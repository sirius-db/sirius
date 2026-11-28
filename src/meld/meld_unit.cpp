/*
 * Copyright 2026, Sirius Contributors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "meld/meld_unit.hpp"

#include "sirius/exception.hpp"
#include "telemetry/nvtx.hpp"

#include <cudf/interop.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/aligned.hpp>
#include <rmm/cuda_device.hpp>
#include <rmm/error.hpp>

#include <arrow/api.h>
#include <arrow/c/abi.h>
#include <arrow/c/bridge.h>
#include <meld-builtin/Builtin.h>
#include <meld/Execution/BoundTable.h>
#include <meld/Execution/DeviceMemory.h>
#include <meld/Execution/Dictionary.h>
#include <meld/Execution/JitEngine.h>
#include <meld/Execution/PtxCompiler.h>
#include <meld/Execution/RunPlan.h>
#include <meld/Execution/Target.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/Parser/Parser.h>

#include <algorithm>
#include <atomic>
#include <mutex>
#include <optional>

namespace sirius::meld {

namespace {

std::atomic<std::uint64_t> batches_run_so_far{0};

void register_once()
{
  static std::once_flag registered;
  std::call_once(registered, [] {
    mlir::meld::registerBuiltinLowering();
    mlir::meld::registerBuiltinExecution();
  });
}

/// A context per call: a context is not shared between threads, and Sirius runs a pipeline on as
/// many as it has. Meld's caches of compiled code outlive it.
class session {
 public:
  session() : _context(make_registry(), mlir::MLIRContext::Threading::DISABLED)
  {
    _handler.emplace(&_context, [this](mlir::Diagnostic& diagnostic) {
      if (!_diagnostics.empty()) { _diagnostics += "\n"; }
      _diagnostics += diagnostic.str();
      for (mlir::Diagnostic const& note : diagnostic.getNotes()) {
        _diagnostics += "\n  note: " + note.str();
      }
      return mlir::success();
    });
  }

  [[nodiscard]] std::string why(std::string const& what) const
  {
    return _diagnostics.empty() ? what : what + ": " + _diagnostics;
  }

  mlir::MLIRContext& context() { return _context; }

 private:
  static mlir::DialectRegistry make_registry()
  {
    mlir::DialectRegistry registry;
    mlir::meld::registerJitDialects(registry);
    mlir::meld::registerPtxDialects(registry);
    return registry;
  }

  mlir::MLIRContext _context;
  std::string _diagnostics;
  std::optional<mlir::ScopedDiagnosticHandler> _handler;
};

struct parsed_plan {
  mlir::OwningOpRef<mlir::ModuleOp> module;
  mlir::meld::PlanInputs inputs;
  mlir::meld::TableInput table;
};

std::optional<parsed_plan> parse(session& session, std::string const& plan)
{
  nvtx_scoped_range nvtx_range{"meld_unit::parse"};
  parsed_plan parsed;
  parsed.module = mlir::parseSourceString<mlir::ModuleOp>(plan, &session.context());
  if (!parsed.module) { return std::nullopt; }
  auto inputs = mlir::meld::planInputs(*parsed.module);
  if (mlir::failed(inputs)) { return std::nullopt; }
  auto tables = mlir::meld::tablesToBind(*inputs, parsed.module->getLoc());
  if (mlir::failed(tables)) { return std::nullopt; }
  if (inputs->size() != 1 || tables->size() != 1) {
    parsed.module->emitError() << "a unit has one root reading one table; this plan has "
                               << inputs->size() << " roots reading " << tables->size()
                               << " tables";
    return std::nullopt;
  }
  parsed.inputs = std::move(*inputs);
  parsed.table  = std::move(tables->front());
  return parsed;
}

/// Meld's device memory from a resource of Sirius's, so that what a run takes is counted and
/// pooled with the rest of what Sirius takes.
class resource_allocator final : public mlir::meld::DeviceAllocator {
 public:
  explicit resource_allocator(rmm::device_async_resource_ref const& mr) : _mr(mr) {}

  arrow::Result<void*> allocate(std::int64_t bytes, void* stream) override
  {
    try {
      return _mr.allocate(
        stream_of(stream), static_cast<std::size_t>(bytes), rmm::CUDA_ALLOCATION_ALIGNMENT);
    } catch (rmm::out_of_memory const& error) {
      return arrow::Status::OutOfMemory(error.what());
    } catch (std::exception const& error) {
      return arrow::Status::ExecutionError(error.what());
    }
  }

  void deallocate(void* pointer, std::int64_t bytes, void* stream) override
  {
    _mr.deallocate(
      stream_of(stream), pointer, static_cast<std::size_t>(bytes), rmm::CUDA_ALLOCATION_ALIGNMENT);
  }

 private:
  static ::cuda::stream_ref stream_of(void* stream)
  {
    return ::cuda::stream_ref{static_cast<cudaStream_t>(stream)};
  }

  rmm::device_async_resource_ref _mr;
};

mlir::meld::RunOptions gpu_options(int device, mlir::meld::DeviceAllocator& memory)
{
  mlir::meld::RunOptions options;
  options.target          = mlir::meld::Target::Gpu;
  options.device          = device;
  options.deviceAllocator = &memory;
  // A column the cache kept would keep a batch's memory past the batch, outside what Sirius
  // accounts for.
  options.hostFragmentCacheBytes   = 0;
  options.deviceFragmentCacheBytes = 0;
  return options;
}

std::optional<cudf::data_type> cudf_type_of(arrow::DataType const& type)
{
  switch (type.id()) {
    case arrow::Type::INT8: return cudf::data_type{cudf::type_id::INT8};
    case arrow::Type::INT16: return cudf::data_type{cudf::type_id::INT16};
    case arrow::Type::INT32: return cudf::data_type{cudf::type_id::INT32};
    case arrow::Type::INT64: return cudf::data_type{cudf::type_id::INT64};
    case arrow::Type::FLOAT: return cudf::data_type{cudf::type_id::FLOAT32};
    case arrow::Type::DOUBLE: return cudf::data_type{cudf::type_id::FLOAT64};
    case arrow::Type::DATE32: return cudf::data_type{cudf::type_id::TIMESTAMP_DAYS};
    default: return std::nullopt;
  }
}

/// The batch's columns as Meld reads them: each one where cuDF holds it, uncopied.
mlir::meld::BoundTable bind_batch(cudf::table_view const& batch,
                                  mlir::meld::TableInput const& table,
                                  std::vector<std::string> const& column_names)
{
  mlir::meld::BoundTable bound;
  bound.rows = batch.num_rows();
  for (auto const& field : table.schema->fields()) {
    auto named = std::find(column_names.begin(), column_names.end(), field->name());
    if (named == column_names.end()) {
      throw internal_exception("Meld reads a column '{}' the batch does not have", field->name());
    }
    cudf::column_view const& column = batch.column(std::distance(column_names.begin(), named));
    auto type                       = cudf_type_of(*field->type());
    if (!type || *type != column.type()) {
      throw internal_exception(
        "Meld reads '{}' as {}, and the batch does not", field->name(), field->type()->ToString());
    }

    mlir::meld::ColumnChunk chunk;
    chunk.rows   = column.size();
    chunk.offset = column.offset();
    chunk.values = mlir::meld::DataBuffer::device(
      column.head(), (column.offset() + column.size()) * cudf::size_of(column.type()), nullptr);
    chunk.nullCount = column.null_count();
    if (chunk.nullCount > 0) {
      chunk.validity = mlir::meld::DataBuffer::device(
        column.null_mask(),
        cudf::bitmask_allocation_size_bytes(column.offset() + column.size()),
        nullptr);
    }
    mlir::meld::BoundColumn bound_column;
    bound_column.field = field;
    bound_column.chunks.push_back(std::move(chunk));
    bound.columns.push_back(std::move(bound_column));
  }
  return bound;
}

std::unique_ptr<cudf::table> to_cudf(arrow::Table const& answer,
                                     ::cuda::stream_ref stream,
                                     rmm::device_async_resource_ref const& mr)
{
  nvtx_scoped_range nvtx_range{"meld_unit::to_cudf"};
  auto batch = answer.CombineChunksToBatch();
  if (!batch.ok()) {
    throw internal_exception("Meld's answer is not one batch: {}", batch.status().ToString());
  }
  ArrowSchema schema;
  ArrowDeviceArray array{};
  if (auto status = arrow::ExportRecordBatch(**batch, &array.array, &schema); !status.ok()) {
    throw internal_exception("Meld's answer cannot be exported: {}", status.ToString());
  }
  array.device_id   = -1;
  array.device_type = ARROW_DEVICE_CPU;
  std::unique_ptr<cudf::table> table;
  try {
    table = cudf::from_arrow_host(&schema, &array, stream, mr);
  } catch (...) {
    schema.release(&schema);
    array.array.release(&array.array);
    throw;
  }
  schema.release(&schema);
  array.array.release(&array.array);
  return table;
}

}  // namespace

std::unique_ptr<meld_unit> meld_unit::check(std::string plan,
                                            std::vector<std::string> column_names,
                                            std::string& why)
{
  nvtx_scoped_range nvtx_range{"meld_unit::check"};
  register_once();
  session session;
  auto parsed = parse(session, plan);
  if (!parsed) {
    why = session.why("Meld cannot read the plan");
    return nullptr;
  }
  for (auto const& field : parsed->table.schema->fields()) {
    if (std::find(column_names.begin(), column_names.end(), field->name()) == column_names.end()) {
      why = "the plan reads a column '" + field->name() + "' the batches do not have";
      return nullptr;
    }
  }

  auto empty = arrow::Table::MakeEmpty(parsed->table.schema);
  if (!empty.ok()) {
    why = "Meld cannot make an empty table: " + empty.status().ToString();
    return nullptr;
  }
  resource_allocator memory(cudf::get_current_device_resource_ref());
  llvm::StringMap<std::shared_ptr<arrow::Table>> tables;
  tables[parsed->table.tableName] = *empty;
  auto answers                    = mlir::meld::runPlan(*parsed->module,
                                     parsed->inputs,
                                     tables,
                                     mlir::meld::DictionaryMap(),
                                     gpu_options(rmm::get_current_cuda_device().value(), memory));
  if (mlir::failed(answers)) {
    why = session.why("Meld cannot run the plan");
    return nullptr;
  }

  std::vector<cudf::data_type> answer_types;
  for (auto const& field : answers->front()->schema()->fields()) {
    auto type = cudf_type_of(*field->type());
    if (!type) {
      why = "Meld answers with a " + field->type()->ToString() + " Sirius does not take";
      return nullptr;
    }
    answer_types.push_back(*type);
  }
  return std::unique_ptr<meld_unit>(
    new meld_unit(std::move(plan), std::move(column_names), std::move(answer_types)));
}

std::unique_ptr<cudf::table> meld_unit::run(cudf::table_view const& batch,
                                            ::cuda::stream_ref stream,
                                            rmm::device_async_resource_ref const& mr) const
{
  nvtx_scoped_range nvtx_range{"meld_unit::run"};
  session session;
  auto parsed = parse(session, _plan);
  if (!parsed) { throw internal_exception(session.why("Meld cannot read a plan it checked")); }

  resource_allocator memory(mr);
  llvm::StringMap<mlir::meld::BoundTable> tables;
  tables[parsed->table.tableName] = bind_batch(batch, parsed->table, _column_names);
  stream.sync();
  auto answers = [&] {
    nvtx_scoped_range run_plan_range{"meld_unit::run_plan"};
    return mlir::meld::runPlan(*parsed->module,
                               parsed->inputs,
                               tables,
                               mlir::meld::DictionaryMap(),
                               gpu_options(rmm::get_current_cuda_device().value(), memory));
  }();
  if (mlir::failed(answers)) {
    throw internal_exception(session.why("Meld could not run the plan"));
  }
  batches_run_so_far.fetch_add(1, std::memory_order_relaxed);
  return to_cudf(*answers->front(), stream, mr);
}

std::uint64_t meld_unit::batches_run()
{
  return batches_run_so_far.load(std::memory_order_relaxed);
}

}  // namespace sirius::meld
