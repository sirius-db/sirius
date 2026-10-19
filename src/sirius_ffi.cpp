/*
 * Copyright 2025, Sirius Contributors.
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

// Implementation of sirius/ffi.hpp. This translation unit sees the heavy internal types so
// consumers (e.g. the Rust bindings) never include sirius_context.hpp.

#include "config.hpp"                                    // duckdb::Config::LOG_*
#include "core_functions_extension.hpp"                  // duckdb::CoreFunctionsExtension
#include "cudf/cudf_utils.hpp"                           // sirius::get_cudf_type
#include "data/data_batch_utils.hpp"                     // sirius::make_data_batch
#include "data/sirius_converter_registry.hpp"            // sirius::converter_registry
#include "duckdb/common/arrow/arrow.hpp"                 // ArrowArray, ArrowSchema, ArrowArrayStream
#include "duckdb/common/enums/optimizer_type.hpp"        // duckdb::OptimizerType
#include "duckdb/execution/column_binding_resolver.hpp"  // duckdb::ColumnBindingResolver
#include "duckdb/main/client_context.hpp"                // duckdb::ClientContext
#include "duckdb/main/config.hpp"                        // duckdb::DBConfig
#include "duckdb/main/connection.hpp"                    // duckdb::Connection
#include "duckdb/main/database.hpp"                      // duckdb::DuckDB
#include "duckdb/main/prepared_statement_data.hpp"       // duckdb::PreparedStatementData
#include "duckdb/main/query_result.hpp"                  // duckdb::QueryResult
#include "duckdb/main/relation.hpp"                      // duckdb::Relation
#include "duckdb/main/settings.hpp"
#include "duckdb/optimizer/optimizer.hpp"                  // duckdb::Optimizer
#include "duckdb/parser/statement/relation_statement.hpp"  // duckdb::RelationStatement
#include "duckdb/planner/planner.hpp"                      // duckdb::Planner
#include "exec/arrow_result_stream.hpp"
#include "exec/stream_bind_catalog.hpp"   // sirius::exec::stream_bind_catalog
#include "exec/stream_plan_bindings.hpp"  // sirius::exec::register_stream_source_function
#include "exec/streaming_fragment.hpp"    // sirius::exec::streaming_fragment, fragment_spec
#include "from_substrait.hpp"             // duckdb::SubstraitToDuckDB (compiled into libsirius)
#include "helper/type_conversions.hpp"    // sirius::from_duckdb
#include "log/logging.hpp"                // SIRIUS_LOG_INFO
#include "parquet_extension.hpp"          // duckdb::ParquetExtension
#include "planner/unsupported_optimizers.hpp"
#include "sirius/ffi.hpp"
#include "sirius_config.hpp"   // sirius::sirius_config
#include "sirius_context.hpp"  // duckdb::SiriusContext

#include <cudf/interop.hpp>
#include <cudf/reduction.hpp>
#include <cudf/round.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/unary.hpp>

#include <rmm/cuda_device.hpp>
#include <rmm/cuda_stream.hpp>

#include <cuda_runtime_api.h>

#include <google/protobuf/util/json_util.h>  // DuckDB-vendored: duckdb::google::protobuf
#include <substrait/plan.pb.h>
#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <stdexcept>

// ArrowDeviceArray, from the Arrow C device data interface
// (https://arrow.apache.org/docs/format/CDeviceDataInterface.html). The interface is ABI-stable
// and its spec asks consumers to copy these definitions rather than depend on libarrow, which
// Sirius does not link; cuDF only forward-declares the struct. The guard skips the copy when
// Arrow's own abi.h was included first.
#ifndef ARROW_C_DEVICE_DATA_INTERFACE
#define ARROW_C_DEVICE_DATA_INTERFACE

// Spec and Documentation: https://arrow.apache.org/docs/format/CDeviceDataInterface.html

// DeviceType for the allocated memory
typedef int32_t ArrowDeviceType;  // NOLINT(modernize-use-using)

// CPU device, same as using ArrowArray directly
#define ARROW_DEVICE_CPU 1
// CUDA GPU Device
#define ARROW_DEVICE_CUDA 2
// Pinned CUDA CPU memory by cudaMallocHost
#define ARROW_DEVICE_CUDA_HOST 3
// OpenCL Device
#define ARROW_DEVICE_OPENCL 4
// Vulkan buffer for next-gen graphics
#define ARROW_DEVICE_VULKAN 7
// Metal for Apple GPU
#define ARROW_DEVICE_METAL 8
// Verilog simulator buffer
#define ARROW_DEVICE_VPI 9
// ROCm GPUs for AMD GPUs
#define ARROW_DEVICE_ROCM 10
// Pinned ROCm CPU memory allocated by hipMallocHost
#define ARROW_DEVICE_ROCM_HOST 11
// Reserved for extension
#define ARROW_DEVICE_EXT_DEV 12
// CUDA managed/unified memory allocated by cudaMallocManaged
#define ARROW_DEVICE_CUDA_MANAGED 13
// unified shared memory allocated on a oneAPI non-partitioned device.
#define ARROW_DEVICE_ONEAPI 14
// GPU support for next-gen WebGPU standard
#define ARROW_DEVICE_WEBGPU 15
// Qualcomm Hexagon DSP
#define ARROW_DEVICE_HEXAGON 16

struct ArrowDeviceArray {
  // the Allocated Array
  //
  // the buffers in the array (along with the buffers of any
  // children) are what is allocated on the device.
  struct ArrowArray array;
  // The device id to identify a specific device
  int64_t device_id;
  // The type of device which can access this memory.
  ArrowDeviceType device_type;
  // An event-like object to synchronize on if needed.
  void* sync_event;
  // Reserved bytes for future expansion.
  int64_t reserved[3];
};

#endif  // ARROW_C_DEVICE_DATA_INTERFACE

namespace sirius::ffi {

namespace {

constexpr const char* kSiriusStateKey   = "sirius_state";
constexpr duckdb::idx_t kArrowBatchSize = 1u << 20;

// DuckDB view name a plan uses to read input stream `id`.
std::string stream_view_name_of(std::uint64_t id) { return "sirius_stream_" + std::to_string(id); }

// The embedded DuckDB never loads the Sirius extension, so nothing on this path reads the
// SIRIUS_LOG_{BACKEND,DIR,LEVEL} environment the transparent path honors
// (SiriusContextExtensionCallback) — the engine would run with the noop sink and no log would
// reach the embedder. Install the same sink here when the embedder asks for one; without any
// of the variables the sink is left untouched (noop by default).
void install_log_sink_from_env()
{
  auto const* backend_env = std::getenv("SIRIUS_LOG_BACKEND");
  auto const* log_dir_env = std::getenv("SIRIUS_LOG_DIR");
  auto const* level_env   = std::getenv("SIRIUS_LOG_LEVEL");
  if (!backend_env && !log_dir_env && !level_env) { return; }
  // Serialize the complete settings update, including rollback if sink creation fails.
  std::lock_guard log_lock(duckdb::Config::logging_mutex);
  auto previous_backend = duckdb::Config::LOG_BACKEND;
  auto previous_log_dir = duckdb::Config::LOG_DIR;
  auto previous_level   = duckdb::Config::LOG_LEVEL;
  try {
    if (backend_env) { duckdb::Config::LOG_BACKEND = backend_env; }
    if (log_dir_env) { duckdb::Config::LOG_DIR = log_dir_env; }
    if (level_env) { duckdb::Config::LOG_LEVEL = level_env; }
    // With no database, an unknown backend leaves the current sink in place.
    duckdb::install_configured_log_sink(nullptr);
  } catch (...) {
    // Sink construction can throw (for example, if LOG_DIR cannot be created).
    // It happens before set_sink(), so the previous sink is still active.
    duckdb::Config::LOG_BACKEND.swap(previous_backend);
    duckdb::Config::LOG_DIR.swap(previous_log_dir);
    duckdb::Config::LOG_LEVEL.swap(previous_level);
    throw;
  }
}

double elapsed_ms(std::chrono::steady_clock::time_point since)
{
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - since)
    .count();
}

// Lower a Substrait plan (serialized bytes, or protobuf JSON when `json`) to a bound+optimized
// DuckDB LogicalOperator.
sirius::exec::bound_plan lower_substrait(duckdb::Connection& conn,
                                         const std::string& substrait_plan,
                                         bool json)
{
  auto& client = *conn.context;

  if (json) {
    // The consumer reports a bad JSON plan as a plain runtime_error; parse it here first so a
    // malformed plan is an argument error, distinct from a lowering failure.
    substrait::Plan parsed;
    auto const status =
      duckdb::google::protobuf::util::JsonStringToMessage(substrait_plan, &parsed);
    if (!status.ok()) {
      throw std::invalid_argument("JSON does not parse as a Substrait Plan: " + status.ToString());
    }
  }
  duckdb::SubstraitToDuckDB transformer(conn.context, substrait_plan, json);
  auto relation = transformer.TransformPlan();

  duckdb::Planner planner(client);
  planner.CreatePlan(duckdb::make_uniq<duckdb::RelationStatement>(relation));

  auto prepared =
    duckdb::make_shared_ptr<duckdb::PreparedStatementData>(duckdb::StatementType::SELECT_STATEMENT);
  prepared->names     = planner.names;
  prepared->types     = planner.types;
  prepared->value_map = std::move(planner.value_map);

  auto logical_plan = std::move(planner.plan);
  if (duckdb::Settings::Get<duckdb::EnableOptimizerSetting>(client)) {
    duckdb::Optimizer optimizer(*planner.binder, client);
    logical_plan = optimizer.Optimize(std::move(logical_plan));
  }
  logical_plan->ResolveOperatorTypes();
  duckdb::ColumnBindingResolver resolver;
  duckdb::ColumnBindingResolver verifier(true);
  verifier.VisitOperator(*logical_plan);
  resolver.VisitOperator(*logical_plan);

  return {std::move(logical_plan), std::move(prepared)};
}

// Run `body` in a transaction on `conn`; a failed rollback does not replace body's error.
// `serial` is held throughout because every Fragment of a Context shares `conn`, and a second
// BEGIN on it fails (and invalidates the open transaction) instead of waiting.
template <typename Body>
void in_transaction(std::mutex& serial, duckdb::Connection& conn, Body&& body)
{
  std::lock_guard<std::mutex> lock(serial);
  conn.BeginTransaction();
  try {
    body();
  } catch (...) {
    try {
      conn.Rollback();
    } catch (...) {  // NOLINT(bugprone-empty-catch)
    }
    throw;
  }
  conn.Commit();
}

// Set while a Context is alive. The engine installs process-wide state, so only one may exist.
std::atomic<bool> g_context_alive{false};

// Claims g_context_alive for one Context and releases it on destruction.
struct live_context_claim {
  live_context_claim()
  {
    if (g_context_alive.exchange(true)) {
      throw std::logic_error("a sirius::ffi::Context already exists in this process");
    }
  }
  ~live_context_claim() { g_context_alive.store(false); }

  live_context_claim(const live_context_claim&)            = delete;
  live_context_claim& operator=(const live_context_claim&) = delete;
};

// Reject a GPU list the engine cannot bring up exactly, naming the offending id.
void validate_gpu_ids(const std::vector<int>& gpu_ids)
{
  if (gpu_ids.empty()) { throw std::invalid_argument("gpu_ids must not be empty"); }
  int device_count  = 0;
  auto const status = cudaGetDeviceCount(&device_count);
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string("cudaGetDeviceCount failed: ") +
                             cudaGetErrorString(status));
  }
  std::set<int> seen;
  for (int id : gpu_ids) {
    if (id < 0) {
      throw std::invalid_argument("gpu_ids: negative device id " + std::to_string(id));
    }
    if (id >= device_count) {
      throw std::invalid_argument("gpu_ids: device id " + std::to_string(id) + " is not visible (" +
                                  std::to_string(device_count) + " visible device(s))");
    }
    if (!seen.insert(id).second) {
      throw std::invalid_argument("gpu_ids: duplicate device id " + std::to_string(id));
    }
  }
}

// Load `config_path` (or an empty `sirius:` document) and pin the topology to `gpu_ids`, with
// every listed GPU available to one query.
sirius::parsed_sirius_config config_with_gpus(const std::string& config_path,
                                              const std::vector<int>& gpu_ids)
{
  YAML::Node root;
  if (config_path.empty()) {
    root["sirius"] = YAML::Node(YAML::NodeType::Map);
  } else {
    try {
      root = YAML::LoadFile(config_path);
    } catch (const std::exception& e) {
      throw sirius::configuration_input_error(config_path + ": " + e.what());
    }
  }
  // A document without a `sirius` map is left as is, so from_node reports it.
  if (root.IsMap() && root["sirius"]) {
    if (root["sirius"].IsNull()) { root["sirius"] = YAML::Node(YAML::NodeType::Map); }
    auto sirius = root["sirius"];
    if (sirius.IsMap()) {
      if (!sirius["topology"] || sirius["topology"].IsNull()) {
        sirius["topology"] = YAML::Node(YAML::NodeType::Map);
      }
      auto topology = sirius["topology"];
      if (topology.IsMap()) {
        // num_gpus and gpu_ids are mutually exclusive.
        topology.remove("num_gpus");
        topology["gpu_ids"] = gpu_ids;
        // 0 = a query may use every active GPU.
        topology["gpus_per_query"] = 0;
      }
    }
  }
  return sirius::parsed_sirius_config::from_node(root, config_path);
}

// ---------------------------------------------------------------------------
// push_arrow helpers
// ---------------------------------------------------------------------------

// Owns one Arrow C struct and releases it on scope exit. Taking over moves the caller's struct
// and marks the caller's copy released, per the Arrow C data interface move semantics.
template <typename T>
struct owned_arrow {
  T value{};

  owned_arrow() = default;
  explicit owned_arrow(T* caller)
  {
    if (caller != nullptr) {
      value           = *caller;
      caller->release = nullptr;
    }
  }
  ~owned_arrow()
  {
    if (value.release != nullptr) { value.release(&value); }
  }

  owned_arrow(const owned_arrow&)            = delete;
  owned_arrow& operator=(const owned_arrow&) = delete;
};

// Arrow format strings push_arrow accepts (C data interface type codes).
enum class arrow_kind : std::uint8_t {
  int32,
  int64,
  float64,
  date32,
  utf8,
  large_utf8,
  utf8_view,
  decimal128,
  other
};

struct arrow_column_type {
  arrow_kind kind{arrow_kind::other};
  int precision{0};
  int scale{0};
  std::string name;  // for messages
};

arrow_column_type parse_arrow_format(const char* format)
{
  std::string const f = format == nullptr ? "" : format;
  arrow_column_type t;
  t.name = f;
  if (f == "i") {
    t = {arrow_kind::int32, 0, 0, "int32"};
  } else if (f == "l") {
    t = {arrow_kind::int64, 0, 0, "int64"};
  } else if (f == "g") {
    t = {arrow_kind::float64, 0, 0, "double"};
  } else if (f == "tdD") {
    t = {arrow_kind::date32, 0, 0, "date32"};
  } else if (f == "u") {
    t = {arrow_kind::utf8, 0, 0, "utf8"};
  } else if (f == "U") {
    t = {arrow_kind::large_utf8, 0, 0, "large_utf8"};
  } else if (f == "vu") {
    t = {arrow_kind::utf8_view, 0, 0, "string_view"};
  } else if (f.rfind("d:", 0) == 0) {
    // d:P,S or d:P,S,BITWIDTH; only the 128-bit layout is accepted.
    int precision = 0, scale = 0, bit_width = 128;
    auto const fields = std::sscanf(f.c_str(), "d:%d,%d,%d", &precision, &scale, &bit_width);
    if (fields >= 2 && bit_width == 128) {
      t = {arrow_kind::decimal128,
           precision,
           scale,
           "decimal128(" + std::to_string(precision) + "," + std::to_string(scale) + ")"};
    } else {
      t.name = "decimal" + std::to_string(bit_width) + "(" + std::to_string(precision) + "," +
               std::to_string(scale) + ")";
    }
  }
  return t;
}

// Whether an Arrow column of `from` converts to the declared type `to`.
bool arrow_converts_to(const arrow_column_type& from, const sirius::logical_type& to)
{
  switch (to.id()) {
    case sirius::type_id::INTEGER: return from.kind == arrow_kind::int32;
    case sirius::type_id::BIGINT: return from.kind == arrow_kind::int64;
    case sirius::type_id::DOUBLE: return from.kind == arrow_kind::float64;
    case sirius::type_id::DATE: return from.kind == arrow_kind::date32;
    case sirius::type_id::VARCHAR:
      return from.kind == arrow_kind::utf8 || from.kind == arrow_kind::large_utf8 ||
             from.kind == arrow_kind::utf8_view;
    case sirius::type_id::DECIMAL:
      // DuckDB stores DECIMAL(p<=4) as INT16, which has no cuDF carrier.
      return from.kind == arrow_kind::decimal128 && from.precision <= 38 &&
             to.decimal_precision() > sirius::logical_type::decimal_max_precision_int16 &&
             to.decimal_precision() <= 38;
    default: return false;
  }
}

// Upper bound on the host bytes of one Arrow column: it sizes rows 0..offset+length, not just the
// `length` rows past `offset`, so a sliced column is over-counted. The device copy is about the
// same size.
std::size_t arrow_column_bytes(const arrow_column_type& type, const ArrowArray& column)
{
  auto const rows            = static_cast<std::size_t>(column.offset + column.length);
  std::size_t const validity = (rows + 7) / 8;
  switch (type.kind) {
    case arrow_kind::int32:
    case arrow_kind::date32: return validity + 4 * rows;
    case arrow_kind::int64:
    case arrow_kind::float64: return validity + 8 * rows;
    case arrow_kind::decimal128: return validity + 16 * rows;
    case arrow_kind::utf8: {
      auto const* offsets = static_cast<const std::int32_t*>(column.buffers[1]);
      auto const chars    = offsets == nullptr ? 0 : static_cast<std::size_t>(offsets[rows]);
      return validity + 4 * (rows + 1) + chars;
    }
    case arrow_kind::large_utf8: {
      auto const* offsets = static_cast<const std::int64_t*>(column.buffers[1]);
      auto const chars    = offsets == nullptr ? 0 : static_cast<std::size_t>(offsets[rows]);
      return validity + 8 * (rows + 1) + chars;
    }
    case arrow_kind::utf8_view: {
      // Buffers: validity, views, variadic data..., variadic sizes (int64 each).
      std::size_t bytes = validity + 16 * rows;
      if (column.n_buffers >= 3) {
        auto const* sizes = static_cast<const std::int64_t*>(column.buffers[column.n_buffers - 1]);
        for (std::int64_t i = 0; sizes != nullptr && i < column.n_buffers - 3; ++i) {
          bytes += static_cast<std::size_t>(sizes[i]);
        }
      }
      return bytes;
    }
    default: return 0;
  }
}

// Whether converting `from` to `to` holds a second device copy of the column at its peak: a
// decimal rescale, rounding or narrowing cast, or cuDF's rebuild of a string_view column.
bool takes_second_copy(const arrow_column_type& from, const sirius::logical_type& to)
{
  if (from.kind == arrow_kind::utf8_view) { return true; }
  if (to.id() != sirius::type_id::DECIMAL) { return false; }
  return from.scale != static_cast<int>(to.decimal_scale()) ||
         sirius::get_cudf_type(to).id() != cudf::type_id::DECIMAL128;
}

// 10^precision - 1, the largest unscaled value of DECIMAL(precision, s).
__int128 decimal_unscaled_limit(int precision)
{
  __int128 limit = 1;
  for (int i = 0; i < precision; ++i) {
    limit *= 10;
  }
  return limit - 1;
}

std::string push_column_context(std::uint64_t stream_id,
                                std::size_t column,
                                const arrow_column_type& from,
                                const sirius::logical_type& to)
{
  return "push_arrow: stream " + std::to_string(stream_id) + " column " + std::to_string(column) +
         " (Arrow " + from.name + ", declared " + to.to_string() + ")";
}

// Convert one Arrow child column to the device carrier of its declared type.
std::unique_ptr<cudf::column> arrow_column_to_declared(const ArrowSchema& schema,
                                                       const ArrowArray& column,
                                                       std::int64_t parent_offset,
                                                       std::int64_t parent_length,
                                                       const arrow_column_type& from,
                                                       const sirius::logical_type& to,
                                                       const std::string& context,
                                                       rmm::cuda_stream_view stream,
                                                       rmm::device_async_resource_ref mr)
{
  // A child of a sliced struct batch spans the parent's rows: narrow the borrowed child to them
  // (cuDF honours ArrowArray::offset). Its null count may not hold for the narrower range, so it
  // becomes unknown (-1, cuDF counts the bitmap) unless the child has no nulls at all.
  if (parent_offset < 0 || parent_length < 0 || column.length < parent_offset + parent_length) {
    throw std::invalid_argument(context + ": the column has " + std::to_string(column.length) +
                                " rows, fewer than the batch's offset plus length");
  }
  ArrowDeviceArray host{};
  host.array = column;  // borrowed: cuDF does not release its input
  if (parent_offset != 0 || parent_length != column.length) {
    host.array.offset += parent_offset;
    host.array.length = parent_length;
    if (host.array.null_count != 0) { host.array.null_count = -1; }
  }
  host.device_id   = -1;
  host.device_type = ARROW_DEVICE_CPU;
  auto converted   = cudf::from_arrow_host_column(&schema, &host, stream, mr);

  auto const target = sirius::get_cudf_type(to);
  if (to.id() == sirius::type_id::DECIMAL) {
    // Round or check at the source scale, rescale in 128 bits, then narrow to the declared
    // carrier (decimal64 for DECIMAL(15,2), as the parquet scan produces). The check compares
    // unscaled values against 10^digits - 1 at the column's scale after rounding.
    auto const precision = static_cast<int>(to.decimal_precision());
    auto const scale     = static_cast<int>(to.decimal_scale());
    if (converted->type().id() != cudf::type_id::DECIMAL128) {
      converted = cudf::cast(converted->view(),
                             cudf::data_type{cudf::type_id::DECIMAL128, converted->type().scale()},
                             stream,
                             mr);
    }
    int digits       = precision;
    bool always_fits = false;
    if (from.scale > scale) {
      // Round half away from zero, as a DuckDB CAST does (1.2350 -> 1.24). Rounding can carry
      // into a new integer digit (9.995 -> 10.00), so the check runs after it: a source below
      // 10^(p-s) in magnitude rounds to at most 10^(p-s).
      converted =
        cudf::round_decimal(converted->view(), scale, cudf::rounding_method::HALF_UP, stream, mr);
      always_fits = (from.precision - from.scale) + scale < precision;
    } else {
      // DECIMAL(P,S) holds |x| < 10^(P-S), i.e. |unscaled| < 10^(P-S+s) at the source scale s.
      // Checked before the upscale, whose multiply by 10^(S-s) is unchecked and could wrap.
      digits      = precision - scale + from.scale;
      always_fits = from.precision <= digits;
    }
    if (!always_fits && converted->size() > converted->null_count()) {
      auto const [lo, hi]  = cudf::minmax(converted->view(), stream, mr);
      using decimal_scalar = cudf::fixed_point_scalar<numeric::decimal128>;
      auto const limit     = decimal_unscaled_limit(digits);
      auto const min_value = static_cast<decimal_scalar&>(*lo).value(stream);
      auto const max_value = static_cast<decimal_scalar&>(*hi).value(stream);
      if (min_value < -limit || max_value > limit) {
        throw std::invalid_argument(context + ": a value overflows the declared precision");
      }
    }
    cudf::data_type const wide{cudf::type_id::DECIMAL128, target.scale()};
    if (converted->type() != wide) { converted = cudf::cast(converted->view(), wide, stream, mr); }
    if (converted->type() != target) {
      converted = cudf::cast(converted->view(), target, stream, mr);
    }
  }
  if (converted->type() != target) {
    throw std::invalid_argument(context + ": converted to cuDF type " +
                                std::to_string(static_cast<int>(converted->type().id())) +
                                ", expected " + std::to_string(static_cast<int>(target.id())));
  }
  return converted;
}

}  // namespace

// PIMPL: holds the engine + embedded DuckDB using DuckDB's own smart pointers
// (duckdb::shared_ptr, so the SiriusContext can register as a ClientContextState).
struct Context::Impl {
  // First member: claimed before anything is brought up, released after everything is torn down.
  live_context_claim claim;
  duckdb::shared_ptr<duckdb::SiriusContext> context;
  duckdb::unique_ptr<duckdb::DuckDB> db;
  duckdb::unique_ptr<duckdb::Connection> conn;
  // Serializes transactions on `conn`; see in_transaction().
  std::mutex conn_mutex;

  // Context::execute_substrait / execute_substrait_json.
  void execute(const std::string& plan, bool json, std::uintptr_t out_stream_addr);

  // Serializes use of upload_stream: a Fragment::push_arrow batch binds its GPU reservation to
  // the stream from attach to reset, so two batches cannot convert on it at once.
  std::mutex upload_mutex;
  // Long-lived because the uploaded columns free their memory on the stream they were allocated
  // on. Every Fragment (and so every pushed batch) is gone before the Context is destroyed.
  std::unique_ptr<rmm::cuda_stream> upload_stream;

  void bring_up(const sirius::parsed_sirius_config& config)
  {
    install_log_sink_from_env();
    context = duckdb::make_shared_ptr<duckdb::SiriusContext>();
    context->initialize(config);

    // Substrait lowering uses core functions and resolves local_files reads to parquet_scan.
    db = duckdb::make_uniq<duckdb::DuckDB>(nullptr);
    db->LoadStaticExtension<duckdb::CoreFunctionsExtension>();
    db->LoadStaticExtension<duckdb::ParquetExtension>();
    // Cache parquet footers across binds. The Substrait consumer builds the plan through the
    // Relation API, which re-binds the whole subtree at every level, so a read of one file is
    // bound once per operator above it; with the cache off each bind re-parses the footer and
    // re-derives the column statistics (~0.5 s per bind for a 25 GB, 5k-row-group TPC-H SF100
    // lineitem: Q21 spent 10 s lowering against 4.6 s executing). Set the option at the
    // database level so every connection on this private DuckDB instance inherits it.
    // This DuckDB instance is private to the engine, and the cache validates file mtimes.
    duckdb::DBConfig::GetConfig(*db->instance)
      .SetOptionByName("parquet_metadata_cache", duckdb::Value::BOOLEAN(true));
    conn = duckdb::make_uniq<duckdb::Connection>(*db);
    // Register the engine on the connection and disable DuckDB optimizer rewrites this
    // no-fallback FFI path cannot safely execute.
    auto& client = *conn->context;
    client.registered_state->Insert(kSiriusStateKey, context);
    // Per-connection Sirius state (guard depths, capture bookkeeping) for the
    // embedded connection, mirroring OnConnectionOpened on the transparent path.
    client.registered_state->Insert("sirius_connection_state",
                                    duckdb::make_shared_ptr<duckdb::SiriusConnectionState>());

    // Fragment bind path: register catalog + sirius_stream_source before any plan binds.
    client.registered_state->Insert(sirius::exec::stream_bind_catalog::kStateKey,
                                    duckdb::make_shared_ptr<sirius::exec::stream_bind_catalog>());
    sirius::exec::register_stream_source_function(*db->instance);
    duckdb::Settings::Set<duckdb::EnableOptimizerSetting>(
      client, duckdb::SetScope::LOCAL, duckdb::Value::BOOLEAN(true));
    auto& disabled = duckdb::DBConfig::GetConfig(client).options.disabled_optimizers;
    disabled.insert(duckdb::OptimizerType::IN_CLAUSE);
    disabled.insert(duckdb::OptimizerType::COMPRESSED_MATERIALIZATION);
    disabled.insert(sirius::planner::unsupported_optimizers.begin(),
                    sirius::planner::unsupported_optimizers.end());
    // Keep this FFI-only restriction until its no-fallback execution path has
    // dedicated GPU_VALUES coverage.
    disabled.insert(duckdb::OptimizerType::STATISTICS_PROPAGATION);
    disabled.insert(duckdb::OptimizerType::COLUMN_LIFETIME);
    // Rewrites an ORDER BY ... LIMIT parquet scan into a semi-join on virtual
    // file_index/file_row_number columns the GPU scan drops.
    disabled.insert(duckdb::OptimizerType::LATE_MATERIALIZATION);
  }
};

Context::Context() : impl_(std::make_unique<Impl>())
{
  sirius::parsed_sirius_config config;
  impl_->bring_up(config);
}

Context::Context(const std::string& config_path) : impl_(std::make_unique<Impl>())
{
  auto config = sirius::parsed_sirius_config::from_file(config_path);
  impl_->bring_up(config);
}

Context::Context(const std::string& config_path, const std::vector<int>& gpu_ids)
  : impl_(std::make_unique<Impl>())
{
  validate_gpu_ids(gpu_ids);
  auto config = config_with_gpus(config_path, gpu_ids);
  impl_->bring_up(config);
}

// Defined here, where the heavy types are complete: destroying `impl_` tears down
// the embedded DuckDB and the initialized engine.
Context::~Context() = default;

void Context::execute_substrait(const std::string& plan, std::uintptr_t out_stream_addr)
{
  impl_->execute(plan, /*json=*/false, out_stream_addr);
}

void Context::execute_substrait_json(const std::string& json_plan, std::uintptr_t out_stream_addr)
{
  impl_->execute(json_plan, /*json=*/true, out_stream_addr);
}

void Context::Impl::execute(const std::string& plan, bool json, std::uintptr_t out_stream_addr)
{
  auto& client = *conn->context;

  // Binding (catalog lookups), optimization, and GPU execution all need an active
  // transaction: scans read DuckDB MVCC state through it. GPU execution is eager (the result
  // is materialized), so the transaction can close before the Arrow stream is consumed.
  duckdb::unique_ptr<duckdb::QueryResult> result;
  // Phase timings (lowering / physical planning / GPU execution), logged per query: the embedder
  // only sees the total, and the query window in the telemetry covers execution alone.
  double lower_ms = 0, plan_ms = 0, execute_ms = 0;
  in_transaction(conn_mutex, *conn, [&] {
    // The same result path as a zero-output Fragment.
    sirius::exec::fragment_spec spec;
    spec.plan_source = [&](duckdb::ClientContext&) {
      auto const lower_started = std::chrono::steady_clock::now();
      auto bound               = lower_substrait(*conn, plan, json);
      lower_ms                 = elapsed_ms(lower_started);
      return bound;
    };
    sirius::exec::streaming_fragment fragment(client, std::move(spec));
    auto const plan_started = std::chrono::steady_clock::now();
    fragment.build();
    plan_ms = elapsed_ms(plan_started) - lower_ms;

    auto const execute_started = std::chrono::steady_clock::now();
    fragment.run();
    result     = fragment.take_result();
    execute_ms = elapsed_ms(execute_started);
  });
  SIRIUS_LOG_INFO(
    "[sirius_ffi] execute_substrait: lower {:.1f} ms, plan {:.1f} ms, execute {:.1f} ms",
    lower_ms,
    plan_ms,
    execute_ms);

  sirius::exec::export_arrow_result(
    std::move(result), *reinterpret_cast<ArrowArrayStream*>(out_stream_addr), kArrowBatchSize);
}

std::unique_ptr<Context> make_context() { return std::make_unique<Context>(); }

std::unique_ptr<Context> make_context_from_config(const std::string& config_path)
{
  return std::make_unique<Context>(config_path);
}

std::unique_ptr<Context> make_context_with_gpus(const std::vector<int>& gpu_ids,
                                                const std::string& config_path)
{
  return std::make_unique<Context>(config_path, gpu_ids);
}

std::unique_ptr<std::vector<int>> Context::active_gpu_ids() const
{
  return std::make_unique<std::vector<int>>(impl_->context->active_gpu_ids());
}

// ---------------------------------------------------------------------------
// Fragment
// ---------------------------------------------------------------------------

// PIMPL around one streaming_fragment.
struct Fragment::Impl {
  explicit Impl(Context::Impl& ctx) : ctx(ctx) {}

  Context::Impl& ctx;

  // An input stream declared before build(). type_names are parsed at build(), inside a
  // transaction, because parsing may need a catalog lookup.
  struct declared_input {
    std::vector<std::string> names;
    std::vector<std::string> type_names;
    std::set<sirius::exec::sender_id_t> expected_senders;
    std::optional<std::uint64_t> estimated_rows;
  };

  std::map<sirius::exec::stream_id_t, declared_input> inputs;
  std::vector<sirius::exec::stream_id_t> outputs;
  bool broadcast_outputs{false};
  std::vector<int> hash_key_columns;

  // One streaming_fragment for both terminals. Empty outputs is a RESULT_COLLECTOR.
  std::unique_ptr<sirius::exec::streaming_fragment> fragment;

  // Fragment::build / build_json.
  void build(const std::string& substrait_plan, bool json);

  // Fragment::push_arrow, after the caller's stream is taken over.
  std::size_t push_arrow(std::uint64_t stream_id, ArrowArrayStream& input, std::uint32_t sender_id);

  // Set when a push_arrow() failed after pushing at least one batch: that input holds a partial
  // stream and its sender is still open, so closing or running would silently compute over part
  // of the input, and a retry would duplicate batches. Guarded by push_failure_mutex.
  mutable std::mutex push_failure_mutex;
  std::string push_failure;

  void record_push_failure(std::string failure)
  {
    std::lock_guard<std::mutex> lock(push_failure_mutex);
    if (push_failure.empty()) { push_failure = std::move(failure); }
  }

  void require_no_failed_push(const char* what) const
  {
    std::lock_guard<std::mutex> lock(push_failure_mutex);
    if (!push_failure.empty()) {
      throw std::logic_error(std::string("Fragment: ") + what + " after an earlier failure (" +
                             push_failure + "); discard this Fragment");
    }
  }

  void require_not_built(const char* what) const
  {
    if (fragment) {
      throw sirius::invalid_input_exception(std::string("Fragment: ") + what +
                                            " must be called before build()");
    }
  }

  void require_built(const char* what) const
  {
    if (!fragment) {
      throw sirius::invalid_input_exception(std::string("Fragment: build() must run before ") +
                                            what);
    }
  }

  std::map<sirius::exec::stream_id_t, sirius::exec::stream_input_spec> resolve_inputs() const
  {
    std::map<sirius::exec::stream_id_t, sirius::exec::stream_input_spec> resolved;
    for (const auto& [id, declared] : inputs) {
      sirius::exec::stream_input_spec spec;
      spec.names = declared.names;
      spec.types.reserve(declared.type_names.size());
      for (const auto& type_name : declared.type_names) {
        spec.types.push_back(
          sirius::from_duckdb(duckdb::TransformStringToLogicalType(type_name, *ctx.conn->context)));
      }
      spec.expected_senders = declared.expected_senders;
      if (spec.expected_senders.empty()) { spec.expected_senders.insert(0); }
      spec.estimated_rows = declared.estimated_rows;
      resolved.emplace(id, std::move(spec));
    }
    return resolved;
  }

  // CREATE OR REPLACE VIEW sirius_stream_<id> AS SELECT * FROM sirius_stream_source(<id>)
  // Needs an open transaction, and the ids declared in the stream_bind_catalog so the view's
  // SELECT binds: call it from the plan source, which streaming_fragment::build() runs after
  // declaring them.
  void create_stream_views()
  {
    for (const auto& [id, _] : inputs) {
      const auto view_name = stream_view_name_of(id);
      const auto sql       = "CREATE OR REPLACE VIEW main." + view_name + " AS SELECT * FROM " +
                       std::string(sirius::exec::kStreamSourceFunctionName) + "(" +
                       std::to_string(id) + ")";
      auto res = ctx.conn->Query(sql);
      if (res->HasError()) { res->ThrowError(); }
    }
  }
};

Fragment::Fragment(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}

Fragment::~Fragment() = default;

void Fragment::declare_input_column(std::uint64_t stream_id,
                                    const std::string& name,
                                    const std::string& type)
{
  impl_->require_not_built("declare_input_column");
  auto& d = impl_->inputs[stream_id];
  d.names.push_back(name);
  d.type_names.push_back(type);
}

void Fragment::declare_input_sender(std::uint64_t stream_id, std::uint32_t sender_id)
{
  impl_->require_not_built("declare_input_sender");
  impl_->inputs[stream_id].expected_senders.insert(sender_id);
}

void Fragment::declare_input_cardinality(std::uint64_t stream_id, std::uint64_t rows)
{
  impl_->require_not_built("declare_input_cardinality");
  impl_->inputs[stream_id].estimated_rows = rows;
}

void Fragment::declare_output(std::uint64_t stream_id)
{
  impl_->require_not_built("declare_output");
  auto& outs = impl_->outputs;
  if (std::find(outs.begin(), outs.end(), stream_id) != outs.end()) {
    throw sirius::invalid_input_exception("Fragment: duplicate output stream id " +
                                          std::to_string(stream_id));
  }
  outs.push_back(stream_id);
}

void Fragment::declare_output_broadcast()
{
  impl_->require_not_built("declare_output_broadcast");
  if (!impl_->hash_key_columns.empty()) {
    throw sirius::invalid_input_exception(
      "Fragment: broadcast and hash-partitioned output are mutually exclusive");
  }
  impl_->broadcast_outputs = true;
}

void Fragment::declare_output_hash_key(std::uint32_t column_index)
{
  impl_->require_not_built("declare_output_hash_key");
  if (impl_->broadcast_outputs) {
    throw sirius::invalid_input_exception(
      "Fragment: broadcast and hash-partitioned output are mutually exclusive");
  }
  impl_->hash_key_columns.push_back(static_cast<int>(column_index));
}

void Fragment::build(const std::string& substrait_plan) { impl_->build(substrait_plan, false); }

void Fragment::build_json(const std::string& json_plan) { impl_->build(json_plan, true); }

void Fragment::Impl::build(const std::string& substrait_plan, bool json)
{
  require_not_built("build");

  std::unique_ptr<sirius::exec::streaming_fragment> built;
  // Type-name parsing, CREATE VIEW, and Substrait lowering all need an active transaction. A
  // failure rolls back the views, so a half-declared fragment leaves nothing behind.
  in_transaction(ctx.conn_mutex, *ctx.conn, [&] {
    sirius::exec::fragment_spec spec;
    spec.inputs      = resolve_inputs();
    spec.outputs     = outputs;
    spec.plan_source = [this, &substrait_plan, json](duckdb::ClientContext&) {
      create_stream_views();
      return lower_substrait(*ctx.conn, substrait_plan, json);
    };
    // streaming_fragment rejects a partition mode on fewer than two outputs.
    if (broadcast_outputs) {
      sirius::op::partition_spec broadcast;
      broadcast.mode    = sirius::op::partition_mode::broadcast;
      spec.partitioning = std::move(broadcast);
    } else if (!hash_key_columns.empty()) {
      // key_cast_types left empty. streaming_fragment::build() fills them from output types.
      sirius::op::partition_spec hash;
      hash.mode         = sirius::op::partition_mode::hash;
      hash.key_columns  = hash_key_columns;
      spec.partitioning = std::move(hash);
    }

    built = std::make_unique<sirius::exec::streaming_fragment>(*ctx.conn->context, std::move(spec));
    built->build();
  });
  fragment = std::move(built);
}

std::size_t Fragment::Impl::push_arrow(std::uint64_t stream_id,
                                       ArrowArrayStream& input,
                                       std::uint32_t sender_id)
{
  if (!fragment) { throw std::logic_error("Fragment: build() must run before push_arrow()"); }
  if (!fragment->accepts_input()) {
    throw std::logic_error("Fragment: push_arrow() must be called before run()");
  }
  require_no_failed_push("push_arrow()");
  auto const* declared = fragment->find_input(stream_id);
  if (declared == nullptr) {
    throw std::invalid_argument("push_arrow: input stream " + std::to_string(stream_id) +
                                " was never declared on this fragment");
  }
  if (declared->expected_senders.count(sender_id) == 0) {
    throw std::invalid_argument("push_arrow: sender " + std::to_string(sender_id) +
                                " is not in the expected set for input stream " +
                                std::to_string(stream_id));
  }
  if (input.release == nullptr || input.get_schema == nullptr || input.get_next == nullptr) {
    throw std::invalid_argument("push_arrow: the ArrowArrayStream is released or incomplete");
  }
  auto const last_error = [&input]() -> std::string {
    char const* message = input.get_last_error != nullptr ? input.get_last_error(&input) : nullptr;
    return message != nullptr ? message : "no error message";
  };

  owned_arrow<ArrowSchema> schema;
  if (input.get_schema(&input, &schema.value) != 0) {
    throw std::runtime_error("push_arrow: get_schema failed: " + last_error());
  }
  auto const& types = declared->types;
  if (static_cast<std::size_t>(schema.value.n_children) != types.size()) {
    throw std::invalid_argument("push_arrow: stream " + std::to_string(stream_id) + " declares " +
                                std::to_string(types.size()) +
                                " columns but the Arrow stream has " +
                                std::to_string(schema.value.n_children));
  }
  std::vector<arrow_column_type> columns;
  std::vector<std::string> contexts;
  columns.reserve(types.size());
  for (std::size_t i = 0; i < types.size(); ++i) {
    columns.push_back(parse_arrow_format(schema.value.children[i]->format));
    contexts.push_back(push_column_context(stream_id, i, columns.back(), types[i]));
    if (!arrow_converts_to(columns.back(), types[i])) {
      throw std::invalid_argument(contexts.back() + ": unconvertible type");
    }
  }
  if (fragment->input_sender_closed(stream_id, sender_id)) {
    throw std::runtime_error("push_arrow: input stream " + std::to_string(stream_id) +
                             " already ended for sender " + std::to_string(sender_id));
  }

  if (ctx.context->active_gpu_ids().empty()) {
    throw std::runtime_error("push_arrow: the engine has no active GPU");
  }
  auto& space = fragment->input_memory_space();
  rmm::cuda_set_device_raii device_guard{rmm::cuda_device_id{space.get_device_id()}};
  auto const mr = space.get_default_allocator();

  std::size_t pushed = 0;
  try {
    for (;;) {
      owned_arrow<ArrowArray> batch;
      if (input.get_next(&input, &batch.value) != 0) {
        throw std::runtime_error("push_arrow: get_next failed: " + last_error());
      }
      if (batch.value.release == nullptr) { break; }
      if (static_cast<std::size_t>(batch.value.n_children) != types.size()) {
        throw std::invalid_argument("push_arrow: stream " + std::to_string(stream_id) +
                                    " batch has " + std::to_string(batch.value.n_children) +
                                    " columns, the schema " + std::to_string(types.size()));
      }

      // Charge the upload to a reservation on the target GPU: the host bytes for the copy, twice
      // for a column whose conversion holds a second copy (see takes_second_copy).
      std::size_t bytes = 0;
      for (std::size_t i = 0; i < types.size(); ++i) {
        auto const column_bytes = arrow_column_bytes(columns[i], *batch.value.children[i]);
        bytes += takes_second_copy(columns[i], types[i]) ? 2 * column_bytes : column_bytes;
      }
      // Reserved before taking the upload lock, which only guards the shared upload stream.
      auto reservation = space.make_reservation(std::max<std::size_t>(bytes, 1));

      std::shared_ptr<cucascade::data_batch> data;
      {
        std::lock_guard<std::mutex> upload_lock(ctx.upload_mutex);
        if (!ctx.upload_stream) { ctx.upload_stream = std::make_unique<rmm::cuda_stream>(); }
        auto const stream = ctx.upload_stream->view();
        auto* allocator   = reservation->get_memory_resource_of<cucascade::memory::Tier::GPU>();
        if (allocator == nullptr ||
            !allocator->attach_reservation_to_tracker(stream, std::move(reservation))) {
          throw std::runtime_error("push_arrow: failed to bind the upload to its GPU reservation");
        }
        try {
          std::vector<std::unique_ptr<cudf::column>> converted;
          converted.reserve(types.size());
          for (std::size_t i = 0; i < types.size(); ++i) {
            converted.push_back(arrow_column_to_declared(*schema.value.children[i],
                                                         *batch.value.children[i],
                                                         batch.value.offset,
                                                         batch.value.length,
                                                         columns[i],
                                                         types[i],
                                                         contexts[i],
                                                         stream,
                                                         mr));
          }
          data = sirius::make_data_batch(cudf::table(std::move(converted)),
                                         space,
                                         stream,
                                         sirius::telemetry::batch_telemetry_info{});
          // Publish only finished data. Same-GPU consumers read a batch already in their memory
          // space without waiting on its writer event; Sirius relies on producers syncing before
          // they publish (as gpu_pipeline_task does). The conversions above are only enqueued on
          // the upload stream.
          stream.synchronize();
        } catch (...) {
          allocator->reset_stream_reservation(stream);
          throw;
        }
        allocator->reset_stream_reservation(stream);
      }

      if (!fragment->push_input(stream_id, std::move(data))) {
        throw std::runtime_error("push_arrow: input stream " + std::to_string(stream_id) +
                                 " refused a batch; it had already ended");
      }
      ++pushed;
    }
    fragment->close_input(stream_id, sender_id);
  } catch (const std::exception& e) {
    if (pushed > 0) {
      record_push_failure("push_arrow into input stream " + std::to_string(stream_id) +
                          " (sender " + std::to_string(sender_id) + ") failed after " +
                          std::to_string(pushed) + " batch(es): " + e.what());
    }
    throw;
  } catch (...) {
    if (pushed > 0) {
      record_push_failure("push_arrow into input stream " + std::to_string(stream_id) +
                          " (sender " + std::to_string(sender_id) + ") failed after " +
                          std::to_string(pushed) + " batch(es)");
    }
    throw;
  }
  return pushed;
}

std::size_t Fragment::push_arrow(std::uint64_t stream_id,
                                 std::uintptr_t arrow_array_stream_addr,
                                 std::uint32_t sender_id)
{
  // Take the stream over before any check, so the caller's copy is released on every path.
  owned_arrow<ArrowArrayStream> input(reinterpret_cast<ArrowArrayStream*>(arrow_array_stream_addr));
  if (arrow_array_stream_addr == 0) {
    throw std::invalid_argument("push_arrow: null ArrowArrayStream address");
  }
  return impl_->push_arrow(stream_id, input.value, sender_id);
}

std::size_t Fragment::relay_from(Fragment& source,
                                 std::uint64_t source_stream_id,
                                 std::uint64_t input_stream_id,
                                 std::uint32_t sender_id)
{
  impl_->require_built("relay_from()");
  impl_->require_no_failed_push("relay_from()");
  if (!source.impl_->fragment) {
    throw sirius::invalid_input_exception(
      "Fragment: relay_from() requires the source fragment to have been built");
  }
  return impl_->fragment->relay_from(
    *source.impl_->fragment, source_stream_id, input_stream_id, sender_id);
}

void Fragment::close_input(std::uint64_t stream_id, std::uint32_t sender_id)
{
  impl_->require_built("close_input()");
  impl_->require_no_failed_push("close_input()");
  impl_->fragment->close_input(stream_id, sender_id);
}

void Fragment::run()
{
  impl_->require_built("run()");
  impl_->require_no_failed_push("run()");
  // Scans read DuckDB MVCC state through the active transaction.
  in_transaction(impl_->ctx.conn_mutex, *impl_->ctx.conn, [&] { impl_->fragment->run(); });
}

void Fragment::result_to_arrow(std::uintptr_t out_stream_addr)
{
  impl_->require_built("result_to_arrow()");
  auto result = impl_->fragment->take_result();
  sirius::exec::export_arrow_result(
    std::move(result), *reinterpret_cast<ArrowArrayStream*>(out_stream_addr), kArrowBatchSize);
}

std::size_t Fragment::output_batch_count(std::uint64_t stream_id) const
{
  if (!impl_->fragment) { return 0; }
  return impl_->fragment->output_batch_count(stream_id);
}

std::uint64_t Fragment::output_row_count(std::uint64_t stream_id) const
{
  impl_->require_built("output_row_count()");
  return impl_->fragment->output_row_count(stream_id);
}

std::unique_ptr<std::vector<std::string>> Fragment::output_types() const
{
  impl_->require_built("output_types()");
  if (impl_->fragment->is_result()) {
    throw sirius::invalid_input_exception(
      "Fragment: output_types() is only valid on an intermediate fragment with output streams");
  }
  auto types = std::make_unique<std::vector<std::string>>();
  types->reserve(impl_->fragment->sink_types().size());
  for (const auto& type : impl_->fragment->sink_types()) {
    types->push_back(type.to_string());
  }
  return types;
}

std::unique_ptr<Fragment> make_fragment(Context& context)
{
  return std::unique_ptr<Fragment>(new Fragment(std::make_unique<Fragment::Impl>(*context.impl_)));
}

std::unique_ptr<std::string> stream_view_name(std::uint64_t stream_id)
{
  return std::make_unique<std::string>(stream_view_name_of(stream_id));
}

}  // namespace sirius::ffi
