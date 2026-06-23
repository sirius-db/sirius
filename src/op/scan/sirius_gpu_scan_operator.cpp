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

// sirius
#include <data/data_batch_utils.hpp>
#include <data/sirius_converter_registry.hpp>
#include <log/logging.hpp>
#include <op/scan/sirius_gpu_scan_operator.hpp>
#include <op/scan/sirius_gpu_scan_operator_data.hpp>
#include <op/sirius_physical_operator.hpp>
#include <scan_manager/split_connector.hpp>

// cudf
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/memory_resource.hpp>

// cucascade
#include <cucascade/data/data_batch.hpp>
#include <cucascade/data/gpu_data_representation.hpp>
#include <cucascade/memory/memory_space.hpp>

// standard library
#include <stdexcept>
#include <utility>
#include <vector>

namespace sirius::op::scan {

//===----------------------------------------------------------------------===//
// scan_operator_with_pinned_table_input out-of-line members
//===----------------------------------------------------------------------===//
void scan_operator_with_pinned_table_input::prepare_for_processing(
  const ::cucascade::memory::memory_space* requested_memory_space, rmm::cuda_stream_view stream)
{
  // Convert host-tier cached batches onto the requested GPU memory space.
  // GPU-tier batches need no work — execute() consumes them directly.
  if (batch && requested_memory_space) {
    bool needs_upload = false;
    {
      auto ro      = batch->to_read_only();
      needs_upload = ro.get_data() && ro.get_current_tier() != ::cucascade::memory::Tier::GPU;
    }
    if (needs_upload) {
      auto& registry = ::sirius::converter_registry::get();
      auto mut       = batch->to_mutable();
      mut.convert_to<::cucascade::gpu_table_representation>(
        registry, requested_memory_space, stream);
    }
  }
  // Capture the actual memory_space the batch lives on (post any conversion)
  // so execute() can tag its output batch with the right placement.
  if (batch) {
    auto ro          = batch->to_read_only();
    gpu_memory_space = ro.get_memory_space();
  }
}

std::size_t scan_operator_with_pinned_table_input::get_estimated_size_in_bytes() const
{
  if (!batch) { return 0; }
  // Use the generic representation accessor so this works for both GPU-resident
  // batches and host_data_representation batches that prepare_for_processing
  // will upgrade to GPU.
  auto ro          = batch->to_read_only();
  auto const* data = ro.get_data();
  if (!data) { return 0; }
  return data->get_size_in_bytes();
}

//===----------------------------------------------------------------------===//
// sirius_gpu_scan_operator
//===----------------------------------------------------------------------===//
sirius_gpu_scan_operator::sirius_gpu_scan_operator(
  duckdb::vector<sirius::logical_type> types,
  duckdb::idx_t estimated_cardinality,
  std::unique_ptr<io::ingestible_table_info> table_info)
  : sirius_physical_operator(
      SiriusPhysicalOperatorType::GPU_SCAN, std::move(types), estimated_cardinality),
    _table_info(std::move(table_info)),
    _split_connector(std::make_unique<scan_manager::split_connector>())
{
  // Start with a closed connector so that an operator wired into a pipeline but
  // never prepared (e.g. fallback to CPU) reports all_ports_empty() and yields
  // nullopt from get_next_task_hint(). The scan_manager replaces this connector
  // with a fresh open one during prepare_for_query.
  _split_connector->close();
}

sirius_gpu_scan_operator::~sirius_gpu_scan_operator() = default;

//===----------------------------------------------------------------------===//
// Source / scheduling interface
//===----------------------------------------------------------------------===//
std::optional<task_creation_hint> sirius_gpu_scan_operator::get_next_task_hint()
{
  if (_split_connector->is_closed()) { return std::nullopt; }
  // Returns READY even when the queue is empty but not yet closed; the dispatched
  // worker parks in split_connector::get_next_split until a split arrives or the
  // connector is closed. See sirius_gpu_parquet_scan_operator::get_next_task_hint
  // for the deeper lifecycle note this preserves.
  return task_creation_hint{TaskCreationHint::READY, this};
}

bool sirius_gpu_scan_operator::all_ports_empty()
{
  // Done when the connector is closed and the ingestible has no buffered work.
  // The unprepared path has no ingestible.
  return _split_connector->is_closed() && (!_ingestible || _ingestible->consumer_drained());
}

std::unique_ptr<op::operator_data> sirius_gpu_scan_operator::get_next_task_input_data()
{
  // Hold the scan-order mutex for the entire pull + batch-ID assignment so that
  // pre_assigned_batch_id values reflect split-pull order (scan order).
  // Without this, concurrent callers could race: thread A pulls split 0 then
  // thread B pulls split 1, but B calls get_next_batch_id() first, giving split 0
  // a higher ID than split 1 and breaking the ordering that first() relies on.
  std::lock_guard<std::mutex> lg(_scan_order_mutex_);

  std::unique_ptr<op::operator_data> input;
  if (_ingestible) {
    input = _ingestible->consume_next_input(*_split_connector);
  } else {
    auto next = _split_connector->get_next_split();
    if (!next.has_value()) { return nullptr; }
    input = std::move(*next);
  }

  if (input) { input->pre_assigned_batch_id = get_next_batch_id(); }
  return input;
}

//===----------------------------------------------------------------------===//
// scan_manager wiring
//===----------------------------------------------------------------------===//
std::unique_ptr<io::ingestible_table_info> sirius_gpu_scan_operator::take_table_info()
{
  return std::move(_table_info);
}

void sirius_gpu_scan_operator::install_ingestible(std::shared_ptr<io::gpu_ingestible> ingestible)
{
  _ingestible = std::move(ingestible);
}

io::gpu_ingestible& sirius_gpu_scan_operator::get_ingestible() const
{
  if (!_ingestible) {
    throw std::runtime_error(
      "[sirius_gpu_scan_operator] get_ingestible() called before install_ingestible(); "
      "scan_manager::prepare_for_query must run first.");
  }
  return *_ingestible;
}

void sirius_gpu_scan_operator::set_split_connector(std::unique_ptr<scan_manager::split_connector> c)
{
  _split_connector = std::move(c);
}

//===----------------------------------------------------------------------===//
// execute()
//===----------------------------------------------------------------------===//
std::unique_ptr<op::operator_data> sirius_gpu_scan_operator::execute(
  const op::operator_data& input_data, rmm::cuda_stream_view stream)
{
  if (!_ingestible) {
    throw std::runtime_error(
      "[sirius_gpu_scan_operator::execute] no ingestible installed; "
      "scan_manager::prepare_for_query must run before execute().");
  }

  std::unique_ptr<cudf::table> table;
  ::cucascade::memory::memory_space* mem_space = nullptr;

  if (auto const* fresh = dynamic_cast<scan_operator_input const*>(&input_data)) {
    if (!fresh->metadata) {
      throw std::runtime_error("[sirius_gpu_scan_operator::execute] fresh input missing metadata.");
    }
    if (!fresh->gpu_memory_space) {
      throw std::runtime_error(
        "[sirius_gpu_scan_operator::execute] fresh input missing gpu_memory_space; "
        "prepare_for_processing must run before execute().");
    }
    auto const& md    = *fresh->metadata;
    auto materialized = _ingestible->materialize_table(md.scan(), *fresh->gpu_memory_space, stream);
    if (md.has_filter() && materialized.state != io::filter_state::ROW_FILTERED_AND_PROJECTED) {
      table = _ingestible->post_filter_and_project(
        std::move(materialized.table), md.filter_and_project(), *fresh->gpu_memory_space, stream);
    } else {
      table = std::move(materialized.table);
    }
    mem_space = fresh->gpu_memory_space;
  } else if (auto const* pinned =
               dynamic_cast<scan_operator_with_pinned_table_input const*>(&input_data)) {
    if (!pinned->batch) {
      throw std::runtime_error(
        "[sirius_gpu_scan_operator::execute] pinned input missing data_batch.");
    }
    if (!pinned->filter_info) {
      // Fast path: forward the cached batch unchanged. Zero-copy; pinned columns
      // remain co-owned via shared_ptr<column> within the batch.
      std::vector<std::shared_ptr<::cucascade::data_batch>> batches;
      batches.push_back(pinned->batch);
      return std::make_unique<pipelineable_operator_data>(std::move(batches));
    }
    auto ro_batch  = pinned->batch->to_read_only();
    auto* batch_mr = ro_batch.get_memory_space();
    if (!batch_mr) {
      throw std::runtime_error(
        "[sirius_gpu_scan_operator::execute] pinned batch has no memory_space; "
        "prepare_for_processing must have produced a GPU-resident batch.");
    }
    auto& gpu_rep     = ro_batch.get_data()->cast<::cucascade::gpu_table_representation>();
    auto owning_input = gpu_rep.release_table(stream);

    table = _ingestible->post_filter_and_project(
      std::move(owning_input), *pinned->filter_info, *batch_mr, stream);
    mem_space = batch_mr;
  } else {
    throw std::runtime_error(
      "[sirius_gpu_scan_operator::execute] unexpected operator_data type; expected "
      "scan_operator_input or scan_operator_with_pinned_table_input.");
  }

  // Use the batch ID pre-assigned in get_next_task_input_data() rather than calling
  // get_next_batch_id() here. Pre-assignment happens under _scan_order_mutex_ so IDs
  // reflect split-pull order (scan order); assigning here would be completion order.
  auto batch =
    sirius::make_data_batch(std::move(table), *mem_space, stream, input_data.pre_assigned_batch_id);
  std::vector<std::shared_ptr<::cucascade::data_batch>> batches;
  batches.push_back(std::move(batch));
  return std::make_unique<pipelineable_operator_data>(std::move(batches));
}

std::size_t sirius_gpu_scan_operator::no_history_peak_memory_estimate(
  const op::input_stats& stats) const
{
  // Match the legacy heuristics: pinned (cached) inputs are pass-throughs in the
  // common case, so the estimate equals the input size. Fresh reads expand the
  // input substantially (decompression + decode), so the parquet operator used
  // an 8× factor. duckdb-native used 4×. Pick 8× as the safe upper bound — the
  // reservation system clamps via downstream operator estimates anyway.
  if (stats.resident) { return stats.bytes; }
  return stats.bytes * 8;
}

}  // namespace sirius::op::scan
