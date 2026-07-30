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
#include "io/cache/types.hpp"

#include <data/data_batch_utils.hpp>
#include <data/sirius_converter_registry.hpp>
#include <helper/numeric_narrowing.hpp>
#include <log/logging.hpp>
#include <op/scan/gpu_ingestible.hpp>
#include <op/scan/sirius_gpu_scan_operator.hpp>
#include <op/scan/sirius_gpu_scan_operator_data.hpp>
#include <op/sirius_physical_operator.hpp>
#include <scan_manager/split_connector.hpp>
#include <sirius/exception.hpp>
#include <sirius_context.hpp>

// cudf
#include <cudf/cudf_utils.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/unary.hpp>
#include <cudf/utilities/memory_resource.hpp>

// cucascade
#include <cucascade/cudf/gpu_data_representation.hpp>
#include <cucascade/data/data_batch.hpp>
#include <cucascade/memory/memory_space.hpp>

// standard library
#include <algorithm>
#include <limits>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

namespace sirius::op::scan {
namespace {
constexpr std::size_t kMaxNumericCarrierExpansion = 8;

constexpr std::size_t saturating_add(std::size_t lhs, std::size_t rhs) noexcept
{
  auto const max = std::numeric_limits<std::size_t>::max();
  return rhs > max - lhs ? max : lhs + rhs;
}

constexpr std::size_t saturating_mul(std::size_t value, std::size_t factor) noexcept
{
  auto const max = std::numeric_limits<std::size_t>::max();
  if (value == 0 || factor == 0) { return 0; }
  return value > max / factor ? max : value * factor;
}

std::unique_ptr<cudf::table> normalize_physical_schema(std::unique_ptr<cudf::table> table,
                                                       const std::vector<cudf::data_type>& targets,
                                                       bool has_explicit_physical_schema,
                                                       duckdb::SiriusContext* observer,
                                                       rmm::cuda_stream_view stream,
                                                       rmm::device_async_resource_ref mr)
{
  if (targets.empty()) { return table; }

  auto const actual_width = static_cast<std::size_t>(table->num_columns());
  if (actual_width != targets.size()) {
    if (!has_explicit_physical_schema) { return table; }
    throw internal_exception(
      "[sirius_gpu_scan_operator] compressed schema width mismatch: materialized {} columns, "
      "planned {}",
      actual_width,
      targets.size());
  }

  // Preflight while every source column remains owned. Without a sidecar, resident batches may
  // only widen a narrow stored carrier to its native type. An explicit sidecar may also narrow a
  // freshly decoded carrier, but only after exact materialized bounds confirm that its values fit
  // the planned target.
  auto const table_view = table->view();
  for (std::size_t column_idx = 0; column_idx < targets.size(); column_idx++) {
    auto const& column = table_view.column(column_idx);
    auto const actual  = column.type();
    auto const target  = targets[column_idx];
    if (actual == target || can_restore_to(actual, target)) { continue; }

    if (!has_explicit_physical_schema) { continue; }
    if (!can_narrow_to(actual, target)) {
      throw internal_exception(
        "[sirius_gpu_scan_operator] invalid compressed carrier for column {}: materialized {}, "
        "planned {}",
        column_idx,
        cudf::type_to_name(actual),
        cudf::type_to_name(target));
    }

    bool const has_values = column.size() != 0 && column.null_count() != column.size();
    auto const exact = has_values ? compute_exact_numeric_range(column, stream, mr) : std::nullopt;
    if (has_values && (!exact || !numeric_range_fits(target, *exact))) {
      throw internal_exception(
        "[sirius_gpu_scan_operator] compressed-materialization metadata invariant violated for "
        "column {}: exact values from {} do not fit planned {}",
        column_idx,
        cudf::type_to_name(actual),
        cudf::type_to_name(target));
    }
  }

  auto columns = table->release();
  for (std::size_t column_idx = 0; column_idx < columns.size(); column_idx++) {
    auto const target    = targets[column_idx];
    auto const actual    = columns[column_idx]->type();
    auto const restoring = can_restore_to(actual, target);
    auto const narrowing = has_explicit_physical_schema && can_narrow_to(actual, target);
    if (actual == target || (!restoring && !narrowing)) { continue; }
    columns[column_idx] = cudf::cast(columns[column_idx]->view(), target, stream, mr);
    if (observer != nullptr) {
      if (narrowing) {
        observer->record_compressed_materialization_scan_columns_narrowed();
      } else {
        observer->record_compressed_materialization_scan_columns_restored();
      }
    }
    SIRIUS_LOG_DEBUG("[compressed_materialization] scan column {} {}: {} -> {}",
                     column_idx,
                     narrowing ? "narrowed" : "restored",
                     cudf::type_to_name(actual),
                     cudf::type_to_name(target));
  }
  return std::make_unique<cudf::table>(std::move(columns));
}

}  // namespace

//===----------------------------------------------------------------------===//
// sirius_gpu_scan_operator
//===----------------------------------------------------------------------===//
sirius_gpu_scan_operator::sirius_gpu_scan_operator(
  duckdb::vector<sirius::logical_type> types,
  duckdb::idx_t estimated_cardinality,
  std::shared_ptr<gpu_ingestible> ingestible,
  duckdb::SiriusContext* compressed_materialization_observer)
  : sirius_physical_operator(
      SiriusPhysicalOperatorType::GPU_SCAN, std::move(types), estimated_cardinality),
    _ingestible(std::move(ingestible)),
    _split_connector(std::make_shared<scan_manager::split_connector>()),
    _compressed_materialization_observer(compressed_materialization_observer)
{
  // `this->types` reads the base-class member; the constructor argument was consumed above.
  _native_physical_types.reserve(this->types.size());
  for (auto const& type : this->types) {
    auto const native = sirius::try_get_cudf_type(type);
    if (!native) {
      _native_physical_types.clear();
      break;
    }
    _native_physical_types.push_back(*native);
  }
}

sirius_gpu_scan_operator::~sirius_gpu_scan_operator() = default;

//===----------------------------------------------------------------------===//
// Source / scheduling interface
//===----------------------------------------------------------------------===//
std::optional<task_creation_hint> sirius_gpu_scan_operator::get_next_task_hint()
{
  if (_split_connector->is_closed()) { return std::nullopt; }
  return task_creation_hint{TaskCreationHint::READY, this};
}

bool sirius_gpu_scan_operator::all_ports_empty() { return _split_connector->is_closed(); }

std::unique_ptr<op::operator_data> sirius_gpu_scan_operator::get_next_task_input_data()
{
  auto next = _split_connector->get_next_split();
  if (!next.has_value()) { return nullptr; }
  if (auto* scan_input = dynamic_cast<scan_operator_input*>(next->get()); scan_input) {
    scan_input->prefetch(io::cache::prefetching_stage::immediate);
  }
  return std::move(*next);
}

//===----------------------------------------------------------------------===//
// scan_manager wiring
//===----------------------------------------------------------------------===//
gpu_ingestible& sirius_gpu_scan_operator::get_ingestible() const { return *_ingestible; }

scan_manager::split_connector& sirius_gpu_scan_operator::get_split_connector()
{
  return *_split_connector;
}

//===----------------------------------------------------------------------===//
// execute()
//===----------------------------------------------------------------------===//
std::unique_ptr<op::operator_data> sirius_gpu_scan_operator::execute(
  const op::operator_data& input_data, rmm::cuda_stream_view stream)
{
  auto scan_input = dynamic_cast<const scan_operator_input*>(&input_data);
  if (!scan_input) {
    throw std::runtime_error(
      "[sirius_gpu_scan_operator::execute] expected input of type scan_operator_input; got " +
      std::string(typeid(input_data).name()));
  }

  ::cucascade::memory::memory_space* mem_space = scan_input->gpu_memory_space;
  std::unique_ptr<cudf::table> output_table;
  auto materialized_table = _ingestible->materialize_table(*scan_input, stream);
  if (materialized_table.state != filter_state::ROW_FILTERED_AND_PROJECTED) {
    output_table =
      _ingestible->post_filter_and_project(std::move(materialized_table), *mem_space, stream);
  } else {
    output_table = materialized_table.table.release(stream, mem_space->get_default_allocator());
  }

  // Cast each batch column to its planned carrier. A resident chunk is normalized even without a
  // sidecar: it may be stored narrow (pinned with the feature on, queried with it off) and must
  // then restore to native.
  auto const has_explicit_physical_schema = has_physical_overrides();
  if (has_explicit_physical_schema || scan_input->is_resident()) {
    output_table = normalize_physical_schema(std::move(output_table),
                                             normalization_targets(),
                                             has_explicit_physical_schema,
                                             _compressed_materialization_observer,
                                             stream,
                                             mem_space->get_default_allocator());
  }
  auto batch =
    sirius::make_data_batch(std::move(output_table), *mem_space, stream, batch_telemetry());
  std::vector<std::shared_ptr<::cucascade::data_batch>> batches{std::move(batch)};
  return std::make_unique<pipelineable_operator_data>(std::move(batches));
}

std::size_t sirius_gpu_scan_operator::no_history_peak_memory_estimate(
  const op::input_stats& stats) const
{
  auto const expanded_bytes = saturating_mul(stats.bytes, kMaxNumericCarrierExpansion);
  if (stats.resident) {
    // The serve site compared each selected column's recorded carrier against the carrier this
    // scan plans for it, so a chunk reports a conversion only when normalize_physical_schema
    // will actually cast one. The cast destination coexists with the resident input/filter
    // working set at peak.
    if (stats.needs_carrier_conversion) {
      // The exact per-column destination, in the target widths cudf::cast allocates. A zero
      // means a converting chunk's row count was unreadable, so the maximum-expansion bound
      // stays.
      if (stats.conversion_destination_bytes > 0) {
        return saturating_add(stats.working_set_bytes, stats.conversion_destination_bytes);
      }
      return saturating_add(stats.working_set_bytes, expanded_bytes);
    }
    if (has_physical_overrides()) {
      // The serve site reported no cast, but this scan carries a plan sidecar. Keep a
      // conversion-sized headroom anyway, bounded by the stored source: a sidecar only ever
      // narrows a resident carrier, so any cast still reachable here fits in the source's bytes.
      // @kevin See issue ticket #1313 on whether this branch is dead code.
      return saturating_add(stats.working_set_bytes, stats.bytes);
    }
    return std::max(stats.bytes, stats.working_set_bytes);
  }

  // Preserve the maximum-width fresh-read heuristic for projected data, then add filter-only
  // columns without expanding the transient working set twice. Explicit physical sidecars retain
  // this conservative estimate because native decoded columns coexist with their narrow result.
  auto const filter_only_bytes =
    stats.working_set_bytes > stats.bytes ? stats.working_set_bytes - stats.bytes : 0;
  return saturating_add(expanded_bytes, filter_only_bytes);
}

}  // namespace sirius::op::scan
