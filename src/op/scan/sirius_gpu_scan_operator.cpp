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
#include <op/scan/owning_table_view.hpp>
#include <op/scan/sirius_gpu_scan_operator.hpp>
#include <op/scan/sirius_gpu_scan_operator_data.hpp>
#include <op/sirius_physical_operator.hpp>
#include <scan_manager/split_connector.hpp>
#include <sirius/exception.hpp>
#include <sirius_context.hpp>

// cudf
#include <cudf/column/column.hpp>
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
#include <any>
#include <limits>
#include <memory>
#include <numeric>
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

// Build one optional replacement per column. Validate the complete source/target shape before
// enqueuing any cast so an invalid schema cannot leave partially normalized output.
std::vector<std::unique_ptr<cudf::column>> normalize_physical_schema_casts(
  cudf::table_view const& view,
  const std::vector<cudf::data_type>& targets,
  bool has_explicit_physical_schema,
  duckdb::SiriusContext* observer,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  if (targets.empty()) { return {}; }

  auto const actual_width = static_cast<std::size_t>(view.num_columns());
  if (actual_width != targets.size()) {
    throw internal_exception(
      "[sirius_gpu_scan_operator] output schema width mismatch: materialized {} columns, expected "
      "{}",
      actual_width,
      targets.size());
  }

  // Preflight every column before casting any. Without a sidecar, resident batches may only widen
  // a narrow stored carrier to its native type. An explicit sidecar may also narrow a freshly
  // decoded carrier, but only after exact materialized bounds confirm that its values fit the
  // planned target.
  for (std::size_t column_idx = 0; column_idx < targets.size(); column_idx++) {
    auto const& column = view.column(static_cast<cudf::size_type>(column_idx));
    auto const actual  = column.type();
    auto const target  = targets[column_idx];
    if (actual == target || can_restore_to(actual, target)) { continue; }

    if (!has_explicit_physical_schema) {
      throw internal_exception(
        "[sirius_gpu_scan_operator] native schema carrier mismatch for column {}: materialized {}, "
        "expected {}",
        column_idx,
        cudf::type_to_name(actual),
        cudf::type_to_name(target));
    }
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

  std::vector<std::unique_ptr<cudf::column>> casts(targets.size());
  for (std::size_t column_idx = 0; column_idx < targets.size(); column_idx++) {
    auto const& column   = view.column(static_cast<cudf::size_type>(column_idx));
    auto const target    = targets[column_idx];
    auto const actual    = column.type();
    auto const restoring = can_restore_to(actual, target);
    auto const narrowing = has_explicit_physical_schema && can_narrow_to(actual, target);
    if (actual == target || (!restoring && !narrowing)) { continue; }
    casts[column_idx] = cudf::cast(column, target, stream, mr);
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
  return casts;
}

// Normalize an owned table in place, preserving it when no column needs a cast.
std::unique_ptr<cudf::table> normalize_physical_schema(std::unique_ptr<cudf::table> table,
                                                       const std::vector<cudf::data_type>& targets,
                                                       bool has_explicit_physical_schema,
                                                       duckdb::SiriusContext* observer,
                                                       rmm::cuda_stream_view stream,
                                                       rmm::device_async_resource_ref mr)
{
  auto casts = normalize_physical_schema_casts(
    table->view(), targets, has_explicit_physical_schema, observer, stream, mr);
  if (std::all_of(casts.begin(), casts.end(), [](auto const& cast) { return cast == nullptr; })) {
    return table;
  }
  // Swap the replacements in. The unique_ptr moves never relocate the columns the casts read
  // from, and a replaced source column is destroyed only after its cast was enqueued on the same
  // stream.
  auto columns = table->release();
  for (std::size_t column_idx = 0; column_idx < casts.size(); column_idx++) {
    if (casts[column_idx]) { columns[column_idx] = std::move(casts[column_idx]); }
  }
  return std::make_unique<cudf::table>(std::move(columns));
}

// Keeps both the surrendered pinned input and newly cast columns alive for a mixed output view.
// This must remain copy-constructible because make_data_batch_from_view stores it in std::any.
struct mixed_scan_owner {
  std::any pinned;
  std::shared_ptr<std::vector<std::unique_ptr<cudf::column>>> casted;
};

// Emit a pure forwarded view when no cast is needed, or combine forwarded and cast columns under a
// composite owner. Attribute cast allocations exactly and estimate only the referenced input
// columns, avoiding a full-input charge after projection.
std::shared_ptr<::cucascade::data_batch> emit_view_forward(
  owning_table_view::released_view forwarded,
  std::vector<std::unique_ptr<cudf::column>> casts,
  std::size_t total_input_bytes,
  ::cucascade::memory::memory_space& mem_space,
  rmm::cuda_stream_view stream,
  const telemetry::batch_telemetry_info& telemetry)
{
  auto const num_casted = static_cast<std::size_t>(
    std::count_if(casts.begin(), casts.end(), [](auto const& cast) { return cast != nullptr; }));

  if (num_casted == 0) {
    std::vector<cudf::size_type> all_columns(
      static_cast<std::size_t>(forwarded.view.num_columns()));
    std::iota(all_columns.begin(), all_columns.end(), 0);
    auto const referenced_bytes =
      sirius::estimate_referenced_column_bytes(forwarded.view, all_columns, total_input_bytes);
    return sirius::make_data_batch_from_view(
      forwarded.view, std::move(forwarded.owner), referenced_bytes, mem_space, stream, telemetry);
  }

  // Views are taken before the cast vector is moved (unique_ptr moves never relocate the
  // columns).
  std::vector<cudf::column_view> views;
  views.reserve(casts.size());
  std::vector<cudf::size_type> forwarded_columns;
  forwarded_columns.reserve(casts.size() - num_casted);
  std::size_t casted_bytes = 0;
  for (std::size_t column_idx = 0; column_idx < casts.size(); column_idx++) {
    if (casts[column_idx]) {
      views.push_back(casts[column_idx]->view());
      casted_bytes += casts[column_idx]->alloc_size();
    } else {
      views.push_back(forwarded.view.column(static_cast<cudf::size_type>(column_idx)));
      forwarded_columns.push_back(static_cast<cudf::size_type>(column_idx));
    }
  }
  auto const alloc_size = casted_bytes + sirius::estimate_referenced_column_bytes(
                                           forwarded.view, forwarded_columns, total_input_bytes);
  auto casted = std::make_shared<std::vector<std::unique_ptr<cudf::column>>>(std::move(casts));
  return sirius::make_data_batch_from_view(
    cudf::table_view{views},
    mixed_scan_owner{std::move(forwarded.owner), std::move(casted)},
    alloc_size,
    mem_space,
    stream,
    telemetry);
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
  _native_physical_types.reserve(this->types.size());
  for (std::size_t column_idx = 0; column_idx < this->types.size(); ++column_idx) {
    auto const& type  = this->types[column_idx];
    auto const native = sirius::try_get_cudf_type(type);
    if (!native) {
      throw internal_exception(
        "[sirius_gpu_scan_operator] output column {} ({}) has no native cuDF carrier",
        column_idx,
        type.to_string());
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
  auto const total_input_bytes                 = scan_input->get_estimated_size_in_bytes();
  auto materialized_table = _ingestible->materialize_table(*scan_input, stream);
  owning_table_view result =
    materialized_table.state != filter_state::ROW_FILTERED_AND_PROJECTED
      ? _ingestible->post_filter_and_project(std::move(materialized_table), *mem_space, stream)
      : std::move(materialized_table.table);

  auto const has_explicit_physical_schema = has_physical_overrides();
  auto const needs_normalization = has_explicit_physical_schema || scan_input->is_resident();

  std::shared_ptr<::cucascade::data_batch> batch;
  if (auto forwarded = result.release_view()) {
    // A copy-requiring view can transfer its owner instead of materializing; raw GPU-pinned inputs
    // are the production path that does so. Preserve that owner and cast only columns whose stored
    // carriers disagree with the plan.
    auto casts = needs_normalization
                   ? normalize_physical_schema_casts(forwarded->view,
                                                     normalization_targets(),
                                                     has_explicit_physical_schema,
                                                     _compressed_materialization_observer,
                                                     stream,
                                                     mem_space->get_default_allocator())
                   : std::vector<std::unique_ptr<cudf::column>>{};
    batch      = emit_view_forward(std::move(*forwarded),
                              std::move(casts),
                              total_input_bytes,
                              *mem_space,
                              stream,
                              batch_telemetry());
  } else {
    auto output_table = result.release(stream, mem_space->get_default_allocator());
    // Cast each batch column to its planned carrier. A resident chunk is normalized even without
    // a sidecar: it may be stored narrow (pinned with the feature on, queried with it off) and
    // must then restore to native.
    if (needs_normalization) {
      output_table = normalize_physical_schema(std::move(output_table),
                                               normalization_targets(),
                                               has_explicit_physical_schema,
                                               _compressed_materialization_observer,
                                               stream,
                                               mem_space->get_default_allocator());
    }
    batch = sirius::make_data_batch(std::move(output_table), *mem_space, stream, batch_telemetry());
  }
  std::vector<std::shared_ptr<::cucascade::data_batch>> batches{std::move(batch)};
  return std::make_unique<pipelineable_operator_data>(std::move(batches));
}

std::size_t sirius_gpu_scan_operator::resident_carrier_conversion_peak_memory_estimate(
  const op::input_stats& stats) noexcept
{
  D_ASSERT(stats.resident && stats.needs_carrier_conversion);
  auto destination_bytes = stats.conversion_destination_bytes;
  if (destination_bytes == 0) {
    destination_bytes = saturating_mul(stats.bytes, kMaxNumericCarrierExpansion);
  }
  return saturating_add(stats.working_set_bytes, destination_bytes);
}

std::size_t sirius_gpu_scan_operator::no_history_peak_memory_estimate(
  const op::input_stats& stats) const
{
  if (stats.resident) {
    // cached_databatch_provider sets this only when normalize_physical_schema will cast the split.
    if (stats.needs_carrier_conversion) {
      return resident_carrier_conversion_peak_memory_estimate(stats);
    }
    if (has_physical_overrides()) {
      // Keep an input-sized fallback for sidecar scans because per-split metadata may be
      // incomplete.
      return saturating_add(stats.working_set_bytes, stats.bytes);
    }
    return std::max(stats.bytes, stats.working_set_bytes);
  }

  // Fresh reads expand projected bytes by the maximum carrier factor. The working set may also
  // contain decoded filter-only columns, so add only the bytes beyond the projected input.
  auto const expanded_bytes = saturating_mul(stats.bytes, kMaxNumericCarrierExpansion);
  auto const filter_only_bytes =
    stats.working_set_bytes > stats.bytes ? stats.working_set_bytes - stats.bytes : 0;
  return saturating_add(expanded_bytes, filter_only_bytes);
}

}  // namespace sirius::op::scan
