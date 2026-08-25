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
#include <op/scan/parquet_gpu_ingestible.hpp>
#include <op/scan/sirius_gpu_scan_operator.hpp>
#include <op/scan/sirius_gpu_scan_operator_data.hpp>
#include <op/sirius_physical_operator.hpp>
#include <pipeline/data_size_estimator.hpp>
#include <scan_manager/split_connector.hpp>
#include <sirius/exception.hpp>
#include <sirius_context.hpp>

// cudf
#include <cudf/binaryop.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_stream.hpp>
#include <cudf/cudf_utils.hpp>
#include <cudf/filling.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/memory_resource.hpp>

// cucascade
#include <cucascade/cudf/gpu_data_representation.hpp>
#include <cucascade/data/data_batch.hpp>
#include <cucascade/memory/memory_space.hpp>

// standard library
#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <utility>
#include <vector>

namespace sirius::op::scan {
namespace {
constexpr std::size_t kMaxNumericCarrierExpansion = 8;

/// Emit the deferred positions as a rowid and placeholders instead of values.
///
/// The ids are a SEQUENCE over the origin's span, which is what makes this
/// cheap: a served batch is one pinned chunk's rows, in order, so row i of the
/// output is global row `range.start + i`. That equality is checked rather than
/// assumed — a batch whose rows were dropped somewhere would still produce a
/// column of the right length, holding ids for rows it no longer contains, and
/// the wrongness would only surface as plausible values at the far end.
///
/// This runs on the FINISHED scan output, which is why v1 refuses any scan that
/// restricts rows (see install_late_materialization). Admitting one means
/// substituting ahead of the filter's gather so the rowid is carried through it.
std::unique_ptr<cudf::table> substitute_deferred_columns(
  std::unique_ptr<cudf::table> output,
  late_mat::deferred_scan_output const& deferred,
  late_mat::scan_batch_origin const& origin,
  cudf::column_view const* survivors,
  std::size_t arity,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  auto const rows = output->num_rows();
  // Two shapes, and the check is what tells them apart. Unfiltered: the batch
  // IS the chunk, so row k is global row start + k. Filtered: the batch holds
  // the survivors, and row k is global row start + survivors[k] — which is why
  // a filtered batch without survivors must fail here rather than emit ids for
  // rows it no longer holds.
  if (survivors != nullptr) {
    if (survivors->size() != rows) {
      throw std::runtime_error("[sirius_gpu_scan_operator] a deferred scan emitted " +
                               std::to_string(rows) + " rows but captured " +
                               std::to_string(survivors->size()) + " survivor positions");
    }
  } else if (static_cast<std::int64_t>(rows) != origin.range.rows) {
    throw std::runtime_error("[sirius_gpu_scan_operator] a deferred scan emitted " +
                             std::to_string(rows) + " rows for a pinned chunk of " +
                             std::to_string(origin.range.rows) +
                             "; the rowid would address rows this batch no longer holds");
  }

  auto columns = output->release();
  // The scan may have ELIDED the deferred columns from the projection, so that
  // realizing the batch never copied values it was about to replace. Then they
  // are missing rather than present-and-overwritten, and this restores the
  // arity by inserting at their positions instead.
  bool const inserting = columns.size() + deferred.output_positions.size() == arity;
  if (!inserting && columns.size() != arity) {
    throw std::runtime_error("[sirius_gpu_scan_operator] a deferred scan produced " +
                             std::to_string(columns.size()) + " columns for an output of " +
                             std::to_string(arity));
  }
  if (inserting) {
    std::vector<std::unique_ptr<cudf::column>> widened;
    widened.reserve(arity);
    std::size_t next_kept = 0;
    for (std::size_t position = 0; position < arity; ++position) {
      if (deferred.defers(position)) {
        widened.push_back(nullptr);  // filled in below, in bundle order
      } else {
        widened.push_back(std::move(columns[next_kept++]));
      }
    }
    columns = std::move(widened);
  }
  for (std::size_t i = 0; i < deferred.output_positions.size(); ++i) {
    auto const position = deferred.output_positions[i];
    if (position >= columns.size()) {
      throw std::runtime_error("[sirius_gpu_scan_operator] deferred position " +
                               std::to_string(position) + " is outside the scan's output");
    }
    if (i == 0) {
      if (survivors != nullptr) {
        // start + survivor position, in the agreed width. cudf::binary_operation
        // casts the INT32 positions into the output type, so the narrow case
        // stays narrow.
        auto const base = cudf::numeric_scalar<std::uint64_t>(
          static_cast<std::uint64_t>(origin.range.start), true, stream, mr);
        auto const type = cudf::data_type{deferred.rowid_type};
        columns[position] =
          cudf::binary_operation(base, *survivors, cudf::binary_operator::ADD, type, stream, mr);
        continue;
      }
      // The width the pair agreed on: a pinned table whose rows fit 32 bits
      // rides half the bytes. The range check is not a formality — a wrong
      // width here wraps the id and materializes a plausible WRONG row.
      if (deferred.rowid_type == late_mat::kNarrowRowidType) {
        auto const last = origin.range.start + origin.range.rows;
        if (last > static_cast<std::int64_t>(std::numeric_limits<std::uint32_t>::max())) {
          throw std::runtime_error(
            "[sirius_gpu_scan_operator] a narrow rowid was installed for a pinned span reaching "
            "row " +
            std::to_string(last) + ", which does not fit 32 bits");
        }
        auto const init = cudf::numeric_scalar<std::uint32_t>(
          static_cast<std::uint32_t>(origin.range.start), true, stream, mr);
        auto const step   = cudf::numeric_scalar<std::uint32_t>(1, true, stream, mr);
        columns[position] = cudf::sequence(rows, init, step, stream, mr);
        continue;
      }
      auto const init = cudf::numeric_scalar<std::uint64_t>(
        static_cast<std::uint64_t>(origin.range.start), true, stream, mr);
      auto const step   = cudf::numeric_scalar<std::uint64_t>(1, true, stream, mr);
      columns[position] = cudf::sequence(rows, init, step, stream, mr);
      continue;
    }
    // A placeholder only has to keep the position occupied and the arity
    // unchanged. Zeroed rather than left uninitialized: nothing between the two
    // halves reads it, but a byte nobody defines is a byte that makes an
    // otherwise deterministic run differ.
    auto const zero   = cudf::numeric_scalar<std::int8_t>(0, true, stream, mr);
    columns[position] = cudf::make_column_from_scalar(zero, rows, stream, mr);
  }
  return std::make_unique<cudf::table>(std::move(columns));
}

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

enum class carrier_conversion_kind : uint8_t { NONE, RESTORE, NARROW };

struct carrier_conversion_plan {
  cudf::data_type actual;
  cudf::data_type target;
  carrier_conversion_kind kind;
};

/// Deferred positions are skipped: substitute_deferred_columns overwrites them with a rowid and
/// placeholders, so preflighting and casting them first is work whose result is discarded -- and
/// on a narrow-stored pin queried natively it is a full widen. `deferred` is null on the
/// transactional-steal path, which never runs for a scan carrying a late-materialization
/// deferral (see execute()).
std::vector<carrier_conversion_plan> preflight_physical_schema(
  cudf::table_view table,
  const std::vector<cudf::data_type>& targets,
  bool has_explicit_physical_schema,
  late_mat::deferred_scan_output const* deferred,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  auto const actual_width = static_cast<std::size_t>(table.num_columns());
  if (actual_width != targets.size()) {
    throw internal_exception(
      "[sirius_gpu_scan_operator] output schema width mismatch: materialized {} columns, expected "
      "{}",
      actual_width,
      targets.size());
  }

  std::vector<carrier_conversion_plan> plan;
  plan.reserve(targets.size());
  for (std::size_t column_idx = 0; column_idx < targets.size(); column_idx++) {
    if (deferred != nullptr && deferred->defers(column_idx)) {
      plan.push_back({cudf::data_type{}, cudf::data_type{}, carrier_conversion_kind::NONE});
      continue;
    }
    auto const& column = table.column(column_idx);
    auto const actual  = column.type();
    auto const target  = targets[column_idx];
    if (actual == target) {
      plan.push_back({actual, target, carrier_conversion_kind::NONE});
      continue;
    }
    if (can_restore_to(actual, target)) {
      plan.push_back({actual, target, carrier_conversion_kind::RESTORE});
      continue;
    }

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
    plan.push_back({actual, target, carrier_conversion_kind::NARROW});
  }
  return plan;
}

scan_operator_input::converted_column_replacements build_carrier_replacements(
  cudf::table_view source,
  const std::vector<carrier_conversion_plan>& plan,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  scan_operator_input::converted_column_replacements replacements(plan.size());
  for (std::size_t column_idx = 0; column_idx < plan.size(); ++column_idx) {
    if (plan[column_idx].kind == carrier_conversion_kind::NONE) { continue; }
    replacements[column_idx] =
      cast_through_rep(source.column(column_idx), plan[column_idx].target, stream, mr);
  }
  return replacements;
}

void record_carrier_conversions(const std::vector<carrier_conversion_plan>& plan,
                                duckdb::SiriusContext* observer)
{
  for (std::size_t column_idx = 0; column_idx < plan.size(); ++column_idx) {
    auto const& conversion = plan[column_idx];
    if (conversion.kind == carrier_conversion_kind::NONE) { continue; }
    auto const narrowing = conversion.kind == carrier_conversion_kind::NARROW;
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
                     cudf::type_to_name(conversion.actual),
                     cudf::type_to_name(conversion.target));
  }
}

/// `deferred` is null unless a late-materialization deferral is installed for this scan; passed
/// through to preflight_physical_schema so deferred positions are skipped rather than cast (they
/// hold a rowid/placeholder by the time this runs -- see execute()).
std::unique_ptr<cudf::table> normalize_physical_schema(
  std::unique_ptr<cudf::table> table,
  const std::vector<cudf::data_type>& targets,
  bool has_explicit_physical_schema,
  late_mat::deferred_scan_output const* deferred,
  duckdb::SiriusContext* observer,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  if (targets.empty()) { return table; }

  // Preflight while every source column remains owned. Without a sidecar, resident batches may
  // only widen a narrow stored carrier to its native type. An explicit sidecar may also narrow a
  // freshly decoded carrier, but only after exact materialized bounds confirm that its values fit
  // the planned target.
  auto const plan = preflight_physical_schema(
    table->view(), targets, has_explicit_physical_schema, deferred, stream, mr);

  auto columns = table->release();
  for (std::size_t column_idx = 0; column_idx < columns.size(); column_idx++) {
    if (plan[column_idx].kind == carrier_conversion_kind::NONE) { continue; }
    // The source's buffers may be dealloc-bound to the pinned-cache upload stream; rebind them
    // to `stream` so the free replacing the column queues behind the cast still reading them.
    auto casted =
      cast_through_rep(columns[column_idx]->view(), plan[column_idx].target, stream, mr);
    auto source         = cudf::rebind_stream(std::move(*columns[column_idx]), stream);
    columns[column_idx] = std::move(casted);
  }
  record_carrier_conversions(plan, observer);
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
  // Resolve the scan's dynamic-filter channel once (null for non-parquet
  // ingestibles): every split gets it stamped so prepare_for_processing can
  // snapshot membership filters at decode time.
  if (_ingestible != nullptr) {
    if (auto const* pq =
          dynamic_cast<parquet_ingestible_table_info const*>(&_ingestible->table_info())) {
      _dynamic_filters_channel = pq->sirius_dynamic_filters;
    }
  }
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

std::optional<std::size_t> sirius_gpu_scan_operator::total_source_input_bytes() const
{
  if (!_split_connector->is_discovery_complete()) { return std::nullopt; }
  // An unsized split may still emit rows, so discovered bytes are not a total.
  if (_split_connector->has_unsized_splits()) { return std::nullopt; }
  return _split_connector->discovered_bytes();
}

std::optional<std::size_t> sirius_gpu_scan_operator::total_source_output_bytes() const
{
  std::size_t rows  = 0;
  std::size_t bytes = 0;
  {
    std::lock_guard<std::mutex> guard(_emitted_mutex);
    rows  = _emitted_rows;
    bytes = _emitted_bytes;
  }
  return pipeline::project_source_output_bytes(estimated_cardinality, rows, bytes);
}

std::unique_ptr<op::operator_data> sirius_gpu_scan_operator::get_next_task_input_data()
{
  auto next = _split_connector->get_next_split();
  if (!next.has_value()) { return nullptr; }
  if (auto* scan_input = dynamic_cast<scan_operator_input*>(next->get()); scan_input) {
    // Share the operator's "compaction is unprofitable" latch with the split
    // BEFORE any reservation estimate runs: one such batch decides the whole
    // scan (uniform per-batch selectivity), and both the working-set estimator
    // and prepare_for_processing consult the latch.
    scan_input->pushdown_selection_unprofitable = _decode_selection_unprofitable;
    // Membership channel for the decode-time snapshot (join builds publish
    // during execution — only a snapshot taken at prepare/decode can see them).
    scan_input->dynamic_filters = _dynamic_filters_channel;
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
  auto const has_explicit_physical_schema      = has_physical_overrides();
  auto const& targets                          = normalization_targets();
  std::vector<carrier_conversion_plan> transactional_plan;
  std::unique_ptr<cudf::table> output_table;
  // A deferral needs to know which rows survived a filter applied here; asking
  // for them costs one gather over a row-index sequence, so only a deferring
  // scan asks.
  std::unique_ptr<cudf::column> survivors;
  auto const wants_survivors = !deferred_output().empty();

  // Only prepare_for_processing arms pending (pipeline contract: prepare runs before execute on
  // the same task), so a non-candidate split skips the builder lambda entirely. A
  // decode-row-filtered split may only bypass post_filter_and_project when the assembly it
  // skips is a leading identity — its width match alone cannot prove that, because trailing
  // pure-filter columns can offset synthesized (partition) output columns. An unfiltered split
  // needs no such proof: it decodes exactly the output columns, so a width match is decisive.
  // Refused outright when a late-materialization deferral is installed (wants_survivors): the
  // transactional steal bypasses post_filter_and_project entirely, so it can neither elide the
  // deferred columns from the projection nor hand substitute_deferred_columns the survivor
  // positions it needs -- that path always falls through to the branch below instead.
  if (!wants_survivors && scan_input->converted_table_steal_pending &&
      (!scan_input->pushdown_row_filtered || _ingestible->output_assembly_is_leading_identity())) {
    output_table = scan_input->transactionally_steal_converted_table(
      targets.size(),
      [&](cudf::table_view source) {
        transactional_plan = preflight_physical_schema(source,
                                                       targets,
                                                       has_explicit_physical_schema,
                                                       /*deferred=*/nullptr,
                                                       stream,
                                                       mem_space->get_default_allocator());
        return build_carrier_replacements(
          source, transactional_plan, stream, mem_space->get_default_allocator());
      },
      stream);
  }

  if (output_table) {
    // Observation follows the commit so a failed cast attempt cannot be counted twice on retry.
    record_carrier_conversions(transactional_plan, _compressed_materialization_observer);
  } else {
    // Every non-candidate path remains unchanged: materialize, filter/project if needed, then
    // normalize the resulting owned table.
    auto const like_swar_fastpath = like_swar_fastpath_enabled();
    auto like_pattern_cache       = like_cache();
    auto materialized_table =
      _ingestible->materialize_table(*scan_input, stream, like_swar_fastpath, like_pattern_cache);
    if (materialized_table.state != filter_state::ROW_FILTERED_AND_PROJECTED) {
      // Elide the deferred columns from the projection: realizing the batch would
      // copy values this scan is about to replace with a rowid.
      output_table = _ingestible->post_filter_and_project(std::move(materialized_table),
                                                          *mem_space,
                                                          stream,
                                                          like_swar_fastpath,
                                                          std::move(like_pattern_cache),
                                                          wants_survivors ? &survivors : nullptr,
                                                          deferred_output().output_positions);
    } else {
      output_table = materialized_table.table.release(stream, mem_space->get_default_allocator());
    }

    // Late materialization, producing half. Installed only where a served batch is
    // one pinned chunk's whole row span, so a batch arriving here without an
    // origin means the install-time conditions and the serve-time stamping have
    // drifted apart — and emitting values for one batch and rowids for the next
    // hands downstream two different schemas.
    if (wants_survivors) {
      if (!scan_input->origin) {
        throw std::runtime_error(
          "[sirius_gpu_scan_operator::execute] a deferral is installed but this split carries no "
          "origin; the scan cannot say where its rows came from");
      }
      std::optional<cudf::column_view> const survivor_view =
        survivors ? std::optional<cudf::column_view>{survivors->view()} : std::nullopt;
      output_table = substitute_deferred_columns(std::move(output_table),
                                                 deferred_output(),
                                                 *scan_input->origin,
                                                 survivor_view ? &*survivor_view : nullptr,
                                                 this->types.size(),
                                                 stream,
                                                 mem_space->get_default_allocator());
    }

    // A resident chunk is normalized even without a sidecar: it may be stored narrow (pinned with
    // the feature on, queried with it off) and must then restore to native. AFTER substitution:
    // the deferred positions then hold a rowid and placeholders, which it skips, and the elided
    // columns are back so the arity check holds.
    if (has_explicit_physical_schema || scan_input->is_resident()) {
      output_table = normalize_physical_schema(std::move(output_table),
                                               targets,
                                               has_explicit_physical_schema,
                                               wants_survivors ? &deferred_output() : nullptr,
                                               _compressed_materialization_observer,
                                               stream,
                                               mem_space->get_default_allocator());
    }
  }

  // Read the size before moving the table into a batch to avoid locking the batch. Publish rows
  // and bytes together because they form one sample.
  auto const emitted_rows  = static_cast<std::size_t>(output_table->num_rows());
  auto const emitted_bytes = output_table->alloc_size();

  auto batch =
    sirius::make_data_batch(std::move(output_table), *mem_space, stream, batch_telemetry());
  // Published only after make_data_batch succeeds: an OOM there reschedules the task into
  // re-running the same split, and these bytes floor total_source_output_bytes().
  {
    std::lock_guard<std::mutex> guard(_emitted_mutex);
    _emitted_rows += emitted_rows;
    _emitted_bytes += emitted_bytes;
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
