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
#include <cucascade/io/cache/types.hpp>
#include <data/data_batch_utils.hpp>
#include <data/sirius_converter_registry.hpp>
#include <helper/numeric_narrowing.hpp>
#include <log/logging.hpp>
#include <memory/size_arithmetic.hpp>
#include <op/scan/detail/read_retirement.hpp>
#include <op/scan/gpu_ingestible.hpp>
#include <op/scan/scan_output_operator_data.hpp>
#include <op/scan/sirius_gpu_scan_operator.hpp>
#include <op/scan/sirius_gpu_scan_operator_data.hpp>
#include <op/sirius_physical_operator.hpp>
#include <pipeline/data_size_estimator.hpp>
#include <scan_manager/split_connector.hpp>
#include <sirius/exception.hpp>
#include <sirius_context.hpp>
#include <transparent/read_view_registry.hpp>

// cudf
#include <cudf/binaryop.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_stream.hpp>
#include <cudf/cudf_utils.hpp>
#include <cudf/filling.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/memory_resource.hpp>

// cucascade
#include <cucascade/cudf/gpu_data_representation.hpp>
#include <cucascade/data/data_batch.hpp>
#include <cucascade/memory/memory_space.hpp>

// standard library
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

namespace sirius::op::scan {
namespace {
constexpr std::size_t kMaxNumericCarrierExpansion = 8;

using sirius::memory::saturating_add;
using sirius::memory::saturating_mul;

struct scan_assembly_state {
  std::unique_ptr<cudf::table> output;
  std::unique_ptr<cudf::column> survivors;
  std::vector<std::unique_ptr<cudf::column>> columns;
};

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
void substitute_deferred_columns(scan_assembly_state& state,
                                 late_mat::deferred_scan_output const& deferred,
                                 late_mat::scan_batch_origin const& origin,
                                 cudf::column_view const* survivors,
                                 std::size_t arity,
                                 ::cuda::stream_ref stream,
                                 rmm::device_async_resource_ref mr)
{
  auto& output    = state.output;
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

  auto& columns = state.columns;
  columns       = output->release();
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
  output = std::make_unique<cudf::table>(std::move(columns));
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
  ::cuda::stream_ref stream,
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
  ::cuda::stream_ref stream,
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
        observer->get_event_publisher().publish_compressed_materialization(
          sirius::event::compressed_materialization_activity::scan_columns_narrowed);
      } else {
        observer->get_event_publisher().publish_compressed_materialization(
          sirius::event::compressed_materialization_activity::scan_columns_restored);
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
void normalize_physical_schema(scan_assembly_state& state,
                               const std::vector<cudf::data_type>& targets,
                               bool has_explicit_physical_schema,
                               late_mat::deferred_scan_output const* deferred,
                               duckdb::SiriusContext* observer,
                               ::cuda::stream_ref stream,
                               rmm::device_async_resource_ref mr)
{
  auto& table = state.output;
  if (targets.empty()) { return; }

  // Preflight while every source column remains owned. Without a sidecar, resident batches may
  // only widen a narrow stored carrier to its native type. An explicit sidecar may also narrow a
  // freshly decoded carrier, but only after exact materialized bounds confirm that its values fit
  // the planned target.
  auto const plan = preflight_physical_schema(
    table->view(), targets, has_explicit_physical_schema, deferred, stream, mr);

  // Each source is freed as soon as its replacement is installed, so the peak holds one source
  // beside the casts. A failed cast leaves the sources it has not replaced in the assembly.
  auto& columns = state.columns;
  columns       = table->release();
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
  table = std::make_unique<cudf::table>(std::move(columns));
}

/// The receipt @p scan starts from: the decode history of @p input's in-range output ordinals
batch_receipt decode_receipt(sirius_gpu_scan_operator const& scan, scan_operator_input const& input)
{
  // History describes rows, not column contents: a deferral preserves row positions and restores
  // the values before the endpoint reads them, so deferred ordinals keep their entries too.
  batch_receipt receipt;
  if (scan.consumer()) {
    receipt.endpoint = scan.consumer()->channel();
    for (auto const& entry : input.decode_attached) {
      if (entry.output_ordinal < scan.types.size()) { receipt.decode_attached.push_back(entry); }
    }
  }
  return receipt;
}

/// @brief Moves the table that `prepare_for_processing` freshly converted out of @p input's cached
/// wrapper, casting its carriers to @p scan's normalization targets on the way
///
/// Returns null, leaving @p input untouched, when @p input is not a candidate.
std::unique_ptr<cudf::table> transfer_converted_table(sirius_gpu_scan_operator const& scan,
                                                      scan_operator_input const& input,
                                                      ::cuda::stream_ref stream)
{
  // Only prepare_for_processing arms pending (pipeline contract: prepare runs before execute on
  // the same task), so a non-candidate split skips the builder lambda entirely. A
  // decode-row-filtered split may only bypass post_filter_and_project when the assembly it
  // skips is a leading identity -- its width match alone cannot prove that, because trailing
  // pure-filter columns can offset synthesized (partition) output columns. An unfiltered split
  // needs no such proof: it decodes exactly the output columns, so a width match is decisive.
  // Refused outright when a late-materialization deferral is installed: the transfer bypasses
  // post_filter_and_project entirely, so it can neither elide the deferred columns from the
  // projection nor hand substitute_deferred_columns the survivor positions it needs.
  if (!scan.deferred_output().empty() || !input.converted_table_steal_pending ||
      (input.pushdown_row_filtered &&
       !scan.get_ingestible().output_assembly_is_leading_identity())) {
    return nullptr;
  }
  auto const& targets = scan.normalization_targets();
  auto const mr       = input.gpu_memory_space->get_default_allocator();
  std::vector<carrier_conversion_plan> plan;
  auto table = input.transactionally_steal_converted_table(
    targets.size(),
    [&](cudf::table_view source) {
      plan = preflight_physical_schema(
        source, targets, scan.has_physical_overrides(), /*deferred=*/nullptr, stream, mr);
      return build_carrier_replacements(source, plan, stream, mr);
    },
    stream);
  // Observation follows the commit so a failed cast attempt cannot be counted twice on retry.
  if (table) { record_carrier_conversions(plan, scan.certificate_observer()); }
  return table;
}

/// @brief Materializes @p input through @p scan's ingestible
///
/// A deferring scan's late materialization addresses whole pinned chunks, so its split must carry
/// an origin and keep every row of that chunk.
filtered_table materialize_source(sirius_gpu_scan_operator const& scan,
                                  scan_operator_input const& input,
                                  ::cuda::stream_ref stream)
{
  auto source = scan.get_ingestible().materialize_table(
    input, stream, scan.like_swar_fastpath_enabled(), scan.like_cache(), scan.expression_policy());
  if (scan.deferred_output().empty()) { return source; }
  // Installed only where a served batch is one pinned chunk's whole row span, so a batch arriving
  // here without an origin means the install-time conditions and the serve-time stamping have
  // drifted apart -- and emitting values for one batch and rowids for the next hands downstream
  // two different schemas.
  if (!input.origin) {
    throw std::runtime_error(
      "[sirius_gpu_scan_operator::execute] a deferral is installed but this split carries no "
      "origin; the scan cannot say where its rows came from");
  }
  // Installation refuses static filters on compressed pins, and decode_consumer() withholds
  // dynamic membership probes, so a deferring scan's decode never removes rows. Survivor positions
  // from a later filter index the decoded rows, so those rows must be the whole chunk. Membership
  // compaction can remove rows without reporting pushdown_row_filtered, so the row count is the
  // evidence.
  if (std::cmp_not_equal(source.table.num_rows(), input.origin->range.rows)) {
    throw internal_exception(
      "[sirius_gpu_scan_operator::execute] the decode removed rows of a scan whose deferral "
      "addresses whole chunks: {} of {} rows remain (static filter applied: {}, membership "
      "probes offered: {})",
      source.table.num_rows(),
      input.origin->range.rows,
      input.pushdown_row_filtered,
      input.decode_attached.size());
  }
  return source;
}

/// The cached checkpoint's application, kept for `dynamic_filter_consumer::finish`, and the
/// placement counters of one `sirius_gpu_scan_operator::execute`
struct row_selection {
  std::optional<application_result> cached;
  sirius::op::dynamic_filter_stats_snapshot placement;
};

/// @brief Runs @p scan's cached dynamic-filter checkpoint over the borrowed resident @p source
///
/// A selection places the projected survivors, and their positions when a deferral needs them, in
/// @p assembly. Otherwise @p source gets its rows back for the ordinary projection.
row_selection run_cached_checkpoint(sirius_gpu_scan_operator const& scan,
                                    scan_operator_input const& input,
                                    batch_receipt const& receipt,
                                    filtered_table& source,
                                    scan_assembly_state& assembly,
                                    ::cuda::stream_ref stream)
{
  row_selection selection;
  auto& placement = selection.placement;
  auto layout = scan.get_ingestible().view_layout(source, scan.deferred_output().output_positions);
  if (!layout) {
    placement.cached_declined_no_layout = 1;
    return selection;
  }
  std::vector<binding> bindings;
  bindings.reserve(layout->output_columns.size());
  for (std::size_t ordinal = 0; ordinal < layout->output_columns.size(); ++ordinal) {
    bindings.push_back({ordinal, layout->output_columns[ordinal]});
  }
  auto const& consumer = *scan.consumer();
  auto* const space    = input.gpu_memory_space;
  auto const view      = source.table.view();
  auto const input_bytes =
    sirius::estimate_referenced_column_bytes(view, layout->projected_columns, source.stored_bytes);
  auto program = consumer.prepare({.source             = view,
                                   .bindings           = bindings,
                                   .input_bytes        = input_bytes,
                                   .prior              = &receipt,
                                   .device_id          = space->get_device_id(),
                                   .cached_layout      = &*layout,
                                   .want_row_positions = !scan.deferred_output().empty()});
  if (!program) {
    placement.cached_declined_no_membership = 1;
    return selection;
  }
  auto const input_rows = source.table.num_rows();
  auto& result          = selection.cached.emplace(consumer.apply(std::move(*program),
                                                         std::move(source.table),
                                                         stream,
                                                         space->get_default_allocator(),
                                                         scan.like_swar_fastpath_enabled(),
                                                         scan.like_cache(),
                                                         scan.expression_policy()));
  assembly.output       = result.take_output();
  assembly.survivors    = result.take_row_positions();
  if (!assembly.output) {
    source.table = result.take_source();
    // A batch on another device passed through without running the program; the consumer counts it
    // in applications_skipped_other_device.
    if (result.completed()) { placement.cached_declined_no_membership = 1; }
    return selection;
  }
  auto const applied = result.contributing_steps();
  // Every probe declined, so the residual alone selected the rows: no membership step ran.
  if (applied == 0) {
    placement.cached_declined_no_membership = 1;
    return selection;
  }
  placement.cached_placements          = 1;
  placement.cached_payload_bytes       = assembly.output->alloc_size();
  placement.cached_position_selections = assembly.survivors ? 1 : 0;
  SIRIUS_LOG_DEBUG(
    "[sirius_gpu_scan_operator::execute] cached checkpoint: {} -> {} rows with {} dynamic filters "
    "folded into the gather",
    input_rows,
    assembly.output->num_rows(),
    applied);
  return selection;
}

/// @brief Filters and projects @p input into @p assembly
///
/// A split that `scan_operator_input::cached_checkpoint_possible` admits first offers its rows to
/// the cached dynamic-filter checkpoint; rows it does not select take the ingestible's ordinary
/// residual filter and projection.
row_selection select_rows(sirius_gpu_scan_operator const& scan,
                          scan_operator_input const& input,
                          batch_receipt const& receipt,
                          scan_assembly_state& assembly,
                          ::cuda::stream_ref stream)
{
  bool const checkpoint = input.cached_checkpoint_possible();
  bool const borrowed   = input.borrows_resident_batch();
  auto source           = materialize_source(scan, input, stream);
  row_selection selection;
  if (checkpoint) {
    selection = run_cached_checkpoint(scan, input, receipt, source, assembly, stream);
  } else if (auto const& consumer = scan.consumer(); consumer && borrowed) {
    // The scan cannot describe a cached view of this borrowed batch.
    if (consumer->cached_placement_enabled()) {
      selection.placement.cached_declined_no_layout = 1;
    } else {
      selection.placement.cached_disabled = 1;
    }
  }
  if (assembly.output) { return selection; }
  auto const& space = *input.gpu_memory_space;
  if (source.state == filter_state::ROW_FILTERED_AND_PROJECTED) {
    assembly.output = source.table.release(stream, space.get_default_allocator());
    return selection;
  }
  // Elide the deferred columns from the projection: realizing the batch would copy values this scan
  // is about to replace with a rowid.
  auto const& deferred = scan.deferred_output();
  assembly.output =
    scan.get_ingestible().post_filter_and_project(std::move(source),
                                                  space,
                                                  stream,
                                                  scan.like_swar_fastpath_enabled(),
                                                  scan.like_cache(),
                                                  deferred.empty() ? nullptr : &assembly.survivors,
                                                  deferred.output_positions,
                                                  scan.expression_policy());
  return selection;
}

/// @brief Restores @p scan's deferred outputs and normalizes the selected rows of @p input to the
/// planned carriers
void finish_assembly(sirius_gpu_scan_operator const& scan,
                     scan_operator_input const& input,
                     scan_assembly_state& assembly,
                     ::cuda::stream_ref stream)
{
  auto const mr        = input.gpu_memory_space->get_default_allocator();
  auto const& deferred = scan.deferred_output();
  if (!deferred.empty()) {
    std::optional<cudf::column_view> const survivors =
      assembly.survivors ? std::optional{assembly.survivors->view()} : std::nullopt;
    substitute_deferred_columns(assembly,
                                deferred,
                                *input.origin,
                                survivors ? &*survivors : nullptr,
                                scan.types.size(),
                                stream,
                                mr);
    detail::check_read_checkpoint(detail::read_checkpoint::SUBSTITUTION);
  }
  // A resident chunk is normalized even without a sidecar: it may be stored narrow (pinned with the
  // feature on, queried with it off) and must then restore to native. AFTER substitution: the
  // deferred positions then hold a rowid and placeholders, which it skips, and the elided columns
  // are back so the arity check holds.
  if (scan.has_physical_overrides() || input.is_resident()) {
    normalize_physical_schema(assembly,
                              scan.normalization_targets(),
                              scan.has_physical_overrides(),
                              deferred.empty() ? nullptr : &deferred,
                              scan.certificate_observer(),
                              stream,
                              mr);
    detail::check_read_checkpoint(detail::read_checkpoint::NORMALIZATION);
  }
}

}  // namespace

std::optional<detail::retained_scan_assembly_storage>
detail::scan_operator_test_access::retained_assembly(
  read_retirement_test_hooks const& hooks) noexcept
{
  auto const* state = hooks.retained.get<scan_assembly_state>();
  if (state == nullptr) { return std::nullopt; }
  return retained_scan_assembly_storage{state->output.get(), state->survivors.get()};
}

//===----------------------------------------------------------------------===//
// sirius_gpu_scan_operator
//===----------------------------------------------------------------------===//
std::vector<binding> scan_decode_bindings(gpu_ingestible const& ingestible,
                                          std::size_t output_width)
{
  if (!ingestible.output_assembly_is_leading_identity() ||
      output_width > ingestible.materialized_column_order().size()) {
    return {};
  }
  return identity_bindings(output_width);
}

sirius_gpu_scan_operator::sirius_gpu_scan_operator(
  duckdb::vector<sirius::logical_type> types,
  duckdb::idx_t estimated_cardinality,
  std::shared_ptr<gpu_ingestible> ingestible,
  scan_contract_id contract_id,
  duckdb::SiriusContext* compressed_materialization_observer,
  std::shared_ptr<transparent::read_view_registry> read_views,
  std::shared_ptr<dynamic_filter_consumer> consumer)
  : sirius_physical_operator(
      SiriusPhysicalOperatorType::GPU_SCAN, std::move(types), estimated_cardinality),
    _ingestible(std::move(ingestible)),
    _read_views(std::move(read_views)),
    _contract_id(contract_id),
    _split_connector(std::make_shared<scan_manager::split_connector>()),
    _dynamic_filter_consumer(std::move(consumer)),
    _compressed_materialization_observer(compressed_materialization_observer)
{
  if (_contract_id == 0) {
    throw std::invalid_argument("GPU scan operator requires a nonzero scan contract ID");
  }

  _native_physical_types.reserve(this->types.size());
  for (std::size_t column_idx = 0; column_idx < this->types.size(); ++column_idx) {
    auto const& type  = this->types[column_idx];
    auto const native = sirius::get_cudf_type(type);
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
    if (scan_input->is_resident()) validate_input(*scan_input, true);
    // Share the operator's "compaction is unprofitable" latch with the split
    // BEFORE any reservation estimate runs: one such batch decides the whole
    // scan (uniform per-batch selectivity), and both the working-set estimator
    // and prepare_for_processing consult the latch.
    scan_input->pushdown_selection_unprofitable = _decode_selection_unprofitable;
    scan_input->consumer                        = decode_consumer();
    scan_input->cached_view_possible            = cached_view_possible();
    scan_input->update(cucascade::io::cache::scan_stage::queued);
  }
  return std::move(*next);
}

bool sirius_gpu_scan_operator::cached_view_possible() const noexcept
{
  return _dynamic_filter_consumer && _dynamic_filter_consumer->cached_placement_enabled() &&
         _ingestible && _ingestible->can_serve_cached_view();
}

//===----------------------------------------------------------------------===//
// scan_manager wiring
//===----------------------------------------------------------------------===//
gpu_ingestible& sirius_gpu_scan_operator::get_ingestible() const { return *_ingestible; }

void sirius_gpu_scan_operator::validate_input(scan_operator_input const& input,
                                              bool admit_resident) const
{
  if (!_contract_id) return;  // standalone unbound operator fixtures
  try {
    if (input.has_scan_metadata()) {
      auto required =
        _read_views ? _read_views->entry(_contract_id).eligibility.later_checks : later_check_set{};
      validate_split_for_gpu(_contract_id, required, _expected_key, input.get_scan_info());
      if (!_read_views)
        throw certificate_incomplete(_contract_id, {}, "bound scan registry missing");
      for (auto const& dependency : input.get_scan_info().dependencies()) {
        if (dependency.profiles && dependency.profiles != _read_views->profiles)
          throw certificate_incomplete(
            _contract_id, {}, "physical profile belongs to another scan registry");
      }
    } else if (input.is_resident()) {
      if (input.resident_contract_id != _contract_id)
        throw transparent::classified_execution_error(transparent::late_failure_cause::certificate,
                                                      "resident scan contract mismatch");
      if (!input.resident_validation)
        throw certificate_incomplete(_contract_id, {}, "resident admission missing");
      if (_ingestible &&
          dynamic_cast<duckdb_native_ingestible_table_info const*>(&_ingestible->table_info()) &&
          (!input.resident_validation->iteration.applies ||
           !input.resident_validation->visibility.applies))
        throw certificate_incomplete(_contract_id, {}, "resident native applicability missing");
      if (admit_resident) {
        admit_resident_batch(_contract_id, *input.resident_validation, _query_token);
      } else if (input.resident_validation->query_token != _query_token) {
        throw certificate_incomplete(_contract_id, {}, "resident query token mismatch");
      }
    } else {
      throw certificate_incomplete(_contract_id, {}, "scan payload missing");
    }
  } catch (certificate_incomplete const&) {
    if (_compressed_materialization_observer)
      _compressed_materialization_observer->record_transparent_certificate_incomplete();
    throw;
  } catch (...) {
    if (_compressed_materialization_observer)
      _compressed_materialization_observer->record_transparent_certificate_mismatch();
    throw;
  }
}

bound_table_scan const& sirius_gpu_scan_operator::scan_contract() const
{
  if (!_read_views || _contract_id == 0) {
    throw std::runtime_error("GPU scan operator has no bound scan contract");
  }
  return contract_of(*_read_views, _contract_id);
}

scan_manager::split_connector& sirius_gpu_scan_operator::get_split_connector()
{
  return *_split_connector;
}

//===----------------------------------------------------------------------===//
// execute()
//===----------------------------------------------------------------------===//
std::unique_ptr<op::operator_data> sirius_gpu_scan_operator::execute(
  const op::operator_data& input_data, ::cuda::stream_ref stream)
{
  auto const* scan_input = dynamic_cast<const scan_operator_input*>(&input_data);
  if (!scan_input) {
    throw std::runtime_error(
      "[sirius_gpu_scan_operator::execute] expected input of type scan_operator_input; got " +
      std::string(typeid(input_data).name()));
  }
  validate_input(*scan_input, false);

  auto receipt        = decode_receipt(*this, *scan_input);
  auto assembly       = std::make_unique<scan_assembly_state>();
  auto const unjoined = detail::prepare_unjoined_failure();
  row_selection selection;
  try {
    // An owned table, transferred here or stolen by prepare_for_processing, bypasses the cached
    // checkpoint.
    assembly->output = transfer_converted_table(*this, *scan_input, stream);
    if (assembly->output) {
      selection.placement.cached_bypassed_transfer = 1;
    } else {
      bool const stolen = scan_input->stolen_table != nullptr;
      selection         = select_rows(*this, *scan_input, receipt, *assembly, stream);
      selection.placement.cached_bypassed_transfer = stolen ? 1 : 0;
      finish_assembly(*this, *scan_input, *assembly, stream);
    }
  } catch (...) {
    detail::retire_reads(assembly, stream, unjoined);
    throw;
  }

  // Read the size before moving the table into a batch to avoid locking the batch. Publish rows
  // and bytes together because they form one sample.
  auto& output_table       = assembly->output;
  auto const emitted_rows  = static_cast<std::size_t>(output_table->num_rows());
  auto const emitted_bytes = output_table->alloc_size();
  auto* const mem_space    = scan_input->gpu_memory_space;
  auto batch =
    sirius::make_data_batch(std::move(output_table), *mem_space, stream, batch_telemetry());
  detail::check_read_checkpoint(detail::read_checkpoint::BATCH);
  auto const batch_id = batch->get_batch_id();
  std::unique_ptr<operator_data> output;
  if (_dynamic_filter_consumer) {
    output = std::make_unique<scan_output_operator_data>(std::move(batch));
  } else {
    std::vector<std::shared_ptr<::cucascade::data_batch>> batches{std::move(batch)};
    output = std::make_unique<pipelineable_operator_data>(std::move(batches));
  }
  detail::check_read_checkpoint(detail::read_checkpoint::PAYLOAD);
  {
    std::lock_guard<std::mutex> guard(_emitted_mutex);
    _emitted_rows += emitted_rows;
    _emitted_bytes += emitted_bytes;
  }
  if (_dynamic_filter_consumer) {
    _dynamic_filter_consumer->record_placement(selection.placement);
    receipt.original_batch_id = batch_id;
    if (selection.cached) {
      receipt = _dynamic_filter_consumer->finish(std::move(*selection.cached), batch_id);
    }
    static_cast<scan_output_operator_data&>(*output).install_receipt(std::move(receipt));
  }
  return output;
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
