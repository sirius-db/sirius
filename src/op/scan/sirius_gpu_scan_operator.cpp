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

#include <codegen/selection/selection.hpp>
#include <data/data_batch_utils.hpp>
#include <data/sirius_converter_registry.hpp>
#include <late_mat/annotated_table_representation.hpp>
#include <late_mat/column_origin.hpp>
#include <late_mat/defer_directive.hpp>
#include <late_mat/rowid_emission.hpp>
#include <log/logging.hpp>
#include <op/scan/gpu_ingestible.hpp>
#include <op/scan/parquet_gpu_ingestible.hpp>
#include <op/scan/sirius_gpu_scan_operator.hpp>
#include <op/scan/sirius_gpu_scan_operator_data.hpp>
#include <op/sirius_physical_operator.hpp>
#include <scan_manager/split_connector.hpp>

// cudf
#include <cudf/binaryop.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/filling.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/device_buffer.hpp>

// cucascade
#include <cucascade/cudf/gpu_data_representation.hpp>
#include <cucascade/data/data_batch.hpp>
#include <cucascade/memory/memory_space.hpp>

// standard library
#include <algorithm>
#include <memory>
#include <utility>
#include <vector>

namespace sirius::op::scan {

//===----------------------------------------------------------------------===//
// sirius_gpu_scan_operator
//===----------------------------------------------------------------------===//
sirius_gpu_scan_operator::sirius_gpu_scan_operator(duckdb::vector<sirius::logical_type> types,
                                                   duckdb::idx_t estimated_cardinality,
                                                   std::shared_ptr<gpu_ingestible> ingestible)
  : sirius_physical_operator(
      SiriusPhysicalOperatorType::GPU_SCAN, std::move(types), estimated_cardinality),
    _ingestible(std::move(ingestible)),
    _split_connector(std::make_shared<scan_manager::split_connector>())
{
  // Resolve the scan's dynamic-filter channel once (null for non-parquet
  // ingestibles): every split gets it stamped so prepare_for_processing can
  // snapshot membership filters at decode time.
  if (auto const* pq = dynamic_cast<parquet_ingestible_table_info const*>(
        &_ingestible->table_info())) {
    _dynamic_filters_channel = pq->sirius_dynamic_filters;
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
    // Share the operator's RULE-2 bail latch with the split BEFORE any
    // reservation estimate runs: one bail decides the whole scan (uniform
    // per-batch selectivity), and both the working-set estimator and
    // prepare_for_processing consult the latch.
    scan_input->fused_bail_flag = _fused_rule2_bailed;
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
const ingestible_table_info& sirius_gpu_scan_operator::peek_table_info() const
{
  return _ingestible->table_info();
}

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
  // v2 count-on-deferred under a static scan filter: substitute BEFORE
  // post_filter_and_project by VIEW-SPLICING the rowid/placeholder columns
  // over the deferred positions (no copy — post_filter's filter-by-copy then
  // compacts the spliced columns with the batch). Sound because the policy
  // proved the filter references no deferred position, and emission is dense
  // over the full chunk at this point. The spliced sources are kept alive by
  // the owning_table_view's owner until the filter copies them.
  bool substituted_pre_filter = false;
  if (late_mat_defer && late_mat_defer->pre_filter && scan_input->late_mat_origin &&
      output_table == nullptr &&
      materialized_table.state != filter_state::ROW_FILTERED_AND_PROJECTED) {
    auto const& origin = *scan_input->late_mat_origin;
    auto const& defer  = *late_mat_defer;
    auto view          = materialized_table.table.view();
    auto mr            = mem_space->get_default_allocator();
    auto const n_rows  = static_cast<std::int64_t>(view.num_rows());
    late_mat::rowid_emission_request req;
    req.range = origin.range;
    req.width = defer.narrow_rowid ? late_mat::rowid_width::u32 : late_mat::rowid_width::u64;
    // Dense only at this point: the batch has not been row-filtered yet.
    auto holder = std::make_shared<std::vector<std::unique_ptr<cudf::column>>>();
    holder->push_back(late_mat::emit_rowid_column(req, n_rows, stream, mr));
    std::vector<cudf::column_view> spliced(view.begin(), view.end());
    for (auto const pos : defer.output_positions) {
      if (pos >= spliced.size()) {
        throw std::runtime_error(
          "[sirius_gpu_scan_operator::execute] pre-filter defer position out of range");
      }
      if (pos == defer.rowid_position()) {
        spliced[pos] = holder->front()->view();
      } else {
        holder->push_back(
          cudf::make_column_from_scalar(cudf::numeric_scalar<std::int8_t>(0, true, stream),
                                        static_cast<cudf::size_type>(n_rows),
                                        stream,
                                        mr));
        spliced[pos] = holder->back()->view();
      }
    }
    // Owner = the previous view's owner AND the spliced columns (shared_ptr
    // pair — std::any requires copy-constructible owners).
    auto previous = std::make_shared<owning_table_view>(std::move(materialized_table.table));
    materialized_table.table =
      owning_table_view(std::make_pair(std::move(previous), std::move(holder)),
                        cudf::table_view(spliced));
    substituted_pre_filter = true;
  }
  if (materialized_table.state != filter_state::ROW_FILTERED_AND_PROJECTED) {
    output_table =
      _ingestible->post_filter_and_project(std::move(materialized_table), *mem_space, stream);
  } else {
    output_table = materialized_table.table.release(stream, mem_space->get_default_allocator());
  }

  std::shared_ptr<::cucascade::data_batch> batch;
  // Late-mat (SIRIUS_EXP_LATE_MAT). Two gated features, both keyed off the
  // provider-stamped origin; gate off ⇒ late_mat_origin is never set and this
  // whole region is a single null check.
  //
  // 1. DEFERRAL SUBSTITUTION (late_mat_defer, installed by the defer policy in
  //    a pair with the consuming port's directive): replace the deferred
  //    output positions with a UINT64 pin-order rowid column (first position;
  //    dense iota for full-chunk batches, captured wave-1 survivor ids for
  //    fused-compacted ones) and INT8 zero placeholders. Arity and positions
  //    are preserved, so every operator between here and the consuming port
  //    carries the narrow columns as ordinary data. A stamped scan whose
  //    batch is neither dense nor mask-captured must FAIL — batches of one
  //    scan must substitute consistently or CONCAT downstream would see
  //    mixed types (never silent wrong data).
  if (late_mat_defer && !substituted_pre_filter && scan_input->late_mat_origin && output_table) {
    auto const& origin = *scan_input->late_mat_origin;
    auto const& defer  = *late_mat_defer;
    auto const n_rows  = static_cast<std::int64_t>(output_table->num_rows());
    auto mr            = mem_space->get_default_allocator();

    // One-line emission via the shared helper (dense iota / captured wave-1
    // mask; u32 only for count-on-deferred bundles). A stamped batch matching
    // neither shape throws inside — batches of one scan must substitute
    // consistently or CONCAT downstream would see mixed types.
    late_mat::rowid_emission_request req;
    req.range = origin.range;
    req.width = defer.narrow_rowid ? late_mat::rowid_width::u32 : late_mat::rowid_width::u64;
    sirius::codegen::selection_mask mask;
    if (n_rows != origin.range.rows && scan_input->late_mat_selection &&
        scan_input->late_mat_selection->kind == late_mat::row_selection_kind::mask &&
        n_rows == scan_input->late_mat_selection->survivor_count) {
      auto const& sel     = *scan_input->late_mat_selection;
      mask.words          = static_cast<std::uint32_t*>(sel.mask_words->data());
      mask.num_rows       = origin.range.rows;
      mask.survivor_count = sel.survivor_count;
      mask.chunk_offsets  = static_cast<std::uint32_t*>(sel.chunk_offsets->data());
      req.mask            = &mask;
    }
    auto rowid_col = late_mat::emit_rowid_column(req, n_rows, stream, mr);

    auto cols = output_table->release();
    for (auto const pos : defer.output_positions) {
      if (pos >= cols.size()) {
        throw std::runtime_error(
          "[sirius_gpu_scan_operator::execute] late-mat defer position out of range");
      }
      if (pos == defer.rowid_position()) {
        cols[pos] = std::move(rowid_col);
      } else {
        cols[pos] = cudf::make_column_from_scalar(
          cudf::numeric_scalar<std::int8_t>(0, true, stream),
          static_cast<cudf::size_type>(n_rows),
          stream,
          mr);
      }
    }
    output_table = std::make_unique<cudf::table>(std::move(cols));
  }
  // 2. ORIGIN ANNOTATION on non-substituted outputs (downstream consumers /
  //    prepare_selection_from_batch):
  //  - dense form when the output demonstrably covers the WHOLE chunk (the
  //    row guard is self-verifying: filters/masks/compaction fall through);
  //  - mask form when the fused decode compacted the batch and the capture
  //    harvested its selection (rows == survivor count);
  //  - the column guard enforces the materialized-order mapping invariant (output
  //    column j == materialized slot j).
  else if (scan_input->late_mat_origin && output_table) {
    auto const& origin = *scan_input->late_mat_origin;
    auto const n_rows  = static_cast<std::int64_t>(output_table->num_rows());
    bool const columns_map =
      origin.columns &&
      static_cast<std::size_t>(output_table->num_columns()) <= origin.columns->size();
    std::shared_ptr<const late_mat::batch_annotation> annotation;
    if (columns_map && n_rows == origin.range.rows) {
      annotation = std::make_shared<const late_mat::batch_annotation>(
        late_mat::batch_annotation{origin, late_mat::row_selection::make_dense(origin.range)});
    } else if (columns_map && scan_input->late_mat_selection &&
               scan_input->late_mat_selection->kind == late_mat::row_selection_kind::mask &&
               n_rows == scan_input->late_mat_selection->survivor_count) {
      annotation = std::make_shared<const late_mat::batch_annotation>(
        late_mat::batch_annotation{origin, *scan_input->late_mat_selection});
    }
    if (annotation) {
      auto annotated_repr = std::make_unique<late_mat::origin_annotated_gpu_table_representation>(
        std::move(output_table), *mem_space, stream, std::move(annotation));
      const auto batch_id = sirius::get_next_batch_id();
      batch               = ::cucascade::data_batch::make(
        batch_id,
        std::move(annotated_repr),
        telemetry::quent_data_batch_probe::create(batch_telemetry(), batch_id));
    }
  }
  if (!batch) {
    batch = sirius::make_data_batch(std::move(output_table), *mem_space, stream, batch_telemetry());
  }
  std::vector<std::shared_ptr<::cucascade::data_batch>> batches{std::move(batch)};
  return std::make_unique<pipelineable_operator_data>(std::move(batches));
}

std::size_t sirius_gpu_scan_operator::no_history_peak_memory_estimate(
  const op::input_stats& stats) const
{
  // Match the legacy 8x fresh-read heuristic for projected data, then add any
  // filter-only columns that must also be decoded. Keeping the latter additive
  // avoids applying the expansion factor twice to the transient working set.
  // Resident (cached) chunks surface their mask/filter copy peaks through the
  // split's working-set estimate; a plain chunk reports it equal to bytes.
  if (stats.resident) { return std::max(stats.bytes, stats.working_set_bytes); }
  auto const filter_only_bytes =
    stats.working_set_bytes > stats.bytes ? stats.working_set_bytes - stats.bytes : 0;
  return stats.bytes * 8 + filter_only_bytes;
}

}  // namespace sirius::op::scan
