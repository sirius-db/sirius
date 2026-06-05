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

#include "planner/sirius_physical_plan_generator.hpp"

#include "config.hpp"
#include "duckdb/common/multi_file/multi_file_states.hpp"
#include "duckdb/common/type_visitor.hpp"
#include "duckdb/common/types/column/column_data_collection.hpp"
#include "duckdb/execution/column_binding_resolver.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/main/config.hpp"
#include "duckdb/main/query_profiler.hpp"
#include "duckdb/main/settings.hpp"
#include "duckdb/planner/operator/list.hpp"
#include "duckdb/planner/operator/logical_extension_operator.hpp"
#include "duckdb/planner/table_filter.hpp"
#include "log/logging.hpp"
#include "op/scan/iceberg_metadata_reader.hpp"
#include "op/scan/parquet_scan_info.hpp"
#include "op/scan/sirius_gpu_parquet_scan_operator.hpp"
#include "op/sirius_dynamic_filter.hpp"
#include "op/sirius_physical_column_data_scan.hpp"
#include "op/sirius_physical_concat.hpp"
#include "op/sirius_physical_cpu_source.hpp"
#include "op/sirius_physical_delim_join.hpp"
#include "op/sirius_physical_duckdb_scan.hpp"
#include "op/sirius_physical_dummy_scan.hpp"
#include "op/sirius_physical_grouped_aggregate.hpp"
#include "op/sirius_physical_grouped_aggregate_merge.hpp"
#include "op/sirius_physical_iceberg_scan.hpp"
#include "op/sirius_physical_merge_sort.hpp"
#include "op/sirius_physical_order.hpp"
#include "op/sirius_physical_partition.hpp"
#include "op/sirius_physical_result_collector.hpp"
#include "op/sirius_physical_sort_partition.hpp"
#include "op/sirius_physical_sort_sample.hpp"
#include "op/sirius_physical_table_scan.hpp"
#include "op/sirius_physical_top_n.hpp"
#include "op/sirius_physical_top_n_merge.hpp"
#include "op/sirius_physical_ungrouped_aggregate.hpp"
#include "op/sirius_physical_ungrouped_aggregate_merge.hpp"
#include "planner/sirius_plan_projection_utils.hpp"
#include "sirius_config.hpp"
#include "sirius_context.hpp"

#include <optional>
#include <utility>

namespace sirius::planner {

namespace {
/// Read the dynamic-filter-pushdown enable flag from the active SiriusContext config. Defaults to
/// disabled when the state is unavailable (no config to consult outside a configured query).
bool dynamic_filter_pushdown_enabled(duckdb::ClientContext& context)
{
  auto state = context.registered_state->Get<duckdb::SiriusContext>("sirius_state");
  if (!state) { return false; }
  return state->get_config().get_operator_params().enable_dynamic_filter_pushdown;
}

//! Insert `factory(std::move(parent.children[i]))` between `parent` and its i-th child. The
//! factory takes ownership of the original child and returns the wrapper subtree (which
//! must already hold the original as one of its own descendants).
//!
//! Move-semantics on `parent.children[i]` guarantees no raw pointer is held across the
//! mutation: the slot is null between the `std::move` and the assignment, so the compiler
//! enforces tree integrity. See the master plan (`Tree-mutation pattern`) for the rationale.
template <typename WrapperFactory>
void wrap_child(sirius::op::sirius_physical_operator& parent,
                std::size_t i,
                WrapperFactory&& factory)
{
  auto original      = std::move(parent.children[i]);
  auto wrapper       = std::forward<WrapperFactory>(factory)(std::move(original));
  parent.children[i] = std::move(wrapper);
}

//! Replace the operator at `slot` with `factory(std::move(slot))`. Used for sink-wrap rewrites
//! (HASH_GROUP_BY, ORDER_BY, TOP_N, etc.) where the original operator becomes a child of a
//! newly-inserted wrapper that sits in the slot it used to occupy. The factory receives
//! ownership of the original and must return a wrapper subtree containing the original.
template <typename WrapperFactory>
void wrap_above(duckdb::unique_ptr<sirius::op::sirius_physical_operator>& slot,
                WrapperFactory&& factory)
{
  auto original = std::move(slot);
  slot          = std::forward<WrapperFactory>(factory)(std::move(original));
}

//! Build a `parquet_scan_info` describing a parquet read by lifting fields out of a
//! `sirius_physical_table_scan`. **Destructive**: `scan_op.table_filters` is moved into the
//! info and `scan_op.table_filters` is left null. Mirrors the field plumbing in today's
//! `sirius_pipeline_converter::insert_parquet_scan_operator` byte-for-byte.
std::unique_ptr<sirius::op::scan::parquet_scan_info> build_parquet_scan_info(
  sirius::op::sirius_physical_table_scan& scan_op, const sirius::operator_params& op_params)
{
  auto const& bind_data = scan_op.bind_data->Cast<duckdb::MultiFileBindData>();
  if (!bind_data.file_list || bind_data.file_list->IsEmpty()) {
    throw std::runtime_error(
      "[sirius_physical_plan_generator::build_parquet_scan_info] No input files to scan");
  }
  std::vector<std::string> file_paths;
  for (auto const& file : bind_data.file_list->GetAllFiles()) {
    file_paths.push_back(file.path);
  }
  auto const& partition_indices = bind_data.reader_bind.hive_partitioning_indexes;

  auto info               = std::make_unique<sirius::op::scan::parquet_scan_info>();
  info->returned_types    = scan_op.returned_types;
  info->file_paths        = std::move(file_paths);
  info->column_ids        = scan_op.column_ids;
  info->projection_ids    = scan_op.projection_ids;
  info->names             = scan_op.names;
  info->table_filters     = std::move(scan_op.table_filters);
  info->partition_indices = partition_indices;
  // Mirror the two trailing fields the legacy `insert_parquet_scan_operator` sets
  // (sirius_pipeline_converter.cpp:297-302). Both came in via post-B.2b upstream commits:
  //   - `scan_output_arity` (upstream #749) drives `scan_info::make_provider`'s expected
  //     column count. Without it, the runtime task emits only the data columns and skips
  //     the hive-partition columns it should inject post-read, producing a 3-vs-5 column
  //     mismatch and downstream vector::_M_range_check.
  //   - `approximate_batch_size` (upstream #792) provides the per-task scan batch sizing.
  info->scan_output_arity      = scan_op.types.size();
  info->approximate_batch_size = op_params.scan_task_batch_size;
  return info;
}

//! Rewrite a TABLE_SCAN node so the tree-based converter's `build_pipelines` walk produces
//! the same pipeline shape the legacy converter assembles at runtime.
//!
//! - For `parquet_scan` / `read_parquet`: the legacy
//!   `sirius_pipeline_converter::insert_parquet_scan_operator` inserts the GPU scan at the
//!   front of the *same* pipeline as the TABLE_SCAN's downstream operators (no extra
//!   pipeline). We mirror that here by REPLACING the slot's TABLE_SCAN with the GPU leaf —
//!   the leaf inherits TABLE_SCAN's position in the plan tree so the walk treats it as the
//!   source-leaf of the existing pipeline, matching the legacy dump.
//! - For `seq_scan` and `iceberg_scan`: the legacy
//!   `sirius_pipeline_converter::split_table_scan_source` creates a *new* pipeline. We
//!   reproduce that by attaching the GPU leaf as the TABLE_SCAN's only child so the walk
//!   spins up a child meta-pipeline for it (the wrap pipeline).
//!
//! For `iceberg_scan`, attaches the pre-fetched `IcebergDeleteData` from
//! `iceberg_delete_data_cache_` so the GPU operator sees the delete-merge set on every read.
//! Path resolution mirrors `resolve_iceberg_table_path` exactly; cache misses leave
//! `delete_data` null, which matches today's converter behavior at
//! `construct_sirius_specific_operator:65-76`.
//!
//! Throws on truly unsupported scan functions to match `construct_sirius_specific_operator`'s
//! behavior at converter line 80.
void wrap_table_scan_source(
  duckdb::unique_ptr<sirius::op::sirius_physical_operator>& table_scan_slot,
  const std::unordered_map<std::string, std::shared_ptr<const sirius::op::scan::IcebergDeleteData>>&
    iceberg_cache,
  const sirius::operator_params& op_params)
{
  // Table-in-out functions wear a TABLE_SCAN with children — skip per the master plan's
  // exclusion rule. Wrapping them would change their child layout in a way the converter
  // and downstream operators don't expect.
  if (!table_scan_slot->children.empty()) { return; }

  auto& scan     = table_scan_slot->Cast<sirius::op::sirius_physical_table_scan>();
  const auto& fn = scan.function.name;

  duckdb::unique_ptr<sirius::op::sirius_physical_operator> leaf;
  bool replace_slot = false;
  if (fn == "seq_scan") {
    leaf = duckdb::make_uniq<sirius::op::sirius_physical_duckdb_scan>(&scan);
  } else if (fn == "parquet_scan" || fn == "read_parquet") {
    auto info = build_parquet_scan_info(scan, op_params);
    leaf      = duckdb::make_uniq<sirius::op::scan::sirius_gpu_parquet_scan_operator>(
      scan.types, scan.estimated_cardinality, std::move(info));
    // Parquet: legacy inlines GPU_PARQUET_SCAN into the current pipeline, so replace the
    // TABLE_SCAN in place rather than attaching as a child (which would make build_pipelines
    // spin up a separate wrap pipeline). The TABLE_SCAN object is dropped — its bind_data
    // and metadata have already been lifted into the parquet_scan_info.
    replace_slot = true;
  } else if (fn == "iceberg_scan") {
    auto iceberg_scan = duckdb::make_uniq<sirius::op::sirius_physical_iceberg_scan>(&scan);
    if (!scan.parameters.empty()) {
      auto const table_path = scan.parameters[0].ToString();
      auto it               = iceberg_cache.find(table_path);
      if (it != iceberg_cache.end()) { iceberg_scan->delete_data = it->second; }
    }
    leaf = std::move(iceberg_scan);
  } else {
    throw std::runtime_error(
      "[sirius_physical_plan_generator::wrap_table_scan_source] Unsupported scan function: " + fn);
  }
  if (replace_slot) {
    table_scan_slot = std::move(leaf);
  } else {
    table_scan_slot->children.push_back(std::move(leaf));
  }
}

//! Attach a leaf CPU_SOURCE operator as the only child of a COLUMN_DATA_SCAN (with non-null
//! collection), EMPTY_RESULT, or DUMMY_SCAN node. Mirrors the legacy converter's
//! `split_cpu_source`. COLUMN_DATA_SCAN with a null collection is the LEFT_DELIM_JOIN cached
//! chunk scan — populated at runtime by the delim-join sink, not by `cpu_source_task` — so
//! we deliberately leave it as-is.
void wrap_cpu_source(sirius::op::sirius_physical_operator& source_op)
{
  if (!source_op.children.empty()) { return; }

  duckdb::unique_ptr<sirius::op::sirius_physical_cpu_source> leaf;
  switch (source_op.type) {
    case sirius::op::SiriusPhysicalOperatorType::COLUMN_DATA_SCAN: {
      auto& col_scan = source_op.Cast<sirius::op::sirius_physical_column_data_scan>();
      if (!col_scan.collection) { return; }  // LEFT_DELIM_JOIN cached chunk scan — skip
      leaf = duckdb::make_uniq<sirius::op::sirius_physical_cpu_source>(
        source_op.types, source_op.estimated_cardinality, std::move(col_scan.collection));
      break;
    }
    case sirius::op::SiriusPhysicalOperatorType::DUMMY_SCAN:
      leaf = duckdb::make_uniq<sirius::op::sirius_physical_cpu_source>(
        source_op.types, source_op.estimated_cardinality, /*produce_single_row=*/true);
      break;
    case sirius::op::SiriusPhysicalOperatorType::EMPTY_RESULT:
      leaf = duckdb::make_uniq<sirius::op::sirius_physical_cpu_source>(
        source_op.types, source_op.estimated_cardinality, /*produce_single_row=*/false);
      break;
    default: return;
  }
  source_op.children.push_back(std::move(leaf));
}

//! Replace a HASH_GROUP_BY slot with `GROUPED_AGGREGATE_MERGE → PARTITION → HASH_GROUP_BY →
//! original_input`. Mirrors the converter's `split_group_aggregate_sink` HASH_GROUP_BY branch.
//! The original HGB is kept as the per-thread state sink; PARTITION buckets its output for
//! the cross-thread merge.
void wrap_hash_group_by(duckdb::unique_ptr<sirius::op::sirius_physical_operator>& slot,
                        const sirius::operator_params& op_params)
{
  wrap_above(slot, [&](duckdb::unique_ptr<sirius::op::sirius_physical_operator> hgb_op) {
    auto* hgb_ptr = hgb_op.get();

    auto partition =
      duckdb::make_uniq<sirius::op::sirius_physical_partition>(hgb_ptr->types,
                                                               hgb_ptr->estimated_cardinality,
                                                               /*key_source=*/hgb_ptr,
                                                               /*is_build=*/false,
                                                               op_params.hash_partition_bytes);
    partition->children.push_back(std::move(hgb_op));

    auto merge = duckdb::make_uniq<sirius::op::sirius_physical_grouped_aggregate_merge>(
      &hgb_ptr->Cast<sirius::op::sirius_physical_grouped_aggregate>());
    merge->children.push_back(std::move(partition));
    return merge;
  });
}

//! Replace an UNGROUPED_AGGREGATE slot with `UNGROUPED_AGGREGATE_MERGE → UNGROUPED_AGGREGATE →
//! original_input`. Mirrors the UNGROUPED branch of `split_group_aggregate_sink`. No PARTITION
//! step is needed: the merge consumes the single per-thread accumulator directly.
void wrap_ungrouped_aggregate(duckdb::unique_ptr<sirius::op::sirius_physical_operator>& slot)
{
  wrap_above(slot, [&](duckdb::unique_ptr<sirius::op::sirius_physical_operator> ungrouped_op) {
    auto* ungrouped_ptr = ungrouped_op.get();
    auto merge          = duckdb::make_uniq<sirius::op::sirius_physical_ungrouped_aggregate_merge>(
      &ungrouped_ptr->Cast<sirius::op::sirius_physical_ungrouped_aggregate>());
    merge->children.push_back(std::move(ungrouped_op));
    return merge;
  });
}

//! Replace a TOP_N slot with `TOP_N_MERGE → TOP_N → original_input`. Mirrors `split_top_n_sink`.
void wrap_top_n(duckdb::unique_ptr<sirius::op::sirius_physical_operator>& slot)
{
  wrap_above(slot, [&](duckdb::unique_ptr<sirius::op::sirius_physical_operator> topn_op) {
    auto* topn_ptr = &topn_op->Cast<sirius::op::sirius_physical_top_n>();
    auto merge     = duckdb::make_uniq<sirius::op::sirius_physical_top_n_merge>(topn_ptr);
    merge->children.push_back(std::move(topn_op));
    return merge;
  });
}

//! Replace an ORDER_BY slot with the sort chain
//! `MERGE_SORT → SORT_PARTITION → SORT_SAMPLE → ORDER_BY → original_input`. Mirrors
//! `split_order_by_sink` field-for-field including the destructive side-effects:
//!   - ORDER_BY's `projections` is overwritten with the identity projection over the input's
//!     types, and its `types` is replaced with the input's types — so the per-batch sort
//!     keeps every column visible to SORT_SAMPLE / SORT_PARTITION.
//!   - SORT_SAMPLE is constructed with the sample-sizing params from `op_params`
//!     (`sort_sample_bytes`, `max_sort_partition_bytes`, `max_sort_partition_memory_fraction`).
//!   - SORT_PARTITION's `set_sample_op` is wired to the SORT_SAMPLE just inserted.
//!   - MERGE_SORT receives the original projection back via `set_final_projections` when the
//!     original was non-identity (otherwise the chain already projects all columns).
//!
//! Post-#866/#876: SORT_SAMPLE has `is_sink()==false`, so under the tree-based build it lands
//! in `operators[]` of the SORT_PARTITION pipeline (3-pipeline shape) — matching what
//! `split_order_by_sink` produces on the legacy path.
void wrap_order_by(duckdb::unique_ptr<sirius::op::sirius_physical_operator>& slot,
                   const sirius::operator_params& op_params)
{
  wrap_above(slot, [&](duckdb::unique_ptr<sirius::op::sirius_physical_operator> order_op) {
    auto* order_ptr = &order_op->Cast<sirius::op::sirius_physical_order>();
    if (order_ptr->children.empty()) {
      throw std::runtime_error(
        "[sirius_physical_plan_generator::wrap_order_by] ORDER_BY has no child input");
    }

    auto original_projections = order_ptr->projections;
    auto const& child_types   = order_ptr->children[0]->types;

    duckdb::vector<std::size_t> identity_proj;
    identity_proj.reserve(child_types.size());
    for (std::size_t col_idx = 0; col_idx < child_types.size(); col_idx++) {
      identity_proj.push_back(col_idx);
    }
    order_ptr->projections = std::move(identity_proj);
    order_ptr->types       = child_types;

    auto sample = duckdb::make_uniq<sirius::op::sirius_physical_sort_sample>(
      order_ptr,
      op_params.sort_sample_bytes,
      op_params.max_sort_partition_bytes,
      op_params.max_sort_partition_memory_fraction);
    auto* sample_ptr = sample.get();
    sample->children.push_back(std::move(order_op));

    auto partition = duckdb::make_uniq<sirius::op::sirius_physical_sort_partition>(order_ptr);
    partition->set_sample_op(sample_ptr);
    partition->children.push_back(std::move(sample));

    auto merge = duckdb::make_uniq<sirius::op::sirius_physical_merge_sort>(order_ptr);

    bool is_identity = (original_projections.size() == order_ptr->types.size());
    if (is_identity) {
      for (std::size_t i = 0; i < original_projections.size(); i++) {
        if (original_projections[i] != i) {
          is_identity = false;
          break;
        }
      }
    }
    if (!is_identity) {
      duckdb::vector<sirius::logical_type> output_types;
      output_types.reserve(original_projections.size());
      for (auto idx : original_projections) {
        output_types.push_back(order_ptr->types[idx]);
      }
      merge->set_final_projections(std::move(original_projections), std::move(output_types));
    }

    merge->children.push_back(std::move(partition));
    return merge;
  });
}

//! Wrap a single child of a HASH_JOIN or NESTED_LOOP_JOIN at `join_op.children[child_idx]`
//! with `CONCAT → PARTITION → original_child`. `is_build` flips the build/probe semantics
//! threaded through both wrappers. `join_op` is passed verbatim as PARTITION's `key_source`
//! (HJ conditions or NLJ shape determine partition keys) and as CONCAT's `downstream_join`
//! (the HJ/NLJ's join type determines CONCAT's batch-coalescing mode). Mirrors the per-side
//! construction in `split_intermediate_joins` (probe) and `split_join_sink` (build) at
//! converter:361-371 and 463-491.
void wrap_join_child(sirius::op::sirius_physical_operator& join_op,
                     std::size_t child_idx,
                     bool is_build,
                     const sirius::operator_params& op_params)
{
  D_ASSERT(join_op.type == sirius::op::SiriusPhysicalOperatorType::HASH_JOIN ||
           join_op.type == sirius::op::SiriusPhysicalOperatorType::NESTED_LOOP_JOIN);
  auto* join_op_ptr = &join_op;
  wrap_child(
    join_op, child_idx, [&](duckdb::unique_ptr<sirius::op::sirius_physical_operator> child_orig) {
      // Capture types and cardinality from the original child BEFORE moving it: PARTITION
      // and CONCAT need them to construct, and after the move into PARTITION the original's
      // members are no longer addressable.
      auto child_types = child_orig->types;
      auto est_card    = child_orig->estimated_cardinality;

      auto concat =
        duckdb::make_uniq<sirius::op::sirius_physical_concat>(child_types,
                                                              est_card,
                                                              /*downstream_join=*/join_op_ptr,
                                                              is_build,
                                                              op_params.concat_batch_bytes);
      auto partition =
        duckdb::make_uniq<sirius::op::sirius_physical_partition>(std::move(child_types),
                                                                 est_card,
                                                                 /*key_source=*/join_op_ptr,
                                                                 is_build,
                                                                 op_params.hash_partition_bytes);
      partition->children.push_back(std::move(child_orig));
      concat->children.push_back(std::move(partition));
      return concat;
    });
}

//! Wrap both children of a HASH_JOIN or NESTED_LOOP_JOIN with the CONCAT/PARTITION feeder
//! chain. Probe side (`children[0]`, `is_build=false`) mirrors
//! `split_intermediate_joins`; build side (`children[1]`, `is_build=true`) mirrors
//! `split_join_sink`. If a side is unexpectedly missing (`children.size() < 2`), it is
//! simply skipped — the join wouldn't be well-formed otherwise and the downstream
//! operators would have already failed.
void wrap_join(sirius::op::sirius_physical_operator& join_op,
               const sirius::operator_params& op_params)
{
  if (join_op.children.size() >= 1) {
    wrap_join_child(join_op, /*child_idx=*/0, /*is_build=*/false, op_params);
  }
  if (join_op.children.size() >= 2) {
    wrap_join_child(join_op, /*child_idx=*/1, /*is_build=*/true, op_params);
  }
}

// Forward declaration so `wrap_delim_join` can recurse into the internal `join`/`distinct`
// subtrees of a DELIM JOIN; those operators are stored as `unique_ptr` fields on the
// delim-join class, not in `children[]`, so the standard tree walk would otherwise skip them.
void insert_gpu_pipeline_operators_recursive(
  duckdb::unique_ptr<sirius::op::sirius_physical_operator>& slot,
  const std::unordered_map<std::string, std::shared_ptr<const sirius::op::scan::IcebergDeleteData>>&
    iceberg_cache,
  const sirius::operator_params& op_params);

//! Replace a DELIM JOIN's `distinct_root` (initially the bare DISTINCT) with the chain
//! `DISTINCT_MERGE -> PARTITION_DISTINCT -> original DISTINCT`. Mirrors `wrap_hash_group_by`
//! structurally, applied to the `distinct_root` slot rather than a `children[]` entry. The
//! original DISTINCT aggregate stays reachable via the non-owning `delim_base.distinct`
//! borrow — the inline per-batch sink path on left/right_delim_join uses that borrow, and
//! the underlying object never relocates (move-of-unique_ptr only transfers ownership).
void wrap_delim_distinct(sirius::op::sirius_physical_delim_join& delim_base,
                         const sirius::operator_params& op_params)
{
  if (!delim_base.distinct_root) { return; }

  // distinct_root currently holds the bare original DISTINCT aggregate.
  auto original          = std::move(delim_base.distinct_root);
  auto* original_agg_ptr = &original->Cast<sirius::op::sirius_physical_grouped_aggregate>();

  auto partition =
    duckdb::make_uniq<sirius::op::sirius_physical_partition>(original->types,
                                                             original->estimated_cardinality,
                                                             /*key_source=*/original.get(),
                                                             /*is_build=*/false,
                                                             op_params.hash_partition_bytes);
  partition->children.push_back(std::move(original));

  auto merge =
    duckdb::make_uniq<sirius::op::sirius_physical_grouped_aggregate_merge>(original_agg_ptr);
  merge->children.push_back(std::move(partition));

  // B.1' (#604): tag the chain top with the owning DELIM_JOIN so
  // compute_repository_wiring_tree_based can redirect its tree-parent wiring
  // (which would otherwise emit merge_top -> DELIM_JOIN) to each delim_scan's
  // consumer pipeline. Mirrors legacy split_delim_join_sink's retarget at
  // sirius_pipeline_converter.cpp:849-852.
  merge->set_owning_delim_join(&delim_base);

  delim_base.distinct_root = std::move(merge);
  // delim_base.distinct stays valid: the original DISTINCT object never relocates, only its
  // owning slot moves from delim_base.distinct_root down through the chain.
}

//! Rewrite the internal subtrees of a DELIM JOIN (LEFT or RIGHT) and wire the sibling
//! pointers that the operator needs at runtime. Mirrors `split_delim_join_sink`'s
//! sibling-pointer assignments (converter:750-755) AND the DISTINCT_MERGE/PARTITION_DISTINCT
//! chain that the legacy converter creates outside the tree (converter:769-841). Both halves
//! live in the tree under USE_TREE_BASED_PIPELINE_BUILD.
//!
//! What this does:
//!   - Recursively walks `delim->join` so source-side wraps (TABLE_SCAN/CPU_SOURCE family)
//!     and sink-side wraps (HASH_GROUP_BY/ORDER_BY/TOP_N/UNGROUPED) inside the internal
//!     join's subtree fire, and so the internal join (if HJ/NLJ) gets the same
//!     CONCAT/PARTITION wraps on its probe + build that Sub-phase B.4 applies to top-level
//!     joins. This is what plants the `partition_join` candidate node.
//!   - Recursively walks the children of the original DISTINCT (via `distinct_root->children`,
//!     because at this point `distinct_root` still holds the bare DISTINCT) so source-side
//!     wraps below it fire. Then calls `wrap_delim_distinct` to wrap DISTINCT_MERGE +
//!     PARTITION_DISTINCT above it. Post-order: source-side wraps first, then the chain
//!     wrap, so the chain wrap doesn't re-visit the freshly-inserted wrappers.
//!   - For RIGHT_DELIM_JOIN: after the internal-join recursion has wrapped the build side
//!     with `CONCAT_build -> PARTITION_build -> original_build`, captures the new
//!     PARTITION_build via `internal_join->children[1]->children[0]` and assigns it to
//!     `right_delim_join.partition_join`. Matches converter:750-751.
//!   - For LEFT_DELIM_JOIN: assigns `left_delim_join.column_data_scan` to the COLUMN_DATA_SCAN
//!     at `internal_join->children[0]`. Matches converter:753-754.
void wrap_delim_join(
  duckdb::unique_ptr<sirius::op::sirius_physical_operator>& slot,
  const std::unordered_map<std::string, std::shared_ptr<const sirius::op::scan::IcebergDeleteData>>&
    iceberg_cache,
  const sirius::operator_params& op_params)
{
  auto& delim_base = slot->Cast<sirius::op::sirius_physical_delim_join>();

  // Recurse into the internal join + distinct subtrees so their source/sink/join wraps fire.
  // This is what produces the CONCAT_build/PARTITION_build chain on the internal join's
  // build side (when the internal join is HJ/NLJ) that the RIGHT_DELIM sibling pointer
  // references.
  if (delim_base.join) {
    insert_gpu_pipeline_operators_recursive(delim_base.join, iceberg_cache, op_params);
  }
  if (delim_base.distinct_root) {
    // At this point `distinct_root` still holds the bare original DISTINCT (wrap_delim_distinct
    // hasn't run yet). Recurse into its children for source-side wraps below DISTINCT, then
    // wrap MERGE/PARTITION above.
    for (auto& child_slot : delim_base.distinct_root->children) {
      insert_gpu_pipeline_operators_recursive(child_slot, iceberg_cache, op_params);
    }
    wrap_delim_distinct(delim_base, op_params);
  }

  if (slot->type == sirius::op::SiriusPhysicalOperatorType::RIGHT_DELIM_JOIN) {
    auto& right_delim = slot->Cast<sirius::op::sirius_physical_right_delim_join>();

    // B.5 (#604): tag the bare DISTINCT so its `build_pipelines` becomes a no-op.
    // RIGHT_DELIM_JOIN::sink runs `distinct->execute` and `distinct->sink` inline,
    // so the bare DISTINCT contributes nothing to any pipeline (matches legacy's
    // partition_distinct pipeline which has operators=[PARTITION] only). Scoped
    // to RIGHT only — LEFT_DELIM_JOIN's distinct keeps standard sink behavior
    // pending separate analysis for q2/q15/q17/q20.
    if (right_delim.distinct) { right_delim.distinct->set_owned_by_delim_join(true); }

    auto* internal_join = delim_base.join.get();
    if (internal_join && internal_join->children.size() >= 2) {
      auto* build_child = internal_join->children[1].get();
      if (build_child && build_child->type == sirius::op::SiriusPhysicalOperatorType::CONCAT &&
          !build_child->children.empty()) {
        auto* partition_build = build_child->children[0].get();
        if (partition_build &&
            partition_build->type == sirius::op::SiriusPhysicalOperatorType::PARTITION) {
          right_delim.partition_join =
            &partition_build->Cast<sirius::op::sirius_physical_partition>();
        }
      }
    }
  } else if (slot->type == sirius::op::SiriusPhysicalOperatorType::LEFT_DELIM_JOIN) {
    auto& left_delim = slot->Cast<sirius::op::sirius_physical_left_delim_join>();

    // LEFT_DELIM_JOIN ownership (#604, mirror of B.5 for RIGHT): tag the bare
    // DISTINCT so its build_pipelines becomes a no-op. LEFT_DELIM_JOIN::sink
    // runs `distinct->execute` and `distinct->sink` inline; the bare DISTINCT
    // contributes nothing to any pipeline.
    //
    // The cached chunk scan (`left_delim.column_data_scan`) is already recorded
    // and flagged in LEFT_DELIM_JOIN's constructor — we can't recover the same
    // pointer here because wrap_join (which already ran above) replaced
    // internal_join->children[0] with a CONCAT/PARTITION wrap chain.
    if (left_delim.distinct) { left_delim.distinct->set_owned_by_delim_join(true); }
  }
}

//! Post-order recursive walk over the physical plan tree. Children are visited (and rewritten)
//! before the dispatch on `slot->type`, so a later `wrap_above` cannot re-enter the freshly-
//! inserted wrapper subtree and double-wrap the original node. Source-side wraps append a leaf
//! to an existing TABLE_SCAN/COLUMN_DATA_SCAN/EMPTY_RESULT/DUMMY_SCAN node, growing it from a
//! leaf into an intermediate; the new leaf has no children of its own, so post-order is
//! equivalent to pre-order in those cases. Sink-side wraps replace the slot with a wrapper
//! subtree whose root sits above the original sink; the new wrapper nodes are not visited
//! because the walk has already moved past the slot. Join-side wraps replace each child of a
//! HJ/NLJ with a CONCAT/PARTITION chain; the chain's original child (the already-walked
//! probe/build subtree) is moved into PARTITION's child slot. DELIM JOIN handling recurses
//! into the internal `join`/`distinct` fields (which live outside `children[]`).
void insert_gpu_pipeline_operators_recursive(
  duckdb::unique_ptr<sirius::op::sirius_physical_operator>& slot,
  const std::unordered_map<std::string, std::shared_ptr<const sirius::op::scan::IcebergDeleteData>>&
    iceberg_cache,
  const sirius::operator_params& op_params)
{
  if (!slot) { return; }

  for (auto& child_slot : slot->children) {
    insert_gpu_pipeline_operators_recursive(child_slot, iceberg_cache, op_params);
  }

  switch (slot->type) {
    case sirius::op::SiriusPhysicalOperatorType::TABLE_SCAN:
      wrap_table_scan_source(slot, iceberg_cache, op_params);
      break;
    case sirius::op::SiriusPhysicalOperatorType::COLUMN_DATA_SCAN:
    case sirius::op::SiriusPhysicalOperatorType::EMPTY_RESULT: wrap_cpu_source(*slot); break;
    case sirius::op::SiriusPhysicalOperatorType::DUMMY_SCAN: {
      // B.3+B.4+B.6 (#604): skip the CPU_SOURCE wrap for the synthetic DUMMY_SCAN
      // inserted as a RIGHT_DELIM_JOIN's build placeholder — it carries no runtime
      // data flow (partition_join executes inline via DELIM_JOIN's sink) so the
      // CPU_SOURCE leaf would only materialize a phantom pipeline. Real DUMMY_SCAN
      // usages (constant-row subqueries) keep the wrap.
      auto& dummy = slot->Cast<sirius::op::sirius_physical_dummy_scan>();
      if (!dummy.is_delim_join_placeholder()) { wrap_cpu_source(*slot); }
      break;
    }
    case sirius::op::SiriusPhysicalOperatorType::HASH_GROUP_BY:
      wrap_hash_group_by(slot, op_params);
      break;
    case sirius::op::SiriusPhysicalOperatorType::UNGROUPED_AGGREGATE:
      wrap_ungrouped_aggregate(slot);
      break;
    case sirius::op::SiriusPhysicalOperatorType::ORDER_BY: wrap_order_by(slot, op_params); break;
    case sirius::op::SiriusPhysicalOperatorType::TOP_N: wrap_top_n(slot); break;
    case sirius::op::SiriusPhysicalOperatorType::HASH_JOIN:
    case sirius::op::SiriusPhysicalOperatorType::NESTED_LOOP_JOIN:
      wrap_join(*slot, op_params);
      break;
    case sirius::op::SiriusPhysicalOperatorType::LEFT_DELIM_JOIN:
    case sirius::op::SiriusPhysicalOperatorType::RIGHT_DELIM_JOIN:
      wrap_delim_join(slot, iceberg_cache, op_params);
      break;
    default: break;
  }
}

}  // namespace

sirius_physical_plan_generator::sirius_physical_plan_generator(duckdb::ClientContext& context)
  : context(context)
{
}

sirius_physical_plan_generator::~sirius_physical_plan_generator() {}

std::shared_ptr<sirius::op::sirius_dynamic_filter_set>
sirius_physical_plan_generator::get_or_create_dynamic_filter_channel(
  duckdb::DynamicTableFilterSet const* key)
{
  if (!key) { return nullptr; }
  // Central gate: when dynamic-filter pushdown is disabled, return no channel so neither the
  // producer (join) nor the consumer (scan) wires anything.
  if (!dynamic_filter_pushdown_enabled(context)) { return nullptr; }
  auto [it, inserted] = dynamic_filter_channels.try_emplace(key, nullptr);
  if (inserted) { it->second = std::make_shared<sirius::op::sirius_dynamic_filter_set>(); }
  return it->second;
}

void sirius_physical_plan_generator::set_parent_ops(sirius::op::sirius_physical_operator& op,
                                                    sirius::op::sirius_physical_operator* parent)
{
  op.set_parent_op(parent);

  // CTE is transparent for data flow on its consumer side (children[1]): the
  // CTE body (children[0]) materializes INTO CTE, while children[1] is the
  // outer query that reads from the CTE via CTE_SCAN and whose result IS
  // CTE's output (`sirius_physical_cte::execute` just forwards children[1]'s
  // batches). Set children[1]'s parent_op to CTE's own parent so the
  // tree-parent walk in `compute_repository_wiring_tree_based` doesn't emit
  // `consumer_sink -> CTE_pipeline` edges. Such an edge, combined with the
  // CTE sink's `CTE_pipeline -> CTE_SCAN_consumer` emissions
  // (`sirius_pipeline_converter.cpp:1146-1158`), would close a cycle that
  // `dump_pipeline_conversion_result::compute_sig` infinite-loops on (q15
  // SIGSEGV — only TPC-H query that exercises CTE_SCAN). Mirrors legacy's
  // wiring graph where the CTE consumer side never wires back to the CTE
  // sink.
  if (op.type == sirius::op::SiriusPhysicalOperatorType::CTE) {
    D_ASSERT(op.children.size() == 2);
    set_parent_ops(*op.children[0], &op);
    set_parent_ops(*op.children[1], parent);
    return;
  }

  for (auto& child : op.children) {
    if (child) { set_parent_ops(*child, &op); }
  }
  // DELIM JOIN stores its internal `join` and `distinct_root` subtrees as unique_ptr fields
  // outside `children[]`. Descend into them so the wrapped operators inside (B.4's
  // CONCAT/PARTITION on the join side, wrap_delim_distinct's MERGE/PARTITION on the distinct
  // side) get their `_parent_op` set to their tree parent. PARTITION's ctor takes a
  // `key_source` argument that is captured for key/type derivation only and never stored
  // (separated from the tree-parent role here), so without this descent PARTITION._parent_op
  // stays nullptr and compute_repository_wiring_tree_based can't resolve its destination.
  if (op.type == sirius::op::SiriusPhysicalOperatorType::LEFT_DELIM_JOIN ||
      op.type == sirius::op::SiriusPhysicalOperatorType::RIGHT_DELIM_JOIN) {
    auto& delim = op.Cast<sirius::op::sirius_physical_delim_join>();
    if (delim.join) { set_parent_ops(*delim.join, &op); }
    if (delim.distinct_root) { set_parent_ops(*delim.distinct_root, &op); }
  }
  // RESULT_COLLECTOR stores its tree child in `plan` (a reference, outside `children[]`) —
  // it's the engine-injected root wrapper added by `sirius_pending_statement_internal`
  // (`src/sirius_interface.cpp:166`), used by BOTH `CALL gpu_execution()` and transparent
  // execution. Without descending here the wrapped sink (e.g. MERGE_TOP_N) gets
  // `_parent_op = nullptr` and `compute_repository_wiring_tree_based` silently skips its emit
  // at the uniform tree-parent lookup (`sirius_pipeline_converter.cpp:1380`), leaving the
  // RESULT_COLLECTOR pipeline with no input source — runtime hang. Not caught by the E.1
  // differential gate because `convert_query_to_dump` builds plans by calling
  // `physical_planner.create_plan()` directly, bypassing the wrapping path.
  if (op.type == sirius::op::SiriusPhysicalOperatorType::RESULT_COLLECTOR) {
    auto& rc = op.Cast<sirius::op::sirius_physical_result_collector>();
    set_parent_ops(rc.plan, &op);
  }
}

void sirius_physical_plan_generator::insert_gpu_pipeline_operators(
  duckdb::unique_ptr<sirius::op::sirius_physical_operator>& plan)
{
  // op_params live on SiriusContext alongside the cache_level / quent config. Sink wraps
  // (HASH_GROUP_BY, ORDER_BY) need `hash_partition_bytes` and `max_sort_partition_bytes` to
  // match the legacy converter's `op_params_` reads at line 544 and line 624. Use empty
  // defaults if SiriusContext is missing — the resulting wraps fall back to the operators'
  // own constructor defaults, which match converter behavior when the context is absent.
  sirius::operator_params op_params;
  if (auto sirius_ctx = context.registered_state->Get<duckdb::SiriusContext>("sirius_state")) {
    op_params = sirius_ctx->get_config().get_operator_params();
  }
  insert_gpu_pipeline_operators_recursive(plan, iceberg_delete_data_cache_, op_params);
}

std::string sirius_physical_plan_generator::resolve_iceberg_table_path(
  sirius::op::sirius_physical_table_scan& scan_op)
{
  if (!scan_op.parameters.empty()) { return scan_op.parameters[0].ToString(); }

  // REST catalog: derive from bind_data file list.
  if (scan_op.bind_data) {
    auto& bind_data = scan_op.bind_data->Cast<duckdb::MultiFileBindData>();
    if (bind_data.file_list && !bind_data.file_list->IsEmpty()) {
      auto files = bind_data.file_list->GetAllFiles();
      if (!files.empty()) {
        auto const& first_path = files[0].path;
        // Strip "/data/<filename>" to get table root.
        auto data_pos = first_path.rfind("/data/");
        if (data_pos != std::string::npos) { return first_path.substr(0, data_pos); }
      }
    }
  }
  return {};
}

void sirius_physical_plan_generator::prefetch_iceberg_delete_data(
  sirius::op::sirius_physical_operator& plan)
{
  // Walk the plan tree and fully materialize delete data for every iceberg scan, mirroring
  // `sirius_engine::prefetch_iceberg_delete_data`. The engine's variant still runs for the
  // flag-off (legacy converter) path until Sub-phase E removes it; this variant feeds the
  // tree-based wrap performed by `wrap_table_scan_source` later in `create_plan`.
  if (plan.type != sirius::op::SiriusPhysicalOperatorType::TABLE_SCAN) {
    if (plan.type == sirius::op::SiriusPhysicalOperatorType::RESULT_COLLECTOR) {
      auto& collector = plan.Cast<sirius::op::sirius_physical_result_collector>();
      prefetch_iceberg_delete_data(collector.plan);
    } else {
      for (auto& child : plan.children) {
        prefetch_iceberg_delete_data(*child);
      }
    }
    return;
  }

  auto& scan_op = plan.Cast<sirius::op::sirius_physical_table_scan>();
  if (scan_op.function.name != "iceberg_scan") { return; }

  std::string const table_path = resolve_iceberg_table_path(scan_op);
  if (table_path.empty()) { return; }
  if (iceberg_delete_data_cache_.count(table_path)) { return; }  // already fetched

  // Extract snapshot parameters if present (for snapshot-aware delete discovery).
  std::optional<uint64_t> snapshot_id;
  auto sid_it = scan_op.named_parameters.find("snapshot_from_id");
  if (sid_it != scan_op.named_parameters.end() && !sid_it->second.IsNull()) {
    snapshot_id = sid_it->second.GetValue<uint64_t>();
  }

  // Opening secondary connections triggers QueryBegin/QueryEnd on the shared SiriusContext;
  // InternalQueryGuard suppresses those side-effects.
  auto sirius_ctx = context.registered_state->Get<duckdb::SiriusContext>("sirius_state");
  if (!sirius_ctx) {
    SIRIUS_LOG_WARN(
      "[sirius_physical_plan_generator] SiriusContext not available; treating iceberg '{}' as V1.",
      table_path);
    iceberg_delete_data_cache_.emplace(table_path,
                                       std::make_shared<sirius::op::scan::IcebergDeleteData>());
    return;
  }

  duckdb::SiriusContext::InternalQueryGuard guard(*sirius_ctx);

  // Iceberg metadata reads use a single GPU's sirius_ioctx (planning-time /
  // pre-execution; not on the multi-GPU column-chunk hot path). Multi-GPU
  // residency for iceberg metadata is deferred. Mirrors the engine's pre-existing
  // read_iceberg_delete_data call site.
  auto const& gpu_ioctxs = sirius_ctx->get_gpu_ioctxs();
  if (gpu_ioctxs.empty()) {
    throw std::runtime_error(
      "[sirius_physical_plan_generator] read_iceberg_delete_data: SiriusContext has no GPU "
      "sirius_ioctxs (kvikio path is forbidden).");
  }
  // Pick the lowest-numbered GPU id (deterministic ordering — get_gpu_ioctxs
  // returns an unordered_map, so use std::min_element rather than .begin()).
  auto lowest = std::min_element(gpu_ioctxs.begin(),
                                 gpu_ioctxs.end(),
                                 [](auto const& a, auto const& b) { return a.first < b.first; });
  auto data =
    sirius::op::scan::read_iceberg_delete_data(context, table_path, lowest->second, snapshot_id);
  iceberg_delete_data_cache_.emplace(table_path, std::move(data));
}

sirius::OrderPreservationType sirius_physical_plan_generator::order_preservation_recursive(
  sirius::op::sirius_physical_operator& op)
{
  if (op.is_source()) { return op.source_order(); }

  std::size_t child_idx = 0;
  for (auto& child : op.children) {
    // Do not take the materialization phase of physical CTEs into account
    if (op.type == sirius::op::SiriusPhysicalOperatorType::CTE && child_idx == 0) {
      child_idx++;
      continue;
    }
    auto child_preservation = order_preservation_recursive(*child);
    if (child_preservation != sirius::OrderPreservationType::INSERTION_ORDER) {
      return child_preservation;
    }
    child_idx++;
  }
  return sirius::OrderPreservationType::INSERTION_ORDER;
}

bool sirius_physical_plan_generator::preserve_insertion_order(
  duckdb::ClientContext& context, sirius::op::sirius_physical_operator& plan)
{
  auto preservation_type = order_preservation_recursive(plan);
  if (preservation_type == sirius::OrderPreservationType::FIXED_ORDER) {
    // always need to maintain preservation order
    return true;
  }
  if (preservation_type == sirius::OrderPreservationType::NO_ORDER) {
    // never need to preserve order
    return false;
  }
  // preserve insertion order - check flags
  if (!duckdb::Settings::Get<duckdb::PreserveInsertionOrderSetting>(context)) {
    // preserving insertion order is disabled by config
    return false;
  }
  return true;
}

bool sirius_physical_plan_generator::preserve_insertion_order(
  sirius::op::sirius_physical_operator& plan)
{
  return preserve_insertion_order(context, plan);
}

duckdb::unique_ptr<sirius::op::sirius_physical_operator>
sirius_physical_plan_generator::create_plan(duckdb::unique_ptr<duckdb::LogicalOperator> op)
{
  auto& profiler = duckdb::QueryProfiler::Get(context);

  // Resolve the types of each operator.
  profiler.StartPhase(duckdb::MetricType::PHYSICAL_PLANNER_RESOLVE_TYPES);
  op->ResolveOperatorTypes();
  profiler.EndPhase();

  // Resolve the column references.
  profiler.StartPhase(duckdb::MetricType::PHYSICAL_PLANNER_COLUMN_BINDING);
  duckdb::ColumnBindingResolver resolver;
  resolver.VisitOperator(*op);
  profiler.EndPhase();

  // then create the main physical plan
  profiler.StartPhase(duckdb::MetricType::PHYSICAL_PLANNER_CREATE_PLAN);
  auto plan = create_plan(*op);
  profiler.EndPhase();

  plan = fold_adjacent_projections(std::move(plan));
  plan->verify();

  // Phase 3 (#601) tree-based pipeline build. When the flag is on, rewrite the plan tree to
  // contain GPU pipeline operators (PARTITION/CONCAT/MERGE_*/SORT_*/scan companions/etc.)
  // so the converter becomes a pure topology pass driven by `build_pipelines` virtuals.
  // Default off; the legacy `sirius_pipeline_converter` is authoritative when the flag is
  // off. Iceberg delete data is pre-fetched before the tree rewrite so
  // `wrap_table_scan_source` can attach `delete_data` to each `sirius_physical_iceberg_scan`
  // it constructs. `set_parent_ops` then derives every operator's `_parent_op` from the
  // final tree, enabling Sub-phase C/D's tree-parent-lookup wiring.
  if (duckdb::Config::USE_TREE_BASED_PIPELINE_BUILD) {
    prefetch_iceberg_delete_data(*plan);
    insert_gpu_pipeline_operators(plan);
    set_parent_ops(*plan, /*parent=*/nullptr);
  }

  return plan;
}

duckdb::unique_ptr<sirius::op::sirius_physical_operator>
sirius_physical_plan_generator::create_plan(duckdb::LogicalOperator& op)
{
  SIRIUS_LOG_DEBUG("Creating sirius physical plan for logical operator type: {}",
                   duckdb::LogicalOperatorToString(op.type));
  op.estimated_cardinality                                      = op.EstimateCardinality(context);
  duckdb::unique_ptr<sirius::op::sirius_physical_operator> plan = nullptr;

  // SQLNULL-typed columns (e.g. an uncast NULL in VALUES) have no cuDF
  // representation — get_cudf_type() / fixed_width_byte_size() reject them at
  // execution time, after the GPU plan is already running. Reject the plan
  // here instead so transparent execution falls back to DuckDB CPU.
  for (const auto& type : op.types) {
    if (duckdb::TypeVisitor::Contains(type, duckdb::LogicalTypeId::SQLNULL)) {
      throw duckdb::NotImplementedException("SQLNULL-typed column not supported");
    }
  }

  switch (op.type) {
    case duckdb::LogicalOperatorType::LOGICAL_GET:
      plan = create_plan(op.Cast<duckdb::LogicalGet>());
      break;
    case duckdb::LogicalOperatorType::LOGICAL_PROJECTION:
      plan = create_plan(op.Cast<duckdb::LogicalProjection>());
      break;
    case duckdb::LogicalOperatorType::LOGICAL_EMPTY_RESULT:
      plan = create_plan(op.Cast<duckdb::LogicalEmptyResult>());
      break;
    case duckdb::LogicalOperatorType::LOGICAL_FILTER:
      plan = create_plan(op.Cast<duckdb::LogicalFilter>());
      break;
    case duckdb::LogicalOperatorType::LOGICAL_AGGREGATE_AND_GROUP_BY:
      plan = create_plan(op.Cast<duckdb::LogicalAggregate>());
      break;
    case duckdb::LogicalOperatorType::LOGICAL_WINDOW:
      throw duckdb::NotImplementedException("Window not supported");
      // plan = create_plan(op.Cast<duckdb::LogicalWindow>());
      break;
    case duckdb::LogicalOperatorType::LOGICAL_UNNEST:
      throw duckdb::NotImplementedException("Unnest not supported");
      // plan = create_plan(op.Cast<duckdb::LogicalUnnest>());
      break;
    case duckdb::LogicalOperatorType::LOGICAL_LIMIT:
      plan = create_plan(op.Cast<duckdb::LogicalLimit>());
      break;
    case duckdb::LogicalOperatorType::LOGICAL_SAMPLE:
      throw duckdb::NotImplementedException("Sample not supported");
      // plan = create_plan(op.Cast<duckdb::LogicalSample>());
      break;
    case duckdb::LogicalOperatorType::LOGICAL_ORDER_BY:
      plan = create_plan(op.Cast<duckdb::LogicalOrder>());
      break;
    case duckdb::LogicalOperatorType::LOGICAL_TOP_N:
      plan = create_plan(op.Cast<duckdb::LogicalTopN>());
      break;
    case duckdb::LogicalOperatorType::LOGICAL_COPY_TO_FILE:
      throw duckdb::NotImplementedException("Copy to file not supported");
      // plan = create_plan(op.Cast<duckdb::LogicalCopyToFile>());
      break;
    case duckdb::LogicalOperatorType::LOGICAL_DUMMY_SCAN:
      plan = create_plan(op.Cast<duckdb::LogicalDummyScan>());
      break;
    case duckdb::LogicalOperatorType::LOGICAL_ANY_JOIN:
      throw duckdb::NotImplementedException("Any join not supported");
      // plan = create_plan(op.Cast<duckdb::LogicalAnyJoin>());
      break;
    case duckdb::LogicalOperatorType::LOGICAL_ASOF_JOIN:
      throw duckdb::NotImplementedException("Asof join not supported");
      break;
    case duckdb::LogicalOperatorType::LOGICAL_DELIM_JOIN:
    case duckdb::LogicalOperatorType::LOGICAL_COMPARISON_JOIN:
      plan = create_plan(op.Cast<duckdb::LogicalComparisonJoin>());
      break;
    case duckdb::LogicalOperatorType::LOGICAL_CROSS_PRODUCT:
      throw duckdb::NotImplementedException("Cross product not supported");
      // plan = create_plan(op.Cast<duckdb::LogicalCrossProduct>());
      break;
    case duckdb::LogicalOperatorType::LOGICAL_POSITIONAL_JOIN:
      throw duckdb::NotImplementedException("Positional join not supported");
      // plan = create_plan(op.Cast<duckdb::LogicalPositionalJoin>());
      break;
    case duckdb::LogicalOperatorType::LOGICAL_UNION:
    case duckdb::LogicalOperatorType::LOGICAL_EXCEPT:
    case duckdb::LogicalOperatorType::LOGICAL_INTERSECT:
      throw duckdb::NotImplementedException("Set operation not supported");
      // plan = create_plan(op.Cast<duckdb::LogicalSetOperation>());
      break;
    case duckdb::LogicalOperatorType::LOGICAL_INSERT:
      throw duckdb::NotImplementedException("Insert not supported");
      // plan = create_plan(op.Cast<duckdb::LogicalInsert>());
      break;
    case duckdb::LogicalOperatorType::LOGICAL_DELETE:
      throw duckdb::NotImplementedException("Delete not supported");
      // plan = create_plan(op.Cast<duckdb::LogicalDelete>());
      break;
    case duckdb::LogicalOperatorType::LOGICAL_CHUNK_GET:
      plan = create_plan(op.Cast<duckdb::LogicalColumnDataGet>());
      break;
    case duckdb::LogicalOperatorType::LOGICAL_DELIM_GET:
      plan = create_plan(op.Cast<duckdb::LogicalDelimGet>());
      break;
    case duckdb::LogicalOperatorType::LOGICAL_EXPRESSION_GET:
      plan = create_plan(op.Cast<duckdb::LogicalExpressionGet>());
      break;
    case duckdb::LogicalOperatorType::LOGICAL_UPDATE:
      throw duckdb::NotImplementedException("Update not supported");
      // plan = create_plan(op.Cast<duckdb::LogicalUpdate>());
      break;
    case duckdb::LogicalOperatorType::LOGICAL_CREATE_TABLE:
      throw duckdb::NotImplementedException("Create table not supported");
      // plan = create_plan(op.Cast<duckdb::LogicalCreateTable>());
      break;
    case duckdb::LogicalOperatorType::LOGICAL_CREATE_INDEX:
      throw duckdb::NotImplementedException("Create index not supported");
      // plan = create_plan(op.Cast<duckdb::LogicalCreateIndex>());
      break;
    case duckdb::LogicalOperatorType::LOGICAL_CREATE_SECRET:
      throw duckdb::NotImplementedException("Create secret not supported");
      // plan = create_plan(op.Cast<duckdb::LogicalCreateSecret>());
      break;
    case duckdb::LogicalOperatorType::LOGICAL_EXPLAIN:
      throw duckdb::NotImplementedException("Explain not supported");
      // plan = create_plan(op.Cast<duckdb::LogicalExplain>());
      break;
    case duckdb::LogicalOperatorType::LOGICAL_DISTINCT:
      throw duckdb::NotImplementedException("Distinct not supported");
      // plan = create_plan(op.Cast<duckdb::LogicalDistinct>());
      break;
    case duckdb::LogicalOperatorType::LOGICAL_PREPARE:
      throw duckdb::NotImplementedException("Prepare not supported");
      // plan = create_plan(op.Cast<duckdb::LogicalPrepare>());
      break;
    case duckdb::LogicalOperatorType::LOGICAL_EXECUTE:
      throw duckdb::NotImplementedException("Execute not supported");
      // plan = create_plan(op.Cast<duckdb::LogicalExecute>());
      break;
    case duckdb::LogicalOperatorType::LOGICAL_CREATE_VIEW:
    case duckdb::LogicalOperatorType::LOGICAL_CREATE_SEQUENCE:
    case duckdb::LogicalOperatorType::LOGICAL_CREATE_SCHEMA:
    case duckdb::LogicalOperatorType::LOGICAL_CREATE_MACRO:
    case duckdb::LogicalOperatorType::LOGICAL_CREATE_TYPE:
      throw duckdb::NotImplementedException("Create not supported");
      // plan = create_plan(op.Cast<duckdb::LogicalCreate>());
      break;
    case duckdb::LogicalOperatorType::LOGICAL_PRAGMA:
      throw duckdb::NotImplementedException("Pragma not supported");
      // plan = create_plan(op.Cast<duckdb::LogicalPragma>());
      break;
    case duckdb::LogicalOperatorType::LOGICAL_VACUUM:
      throw duckdb::NotImplementedException("Vacuum not supported");
      // plan = create_plan(op.Cast<duckdb::LogicalVacuum>());
      break;
    case duckdb::LogicalOperatorType::LOGICAL_TRANSACTION:
    case duckdb::LogicalOperatorType::LOGICAL_ALTER:
    case duckdb::LogicalOperatorType::LOGICAL_DROP:
    case duckdb::LogicalOperatorType::LOGICAL_LOAD:
    case duckdb::LogicalOperatorType::LOGICAL_ATTACH:
    case duckdb::LogicalOperatorType::LOGICAL_DETACH:
      throw duckdb::NotImplementedException("Simple not supported");
      // plan = create_plan(op.Cast<duckdb::LogicalSimple>());
      break;
    case duckdb::LogicalOperatorType::LOGICAL_RECURSIVE_CTE:
      throw duckdb::NotImplementedException("Recursive CTE not supported");
      // plan = create_plan(op.Cast<duckdb::LogicalRecursiveCTE>());
      break;
    case duckdb::LogicalOperatorType::LOGICAL_MATERIALIZED_CTE:
      plan = create_plan(op.Cast<duckdb::LogicalMaterializedCTE>());
      break;
    case duckdb::LogicalOperatorType::LOGICAL_CTE_REF:
      plan = create_plan(op.Cast<duckdb::LogicalCTERef>());
      break;
    case duckdb::LogicalOperatorType::LOGICAL_EXPORT:
      throw duckdb::NotImplementedException("Export not supported");
      // plan = create_plan(op.Cast<duckdb::LogicalExport>());
      break;
    case duckdb::LogicalOperatorType::LOGICAL_SET:
      throw duckdb::NotImplementedException("Set not supported");
      // plan = create_plan(op.Cast<duckdb::LogicalSet>());
      break;
    case duckdb::LogicalOperatorType::LOGICAL_RESET:
      throw duckdb::NotImplementedException("Reset not supported");
      // plan = create_plan(op.Cast<duckdb::LogicalReset>());
      break;
    case duckdb::LogicalOperatorType::LOGICAL_PIVOT:
      throw duckdb::NotImplementedException("Pivot not supported");
      // plan = create_plan(op.Cast<duckdb::LogicalPivot>());
      break;
    case duckdb::LogicalOperatorType::LOGICAL_COPY_DATABASE:
      throw duckdb::NotImplementedException("Copy database not supported");
      // plan = create_plan(op.Cast<duckdb::LogicalCopyDatabase>());
      break;
    case duckdb::LogicalOperatorType::LOGICAL_UPDATE_EXTENSIONS:
      throw duckdb::NotImplementedException("Update extensions not supported");
      // plan = create_plan(op.Cast<duckdb::LogicalSimple>());
      break;
    case duckdb::LogicalOperatorType::LOGICAL_EXTENSION_OPERATOR:
      throw duckdb::NotImplementedException("Extension operator not supported");
      // plan = op.Cast<duckdb::LogicalExtensionOperator>().create_plan(context, *this);

      // if (!plan) {
      // 	throw duckdb::InternalException("Missing sirius_physical_operator for Extension
      // Operator");
      // }
      break;
    case duckdb::LogicalOperatorType::LOGICAL_JOIN:
    case duckdb::LogicalOperatorType::LOGICAL_DEPENDENT_JOIN:
    case duckdb::LogicalOperatorType::LOGICAL_INVALID: {
      throw duckdb::NotImplementedException("Unimplemented logical operator type!");
    }
    default: throw duckdb::NotImplementedException("Unimplemented logical operator type");
  }
  if (!plan) { throw duckdb::InternalException("Physical plan generator - no plan generated"); }

  plan->estimated_cardinality = op.estimated_cardinality;
#ifdef DUCKDB_VERIFY_VECTOR_OPERATOR
  auto verify = duckdb::make_uniq<duckdb::PhysicalVerifyVector>(std::move(plan));
  plan        = std::move(verify);
#endif

  return plan;
}

}  // namespace sirius::planner
