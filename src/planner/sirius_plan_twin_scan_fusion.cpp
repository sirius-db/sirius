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

//! Twin-scan fusion: fuse two near-duplicate probe-side scans of the same table into ONE
//! fan-out scan pipeline that decodes and dynamic-filters the shared stream once.
//!
//! Motivating shape (TPC-H q21): DuckDB's delim decomposition of chained EXISTS / NOT EXISTS
//! leaves two full scans of the same table whose column sets nest, whose static filters differ
//! only by one residual predicate, and whose dynamic membership filters are Blooms over provably
//! nested key sets. Fusing them halves the decode + Bloom-probe work.
//!
//! The pass runs on the physical operator tree after `create_plan` / the compressed-schema
//! passes and BEFORE `insert_gpu_pipeline_operators` (the scans are still TABLE_SCAN nodes
//! carrying their `sirius_dynamic_filters` channels; see sirius_physical_plan_generator.cpp).
//! It is decomposed into `collect_sites` (candidate shapes), `match_scan_geometry` (is fusing
//! mechanically well-formed), `prove_channel_subsumption` (is dropping B's channel semantically
//! safe), and `fuse_pair` (the rewrite). Each rejected condition names a
//! `twin_scan_rejection_reason`.
//!
//! Safety invariants (A = the bare scan whose slot becomes the split, B = the residual-filtered
//! scan that becomes the fused scan):
//!
//!  - I1 -- Prefix property (index remapping is structurally impossible). A's `column_ids` must
//!    be a strict prefix of B's, AND A's effective output layout (projection_ids or identity),
//!    output types, and physical-carrier sidecar entries must be prefixes of B's. Consequences
//!    relied on: (a) every producer-side filter-target index of A's channel -- which is in
//!    column_ids space and remapped to output positions by the fused scan's own ingestible via
//!    `set_consumer_column_remap` -- is valid on the fused scan unchanged; (b) out-A is the
//!    identity projection of the first `width_A` fused output columns, so out-A is row-for-row
//!    A's original output. There is deliberately NO remapping code anywhere; any future
//!    relaxation of the prefix condition must reject, not remap. Checks:
//!    `columns_not_strict_prefix`, `output_layout_not_prefix`, `output_types_not_prefix`,
//!    `physical_carriers_differ`.
//!  - I2 -- Keyset subsumption, proven structurally over multi-key delim tuples. keys(B) must be
//!    a subset of keys(A) so A's membership filters never drop a row B's authoritative join
//!    would keep. Proof obligations: B's delim join's input is DIRECTLY A's delim join
//!    (`delim_chain_not_direct`); A's join-back is RIGHT_SEMI / RIGHT_ANTI -- the only join
//!    types emitting a row-subset of A's delim input with schema preserved
//!    (`join_back_not_row_subset`); both delim distincts have identical group references and
//!    types (`delim_distinct_missing`, `delim_key_refs_differ`) -- then the delim key sets nest
//!    as TUPLE sets even for multi-column delim keys, and tuple containment projects to
//!    containment on any single key column; finally both producer joins must key on the same
//!    single build-side column position (`producer_key_not_single_equality`,
//!    `producer_keys_differ`, `producer_key_outside_delim_output`), so the published membership
//!    filters are over that same projected column.
//!  - I3 -- Advisory filters only; never a static/correctness filter. The only thing the pass
//!    ever drops is B's dynamic (advisory) channel -- legal solely because of I2 plus the fact
//!    that the downstream join is authoritative for extra rows (out-B may carry keys(A) minus
//!    keys(B) rows; the join drops them). Static pushed table filters are never dropped: they
//!    must be identical on both scans or the pair is rejected (`static_filters_differ`). B's
//!    residual predicate is never dropped: it moves into the split and is evaluated per batch.
//!  - I4 -- The proof must cover every producer of both channels. Each channel must have
//!    producers (`channel_without_producer`), none unscoped -- an unscoped producer may publish
//!    on any column and `planned_target_columns()` is documented as meaningless then
//!    (`channel_unscoped_producer`) -- exactly one planned target column
//!    (`channel_multi_target`) on the same underlying table column, not rowid, in range
//!    (`channel_target_invalid`, `channel_targets_differ`), and exactly one hash join whose
//!    `dynamic_filter_plan().probe_targets()` reference the channel
//!    (`producer_join_not_unique`). Channels must be distinct objects with distinct producer
//!    joins (`channel_shared`, `producer_joins_identical`).
//!  - I5 -- Output equivalence. out-A is A's baseline output row-for-row and byte-for-byte (same
//!    columns, same static filters, same channel). out-B is a superset of B's baseline output,
//!    superset only by keys(A) minus keys(B) rows, re-filtered by B's authoritative downstream
//!    join. The split's finalize log line (`rows_in/out_a/out_b`) exists to validate exactly
//!    this against an unfused run.
//!  - I6 -- Tree/lifetime invariant. The split holds a non-owning `TWIN_SCAN_REF*`; both nodes
//!    are installed atomically into the same plan tree and destroyed with it. No pass may detach
//!    one without the other; today none runs between fusion and pipeline insertion, and the
//!    converter / `sink()` consistency checks fail loudly if the pairing breaks.
//!
//! Rewrite: A's slot becomes TWIN_SCAN_SPLIT over B's scan (which keeps its own column set but
//! now consumes A's dynamic-filter channel); B's slot becomes the routing-only TWIN_SCAN_REF.
//! B's channel is closed so its producer skips filter construction.

#include "planner/sirius_plan_twin_scan_fusion.hpp"

#include "duckdb/catalog/catalog_entry/table_catalog_entry.hpp"
#include "duckdb/function/table/table_scan.hpp"
#include "duckdb/planner/table_filter.hpp"
#include "log/logging.hpp"
#include "op/sirius_dynamic_filter.hpp"
#include "op/sirius_physical_delim_join.hpp"
#include "op/sirius_physical_filter.hpp"
#include "op/sirius_physical_grouped_aggregate.hpp"
#include "op/sirius_physical_hash_join.hpp"
#include "op/sirius_physical_table_scan.hpp"
#include "op/sirius_physical_twin_scan_split.hpp"
#include "planner/sirius_physical_plan_generator.hpp"

#include <format>
#include <numeric>
#include <optional>
#include <string>
#include <vector>

namespace sirius::planner {

namespace {

using sirius::op::SiriusPhysicalOperatorType;
using op_ptr = duckdb::unique_ptr<sirius::op::sirius_physical_operator>;

//! A fusion candidate: a probe-side slot holding either a bare TABLE_SCAN (A shape) or a
//! FILTER directly over a TABLE_SCAN (B shape).
struct twin_site {
  op_ptr* slot;
  sirius::op::sirius_physical_table_scan* scan;
  sirius::op::sirius_physical_filter* residual;  // nullptr for the bare shape
};

//! Every fusion candidate site in one plan tree plus the full operator census the
//! subsumption proof searches for producer joins and delim joins.
struct site_census {
  std::vector<twin_site> sites;
  std::vector<sirius::op::sirius_physical_operator*> all;
};

//! Post-order walk collecting `slot` and its subtree (including delim-join internal subtrees)
//! into @p census. @p child_idx is `slot`'s index in @p parent's `children[]`, or nullopt for
//! the delim-internal `join`/`distinct_root` edges, which are never probe edges.
void collect_sites_recursive(op_ptr& slot,
                             sirius::op::sirius_physical_operator* parent,
                             std::optional<std::size_t> child_idx,
                             site_census& census)
{
  if (!slot) { return; }
  census.all.push_back(slot.get());

  const bool probe_child = parent != nullptr &&
                           parent->type == SiriusPhysicalOperatorType::HASH_JOIN &&
                           child_idx == std::size_t{0};
  if (probe_child) {
    if (slot->type == SiriusPhysicalOperatorType::TABLE_SCAN && slot->children.empty()) {
      census.sites.push_back(
        {&slot, &slot->Cast<sirius::op::sirius_physical_table_scan>(), nullptr});
    } else if (slot->type == SiriusPhysicalOperatorType::FILTER && slot->children.size() == 1 &&
               slot->children[0]->type == SiriusPhysicalOperatorType::TABLE_SCAN &&
               slot->children[0]->children.empty()) {
      census.sites.push_back({&slot,
                              &slot->children[0]->Cast<sirius::op::sirius_physical_table_scan>(),
                              &slot->Cast<sirius::op::sirius_physical_filter>()});
    }
  }

  for (std::size_t i = 0; i < slot->children.size(); ++i) {
    collect_sites_recursive(slot->children[i], slot.get(), i, census);
  }
  if (slot->type == SiriusPhysicalOperatorType::LEFT_DELIM_JOIN ||
      slot->type == SiriusPhysicalOperatorType::RIGHT_DELIM_JOIN) {
    auto& delim = slot->Cast<sirius::op::sirius_physical_delim_join>();
    collect_sites_recursive(delim.join, slot.get(), std::nullopt, census);
    collect_sites_recursive(delim.distinct_root, slot.get(), std::nullopt, census);
  }
}

//! Collect every fusion candidate site and the full operator census of @p plan.
site_census collect_sites(op_ptr& plan)
{
  site_census census;
  collect_sites_recursive(plan, nullptr, std::nullopt, census);
  return census;
}

//! One-line scan geometry summary for the detection logs and rejection records: column ids,
//! projection ids, output width, physical-sidecar presence, and the dynamic-filter channel
//! state. Planned target columns are shown only when their accessor is meaningful
//! (producer-backed, no unscoped producer).
std::string describe_scan_geometry(const sirius::op::sirius_physical_table_scan& scan)
{
  std::string cols;
  for (auto const& ci : scan.column_ids) {
    cols += (cols.empty() ? "" : ",") +
            (ci.IsRowIdColumn() ? std::string("rowid") : std::to_string(ci.GetPrimaryIndex()));
  }
  std::string projs;
  for (auto const& p : scan.projection_ids) {
    projs += (projs.empty() ? "" : ",") + std::to_string(p);
  }
  std::string channel_state = "none";
  std::string planned;
  if (scan.sirius_dynamic_filters) {
    auto const& channel = *scan.sirius_dynamic_filters;
    if (!channel.has_producers()) {
      channel_state = "no-producers";
    } else if (channel.has_unscoped_producer()) {
      channel_state = "unscoped-producer";
    } else {
      channel_state = "producer-backed";
      for (auto const p : channel.planned_target_columns()) {
        planned += (planned.empty() ? "" : ",") + std::to_string(p);
      }
    }
  }
  return std::format("col_ids=[{}] proj_ids=[{}] out_width={} phys={} chan={} planned=[{}]",
                     cols,
                     projs,
                     scan.types.size(),
                     scan.has_physical_overrides(),
                     channel_state,
                     planned);
}

//! Same-relation identity: seq_scan by resolved catalog entry, parquet family by resolved
//! file path list. Anything unresolvable is not the same table.
bool same_table_identity(const sirius::op::sirius_physical_table_scan& x,
                         const sirius::op::sirius_physical_table_scan& y)
{
  if (x.function.name != y.function.name) { return false; }
  if (x.function.name == "seq_scan") {
    auto const* bx = dynamic_cast<duckdb::TableScanBindData const*>(x.bind_data.get());
    auto const* by = dynamic_cast<duckdb::TableScanBindData const*>(y.bind_data.get());
    return bx != nullptr && by != nullptr && &bx->table == &by->table;
  }
  if (x.function.name == "parquet_scan" || x.function.name == "read_parquet" ||
      x.function.name == "sirius_read_parquet") {
    auto px = resolve_parquet_scan_file_paths(x.function.name, x.bind_data.get(), x.parameters);
    auto py = resolve_parquet_scan_file_paths(y.function.name, y.bind_data.get(), y.parameters);
    return !px.empty() && px == py;
  }
  return false;
}

//! Effective output layout of a scan: projection_ids, or the identity over column_ids.
std::vector<std::size_t> effective_output(const sirius::op::sirius_physical_table_scan& scan)
{
  if (!scan.projection_ids.empty()) {
    return {scan.projection_ids.begin(), scan.projection_ids.end()};
  }
  std::vector<std::size_t> identity(scan.column_ids.size());
  std::iota(identity.begin(), identity.end(), 0);
  return identity;
}

//! Whether fusing (a := shared output, b := residual-filtered output) is mechanically
//! well-formed: the I1 prefix ladder, identical static filters (I3), and agreeing physical
//! carriers on the shared prefix. Pure scan-vs-scan; no channel or join knowledge. The caller
//! has already established same-table identity via `same_table_identity` (cross-table pairs
//! are not candidates and are never recorded). Returns the rejection reason, or nullopt on a
//! match. @p sa / @p sb are non-const because `TableFilterSet::Equals` is a non-const member.
std::optional<twin_scan_rejection_reason> match_scan_geometry(
  sirius::op::sirius_physical_table_scan& sa, sirius::op::sirius_physical_table_scan& sb)
{
  // --- Column geometry: A must be a strict prefix of B, in column_ids AND output layout. ---
  // This is what makes A's channel indexes (column_ids space on the producer side) valid on
  // the fused scan without remapping.
  if (sa.column_ids.size() >= sb.column_ids.size()) {
    return twin_scan_rejection_reason::columns_not_strict_prefix;
  }
  for (std::size_t i = 0; i < sa.column_ids.size(); ++i) {
    if (!(sa.column_ids[i] == sb.column_ids[i])) {
      return twin_scan_rejection_reason::columns_not_strict_prefix;
    }
  }
  auto eff_a = effective_output(sa);
  auto eff_b = effective_output(sb);
  if (eff_a.size() > eff_b.size()) { return twin_scan_rejection_reason::output_layout_not_prefix; }
  for (std::size_t i = 0; i < eff_a.size(); ++i) {
    if (eff_a[i] != eff_b[i]) { return twin_scan_rejection_reason::output_layout_not_prefix; }
  }
  if (sa.types.size() != eff_a.size() || sa.types.size() >= sb.types.size()) {
    return twin_scan_rejection_reason::output_types_not_prefix;
  }
  for (std::size_t i = 0; i < sa.types.size(); ++i) {
    if (!(sa.types[i] == sb.types[i])) {
      return twin_scan_rejection_reason::output_types_not_prefix;
    }
  }

  // --- Physical carriers on the shared prefix must agree (downstream ops were planned
  //     against A's carriers; the fused scan materializes B's). Requiring equal sidecar
  //     PRESENCE even when only B's residual-only columns are narrowed is deliberate
  //     over-rejection: it keeps this check independent of how sidecars are derived. ---
  if (sa.has_physical_overrides() != sb.has_physical_overrides()) {
    return twin_scan_rejection_reason::physical_carriers_differ;
  }
  if (sa.has_physical_overrides()) {
    auto const& pa = sa.get_physical_types();
    auto const& pb = sb.get_physical_types();
    // Sidecar sizes track the column widths already prefix-checked above; the size guard is
    // defensive so the indexing below can never rely on that invariant silently.
    if (pa.size() > pb.size()) { return twin_scan_rejection_reason::physical_carriers_differ; }
    for (std::size_t i = 0; i < pa.size(); ++i) {
      if (!(pa[i] == pb[i])) { return twin_scan_rejection_reason::physical_carriers_differ; }
    }
  }

  // --- Static filters must be identical: the fused scan runs B's, so A's must equal them. ---
  const bool empty_a = !sa.table_filters || sa.table_filters->filters.empty();
  const bool empty_b = !sb.table_filters || sb.table_filters->filters.empty();
  if (empty_a != empty_b) { return twin_scan_rejection_reason::static_filters_differ; }
  if (!empty_a && !sa.table_filters->Equals(*sb.table_filters)) {
    return twin_scan_rejection_reason::static_filters_differ;
  }

  return std::nullopt;
}

//! The unique hash join publishing into `channel`, or nullptr (none, or more than one).
sirius::op::sirius_physical_hash_join* unique_producer_join(
  const std::vector<sirius::op::sirius_physical_operator*>& all,
  const std::shared_ptr<sirius::op::sirius_dynamic_filter_set>& channel)
{
  sirius::op::sirius_physical_hash_join* found = nullptr;
  for (auto* node : all) {
    if (node->type != SiriusPhysicalOperatorType::HASH_JOIN) { continue; }
    auto& join = node->Cast<sirius::op::sirius_physical_hash_join>();
    for (auto const& target : join.dynamic_filter_plan().probe_targets()) {
      if (target.filter_set.get() != channel.get()) { continue; }
      if (found != nullptr && found != &join) { return nullptr; }
      found = &join;
    }
  }
  return found;
}

//! The delim join whose duplicate-eliminated keys feed `join`'s build side (children[1]
//! descends through unary operators to a DELIM_SCAN owned by that delim join), or nullptr.
sirius::op::sirius_physical_delim_join* delim_feeding_build(
  const std::vector<sirius::op::sirius_physical_operator*>& all,
  sirius::op::sirius_physical_hash_join& join)
{
  if (join.children.size() < 2) { return nullptr; }
  auto* node = join.children[1].get();
  while (node != nullptr && node->type != SiriusPhysicalOperatorType::DELIM_SCAN &&
         node->children.size() == 1) {
    node = node->children[0].get();
  }
  if (node == nullptr || node->type != SiriusPhysicalOperatorType::DELIM_SCAN) { return nullptr; }
  for (auto* candidate : all) {
    if (candidate->type != SiriusPhysicalOperatorType::LEFT_DELIM_JOIN &&
        candidate->type != SiriusPhysicalOperatorType::RIGHT_DELIM_JOIN) {
      continue;
    }
    auto& delim = candidate->Cast<sirius::op::sirius_physical_delim_join>();
    for (auto const& scan_ref : delim.delim_scans) {
      if (&scan_ref.get() == node) { return &delim; }
    }
  }
  return nullptr;
}

//! The build-side column position of @p join's single equality condition, or nullopt when
//! the join has more or fewer than one equality, or its build side is not a plain column
//! reference. The membership filters a join publishes are built from its equality build
//! keys, so a unique plain-reference key pins down exactly which build column feeds them.
std::optional<uint32_t> equality_build_key(const sirius::op::sirius_physical_hash_join& join)
{
  std::optional<uint32_t> key;
  for (auto const& cond : join.conditions) {
    if (cond.comparison != sirius::comparison_type::equal) { continue; }
    if (key.has_value()) { return std::nullopt; }  // more than one equality
    if (!cond.right || !cond.right->holds<sirius::ast::reference>()) { return std::nullopt; }
    key = cond.right->get<sirius::ast::reference>().column_index;
  }
  return key;
}

//! Whether dropping B's dynamic-filter channel in favor of A's is semantically safe: the
//! channel obligations of I4 followed by the I2 delim-chain proof of keys(B) contained in
//! keys(A). Returns the rejection reason, or nullopt when the proof holds.
std::optional<twin_scan_rejection_reason> prove_channel_subsumption(
  const sirius::op::sirius_physical_table_scan& sa,
  const sirius::op::sirius_physical_table_scan& sb,
  const std::vector<sirius::op::sirius_physical_operator*>& all)
{
  // --- Dynamic-filter channels: single-column targets on the same table column (I4). ---
  auto chan_a = sa.sirius_dynamic_filters;
  auto chan_b = sb.sirius_dynamic_filters;
  if (!chan_a || !chan_b) { return twin_scan_rejection_reason::channel_missing; }
  if (chan_a == chan_b) { return twin_scan_rejection_reason::channel_shared; }
  if (!chan_a->has_producers() || !chan_b->has_producers()) {
    return twin_scan_rejection_reason::channel_without_producer;
  }
  // An unscoped producer may publish filters on any column, and planned_target_columns() is
  // documented as meaningful only when no producer is unscoped -- the proof below cannot
  // cover such a producer, so reject.
  if (chan_a->has_unscoped_producer() || chan_b->has_unscoped_producer()) {
    return twin_scan_rejection_reason::channel_unscoped_producer;
  }
  auto planned_a = chan_a->planned_target_columns();
  auto planned_b = chan_b->planned_target_columns();
  if (planned_a.size() != 1 || planned_b.size() != 1) {
    return twin_scan_rejection_reason::channel_multi_target;
  }
  if (planned_a[0] >= sa.column_ids.size() || planned_b[0] >= sb.column_ids.size()) {
    return twin_scan_rejection_reason::channel_target_invalid;
  }
  if (sa.column_ids[planned_a[0]].IsRowIdColumn() || sb.column_ids[planned_b[0]].IsRowIdColumn()) {
    return twin_scan_rejection_reason::channel_target_invalid;
  }
  if (sa.column_ids[planned_a[0]].GetPrimaryIndex() !=
      sb.column_ids[planned_b[0]].GetPrimaryIndex()) {
    return twin_scan_rejection_reason::channel_targets_differ;
  }

  // --- Key-set subsumption keys(B) in keys(A): the delim-chain structural proof (I2). ---
  auto* join_a = unique_producer_join(all, chan_a);
  auto* join_b = unique_producer_join(all, chan_b);
  if (join_a == nullptr || join_b == nullptr) {
    return twin_scan_rejection_reason::producer_join_not_unique;
  }
  if (join_a == join_b) { return twin_scan_rejection_reason::producer_joins_identical; }
  auto* delim_a = delim_feeding_build(all, *join_a);
  auto* delim_b = delim_feeding_build(all, *join_b);
  if (delim_a == nullptr || delim_b == nullptr) {
    return twin_scan_rejection_reason::build_not_delim_replay;
  }
  if (delim_a == delim_b) { return twin_scan_rejection_reason::delim_joins_identical; }
  if (delim_b->children.empty() || delim_b->children[0].get() != delim_a) {
    return twin_scan_rejection_reason::delim_chain_not_direct;
  }
  if (!delim_a->join || delim_a->join->type != SiriusPhysicalOperatorType::HASH_JOIN) {
    return twin_scan_rejection_reason::join_back_not_row_subset;
  }
  auto const join_back_type =
    delim_a->join->Cast<sirius::op::sirius_physical_hash_join>().join_type;
  if (join_back_type != duckdb::JoinType::RIGHT_SEMI &&
      join_back_type != duckdb::JoinType::RIGHT_ANTI) {
    // Only these emit a row-subset of A's delim input with the schema preserved, which is
    // what lets equal distinct group references prove the key containment below.
    return twin_scan_rejection_reason::join_back_not_row_subset;
  }
  auto* distinct_a = delim_a->distinct;
  auto* distinct_b = delim_b->distinct;
  if (distinct_a == nullptr || distinct_b == nullptr) {
    return twin_scan_rejection_reason::delim_distinct_missing;
  }
  if (distinct_a->group_idx != distinct_b->group_idx || !(distinct_a->types == distinct_b->types)) {
    return twin_scan_rejection_reason::delim_key_refs_differ;
  }
  // Multi-column delim keys are fine: keys(D_B) is contained in keys(D_A) as TUPLE sets
  // (row-subset input + identical group refs), and tuple containment projects to containment
  // on any one key column. What that projection argument needs is that both producer joins
  // build their membership filter from the SAME delim key column: each join must have exactly
  // one equality condition, keyed on the same build-side column position.
  auto build_key   = equality_build_key(*join_a);
  auto build_key_b = equality_build_key(*join_b);
  if (!build_key.has_value() || !build_key_b.has_value()) {
    return twin_scan_rejection_reason::producer_key_not_single_equality;
  }
  if (*build_key != *build_key_b) { return twin_scan_rejection_reason::producer_keys_differ; }
  if (*build_key >= distinct_a->types.size()) {
    return twin_scan_rejection_reason::producer_key_outside_delim_output;
  }

  return std::nullopt;
}

//! Rewrite one matched pair. See the file comment for the resulting shape. The ref is
//! installed into B's slot first (destroying the residual FILTER whose residual, output mask,
//! and scan have already been moved out), then the split into A's slot; A is a leaf and the
//! two slots live in unrelated parents, so neither write invalidates the other (I6).
void fuse_pair(twin_site& a, twin_site& b)
{
  auto& residual_filter = *b.residual;

  auto ref = duckdb::make_uniq<sirius::op::sirius_physical_twin_scan_ref>(
    residual_filter.types, residual_filter.estimated_cardinality);
  if (residual_filter.has_physical_overrides()) {
    ref->set_physical_types(residual_filter.get_physical_types());
  }
  auto* ref_ptr = ref.get();

  // Lift the residual out of the FILTER and take B's scan as the fused scan before the
  // FILTER node is destroyed by the slot replacement below.
  auto residual_expr = std::move(residual_filter.expression);
  auto output_mask_b = std::move(residual_filter.output_columns);
  auto types_b       = residual_filter.types;
  auto fused_scan    = std::move(residual_filter.children[0]);

  auto& scan_b = fused_scan->Cast<sirius::op::sirius_physical_table_scan>();
  auto chan_b  = scan_b.sirius_dynamic_filters;
  // The fused scan consumes A's channel; A's producer indexes stay valid because A's
  // column_ids are a prefix of B's (checked by match_scan_geometry). B's channel loses its
  // only consumer -- close it so its producer join skips filter construction entirely.
  scan_b.sirius_dynamic_filters = a.scan->sirius_dynamic_filters;
  scan_b.dynamic_filters        = a.scan->dynamic_filters;
  if (chan_b) { chan_b->close_for_new_filters(); }

  std::vector<cudf::size_type> output_indices_a(a.scan->types.size());
  std::iota(output_indices_a.begin(), output_indices_a.end(), 0);

  auto split =
    duckdb::make_uniq<sirius::op::sirius_physical_twin_scan_split>(a.scan->types,
                                                                   std::move(output_indices_a),
                                                                   std::move(residual_expr),
                                                                   std::move(output_mask_b),
                                                                   std::move(types_b),
                                                                   a.scan->estimated_cardinality,
                                                                   ref_ptr);
  if (a.scan->has_physical_overrides()) { split->set_physical_types(a.scan->get_physical_types()); }
  split->children.push_back(std::move(fused_scan));

  *b.slot = std::move(ref);    // destroys the residual FILTER node
  *a.slot = std::move(split);  // destroys A's scan node
}

}  // namespace

std::string_view to_string(twin_scan_rejection_reason reason) noexcept
{
  using enum twin_scan_rejection_reason;
  switch (reason) {
    case columns_not_strict_prefix: return "columns_not_strict_prefix";
    case output_layout_not_prefix: return "output_layout_not_prefix";
    case output_types_not_prefix: return "output_types_not_prefix";
    case physical_carriers_differ: return "physical_carriers_differ";
    case static_filters_differ: return "static_filters_differ";
    case channel_missing: return "channel_missing";
    case channel_shared: return "channel_shared";
    case channel_without_producer: return "channel_without_producer";
    case channel_unscoped_producer: return "channel_unscoped_producer";
    case channel_multi_target: return "channel_multi_target";
    case channel_target_invalid: return "channel_target_invalid";
    case channel_targets_differ: return "channel_targets_differ";
    case producer_join_not_unique: return "producer_join_not_unique";
    case producer_joins_identical: return "producer_joins_identical";
    case build_not_delim_replay: return "build_not_delim_replay";
    case delim_joins_identical: return "delim_joins_identical";
    case delim_chain_not_direct: return "delim_chain_not_direct";
    case join_back_not_row_subset: return "join_back_not_row_subset";
    case delim_distinct_missing: return "delim_distinct_missing";
    case delim_key_refs_differ: return "delim_key_refs_differ";
    case producer_key_not_single_equality: return "producer_key_not_single_equality";
    case producer_keys_differ: return "producer_keys_differ";
    case producer_key_outside_delim_output: return "producer_key_outside_delim_output";
  }
  return "unknown_rejection_reason";
}

twin_scan_fusion_report fuse_twin_scans(
  duckdb::unique_ptr<sirius::op::sirius_physical_operator>& plan)
{
  twin_scan_fusion_report report;
  bool changed = true;
  // Re-walk after every rewrite: a fusion invalidates collected slots. O(sites^2) per
  // iteration is irrelevant at planner scale. A same-table rejection is recorded once per
  // ordered pair per walk iteration.
  while (changed) {
    changed     = false;
    auto census = collect_sites(plan);

    // Candidate census: one line per collected site so a detection gap is diagnosable from
    // the log without a debugger (which shape was collected, with what geometry).
    for (auto const& s : census.sites) {
      SIRIUS_LOG_DEBUG("[twin_scan_fusion] site fn={} residual={} {}",
                       s.scan->function.name,
                       s.residual != nullptr,
                       describe_scan_geometry(*s.scan));
    }

    for (auto& a : census.sites) {
      if (a.residual != nullptr) { continue; }
      for (auto& b : census.sites) {
        if (b.residual == nullptr || a.scan == b.scan) { continue; }
        if (!same_table_identity(*a.scan, *b.scan)) { continue; }
        auto reject = match_scan_geometry(*a.scan, *b.scan);
        if (!reject.has_value()) {
          reject = prove_channel_subsumption(*a.scan, *b.scan, census.all);
        }
        if (reject.has_value()) {
          // A same-table candidate pair that fails a condition is rare (a handful per query
          // at most) and names an actionable detection gap -- worth INFO and a report record.
          auto geometry_a = describe_scan_geometry(*a.scan);
          auto geometry_b = describe_scan_geometry(*b.scan);
          SIRIUS_LOG_INFO("[twin_scan_fusion] same-table pair rejected: {} | A: {} | B: {}",
                          to_string(*reject),
                          geometry_a,
                          geometry_b);
          report.same_table_rejections.push_back(
            {*reject, std::move(geometry_a), std::move(geometry_b)});
          continue;
        }
        std::string table_name = a.scan->function.name;
        if (auto const* bind =
              dynamic_cast<duckdb::TableScanBindData const*>(a.scan->bind_data.get())) {
          table_name = bind->table.name;
        }
        SIRIUS_LOG_INFO(
          "[twin_scan_fusion] fusing twin scans of '{}': shared {} columns + {} residual-only "
          "columns, shared dynamic-filter channel adopted from the wider key set",
          table_name,
          a.scan->types.size(),
          b.scan->types.size() - a.scan->types.size());
        fuse_pair(a, b);
        ++report.fused_pairs;
        changed = true;
        break;
      }
      if (changed) { break; }
    }
  }
  return report;
}

}  // namespace sirius::planner
