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

#include "pipeline/sirius_pipeline_converter.hpp"

#include "duckdb/common/shared_ptr_ipp.hpp"
#include "log/logging.hpp"
#include "op/scan/duckdb_native_gpu_ingestible.hpp"
#include "op/scan/parquet_gpu_ingestible.hpp"
#include "op/scan/sirius_gpu_scan_operator.hpp"
#include "op/sirius_physical_column_data_scan.hpp"
#include "op/sirius_physical_concat.hpp"
#include "op/sirius_physical_cte.hpp"
#include "op/sirius_physical_delim_join.hpp"
#include "op/sirius_physical_grouped_aggregate.hpp"
#include "op/sirius_physical_hash_join.hpp"
#include "op/sirius_physical_operator.hpp"
#include "op/sirius_physical_operator_type.hpp"
#include "op/sirius_physical_partition.hpp"

#include <algorithm>
#include <functional>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace sirius::pipeline {

sirius_pipeline_converter::sirius_pipeline_converter(const pipeline_build_context& ctx,
                                                     const sirius::operator_params& op_params)
  : build_ctx_(ctx), op_params_(op_params)
{
}

pipeline_conversion_result sirius_pipeline_converter::convert(sirius_meta_pipeline& root_pipeline)
{
  scheduled_.clear();
  repository_wirings_.clear();

  scheduled_ = schedule_pipelines(root_pipeline);
  compute_repository_wiring(root_pipeline.get_state());
  setup_pipeline_parents();
  finalize_pipeline_structure();
  link_join_partition_siblings();
  configure_partition_min_partitions();
  // Must run after finalize_pipeline_structure (populates `dependencies`) and after
  // link_join_partition_siblings (reads dependencies[0]/[1] positionally pre-reorder).
  reorder_pipelines_topologically(scheduled_);

  return {std::move(scheduled_), std::move(repository_wirings_), meta_pipeline_count_};
}

void reorder_pipelines_topologically(duckdb::vector<duckdb::shared_ptr<sirius_pipeline>>& pipelines)
{
  duckdb::vector<duckdb::shared_ptr<sirius_pipeline>> ordered;
  ordered.reserve(pipelines.size());
  std::unordered_set<const sirius_pipeline*> emitted;
  std::unordered_set<const sirius_pipeline*> in_progress;

  auto emit = [&](auto&& self, const duckdb::shared_ptr<sirius_pipeline>& pipeline) -> void {
    // `in_progress` breaks dependency cycles (delim-join distribution edges); the
    // pipeline is still emitted when its own frame completes.
    if (emitted.contains(pipeline.get()) || in_progress.contains(pipeline.get())) { return; }
    in_progress.insert(pipeline.get());
    for (const auto& producer : pipeline->dependencies) {
      self(self, producer);
    }
    in_progress.erase(pipeline.get());
    emitted.insert(pipeline.get());
    ordered.push_back(pipeline);
  };

  for (const auto& pipeline : pipelines) {
    if (pipeline->get_parents().empty()) { emit(emit, pipeline); }
  }
  // Safety net for pipelines unreachable from a sink-side root.
  for (const auto& pipeline : pipelines) {
    emit(emit, pipeline);
  }
  D_ASSERT(ordered.size() == pipelines.size());
  pipelines = std::move(ordered);

  for (size_t i = 0; i < pipelines.size(); i++) {
    pipelines[i]->set_pipeline_id(i);
  }
  for (const auto& pipeline : pipelines) {
    std::sort(pipeline->dependencies.begin(),
              pipeline->dependencies.end(),
              [](const duckdb::shared_ptr<sirius_pipeline>& a,
                 const duckdb::shared_ptr<sirius_pipeline>& b) {
                return a->get_pipeline_id() < b->get_pipeline_id();
              });
  }
#ifdef DEBUG
  // Join dependencies are build-first (finalize_pipeline_structure) and the walk above
  // visits slot 0 first, so every build concat is emitted — and numbered — before its
  // probe sibling and the ascending re-sort keeps it in slot 0. Delim-join cycle
  // breaking is the one path that could reorder them; catch that here instead of
  // silently regressing dynamic-filter publish-before-probe scheduling.
  for (const auto& pipeline : pipelines) {
    if (pipeline->get_source()->type != op::SiriusPhysicalOperatorType::HASH_JOIN &&
        pipeline->get_source()->type != op::SiriusPhysicalOperatorType::NESTED_LOOP_JOIN) {
      continue;
    }
    auto build_sink = pipeline->dependencies[0]->get_sink();
    D_ASSERT(build_sink->type == op::SiriusPhysicalOperatorType::CONCAT &&
             build_sink->Cast<op::sirius_physical_concat>().is_build_concat());
  }
#endif
}

duckdb::vector<duckdb::shared_ptr<sirius_pipeline>> sirius_pipeline_converter::schedule_pipelines(
  sirius_meta_pipeline& root_pipeline)
{
  duckdb::vector<duckdb::shared_ptr<sirius_meta_pipeline>> to_schedule;
  duckdb::vector<duckdb::shared_ptr<sirius_pipeline>> sirius_scheduled;
  scheduled_.clear();
  root_pipeline.get_meta_pipelines(to_schedule, true, true);

  // number of 'PipelineCompleteEvent's is equal to the number of meta pipelines, so we have to
  // set it here
  meta_pipeline_count_ = to_schedule.size();

  SIRIUS_LOG_DEBUG("Total meta pipelines {}", to_schedule.size());
  int schedule_count = 0;
  int meta           = 0;
  while (schedule_count < to_schedule.size()) {
    duckdb::vector<duckdb::shared_ptr<sirius_meta_pipeline>> children;
    to_schedule[to_schedule.size() - 1 - meta]->get_meta_pipelines(children, false, true);
    auto base_pipeline   = to_schedule[to_schedule.size() - 1 - meta]->get_base_pipeline();
    bool should_schedule = true;

    // already scheduled
    if (std::ranges::find(sirius_scheduled, base_pipeline) != sirius_scheduled.end()) {
      should_schedule = false;
    } else {
      // check if all children are scheduled
      for (auto& child : children) {
        if (std::ranges::find(sirius_scheduled, child->get_base_pipeline()) ==
            sirius_scheduled.end()) {
          should_schedule = false;
          break;
        }
      }
      // check if all dependencies are scheduled
      for (const auto& dependency : base_pipeline->dependencies) {
        if (std::ranges::find(sirius_scheduled, dependency) == sirius_scheduled.end()) {
          should_schedule = false;
          break;
        }
      }
    }
    if (should_schedule) {
      duckdb::vector<duckdb::shared_ptr<sirius_pipeline>> pipeline_inside;
      to_schedule[to_schedule.size() - 1 - meta]->get_pipelines(pipeline_inside, false);
      for (auto& pipeline : pipeline_inside) {
        sirius_scheduled.push_back(pipeline);
      }
      schedule_count++;
    }
    meta = (meta + 1) % to_schedule.size();
  }

  // The pipelines are already final-shape; return the originals so pointer-identity
  // lookups (`state.cte_scan_consumers`) still resolve.
  return sirius_scheduled;
}

std::string_view sirius_pipeline_converter::resolve_port_id(
  const op::sirius_physical_operator& sink, const op::sirius_physical_operator& /*parent*/)
{
  using T = op::SiriusPhysicalOperatorType;
  // Build-side CONCATs feed the join's "build" port; everything else feeds "default".
  if (sink.type == T::CONCAT) {
    return sink.Cast<op::sirius_physical_concat>().is_build_concat() ? "build" : "default";
  }
  return "default";
}

op::MemoryBarrierType sirius_pipeline_converter::resolve_barrier(
  const op::sirius_physical_operator& sink, const sirius_pipeline& dest)
{
  using T = op::SiriusPhysicalOperatorType;
  // Sort sinks process batches as they arrive. GPU_SCAN is intentionally excluded
  // (legacy gives it FULL/PARTIAL, not PIPELINE); SORT_SAMPLE is never a pipeline
  // sink (it runs as an intermediate in the SORT_PARTITION pipeline).
  if (sink.type == T::ORDER_BY) { return op::MemoryBarrierType::PIPELINE; }
  // Producers that feed CONCAT can drain incrementally (PARTIAL); otherwise wait
  // for the upstream pipeline to finish (FULL).
  if (sink.type == T::PARTITION || sink.type == T::UNGROUPED_AGGREGATE || sink.type == T::TOP_N ||
      sink.type == T::SORT_PARTITION) {
    const auto ops                  = dest.get_operators();
    const bool downstream_is_concat = (!ops.empty() && ops[0].get().type == T::CONCAT) ||
                                      (dest.get_sink() && dest.get_sink()->type == T::CONCAT);
    return downstream_is_concat ? op::MemoryBarrierType::PARTIAL : op::MemoryBarrierType::FULL;
  }
  // Probe-side PARTITION under a join streams batches → PARTIAL; build-side PARTITION is
  // FULL (build must accumulate every partition before the probe can join). Aggregate-
  // fanout PARTITIONs are also FULL — the merge operator needs every per-thread bucket.
  // Join-feeders sit under a CONCAT (wrap_join_child's wrap chain); aggregate-fanouts sit
  // under MERGE_GROUP_BY / GROUPED_AGGREGATE_MERGE. Exception: a RIGHT-family join sizes
  // from the complete probe input (CONCAT retains the whole probe partition), so its probe
  // PARTITION stays FULL — unless it is a RIGHT_DELIM_JOIN's internal join, which
  // bootstraps its probe subtree from build-side distinct data.
  if (dest.get_sink() && dest.get_sink()->type == T::PARTITION) {
    auto& partition        = dest.get_sink()->Cast<op::sirius_physical_partition>();
    auto* partition_parent = partition.get_parent_op();
    const bool join_feeder = partition_parent != nullptr && partition_parent->type == T::CONCAT;
    // Tree-parent walk: PARTITION -> CONCAT -> owning join (stamped at plan-gen).
    auto* join = join_feeder ? partition_parent->get_parent_op() : nullptr;
    const bool right_family_full =
      join && join->type == T::HASH_JOIN &&
      join->Cast<op::sirius_physical_hash_join>().is_right_family() &&
      !(join->get_parent_op() && join->get_parent_op()->type == T::RIGHT_DELIM_JOIN);
    if (!partition.is_build_partition() && join_feeder && !right_family_full) {
      return op::MemoryBarrierType::PARTIAL;
    }
  }
  return op::MemoryBarrierType::FULL;
}

void sirius_pipeline_converter::compute_repository_wiring(sirius_pipeline_build_state& state)
{
  // Lookup: operator -> the pipeline that starts at it, i.e. its operators[0]
  // (entry-point post-reverse) or its sink for sink-only pipelines.
  std::unordered_map<const op::sirius_physical_operator*, duckdb::shared_ptr<sirius_pipeline>>
    dest_for_op;
  for (const auto& pipeline : scheduled_) {
    const auto ops = pipeline->get_operators();
    if (!ops.empty()) { dest_for_op[&ops[0].get()] = pipeline; }
    if (pipeline->get_sink()) { dest_for_op[pipeline->get_sink().get()] = pipeline; }
  }

  // Assign pipeline IDs before emitting wiring descriptors. Runtime materialization
  // uses these to sort `_ports_list` deterministically.
  for (size_t i = 0; i < scheduled_.size(); i++) {
    scheduled_[i]->set_pipeline_id(i);
  }

  auto emit = [&](std::string_view port_id,
                  op::MemoryBarrierType barrier,
                  op::sirius_physical_operator* source_op,
                  const duckdb::shared_ptr<sirius_pipeline>& src,
                  const duckdb::shared_ptr<sirius_pipeline>& dst) {
    repository_wirings_.push_back({port_id, barrier, source_op, src, dst});
  };

  for (auto& pipeline : scheduled_) {
    auto* sink_op = pipeline->get_sink().get();
    if (!sink_op) { continue; }

    using T = op::SiriusPhysicalOperatorType;

    // RESULT_COLLECTOR is a terminal sink — nothing to emit.
    if (sink_op->type == T::RESULT_COLLECTOR) { continue; }

    // CTE fans out to its sibling `cte_scans` (parent_op alone doesn't encode them).
    // CTE_SCAN never lands in any pipeline's operators[], so consumers resolve via
    // `state.cte_scan_consumers` instead of `dest_for_op`.
    if (sink_op->type == T::CTE) {
      auto& cte_op = sink_op->Cast<op::sirius_physical_cte>();
      for (auto cte_scan : cte_op.cte_scans) {
        auto it = state.cte_scan_consumers.find(cte_scan);
        if (it == state.cte_scan_consumers.end()) { continue; }
        auto dest_pipeline = it->second.get().shared_from_this();
        // Per-consumer barrier: probe-side CTE_SCAN consumers resolve to PARTIAL via the
        // join-feeder rule; build-side consumers resolve to FULL.
        emit(
          "default", resolve_barrier(*sink_op, *dest_pipeline), sink_op, pipeline, dest_pipeline);
      }
      continue;
    }

    // A delim join is a fan-out sink: its base sink() pushes each input batch to both branch
    // pipelines, which are first-class pipelines sourced from the delim join. Emit one edge per
    // branch with the delim join as the producer; the sub-ops' outward edges (PARTITION_build →
    // CONCAT_build, DISTINCT → PARTITION_distinct, column_data_scan → PARTITION_probe) come from
    // the uniform tree-parent lookup below.
    if (sink_op->type == T::RIGHT_DELIM_JOIN || sink_op->type == T::LEFT_DELIM_JOIN) {
      auto& delim = sink_op->Cast<op::sirius_physical_delim_join>();
      // Branch 1: the side that feeds the inner join (RHS build / LHS probe).
      op::sirius_physical_operator* join_side = nullptr;
      if (sink_op->type == T::RIGHT_DELIM_JOIN) {
        join_side = sink_op->Cast<op::sirius_physical_right_delim_join>().partition_join;
      } else {
        join_side = sink_op->Cast<op::sirius_physical_left_delim_join>().column_data_scan;
      }
      // Branch 2: the duplicate-elimination (distinct) side.
      op::sirius_physical_operator* distinct_op = delim.distinct;
      for (auto* branch : {join_side, distinct_op}) {
        if (!branch) { continue; }
        auto it = dest_for_op.find(branch);
        if (it == dest_for_op.end()) { continue; }
        emit(resolve_port_id(*sink_op, *branch),
             resolve_barrier(*sink_op, *it->second),
             sink_op,
             pipeline,
             it->second);
      }
      continue;
    }

    // A DELIM_JOIN's distinct chain top (MERGE_GROUP_BY) sits under DELIM_JOIN in the
    // tree, but its merged output retargets to each delim_scan's downstream consumer
    // (the inner-HJ probe partition). Only the distinct_root carries the
    // `_owning_delim_join` back-pointer.
    if (auto* owning_delim = sink_op->owning_delim_join()) {
      for (auto& delim_scan_ref : owning_delim->delim_scans) {
        auto& delim_scan  = delim_scan_ref.get();
        auto* scan_parent = delim_scan.get_parent_op();
        if (!scan_parent) { continue; }
        auto cit = dest_for_op.find(scan_parent);
        if (cit == dest_for_op.end()) { continue; }
        emit(resolve_port_id(*sink_op, *scan_parent),
             resolve_barrier(*sink_op, *cit->second),
             sink_op,
             pipeline,
             cit->second);
      }
      continue;
    }

    // Uniform tree-parent lookup for everything else.
    auto* parent_op = sink_op->get_parent_op();
    if (!parent_op) { continue; }

    // A RIGHT_DELIM_JOIN's `delim.join` wires out to the RDJ's tree parent, skipping the
    // RDJ itself. Otherwise it and the root HJ of
    // `RDJ.children[0]` would both resolve to the RDJ pipeline, and the RDJ-sink
    // emission's CONCAT fallback would close a cycle through the inner HJ's build CONCAT.
    if ((parent_op->type == T::RIGHT_DELIM_JOIN) &&
        parent_op->Cast<op::sirius_physical_delim_join>().join.get() == sink_op) {
      auto* grand = parent_op->get_parent_op();
      if (grand) {
        auto it_gp = dest_for_op.find(grand);
        if (it_gp != dest_for_op.end()) {
          emit(resolve_port_id(*sink_op, *grand),
               resolve_barrier(*sink_op, *it_gp->second),
               sink_op,
               pipeline,
               it_gp->second);
          continue;
        }
      }
    }

    auto it = dest_for_op.find(parent_op);
    if (it == dest_for_op.end()) { continue; }

    const auto& dest = it->second;
    emit(resolve_port_id(*sink_op, *parent_op),
         resolve_barrier(*sink_op, *dest),
         sink_op,
         pipeline,
         dest);
  }
}

void sirius_pipeline_converter::setup_pipeline_parents()
{
  // Derive parents off the wiring descriptors instead of reading materialised ports —
  // ports aren't attached until `materialize_repository_wiring()` runs after `convert()`
  // returns. Each descriptor encodes a `source_pipeline -> dest_pipeline` edge that the
  // old code derived from `add_next_port_after_sink({next_op, port_id})`
  for (const auto& pipeline : scheduled_) {
    pipeline->parents.clear();
    pipeline->dependencies.clear();
  }
  for (const auto& wiring : repository_wirings_) {
    wiring.source_pipeline->parents.push_back(
      duckdb::weak_ptr<sirius_pipeline>(wiring.dest_pipeline));
  }
}

void sirius_pipeline_converter::finalize_pipeline_structure()
{
  // `is_ready` already derived source/sink from operators[]; all that remains is the
  // parent->dependency population.
  for (const auto& pipeline : scheduled_) {
    // For each parent pipeline, add the current pipeline to its dependencies. Join
    // dependencies must end up build-side-first: link_join_partition_siblings() reads
    // dependencies[0]/[1] positionally, and reorder_pipelines_topologically() visits
    // producers in slot order — build-first is what schedules a join's dynamic-filter
    // publication before the probe-side scans it prunes. The meta-sweep emits
    // probe-first, so build concats are inserted at the front instead.
    for (auto& parent : pipeline->parents) {
      auto locked_parent = parent.lock();
      if (!locked_parent) { continue; }
      bool const build_side_of_join =
        (locked_parent->source->type == op::SiriusPhysicalOperatorType::HASH_JOIN ||
         locked_parent->source->type == op::SiriusPhysicalOperatorType::NESTED_LOOP_JOIN) &&
        pipeline->sink->type == op::SiriusPhysicalOperatorType::CONCAT &&
        pipeline->sink->Cast<op::sirius_physical_concat>().is_build_concat();
      if (build_side_of_join) {
        locked_parent->dependencies.insert(locked_parent->dependencies.begin(), pipeline);
      } else {
        locked_parent->dependencies.push_back(pipeline);
      }
    }
  }
}

void sirius_pipeline_converter::link_join_partition_siblings()
{
  for (const auto& pipeline : scheduled_) {
    // Both join types get the same CONCAT/PARTITION build+probe wrap from `wrap_join`, and
    // both can stream the probe, so the upstream→probe-partition edge is PARTIAL for both.
    if (pipeline->source->type == op::SiriusPhysicalOperatorType::HASH_JOIN ||
        pipeline->source->type == op::SiriusPhysicalOperatorType::NESTED_LOOP_JOIN) {
      auto build_concat_pipeline    = pipeline->dependencies[0];
      auto build_partition_pipeline = build_concat_pipeline->dependencies[0];
      auto probe_concat_pipeline    = pipeline->dependencies[1];
      auto probe_partition_pipeline = probe_concat_pipeline->dependencies[0];
      // Positional roles are guaranteed by finalize_pipeline_structure() (tree path) /
      // emission order (legacy). A swap here mislinks sibling partitions and, for
      // right-family joins, drives_partition_count.
      D_ASSERT(
        build_concat_pipeline->get_sink()->type == op::SiriusPhysicalOperatorType::CONCAT &&
        build_concat_pipeline->get_sink()->Cast<op::sirius_physical_concat>().is_build_concat());
      // A RIGHT_DELIM_JOIN's inner join bootstraps its probe subtree from build-side (distinct)
      // data, so the build side drives the partition count. Detect it off the join's tree parent
      // (the same predicate resolve_barrier uses), since the build partition is a plain PARTITION
      // that the sink type alone does not distinguish.
      bool const inner_join_of_rdj =
        pipeline->source->get_parent_op() != nullptr &&
        pipeline->source->get_parent_op()->type == op::SiriusPhysicalOperatorType::RIGHT_DELIM_JOIN;
      // Right-family sizing applies to hash joins only — NLJ probe partitions always stream.
      bool const probe_drives_partition_count =
        pipeline->source->type == op::SiriusPhysicalOperatorType::HASH_JOIN &&
        pipeline->source->Cast<op::sirius_physical_hash_join>().is_right_family() &&
        !inner_join_of_rdj;

      // partition pipeline only has one operator, so sink and source are the same
      auto& build_partition_op =
        build_partition_pipeline->get_sink()->Cast<op::sirius_physical_partition>();
      auto& probe_partition_op =
        probe_partition_pipeline->get_sink()->Cast<op::sirius_physical_partition>();
      build_partition_op.set_sibling_partition_op(&probe_partition_op);
      probe_partition_op.set_sibling_partition_op(&build_partition_op);
      if (probe_drives_partition_count) {
        build_partition_op.set_drives_partition_count(false);
        probe_partition_op.set_drives_partition_count(true);
      }
    }
  }
}

void sirius_pipeline_converter::configure_partition_min_partitions()
{
  // Pull num_gpus from the build context (populated from sirius_engine's
  // hardware topology at convert time). Single-GPU runs keep the default
  // min of 1 (no-op). For multi-GPU we force a floor equal to num_gpus on
  // big-enough inputs; small_table_bytes keeps tiny aggregations on a
  // single GPU to avoid cross-device overhead.
  const int num_gpus = build_ctx_.num_gpus();
  if (num_gpus <= 1) return;
  // Heuristic threshold: below ~16 MiB per GPU the partition overhead
  // dominates. Configurable later if we find a workload where this matters.
  const uint64_t small_table_bytes = static_cast<uint64_t>(num_gpus) * uint64_t{16} * 1024 * 1024;

  auto apply_to_op = [&](op::sirius_physical_operator* op) {
    if (op && op->type == op::SiriusPhysicalOperatorType::PARTITION) {
      static_cast<op::sirius_physical_partition*>(op)->set_min_num_partitions(num_gpus,
                                                                              small_table_bytes);
    }
  };
  for (auto& pipe : scheduled_) {
    if (!pipe) continue;
    auto sink   = pipe->get_sink();
    auto source = pipe->get_source();
    if (sink) apply_to_op(sink.get());
    if (source) apply_to_op(source.get());
  }
}

namespace {

std::string dump_op_name(const op::sirius_physical_operator* op)
{
  return op == nullptr ? std::string{"(null)"} : op::SiriusPhysicalOperatorToString(op->type);
}

std::string dump_barrier_name(op::MemoryBarrierType b)
{
  switch (b) {
    case op::MemoryBarrierType::PIPELINE: return "PIPELINE";
    case op::MemoryBarrierType::PARTIAL: return "PARTIAL";
    case op::MemoryBarrierType::FULL: return "FULL";
  }
  return "?";
}

//! Scan identity: serialize what the ingestible will scan, so a conversion that drops
//! identity fields (e.g. the duckdb-native pin-cache qualified name, or a parquet file
//! list) fails the dump byte-diff instead of passing on an identical operator-type chain.
//! Conversion-time only: table_info is still parked on the operator (scan_manager takes it
//! at prepare_for_query).
void dump_scan_identity(std::ostringstream& out, const op::sirius_physical_operator& op)
{
  if (op.type != op::SiriusPhysicalOperatorType::GPU_SCAN) { return; }
  auto const& info = op.Cast<op::scan::sirius_gpu_scan_operator>().peek_table_info();
  if (auto const* pq = dynamic_cast<op::scan::parquet_ingestible_table_info const*>(&info)) {
    out << "      scan: parquet files=[";
    for (std::size_t f = 0; f < pq->resolved_file_paths.size(); ++f) {
      out << (f == 0 ? "" : ",") << pq->resolved_file_paths[f];
    }
    out << "]\n";
  } else if (auto const* nat =
               dynamic_cast<op::scan::duckdb_native_ingestible_table_info const*>(&info)) {
    out << "      scan: duckdb table=" << nat->catalog_name << "." << nat->schema_name << "."
        << nat->table_name << "\n";
  }
}

//! One `[pipeline N]` block: source/sink/operators with per-scan identity, shared by the
//! canonical and raw dumps so both describe a single pipeline identically.
void dump_pipeline_block(std::ostringstream& out, std::size_t index, const sirius_pipeline& p)
{
  out << "[pipeline " << index << "]\n";
  out << "  source: " << dump_op_name(p.get_source().get()) << "\n";
  out << "  sink: " << dump_op_name(p.get_sink().get()) << "\n";
  const auto ops = p.get_operators();
  out << "  operators (" << ops.size() << "):\n";
  std::size_t op_idx = 0;
  for (const auto& op_ref : ops) {
    out << "    [" << op_idx++ << "] " << dump_op_name(&op_ref.get()) << "\n";
    dump_scan_identity(out, op_ref.get());
  }
}

//! Wiring lines with endpoints resolved through `index_of`, sorted by those indices —
//! insensitive to wiring emission order, sensitive to the caller's pipeline order.
void dump_wirings(
  std::ostringstream& out,
  const std::vector<repository_wiring>& wirings,
  const std::function<std::size_t(const duckdb::shared_ptr<sirius_pipeline>&)>& index_of)
{
  auto pipeline_index = [&](const duckdb::shared_ptr<sirius_pipeline>& p) -> std::string {
    auto idx = index_of(p);
    return idx == std::numeric_limits<std::size_t>::max() ? std::string{"?"} : std::to_string(idx);
  };
  std::vector<repository_wiring> sorted_wirings(wirings.begin(), wirings.end());
  std::sort(sorted_wirings.begin(),
            sorted_wirings.end(),
            [&](const repository_wiring& a, const repository_wiring& b) -> bool {
              auto a_src = index_of(a.source_pipeline);
              auto b_src = index_of(b.source_pipeline);
              if (a_src != b_src) { return a_src < b_src; }
              auto a_dst = index_of(a.dest_pipeline);
              auto b_dst = index_of(b.dest_pipeline);
              if (a_dst != b_dst) { return a_dst < b_dst; }
              return a.port_id < b.port_id;
            });

  out << "\n=== repository_wirings (" << sorted_wirings.size() << ") ===\n";
  for (std::size_t i = 0; i < sorted_wirings.size(); ++i) {
    const auto& w = sorted_wirings[i];
    out << "[wiring " << i << "]\n";
    out << "  port_id: " << w.port_id << "\n";
    out << "  barrier: " << dump_barrier_name(w.barrier_type) << "\n";
    out << "  source_op: " << dump_op_name(w.source_op) << "\n";
    out << "  src_pipeline: " << pipeline_index(w.source_pipeline) << "\n";
    out << "  dest_pipeline: " << pipeline_index(w.dest_pipeline) << "\n";
  }
}

}  // namespace

std::string dump_pipeline_conversion_result(const pipeline_conversion_result& result)
{
  // Per-pipeline local signature: sink|source|operators...
  auto local_sig = [&](const sirius_pipeline* p) -> std::string {
    std::string s = dump_op_name(p->get_sink().get());
    s += "|" + dump_op_name(p->get_source().get());
    for (const auto& op_ref : p->get_operators()) {
      s += "|" + dump_op_name(&op_ref.get());
    }
    return s;
  };

  // Downstream-aware signature: local shape plus the sorted (port_id, downstream_signature)
  // list, so pipelines only match when both shape and consumer set match. Memoized.
  std::unordered_map<const sirius_pipeline*, std::string> sig_cache;
  // Cycle guard: return "CYCLE" on re-entry instead of recursing forever, so a buggy
  // cyclic wiring graph surfaces as a dump mismatch rather than a hang.
  std::unordered_set<const sirius_pipeline*> sig_visiting;
  std::function<std::string(const sirius_pipeline*)> compute_sig =
    [&](const sirius_pipeline* p) -> std::string {
    auto it = sig_cache.find(p);
    if (it != sig_cache.end()) { return it->second; }
    if (sig_visiting.count(p)) { return "CYCLE"; }
    sig_visiting.insert(p);
    std::vector<std::string> down;
    for (const auto& w : result.repository_wirings) {
      if (w.source_pipeline.get() == p) {
        down.push_back(std::string{w.port_id} + ":" + compute_sig(w.dest_pipeline.get()));
      }
    }
    sig_visiting.erase(p);
    std::sort(down.begin(), down.end());
    std::string s = local_sig(p);
    for (const auto& d : down) {
      s += "|>" + d;
    }
    sig_cache[p] = s;
    return s;
  };

  // Canonical order: sort by signature so the dump is independent of emission order —
  // equivalent graphs print byte-identical output.
  duckdb::vector<duckdb::shared_ptr<sirius_pipeline>> ordered = result.scheduled_pipelines;
  std::sort(ordered.begin(), ordered.end(), [&](const auto& a, const auto& b) -> bool {
    return compute_sig(a.get()) < compute_sig(b.get());
  });

  std::unordered_map<const sirius_pipeline*, std::size_t> pipeline_to_index;
  for (std::size_t i = 0; i < ordered.size(); ++i) {
    pipeline_to_index[ordered[i].get()] = i;
  }
  auto idx_of = [&](const duckdb::shared_ptr<sirius_pipeline>& p) -> std::size_t {
    auto it = pipeline_to_index.find(p.get());
    return it == pipeline_to_index.end() ? std::numeric_limits<std::size_t>::max() : it->second;
  };

  std::ostringstream out;
  out << "=== pipelines (" << ordered.size() << ") ===\n";
  for (std::size_t i = 0; i < ordered.size(); ++i) {
    dump_pipeline_block(out, i, *ordered[i]);
  }

  // Wirings keyed by canonical pipeline indices so wiring order is also path-independent.
  dump_wirings(out, result.repository_wirings, idx_of);
  return out.str();
}

std::string dump_pipeline_schedule_raw(const pipeline_conversion_result& result)
{
  const auto& scheduled = result.scheduled_pipelines;
  std::unordered_map<const sirius_pipeline*, std::size_t> position;
  for (std::size_t i = 0; i < scheduled.size(); ++i) {
    position[scheduled[i].get()] = i;
  }
  auto idx_of = [&](const duckdb::shared_ptr<sirius_pipeline>& p) -> std::size_t {
    auto it = position.find(p.get());
    return it == position.end() ? std::numeric_limits<std::size_t>::max() : it->second;
  };
  auto idx_str = [&](const duckdb::shared_ptr<sirius_pipeline>& p) -> std::string {
    auto idx = idx_of(p);
    return idx == std::numeric_limits<std::size_t>::max() ? std::string{"?"} : std::to_string(idx);
  };

  std::ostringstream out;
  out << "=== scheduled pipelines (" << scheduled.size() << ") ===\n";
  for (std::size_t i = 0; i < scheduled.size(); ++i) {
    dump_pipeline_block(out, i, *scheduled[i]);
    out << "  deps: [";
    for (std::size_t d = 0; d < scheduled[i]->dependencies.size(); ++d) {
      out << (d == 0 ? "" : ",") << idx_str(scheduled[i]->dependencies[d]);
    }
    out << "]\n";
  }
  dump_wirings(out, result.repository_wirings, idx_of);
  return out.str();
}

}  // namespace sirius::pipeline
