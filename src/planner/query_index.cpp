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

#include "planner/query_index.hpp"

#include "planner/query.hpp"

#include <algorithm>
#include <limits>
#include <string_view>
#include <unordered_set>
#include <utility>

namespace sirius::planner {

namespace {

using pipeline_ptr = query_index::pipeline_ptr;

/// A directed data-flow edge between two pipelines, tagged with the memory barrier of the port
/// through which the producer feeds the consumer.
struct dag_edge {
  pipeline_ptr other;  ///< producer (for an incoming edge) or consumer (for an outgoing edge)
  op::MemoryBarrierType barrier;
};

/// Pipeline-level data-flow graph derived from the operator ports. `outgoing[P]` lists the
/// pipelines P feeds; `incoming[C]` lists the pipelines that feed C, with the connecting barrier.
struct pipeline_dag {
  std::unordered_map<pipeline_ptr, std::vector<dag_edge>> outgoing;
  std::unordered_map<pipeline_ptr, std::vector<dag_edge>> incoming;

  const std::vector<dag_edge>& out(pipeline_ptr p) const { return lookup(outgoing, p); }
  const std::vector<dag_edge>& in(pipeline_ptr p) const { return lookup(incoming, p); }

  /// A consumer is "multiport" (a fan-in branch point) when more than one pipeline feeds it.
  bool is_multiport(pipeline_ptr c) const { return in(c).size() > 1; }

  /// An edge into `consumer` (carrying `barrier`) cuts the branch when the consumer is multiport,
  /// and -- in barrier_order -- only when that edge is a FULL barrier.
  bool cuts(pipeline_ptr consumer, op::MemoryBarrierType barrier, bool barrier_aware) const
  {
    if (!is_multiport(consumer)) { return false; }
    return !barrier_aware || barrier == op::MemoryBarrierType::FULL;
  }

 private:
  static const std::vector<dag_edge>& lookup(
    const std::unordered_map<pipeline_ptr, std::vector<dag_edge>>& m, pipeline_ptr p)
  {
    static const std::vector<dag_edge> empty;
    auto it = m.find(p);
    return it == m.end() ? empty : it->second;
  }
};

/// The build side of a HASH_JOIN feeds its "build" port; every other input port is a probe side.
constexpr std::string_view kHashJoinBuildPort = "build";

/// Build the pipeline DAG from the sink operators' next-ports. Each next-port names the downstream
/// consumer operator and the port it pushes into; that port carries the barrier and identifies the
/// consumer pipeline via the operator's owning pipeline.
///
/// @param probe_as_pipeline When true (build_probe strategy), an edge feeding the probe side of a
///        HASH_JOIN consumer is recorded as a PIPELINE barrier regardless of the port's real
///        barrier, so the probe pipeline extends through the join.
pipeline_dag build_dag(
  const duckdb::vector<duckdb::shared_ptr<pipeline::sirius_pipeline>>& pipelines,
  bool probe_as_pipeline)
{
  pipeline_dag dag;
  for (const auto& producer_sp : pipelines) {
    pipeline_ptr producer = producer_sp.get();
    if (producer == nullptr) { continue; }
    for (const auto& next : producer->get_next_ports_after_sink()) {
      auto* consumer_op = next.next_operator;
      if (consumer_op == nullptr) { continue; }
      pipeline_ptr consumer = consumer_op->get_pipeline().get();
      if (consumer == nullptr || consumer == producer) { continue; }
      auto barrier = op::MemoryBarrierType::FULL;
      if (auto* port = consumer_op->get_port(next.next_operator_port_name)) {
        barrier = port->type;
      }
      if (probe_as_pipeline && consumer_op->type == op::SiriusPhysicalOperatorType::HASH_JOIN &&
          next.next_operator_port_name != kHashJoinBuildPort) {
        barrier = op::MemoryBarrierType::PIPELINE;  // probe side always pipelines through the join
      }
      dag.outgoing[producer].push_back({consumer, barrier});
      dag.incoming[consumer].push_back({producer, barrier});
    }
  }
  return dag;
}

/// A pipeline heads a branch unless it is absorbed into a producer's branch -- i.e. unless some
/// producer edge into it is not a cut AND that producer feeds only this pipeline (so the producer's
/// forward walk continues into it).
bool is_branch_head(const pipeline_dag& dag, pipeline_ptr p, bool barrier_aware)
{
  const auto& producers = dag.in(p);
  if (producers.empty()) { return true; }  // a scan (no producers) always heads a branch
  for (const auto& edge : producers) {
    const bool cut = dag.cuts(p, edge.barrier, barrier_aware);
    if (!cut && dag.out(edge.other).size() == 1) { return false; }
  }
  return true;
}

/// Walk consumer edges from a branch head, absorbing pipelines until the chain forks, ends, or
/// hits a cut edge.
std::vector<pipeline_ptr> walk_branch(const pipeline_dag& dag,
                                      pipeline_ptr head,
                                      bool barrier_aware)
{
  std::vector<pipeline_ptr> chain{head};
  std::unordered_set<pipeline_ptr> visited{head};
  pipeline_ptr cur = head;
  while (true) {
    const auto& consumers = dag.out(cur);
    if (consumers.size() != 1) { break; }  // fork or dead end ends the branch
    const dag_edge& edge = consumers.front();
    if (dag.cuts(edge.other, edge.barrier, barrier_aware)) { break; }
    if (!visited.insert(edge.other).second) { break; }  // defensive cycle guard
    chain.push_back(edge.other);
    cur = edge.other;
  }
  return chain;
}

// ---------------------------------------------------------------------------
// Prefetch ordering
// ---------------------------------------------------------------------------
//
// A second, port-level view of the same DAG. build_dag above is deliberately not reused: it
// drops the consumer *operator* (needed here to name the branch), and build_probe rewrites
// hash-join probe barriers to PIPELINE, which would erase exactly the FULL edges this
// traversal keys on.

/// An edge tagged with the barrier of the port it enters and the operator owning that port.
struct pf_edge {
  pipeline_ptr other;  ///< producer for an incoming edge, consumer for an outgoing one
  op::MemoryBarrierType barrier;
  op::sirius_physical_operator* consumer_op;  ///< owns the port; the branch operator at a fan-in
};

struct pf_dag {
  std::unordered_map<pipeline_ptr, std::vector<pf_edge>> outgoing;
  std::unordered_map<pipeline_ptr, std::vector<pf_edge>> incoming;

  [[nodiscard]] const std::vector<pf_edge>& out(pipeline_ptr p) const
  {
    return lookup(outgoing, p);
  }
  [[nodiscard]] const std::vector<pf_edge>& in(pipeline_ptr p) const { return lookup(incoming, p); }

  /// A fan-in: more than one pipeline feeds it. This is what "branch" means here.
  [[nodiscard]] bool is_branch(pipeline_ptr p) const { return in(p).size() > 1; }

 private:
  static const std::vector<pf_edge>& lookup(
    const std::unordered_map<pipeline_ptr, std::vector<pf_edge>>& m, pipeline_ptr p)
  {
    static const std::vector<pf_edge> empty;
    auto it = m.find(p);
    return it == m.end() ? empty : it->second;
  }
};

pf_dag build_pf_dag(const std::vector<pipeline_ptr>& pipelines)
{
  pf_dag dag;
  for (pipeline_ptr producer : pipelines) {
    if (producer == nullptr) { continue; }
    for (const auto& next : producer->get_next_ports_after_sink()) {
      auto* consumer_op = next.next_operator;
      if (consumer_op == nullptr) { continue; }
      pipeline_ptr consumer = consumer_op->get_pipeline().get();
      if (consumer == nullptr || consumer == producer) { continue; }
      auto barrier = op::MemoryBarrierType::FULL;  // conservative when the port is unreadable
      if (auto* port = consumer_op->get_port(next.next_operator_port_name)) {
        barrier = port->type;
      }
      dag.outgoing[producer].push_back({consumer, barrier, consumer_op});
      dag.incoming[consumer].push_back({producer, barrier, consumer_op});
    }
  }
  return dag;
}

[[nodiscard]] bool is_gpu_scan(const op::sirius_physical_operator* o)
{
  return o != nullptr && o->type == op::SiriusPhysicalOperatorType::GPU_SCAN;
}

/// The GPU scan a leaf pipeline reads through, or null. A scan pipeline may stack operators
/// above the scan (e.g. DYNAMIC_FILTER), so the source is checked first and then the chain.
op::sirius_physical_operator* scan_of(pipeline_ptr p)
{
  if (p == nullptr) { return nullptr; }
  if (auto source = p->get_source(); is_gpu_scan(source.get())) { return source.get(); }
  for (auto& ref : p->get_operators()) {
    if (is_gpu_scan(&ref.get())) { return &ref.get(); }
  }
  return nullptr;
}

/// Classify a scan by walking *downstream* from its pipeline until the barrier that gates it
/// is known.
///
///   - first branch reached through a FULL port          -> barrier_all
///   - a later branch reached through a FULL port        -> barrier_serial
///   - no branch is ever reached through a FULL port     -> pipeline
///
/// The branch id reported is the branch whose FULL port decided the answer. In the @c pipeline
/// case nothing gates the scan, so no branch is named here and the caller substitutes the first
/// branch of the traversal -- note a branch is only this scan's gate if the scan's own data
/// passes through its FULL port; a branch with a FULL port on some *other* side does not gate it.
std::pair<scheduling_mode, std::size_t> classify_scan(const pf_dag& dag, pipeline_ptr scan_pipe)
{
  bool at_first_branch = true;
  std::unordered_set<pipeline_ptr> seen;

  pipeline_ptr cur = scan_pipe;
  while (cur != nullptr && seen.insert(cur).second) {
    const auto& outs = dag.out(cur);
    if (outs.empty()) { break; }  // reached the plan's final operator
    const auto& edge = outs.front();

    if (dag.is_branch(edge.other)) {
      if (edge.barrier == op::MemoryBarrierType::FULL) {
        auto const branch_id =
          edge.consumer_op != nullptr ? edge.consumer_op->get_operator_id() : 0;
        return {at_first_branch ? scheduling_mode::barrier_all : scheduling_mode::barrier_serial,
                branch_id};
      }
      at_first_branch = false;  // passed a branch without a FULL gate; keep looking downstream
    }
    cur = edge.other;
  }
  return {scheduling_mode::pipeline, 0};
}

}  // namespace

std::shared_ptr<const query_index> query_index::build_index(const query& q,
                                                            build_index_options options)
{
  return build_index(q.get_pipelines(), options);
}

std::shared_ptr<const query_index> query_index::build_index(
  const duckdb::vector<duckdb::shared_ptr<pipeline::sirius_pipeline>>& pipelines,
  build_index_options options)
{
  // barrier_order and build_probe both honor barriers when cutting; only pipeline_order ignores
  // them. build_probe additionally rewrites hash-join probe edges to PIPELINE (see build_dag).
  const bool barrier_aware     = !std::holds_alternative<pipeline_order>(options.branch_order);
  const bool probe_as_pipeline = std::holds_alternative<build_probe>(options.branch_order);
  const pipeline_dag dag       = build_dag(pipelines, probe_as_pipeline);

  // Not std::make_shared: the constructor is private.
  std::shared_ptr<query_index> index(new query_index());

  index->_pipelines.reserve(pipelines.size());
  for (const auto& p : pipelines) {
    if (p != nullptr) { index->_pipelines.push_back(p.get()); }
  }

  // Emit one branch per branch head, iterating pipelines in execution order so branches come out
  // in plan order (scans first).
  for (const auto& head_sp : pipelines) {
    pipeline_ptr head = head_sp.get();
    if (head == nullptr || !is_branch_head(dag, head, barrier_aware)) { continue; }
    index->_branches.push_back(walk_branch(dag, head, barrier_aware));
  }

  // Second pass (after _branches is fully populated so the inner-vector buffers are stable):
  // build the span views and the head-operator lookup.
  index->_branch_views.reserve(index->_branches.size());
  for (std::size_t i = 0; i < index->_branches.size(); ++i) {
    const auto& chain = index->_branches[i];
    index->_branch_views.emplace_back(chain.data(), chain.size());
    if (const auto source = chain.front()->get_source()) {
      // First branch headed by this operator wins (a fan-out head owns several branches).
      index->_head_op_to_branch.try_emplace(source->get_operator_id(), i);
    }
  }

  return index;
}

query_index::branch query_index::get_consumer_pipelines_till_next_branch(
  const op::sirius_physical_operator* op) const
{
  if (op == nullptr) { return {}; }
  auto it = _head_op_to_branch.find(op->get_operator_id());
  if (it == _head_op_to_branch.end()) { return {}; }
  return _branch_views[it->second];
}

std::vector<prefetch_step> query_index::prefetching_orders(std::size_t concat_batch_bytes,
                                                           std::size_t scan_task_batch_size) const
{
  std::vector<prefetch_step> steps;
  if (_pipelines.empty()) { return steps; }

  const pf_dag dag = build_pf_dag(_pipelines);

  // A concat batch's worth of scan splits, at least one.
  auto const serial_count =
    std::max<std::size_t>(concat_batch_bytes / std::max<std::size_t>(scan_task_batch_size, 1), 1);

  std::unordered_set<pipeline_ptr> visited;
  // The first branch the walk enters -- the fallback owner for scans nothing gates. Recorded
  // during the walk rather than derived per scan: an ungated scan belongs to the traversal's
  // leading branch, which is not necessarily the last branch on that scan's own path once a
  // plan has more than one independent branch subtree.
  std::size_t first_branch_id = 0;
  bool have_first_branch      = false;

  // Upstream DFS. At a fan-in, the FULL side goes first: nothing downstream of that branch can
  // produce a task until the FULL side has run to completion, so its scans are wanted first.
  auto visit = [&](pipeline_ptr p, auto&& self) -> void {
    if (p == nullptr || !visited.insert(p).second) { return; }

    auto producers = dag.in(p);  // by value: sorted below
    if (producers.empty()) {
      if (auto* scan = scan_of(p)) {
        auto [mode, branch_id] = classify_scan(dag, p);
        // Nothing imposes a FULL barrier on this scan, so it is owned by the leading branch.
        if (mode == scheduling_mode::pipeline) { branch_id = first_branch_id; }
        auto const count = mode == scheduling_mode::barrier_all
                             ? std::numeric_limits<std::size_t>::max()
                           : mode == scheduling_mode::barrier_serial ? serial_count
                                                                     : 1;
        steps.push_back({scan, branch_id, mode, count});
      }
      return;
    }

    if (!have_first_branch && producers.size() > 1) {
      auto* branch_op   = producers.front().consumer_op;
      first_branch_id   = branch_op != nullptr ? branch_op->get_operator_id() : 0;
      have_first_branch = true;
    }

    // FULL first; ties (neither FULL, or both) broken by the lower pipeline id so the walk is
    // deterministic rather than dependent on port registration order.
    std::stable_sort(producers.begin(), producers.end(), [](const pf_edge& a, const pf_edge& b) {
      bool const a_full = a.barrier == op::MemoryBarrierType::FULL;
      bool const b_full = b.barrier == op::MemoryBarrierType::FULL;
      if (a_full != b_full) { return a_full; }
      return a.other->get_pipeline_id() < b.other->get_pipeline_id();
    });

    for (const auto& edge : producers) {
      self(edge.other, self);
    }
  };

  // Roots: pipelines nothing consumes. Walking from each covers every reachable pipeline; the
  // visited set makes a shared subtree emit its scans once, at its first (highest-priority)
  // encounter.
  for (pipeline_ptr p : _pipelines) {
    if (dag.out(p).empty()) { visit(p, visit); }
  }
  // Defensive: a cycle (delim-join distribution edges) can leave a pipeline with no root above
  // it. Sweep anything still unvisited so no scan is silently dropped.
  for (pipeline_ptr p : _pipelines) {
    visit(p, visit);
  }

  return steps;
}

}  // namespace sirius::planner
