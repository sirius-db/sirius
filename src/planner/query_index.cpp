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

#include <unordered_set>

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

/// Build the pipeline DAG from the sink operators' next-ports. Each next-port names the downstream
/// consumer operator and the port it pushes into; that port carries the barrier and identifies the
/// consumer pipeline via the operator's owning pipeline.
pipeline_dag build_dag(
  const duckdb::vector<duckdb::shared_ptr<pipeline::sirius_pipeline>>& pipelines)
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
  const bool barrier_aware = std::holds_alternative<barrier_order>(options.branch_order);
  const pipeline_dag dag   = build_dag(pipelines);

  // Not std::make_shared: the constructor is private.
  std::shared_ptr<query_index> index(new query_index());

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

}  // namespace sirius::planner
