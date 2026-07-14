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

namespace sirius::planner {

namespace {

using pipeline_ptr = query_index::pipeline_ptr;

/// Downstream consumer pipelines (pipelines this one feeds).
std::vector<pipeline_ptr> consumers_of(pipeline_ptr p) { return p->get_parents(); }

/// Number of upstream producer pipelines (pipelines that feed this one).
std::size_t producer_count(pipeline_ptr p) { return p->dependencies.size(); }

/// A pipeline is "internal" (mid-branch) only when it is a simple pass-through: exactly one
/// producer and exactly one consumer. Everything else -- scans (0 producers), joins/merges
/// (>1 producer), forks (>1 consumer), and terminal results (0 consumers) -- is a branch point.
bool is_internal(pipeline_ptr p) { return producer_count(p) == 1 && p->get_parents().size() == 1; }

}  // namespace

std::shared_ptr<const query_index> query_index::build_index(const query& q,
                                                            build_index_options options)
{
  return build_index(q.get_pipelines(), options);
}

std::shared_ptr<const query_index> query_index::build_index(
  const duckdb::vector<duckdb::shared_ptr<pipeline::sirius_pipeline>>& pipelines,
  build_index_options /*options*/)
{
  // Not std::make_shared: the constructor is private.
  std::shared_ptr<query_index> index(new query_index());

  // Enumerate branches by walking, from each branch-point pipeline, along every consumer edge
  // through internal pass-through pipelines until the next branch point. Iterating the query's
  // pipelines in execution order makes the emitted branch order plan order (scans first).
  for (const auto& head_sp : pipelines) {
    pipeline_ptr head = head_sp.get();
    if (head == nullptr || is_internal(head)) { continue; }  // only start at branch points

    for (pipeline_ptr next : consumers_of(head)) {
      std::vector<pipeline_ptr> chain{head};
      pipeline_ptr cur = next;
      while (cur != nullptr && is_internal(cur)) {
        chain.push_back(cur);
        cur = cur->get_parents().front();  // internal => exactly one consumer
      }
      if (cur != nullptr) { chain.push_back(cur); }  // the next branch point (branch tail)
      index->_branches.push_back(std::move(chain));
    }
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
