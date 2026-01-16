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

#include "pipeline/sirius_meta_pipeline.hpp"

#include "duckdb/common/enums/physical_operator_type.hpp"
#include "gpu_executor.hpp"
#include "log/logging.hpp"

namespace sirius {
namespace pipeline {

sirius_meta_pipeline::sirius_meta_pipeline(
  duckdb::GPUExecutor& executor_p,
  sirius_pipeline_build_state& state_p,
  duckdb::optional_ptr<op::sirius_physical_operator> sink_p)
  : executor(executor_p), state(state_p), sink(sink_p), recursive_cte(false), next_batch_index(0)
{
  create_pipeline();
}

duckdb::GPUExecutor& sirius_meta_pipeline::get_executor() const { return executor; }

sirius_pipeline_build_state& sirius_meta_pipeline::get_state() const { return state; }

duckdb::optional_ptr<op::sirius_physical_operator> sirius_meta_pipeline::get_sink() const
{
  return sink;
}

duckdb::optional_ptr<sirius_pipeline> sirius_meta_pipeline::get_parent() const { return parent; }

duckdb::shared_ptr<sirius_pipeline>& sirius_meta_pipeline::get_base_pipeline()
{
  return pipelines[0];
}

void sirius_meta_pipeline::get_pipelines(
  duckdb::vector<duckdb::shared_ptr<sirius_pipeline>>& result, bool recursive)
{
  result.insert(result.end(), pipelines.begin(), pipelines.end());
  if (recursive) {
    for (auto& child : children) {
      child->get_pipelines(result, true);
    }
  }
}

void sirius_meta_pipeline::get_meta_pipelines(
  duckdb::vector<duckdb::shared_ptr<sirius_meta_pipeline>>& result, bool recursive, bool skip)
{
  if (!skip) {
    result.push_back(duckdb::enable_shared_from_this<sirius_meta_pipeline>::shared_from_this());
  }
  if (recursive) {
    for (auto& child : children) {
      child->get_meta_pipelines(result, true, false);
    }
  }
}

sirius_meta_pipeline& sirius_meta_pipeline::get_last_child()
{
  if (children.empty()) { return *this; }
  duckdb::reference<const duckdb::vector<duckdb::shared_ptr<sirius_meta_pipeline>>>
    current_children = children;
  while (!current_children.get().back()->children.empty()) {
    current_children = current_children.get().back()->children;
  }
  return *current_children.get().back();
}

duckdb::optional_ptr<const duckdb::vector<duckdb::reference<sirius_pipeline>>>
sirius_meta_pipeline::get_dependencies(sirius_pipeline& dependent) const
{
  auto it = dependencies.find(dependent);
  if (it == dependencies.end()) {
    return nullptr;
  } else {
    return &it->second;
  }
}

bool sirius_meta_pipeline::has_recursive_cte() const { return recursive_cte; }

void sirius_meta_pipeline::set_recursive_cte() { recursive_cte = true; }

void sirius_meta_pipeline::assign_next_batch_index(sirius_pipeline& pipeline)
{
  pipeline.base_batch_index = next_batch_index++ * sirius_pipeline_build_state::BATCH_INCREMENT;
}

void sirius_meta_pipeline::build(op::sirius_physical_operator& op)
{
  D_ASSERT(pipelines.size() == 1);
  D_ASSERT(children.empty());
  // SIRIUS_LOG_DEBUG("op.type = {}", duckdb::PhysicalOperatorToString(op.type));
  op.build_pipelines(*pipelines.back(), *this);
}

void sirius_meta_pipeline::ready()
{
  for (auto& pipeline : pipelines) {
    pipeline->is_ready();
  }
  for (auto& child : children) {
    child->ready();
  }
}

sirius_meta_pipeline& sirius_meta_pipeline::create_child_meta_pipeline(
  sirius_pipeline& current, op::sirius_physical_operator& op)
{
  children.push_back(duckdb::make_shared_ptr<sirius_meta_pipeline>(executor, state, &op));
  auto child_meta_pipeline = children.back().get();
  // store the parent
  child_meta_pipeline->parent = &current;
  // child sirius_meta_pipeline must finish completely before this sirius_meta_pipeline can start
  current.add_dependency(child_meta_pipeline->get_base_pipeline());
  // child meta pipeline is part of the recursive CTE too
  child_meta_pipeline->recursive_cte = recursive_cte;
  return *child_meta_pipeline;
}

sirius_pipeline& sirius_meta_pipeline::create_pipeline()
{
  pipelines.emplace_back(duckdb::make_shared_ptr<sirius_pipeline>(executor));
  state.set_pipeline_sink(*pipelines.back(), sink, next_batch_index++);
  return *pipelines.back();
}

void sirius_meta_pipeline::add_dependencies_from(sirius_pipeline& dependent,
                                                 sirius_pipeline& start,
                                                 bool including)
{
  // find 'start'
  auto it = pipelines.begin();
  for (; !duckdb::RefersToSameObject(**it, start); it++) {}

  if (!including) { it++; }

  // collect pipelines that were created from then
  duckdb::vector<duckdb::reference<pipeline::sirius_pipeline>> created_pipelines;
  for (; it != pipelines.end(); it++) {
    if (duckdb::RefersToSameObject(**it, dependent)) {
      // cannot depend on itself
      continue;
    }
    created_pipelines.push_back(**it);
  }

  // add them to the dependencies
  auto& deps = dependencies[dependent];
  deps.insert(deps.begin(), created_pipelines.begin(), created_pipelines.end());
}

void sirius_meta_pipeline::add_recursive_dependencies(
  const duckdb::vector<duckdb::shared_ptr<sirius_pipeline>>& new_dependencies,
  const sirius_meta_pipeline& last_child)
{
  if (recursive_cte) {
    return;  // let's not burn our fingers on this for now
  }

  duckdb::vector<duckdb::shared_ptr<sirius_meta_pipeline>> child_meta_pipelines;
  this->get_meta_pipelines(child_meta_pipelines, true, false);

  // find the meta pipeline that has the same sink as 'pipeline'
  auto it = child_meta_pipelines.begin();
  for (; !duckdb::RefersToSameObject(last_child, **it); it++) {}
  D_ASSERT(it != child_meta_pipelines.end());

  // skip over it
  it++;

  // we try to limit the performance impact of these dependencies on smaller workloads,
  // by only adding the dependencies if the source operator can likely keep all threads busy
  // const auto thread_count =
  // duckdb::NumericCast<idx_t>(duckdb::TaskScheduler::GetScheduler(executor.context).NumberOfThreads());
  // for (; it != child_meta_pipelines.end(); it++) {
  // 	for (auto &pipeline : it->get()->pipelines) {
  // 		if (!PipelineExceedsThreadCount(*pipeline, thread_count)) {
  // 			continue;
  // 		}
  // 		auto &pipeline_deps = pipeline_dependencies[*pipeline];
  // 		for (auto &new_dependency : new_dependencies) {
  // 			if (!PipelineExceedsThreadCount(*new_dependency, thread_count)) {
  // 				continue;
  // 			}
  // 			pipeline_deps.push_back(*new_dependency);
  // 		}
  // 	}
  // }
}

void sirius_meta_pipeline::add_finish_event(sirius_pipeline& pipeline)
{
  D_ASSERT(finish_pipelines.find(pipeline) == finish_pipelines.end());
  finish_pipelines.insert(pipeline);

  // add all pipelines that were added since 'pipeline' was added (including 'pipeline') to the
  // finish group
  auto it = pipelines.begin();
  for (; !duckdb::RefersToSameObject(**it, pipeline); it++) {}
  it++;
  for (; it != pipelines.end(); it++) {
    finish_map.emplace(**it, pipeline);
  }
}

bool sirius_meta_pipeline::has_finish_event(sirius_pipeline& pipeline) const
{
  return finish_pipelines.find(pipeline) != finish_pipelines.end();
}

duckdb::optional_ptr<sirius_pipeline> sirius_meta_pipeline::get_finish_group(
  sirius_pipeline& pipeline) const
{
  auto it = finish_map.find(pipeline);
  return it == finish_map.end() ? nullptr : &it->second;
}

sirius_pipeline& sirius_meta_pipeline::create_union_pipeline(sirius_pipeline& current,
                                                             bool order_matters)
{
  // create the union pipeline (batch index 0, should be set correctly afterwards)
  auto& union_pipeline = create_pipeline();
  state.set_pipeline_operators(union_pipeline, state.get_pipeline_operators(current));
  state.set_pipeline_sink(union_pipeline, sink, 0);

  // 'union_pipeline' inherits ALL dependencies of 'current' (within this sirius_meta_pipeline, and
  // across MetaPipelines)
  union_pipeline.dependencies = current.dependencies;
  auto current_deps           = get_dependencies(current);
  if (current_deps) { dependencies[union_pipeline] = *current_deps; }

  if (order_matters) {
    // if we need to preserve order, or if the sink is not parallel, we set a dependency
    dependencies[union_pipeline].push_back(current);
  }

  return union_pipeline;
}

void sirius_meta_pipeline::create_child_pipeline(sirius_pipeline& current,
                                                 op::sirius_physical_operator& op,
                                                 sirius_pipeline& last_pipeline)
{
  // rule 2: 'current' must be fully built (down to the source) before creating the child pipeline
  D_ASSERT(current.source);

  // create the child pipeline (same batch index)
  pipelines.emplace_back(state.create_child_pipeline(executor, current, op));
  auto& child_pipeline            = *pipelines.back();
  child_pipeline.base_batch_index = current.base_batch_index;

  // child pipeline has a dependency (within this sirius_meta_pipeline on all pipelines that were
  // scheduled between 'current' and now (including 'current') - set them up
  dependencies[child_pipeline].push_back(current);
  add_dependencies_from(child_pipeline, last_pipeline, false);
  D_ASSERT(!get_dependencies(child_pipeline)->empty());
}

}  // namespace pipeline
}  // namespace sirius
