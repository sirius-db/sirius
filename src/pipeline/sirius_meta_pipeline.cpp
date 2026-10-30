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

namespace sirius {
namespace pipeline {

sirius_meta_pipeline::sirius_meta_pipeline(
  const pipeline_build_context& ctx,
  sirius_pipeline_build_state& state_p,
  sirius::optional_ptr<op::sirius_physical_operator> sink_p)
  : build_ctx(ctx), state(state_p), sink(sink_p)
{
  create_pipeline();
}

sirius_pipeline_build_state& sirius_meta_pipeline::get_state() const { return state; }

std::shared_ptr<sirius_pipeline>& sirius_meta_pipeline::get_base_pipeline() { return pipelines[0]; }

void sirius_meta_pipeline::get_pipelines(std::vector<std::shared_ptr<sirius_pipeline>>& result,
                                         bool recursive)
{
  result.insert(result.end(), pipelines.begin(), pipelines.end());
  if (recursive) {
    for (auto& child : children) {
      child->get_pipelines(result, true);
    }
  }
}

void sirius_meta_pipeline::get_meta_pipelines(
  std::vector<std::shared_ptr<sirius_meta_pipeline>>& result, bool recursive, bool skip)
{
  if (!skip) { result.push_back(shared_from_this()); }
  if (recursive) {
    for (auto& child : children) {
      child->get_meta_pipelines(result, true, false);
    }
  }
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
  children.push_back(std::make_shared<sirius_meta_pipeline>(build_ctx, state, &op));
  auto child_meta_pipeline = children.back().get();
  // child sirius_meta_pipeline must finish completely before this sirius_meta_pipeline can start
  current.add_dependency(child_meta_pipeline->get_base_pipeline());
  return *child_meta_pipeline;
}

sirius_pipeline& sirius_meta_pipeline::create_pipeline()
{
  pipelines.emplace_back(std::make_shared<sirius_pipeline>(build_ctx));
  state.set_pipeline_sink(*pipelines.back(), sink);
  if (sink) {
    // Pre-populate operators with [sink] so it lands at operators.back() after `is_ready`
    // reverses; intermediates/sources are appended as build_pipelines recurses.
    state.add_pipeline_operator(*pipelines.back(), *sink);
  }
  return *pipelines.back();
}

}  // namespace pipeline
}  // namespace sirius
