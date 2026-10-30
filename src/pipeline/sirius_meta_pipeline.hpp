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

#pragma once

#include "common/optional_ptr.hpp"
#include "op/sirius_physical_operator.hpp"
#include "pipeline/pipeline_build_context.hpp"
#include "pipeline/sirius_pipeline.hpp"

#include <memory>
#include <vector>

namespace sirius {

namespace op {
class sirius_physical_operator;
}  // namespace op

namespace pipeline {

class sirius_pipeline_build_state;
class sirius_pipeline;

//! sirius_meta_pipeline represents a set of pipelines that all have the same sink
class sirius_meta_pipeline : public std::enable_shared_from_this<sirius_meta_pipeline> {
 public:
  //! Create a sirius_meta_pipeline with the given sink
  sirius_meta_pipeline(const pipeline_build_context& ctx,
                       sirius_pipeline_build_state& state,
                       sirius::optional_ptr<op::sirius_physical_operator> sink);

 public:
  //! Get the pipeline_build_state for this sirius_meta_pipeline
  sirius_pipeline_build_state& get_state() const;
  //! Get the initial pipeline of this sirius_meta_pipeline
  std::shared_ptr<sirius_pipeline>& get_base_pipeline();
  //! Get the pipelines of this sirius_meta_pipeline
  void get_pipelines(std::vector<std::shared_ptr<sirius_pipeline>>& result, bool recursive);
  //! Get the sirius_meta_pipeline children of this sirius_meta_pipeline
  void get_meta_pipelines(std::vector<std::shared_ptr<sirius_meta_pipeline>>& result,
                          bool recursive,
                          bool skip);

 public:
  //! Build the sirius_meta_pipeline with 'op' as the first operator (excl. the shared sink)
  void build(op::sirius_physical_operator& op);
  //! Ready all the pipelines (recursively)
  void ready();

  //! Create an empty pipeline within this sirius_meta_pipeline
  sirius_pipeline& create_pipeline();
  //! Create a sirius_meta_pipeline child that 'current' depends on
  sirius_meta_pipeline& create_child_meta_pipeline(sirius_pipeline& current,
                                                   op::sirius_physical_operator& op);

 private:
  //! Plan-time build context for all MetaPipelines in the query plan
  const pipeline_build_context build_ctx;
  //! The pipeline_build_state for all MetaPipelines in the query plan
  sirius_pipeline_build_state& state;
  //! The sink of all pipelines within this sirius_meta_pipeline
  sirius::optional_ptr<op::sirius_physical_operator> sink;
  //! All pipelines with a different source, but the same sink
  std::vector<std::shared_ptr<sirius_pipeline>> pipelines;
  //! Other MetaPipelines that this sirius_meta_pipeline depends on
  std::vector<std::shared_ptr<sirius_meta_pipeline>> children;
};

}  // namespace pipeline
}  // namespace sirius
