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

#include "duckdb/common/reference_map.hpp"
#include "op/sirius_physical_operator.hpp"
#include "pipeline/sirius_pipeline.hpp"

namespace sirius {

class sirius_engine;

namespace op {
class sirius_physical_operator;
}  // namespace op

namespace pipeline {

enum class MetaPipelineType : uint8_t {
  REGULAR    = 0,  //! The shared sink is regular
  JOIN_BUILD = 1   //! The shared sink is a join build
};

class sirius_pipeline_build_state;
class sirius_pipeline;

//! sirius_meta_pipeline represents a set of pipelines that all have the same sink
class sirius_meta_pipeline : public duckdb::enable_shared_from_this<sirius_meta_pipeline> {
  //! We follow these rules when building:
  //! 1. For joins, build out the blocking side before going down the probe side
  //!     - The current streaming pipeline will have a dependency on it (dependency across
  //!     MetaPipelines)
  //!     - Unions of this streaming pipeline will automatically inherit this dependency
  //! 2. Build child pipelines last (e.g., Hash Join becomes source after probe is done: scan HT for
  //! FULL OUTER JOIN)
  //!     - 'last' means after building out all other pipelines associated with this operator
  //!     - The child pipeline automatically has dependencies (within this sirius_meta_pipeline) on:
  //!         * The 'current' streaming pipeline
  //!         * And all pipelines that were added to the sirius_meta_pipeline after 'current'
 public:
  //! Create a sirius_meta_pipeline with the given sink
  sirius_meta_pipeline(sirius_engine& engine,
                       sirius_pipeline_build_state& state,
                       duckdb::optional_ptr<op::sirius_physical_operator> sink);

 public:
  //! Get the gpu_executor for this sirius_meta_pipeline
  sirius_engine& get_engine() const;
  //! Get the pipeline_build_state for this sirius_meta_pipeline
  sirius_pipeline_build_state& get_state() const;
  //! Get the sink operator for this sirius_meta_pipeline
  duckdb::optional_ptr<op::sirius_physical_operator> get_sink() const;
  //! Get the parent pipeline
  duckdb::optional_ptr<sirius_pipeline> get_parent() const;

  //! Get the initial pipeline of this sirius_meta_pipeline
  duckdb::shared_ptr<sirius_pipeline>& get_base_pipeline();
  //! Get the pipelines of this sirius_meta_pipeline
  void get_pipelines(duckdb::vector<duckdb::shared_ptr<sirius_pipeline>>& result, bool recursive);
  //! Get the sirius_meta_pipeline children of this sirius_meta_pipeline
  void get_meta_pipelines(duckdb::vector<duckdb::shared_ptr<sirius_meta_pipeline>>& result,
                          bool recursive,
                          bool skip);
  //! Recursively gets the last child added
  sirius_meta_pipeline& get_last_child();
  //! Get the dependencies (within this sirius_meta_pipeline) of the given Pipeline
  duckdb::optional_ptr<const duckdb::vector<duckdb::reference<sirius_pipeline>>> get_dependencies(
    sirius_pipeline& dependent) const;
  //! Whether the sink of this pipeline is a join build
  MetaPipelineType Type() const;
  //! Whether this sirius_meta_pipeline has a recursive CTE
  bool has_recursive_cte() const;
  //! Set the flag that this sirius_meta_pipeline is a recursive CTE pipeline
  void set_recursive_cte();
  //! Assign a batch index to the given pipeline
  void assign_next_batch_index(sirius_pipeline& pipeline);
  //! Let 'dependant' depend on all pipeline that were created since 'start',
  //! where 'including' determines whether 'start' is added to the dependencies
  void add_dependencies_from(sirius_pipeline& dependent, sirius_pipeline& start, bool including);
  //! Recursively makes all children of this MetaPipeline depend on the given Pipeline
  void add_recursive_dependencies(
    const duckdb::vector<duckdb::shared_ptr<sirius_pipeline>>& new_dependencies,
    const sirius_meta_pipeline& last_child);
  //! Make sure that the given pipeline has its own PipelineFinishEvent (e.g., for IEJoin - double
  //! Finalize)
  void add_finish_event(sirius_pipeline& pipeline);
  //! Whether the pipeline needs its own PipelineFinishEvent
  bool has_finish_event(sirius_pipeline& pipeline) const;
  //! Whether this pipeline is part of a PipelineFinishEvent
  duckdb::optional_ptr<sirius_pipeline> get_finish_group(sirius_pipeline& pipeline) const;

  void build_sirius_pipelines(op::sirius_physical_operator& node, sirius_pipeline& current);

 public:
  //! Build the sirius_meta_pipeline with 'op' as the first operator (excl. the shared sink)
  void build(op::sirius_physical_operator& op);
  //! Ready all the pipelines (recursively)
  void ready();

  //! Create an empty pipeline within this sirius_meta_pipeline
  sirius_pipeline& create_pipeline();
  //! Create a union pipeline (clone of 'current')
  sirius_pipeline& create_union_pipeline(sirius_pipeline& current, bool order_matters);
  //! Create a child pipeline op 'current' starting at 'op',
  //! where 'last_pipeline' is the last pipeline added before building out 'current'
  void create_child_pipeline(sirius_pipeline& current,
                             op::sirius_physical_operator& op,
                             sirius_pipeline& last_pipeline);
  //! Create a sirius_meta_pipeline child that 'current' depends on
  sirius_meta_pipeline& create_child_meta_pipeline(sirius_pipeline& current,
                                                   op::sirius_physical_operator& op);

 private:
  //! The executor for all MetaPipelines in the query plan
  sirius_engine& engine;
  //! The pipeline_build_state for all MetaPipelines in the query plan
  sirius_pipeline_build_state& state;
  //! Parent pipeline (optional)
  duckdb::optional_ptr<sirius_pipeline> parent;
  //! The sink of all pipelines within this sirius_meta_pipeline
  duckdb::optional_ptr<op::sirius_physical_operator> sink;
  //! The type of this MetaPipeline (regular, join build)
  MetaPipelineType type;
  //! Whether this sirius_meta_pipeline is a the recursive pipeline of a recursive CTE
  bool recursive_cte;
  //! All pipelines with a different source, but the same sink
  duckdb::vector<duckdb::shared_ptr<sirius_pipeline>> pipelines;
  //! Dependencies within this sirius_meta_pipeline
  duckdb::reference_map_t<sirius_pipeline, duckdb::vector<duckdb::reference<sirius_pipeline>>>
    dependencies;
  //! Other MetaPipelines that this sirius_meta_pipeline depends on
  duckdb::vector<duckdb::shared_ptr<sirius_meta_pipeline>> children;
  //! Next batch index
  duckdb::idx_t next_batch_index;
  //! Pipelines (other than the base pipeline) that need their own PipelineFinishEvent (e.g., for
  //! IEJoin)
  duckdb::reference_set_t<sirius_pipeline> finish_pipelines;
  //! Mapping from pipeline (e.g., child or union) to finish pipeline
  duckdb::reference_map_t<sirius_pipeline, sirius_pipeline&> finish_map;
};

}  // namespace pipeline
}  // namespace sirius
