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

#include "duckdb/common/common.hpp"
#include "op/sirius_physical_operator.hpp"

#include <cucascade/data/data_repository_manager.hpp>

#include <string_view>
#include <vector>

namespace sirius {
namespace pipeline {

class sirius_pipeline;

//! Plan-time description of a single sink->source connection between pipelines.
//!
//! Produced by `sirius_pipeline_converter::compute_repository_wiring()` and consumed by
//! `materialize_repository_wiring()` at runtime, which creates the backing
//! `cucascade::shared_data_repository` and attaches `sirius_physical_operator::port`s.
//!
//! The destination operator (the first operator of `dest_pipeline`, or its sink if the
//! pipeline has no operators) is resolved at materialization time, so the descriptor
//! itself stays purely topological.
struct repository_wiring {
  //! Logical channel name on the destination operator (e.g. "default", "build", "scan").
  std::string_view port_id;
  //! Memory barrier semantics between source and destination.
  op::MemoryBarrierType barrier_type;
  //! Operator that emits via `add_next_port_after_sink`. For ordinary pipeline sinks
  //! this is `source_pipeline->get_sink().get()`; for delim joins and build CONCATs it
  //! is a sub-operator of the sink.
  op::sirius_physical_operator* source_op;
  //! Pipeline that produces the data.
  duckdb::shared_ptr<sirius_pipeline> source_pipeline;
  //! Pipeline that consumes the data.
  duckdb::shared_ptr<sirius_pipeline> dest_pipeline;
};

//! Materialize a list of plan-time `repository_wiring` descriptors into runtime state.
//!
//! For each descriptor this:
//!   1. Creates a new `cucascade::shared_data_repository` in `data_repo_manager`,
//!      keyed by the destination operator's id and the descriptor's port id.
//!   2. Attaches an `op::sirius_physical_operator::port` to the destination operator
//!      (the first operator of `dest_pipeline`, or its sink if there are no operators).
//!   3. Records the destination on the source operator via `add_next_port_after_sink`.
//!
//! Precondition: every `source_pipeline` and `dest_pipeline` referenced in `wirings` must
//! already have its pipeline id assigned (see `sirius_pipeline::set_pipeline_id`). Port
//! insertion uses pipeline ids to keep `_ports_list` ordered.
void materialize_repository_wiring(const std::vector<repository_wiring>& wirings,
                                   cucascade::shared_data_repository_manager& data_repo_manager);

/**
 * @brief Stamp every operator in @p pipelines with a dense, 0-based id for this query.
 *
 * Operators are constructed with `sirius_physical_operator::invalid_operator_id` so that plan
 * construction carries no cross-query state and two queries can be planned concurrently. This
 * pass runs once, after the pipeline converter has produced the final pipeline set and before
 * anything reads an operator id — `materialize_repository_wiring` above is the first consumer,
 * since it keys each repository on the destination operator's id.
 *
 * Ids are assigned in pipeline-scheduling order and, within a pipeline, in
 * source -> operators -> sink order. Each operator is numbered on first visit, so an operator
 * appearing in several pipelines keeps one stable id. The resulting ids are `0..N-1`, no gaps.
 *
 * The pipeline set — not the plan tree — is the traversal root on purpose:
 * `sirius_physical_delim_join` owns its `join` and `distinct_root` subtrees outside `children`
 * and does not override `get_children()`, so a tree walk would miss them. Every operator that
 * executes belongs to a pipeline.
 *
 * @param pipelines The query's pipelines, in scheduling order.
 * @return The number of operators numbered (i.e. one past the highest id assigned).
 */
size_t assign_operator_ids(const duckdb::vector<duckdb::shared_ptr<sirius_pipeline>>& pipelines);

}  // namespace pipeline
}  // namespace sirius
