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

#include <string_view>

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

}  // namespace pipeline
}  // namespace sirius
