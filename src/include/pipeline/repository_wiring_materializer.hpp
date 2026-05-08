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

#include "pipeline/repository_wiring.hpp"

#include <cucascade/data/data_repository_manager.hpp>

#include <vector>

namespace sirius {
namespace pipeline {

//! Materialize a list of plan-time `repository_wiring` descriptors into runtime state.
//!
//! For each descriptor this:
//!   1. Creates a new `cucascade::shared_data_repository` in `data_repo_manager`,
//!      keyed by the destination operator's id and the descriptor's port id.
//!   2. Attaches an `op::sirius_physical_operator::port` to the destination operator
//!      (the first operator of `dest_pipeline`, or its sink if there are no operators).
//!   3. Records the destination on the source operator via `add_next_port_after_sink`.
//!
//! Special cases preserved from the previous in-engine implementation:
//!   - When the destination's type is `RIGHT_DELIM_JOIN`, an additional `FULL`-barrier
//!     port is also added to its `partition_join` sibling so the partitioned join branch
//!     can read the same repository.
//!   - When the destination's type is `LEFT_DELIM_JOIN`, this throws — a left delim join
//!     should never act as a destination/source.
//!
//! Precondition: every `source_pipeline` and `dest_pipeline` referenced in `wirings` must
//! already have its pipeline id assigned (see `sirius_pipeline::set_pipeline_id`). Port
//! insertion uses pipeline ids to keep `_ports_list` ordered.
void materialize_repository_wiring(const std::vector<repository_wiring>& wirings,
                                   cucascade::shared_data_repository_manager& data_repo_manager);

}  // namespace pipeline
}  // namespace sirius
