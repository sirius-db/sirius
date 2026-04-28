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

#include <rmm/cuda_stream_view.hpp>

#include <cucascade/data/common.hpp>
#include <cucascade/data/gpu_data_representation.hpp>
#include <cucascade/memory/memory_space.hpp>

#include <memory>

namespace sirius::data {

/**
 * @brief Sirius-side GPU->GPU converter override (MGPU-06).
 *
 * Replaces cucascade's built-in gpu_table_representation ->
 * gpu_table_representation converter to fix a cross-stream race on the
 * GPU1 -> GPU0 return leg. cucascade's body at
 * cucascade/src/data/representation_converter.cpp:173 issues
 * cudaMemcpyPeerAsync on the *caller's* stream, then immediately constructs
 * a cudf::table on the *target_stream* without inserting a cuda event
 * between the two streams. On N=2 hosts where the caller's stream lives on
 * a different device than target_stream, the unpack on target_stream can
 * dereference not-yet-landed bytes (cudaErrorIllegalAddress surfaces in
 * the next kernel launch, typically thrust::reduce_by_key inside the table
 * constructor). Pattern 2 in
 * .planning/phases/07-p2p-direct-transfer-adaptive-scan-partitioning/07-RESEARCH.md documents this
 * fix.
 *
 * This factory issues the peer copy on target_stream (same stream used to
 * allocate the destination buffer AND to construct the target cudf::table),
 * which guarantees in-order dependency without cross-stream events.
 *
 * Pre-condition: Plan 07-01's peer-access enable loop has run, OR the
 * caller has otherwise enabled CUDA driver-level peer mapping for the pair.
 * When peer access is NOT enabled, cudaMemcpyPeerAsync falls back to a
 * host-staged copy at the driver level (still correct, slower).
 *
 * Registered in sirius_extension.cpp immediately after
 * sirius::converter_registry::initialize() so the override replaces
 * cucascade's built-in before any query runs.
 */
std::unique_ptr<cucascade::idata_representation> sirius_p2p_converter_factory(
  cucascade::idata_representation& source,
  const cucascade::memory::memory_space* target_memory_space,
  rmm::cuda_stream_view stream);

}  // namespace sirius::data
