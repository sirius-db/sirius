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
#include <cucascade/memory/memory_space.hpp>

#include <memory>

namespace sirius::data {

/**
 * @brief FIX-02 (v1.2) Sirius-side override for the
 *        host_data_representation -> gpu_table_representation conversion.
 *
 * Replaces cucascade's built-in convert_host_fast_to_gpu on paths that
 * exhibited cudaErrorInvalidValue at cuda_memcpy.cu when num_gpus == 2.
 * See .planning/phases/08-multi-gpu-sql-pipeline-fix/08-02-PROBE.md for
 * the failing-test reproduction and v1.1 bug-signature match.
 *
 * Root cause: cucascade's body (representation_converter.cpp:825-856) sets
 *   rmm::cuda_set_device_raii{ target_device_id }
 * at L837 but issues the H2D batched copy on the CALLER's stream at L849.
 * If the caller's stream is bound to a different device than target_device_id,
 * the cudaMemcpyBatchAsync call raises cudaErrorInvalidValue because the
 * stream and the current device (under the RAII guard) do not belong to the
 * same CUDA context.
 *
 * The Sirius override mirrors Pattern 2 shape from sirius_p2p_converter.cpp:
 *   1. Sync caller's stream so any upstream work is flushed (safe: caller
 *      stream may live on any device; sync is a no-op at driver level when
 *      there is no outstanding work).
 *   2. Acquire a target-bound stream via target_memory_space->acquire_stream()
 *      — a cuda_stream_view whose device matches target_device_id.
 *   3. Enter rmm::cuda_set_device_raii for target_device_id before any
 *      allocation or H2D copy (matches cucascade's device_guard).
 *   4. Reconstruct each column onto the target device using target_stream
 *      throughout (never the caller's stream).
 *   5. Synchronize target_stream before returning so the caller observes
 *      finished GPU data.
 *
 * The reconstruct_column / BatchCopyAccumulator helpers in cucascade are
 * file-private (not exposed in cucascade/include/). This override reimplements
 * the traversal inline using only cucascade's PUBLIC host_data_representation
 * + column_metadata + fixed_multiple_blocks_allocation API plus cudf column
 * factories.
 */
std::unique_ptr<cucascade::idata_representation> sirius_host_fast_to_gpu_factory(
  cucascade::idata_representation& source,
  const cucascade::memory::memory_space* target_memory_space,
  rmm::cuda_stream_view stream);

}  // namespace sirius::data
