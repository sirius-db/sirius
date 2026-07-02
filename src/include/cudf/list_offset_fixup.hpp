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

#include <cucascade/data/data_batch.hpp>
#include <cucascade/memory/memory_space.hpp>

namespace sirius {

/**
 * @brief Rewrite every LIST column in a freshly H2D converted batch so its offsets
 *        child is INT32, as cuDF requires.
 *
 * cuCascade's H2D reconstruction promotes the offsets child of both STRING and LIST columns
 * from INT32 to INT64. For STRING that matches cuDF's large-string convention, but cuDF LIST
 * columns use INT32 (cudf::size_type) offsets. A LIST whose offsets child is INT64 is misread
 * by every list algorithm, collapsing the sublists.
 *
 * Workaround for the upstream cuCascade bug NVIDIA/cuCascade#147; remove this once the LIST
 * offsets are no longer promoted to INT64 during reconstruction.
 *
 * This is the single, central place where that promotion is reversed: it walks the GPU table
 * held by @p batch and rebuilds each LIST column (recursing into nested lists) with INT32
 * offsets, moving the values child and null mask (no payload copy) and reallocating only the
 * small offsets buffer via cudf::cast. The LIST column is assembled with the cudf::column
 * constructor rather than make_lists_column, so no purge_nonempty_nulls runs and the fixed
 * stride is preserved even for null array rows.
 *
 * No-op when the batch holds no LIST columns or the offsets are already INT32 (e.g. a GPU table
 * produced by a cudf operation rather than a host->GPU conversion).
 *
 * Must be called while holding a mutable lock, immediately after converting the batch to a
 * gpu_table_representation.
 *
 * @param batch     Mutable accessor to the just-converted GPU batch.
 * @param gpu_space The GPU memory space the batch now resides in.
 * @param stream    CUDA stream used for the offsets cast.
 */
void normalize_gpu_list_offsets(cucascade::mutable_data_batch& batch,
                                const cucascade::memory::memory_space* gpu_space,
                                rmm::cuda_stream_view stream);

}  // namespace sirius
