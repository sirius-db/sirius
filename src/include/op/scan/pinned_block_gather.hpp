/*
 * Copyright 2026, Sirius Contributors.
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

#include <cudf/types.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <cstddef>

namespace sirius::op::scan {

/**
 * @brief Gather fixed-width elements straight out of block-fragmented pinned host memory into a
 * device buffer, by row index.
 *
 * The pinned-cache tier stores column buffers in fixed-size host blocks
 * (`fixed_size_host_memory_resource::multiple_blocks_allocation`) that are
 * `cudaHostRegister(..., Mapped)`-registered, so a kernel can dereference them directly over
 * UVA / coherent links. A column's bytes are contiguous only in the allocation's *virtual*
 * offset space; this kernel performs the block translation
 * (`off / block_size`, `off % block_size`) per element.
 *
 * This is the serve-side alternative to bulk H2D + device-side `apply_boolean_mask`: only the
 * surviving rows' bytes ever cross the link.
 *
 * @param device_block_ptrs Device array of the allocation's host block base pointers.
 * @param block_size        Bytes per host block (power of two not required).
 * @param data_offset       Virtual offset of the column's element 0 inside the allocation.
 * @param element_size      Element width in bytes: 1, 2, 4, 8, or 16.
 * @param device_row_indices Device array of ascending-or-not row indices to gather.
 * @param num_rows          Number of indices / output elements.
 * @param device_out        Device output buffer of at least num_rows * element_size bytes.
 * @throws std::invalid_argument for an unsupported element_size.
 */
void gather_fixed_width_from_host_blocks(std::byte const* const* device_block_ptrs,
                                         std::size_t block_size,
                                         std::size_t data_offset,
                                         std::size_t element_size,
                                         cudf::size_type const* device_row_indices,
                                         cudf::size_type num_rows,
                                         std::byte* device_out,
                                         rmm::cuda_stream_view stream);

}  // namespace sirius::op::scan
