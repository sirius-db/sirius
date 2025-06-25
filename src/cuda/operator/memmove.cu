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

#include "cuda_helper.cuh"
#include "log/logging.hpp"
#include <chrono>
#include <cub/cub.cuh>
#include <thrust/device_vector.h>

namespace duckdb {

// The scan op for decoupled look-back
struct MemmoveScanOp
{
  __device__ __forceinline__ bool operator()(bool l, bool r) const
  {
    return l && r;
  }
};

// Initializes state for decoupled look-back
template <typename ScanTileStateT>
__global__ void scan_tile_state_init_kernel(ScanTileStateT scan_tile_state, int32_t num_tiles)
{
  scan_tile_state.InitializeStatus(num_tiles);
}

// For overlapping ranges
template <std::int32_t B, std::int32_t I>
__global__ void memmove_kernel(std::uint8_t* destination,
                               std::uint8_t* source, // Assumes 4B alignment
                               std::size_t num,
                               cub::ScanTileState<bool> scan_tile_state)
{
  static constexpr std::int32_t TILE_ITEMS = B * I;

  using block_load_t =
    cub::BlockLoad<std::uint32_t, B, I, cub::BLOCK_LOAD_STRIPED>;
  using block_load_storage_t = typename block_load_t::TempStorage;
  using block_store_t =
    cub::BlockStore<std::uint32_t, B, I, cub::BLOCK_STORE_STRIPED>;
  using block_store_storage_t = typename block_store_t::TempStorage;
  using scan_op_t             = MemmoveScanOp;
  using tile_prefix_op_t    = cub::TilePrefixCallbackOp<bool, scan_op_t, cub::ScanTileState<bool>>;
  using prefix_op_storage_t = typename tile_prefix_op_t::TempStorage;

  // Shared memory
  __shared__ block_load_storage_t load_storage;
  __shared__ prefix_op_storage_t prefix_storage;
  __shared__ block_store_storage_t store_storage;

  // Thread memory
  std::uint32_t thread_chunks[I];
  auto destination_chunks = reinterpret_cast<std::uint32_t*>(destination);
  auto source_chunks      = reinterpret_cast<std::uint32_t*>(source);
  std::size_t num_chunks  = num / sizeof(std::uint32_t);
  std::size_t offset      = static_cast<std::size_t>(blockIdx.x) * TILE_ITEMS;
  std::int32_t num_tile_chunks =
    static_cast<std::int32_t>(cub::min(num_chunks - offset, TILE_ITEMS));
  bool is_last_tile = (offset + TILE_ITEMS) >= num_chunks;

  // Load the data
  if (is_last_tile)
  {
    block_load_t{load_storage}.Load(source_chunks + offset, thread_chunks, num_tile_chunks);
  }
  else
  {
    block_load_t{load_storage}.Load(source_chunks + offset, thread_chunks);
  }

  // Do decoupled lookback
  if (blockIdx.x == 0)
  {
    if (threadIdx.x == 0)
    {
      scan_tile_state.SetInclusive(blockIdx.x, true);
    }
  }
  else
  {
    // The first warp does the look-back
    if (threadIdx.x < CUB_PTX_WARP_THREADS)
    {
      // Initialize the prefix op
      tile_prefix_op_t prefix_op(scan_tile_state, prefix_storage, scan_op_t{});

      // Do the decoupled look-back
      prefix_op(true);
    }
  }
  __syncthreads();

  // Now we can store
  if (is_last_tile)
  {
    block_store_t{store_storage}.Store(destination_chunks + offset, thread_chunks, num_tile_chunks);
  }
  else
  {
    block_store_t{store_storage}.Store(destination_chunks + offset, thread_chunks);
  }
}

// The wrapper (on L4, B = 256, I = 6 seems best)
void cudaMemmove(uint8_t* destination,
                             uint8_t* source,
                             size_t num)
{
  static constexpr int32_t TILE_ITEMS = 256 * 6;

  // Determine the number of thread blocks needed
  size_t num_chunks = num / sizeof(uint32_t);
  auto ceil_div = [](size_t a, size_t b) { return (a + b - 1) / b; };
  auto num_tiles         = static_cast<uint32_t>(ceil_div(num_chunks, TILE_ITEMS));

  // Determine if there is memory overlap
  if (destination + num < source)
  {
    SIRIUS_LOG_DEBUG("This is just cudaMemcpy");
    cudaMemcpy(destination, source, num, cudaMemcpyDeviceToDevice);
    CHECK_ERROR();
  }
  else
  {
    SIRIUS_LOG_DEBUG("This is actual memmove");
    // We need memmove
    using scan_tile_state_t = cub::ScanTileState<bool>;

    // Determine the temporary storage needed for decoupled look-back
    size_t tile_state_storage_bytes = 0;
    scan_tile_state_t::AllocationSize(num_tiles, tile_state_storage_bytes);

    /// TODO: BELOW, USE THE SIRIUS API!!! (e.g., something like the next line) ///
    // std::uint8_t* tile_state_storage = allocator.allocate(tile_state_storage_bytes);
    // thrust::device_vector<std::uint8_t> tile_state_storage_vec(tile_state_storage_bytes);
    // std::uint8_t* tile_state_storage = thrust::raw_pointer_cast(tile_state_storage_vec.data());
    GPUBufferManager* gpuBufferManager = &(GPUBufferManager::GetInstance());
    uint8_t* tile_state_storage = gpuBufferManager->customCudaMalloc<uint8_t>(tile_state_storage_bytes, 0, 0);

    // Initialize the temporary storage needed for decoupled look-back
    scan_tile_state_t scan_tile_state{};
    scan_tile_state.Init(num_tiles, tile_state_storage, tile_state_storage_bytes);
    scan_tile_state_init_kernel<<<ceil_div(num_tiles, 256), 256>>>(
      scan_tile_state,
      num_tiles);

    CHECK_ERROR();

    // Invoke the move kernel
    memmove_kernel<256, 6><<<num_tiles, 256>>>(destination, source, num, scan_tile_state);
    CHECK_ERROR();
  }
}

}