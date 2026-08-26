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

// Gathering by GLOBAL row id, across a pinned table's batches, in one pass.
//
// cudf::gather works on one column, so a selection spanning several pinned
// batches would otherwise have to be canonicalized first — sorted, split by
// batch, gathered per batch and reassembled. For an uncompressed origin that
// whole apparatus buys nothing: a gather needs neither sorted nor unique ids,
// and produces exactly as many rows as it is given, in the order it is given
// them. All that is really missing is which batch an id belongs to, and that is
// a binary search over B row starts.
//
// So this reads the ids as they came, finds each one's batch, and copies its
// element. Duplicates and disorder are ordinary gather semantics; the output is
// in the caller's order with no restoring pass.
//
// Fixed-width columns only — a variable-width one has no element size to copy,
// and its offsets would need reconstructing. Those take the canonical path.

#include "late_mat/multi_source_gather.hpp"

#include <cudf/utilities/bit.hpp>

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace sirius::late_mat {

namespace {

constexpr int kBlock = 256;

inline void throw_on_cuda(cudaError_t err, char const* what)
{
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string("multi_source_gather: ") + what + ": " +
                             cudaGetErrorString(err));
  }
}

/// The batch owning `id`: the last start not greater than it. B is a handful,
/// so the starts stay cache-resident and the search is a few steps.
__device__ inline int find_batch(std::int64_t const* __restrict__ row_start,
                                 int num_batches,
                                 std::int64_t id)
{
  int lo = 0;
  int hi = num_batches;  // searching [0, B) over starts[0..B)
  while (lo + 1 < hi) {
    int const mid = lo + (hi - lo) / 2;
    if (row_start[mid] <= id) {
      lo = mid;
    } else {
      hi = mid;
    }
  }
  return lo;
}

template <typename T>
__global__ void gather_fixed_kernel(void const* const* __restrict__ bases,
                                    std::int64_t const* __restrict__ row_start,
                                    int num_batches,
                                    std::uint64_t const* __restrict__ ids,
                                    std::int64_t count,
                                    T* __restrict__ out,
                                    cudf::bitmask_type const* const* __restrict__ masks,
                                    cudf::bitmask_type* __restrict__ out_mask)
{
  auto const stride = static_cast<std::int64_t>(gridDim.x) * blockDim.x;
  for (std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < count;
       i += stride) {
    auto const id    = static_cast<std::int64_t>(ids[i]);
    int const b      = find_batch(row_start, num_batches, id);
    auto const local = static_cast<cudf::size_type>(id - row_start[b]);
    auto const* src  = static_cast<T const*>(bases[b]);
    out[i]           = src[local];
    if (out_mask != nullptr) {
      auto const* src_mask = masks[b];
      bool const valid     = src_mask == nullptr || cudf::bit_is_set(src_mask, local);
      if (valid) {
        cudf::set_bit(out_mask, static_cast<cudf::size_type>(i));
      } else {
        cudf::clear_bit(out_mask, static_cast<cudf::size_type>(i));
      }
    }
  }
}

__global__ void gather_validity_kernel(cudf::bitmask_type const* __restrict__ source_mask,
                                       std::int32_t const* __restrict__ map,
                                       std::int64_t count,
                                       cudf::bitmask_type* __restrict__ out_mask)
{
  auto const stride = static_cast<std::int64_t>(gridDim.x) * blockDim.x;
  for (std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < count;
       i += stride) {
    if (cudf::bit_is_set(source_mask, static_cast<cudf::size_type>(map[i]))) {
      cudf::set_bit(out_mask, static_cast<cudf::size_type>(i));
    } else {
      cudf::clear_bit(out_mask, static_cast<cudf::size_type>(i));
    }
  }
}

}  // namespace

void gather_validity_bits(std::uint32_t const* source_mask,
                          std::int32_t const* map,
                          std::int64_t count,
                          std::uint32_t* out_mask,
                          rmm::cuda_stream_view stream)
{
  if (count == 0) { return; }
  if (source_mask == nullptr || map == nullptr || out_mask == nullptr) {
    throw std::runtime_error("multi_source_gather: validity gather over unbound buffers");
  }

  std::int64_t blocks = (count + kBlock - 1) / kBlock;
  if (blocks > 4096) { blocks = 4096; }  // grid-stride covers the rest

  gather_validity_kernel<<<static_cast<unsigned>(blocks), kBlock, 0, stream.value()>>>(
    reinterpret_cast<cudf::bitmask_type const*>(source_mask),
    map,
    count,
    reinterpret_cast<cudf::bitmask_type*>(out_mask));
  throw_on_cuda(cudaPeekAtLastError(), "gather_validity launch");
}

void multi_source_gather_fixed(void const* const* bases_dev,
                               std::int64_t const* row_start_dev,
                               int num_batches,
                               std::size_t elem_size,
                               std::uint64_t const* ids,
                               std::int64_t count,
                               void* out,
                               std::uint32_t const* const* masks_dev,
                               std::uint32_t* out_mask,
                               rmm::cuda_stream_view stream)
{
  if (count == 0) { return; }
  if (bases_dev == nullptr || row_start_dev == nullptr || ids == nullptr || out == nullptr ||
      num_batches <= 0) {
    throw std::runtime_error("multi_source_gather: gather over unbound buffers");
  }

  std::int64_t blocks = (count + kBlock - 1) / kBlock;
  if (blocks > 4096) { blocks = 4096; }  // grid-stride covers the rest
  auto const grid = static_cast<unsigned>(blocks);

  auto const* masks = reinterpret_cast<cudf::bitmask_type const* const*>(masks_dev);
  auto* out_bits    = reinterpret_cast<cudf::bitmask_type*>(out_mask);

  // One instantiation per element width rather than a byte loop: the copy is
  // the whole kernel, so it should be one aligned load and store.
  switch (elem_size) {
    case 1:
      gather_fixed_kernel<std::uint8_t>
        <<<grid, kBlock, 0, stream.value()>>>(bases_dev,
                                              row_start_dev,
                                              num_batches,
                                              ids,
                                              count,
                                              static_cast<std::uint8_t*>(out),
                                              masks,
                                              out_bits);
      break;
    case 2:
      gather_fixed_kernel<std::uint16_t>
        <<<grid, kBlock, 0, stream.value()>>>(bases_dev,
                                              row_start_dev,
                                              num_batches,
                                              ids,
                                              count,
                                              static_cast<std::uint16_t*>(out),
                                              masks,
                                              out_bits);
      break;
    case 4:
      gather_fixed_kernel<std::uint32_t>
        <<<grid, kBlock, 0, stream.value()>>>(bases_dev,
                                              row_start_dev,
                                              num_batches,
                                              ids,
                                              count,
                                              static_cast<std::uint32_t*>(out),
                                              masks,
                                              out_bits);
      break;
    case 8:
      gather_fixed_kernel<std::uint64_t>
        <<<grid, kBlock, 0, stream.value()>>>(bases_dev,
                                              row_start_dev,
                                              num_batches,
                                              ids,
                                              count,
                                              static_cast<std::uint64_t*>(out),
                                              masks,
                                              out_bits);
      break;
    case 16:
      gather_fixed_kernel<uint4><<<grid, kBlock, 0, stream.value()>>>(bases_dev,
                                                                      row_start_dev,
                                                                      num_batches,
                                                                      ids,
                                                                      count,
                                                                      static_cast<uint4*>(out),
                                                                      masks,
                                                                      out_bits);
      break;
    default:
      throw std::runtime_error("multi_source_gather: element width " + std::to_string(elem_size) +
                               " is not one of 1, 2, 4, 8, 16");
  }
  throw_on_cuda(cudaPeekAtLastError(), "gather_fixed launch");
}

}  // namespace sirius::late_mat
