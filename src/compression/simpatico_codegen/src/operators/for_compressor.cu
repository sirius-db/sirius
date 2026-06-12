/**
 * Frame-of-Reference (FOR) compressor: store reference (min) per chunk, then
 * deltas (value - reference). Uses variance-based chunking similar to bitpack.
 * Deltas can be further compressed with bitpack for optimal compression.
 *
 * Structure:
 * - references: one value per chunk (the minimum value in that chunk)
 * - reference_offsets: starting index of each chunk (INT32)
 * - deltas: (value - reference) for each element (same type as original)
 */

#include "codegen/plan/representation.hpp"

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/types.hpp>

#include <rmm/device_buffer.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>
#include <rmm/mr/per_device_resource.hpp>
#include <rmm/resource_ref.hpp>

#include <cuda_runtime.h>
#include <thrust/adjacent_difference.h>
#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/for_each.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/sequence.h>
#include <thrust/transform.h>

#include <cstdint>
#include <memory>
#include <stdexcept>

namespace simpatico {

namespace {

constexpr int kForVarianceFactor = 2;  // when bits_needed > initial_bits * this, start new chunk

// Device function: bits needed to represent a range (at least 1)
__device__ __host__ inline int for_bits_for_range(uint64_t range)
{
  if (range == 0) return 1;
#ifdef __CUDA_ARCH__
  return 64 - __clzll(static_cast<unsigned long long>(range));
#else
  return static_cast<int>(64 - __builtin_clzll(static_cast<unsigned long long>(range)));
#endif
}

// Device function: binary search to find which chunk contains element i
// Returns chunk index c such that chunk_start[c] <= i < chunk_start[c+1]
__device__ inline cudf::size_type for_find_chunk_binary_search(
  cudf::size_type const* __restrict__ chunk_start, cudf::size_type num_chunks, cudf::size_type i)
{
  cudf::size_type left  = 0;
  cudf::size_type right = num_chunks - 1;
  while (left < right) {
    cudf::size_type mid = left + (right - left + 1) / 2;
    if (chunk_start[mid] <= i) {
      left = mid;
    } else {
      right = mid - 1;
    }
  }
  return left;
}

// Kernel: compute bits needed at each position from prefix min/max range
template <typename T>
__global__ void for_compute_bits_kernel(T const* __restrict__ prefix_min,
                                        T const* __restrict__ prefix_max,
                                        cudf::size_type n,
                                        uint8_t* __restrict__ bits_out)
{
  cudf::size_type i = static_cast<cudf::size_type>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n) return;
  uint64_t range = static_cast<uint64_t>(prefix_max[i] - prefix_min[i]);
  int bits       = for_bits_for_range(range);
  bits_out[i]    = static_cast<uint8_t>(bits > 64 ? 64 : (bits < 1 ? 1 : bits));
}

// Kernel: mark chunk boundaries based on variance factor
__global__ void for_mark_chunk_boundaries_kernel(uint8_t const* __restrict__ bits,
                                                 cudf::size_type n,
                                                 int variance_factor,
                                                 int32_t* __restrict__ boundary_flags)
{
  cudf::size_type i = static_cast<cudf::size_type>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n) return;

  if (i == 0) {
    boundary_flags[i] = 1;  // First element always starts a chunk
    return;
  }

  // Simple heuristic: start new chunk when bits[i] > bits[i-1] * variance_factor
  int prev_bits     = static_cast<int>(bits[i - 1]);
  int curr_bits     = static_cast<int>(bits[i]);
  boundary_flags[i] = (curr_bits > prev_bits * variance_factor) ? 1 : 0;
}

// Kernel: compute deltas (value - reference) for each element
template <typename T>
__global__ void for_compute_deltas_kernel(T const* __restrict__ data,
                                          T const* __restrict__ chunk_refs,
                                          cudf::size_type const* __restrict__ chunk_start,
                                          cudf::size_type num_chunks,
                                          cudf::size_type n,
                                          T* __restrict__ deltas)
{
  cudf::size_type i = static_cast<cudf::size_type>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n) return;

  // Find which chunk this element belongs to
  cudf::size_type c = for_find_chunk_binary_search(chunk_start, num_chunks, i);
  T ref             = chunk_refs[c];

  // Delta is value - reference (will be >= 0 since ref is min)
  deltas[i] = data[i] - ref;
}

// Kernel: reconstruct original values (reference + delta)
template <typename T>
__global__ void for_reconstruct_kernel(T const* __restrict__ deltas,
                                       T const* __restrict__ chunk_refs,
                                       cudf::size_type const* __restrict__ chunk_start,
                                       cudf::size_type num_chunks,
                                       cudf::size_type n,
                                       T* __restrict__ out)
{
  cudf::size_type i = static_cast<cudf::size_type>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n) return;

  // Find which chunk this element belongs to
  cudf::size_type c = for_find_chunk_binary_search(chunk_start, num_chunks, i);
  T ref             = chunk_refs[c];

  // Original value = reference + delta
  out[i] = ref + deltas[i];
}

template <typename T>
std::unique_ptr<for_compressed_representation> for_compress_impl(cudf::column_view const& col,
                                                                 rmm::cuda_stream_view stream,
                                                                 rmm::device_async_resource_ref mr)
{
  auto const n = col.size();
  if (n == 0) {
    auto empty_refs =
      cudf::make_fixed_width_column(col.type(), 0, cudf::mask_state::UNALLOCATED, stream, mr);
    auto empty_offsets = cudf::make_fixed_width_column(
      cudf::data_type(cudf::type_id::INT32), 0, cudf::mask_state::UNALLOCATED, stream, mr);
    auto empty_deltas =
      cudf::make_fixed_width_column(col.type(), 0, cudf::mask_state::UNALLOCATED, stream, mr);
    return std::make_unique<for_compressed_representation>(
      col.type(), 0, std::move(empty_refs), std::move(empty_offsets), std::move(empty_deltas));
  }

  T const* data = col.data<T>();
  auto exec     = rmm::exec_policy_nosync(stream, mr);

  // Pass 1: Running min and running max (inclusive scan) for chunking decisions.
  rmm::device_uvector<T> d_prefix_min(n, stream, mr);
  rmm::device_uvector<T> d_prefix_max(n, stream, mr);
  thrust::inclusive_scan(exec,
                         thrust::device_pointer_cast(data),
                         thrust::device_pointer_cast(data + n),
                         thrust::device_pointer_cast(d_prefix_min.data()),
                         thrust::minimum<T>());
  thrust::inclusive_scan(exec,
                         thrust::device_pointer_cast(data),
                         thrust::device_pointer_cast(data + n),
                         thrust::device_pointer_cast(d_prefix_max.data()),
                         thrust::maximum<T>());

  // Pass 1b: Compute bits needed at each position (GPU)
  rmm::device_uvector<uint8_t> d_bits(n, stream, mr);
  for_compute_bits_kernel<T><<<(n + 255) / 256, 256, 0, stream.value()>>>(
    d_prefix_min.data(), d_prefix_max.data(), n, d_bits.data());

  // Pass 1c: Mark chunk boundaries (GPU)
  rmm::device_uvector<int32_t> d_boundary_flags(n, stream, mr);
  for_mark_chunk_boundaries_kernel<<<(n + 255) / 256, 256, 0, stream.value()>>>(
    d_bits.data(), n, kForVarianceFactor, d_boundary_flags.data());

  // Pass 1d: Prefix sum of boundary flags to get segment IDs
  rmm::device_uvector<int32_t> d_segment_id(n, stream, mr);
  thrust::inclusive_scan(exec,
                         thrust::device_pointer_cast(d_boundary_flags.data()),
                         thrust::device_pointer_cast(d_boundary_flags.data() + n),
                         thrust::device_pointer_cast(d_segment_id.data()));

  // Compute number of chunks (last element of prefix sum)
  cudf::size_type num_chunks;
  cudaMemcpyAsync(&num_chunks,
                  d_segment_id.data() + n - 1,
                  sizeof(cudf::size_type),
                  cudaMemcpyDeviceToHost,
                  stream.value());
  cudaStreamSynchronize(stream.value());

  // Pass 1e: Extract chunk start positions using stream compaction
  rmm::device_uvector<cudf::size_type> d_chunk_start(num_chunks + 1, stream, mr);

  auto counting_begin = thrust::make_counting_iterator(cudf::size_type{0});
  thrust::copy_if(exec,
                  counting_begin,
                  counting_begin + n,
                  thrust::device_pointer_cast(d_boundary_flags.data()),
                  thrust::device_pointer_cast(d_chunk_start.data()),
                  [] __device__(int32_t flag) { return flag == 1; });

  // Add n as the final chunk end
  cudf::size_type final_end = n;
  cudaMemcpyAsync(d_chunk_start.data() + num_chunks,
                  &final_end,
                  sizeof(cudf::size_type),
                  cudaMemcpyHostToDevice,
                  stream.value());

  // Convert segment IDs from 1-indexed to 0-indexed
  thrust::transform(exec,
                    thrust::device_pointer_cast(d_segment_id.data()),
                    thrust::device_pointer_cast(d_segment_id.data() + n),
                    thrust::device_pointer_cast(d_segment_id.data()),
                    [] __device__(int32_t x) { return x - 1; });

  // Pass 2: Compute min (reference) per chunk using segmented reduction
  rmm::device_uvector<int32_t> d_chunk_keys(num_chunks, stream, mr);
  rmm::device_uvector<T> d_chunk_refs(num_chunks, stream, mr);
  thrust::sequence(exec,
                   thrust::device_pointer_cast(d_chunk_keys.data()),
                   thrust::device_pointer_cast(d_chunk_keys.data() + num_chunks),
                   0);
  thrust::reduce_by_key(exec,
                        thrust::device_pointer_cast(d_segment_id.data()),
                        thrust::device_pointer_cast(d_segment_id.data() + n),
                        thrust::device_pointer_cast(data),
                        thrust::device_pointer_cast(d_chunk_keys.data()),
                        thrust::device_pointer_cast(d_chunk_refs.data()),
                        thrust::equal_to<int32_t>(),
                        thrust::minimum<T>());

  // Pass 3: Compute deltas (value - reference) for each element
  rmm::device_uvector<T> d_deltas(n, stream, mr);
  for_compute_deltas_kernel<T><<<(n + 255) / 256, 256, 0, stream.value()>>>(
    data, d_chunk_refs.data(), d_chunk_start.data(), num_chunks, n, d_deltas.data());

  cudaError_t err = cudaStreamSynchronize(stream.value());
  if (err != cudaSuccess) { throw std::runtime_error("for_compress_impl sync failed"); }

  // Build cudf::columns
  auto refs_col =
    std::make_unique<cudf::column>(std::move(d_chunk_refs), rmm::device_buffer(0, stream, mr), 0);

  // Convert chunk_start to INT32 for reference_offsets (excluding the final sentinel)
  rmm::device_uvector<int32_t> d_ref_offsets(num_chunks, stream, mr);
  thrust::transform(exec,
                    thrust::device_pointer_cast(d_chunk_start.data()),
                    thrust::device_pointer_cast(d_chunk_start.data() + num_chunks),
                    thrust::device_pointer_cast(d_ref_offsets.data()),
                    [] __device__(cudf::size_type x) { return static_cast<int32_t>(x); });

  auto offsets_col =
    std::make_unique<cudf::column>(std::move(d_ref_offsets), rmm::device_buffer(0, stream, mr), 0);

  auto deltas_col =
    std::make_unique<cudf::column>(std::move(d_deltas), rmm::device_buffer(0, stream, mr), 0);

  return std::make_unique<for_compressed_representation>(
    col.type(), n, std::move(refs_col), std::move(offsets_col), std::move(deltas_col));
}

std::unique_ptr<compressed_representation> for_compress_dispatch(cudf::column_view const& col,
                                                                 rmm::cuda_stream_view stream,
                                                                 rmm::device_async_resource_ref mr)
{
  switch (col.type().id()) {
    case cudf::type_id::INT8: return for_compress_impl<int8_t>(col, stream, mr);
    case cudf::type_id::INT16: return for_compress_impl<int16_t>(col, stream, mr);
    case cudf::type_id::INT32: return for_compress_impl<int32_t>(col, stream, mr);
    case cudf::type_id::INT64: return for_compress_impl<int64_t>(col, stream, mr);
    case cudf::type_id::UINT8: return for_compress_impl<uint8_t>(col, stream, mr);
    case cudf::type_id::UINT16: return for_compress_impl<uint16_t>(col, stream, mr);
    case cudf::type_id::UINT32: return for_compress_impl<uint32_t>(col, stream, mr);
    case cudf::type_id::UINT64: return for_compress_impl<uint64_t>(col, stream, mr);
    case cudf::type_id::DECIMAL32: return for_compress_impl<int32_t>(col, stream, mr);
    case cudf::type_id::DECIMAL64: return for_compress_impl<int64_t>(col, stream, mr);
    case cudf::type_id::TIMESTAMP_DAYS:
    case cudf::type_id::DURATION_DAYS:
      return for_compress_impl<int32_t>(col, stream, mr);  // 32-bit types
    case cudf::type_id::TIMESTAMP_SECONDS:
    case cudf::type_id::TIMESTAMP_MILLISECONDS:
    case cudf::type_id::TIMESTAMP_MICROSECONDS:
    case cudf::type_id::TIMESTAMP_NANOSECONDS:
    case cudf::type_id::DURATION_SECONDS:
    case cudf::type_id::DURATION_MILLISECONDS:
    case cudf::type_id::DURATION_MICROSECONDS:
    case cudf::type_id::DURATION_NANOSECONDS:
      return for_compress_impl<int64_t>(col, stream, mr);  // 64-bit types
    default:
      throw std::runtime_error("for_compressor: unsupported column type '" +
                               type_id_to_name(col.type()) + "'");
  }
}

}  // namespace

// -----------------------------------------------------------------------------
// for_compressed_representation
// -----------------------------------------------------------------------------

for_compressed_representation::for_compressed_representation(
  cudf::data_type type,
  cudf::size_type n_rows,
  std::unique_ptr<cudf::column> refs,
  std::unique_ptr<cudf::column> ref_offsets,
  std::unique_ptr<cudf::column> dealt)
  : compressed_representation(type, n_rows),
    references(std::move(refs)),
    reference_offsets(std::move(ref_offsets)),
    deltas(std::move(dealt))
{
}

std::unique_ptr<cudf::column> for_compressed_representation::decompress(
  rmm::cuda_stream_view stream, rmm::device_async_resource_ref mr) const
{
  if (num_rows == 0) {
    return cudf::make_fixed_width_column(
      original_type, 0, cudf::mask_state::UNALLOCATED, stream, mr);
  }

  auto exec = rmm::exec_policy_nosync(stream, mr);

  cudf::column_view refs_view    = references->view();
  cudf::column_view offsets_view = reference_offsets->view();
  cudf::column_view deltas_view  = deltas->view();

  int32_t const* offsets_data = offsets_view.data<int32_t>();

  cudf::size_type num_chunks = references->size();

  // Build chunk_start array (add sentinel at end)
  rmm::device_uvector<cudf::size_type> d_chunk_start(num_chunks + 1, stream, mr);
  thrust::transform(exec,
                    thrust::device_pointer_cast(offsets_data),
                    thrust::device_pointer_cast(offsets_data + num_chunks),
                    thrust::device_pointer_cast(d_chunk_start.data()),
                    [] __device__(int32_t x) { return static_cast<cudf::size_type>(x); });
  cudf::size_type num_rows_val = num_rows;
  cudaMemcpyAsync(d_chunk_start.data() + num_chunks,
                  &num_rows_val,
                  sizeof(cudf::size_type),
                  cudaMemcpyHostToDevice,
                  stream.value());
  // No sync needed — num_rows_val stays in scope through function return,
  // and the H2D copy is ordered before the kernel on the same stream.

  // Dispatch by original_type
#define FOR_RECONSTRUCT_AND_BUILD(T)                                                               \
  do {                                                                                             \
    T const* refs_ptr   = refs_view.data<T>();                                                     \
    T const* deltas_ptr = deltas_view.data<T>();                                                   \
    rmm::device_uvector<T> d_out(num_rows, stream, mr);                                            \
    for_reconstruct_kernel<T><<<(num_rows + 255) / 256, 256, 0, stream.value()>>>(                 \
      deltas_ptr, refs_ptr, d_chunk_start.data(), num_chunks, num_rows, d_out.data());             \
    /* No sync — stream ordering guarantees kernel completes before any subsequent ops */          \
    return std::make_unique<cudf::column>(std::move(d_out), rmm::device_buffer(0, stream, mr), 0); \
  } while (0)

  switch (original_type.id()) {
    case cudf::type_id::INT8: FOR_RECONSTRUCT_AND_BUILD(int8_t); break;
    case cudf::type_id::INT16: FOR_RECONSTRUCT_AND_BUILD(int16_t); break;
    case cudf::type_id::INT32: FOR_RECONSTRUCT_AND_BUILD(int32_t); break;
    case cudf::type_id::INT64: FOR_RECONSTRUCT_AND_BUILD(int64_t); break;
    case cudf::type_id::UINT8: FOR_RECONSTRUCT_AND_BUILD(uint8_t); break;
    case cudf::type_id::UINT16: FOR_RECONSTRUCT_AND_BUILD(uint16_t); break;
    case cudf::type_id::UINT32: FOR_RECONSTRUCT_AND_BUILD(uint32_t); break;
    case cudf::type_id::UINT64: FOR_RECONSTRUCT_AND_BUILD(uint64_t); break;
    case cudf::type_id::DECIMAL32: FOR_RECONSTRUCT_AND_BUILD(int32_t); break;
    case cudf::type_id::DECIMAL64: FOR_RECONSTRUCT_AND_BUILD(int64_t); break;
    case cudf::type_id::TIMESTAMP_DAYS:
    case cudf::type_id::DURATION_DAYS:
      FOR_RECONSTRUCT_AND_BUILD(int32_t);  // 32-bit types
      break;
    case cudf::type_id::TIMESTAMP_SECONDS:
    case cudf::type_id::TIMESTAMP_MILLISECONDS:
    case cudf::type_id::TIMESTAMP_MICROSECONDS:
    case cudf::type_id::TIMESTAMP_NANOSECONDS:
    case cudf::type_id::DURATION_SECONDS:
    case cudf::type_id::DURATION_MILLISECONDS:
    case cudf::type_id::DURATION_MICROSECONDS:
    case cudf::type_id::DURATION_NANOSECONDS:
      FOR_RECONSTRUCT_AND_BUILD(int64_t);  // 64-bit types
      break;
    default: FOR_RECONSTRUCT_AND_BUILD(int64_t); break;
  }
#undef FOR_RECONSTRUCT_AND_BUILD
  return nullptr;  // unreachable
}

// -----------------------------------------------------------------------------
// for_compressor
// -----------------------------------------------------------------------------

std::unique_ptr<compressed_representation> for_compressor::compress(
  cudf::column_view column_to_compress,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  return for_compress_dispatch(column_to_compress, stream, mr);
}

std::unique_ptr<cudf::column> for_compressor::decompress(
  compressed_representation const& data_to_decompress,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  auto const* repr = dynamic_cast<for_compressed_representation const*>(&data_to_decompress);
  if (repr == nullptr) return nullptr;
  return repr->decompress(stream, mr);
}

}  // namespace simpatico
