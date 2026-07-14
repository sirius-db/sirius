// SPDX-License-Identifier: Apache-2.0
//
// offsets_cumsum.cu — device-side cumulative-sum helpers for the C++ JIT
// bridge's offset metadata buffers. Both are exposed as plain C entry points so the
// bridge (compiled without nvcc) can link them without dragging
// thrust/cub into its TU. Two unrelated scans share this file because
// they're the same shape (CUB DeviceScan, two-call probe/execute):
//
//   * Bitpack bp_offsets (decode side): derive n_words[c] from
//     (chunk_count[c]*chunk_bits[c]+31)>>5, exclusive-scan into
//     bp_offsets[0..num_chunks); the [num_chunks] sentinel (= total
//     packed words) is patched by a 1-thread tail kernel. A fused
//     transform iterator computes n_words on the fly, so the
//     intermediate array is never materialised.
//   * Rle rle_runs_offsets (encode side): in-place inclusive scan of a
//     pre-staged [0, nruns0, nruns1, …] buffer into the exclusive-prefix
//     layout the decoder expects.

#include <cub/device/device_scan.cuh>
#include <cuda_runtime.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>

#include <cstdint>

namespace {

// Functor instead of a lambda so CUB's DeviceScan instantiation sees a
// stable type across calls — keeps nvcc/sccache caches warm.
struct NwordsFromChunk {
  const std::int32_t* counts;
  const std::uint8_t* bits;

  __host__ __device__ std::int32_t operator()(std::int32_t c) const noexcept
  {
    const std::int32_t cnt = counts[c];
    if (cnt == 0) return 1;
    const std::int64_t total_bits =
      static_cast<std::int64_t>(cnt) * static_cast<std::int64_t>(bits[c]);
    return static_cast<std::int32_t>((total_bits + 31) >> 5);
  }
};

// Tail-patch kernel: writes the sentinel bp_offsets[num_chunks].  The
// exclusive scan produces only [0, num_chunks); the sentinel is the
// inclusive sum at position num_chunks-1 + the last n_words.
__global__ void bp_offsets_tail_kernel(const std::int32_t* counts,
                                       const std::uint8_t* bits,
                                       std::int32_t num_chunks,
                                       std::int32_t* offsets)
{
  if (num_chunks <= 0) return;
  NwordsFromChunk fn{counts, bits};
  offsets[num_chunks] = offsets[num_chunks - 1] + fn(num_chunks - 1);
}

// Single-thread in-place inclusive scan for small problems. CUB's vectorized
// scan kernel reads whole 16-byte quads, over-reading a tight allocation
// whose byte size is not a 16-byte multiple — real OOB on the tiny per-chunk
// metadata buffers this file scans. At these sizes the launch latency
// dominates anyway, so a serial kernel costs nothing and reads exactly n
// elements.
inline constexpr std::int32_t kSerialScanMaxItems = 1024;

__global__ void inclusive_scan_serial_kernel(std::int32_t* buf, std::int32_t n)
{
  std::int32_t acc = 0;
  for (std::int32_t i = 0; i < n; ++i) {
    acc += buf[i];
    buf[i] = acc;
  }
}

}  // namespace

extern "C" {

// Probe + execute exclusive-scan of derived n_words into bp_offsets.
//
// Two-call CUB protocol:
//   * First call (d_temp_storage == nullptr): writes the required
//     scratch byte count into ``*d_temp_storage_bytes_inout``.
//     Returns 0 without touching bp_offsets.
//   * Second call (d_temp_storage != nullptr): performs the scan
//     using the supplied scratch.  Caller must have allocated
//     ``*d_temp_storage_bytes_inout`` bytes.  Does NOT write the
//     sentinel ``bp_offsets[num_chunks]`` — call
//     ``simpatico_compute_bp_offsets_tail`` for that.
//
// All pointers are device-resident.  Returns 0 on success, a
// cudaError_t cast to int on failure.
int simpatico_compute_bp_offsets_scan(const void* d_chunk_count_v,  // const int32_t*  num_chunks
                                      const void* d_chunk_bits_v,   // const uint8_t*  num_chunks
                                      std::int32_t num_chunks,
                                      void* d_bp_offsets_v,  // int32_t*  num_chunks (or +1)
                                      void* d_temp_storage,
                                      std::size_t* d_temp_storage_bytes_inout,
                                      void* stream_v)
{
  if (num_chunks < 0 || d_temp_storage_bytes_inout == nullptr) { return 1; }
  if (num_chunks == 0) {
    if (d_temp_storage != nullptr && d_bp_offsets_v != nullptr) {
      const std::int32_t zero = 0;
      cudaError_t e           = cudaMemcpyAsync(d_bp_offsets_v,
                                      &zero,
                                      sizeof(zero),
                                      cudaMemcpyHostToDevice,
                                      static_cast<cudaStream_t>(stream_v));
      if (e != cudaSuccess) return static_cast<int>(e);
    }
    *d_temp_storage_bytes_inout = 0;
    return 0;
  }

  auto* counts = static_cast<const std::int32_t*>(d_chunk_count_v);
  auto* bits   = static_cast<const std::uint8_t*>(d_chunk_bits_v);
  auto* out    = static_cast<std::int32_t*>(d_bp_offsets_v);
  auto stream  = static_cast<cudaStream_t>(stream_v);

  auto counting = thrust::counting_iterator<std::int32_t>(0);
  auto input    = thrust::make_transform_iterator(counting, NwordsFromChunk{counts, bits});

  std::size_t tmp_bytes = *d_temp_storage_bytes_inout;
  cudaError_t e =
    cub::DeviceScan::ExclusiveSum(d_temp_storage, tmp_bytes, input, out, num_chunks, stream);
  if (e != cudaSuccess) return static_cast<int>(e);
  *d_temp_storage_bytes_inout = tmp_bytes;
  return 0;
}

// Writes the sentinel bp_offsets[num_chunks] = bp_offsets[num_chunks-1]
// + n_words[num_chunks-1].  Must be called after the scan above
// finishes (i.e. on the same stream so the dependency is implicit).
// Single-thread, single-block kernel — negligible runtime; the
// motivation is purely to avoid a device-to-host sync for the tail
// element.  Returns 0 on success, a cudaError_t cast to int on
// failure.
int simpatico_compute_bp_offsets_tail(const void* d_chunk_count_v,
                                      const void* d_chunk_bits_v,
                                      std::int32_t num_chunks,
                                      void* d_bp_offsets_v,
                                      void* stream_v)
{
  if (num_chunks <= 0) return 0;
  bp_offsets_tail_kernel<<<1, 1, 0, static_cast<cudaStream_t>(stream_v)>>>(
    static_cast<const std::int32_t*>(d_chunk_count_v),
    static_cast<const std::uint8_t*>(d_chunk_bits_v),
    num_chunks,
    static_cast<std::int32_t*>(d_bp_offsets_v));
  return static_cast<int>(cudaPeekAtLastError());
}

// In-place inclusive-scan over the encode-side ``rle_runs_offsets`` buffer.
// Two-call CUB protocol matching ``simpatico_compute_bp_offsets_scan``.
// ``length`` is ``num_chunks + 1`` (the full pre-staged buffer); CUB
// supports d_in == d_out. Returns 0 on success, cudaError_t cast to int
// on failure.
int simpatico_compute_rle_offsets_inclusive_scan(void* d_rle_offsets_v,  // int32_t*  length
                                                 std::int32_t length,
                                                 void* d_temp_storage,
                                                 std::size_t* d_temp_storage_bytes_inout,
                                                 void* stream_v)
{
  if (length < 0 || d_temp_storage_bytes_inout == nullptr) { return 1; }
  if (length <= 1) {
    // length=1 means num_chunks=0 — slot [0] is already 0, nothing to scan.
    *d_temp_storage_bytes_inout = 0;
    return 0;
  }

  auto* buf   = static_cast<std::int32_t*>(d_rle_offsets_v);
  auto stream = static_cast<cudaStream_t>(stream_v);

  if (length <= kSerialScanMaxItems) {
    if (d_temp_storage == nullptr) {
      // Probe call. Report a token scratch size so the caller's execute call
      // arrives with d_temp_storage != nullptr and is distinguishable from
      // another probe (the serial kernel must run exactly once).
      *d_temp_storage_bytes_inout = 16;
      return 0;
    }
    inclusive_scan_serial_kernel<<<1, 1, 0, stream>>>(buf, length);
    return static_cast<int>(cudaPeekAtLastError());
  }

  std::size_t tmp_bytes = *d_temp_storage_bytes_inout;
  cudaError_t e =
    cub::DeviceScan::InclusiveSum(d_temp_storage, tmp_bytes, buf, buf, length, stream);
  if (e != cudaSuccess) return static_cast<int>(e);
  *d_temp_storage_bytes_inout = tmp_bytes;
  return 0;
}

}  // extern "C"

// Compact a padded-stride raw values buffer into a dense layout.
// Templated on element type for vectorised stores (avoids byte-by-byte loop).
namespace {
template <typename T>
__global__ void compact_raw_values_kernel(
  const T* __restrict__ padded,              // [num_chunks * chunk_size]
  T* __restrict__ compact,                   // [total_runs]
  const std::int32_t* __restrict__ offsets,  // exclusive prefix [num_chunks+1]
  std::int32_t chunk_size)
{
  std::int32_t chunk = blockIdx.x;
  std::int32_t nruns = offsets[chunk + 1] - offsets[chunk];
  if (nruns <= 0) return;
  const T* src = padded + static_cast<std::int64_t>(chunk) * chunk_size;
  T* dst       = compact + static_cast<std::int64_t>(offsets[chunk]);
  for (std::int32_t i = threadIdx.x; i < nruns; i += blockDim.x)
    dst[i] = src[i];
}
}  // namespace

extern "C" {

int simpatico_compact_raw_values(
  const void* d_padded_v,
  void* d_compact_v,
  const void* d_offsets_v,  // int32_t* exclusive prefix [num_chunks+1]
  std::int32_t num_chunks,
  std::int32_t chunk_size,
  std::int32_t elem_size,
  void* stream_v)
{
  if (num_chunks <= 0 || !d_padded_v || !d_compact_v || !d_offsets_v) return 0;
  auto stream = static_cast<cudaStream_t>(stream_v);
  auto* offs  = static_cast<const std::int32_t*>(d_offsets_v);
  if (elem_size == 8) {
    compact_raw_values_kernel<std::int64_t>
      <<<num_chunks, 128, 0, stream>>>(static_cast<const std::int64_t*>(d_padded_v),
                                       static_cast<std::int64_t*>(d_compact_v),
                                       offs,
                                       chunk_size);
  } else if (elem_size == 1) {
    compact_raw_values_kernel<std::int8_t>
      <<<num_chunks, 128, 0, stream>>>(static_cast<const std::int8_t*>(d_padded_v),
                                       static_cast<std::int8_t*>(d_compact_v),
                                       offs,
                                       chunk_size);
  } else {
    compact_raw_values_kernel<std::int32_t>
      <<<num_chunks, 128, 0, stream>>>(static_cast<const std::int32_t*>(d_padded_v),
                                       static_cast<std::int32_t*>(d_compact_v),
                                       offs,
                                       chunk_size);
  }
  return static_cast<int>(cudaGetLastError());
}

}  // extern "C"
