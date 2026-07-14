// SPDX-License-Identifier: Apache-2.0
// Implementation of the low-level (batched) nvcomp codec driver. See the header
// for the frame format and rationale.

#include "codegen/util/cuda_check.hpp"
#include "nvcomp_batched_codec.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace simpatico {
namespace detail {

namespace {

inline std::size_t align_up(std::size_t x, std::size_t a) { return (x + a - 1) / a * a; }

// nvcomp-status check stays local; CUDA-error checks use the shared
// throw_if_cuda_error (cuda_check.hpp).
inline void check(nvcompStatus_t s, char const* what)
{
  if (s != nvcompSuccess) {
    throw std::runtime_error(std::string("nvcomp batched: ") + what + " failed (status " +
                             std::to_string(static_cast<int>(s)) + ")");
  }
}

// One block per chunk; the block's threads cooperatively copy the chunk's
// compressed bytes from the padded scratch layout into the compact frame.
__global__ void scatter_chunks_kernel(std::uint8_t const* __restrict__ src,
                                      std::size_t src_stride,
                                      std::uint8_t* __restrict__ dst,
                                      std::size_t const* __restrict__ dst_offsets,
                                      std::size_t const* __restrict__ sizes,
                                      std::size_t num_chunks)
{
  std::size_t const chunk = blockIdx.x;
  if (chunk >= num_chunks) return;
  std::uint8_t const* s = src + chunk * src_stride;
  std::uint8_t* d       = dst + dst_offsets[chunk];
  std::size_t const n   = sizes[chunk];
  for (std::size_t i = threadIdx.x; i < n; i += blockDim.x)
    d[i] = s[i];
}

}  // namespace

std::pair<std::unique_ptr<rmm::device_buffer>, std::size_t> batched_compress_bytes(
  batched_codec_ops const& ops,
  void const* src,
  std::size_t n_bytes,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  cudaStream_t const s = stream.value();
  if (n_bytes == 0) { return {std::make_unique<rmm::device_buffer>(0, stream, mr), 0}; }

  std::size_t const chunk      = ops.chunk_size;
  std::size_t const num_chunks = (n_bytes + chunk - 1) / chunk;

  // Uncompressed chunk pointers/sizes into the source (host-built, uploaded).
  std::vector<void const*> h_uncomp_ptrs(num_chunks);
  std::vector<std::size_t> h_uncomp_bytes(num_chunks);
  auto const* base = static_cast<std::uint8_t const*>(src);
  for (std::size_t i = 0; i < num_chunks; ++i) {
    h_uncomp_ptrs[i]  = base + i * chunk;
    h_uncomp_bytes[i] = std::min(chunk, n_bytes - i * chunk);
  }

  std::size_t max_out = 0;
  check(ops.compress_get_max_output(chunk, &max_out), "compress_get_max_output");
  std::size_t const out_stride = align_up(max_out, kFrameAlign);

  // Per-chunk output slots (worst-case sized, padded for alignment).
  rmm::device_buffer scratch(out_stride * num_chunks, stream, mr);
  std::vector<void*> h_comp_ptrs(num_chunks);
  for (std::size_t i = 0; i < num_chunks; ++i)
    h_comp_ptrs[i] = static_cast<std::uint8_t*>(scratch.data()) + i * out_stride;

  std::size_t temp_bytes = 0;
  check(ops.compress_get_temp_size(num_chunks, chunk, &temp_bytes, n_bytes),
        "compress_get_temp_size");
  rmm::device_buffer temp(temp_bytes, stream, mr);

  rmm::device_buffer d_uncomp_ptrs(num_chunks * sizeof(void*), stream, mr);
  rmm::device_buffer d_uncomp_bytes(num_chunks * sizeof(std::size_t), stream, mr);
  rmm::device_buffer d_comp_ptrs(num_chunks * sizeof(void*), stream, mr);
  rmm::device_buffer d_comp_bytes(num_chunks * sizeof(std::size_t), stream, mr);
  rmm::device_buffer d_statuses(num_chunks * sizeof(nvcompStatus_t), stream, mr);
  throw_if_cuda_error(cudaMemcpyAsync(d_uncomp_ptrs.data(),
                                      h_uncomp_ptrs.data(),
                                      num_chunks * sizeof(void*),
                                      cudaMemcpyHostToDevice,
                                      s),
                      "H2D uncomp ptrs");
  throw_if_cuda_error(cudaMemcpyAsync(d_uncomp_bytes.data(),
                                      h_uncomp_bytes.data(),
                                      num_chunks * sizeof(std::size_t),
                                      cudaMemcpyHostToDevice,
                                      s),
                      "H2D uncomp bytes");
  throw_if_cuda_error(cudaMemcpyAsync(d_comp_ptrs.data(),
                                      h_comp_ptrs.data(),
                                      num_chunks * sizeof(void*),
                                      cudaMemcpyHostToDevice,
                                      s),
                      "H2D comp ptrs");

  check(ops.compress_async(static_cast<void const* const*>(d_uncomp_ptrs.data()),
                           static_cast<std::size_t const*>(d_uncomp_bytes.data()),
                           chunk,
                           num_chunks,
                           temp.data(),
                           temp_bytes,
                           static_cast<void* const*>(d_comp_ptrs.data()),
                           static_cast<std::size_t*>(d_comp_bytes.data()),
                           static_cast<nvcompStatus_t*>(d_statuses.data()),
                           s),
        "compress_async");

  std::vector<std::size_t> h_comp_bytes(num_chunks);
  throw_if_cuda_error(cudaMemcpyAsync(h_comp_bytes.data(),
                                      d_comp_bytes.data(),
                                      num_chunks * sizeof(std::size_t),
                                      cudaMemcpyDeviceToHost,
                                      s),
                      "D2H comp bytes");
  throw_if_cuda_error(cudaStreamSynchronize(s), "sync after compress");

  // Lay out the compact frame: header + per-chunk sizes, then 256-aligned chunks.
  std::size_t const header = align_up(16 + num_chunks * sizeof(std::size_t), kFrameAlign);
  std::vector<std::size_t> h_dst_off(num_chunks);
  std::size_t cursor = header;
  for (std::size_t i = 0; i < num_chunks; ++i) {
    h_dst_off[i] = cursor;
    cursor       = align_up(cursor + h_comp_bytes[i], kFrameAlign);
  }
  std::size_t const frame_size = h_dst_off[num_chunks - 1] + h_comp_bytes[num_chunks - 1];

  auto frame = std::make_unique<rmm::device_buffer>(frame_size, stream, mr);

  // Header + sizes assembled on host, one H2D.
  std::vector<std::uint8_t> h_hdr(header, 0);
  std::uint64_t const nc = num_chunks;
  std::uint64_t const cs = chunk;
  std::memcpy(h_hdr.data(), &nc, sizeof(nc));
  std::memcpy(h_hdr.data() + 8, &cs, sizeof(cs));
  std::memcpy(h_hdr.data() + 16, h_comp_bytes.data(), num_chunks * sizeof(std::size_t));
  throw_if_cuda_error(
    cudaMemcpyAsync(frame->data(), h_hdr.data(), header, cudaMemcpyHostToDevice, s),
    "H2D frame header");

  rmm::device_buffer d_dst_off(num_chunks * sizeof(std::size_t), stream, mr);
  throw_if_cuda_error(cudaMemcpyAsync(d_dst_off.data(),
                                      h_dst_off.data(),
                                      num_chunks * sizeof(std::size_t),
                                      cudaMemcpyHostToDevice,
                                      s),
                      "H2D dst offsets");

  scatter_chunks_kernel<<<static_cast<unsigned>(num_chunks), 256, 0, s>>>(
    static_cast<std::uint8_t const*>(scratch.data()),
    out_stride,
    static_cast<std::uint8_t*>(frame->data()),
    static_cast<std::size_t const*>(d_dst_off.data()),
    static_cast<std::size_t const*>(d_comp_bytes.data()),
    num_chunks);
  throw_if_cuda_error(cudaGetLastError(), "scatter kernel launch");
  throw_if_cuda_error(cudaStreamSynchronize(s), "sync after scatter");

  return {std::move(frame), frame_size};
}

void batched_decompress_bytes(batched_codec_ops const& ops,
                              void const* frame,
                              std::size_t /*frame_size*/,
                              void* dst,
                              std::size_t out_bytes,
                              rmm::cuda_stream_view stream,
                              rmm::device_async_resource_ref mr)
{
  cudaStream_t const s = stream.value();
  if (out_bytes == 0) return;

  auto const* fbase = static_cast<std::uint8_t const*>(frame);

  std::uint64_t hdr[2] = {0, 0};
  throw_if_cuda_error(cudaMemcpyAsync(hdr, fbase, sizeof(hdr), cudaMemcpyDeviceToHost, s),
                      "D2H frame header");
  throw_if_cuda_error(cudaStreamSynchronize(s), "sync frame header");
  std::size_t const num_chunks = static_cast<std::size_t>(hdr[0]);
  std::size_t const chunk      = static_cast<std::size_t>(hdr[1]);
  if (num_chunks == 0 || chunk == 0) {
    throw std::runtime_error("nvcomp batched: corrupt frame header");
  }

  std::vector<std::size_t> h_comp_bytes(num_chunks);
  throw_if_cuda_error(
    cudaMemcpyAsync(
      h_comp_bytes.data(), fbase + 16, num_chunks * sizeof(std::size_t), cudaMemcpyDeviceToHost, s),
    "D2H comp sizes");
  throw_if_cuda_error(cudaStreamSynchronize(s), "sync comp sizes");

  std::size_t const header = align_up(16 + num_chunks * sizeof(std::size_t), kFrameAlign);
  std::vector<void const*> h_comp_ptrs(num_chunks);
  std::vector<std::size_t> h_uncomp_bytes(num_chunks);
  std::vector<void*> h_uncomp_ptrs(num_chunks);
  auto* d            = static_cast<std::uint8_t*>(dst);
  std::size_t cursor = header;
  for (std::size_t i = 0; i < num_chunks; ++i) {
    h_comp_ptrs[i]    = fbase + cursor;
    cursor            = align_up(cursor + h_comp_bytes[i], kFrameAlign);
    h_uncomp_bytes[i] = std::min(chunk, out_bytes - i * chunk);
    h_uncomp_ptrs[i]  = d + i * chunk;
  }

  rmm::device_buffer d_comp_ptrs(num_chunks * sizeof(void*), stream, mr);
  rmm::device_buffer d_comp_bytes(num_chunks * sizeof(std::size_t), stream, mr);
  rmm::device_buffer d_uncomp_bytes(num_chunks * sizeof(std::size_t), stream, mr);
  rmm::device_buffer d_uncomp_ptrs(num_chunks * sizeof(void*), stream, mr);
  throw_if_cuda_error(cudaMemcpyAsync(d_comp_ptrs.data(),
                                      h_comp_ptrs.data(),
                                      num_chunks * sizeof(void*),
                                      cudaMemcpyHostToDevice,
                                      s),
                      "H2D comp ptrs");
  throw_if_cuda_error(cudaMemcpyAsync(d_comp_bytes.data(),
                                      h_comp_bytes.data(),
                                      num_chunks * sizeof(std::size_t),
                                      cudaMemcpyHostToDevice,
                                      s),
                      "H2D comp bytes");
  throw_if_cuda_error(cudaMemcpyAsync(d_uncomp_bytes.data(),
                                      h_uncomp_bytes.data(),
                                      num_chunks * sizeof(std::size_t),
                                      cudaMemcpyHostToDevice,
                                      s),
                      "H2D uncomp bytes");
  throw_if_cuda_error(cudaMemcpyAsync(d_uncomp_ptrs.data(),
                                      h_uncomp_ptrs.data(),
                                      num_chunks * sizeof(void*),
                                      cudaMemcpyHostToDevice,
                                      s),
                      "H2D uncomp ptrs");

  std::size_t temp_bytes = 0;
  check(ops.decompress_get_temp_size(num_chunks, chunk, &temp_bytes, out_bytes),
        "decompress_get_temp_size");
  rmm::device_buffer temp(temp_bytes, stream, mr);

  // Some codecs (e.g. Cascaded) require the actual-size and status output arrays
  // to be non-null, unlike LZ4/Snappy where they are optional. Always provide
  // them; we do not read them back (correctness is verified upstream).
  rmm::device_buffer d_actual(num_chunks * sizeof(std::size_t), stream, mr);
  rmm::device_buffer d_statuses(num_chunks * sizeof(nvcompStatus_t), stream, mr);

  check(ops.decompress_async(static_cast<void const* const*>(d_comp_ptrs.data()),
                             static_cast<std::size_t const*>(d_comp_bytes.data()),
                             static_cast<std::size_t const*>(d_uncomp_bytes.data()),
                             static_cast<std::size_t*>(d_actual.data()),
                             num_chunks,
                             temp.data(),
                             temp_bytes,
                             static_cast<void* const*>(d_uncomp_ptrs.data()),
                             static_cast<nvcompStatus_t*>(d_statuses.data()),
                             s),
        "decompress_async");
  throw_if_cuda_error(cudaStreamSynchronize(s), "sync after decompress");
}

}  // namespace detail
}  // namespace simpatico
