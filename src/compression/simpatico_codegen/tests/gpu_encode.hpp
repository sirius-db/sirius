// Shared GPU-encode helper for JIT roundtrip tests.
//
// Every test produces encoded device buffers by running the *actual* GPU
// encode kernel that `encode::jit::render` emits, then verifies
// correctness by JIT-decoding and comparing against the original input.
//
// `gpu_encode_tree` returns device `LabeledBuffers` (OverAllocate layout)
// plus the rmm `device_buffer`s that own the allocations.  Feed `buffers`
// to `jit_decode.hpp::jit_decode_tree` (plain-CUDA decode renderer +
// `KernelCache::get_or_compile_plain`).
//
// Post-processing the helper performs so callers don't have to:
//   * rle_runs_offsets — the kernel writes raw per-chunk nruns at index
//     c+1 (index 0 == 0); we inclusive-scan in place to the
//     decoder-facing exclusive run prefix (mirrors the production
//     launcher's device-side scan).
//   * bp_offsets — the OverAllocate encoder never produces it, but the Bitpack
//     decode manifest needs it. We synthesize the per-chunk offsets {0, stride,
//     2*stride, ...} from the OverAllocate slot stride so the Compact decode
//     gather reads the padded buffer directly.
//
// Mixing note: device buffers come from the RMM async pool (runtime API);
// the encode kernel is launched via the driver API (`cuLaunchKernel` — it's
// a JIT'd `CUfunction`).  Both share device 0's primary context, so the
// pointers are interchangeable with anything else using that context. The
// async allocs are stream-ordered, so the helper syncs before the launch
// (see below).

#pragma once

#include "codegen/encode/jit/renderer.hpp"
#include "codegen/jit/fused_tree.hpp"
#include "codegen/jit/kernel_cache.hpp"
#include "codegen/jit/nvrtc_compiler.hpp"

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/mr/per_device_resource.hpp>

#include <cuda.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace codegen_test {

struct GpuEncoded {
  codegen::jit::LabeledBuffers buffers;
  std::vector<rmm::device_buffer> owned;  // RAII for the device allocations
  std::int32_t num_chunks = 0;

  // Allocate an (uninitialised) device buffer from the RMM pool, kept alive
  // by *this.  Synchronous — like the cuMemAlloc it replaces — so the
  // returned pointer is immediately usable by a driver launch or copy.
  CUdeviceptr alloc(std::size_t bytes)
  {
    const rmm::cuda_stream_view s{};
    owned.emplace_back(
      bytes == 0 ? std::size_t{1} : bytes, s, rmm::mr::get_current_device_resource_ref());
    cudaStreamSynchronize(s.value());
    return reinterpret_cast<CUdeviceptr>(owned.back().data());
  }

  // alloc + synchronous H2D copy of `bytes` from host `src`.
  CUdeviceptr upload_bytes(const void* src, std::size_t bytes)
  {
    CUdeviceptr d = alloc(bytes);
    cudaMemcpy(reinterpret_cast<void*>(d), src, bytes, cudaMemcpyHostToDevice);
    return d;
  }
};

// Typed device-pointer accessor for the JIT decode path.  Throws if the
// (node_id, field) buffer is missing.
template <typename T>
inline const T* device_ptr(const GpuEncoded& enc, std::int32_t node_id, const std::string& field)
{
  const auto key = codegen::jit::buffer_key(node_id, field);
  auto it        = enc.buffers.find(key);
  if (it == enc.buffers.end()) {
    throw std::runtime_error("gpu_encode_tree: missing buffer '" + key + "'");
  }
  return reinterpret_cast<const T*>(it->second.ptr);
}

// GPU-encode `data` (n elements) through `tree`.  `element_dtype` is the
// column scalar type string ("int32_t"/"int64_t"); `arch_cc` is the
// detected compute capability (e.g. 89).  Throws std::runtime_error /
// RenderError / CompileError on failure.
template <typename Element>
inline GpuEncoded gpu_encode_tree(const codegen::jit::FusedTree& tree,
                                  const std::string& element_dtype,
                                  const Element* data,
                                  std::int64_t n,
                                  int arch_cc)
{
  namespace cc  = codegen;
  namespace cje = codegen::encode::jit;
  namespace jit = codegen::jit;

  auto cu_check = [](CUresult r, const char* what) {
    if (r != CUDA_SUCCESS) {
      const char* s = nullptr;
      cuGetErrorString(r, &s);
      throw std::runtime_error(std::string("gpu_encode_tree: ") + what + ": " + (s ? s : "?"));
    }
  };
  auto rt_check = [](cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
      throw std::runtime_error(std::string("gpu_encode_tree: ") + what + ": " +
                               cudaGetErrorString(e));
    }
  };

  // Device allocations come from the RMM async pool; `out.owned` keeps them
  // alive for the caller (the decode that follows reads these buffers).
  const rmm::cuda_stream_view stream{};
  cudaStream_t cs = stream.value();
  auto mr         = rmm::mr::get_current_device_resource_ref();

  GpuEncoded out;
  out.num_chunks = cc::num_chunks_for(n);

  // 1. Render + compile the encode kernel for this shape. Route through the
  //    shared KernelCache (same path the real pipeline and the operator sweep
  //    use) so the on-disk cubin cache covers fused_operator_sweep's encode too.
  cje::EncodeKernelSpec spec = cje::render(tree, element_dtype, out.num_chunks);
  jit::CompileOptions opts;
  opts.arch_cc = arch_cc;
  const jit::CompiledKernel* kernel =
    jit::KernelCache::instance().get_or_compile_plain(spec.source, spec.entry_symbol, opts);

  // Reserve so the per-spec emplace_backs never reallocate `owned` (keeps
  // the captured d_flat / buf_ptrs raw pointers below trivially valid).
  out.owned.reserve(1 + spec.buffers.size());

  // 2. Upload the flat input column (pool buffer + async H2D).
  const std::size_t flat_bytes =
    (n > 0 ? static_cast<std::size_t>(n) : std::size_t{1}) * sizeof(Element);
  out.owned.emplace_back(flat_bytes, stream, mr);
  CUdeviceptr d_flat = reinterpret_cast<CUdeviceptr>(out.owned.back().data());
  if (n > 0) {
    rt_check(cudaMemcpyAsync(reinterpret_cast<void*>(d_flat),
                             data,
                             static_cast<std::size_t>(n) * sizeof(Element),
                             cudaMemcpyHostToDevice,
                             cs),
             "memcpy flat H2D");
  }

  // 3. One zeroed pool buffer per spec; record in LabeledBuffers and keep
  //    the raw pointers in spec order for the launch arg vector.
  std::vector<CUdeviceptr> buf_ptrs;
  buf_ptrs.reserve(spec.buffers.size());
  for (const auto& b : spec.buffers) {
    const std::size_t bytes       = b.length * b.elem_size;
    const std::size_t alloc_bytes = bytes == 0 ? 1 : bytes;
    out.owned.emplace_back(alloc_bytes, stream, mr);
    CUdeviceptr d = reinterpret_cast<CUdeviceptr>(out.owned.back().data());
    rt_check(cudaMemsetAsync(reinterpret_cast<void*>(d), 0, alloc_bytes, cs), "memset buffer");
    buf_ptrs.push_back(d);
    out.buffers[jit::buffer_key(b.node_id, b.field)] =
      jit::LabeledBuffer{reinterpret_cast<const void*>(d), b.length, b.elem_size};
  }

  // The pool allocs + H2D/memset are stream-ordered, but the pool doesn't
  // reliably order against the driver `cuLaunchKernel` below, so sync first.
  rt_check(cudaStreamSynchronize(cs), "sync before encode launch");

  // 4. Launch: args = [flat, n, buf0, buf1, ...] in spec order.
  std::vector<void*> args;
  args.reserve(2 + buf_ptrs.size());
  args.push_back(&d_flat);
  args.push_back(&n);
  for (auto& p : buf_ptrs)
    args.push_back(&p);

  {
    CUfunction fn_enc   = kernel->func_for_current_device();
    int static_smem_enc = 0;
    cuFuncGetAttribute(&static_smem_enc, CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, fn_enc);
    if (static_smem_enc + static_cast<int>(spec.shared_bytes) > 48 * 1024) {
      cu_check(cuFuncSetAttribute(
                 fn_enc, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, spec.shared_bytes),
               "cuFuncSetAttribute");
    }
  }
  cu_check(cuLaunchKernel(kernel->func_for_current_device(),
                          static_cast<unsigned>(out.num_chunks),
                          1,
                          1,
                          static_cast<unsigned>(spec.block_x),
                          1,
                          1,
                          static_cast<unsigned>(spec.shared_bytes),
                          cs,
                          args.data(),
                          nullptr),
           "cuLaunchKernel(encode)");
  cu_check(cuCtxSynchronize(), "cuCtxSynchronize(encode)");

  // 5. rle_runs_offsets fixup — inclusive scan in place.
  for (const auto& b : spec.buffers) {
    if (b.field != "rle_runs_offsets") continue;
    const auto& lb = out.buffers.at(jit::buffer_key(b.node_id, b.field));
    std::vector<std::int32_t> host(b.length);
    rt_check(cudaMemcpyAsync(host.data(),
                             reinterpret_cast<const void*>(lb.ptr),
                             b.length * sizeof(std::int32_t),
                             cudaMemcpyDeviceToHost,
                             cs),
             "rle_runs_offsets D2H");
    rt_check(cudaStreamSynchronize(cs), "rle_runs_offsets D2H sync");
    for (std::size_t i = 1; i < host.size(); ++i)
      host[i] += host[i - 1];
    rt_check(cudaMemcpyAsync(const_cast<void*>(lb.ptr),
                             host.data(),
                             b.length * sizeof(std::int32_t),
                             cudaMemcpyHostToDevice,
                             cs),
             "rle_runs_offsets H2D");
    rt_check(cudaStreamSynchronize(cs), "rle_runs_offsets H2D sync");
  }

  // 6. bp_offsets injection — real per-chunk offsets per Bitpack node.
  //    The encoder lays ``packed`` out in OverAllocate slots
  //    (chunk ``c`` at word ``c * stride_words``), so the Compact decode
  //    offsets are the arithmetic sequence {0, stride, 2*stride, ...}.
  //    stride_words is recovered from the packed buffer length
  //    (= num_chunks * stride_words words).  This lets the decoder read the
  //    OverAllocate buffer through the uniform Compact gather.
  for (const auto& b : spec.buffers) {
    if (b.field != "packed") continue;  // exactly one per Bitpack node
    const std::int32_t stride_words =
      out.num_chunks > 0
        ? static_cast<std::int32_t>(b.length / static_cast<std::size_t>(out.num_chunks))
        : 0;
    std::vector<std::int32_t> bp_offsets(static_cast<std::size_t>(out.num_chunks) + 1);
    for (std::int32_t c = 0; c <= out.num_chunks; ++c)
      bp_offsets[static_cast<std::size_t>(c)] = c * stride_words;
    CUdeviceptr d_off =
      out.upload_bytes(bp_offsets.data(), bp_offsets.size() * sizeof(std::int32_t));
    out.buffers[jit::buffer_key(b.node_id, "bp_offsets")] = jit::LabeledBuffer{
      reinterpret_cast<const void*>(d_off), bp_offsets.size(), sizeof(std::int32_t)};
  }

  // 7. Variant C decode-transient injection — per-chunk flag buffer
  //    (`rle_scratch`, int32, zeroed) + per-chunk run-value buffer
  //    (`rle_run_values`, Element, uninitialised).  Mirror the production
  //    decode driver's gate: real transients only for a *root* RLE
  //    (node 0); nullptr for every other RLE node (broadcast-compare
  //    fallback).  Any values child is allowed — the values child is
  //    decoded at contiguous run indices in Phase A, so Delta composes.
  //    This lets the JIT roundtrips exercise both paths.
  const bool root_rle_variant_c     = (tree.op == cc::OpKind::Rle);
  const std::size_t rle_scratch_len = static_cast<std::size_t>(out.num_chunks) * cc::kChunkSize;
  for (const auto& b : spec.buffers) {
    if (b.field != "rle_runs_offsets") continue;  // one per Rle node
    const void* sptr  = nullptr;
    const void* rvptr = nullptr;
    if (b.node_id == 0 && root_rle_variant_c && rle_scratch_len > 0) {
      CUdeviceptr d = out.alloc(rle_scratch_len * sizeof(std::int32_t));
      rt_check(cudaMemset(reinterpret_cast<void*>(d), 0, rle_scratch_len * sizeof(std::int32_t)),
               "memset rle_scratch");
      sptr = reinterpret_cast<const void*>(d);

      CUdeviceptr dv = out.alloc(rle_scratch_len * sizeof(Element));
      rvptr          = reinterpret_cast<const void*>(dv);
    }
    out.buffers[jit::buffer_key(b.node_id, "rle_scratch")] =
      jit::LabeledBuffer{sptr, rle_scratch_len, sizeof(std::int32_t)};
    out.buffers[jit::buffer_key(b.node_id, "rle_run_values")] =
      jit::LabeledBuffer{rvptr, rle_scratch_len, sizeof(Element)};
  }

  return out;
}

}  // namespace codegen_test
