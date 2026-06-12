// Shared JIT decode helper for codegen tests.
//
// Mirrors the production decode path in
// src/bridge/codegen_runtime.cpp::run_rendered_decode:
//   decode::jit::render -> KernelCache::get_or_compile_plain -> flat-buffer launch.
//
// Encode-side fixtures come from gpu_encode.hpp (OverAllocate layout,
// fixed_stride=true on Bitpack nodes).  The decode FusedTree must match
// the layout the renderer expects (fixed_stride on Bitpack nodes).

#pragma once

#include "codegen/decode/jit/renderer.hpp"
#include "codegen/jit/fused_tree.hpp"
#include "codegen/jit/kernel_cache.hpp"
#include "codegen/jit/nvrtc_compiler.hpp"
#include "gpu_encode.hpp"
#include "test_utils.hpp"

#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace codegen_test {

namespace cdj = codegen::decode::jit;
namespace jit = codegen::jit;

inline void cu_check(CUresult r, const char* what)
{
  if (r != CUDA_SUCCESS) {
    const char* s = nullptr;
    cuGetErrorString(r, &s);
    throw std::runtime_error(std::string(what) + ": " + (s ? s : "?"));
  }
}

inline void rt_check(cudaError_t e, const char* what)
{
  if (e != cudaSuccess) {
    throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
  }
}

// Launch the plain-CUDA JIT decode kernel for `tree` into a freshly
// allocated device buffer, then copy the reconstructed column to host.
// `labeled` must carry every channel listed in DecodeKernelSpec.buffers
// (keyed by buffer_key(node_id, field)).
template <typename Element>
inline std::vector<Element> jit_decode_tree(const jit::FusedTree& tree,
                                            const std::string& element_dtype,
                                            std::int64_t n,
                                            const jit::LabeledBuffers& labeled,
                                            GpuEncoded& scratch,
                                            int arch_cc)
{
  jit::ensure_cuda_context();

  const std::int32_t num_chunks = codegen::num_chunks_for(n);

  cdj::DecodeKernelSpec spec;
  try {
    spec = cdj::render(tree, element_dtype, num_chunks);
  } catch (const cdj::RenderError& e) {
    throw std::runtime_error(std::string("decode render: ") + e.what());
  }

  jit::CompileOptions opts;
  opts.arch_cc        = arch_cc;
  opts.default_device = true;

  const jit::CompiledKernel* kernel = nullptr;
  try {
    kernel =
      jit::KernelCache::instance().get_or_compile_plain(spec.source, spec.entry_symbol, opts);
  } catch (const jit::CompileError& e) {
    throw std::runtime_error(std::string("decode compile: ") + e.what() + "\n--- log ---\n" +
                             e.log);
  }
  if (kernel == nullptr || kernel->func == nullptr) {
    throw std::runtime_error("decode compile returned null kernel");
  }

  std::vector<CUdeviceptr> dptrs;
  dptrs.reserve(spec.buffers.size());
  for (const auto& b : spec.buffers) {
    const std::string key = jit::buffer_key(b.node_id, b.field);
    auto it               = labeled.find(key);
    if (it == labeled.end()) {
      throw std::runtime_error("jit_decode_tree: missing labeled buffer '" + key + "'");
    }
    dptrs.push_back(reinterpret_cast<CUdeviceptr>(it->second.ptr));
  }

  CUdeviceptr d_out = scratch.alloc(static_cast<std::size_t>(n > 0 ? n : 1) * sizeof(Element));

  CUdeviceptr d_out_arg = d_out;
  std::int64_t total_n  = n;
  std::vector<void*> args;
  args.reserve(dptrs.size() + 2);
  for (auto& p : dptrs)
    args.push_back(&p);
  args.push_back(&d_out_arg);
  args.push_back(&total_n);

  int static_smem = 0;
  cu_check(cuFuncGetAttribute(&static_smem, CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, kernel->func),
           "cuFuncGetAttribute");
  if (static_smem + spec.shared_bytes > 48 * 1024) {
    cu_check(cuFuncSetAttribute(
               kernel->func, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, spec.shared_bytes),
             "cuFuncSetAttribute");
  }

  cu_check(cuLaunchKernel(kernel->func,
                          static_cast<unsigned>(num_chunks),
                          1,
                          1,
                          static_cast<unsigned>(spec.block_x),
                          1,
                          1,
                          static_cast<unsigned>(spec.shared_bytes),
                          nullptr,
                          args.data(),
                          nullptr),
           "cuLaunchKernel(decode)");
  cu_check(cuCtxSynchronize(), "cuCtxSynchronize(decode)");

  std::vector<Element> out(static_cast<std::size_t>(n));
  if (n > 0) {
    rt_check(cudaMemcpy(out.data(),
                        reinterpret_cast<const void*>(d_out),
                        out.size() * sizeof(Element),
                        cudaMemcpyDeviceToHost),
             "memcpy decode D2H");
  }
  return out;
}

inline bool columns_equal(const std::vector<std::int32_t>& a, const std::vector<std::int32_t>& b)
{
  if (a.size() != b.size()) return false;
  for (std::size_t i = 0; i < a.size(); ++i) {
    if (a[i] != b[i]) return false;
  }
  return true;
}

}  // namespace codegen_test
