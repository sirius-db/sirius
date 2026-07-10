// Plain NVRTC compile path for rendered codegen kernels.
//
// Both the encode and decode codegen renderers emit hand-written CUDA
// (`extern "C" __global__`), compiled by nvrtc with the regular
// "-arch=sm_XX -std=c++17" recipe and loaded from a vanilla cubin. The
// kernels are emitted with their plain entry symbol, so there is no
// lowered-name dance — the symbol is the entry name the caller passed in.

#pragma once

#include "codegen/jit/nvrtc_compiler.hpp"  // for CompileOptions, CompileError, CompiledKernel

#include <string>
#include <vector>

namespace codegen::encode::jit {

// Re-exported so callers can write a single `using jit::CompileError`
// without crossing namespaces back to the decode pipeline.
using ::codegen::jit::CompiledKernel;
using ::codegen::jit::CompileError;
using ::codegen::jit::CompileOptions;

// Compile a raw CUDA-C++ source string into a launchable kernel.
//
// `source`        : full translation unit.  Must declare exactly one
//                   `extern "C" __global__ void <entry_symbol>(...)`.
//                   Other declarations are fine (helpers, structs).
// `entry_symbol`  : the kernel's externally-visible C symbol name.
//                   Looked up via cuLibraryGetKernel post-load.
// `opts`          : arch + (unused) kernel_name. `kernel_name` is IGNORED
//                   here — `entry_symbol` is the single source of truth.
//
// Throws `CompileError` on nvrtc rejection (with .log and .source)
// and `std::runtime_error` on driver-API failures.
//
// The returned CompiledKernel stores a device-independent CUkernel handle.
// Call CompiledKernel::func_for_current_device() to obtain the CUfunction
// for whichever GPU is active on the calling thread.
CompiledKernel compile_plain_kernel(const std::string& source,
                                    const std::string& entry_symbol,
                                    const CompileOptions& opts = {});

// Build a launchable CompiledKernel from an already-compiled cubin (skips
// nvrtc). Used by the persistent kernel cache to load a cubin read from disk.
// `rendered_source` is carried into the result only for diagnostics; it does
// not have to be the exact source the cubin was compiled from.
//
// Throws std::runtime_error on any driver-API failure (bad/incompatible
// cubin), so callers can catch and fall back to a fresh compile.
CompiledKernel load_kernel_from_cubin(std::vector<char> cubin,
                                      const std::string& entry_symbol,
                                      std::string rendered_source = {});

}  // namespace codegen::encode::jit
