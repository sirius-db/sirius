// stdint typedef shim — picks the right header for AOT vs JIT compiles.
//
// Background.  Our decode headers use the fixed-width types `int32_t`,
// `uint32_t`, `uint8_t`, etc.  In AOT mode (built by nvcc via CMake)
// `#include <cstdint>` resolves to the host compiler's libstdc++ copy
// which works.  In JIT mode (built by nvrtc at runtime) nvrtc does
// NOT ship libstdc++; `<cstdint>` is unfindable, so the same headers
// fail to compile.
//
// Both modes have access to CCCL's `<cuda/std/cstdint>` though:
//   - nvcc auto-adds the CUDA Toolkit's CCCL include path.
//   - nvrtc gets it via a CCCL include path the JIT driver resolves from the
//     runtime CUDA environment (see cccl_include_dir() in nvrtc_compiler.cpp;
//     override with SIMPATICO_JIT_CCCL_INCLUDE). This header itself is embedded
//     into the binary and fed to nvrtc as an in-memory header.
//
// So: include this shim *instead of* `<cstdint>` in every header.  It
// dispatches and pulls the fixed-width typedefs into the global
// namespace either way.

#pragma once

#if defined(_CODEGEN_CODEGEN_JIT) || defined(__CUDACC_RTC__)
// nvrtc / JIT path — no host stdlib available.  Use CCCL.
//   - fixed-width integer types live in <cuda/std/cstdint>
//   - size_t / ptrdiff_t live in <cuda/std/cstddef>
#include <cuda/std/cstddef>
#include <cuda/std/cstdint>
using cuda::std::int16_t;
using cuda::std::int32_t;
using cuda::std::int64_t;
using cuda::std::int8_t;
using cuda::std::ptrdiff_t;
using cuda::std::size_t;
using cuda::std::uint16_t;
using cuda::std::uint32_t;
using cuda::std::uint64_t;
using cuda::std::uint8_t;
#else
// AOT path — use the host stdlib's <cstdint>, which puts the typedefs
// in both the std:: namespace and (transitively, via <stdint.h>) at
// global scope.  Same surface from the caller's perspective.
#include <cstddef>
#include <cstdint>
#endif
