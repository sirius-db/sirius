// Internal request shared by cache lookup and the NVRTC call. No test overrides.
#pragma once

#include "cache_identity.hpp"
#include "codegen/jit/nvrtc_compiler.hpp"

#include <array>
#include <limits>

namespace codegen::jit::detail {

struct CompilationRequest {
  CompilationRequest(const std::string& source,
                     const std::string& entry,
                     const CompileOptions& opts);
  CompilationRequest(const CompilationRequest&)            = delete;
  CompilationRequest& operator=(const CompilationRequest&) = delete;

  RequestView identity_view() const;
  bool external_headers() const { return !include_option.empty(); }

  const std::string& source;
  const std::string& entry;
  static constexpr const char* program_name = "codegen_jit.cu";
  // Prefix, every possible decimal int digit, sign, and terminator.
  std::array<char, sizeof("-arch=sm_") + std::numeric_limits<int>::digits10 + 2> architecture{};
  std::string include_option;
  std::array<const char*, 5> options{};
  std::array<std::string_view, 5> option_views{};
  int option_count = 0;
};

CompiledKernel compile_request(const CompilationRequest& request);

struct CacheEnvironment {
  Digest identity{};
  bool persistent = false;
};

const CacheEnvironment& cache_environment();

}  // namespace codegen::jit::detail
