// Internal request shared by cache lookup and the NVRTC call. No test overrides.
#pragma once

#include "cache_identity.hpp"
#include "codegen/jit/nvrtc_compiler.hpp"

#include <array>
#include <cstdio>
#include <limits>
#include <stdexcept>

namespace codegen::jit::detail {

struct CompilationRequest {
  CompilationRequest(const std::string& source,
                     const std::string& entry,
                     const CompileOptions& opts)
    : source(source), entry(entry), option_count(opts.device_int128 ? 5U : 4U)
  {
    if (opts.arch_cc <= 0) throw std::invalid_argument("NVRTC architecture must be positive");
    std::snprintf(architecture.data(), architecture.size(), "-arch=sm_%d", opts.arch_cc);
    options = {"-std=c++20",
               architecture.data(),
               "--no-source-include",
               "-default-device",
               "-device-int128"};
    for (std::size_t i = 0; i < options.size(); ++i)
      option_views[i] = options[i];
  }
  CompilationRequest(const CompilationRequest&)            = delete;
  CompilationRequest& operator=(const CompilationRequest&) = delete;

  RequestView identity_view() const
  {
    return {source, entry, program_name, std::span(option_views).first(option_count)};
  }

  const std::string& source;
  const std::string& entry;
  std::size_t option_count{4};
  static constexpr const char* program_name = "codegen_jit.cu";
  // Prefix, every possible decimal int digit, sign, and terminator.
  std::array<char, sizeof("-arch=sm_") + std::numeric_limits<int>::digits10 + 2> architecture{};
  std::array<const char*, 5> options{};
  std::array<std::string_view, 5> option_views{};
};

CompiledKernel compile_request(const CompilationRequest& request);

// Fixed for this process: embedded header digests and NVRTC major/minor.
// Throws on a failed version query; never silently identifies a version as 0.0.
const Digest& cache_environment();

}  // namespace codegen::jit::detail
