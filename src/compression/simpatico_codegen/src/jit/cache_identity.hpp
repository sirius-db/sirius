// Internal, host-only description of the inputs to one NVRTC compilation.
#pragma once

#include <array>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>

namespace codegen::jit::detail {

using Digest = std::array<unsigned char, 32>;  // SHA-256, binary representation

struct RequestView {
  std::string_view source;
  std::string_view entry;
  std::string_view program;
  std::span<const std::string_view> options;
};

struct EnvironmentView {
  std::string_view provider;
  std::string_view project_headers;
  std::string_view cccl_headers;
  std::string_view compiler;
  uint32_t cuda_runtime;
  uint32_t driver;
};

Digest request_identity(const RequestView& request);
Digest environment_identity(const EnvironmentView& environment);
Digest content_identity(std::string_view content);
// Returns false on unreadable/changed files. Used during compiler discovery only.
bool file_identity(const std::string& path, Digest& result);
std::string hex_digest(const Digest& digest);

struct CompilationIdentity {
  Digest environment;
  Digest request;

  std::string relative_path() const;
};

struct DigestHash {
  std::size_t operator()(const Digest& digest) const noexcept;
};

}  // namespace codegen::jit::detail
