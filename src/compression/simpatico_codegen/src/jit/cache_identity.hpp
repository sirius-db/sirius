// Internal, host-only description of the inputs to one NVRTC compilation.
#pragma once

#include <array>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>

namespace codegen::jit::detail {

// XXH3-128 in canonical (big-endian) byte order, matching xxhsum -H128.
using Digest = std::array<unsigned char, 16>;

struct RequestView {
  std::string_view source;
  std::string_view entry;
  std::string_view program;
  std::span<const std::string_view> options;
};

struct EnvironmentView {
  std::string_view project_headers;
  std::string_view cccl_headers;
  uint32_t nvrtc_major;
  uint32_t nvrtc_minor;
};

Digest request_identity(const RequestView& request);
Digest environment_identity(const EnvironmentView& environment);
Digest content_identity(std::string_view content);
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
