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

// Everything handed to NVRTC for one compilation that can vary within a build:
// the rendered source, entry symbol, program name, and the effective options
// (including -arch). Embedded headers are fixed per build and covered by the
// cache epoch (codegen/jit/jit_epoch.h).
struct RequestView {
  std::string_view source;
  std::string_view entry;
  std::string_view program;
  std::span<const std::string_view> options;
};

Digest request_identity(const RequestView& request);
Digest content_identity(std::string_view content);
std::string hex_digest(const Digest& digest);

struct DigestHash {
  std::size_t operator()(const Digest& digest) const noexcept;
};

}  // namespace codegen::jit::detail
