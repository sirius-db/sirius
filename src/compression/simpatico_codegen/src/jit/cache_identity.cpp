#include "cache_identity.hpp"

// Keep xxHash implementation and state private: no library linkage, exported
// xxHash symbols, or heap allocation for the streaming state.
#define XXH_INLINE_ALL
#include <xxhash.h>

#include <algorithm>
#include <array>

namespace codegen::jit::detail {
namespace {

Digest canonical_digest(XXH128_hash_t hash)
{
  XXH128_canonical_t canonical{};
  XXH128_canonicalFromHash(&canonical, hash);
  Digest result{};
  std::copy(std::begin(canonical.digest), std::end(canonical.digest), result.begin());
  return result;
}

class Encoder {
 public:
  explicit Encoder(std::string_view domain)
  {
    XXH3_128bits_reset(&state_);
    field(domain);
  }

  void number(uint64_t value)
  {
    std::array<unsigned char, 8> bytes{};
    for (int i = 7; i >= 0; --i) {
      bytes[i] = static_cast<unsigned char>(value);
      value >>= 8;
    }
    XXH3_128bits_update(&state_, bytes.data(), bytes.size());
  }

  void field(std::string_view value)
  {
    number(value.size());
    XXH3_128bits_update(&state_, value.data(), value.size());
  }

  Digest finish() { return canonical_digest(XXH3_128bits_digest(&state_)); }

 private:
  XXH3_state_t state_{};
};

}  // namespace

Digest request_identity(const RequestView& request)
{
  Encoder hash("simpatico-request-v2");
  hash.field(request.source);
  hash.field(request.entry);
  hash.field(request.program);
  hash.number(request.options.size());
  for (const auto option : request.options)
    hash.field(option);
  return hash.finish();
}

Digest environment_identity(const EnvironmentView& environment)
{
  Encoder hash("simpatico-environment-v3");
  hash.field(environment.project_headers);
  hash.field(environment.cccl_headers);
  hash.number(environment.nvrtc_major);
  hash.number(environment.nvrtc_minor);
  return hash.finish();
}

Digest content_identity(std::string_view content)
{
  return canonical_digest(XXH3_128bits(content.data(), content.size()));
}

std::string hex_digest(const Digest& digest)
{
  constexpr char digits[] = "0123456789abcdef";
  std::string result(digest.size() * 2, '0');
  for (std::size_t i = 0; i < digest.size(); ++i) {
    result[2 * i]     = digits[digest[i] >> 4];
    result[2 * i + 1] = digits[digest[i] & 15];
  }
  return result;
}

std::string CompilationIdentity::relative_path() const
{
  return "v2/" + hex_digest(environment) + "/" + hex_digest(request) + ".cubin";
}

std::size_t DigestHash::operator()(const Digest& digest) const noexcept
{
  // The full digest is compared for equality; this only selects a hash bucket.
  std::size_t result = 0;
  for (std::size_t i = 0; i < sizeof(result); ++i)
    result = (result << 8) | digest[i];
  return result;
}

}  // namespace codegen::jit::detail
