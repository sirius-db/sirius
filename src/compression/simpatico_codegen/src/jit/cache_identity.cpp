#include "cache_identity.hpp"

// Keep xxHash implementation and state private: no library linkage, exported
// xxHash symbols, or heap allocation for the streaming state.
#define XXH_INLINE_ALL
#include <sys/stat.h>
#include <xxhash.h>

#include <algorithm>
#include <array>
#include <fstream>

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
  Encoder hash("simpatico-environment-v2");
  hash.field(environment.provider);
  hash.field(environment.project_headers);
  hash.field(environment.cccl_headers);
  hash.field(environment.compiler);
  hash.number(environment.cuda_runtime);
  hash.number(environment.driver);
  return hash.finish();
}

Digest content_identity(std::string_view content)
{
  return canonical_digest(XXH3_128bits(content.data(), content.size()));
}

bool file_identity(const std::string& path, Digest& result)
{
  struct stat before{}, after{};
  if (::stat(path.c_str(), &before) != 0 || !S_ISREG(before.st_mode)) return false;
  std::ifstream input(path, std::ios::binary);
  if (!input) return false;
  XXH3_state_t hash{};
  XXH3_128bits_reset(&hash);
  std::array<char, 64 * 1024> buffer{};
  while (input) {
    input.read(buffer.data(), buffer.size());
    XXH3_128bits_update(&hash, buffer.data(), static_cast<std::size_t>(input.gcount()));
  }
  if (!input.eof() || ::stat(path.c_str(), &after) != 0 || before.st_dev != after.st_dev ||
      before.st_ino != after.st_ino || before.st_size != after.st_size ||
      before.st_mtim.tv_sec != after.st_mtim.tv_sec ||
      before.st_mtim.tv_nsec != after.st_mtim.tv_nsec)
    return false;
  result = canonical_digest(XXH3_128bits_digest(&hash));
  return true;
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
