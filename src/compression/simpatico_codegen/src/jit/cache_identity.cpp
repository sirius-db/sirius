#include "cache_identity.hpp"

// OpenSSL's fixed-size SHA-256 context avoids an EVP heap allocation per lookup.
// Isolate the deprecated OpenSSL 3 API here; no crypto context escapes this TU.
#define OPENSSL_SUPPRESS_DEPRECATED
#include <openssl/sha.h>
#include <sys/stat.h>

#include <array>
#include <fstream>

namespace codegen::jit::detail {
namespace {

class Encoder {
 public:
  explicit Encoder(std::string_view domain)
  {
    SHA256_Init(&state_);
    field(domain);
  }

  void number(uint64_t value)
  {
    std::array<unsigned char, 8> bytes{};
    for (int i = 7; i >= 0; --i) {
      bytes[i] = static_cast<unsigned char>(value);
      value >>= 8;
    }
    SHA256_Update(&state_, bytes.data(), bytes.size());
  }

  void field(std::string_view value)
  {
    number(value.size());
    SHA256_Update(&state_, value.data(), value.size());
  }

  Digest finish()
  {
    Digest result{};
    SHA256_Final(result.data(), &state_);
    return result;
  }

 private:
  SHA256_CTX state_{};
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
  Digest result{};
  SHA256(reinterpret_cast<const unsigned char*>(content.data()), content.size(), result.data());
  return result;
}

bool file_identity(const std::string& path, Digest& result)
{
  struct stat before{}, after{};
  if (::stat(path.c_str(), &before) != 0 || !S_ISREG(before.st_mode)) return false;
  std::ifstream input(path, std::ios::binary);
  if (!input) return false;
  SHA256_CTX hash{};
  SHA256_Init(&hash);
  std::array<char, 64 * 1024> buffer{};
  while (input) {
    input.read(buffer.data(), buffer.size());
    SHA256_Update(&hash, buffer.data(), static_cast<std::size_t>(input.gcount()));
  }
  if (!input.eof() || ::stat(path.c_str(), &after) != 0 || before.st_dev != after.st_dev ||
      before.st_ino != after.st_ino || before.st_size != after.st_size ||
      before.st_mtim.tv_sec != after.st_mtim.tv_sec ||
      before.st_mtim.tv_nsec != after.st_mtim.tv_nsec)
    return false;
  SHA256_Final(result.data(), &hash);
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
