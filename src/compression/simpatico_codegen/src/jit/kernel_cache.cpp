#include "codegen/jit/kernel_cache.hpp"

#include "codegen/encode/jit/plain_compile.hpp"

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <cstdint>
#include <cstdio>
#include <mutex>
#include <string>

namespace codegen::jit {

namespace {

constexpr uint64_t kFnvOffsetBasis = 0xcbf29ce484222325ULL;
constexpr uint64_t kFnvPrime       = 0x100000001b3ULL;

uint64_t fnv1a_64(const void* data, std::size_t len) noexcept
{
  auto p     = static_cast<const unsigned char*>(data);
  uint64_t h = kFnvOffsetBasis;
  for (std::size_t i = 0; i < len; ++i) {
    h ^= p[i];
    h *= kFnvPrime;
  }
  return h;
}

std::string to_hex16(uint64_t v)
{
  char buf[17];
  std::snprintf(buf, sizeof(buf), "%016lx", static_cast<unsigned long>(v));
  return std::string(buf, 16);
}

uint64_t mix(uint64_t a, uint64_t b) noexcept
{
  return a ^ (b + 0x9e3779b97f4a7c15ULL + (a << 6) + (a >> 2));
}

}  // namespace

std::string source_digest(const std::string& rendered_source)
{
  return to_hex16(fnv1a_64(rendered_source.data(), rendered_source.size()));
}

std::size_t ShapeKeyHash::operator()(const ShapeKey& k) const noexcept
{
  uint64_t h = fnv1a_64(k.source_hash.data(), k.source_hash.size());
  h          = mix(h, static_cast<uint64_t>(k.arch_cc));
  h          = mix(h, static_cast<uint64_t>(k.cuda_runtime));
  h          = mix(h, static_cast<uint64_t>(k.driver_version));
  return static_cast<std::size_t>(h);
}

ShapeKey shape_key_from(const std::string& rendered_source, int arch_cc)
{
  int driver_version = 0;
  cuDriverGetVersion(&driver_version);

  return ShapeKey{
    source_digest(rendered_source),
    arch_cc,
    static_cast<uint32_t>(CUDART_VERSION),
    static_cast<uint32_t>(driver_version),
  };
}

KernelCache& KernelCache::instance()
{
  static KernelCache c;
  return c;
}

const CompiledKernel* KernelCache::get_or_compile_plain(const std::string& source,
                                                        const std::string& entry_symbol,
                                                        const CompileOptions& opts)
{
  std::string key_material = source;
  key_material += '|';
  key_material += entry_symbol;

  ShapeKey key = shape_key_from(key_material, opts.arch_cc);

  {
    std::lock_guard<std::mutex> lock(mu_);
    if (auto it = table_.find(key); it != table_.end()) { return &it->second; }
  }

  CompiledKernel fresh = ::codegen::encode::jit::compile_plain_kernel(source, entry_symbol, opts);

  std::lock_guard<std::mutex> lock(mu_);
  auto [it, inserted] = table_.emplace(std::move(key), std::move(fresh));
  (void)inserted;
  return &it->second;
}

std::size_t KernelCache::size() const
{
  std::lock_guard<std::mutex> lock(mu_);
  return table_.size();
}

void KernelCache::clear()
{
  std::lock_guard<std::mutex> lock(mu_);
  table_.clear();
}

}  // namespace codegen::jit
