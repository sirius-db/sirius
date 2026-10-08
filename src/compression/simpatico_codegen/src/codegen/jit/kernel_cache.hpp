// In-process shape→CompiledKernel deduplication for plain-CUDA nvrtc JIT.
#pragma once

#include "fused_tree.hpp"
#include "nvrtc_compiler.hpp"

#include <array>
#include <cstddef>
#include <mutex>
#include <string>
#include <unordered_map>

namespace codegen::jit {

// Persistent (on-disk) cubin cache, shared across processes and runs. A cubin
// is stored as <dir>/<epoch>/nvrtc-<major>.<minor>/<request>.cubin, where
// <epoch> identifies the code that generates kernels (codegen/jit/jit_epoch.h)
// and <request> hashes the rendered source, entry symbol, and NVRTC options
// (jit/cache_identity.hpp). A rebuild that changes kernel generation therefore
// starts a new epoch instead of reading stale cubins; old epochs are pruned
// automatically. Location resolves to $SIMPATICO_JIT_CACHE_DIR, else
// ${XDG_CACHE_HOME:-$HOME/.cache}/simpatico/jit; set SIMPATICO_JIT_CACHE_DIR
// to "off" (or empty) to disable and fall back to in-memory only.
//
// clear_jit_disk_cache() removes every cached cubin (best-effort), e.g. to time
// cold compiles; call it before any compilation happens. Correctness never
// requires it.
void clear_jit_disk_cache();

class KernelCache {
 public:
  static KernelCache& instance();

  const CompiledKernel* get_or_compile_plain(const std::string& source,
                                             const std::string& entry_symbol,
                                             const CompileOptions& opts = {});

  std::size_t size() const;
  void clear();

  KernelCache(const KernelCache&)            = delete;
  KernelCache& operator=(const KernelCache&) = delete;

 private:
  // XXH3-128 request identity; see jit/cache_identity.hpp.
  using Key = std::array<unsigned char, 16>;
  struct KeyHash {
    std::size_t operator()(const Key& key) const noexcept;
  };

  KernelCache()  = default;
  ~KernelCache() = default;

  mutable std::mutex mu_;
  std::unordered_map<Key, CompiledKernel, KeyHash> table_;
};

}  // namespace codegen::jit
