// In-process shape→CompiledKernel deduplication for plain-CUDA nvrtc JIT.
#pragma once

#include "fused_tree.hpp"
#include "jit/cache_identity.hpp"
#include "nvrtc_compiler.hpp"

#include <cstdint>
#include <list>
#include <mutex>
#include <string>
#include <unordered_map>

namespace codegen::jit {

// Persistent cubins use the versioned compilation identity (see CACHE_FORMAT.md).
// The process-local table holds the request portion for its fixed environment.
// Location resolves to $SIMPATICO_JIT_CACHE_DIR, else
// ${XDG_CACHE_HOME:-$HOME/.cache}/simpatico/jit; set SIMPATICO_JIT_CACHE_DIR
// to "off" (or empty) to disable and fall back to in-memory only.
//
// clear_jit_disk_cache() removes recognized legacy/v2 records and abandoned
// temporaries (best-effort, without following symlinks). Stop writers first.
// No automatic eviction: other-build namespaces persist until explicit cleanup.
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
  KernelCache()  = default;
  ~KernelCache() = default;

  mutable std::mutex mu_;
  std::unordered_map<detail::Digest, CompiledKernel, detail::DigestHash> table_;
  // External headers may change between calls. Retain returned handles without
  // reusing them, so their lifetime still ends only at clear()/destruction.
  std::list<CompiledKernel> uncached_;
};

}  // namespace codegen::jit
