// In-process shape→CompiledKernel deduplication for plain-CUDA nvrtc JIT.
#pragma once

#include "fused_tree.hpp"
#include "jit/cache_identity.hpp"
#include "nvrtc_compiler.hpp"

#include <cstdint>
#include <memory>
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
// clear_jit_disk_cache() removes every cached cubin (best-effort); call it
// before any compilation happens (e.g. from a test's main / orchestrator).
void clear_jit_disk_cache();

/**
 * @brief Deduplicates compiled kernels while sharing their ownership with callers.
 *
 * Returned handles keep the CUDA library loaded independently of cache membership. Callers must
 * retain a handle until every launch using it completes; `clear()` releases only cache ownership.
 * Cache operations are thread-safe.
 */
class KernelCache {
 public:
  static KernelCache& instance();

  std::shared_ptr<const CompiledKernel> get_or_compile_plain(const std::string& source,
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
  std::unordered_map<detail::Digest, std::shared_ptr<const CompiledKernel>, detail::DigestHash>
    table_;
};

}  // namespace codegen::jit
