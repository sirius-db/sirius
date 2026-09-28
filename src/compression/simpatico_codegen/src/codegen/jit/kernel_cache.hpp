// In-process shape→CompiledKernel deduplication for plain-CUDA nvrtc JIT.
#pragma once

#include "fused_tree.hpp"
#include "nvrtc_compiler.hpp"

#include <cstdint>
#include <mutex>
#include <string>
#include <unordered_map>

namespace codegen::jit {

struct ShapeKey {
  std::string source_hash;
  int arch_cc;
  uint32_t cuda_runtime;
  uint32_t driver_version;

  bool operator==(const ShapeKey& other) const noexcept
  {
    return arch_cc == other.arch_cc && cuda_runtime == other.cuda_runtime &&
           driver_version == other.driver_version && source_hash == other.source_hash;
  }
};

struct ShapeKeyHash {
  std::size_t operator()(const ShapeKey& k) const noexcept;
};

std::string source_digest(const std::string& rendered_source);
ShapeKey shape_key_from(const std::string& rendered_source, int arch_cc);

// Persistent (on-disk) cubin cache. Compiled kernels are keyed by the same
// (source, arch, cuda runtime, driver) tuple as the in-memory cache and stored
// as <dir>/<hash>.cubin, so a shape compiled once is reused across processes
// and across runs. Location resolves to $SIMPATICO_JIT_CACHE_DIR, else
// ${XDG_CACHE_HOME:-$HOME/.cache}/simpatico/jit; set SIMPATICO_JIT_CACHE_DIR
// to "off" (or empty) to disable and fall back to in-memory only.
//
// clear_jit_disk_cache() removes every cached cubin (best-effort); call it
// before any compilation happens (e.g. from a test's main / orchestrator).
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
  std::unordered_map<ShapeKey, CompiledKernel, ShapeKeyHash> table_;
};

}  // namespace codegen::jit
