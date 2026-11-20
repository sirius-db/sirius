// In-process shape→CompiledKernel deduplication for plain-CUDA nvrtc JIT.
#pragma once

#include "fused_tree.hpp"
#include "jit/cache_identity.hpp"
#include "nvrtc_compiler.hpp"

#include <atomic>
#include <cstdint>
#include <future>
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
// clear_jit_disk_cache() removes recognized legacy/v2/v3 records and abandoned
// temporaries (best-effort, following only the trusted root). Stop writers first.
// First disk use also prunes old environments: keep current + three newest
// others and a one-hour grace period. This is best effort, not an activity lease.
void clear_jit_disk_cache();

/**
 * @brief Cumulative counts of how the requests to `KernelCache` were answered in this process.
 */
struct KernelCacheStats {
  std::uint64_t memory_hits{0};  ///< Requests answered from the in-memory table
  std::uint64_t disk_hits{0};    ///< Requests that loaded a cubin from the disk cache
  std::uint64_t compiles{0};     ///< Requests that compiled with NVRTC
  std::uint64_t compile_us{0};   ///< Microseconds those compilations took, cubin load included
  std::uint64_t joined{0};       ///< Requests that waited for an identical one already under way
};

/**
 * @brief Deduplicates compiled kernels while sharing their ownership with callers.
 *
 * The cache holds one `CompiledModule` per compiled source and resolves kernels from it, so the
 * kernels of a source that defines several are compiled together, once.
 *
 * Returned handles keep the CUDA library loaded independently of cache membership. Callers must
 * retain a handle until every launch using it completes; `clear()` releases only cache ownership
 * and advances `generation()`, so a caller that memoizes handles can tell that its copies are no
 * longer the cache's. Cache operations are thread-safe.
 */
class KernelCache {
 public:
  static KernelCache& instance();

  /**
   * @brief Returns the kernel @p entry_symbol of @p source, loading the source's module from the
   *        disk cache or compiling it with NVRTC when the in-memory table does not hold it.
   *
   * Requests for one source made at the same time share one disk load or compilation, whichever
   * kernels they ask for: the first requester does the work and the others wait for its result.
   * Requests that differ in the source or a compile option proceed concurrently.
   *
   * A disk load or compilation that fails reaches the first requester and every requester waiting
   * on it as one exception object, which they share and must only read. The failure is not
   * remembered: the next request for the same source starts over. A source that compiles but
   * defines no kernel named @p entry_symbol is cached like any other, and every request for that
   * kernel fails.
   *
   * @throws CompileError when NVRTC rejects @p source.
   * @throws std::runtime_error when an NVRTC or CUDA driver call fails, or when @p source defines
   *         no kernel named @p entry_symbol.
   */
  std::shared_ptr<const CompiledKernel> get_or_compile_plain(const std::string& source,
                                                             const std::string& entry_symbol,
                                                             const CompileOptions& opts = {});

  /** @brief Returns the number of compiled sources the in-memory table holds. */
  std::size_t size() const;

  /**
   * @brief Empties the in-memory table and advances `generation()`.
   *
   * Work under way for earlier requests still completes and reaches its requesters, but its
   * result is not added to the table, and a request made after the call never joins it.
   */
  void clear();

  // Number of clear() calls so far.
  std::uint64_t generation() const noexcept;

  /**
   * @brief Returns a snapshot of the request counters. A waiting request is counted in
   *        `KernelCacheStats::joined` when its wait begins.
   */
  [[nodiscard]] KernelCacheStats stats() const noexcept;

  KernelCache(const KernelCache&)            = delete;
  KernelCache& operator=(const KernelCache&) = delete;

 private:
  using Module = std::shared_ptr<const CompiledModule>;

  KernelCache()  = default;
  ~KernelCache() = default;

  class InFlightRegistration;

  // A disk record is accepted once `entry_symbol` resolves from it: the CUDA driver validates an
  // image only then. Kernels requested from the module later are not part of that check.
  Module get_or_compile_module(const std::string& source,
                               const std::string& entry_symbol,
                               const CompileOptions& opts);

  mutable std::mutex mu_;
  std::atomic<std::uint64_t> generation_{0};
  std::unordered_map<detail::Digest, Module, detail::DigestHash> table_;
  // Requests of the current generation that their first requester is still loading or compiling.
  std::unordered_map<detail::Digest, std::shared_future<Module>, detail::DigestHash> in_flight_;
};

}  // namespace codegen::jit
