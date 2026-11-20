#include "codegen/jit/kernel_cache.hpp"

#include "compilation_request.hpp"
#include "disk_cache.hpp"

#include <unistd.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <future>
#include <mutex>
#include <string>
#include <vector>

namespace codegen::jit {

// Instrumentation (SIMPATICO_JIT_STATS): count in-memory hits, on-disk hits,
// and nvrtc compiles + total compile wall time, printed once at process exit.
static std::atomic<uint64_t> g_jit_hits{0};
static std::atomic<uint64_t> g_jit_disk_hits{0};
static std::atomic<uint64_t> g_jit_compiles{0};
static std::atomic<uint64_t> g_jit_compile_us{0};
static std::atomic<uint64_t> g_jit_joined{0};
namespace {
struct JitStatsReporter {
  ~JitStatsReporter()
  {
    if (std::getenv("SIMPATICO_JIT_STATS") == nullptr) return;
    std::fprintf(stderr,
                 "[jit-stats pid=%d] compiles=%llu mem_hits=%llu disk_hits=%llu compile_ms=%.0f "
                 "joined=%llu\n",
                 static_cast<int>(::getpid()),
                 static_cast<unsigned long long>(g_jit_compiles.load()),
                 static_cast<unsigned long long>(g_jit_hits.load()),
                 static_cast<unsigned long long>(g_jit_disk_hits.load()),
                 g_jit_compile_us.load() / 1000.0,
                 static_cast<unsigned long long>(g_jit_joined.load()));
  }
};
JitStatsReporter g_jit_stats_reporter;

// ---- persistent on-disk cubin cache ----------------------------------------

// Resolved once per process. "" => disabled (in-memory only).
const std::string& disk_cache_dir()
{
  static const std::string dir = []() -> std::string {
    if (const char* e = std::getenv("SIMPATICO_JIT_CACHE_DIR")) {
      std::string s(e);
      if (s.empty() || s == "off" || s == "0") return "";  // explicitly disabled
      return s;
    }
    if (const char* xdg = std::getenv("XDG_CACHE_HOME"); xdg && *xdg)
      return std::string(xdg) + "/simpatico/jit";
    if (const char* home = std::getenv("HOME"); home && *home)
      return std::string(home) + "/.cache/simpatico/jit";
    return "";  // no writable home => disabled
  }();
  return dir;
}

// Loads the requested module from the disk cache, or compiles it and publishes its cubin there.
std::shared_ptr<const CompiledModule> load_or_compile(const detail::CompilationRequest& request,
                                                      const detail::Digest& key,
                                                      const std::string& entry_symbol)
{
  // On-disk cache: a shape another process/run already compiled loads from its
  // cubin (skips requested-kernel compilation). Invalid storage records and
  // reported CUDA load errors fall through to a fresh compile.
  const std::string* cdir = nullptr;
  try {
    cdir = &disk_cache_dir();
  } catch (...) {
    // Resolving storage is best effort; an identity-query failure below is not.
  }
  std::string path;
  if (cdir && !cdir->empty()) {
    // Do not swallow an NVRTC version error in the storage-miss fallback.
    const detail::CompilationIdentity identity{detail::cache_environment(), key};
    try {
      // One best-effort maintenance pass per process, including first-use disk
      // hits. The static initializer serializes concurrent cold callers.
      static const bool maintained = [&] {
        detail::prune_disk_cache(*cdir, identity.environment);
        return true;
      }();
      (void)maintained;
      path = *cdir + "/" + identity.relative_path();
      std::vector<char> bytes;
      if (detail::read_cubin_file(path, bytes)) {
        try {
          auto loaded = CompiledModule::load(std::move(bytes), request.source);
          // The driver validates an image only when a kernel is first resolved from it, so
          // resolving the requested kernel is what completes the load.
          (void)loaded->kernel(entry_symbol);
          ++g_jit_disk_hits;
          return loaded;
        } catch (...) {
          // fall through to recompile below
        }
      }
    } catch (...) {
      path.clear();
    }
  }

  auto _t0   = std::chrono::steady_clock::now();
  auto fresh = detail::compile_request(request);
  auto _us =
    std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - _t0)
      .count();
  g_jit_compiles.fetch_add(1, std::memory_order_relaxed);
  g_jit_compile_us.fetch_add(static_cast<uint64_t>(_us), std::memory_order_relaxed);

  if (!path.empty()) detail::write_cubin_file_atomic(path, fresh->cubin());

  return fresh;
}

}  // namespace

void clear_jit_disk_cache()
{
  try {
    detail::clear_disk_cache(disk_cache_dir());
  } catch (...) {
    // Even resolving the configured path is best effort.
  }
}

KernelCache& KernelCache::instance()
{
  static KernelCache c;
  return c;
}

/**
 * @brief Removes, when destroyed, the entry the first requester of a key added to `in_flight_`.
 *
 * A `clear()` since the registration has already dropped the entry and may have let a newer request
 * register the same key, so the maps are only touched while the generation is unchanged.
 */
class KernelCache::InFlightRegistration {
 public:
  InFlightRegistration(KernelCache& cache,
                       const detail::Digest& key,
                       std::uint64_t generation) noexcept
    : cache_(cache), key_(key), generation_(generation)
  {
  }
  ~InFlightRegistration()
  {
    const std::lock_guard<std::mutex> lock(cache_.mu_);
    if (current()) cache_.in_flight_.erase(key_);
  }
  InFlightRegistration(const InFlightRegistration&)            = delete;
  InFlightRegistration& operator=(const InFlightRegistration&) = delete;

  /** @brief Moves the key from `in_flight_` to the table while holding the lock throughout. */
  void complete(const Module& module)
  {
    const std::lock_guard<std::mutex> lock(cache_.mu_);
    if (!current()) return;
    cache_.table_.emplace(key_, module);
    cache_.in_flight_.erase(key_);
  }

 private:
  [[nodiscard]] bool current() const noexcept
  {
    return cache_.generation_.load(std::memory_order_relaxed) == generation_;
  }

  KernelCache& cache_;
  detail::Digest key_;
  std::uint64_t generation_;
};

std::shared_ptr<const CompiledKernel> KernelCache::get_or_compile_plain(
  const std::string& source, const std::string& entry_symbol, const CompileOptions& opts)
{
  return get_or_compile_module(source, entry_symbol, opts)->kernel(entry_symbol);
}

KernelCache::Module KernelCache::get_or_compile_module(const std::string& source,
                                                       const std::string& entry_symbol,
                                                       const CompileOptions& opts)
{
  const detail::CompilationRequest request(source, opts);
  const auto key = detail::request_identity(request.identity_view());

  std::unique_lock<std::mutex> lock(mu_);
  if (auto it = table_.find(key); it != table_.end()) {
    ++g_jit_hits;
    return it->second;
  }
  if (auto it = in_flight_.find(key); it != in_flight_.end()) {
    const auto pending = it->second;
    ++g_jit_joined;
    lock.unlock();
    return pending.get();
  }

  // First requester: the task hands its module or its exception to this thread and to every request
  // that joins meanwhile.
  const auto generation = generation_.load(std::memory_order_relaxed);
  std::packaged_task<Module()> work([&] {
    InFlightRegistration registration(*this, key, generation);
    auto module = load_or_compile(request, key, entry_symbol);
    registration.complete(module);
    return module;
  });
  const auto pending = work.get_future().share();
  in_flight_.emplace(key, pending);
  lock.unlock();

  work();
  return pending.get();
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
  in_flight_.clear();
  generation_.fetch_add(1, std::memory_order_release);
}

std::uint64_t KernelCache::generation() const noexcept
{
  return generation_.load(std::memory_order_acquire);
}

KernelCacheStats KernelCache::stats() const noexcept
{
  return {.memory_hits = g_jit_hits.load(std::memory_order_relaxed),
          .disk_hits   = g_jit_disk_hits.load(std::memory_order_relaxed),
          .compiles    = g_jit_compiles.load(std::memory_order_relaxed),
          .compile_us  = g_jit_compile_us.load(std::memory_order_relaxed),
          .joined      = g_jit_joined.load(std::memory_order_relaxed)};
}

}  // namespace codegen::jit
