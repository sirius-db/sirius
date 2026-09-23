#include "codegen/jit/kernel_cache.hpp"

#include "compilation_request.hpp"
#include "disk_cache.hpp"

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <unistd.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
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
namespace {
struct JitStatsReporter {
  ~JitStatsReporter()
  {
    if (std::getenv("SIMPATICO_JIT_STATS") == nullptr) return;
    std::fprintf(stderr,
                 "[jit-stats pid=%d] compiles=%llu mem_hits=%llu disk_hits=%llu compile_ms=%.0f\n",
                 static_cast<int>(::getpid()),
                 static_cast<unsigned long long>(g_jit_compiles.load()),
                 static_cast<unsigned long long>(g_jit_hits.load()),
                 static_cast<unsigned long long>(g_jit_disk_hits.load()),
                 g_jit_compile_us.load() / 1000.0);
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

const CompiledKernel* KernelCache::get_or_compile_plain(const std::string& source,
                                                        const std::string& entry_symbol,
                                                        const CompileOptions& opts)
{
  const detail::CompilationRequest request(source, entry_symbol, opts);
  const bool reuse = !request.external_headers();
  const auto key   = detail::request_identity(request.identity_view());
  if (reuse) {
    std::lock_guard<std::mutex> lock(mu_);
    if (auto it = table_.find(key); it != table_.end()) {
      ++g_jit_hits;
      return &it->second;
    }
  }

  // On-disk cache: a shape another process/run already compiled loads from its
  // cubin (skips requested-kernel compilation). A corrupt or incompatible file fails
  // the load and falls through to a fresh compile.
  std::string path;
  try {
    const std::string& cdir = disk_cache_dir();
    if (reuse && !cdir.empty() && detail::cache_environment().persistent) {
      const detail::CompilationIdentity identity{detail::cache_environment().identity, key};
      path = cdir + "/" + identity.relative_path();
      std::vector<char> bytes;
      if (detail::read_cubin_file(path, bytes)) {
        try {
          CompiledKernel loaded = load_kernel_from_cubin(std::move(bytes), entry_symbol, source);
          ++g_jit_disk_hits;
          std::lock_guard<std::mutex> lock(mu_);
          auto [it, inserted] = table_.emplace(std::move(key), std::move(loaded));
          (void)inserted;
          return &it->second;
        } catch (...) {
          // fall through to recompile below
        }
      }
    }
  } catch (...) {
    path.clear();
  }

  auto _t0             = std::chrono::steady_clock::now();
  CompiledKernel fresh = detail::compile_request(request);
  auto _us =
    std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - _t0)
      .count();
  g_jit_compiles.fetch_add(1, std::memory_order_relaxed);
  g_jit_compile_us.fetch_add(static_cast<uint64_t>(_us), std::memory_order_relaxed);

  if (!path.empty()) detail::write_cubin_file_atomic(path, fresh.cubin);

  std::lock_guard<std::mutex> lock(mu_);
  if (!reuse) {
    uncached_.push_back(std::move(fresh));
    return &uncached_.back();
  }
  auto [it, inserted] = table_.emplace(std::move(key), std::move(fresh));
  (void)inserted;
  return &it->second;
}

std::size_t KernelCache::size() const
{
  std::lock_guard<std::mutex> lock(mu_);
  return table_.size() + uncached_.size();
}

void KernelCache::clear()
{
  std::lock_guard<std::mutex> lock(mu_);
  table_.clear();
  uncached_.clear();
}

}  // namespace codegen::jit
