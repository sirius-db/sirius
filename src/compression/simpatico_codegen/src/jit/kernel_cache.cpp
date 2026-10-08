#include "codegen/jit/kernel_cache.hpp"

#include "codegen/jit/jit_epoch.h"
#include "jit/cache_identity.hpp"

#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <mutex>
#include <regex>
#include <string>
#include <string_view>
#include <system_error>
#include <type_traits>
#include <utility>
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

namespace fs = std::filesystem;

// Epochs other than the current one are kept while among the most recently
// used kKeptEpochs, or while used within kEpochGracePeriod: another process
// from a different build may share the directory.
constexpr std::size_t kKeptEpochs = 4;
constexpr auto kEpochGracePeriod  = std::chrono::hours(1);

bool is_epoch_name(const std::string& name)
{
  return name.size() == 32 && std::all_of(name.begin(), name.end(), [](char c) {
           return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
         });
}

// <16 hex>_a<arch>_c<cudart>_d<driver>.cubin (+ ".tmp.<pid>"), written by
// builds before the epoch layout. Never read by this scheme.
bool is_legacy_cubin_name(const std::string& name)
{
  static const std::regex legacy(R"([0-9a-f]{16}_a\d+_c\d+_d\d+\.cubin(\.tmp\.\d+)?)");
  return std::regex_match(name, legacy);
}

// Directory holding this build's cubins. Pruning runs once per process, the
// first time the disk cache is used; failures are ignored.
const std::string& epoch_dir()
{
  static const std::string dir = []() -> std::string {
    const std::string& root = disk_cache_dir();
    if (root.empty()) return "";
    std::string current = root + "/" + kJitEpoch;
    std::error_code ec;
    fs::create_directories(current, ec);
    // Directory mtimes record last use, so the pruning below sees this epoch
    // as live even when every lookup hits and nothing is written.
    fs::last_write_time(current, fs::file_time_type::clock::now(), ec);

    std::vector<std::pair<fs::file_time_type, fs::path>> others;
    for (const auto& entry : fs::directory_iterator(root, ec)) {
      const std::string name = entry.path().filename().string();
      if (entry.is_directory(ec) && is_epoch_name(name) && name != kJitEpoch) {
        others.emplace_back(entry.last_write_time(ec), entry.path());
      } else if (entry.is_regular_file(ec) && is_legacy_cubin_name(name)) {
        fs::remove(entry.path(), ec);
      }
    }
    std::sort(others.begin(), others.end(), std::greater<>{});
    const auto cutoff = fs::file_time_type::clock::now() - kEpochGracePeriod;
    for (std::size_t i = kKeptEpochs - 1; i < others.size(); ++i) {
      if (others[i].first < cutoff) fs::remove_all(others[i].second, ec);
    }
    return current;
  }();
  return dir;
}

std::string cubin_path_for(const detail::Digest& key)
{
  const int nvrtc = nvrtc_version();
  return epoch_dir() + "/nvrtc-" + std::to_string(nvrtc / 1000) + "." +
         std::to_string((nvrtc % 1000) / 10) + "/" + detail::hex_digest(key) + ".cubin";
}

bool read_cubin_file(const std::string& path, std::vector<char>& out)
{
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f) return false;
  const std::streamsize n = f.tellg();
  if (n <= 0) return false;
  out.resize(static_cast<std::size_t>(n));
  f.seekg(0);
  return static_cast<bool>(f.read(out.data(), n));
}

// Atomic publish: write to a pid-unique temp then rename into place, so
// concurrent shard processes can never observe a half-written cubin.
void write_cubin_file_atomic(const std::string& path, const std::vector<char>& bytes)
{
  std::error_code ec;
  std::filesystem::create_directories(std::filesystem::path(path).parent_path(), ec);
  const std::string tmp = path + ".tmp." + std::to_string(static_cast<long>(::getpid()));
  {
    std::ofstream f(tmp, std::ios::binary | std::ios::trunc);
    if (!f) return;
    f.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    if (!f) {
      f.close();
      std::remove(tmp.c_str());
      return;
    }
  }
  if (std::rename(tmp.c_str(), path.c_str()) != 0) std::remove(tmp.c_str());
}
}  // namespace

void clear_jit_disk_cache()
{
  const std::string& d = disk_cache_dir();
  if (d.empty()) return;
  std::error_code ec;
  for (auto const& entry : fs::directory_iterator(d, ec)) {
    const std::string name = entry.path().filename().string();
    if (entry.is_directory(ec) && is_epoch_name(name)) {
      fs::remove_all(entry.path(), ec);
    } else if (entry.is_regular_file(ec) && is_legacy_cubin_name(name)) {
      fs::remove(entry.path(), ec);
    }
  }
}

static_assert(std::is_same_v<std::array<unsigned char, 16>, detail::Digest>);

std::size_t KernelCache::KeyHash::operator()(const Key& key) const noexcept
{
  return detail::DigestHash{}(key);
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
  const std::vector<std::string> options = nvrtc_options(opts);
  const std::vector<std::string_view> option_views(options.begin(), options.end());
  Key key = detail::request_identity({source, entry_symbol, kNvrtcProgramName, option_views});

  {
    std::lock_guard<std::mutex> lock(mu_);
    if (auto it = table_.find(key); it != table_.end()) {
      ++g_jit_hits;
      return &it->second;
    }
  }

  // On-disk cache: a shape another process/run already compiled loads from its
  // cubin (skips nvrtc). A corrupt or toolchain-incompatible file just fails
  // the load and falls through to a fresh compile.
  const std::string& cdir = epoch_dir();
  std::string path;
  if (!cdir.empty()) {
    path = cubin_path_for(key);
    std::vector<char> bytes;
    if (read_cubin_file(path, bytes)) {
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

  auto _t0             = std::chrono::steady_clock::now();
  CompiledKernel fresh = compile_plain_kernel(source, entry_symbol, opts);
  auto _us =
    std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - _t0)
      .count();
  g_jit_compiles.fetch_add(1, std::memory_order_relaxed);
  g_jit_compile_us.fetch_add(static_cast<uint64_t>(_us), std::memory_order_relaxed);

  if (!cdir.empty()) write_cubin_file_atomic(path, fresh.cubin);

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
