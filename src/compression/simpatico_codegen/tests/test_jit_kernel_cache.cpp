// Layer-1 decode-side cache smoke test (plain-CUDA renderer).
//
// Properties to pin down:
//   1. Cubins land in <dir>/<epoch>/nvrtc-<version>/<request>.cubin, and the
//      first disk-cache use prunes stale epochs and legacy flat cubins.
//   2. Two structurally identical rendered sources hit the same cache slot.
//   3. A different shape gets its own slot.

#include "codegen/decode/jit/renderer.hpp"
#include "codegen/jit/fused_tree.hpp"
#include "codegen/jit/jit_epoch.h"
#include "codegen/jit/kernel_cache.hpp"
#include "test_utils.hpp"

#include <unistd.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>

namespace cdj = codegen::decode::jit;
namespace jit = codegen::jit;
using codegen::OpKind;

static int report_fail(const char* what, const std::string& details = "")
{
  std::fprintf(stderr, "FAIL: %s\n", what);
  if (!details.empty()) std::fprintf(stderr, "--- details ---\n%s\n", details.c_str());
  return 1;
}

template <typename F>
static double timed_ms(F&& fn)
{
  auto t0 = std::chrono::steady_clock::now();
  fn();
  auto t1 = std::chrono::steady_clock::now();
  return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

namespace fs = std::filesystem;

// A private disk cache seeded with other epochs and a legacy flat cubin. Must be
// set up before the first KernelCache lookup, which resolves the directory and
// prunes it once per process.
static fs::path seed_disk_cache()
{
  const fs::path dir =
    fs::temp_directory_path() / ("simpatico_jit_cache_test_" + std::to_string(::getpid()));
  fs::remove_all(dir);
  const auto now = fs::file_time_type::clock::now();
  auto make_dir  = [&](const std::string& name, std::chrono::hours age) {
    fs::create_directories(dir / name / "nvrtc-1.0");
    std::ofstream(dir / name / "nvrtc-1.0" / "stale.cubin") << "stale";
    fs::last_write_time(dir / name, now - age);
  };
  // a, b, c are the three most recently used other epochs and survive (b and c
  // only because of the kept count); d is older and past the grace period.
  make_dir(std::string(32, 'a'), std::chrono::hours(0));
  make_dir(std::string(32, 'b'), std::chrono::hours(48));
  make_dir(std::string(32, 'c'), std::chrono::hours(72));
  make_dir(std::string(32, 'd'), std::chrono::hours(96));
  make_dir("v2", std::chrono::hours(96));  // not an epoch: left alone
  std::ofstream(dir / "0123456789abcdef_a89_c12090_d12090.cubin") << "legacy";
  std::ofstream(dir / "notes.cubin") << "not ours";
  setenv("SIMPATICO_JIT_CACHE_DIR", dir.c_str(), 1);
  return dir;
}

static int check_disk_cache(const fs::path& dir)
{
  const fs::path epoch = dir / jit::kJitEpoch;
  int cubins           = 0;
  std::error_code ec;
  for (const auto& entry : fs::recursive_directory_iterator(epoch, ec)) {
    if (entry.path().extension() != ".cubin") continue;
    if (entry.path().parent_path().filename().string().rfind("nvrtc-", 0) != 0 ||
        entry.path().stem().string().size() != 32)
      return report_fail("unexpected cubin path", entry.path().string());
    ++cubins;
  }
  if (cubins != 1) return report_fail("expected one cubin in the epoch directory");
  for (char kept : {'a', 'b', 'c'}) {
    if (!fs::exists(dir / std::string(32, kept)))
      return report_fail("recent epoch was pruned", std::string(1, kept));
  }
  if (fs::exists(dir / std::string(32, 'd'))) return report_fail("stale epoch not pruned");
  if (fs::exists(dir / "0123456789abcdef_a89_c12090_d12090.cubin"))
    return report_fail("legacy cubin not pruned");
  if (!fs::exists(dir / "v2") || !fs::exists(dir / "notes.cubin"))
    return report_fail("pruned a file the cache does not own");
  return 0;
}

int main()
{
  if (cudaSetDevice(0) != cudaSuccess) return report_fail("cudaSetDevice(0) failed");

  const fs::path cache_dir = seed_disk_cache();

  jit::CompileOptions opts;
  opts.arch_cc = jit::arch_cc_for_current_device();

  auto tree_bp = jit::FusedTree::make(OpKind::Bitpack);
  // Decode is Compact-only (drop-overalloc).

  cdj::DecodeKernelSpec spec_a;
  try {
    spec_a = cdj::render(*tree_bp, "int32_t", 8);
  } catch (const std::exception& e) {
    return report_fail("render Bitpack failed", e.what());
  }

  auto& cache = jit::KernelCache::instance();
  cache.clear();

  const jit::CompiledKernel* k1 = nullptr;
  double cold_ms                = 0;
  try {
    cold_ms =
      timed_ms([&] { k1 = cache.get_or_compile_plain(spec_a.source, spec_a.entry_symbol, opts); });
  } catch (const jit::CompileError& e) {
    return report_fail(e.what(), "log:\n" + e.log);
  } catch (const std::exception& e) {
    return report_fail(e.what());
  }
  if (!k1 || !k1->kern) return report_fail("first compile returned null");
  if (cache.size() != 1) return report_fail("cache size != 1 after first insert");

  if (k1->rendered_source.find("simpatico_bp_at") == std::string::npos) {
    return report_fail("rendered_source missing simpatico_bp_at decode primitive");
  }
  if (int rc = check_disk_cache(cache_dir); rc != 0) return rc;

  const jit::CompiledKernel* k2 = nullptr;
  double warm_ms                = 0;
  try {
    warm_ms =
      timed_ms([&] { k2 = cache.get_or_compile_plain(spec_a.source, spec_a.entry_symbol, opts); });
  } catch (const std::exception& e) {
    return report_fail(e.what());
  }
  if (k1 != k2) return report_fail("warm lookup returned different pointer");
  if (cache.size() != 1) return report_fail("cache size grew on warm hit");
  if (warm_ms * 20.0 > cold_ms) {
    return report_fail(
      "warm not enough faster than cold",
      "cold_ms=" + std::to_string(cold_ms) + " warm_ms=" + std::to_string(warm_ms));
  }

  auto tree_delta_bp =
    jit::FusedTree::make(OpKind::Delta,
                         {
                           {"differences", jit::FusedTree::make(OpKind::Bitpack)},
                         });
  cdj::DecodeKernelSpec spec_b;
  try {
    spec_b = cdj::render(*tree_delta_bp, "int32_t", 8);
  } catch (const std::exception& e) {
    return report_fail("render Delta>Bitpack failed", e.what());
  }

  const jit::CompiledKernel* k3 = nullptr;
  try {
    k3 = cache.get_or_compile_plain(spec_b.source, spec_b.entry_symbol, opts);
  } catch (const std::exception& e) {
    return report_fail(e.what());
  }
  if (!k3 || k3 == k1) return report_fail("different shape did not get own slot");
  if (cache.size() != 2) return report_fail("cache size != 2 after second shape");

  cdj::DecodeKernelSpec spec_c;
  try {
    spec_c = cdj::render(*tree_bp, "int64_t", 8);
  } catch (const std::exception& e) {
    return report_fail("render int64 failed", e.what());
  }
  const jit::CompiledKernel* k4 = nullptr;
  try {
    k4 = cache.get_or_compile_plain(spec_c.source, spec_c.entry_symbol, opts);
  } catch (const std::exception& e) {
    return report_fail(e.what());
  }
  if (!k4 || k4 == k1) return report_fail("dtype change did not change cache slot");
  if (cache.size() != 3) return report_fail("cache size != 3 after dtype variant");

  fs::remove_all(cache_dir);
  std::printf("test_jit_kernel_cache: OK (cold=%.1f ms, warm=%.3f ms, size=%zu)\n",
              cold_ms,
              warm_ms,
              cache.size());
  return 0;
}
