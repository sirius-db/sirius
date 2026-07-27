// Layer-1 decode-side cache smoke test (plain-CUDA renderer).
//
// Three properties to pin down:
//   1. `source_digest` is deterministic and hex-encoded.
//   2. Two structurally identical rendered sources hit the same cache slot.
//   3. A different shape gets its own slot.

#include "codegen/decode/jit/renderer.hpp"
#include "codegen/jit/fused_tree.hpp"
#include "codegen/jit/kernel_cache.hpp"
#include "test_utils.hpp"

#include <chrono>
#include <cstdio>
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

int main()
{
  if (cudaSetDevice(0) != cudaSuccess) return report_fail("cudaSetDevice(0) failed");

  {
    auto a = jit::source_digest("hello world");
    auto b = jit::source_digest("hello world");
    auto c = jit::source_digest("hello world!");
    if (a != b) return report_fail("digest not deterministic");
    if (a == c) return report_fail("digest collided across inputs");
    if (a.size() != 16) return report_fail("digest length != 16", "got: " + a);
    for (char ch : a) {
      if (!((ch >= '0' && ch <= '9') || (ch >= 'a' && ch <= 'f')))
        return report_fail("digest contains non-hex char", "got: " + a);
    }
  }

  jit::CompileOptions opts;
  opts.arch_cc        = jit::arch_cc_for_current_device();
  opts.default_device = true;

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

  std::printf("test_jit_kernel_cache: OK (cold=%.1f ms, warm=%.3f ms, size=%zu)\n",
              cold_ms,
              warm_ms,
              cache.size());
  return 0;
}
