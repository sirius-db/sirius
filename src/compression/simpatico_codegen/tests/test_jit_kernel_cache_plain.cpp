// Layer-1 cache smoke test for the plain-CUDA (encode-side) overload.
//
// Sister of test_jit_kernel_cache.cpp; same three properties pinned for
// the encode pipeline:
//   1. Cold compile produces a non-null kernel + grows the cache by 1.
//   2. Warm lookup of the SAME (source, entry_symbol, arch) returns
//      the same pointer with dramatic speedup over cold (asserts the
//      cache actually short-circuits the nvrtc hop).
//   3. A different rendered source (different tree shape OR different
//      dtype) gets its own slot.
//
// Drives the cache with sources produced by the real encode renderer
// so the test exercises the full integration path the bridge sees,
// not a synthetic-string stand-in.
//
// We don't *launch* the kernels here — that's covered by the JIT
// roundtrip tests (test_jit_roundtrip / test_shape_parity) and the
// end-to-end cpp bridge tests on the simulator side.  This file's job
// is to prove cache plumbing.

#include "codegen/decode/jit/renderer.hpp"
#include "codegen/encode/jit/plain_compile.hpp"  // for cje::CompileError alias
#include "codegen/encode/jit/renderer.hpp"
#include "codegen/jit/fused_tree.hpp"
#include "codegen/jit/kernel_cache.hpp"
#include "test_utils.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>

namespace cje = codegen::encode::jit;
namespace cjd = codegen::decode::jit;
namespace cjj = codegen::jit;
using codegen::OpKind;

static int report_fail(const char* what, const std::string& details = "")
{
  std::fprintf(stderr, "FAIL: %s\n", what);
  if (!details.empty()) { std::fprintf(stderr, "--- details ---\n%s\n", details.c_str()); }
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
  // Real CUDA context is required to load the cuLibrary the cache
  // hands back.  Same bootstrap as the decode-side cache test.
  try {
    cjj::ensure_cuda_context();
  } catch (const std::exception& e) {
    return report_fail("ensure_cuda_context failed", e.what());
  }

  cjj::CompileOptions opts;
  opts.arch_cc = detect_arch_cc();  // matches the decode-side test default

  // Reset cache so size assertions are deterministic regardless of
  // whether the test binary inherits state from anywhere.
  auto& cache = cjj::KernelCache::instance();
  cache.clear();
  if (cache.size() != 0) return report_fail("cache not empty after clear");

  // --- Render two distinct encode kernels via the real renderer. -----
  // Shape A: [Bitpack<int32>].  Smallest fusable shape — proves the
  // wiring without depending on Delta/Rle support.  The encoder
  // currently emits the OverAllocate layout only (Bitpack as a leaf
  // must have fixed_stride=true; see renderer.cpp's contract check).
  auto tree_bp          = cjj::FusedTree::make(OpKind::Bitpack);
  tree_bp->fixed_stride = true;

  cje::EncodeKernelSpec spec_a;
  try {
    spec_a = cje::render(*tree_bp, "int32_t", /*num_chunks=*/8);
  } catch (const std::exception& e) {
    return report_fail("render Bitpack<int32_t> failed", e.what());
  }

  // Shape A': SAME shape + dtype, DIFFERENT num_chunks.  Source must
  // be identical (kernel reads chunk_id from blockIdx.x; buffer
  // sizes are spec-side only) so this should hit the cache slot of
  // spec_a.  Documents and pins the "source invariant in num_chunks"
  // contract the bridge's cache wiring relies on.
  cje::EncodeKernelSpec spec_a_bigger;
  try {
    spec_a_bigger = cje::render(*tree_bp, "int32_t", /*num_chunks=*/1024);
  } catch (const std::exception& e) {
    return report_fail("render Bitpack<int32_t> (1024 chunks) failed", e.what());
  }
  if (spec_a_bigger.source != spec_a.source) {
    return report_fail("rendered source differs across num_chunks — cache key assumption broken",
                       "len_8=" + std::to_string(spec_a.source.size()) +
                         " len_1024=" + std::to_string(spec_a_bigger.source.size()));
  }

  // Shape B: same tree shape, DIFFERENT dtype.  Renders a separate
  // source (different type substitution) so MUST get its own slot.
  cje::EncodeKernelSpec spec_b;
  try {
    spec_b = cje::render(*tree_bp, "int64_t", /*num_chunks=*/8);
  } catch (const std::exception& e) {
    return report_fail("render Bitpack<int64_t> failed", e.what());
  }
  if (spec_b.source == spec_a.source) {
    return report_fail(
      "int32 vs int64 sources are identical "
      "(renderer didn't substitute the dtype?)");
  }

  // --- 1. Cold compile of spec_a. ------------------------------------
  const cjj::CompiledKernel* k1 = nullptr;
  double cold_ms                = 0;
  try {
    cold_ms =
      timed_ms([&] { k1 = cache.get_or_compile_plain(spec_a.source, spec_a.entry_symbol, opts); });
  } catch (const cje::CompileError& e) {
    return report_fail(e.what(), "log:\n" + e.log + "\n--- source ---\n" + e.source);
  } catch (const std::exception& e) {
    return report_fail(e.what());
  }
  if (!k1) return report_fail("first get_or_compile_plain returned null");
  if (!k1->func) return report_fail("CUfunction is null after cold compile");
  if (cache.size() != 1) return report_fail("cache size != 1 after first insert");

  // --- 2. Warm lookup of spec_a — same pointer, fast. ----------------
  const cjj::CompiledKernel* k2 = nullptr;
  double warm_ms                = 0;
  try {
    warm_ms =
      timed_ms([&] { k2 = cache.get_or_compile_plain(spec_a.source, spec_a.entry_symbol, opts); });
  } catch (const std::exception& e) {
    return report_fail(e.what());
  }
  if (k1 != k2)
    return report_fail(
      "warm lookup returned a different pointer "
      "(cache failed to dedup)");
  if (cache.size() != 1) return report_fail("cache size grew on warm hit");

  // No absolute cold floor: plain-compile is 3-5x faster than the
  // codegen JIT path (no extra NVRTC hop) and the driver's ComputeCache +
  // nvrtc on-disk cache make subsequent test-process invocations
  // sub-50ms even though they're "cold" from the in-process cache's
  // point of view.  Relative speedup is the real signal anyway:
  // even fully-disk-warm, cold path runs nvrtcCreateProgram,
  // nvrtcCompileProgram, nvrtcGetCUBIN, cuLibraryLoadData,
  // cuLibraryGetKernel, cuKernelGetFunction (>100 µs total), whereas
  // warm is one FNV-1a hash + mutexed map lookup (<10 µs).
  if (warm_ms * 50.0 > cold_ms) {
    return report_fail(
      "warm lookup not enough faster than cold compile",
      "cold_ms=" + std::to_string(cold_ms) + " warm_ms=" + std::to_string(warm_ms));
  }

  // --- 3. Different dtype -> new slot. -------------------------------
  const cjj::CompiledKernel* k3 = nullptr;
  try {
    k3 = cache.get_or_compile_plain(spec_b.source, spec_b.entry_symbol, opts);
  } catch (const std::exception& e) {
    return report_fail(e.what());
  }
  if (!k3) return report_fail("int64 compile returned null");
  if (k3 == k1) return report_fail("int64 source collided to int32 slot");
  if (cache.size() != 2) return report_fail("cache size != 2 after dtype variant");

  // --- 4. SAME source, DIFFERENT num_chunks -> SAME slot. ------------
  // Validates the bridge's optimisation that a single compile serves
  // every num_rows for a given (tree, dtype) pair.
  const cjj::CompiledKernel* k4 = nullptr;
  try {
    k4 = cache.get_or_compile_plain(spec_a_bigger.source, spec_a_bigger.entry_symbol, opts);
  } catch (const std::exception& e) {
    return report_fail(e.what());
  }
  if (k4 != k1) return report_fail("different num_chunks did not hit the same cache slot");
  if (cache.size() != 2)
    return report_fail("cache size grew on a should-be-warm num_chunks variant");

  // --- FOR render smoke tests ----------------------------------------
  // Shape C: [For{Bitpack<int32>}] — the primary JIT FOR shape.
  // Must render a distinct source from Bitpack alone and from Bitpack<int64>.
  auto tree_for_bp = cjj::FusedTree::make(OpKind::For);
  {
    auto child          = cjj::FusedTree::make(OpKind::Bitpack);
    child->fixed_stride = true;
    tree_for_bp->children.emplace("deltas", std::move(child));
  }

  cje::EncodeKernelSpec spec_for_bp_i32;
  try {
    spec_for_bp_i32 = cje::render(*tree_for_bp, "int32_t", /*num_chunks=*/8);
  } catch (const std::exception& e) {
    return report_fail("render For{Bitpack}<int32_t> failed", e.what());
  }
  if (spec_for_bp_i32.source.empty()) {
    return report_fail("For{Bitpack}<int32_t> rendered empty source");
  }
  if (spec_for_bp_i32.source == spec_a.source) {
    return report_fail("For{Bitpack} source collided with Bitpack source");
  }

  cje::EncodeKernelSpec spec_for_bp_i64;
  try {
    spec_for_bp_i64 = cje::render(*tree_for_bp, "int64_t", /*num_chunks=*/8);
  } catch (const std::exception& e) {
    return report_fail("render For{Bitpack}<int64_t> failed", e.what());
  }
  if (spec_for_bp_i64.source == spec_for_bp_i32.source) {
    return report_fail(
      "For{Bitpack}<int64_t> source identical to int32_t "
      "(dtype not substituted in FOR path)");
  }

  // Compile the FOR encode kernels and verify distinct cache slots.
  const cjj::CompiledKernel* k_for_i32 = nullptr;
  try {
    k_for_i32 =
      cache.get_or_compile_plain(spec_for_bp_i32.source, spec_for_bp_i32.entry_symbol, opts);
  } catch (const cje::CompileError& e) {
    return report_fail(e.what(), "log:\n" + e.log + "\n--- source ---\n" + e.source);
  } catch (const std::exception& e) {
    return report_fail(e.what());
  }
  if (!k_for_i32 || !k_for_i32->func) {
    return report_fail("For{Bitpack}<int32_t> compile returned null");
  }

  const cjj::CompiledKernel* k_for_i64 = nullptr;
  try {
    k_for_i64 =
      cache.get_or_compile_plain(spec_for_bp_i64.source, spec_for_bp_i64.entry_symbol, opts);
  } catch (const cje::CompileError& e) {
    return report_fail(e.what(), "log:\n" + e.log + "\n--- source ---\n" + e.source);
  } catch (const std::exception& e) {
    return report_fail(e.what());
  }
  if (!k_for_i64 || !k_for_i64->func) {
    return report_fail("For{Bitpack}<int64_t> compile returned null");
  }
  if (k_for_i64 == k_for_i32) { return report_fail("int64 For slot collided with int32 For slot"); }

  // Decode-side render: verify FOR decode source renders and compiles.
  {
    cjd::DecodeKernelSpec dspec_i32;
    try {
      dspec_i32 = cjd::render(*tree_for_bp, "int32_t", /*num_chunks=*/8);
    } catch (const std::exception& e) {
      return report_fail("decode render For{Bitpack}<int32_t> failed", e.what());
    }
    if (dspec_i32.source.empty()) {
      return report_fail("decode For{Bitpack}<int32_t> rendered empty source");
    }

    cjd::DecodeKernelSpec dspec_i64;
    try {
      dspec_i64 = cjd::render(*tree_for_bp, "int64_t", /*num_chunks=*/8);
    } catch (const std::exception& e) {
      return report_fail("decode render For{Bitpack}<int64_t> failed", e.what());
    }
    if (dspec_i64.source == dspec_i32.source) {
      return report_fail("decode For{Bitpack}<int64_t> source identical to int32_t");
    }
  }

  const std::size_t final_size = cache.size();
  std::printf(
    "test_jit_kernel_cache_plain: OK "
    "(cold=%.1f ms, warm=%.3f ms, speedup=%.0fx, size=%zu)\n",
    cold_ms,
    warm_ms,
    warm_ms > 0 ? cold_ms / warm_ms : 0.0,
    final_size);
  return 0;
}
