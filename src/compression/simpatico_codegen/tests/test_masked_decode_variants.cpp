// Fused scan-filter decode variants (K1 mask_out / K3 mask_consume) —
// correctness roundtrip against plain decode + host-side filtering.
//
// Pipeline under test (production entry points, not test re-implementations):
//   gpu_encode_tree (real encode kernel, OverAllocate)  ->  host compaction to
//   the dense Compact ``packed`` layout the production decode contract expects
//   ->  launch_decode_fused_tree           (plain reference)
//   ->  launch_decode_fused_tree_mask_out  (K1: decode+range-pred -> mask)
//   ->  host CNT-equivalent (per-chunk popcount + exclusive scan)
//   ->  launch_decode_fused_tree_mask_consume (K3: masked compacting decode)
//
// Verified properties:
//   1. plain render is byte-identical with and without the variant argument,
//      and the mask variants get distinct entry symbols + sources (their own
//      JIT-cache entries); non-Bitpack-leaf roots are rejected.
//   2. K1 mask words match a host-computed mask bit-for-bit, INCLUDING the
//      zeroed tail bits/words of a partial last chunk (buffer is pre-filled
//      0xFF to prove the kernel writes the zeros).
//   3. K3 compacted output equals plain-decode + host filter, row order
//      preserved, for mid/all/none selectivities; a zero-survivor chunk
//      exercises the in-kernel early return; a constant (bits==0) chunk
//      exercises the chunk_min short-circuit.
//   4. mask_consume without chunk_offsets (CNT not run) fails cleanly.
//
// GPU required (encode/decode kernels + NVRTC). Same standalone-main harness
// as the other tests in this directory.

#include "codegen/codegen_bridge.hpp"
#include "codegen/decode/jit/renderer.hpp"
#include "codegen/decode/masked_launch.hpp"
#include "codegen/jit/fused_tree.hpp"
#include "codegen/jit/nvrtc_compiler.hpp"
#include "codegen/selection/selection.hpp"
#include "gpu_encode.hpp"

#include <cuda_runtime.h>

#include <rmm/cuda_stream_view.hpp>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace jit = codegen::jit;
namespace cdj = codegen::decode::jit;
using codegen::OpKind;
using codegen_test::device_ptr;
using codegen_test::GpuEncoded;
using sirius::codegen::range_predicate;
using sirius::codegen::selection_mask;

namespace {

constexpr std::int64_t kChunk = codegen::kChunkSize;  // 1024
constexpr int kWordsPerChunk  = static_cast<int>(kChunk / 32);

int g_failures = 0;

#define REQUIRE_MSG(cond, ...)                     \
  do {                                             \
    if (!(cond)) {                                 \
      std::fprintf(stderr, "FAIL: " __VA_ARGS__);  \
      std::fprintf(stderr, "\n");                  \
      ++g_failures;                                \
      return false;                                \
    }                                              \
  } while (0)

std::uint64_t splitmix64(std::uint64_t& s)
{
  s += 0x9E3779B97F4A7C15ull;
  std::uint64_t z = s;
  z               = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
  z               = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
  return z ^ (z >> 31);
}

// Chunk-shaped test data: base + [0, range) pseudo-random, with
//   chunk 2 = constant (bits==0 short-circuit),
//   chunk 4 = confined to the top quarter of the domain (zero survivors
//             under the mid-selectivity predicate -> K3 early return).
template <typename Element>
std::vector<Element> gen_data(std::int64_t n, std::int64_t base, std::int64_t range)
{
  std::vector<Element> v(static_cast<std::size_t>(n));
  for (std::int64_t i = 0; i < n; ++i) {
    const std::int64_t c = i / kChunk;
    std::uint64_t s      = 0xC0FFEEull ^ (0xD1B54A32D192ED03ull * static_cast<std::uint64_t>(i));
    const std::int64_t r = static_cast<std::int64_t>(splitmix64(s) % static_cast<std::uint64_t>(range));
    std::int64_t val;
    if (c == 2) {
      val = base + range / 3;  // constant chunk, inside the mid predicate
    } else if (c == 4) {
      val = base + (3 * range) / 4 + r % (range / 4 + 1);  // above the mid predicate
    } else {
      val = base + r;
    }
    v[static_cast<std::size_t>(i)] = static_cast<Element>(val);
  }
  return v;
}

// Host reference mask: nc*32 words, bit r%32 of word (r/32) set iff
// data[r] in [lo, hi]; tail bits/words zero.
template <typename Element>
std::vector<std::uint32_t> host_mask(const std::vector<Element>& data,
                                     std::int64_t nc,
                                     range_predicate pred)
{
  std::vector<std::uint32_t> m(static_cast<std::size_t>(nc) * kWordsPerChunk, 0u);
  for (std::size_t r = 0; r < data.size(); ++r) {
    const std::int64_t v = static_cast<std::int64_t>(data[r]);
    if (v >= pred.lo && v <= pred.hi) m[r / 32] |= (1u << (r % 32));
  }
  return m;
}

// Replace the OverAllocate ``packed`` channel of bitpack node ``node_id``
// with the dense Compact layout (chunk c's live words at the exclusive scan
// of ceil(count*bits/32), plus 3 zeroed guard words) so the production
// launchers' synthesized bp_offsets match the buffer. Mirrors
// compact_bitpack_packed's layout on host.
bool compact_packed_on_host(GpuEncoded& enc, std::int64_t nc, std::int32_t node_id = 0)
{
  const auto key_packed = jit::buffer_key(node_id, "packed");
  auto it               = enc.buffers.find(key_packed);
  REQUIRE_MSG(it != enc.buffers.end(), "missing %d.packed", node_id);
  const std::size_t total_words = it->second.length;
  const std::size_t stride      = total_words / static_cast<std::size_t>(nc);

  std::vector<std::uint32_t> over(total_words);
  std::vector<std::uint8_t> bits(static_cast<std::size_t>(nc));
  std::vector<std::int32_t> count(static_cast<std::size_t>(nc));
  if (cudaMemcpy(over.data(), it->second.ptr, total_words * 4, cudaMemcpyDeviceToHost) !=
        cudaSuccess ||
      cudaMemcpy(bits.data(),
                 device_ptr<std::uint8_t>(enc, node_id, "chunk_bits"),
                 bits.size(),
                 cudaMemcpyDeviceToHost) != cudaSuccess ||
      cudaMemcpy(count.data(),
                 device_ptr<std::int32_t>(enc, node_id, "chunk_count"),
                 count.size() * 4,
                 cudaMemcpyDeviceToHost) != cudaSuccess) {
    REQUIRE_MSG(false, "compact_packed_on_host: D2H failed");
  }

  std::vector<std::uint32_t> dense;
  dense.reserve(total_words + 3);
  for (std::int64_t c = 0; c < nc; ++c) {
    const std::size_t live =
      (static_cast<std::size_t>(count[static_cast<std::size_t>(c)]) *
         bits[static_cast<std::size_t>(c)] +
       31) /
      32;
    const std::uint32_t* src = over.data() + static_cast<std::size_t>(c) * stride;
    dense.insert(dense.end(), src, src + live);
  }
  dense.insert(dense.end(), 3, 0u);  // guard words for the 3-word gather

  CUdeviceptr d = enc.upload_bytes(dense.data(), dense.size() * 4);
  enc.buffers[key_packed] =
    jit::LabeledBuffer{reinterpret_cast<const void*>(d), dense.size(), sizeof(std::uint32_t)};
  // Drop the harness-injected OverAllocate-stride bp_offsets: the launchers
  // trust a pre-bound bp_offsets entry (rep-level memoization), and the
  // stride offsets are wrong for the dense layout just uploaded — erasing
  // makes the launcher synthesize the matching dense offsets per launch.
  enc.buffers.erase(jit::buffer_key(node_id, "bp_offsets"));
  return true;
}

// Arbitrary host-side selection mask (the consuming variants accept ANY mask,
// e.g. one ANDed together from other columns' K1 outputs): ~keep_pct% bits
// set, chunk ``zero_chunk`` fully cleared (K3/K5 early-return path), tail
// bits/words beyond n zero (contract).
std::vector<std::uint32_t> make_host_mask(std::int64_t n,
                                          std::int64_t nc,
                                          unsigned seed,
                                          unsigned keep_pct,
                                          std::int64_t zero_chunk)
{
  std::vector<std::uint32_t> m(static_cast<std::size_t>(nc) * kWordsPerChunk, 0u);
  for (std::int64_t r = 0; r < n; ++r) {
    if (r / kChunk == zero_chunk) continue;
    std::uint64_t s = seed ^ (0x9E3779B97F4A7C15ull * static_cast<std::uint64_t>(r + 1));
    if (splitmix64(s) % 100ull < keep_pct)
      m[static_cast<std::size_t>(r) / 32] |= (1u << (r % 32));
  }
  return m;
}

// Exclusive per-chunk survivor prefix (host CNT-equivalent). Returns total.
std::int64_t host_cnt(const std::vector<std::uint32_t>& mask,
                      std::int64_t nc,
                      std::vector<std::uint32_t>& chunk_offsets)
{
  chunk_offsets.assign(static_cast<std::size_t>(nc) + 1, 0u);
  std::uint32_t acc = 0;
  for (std::int64_t c = 0; c < nc; ++c) {
    chunk_offsets[static_cast<std::size_t>(c)] = acc;
    for (int w = 0; w < kWordsPerChunk; ++w)
      acc += static_cast<std::uint32_t>(
        __builtin_popcount(mask[static_cast<std::size_t>(c) * kWordsPerChunk + w]));
  }
  chunk_offsets[static_cast<std::size_t>(nc)] = acc;
  return acc;
}

template <typename Element>
bool run_roundtrip(const std::string& dtype, std::int64_t base, std::int64_t range, int arch)
{
  const std::int64_t n  = 5 * kChunk + 700;  // partial tail chunk
  const std::int64_t nc = codegen::num_chunks_for(n);
  const rmm::cuda_stream_view stream{};

  const std::vector<Element> data = gen_data<Element>(n, base, range);

  auto tree = jit::FusedTree::make(OpKind::Bitpack);

  GpuEncoded enc = codegen_test::gpu_encode_tree<Element>(*tree, dtype, data.data(), n, arch);
  if (!compact_packed_on_host(enc, nc)) return false;

  // --- Plain reference through the production launcher. -------------------
  CUdeviceptr d_plain = enc.alloc(static_cast<std::size_t>(n) * sizeof(Element));
  REQUIRE_MSG(simpatico::launch_decode_fused_tree(
                *tree, enc.buffers, dtype.c_str(), n, reinterpret_cast<void*>(d_plain), stream),
              "[%s] plain launch_decode_fused_tree failed",
              dtype.c_str());
  std::vector<Element> plain(static_cast<std::size_t>(n));
  cudaMemcpy(plain.data(),
             reinterpret_cast<const void*>(d_plain),
             plain.size() * sizeof(Element),
             cudaMemcpyDeviceToHost);
  REQUIRE_MSG(plain == data, "[%s] plain decode != original input", dtype.c_str());

  // --- K1 -> host CNT-equivalent -> K3 for three selectivities. ------------
  const range_predicate preds[] = {
    {base + range / 4, base + range / 2},  // mid (chunk 4 has zero survivors)
    {base, base + range},                  // all pass
    {base + 2 * range, base + 3 * range},  // none pass
  };
  const char* pred_names[] = {"mid", "all", "none"};

  const std::size_t nwords = static_cast<std::size_t>(nc) * kWordsPerChunk;
  for (int p = 0; p < 3; ++p) {
    CUdeviceptr d_mask = enc.alloc(nwords * 4);
    cudaMemset(reinterpret_cast<void*>(d_mask), 0xFF, nwords * 4);  // prove tail zeroing

    selection_mask sm;
    sm.words    = reinterpret_cast<std::uint32_t*>(d_mask);
    sm.num_rows = n;

    // mask_consume before CNT ran (no chunk_offsets) must fail cleanly.
    if (p == 0) {
      REQUIRE_MSG(!simpatico::launch_decode_fused_tree_mask_consume(
                    *tree, enc.buffers, dtype.c_str(), n, sm, reinterpret_cast<void*>(d_plain), stream),
                  "[%s] mask_consume without chunk_offsets should fail",
                  dtype.c_str());
    }

    REQUIRE_MSG(simpatico::launch_decode_fused_tree_mask_out(
                  *tree, enc.buffers, dtype.c_str(), n, preds[p], sm, stream),
                "[%s/%s] launch_decode_fused_tree_mask_out failed",
                dtype.c_str(),
                pred_names[p]);

    std::vector<std::uint32_t> mask_dev(nwords);
    cudaMemcpy(mask_dev.data(), reinterpret_cast<const void*>(d_mask), nwords * 4,
               cudaMemcpyDeviceToHost);
    const std::vector<std::uint32_t> mask_ref = host_mask(data, nc, preds[p]);
    REQUIRE_MSG(mask_dev == mask_ref,
                "[%s/%s] K1 mask mismatch vs host reference",
                dtype.c_str(),
                pred_names[p]);

    // Host CNT-equivalent: per-chunk popcount + exclusive prefix scan.
    std::vector<std::uint32_t> chunk_offsets(static_cast<std::size_t>(nc) + 1, 0u);
    std::uint32_t acc = 0;
    for (std::int64_t c = 0; c < nc; ++c) {
      chunk_offsets[static_cast<std::size_t>(c)] = acc;
      for (int w = 0; w < kWordsPerChunk; ++w)
        acc += static_cast<std::uint32_t>(
          __builtin_popcount(mask_ref[static_cast<std::size_t>(c) * kWordsPerChunk + w]));
    }
    chunk_offsets[static_cast<std::size_t>(nc)] = acc;
    const std::int64_t survivors                = acc;

    CUdeviceptr d_offs = enc.upload_bytes(chunk_offsets.data(), chunk_offsets.size() * 4);
    sm.chunk_offsets   = reinterpret_cast<std::uint32_t*>(d_offs);
    sm.survivor_count  = survivors;

    CUdeviceptr d_out =
      enc.alloc(static_cast<std::size_t>(survivors > 0 ? survivors : 1) * sizeof(Element));
    cudaMemset(reinterpret_cast<void*>(d_out), 0xAB,
               static_cast<std::size_t>(survivors > 0 ? survivors : 1) * sizeof(Element));

    REQUIRE_MSG(simpatico::launch_decode_fused_tree_mask_consume(
                  *tree, enc.buffers, dtype.c_str(), n, sm, reinterpret_cast<void*>(d_out), stream),
                "[%s/%s] launch_decode_fused_tree_mask_consume failed",
                dtype.c_str(),
                pred_names[p]);

    // Reference: plain decode + host filter (row order preserved).
    std::vector<Element> expect;
    expect.reserve(static_cast<std::size_t>(survivors));
    for (std::size_t r = 0; r < data.size(); ++r) {
      const std::int64_t v = static_cast<std::int64_t>(data[r]);
      if (v >= preds[p].lo && v <= preds[p].hi) expect.push_back(data[r]);
    }
    REQUIRE_MSG(static_cast<std::int64_t>(expect.size()) == survivors,
                "[%s/%s] internal: survivor count mismatch",
                dtype.c_str(),
                pred_names[p]);

    std::vector<Element> got(static_cast<std::size_t>(survivors));
    if (survivors > 0) {
      cudaMemcpy(got.data(), reinterpret_cast<const void*>(d_out),
                 got.size() * sizeof(Element), cudaMemcpyDeviceToHost);
    }
    REQUIRE_MSG(got == expect,
                "[%s/%s] K3 compacted output != plain decode + host filter (%zu survivors)",
                dtype.c_str(),
                pred_names[p],
                expect.size());

    // K4 (index_consume) must produce the exact same compacted output from
    // the ascending global row-index list (mask->indices wave equivalent).
    std::vector<std::int32_t> row_indices;
    row_indices.reserve(static_cast<std::size_t>(survivors));
    for (std::int64_t r = 0; r < n; ++r)
      if ((mask_ref[static_cast<std::size_t>(r) / 32] >> (r % 32)) & 1u)
        row_indices.push_back(static_cast<std::int32_t>(r));
    CUdeviceptr d_idx = enc.upload_bytes(
      row_indices.data(), row_indices.size() * sizeof(std::int32_t));

    CUdeviceptr d_out4 =
      enc.alloc(static_cast<std::size_t>(survivors > 0 ? survivors : 1) * sizeof(Element));
    cudaMemset(reinterpret_cast<void*>(d_out4), 0xCD,
               static_cast<std::size_t>(survivors > 0 ? survivors : 1) * sizeof(Element));
    REQUIRE_MSG(simpatico::launch_decode_fused_tree_index_consume(
                  *tree,
                  enc.buffers,
                  dtype.c_str(),
                  n,
                  sm,
                  reinterpret_cast<const std::int32_t*>(d_idx),
                  reinterpret_cast<void*>(d_out4),
                  stream),
                "[%s/%s] launch_decode_fused_tree_index_consume failed",
                dtype.c_str(),
                pred_names[p]);
    std::vector<Element> got4(static_cast<std::size_t>(survivors));
    if (survivors > 0) {
      cudaMemcpy(got4.data(), reinterpret_cast<const void*>(d_out4),
                 got4.size() * sizeof(Element), cudaMemcpyDeviceToHost);
    }
    REQUIRE_MSG(got4 == expect,
                "[%s/%s] K4 index-list output != K3/host reference (%zu survivors)",
                dtype.c_str(),
                pred_names[p],
                expect.size());

    std::printf("PASS: %s/%s k3+k4 (survivors=%lld of %lld)\n",
                dtype.c_str(),
                pred_names[p],
                static_cast<long long>(survivors),
                static_cast<long long>(n));
  }

  // K4 singleton: exactly one listed row (deep in a late chunk) — exercises
  // the one-survivor block path and all-other-blocks early return.
  {
    const std::int64_t row = 3 * kChunk + 321;
    std::vector<std::uint32_t> chunk_offsets(static_cast<std::size_t>(nc) + 1, 0u);
    for (std::int64_t c = 0; c <= nc; ++c)
      chunk_offsets[static_cast<std::size_t>(c)] = (c > 3) ? 1u : 0u;
    const std::int32_t idx_host = static_cast<std::int32_t>(row);

    selection_mask sm1;
    sm1.words          = reinterpret_cast<std::uint32_t*>(d_plain);  // unused by K4
    sm1.num_rows       = n;
    sm1.chunk_offsets  = reinterpret_cast<std::uint32_t*>(
      enc.upload_bytes(chunk_offsets.data(), chunk_offsets.size() * 4));
    sm1.survivor_count = 1;

    CUdeviceptr d_idx1 = enc.upload_bytes(&idx_host, sizeof(idx_host));
    CUdeviceptr d_out1 = enc.alloc(sizeof(Element));
    REQUIRE_MSG(simpatico::launch_decode_fused_tree_index_consume(
                  *tree,
                  enc.buffers,
                  dtype.c_str(),
                  n,
                  sm1,
                  reinterpret_cast<const std::int32_t*>(d_idx1),
                  reinterpret_cast<void*>(d_out1),
                  stream),
                "[%s] K4 singleton launch failed",
                dtype.c_str());
    Element got1{};
    cudaMemcpy(&got1, reinterpret_cast<const void*>(d_out1), sizeof(Element),
               cudaMemcpyDeviceToHost);
    REQUIRE_MSG(got1 == data[static_cast<std::size_t>(row)],
                "[%s] K4 singleton decoded wrong value",
                dtype.c_str());
    std::printf("PASS: %s k4-singleton (row=%lld)\n",
                dtype.c_str(),
                static_cast<long long>(row));
  }

  // Exported compute_bp_offsets + memoization guard: pre-bind dense offsets
  // once, verify the launcher reuses them (correct decode) and does NOT
  // erase the pre-bound entry (only per-launch synthesized transients are
  // dropped).
  {
    const auto& cc    = enc.buffers.at(jit::buffer_key(0, "chunk_count"));
    const auto& cb    = enc.buffers.at(jit::buffer_key(0, "chunk_bits"));
    CUdeviceptr d_off = enc.alloc((static_cast<std::size_t>(nc) + 1) * sizeof(std::int32_t));
    REQUIRE_MSG(
      simpatico::compute_bp_offsets(
        cc.ptr,
        cb.ptr,
        static_cast<std::int32_t>(nc),
        reinterpret_cast<void*>(d_off),
        [&](std::size_t bytes) { return reinterpret_cast<void*>(enc.alloc(bytes)); },
        stream.value()) == 0,
      "[%s] exported compute_bp_offsets failed",
      dtype.c_str());
    enc.buffers[jit::buffer_key(0, "bp_offsets")] = jit::LabeledBuffer{
      reinterpret_cast<const void*>(d_off), static_cast<std::size_t>(nc) + 1, sizeof(std::int32_t)};

    REQUIRE_MSG(simpatico::launch_decode_fused_tree(
                  *tree, enc.buffers, dtype.c_str(), n, reinterpret_cast<void*>(d_plain), stream),
                "[%s] memoized plain decode failed",
                dtype.c_str());
    std::vector<Element> memo(static_cast<std::size_t>(n));
    cudaMemcpy(memo.data(),
               reinterpret_cast<const void*>(d_plain),
               memo.size() * sizeof(Element),
               cudaMemcpyDeviceToHost);
    REQUIRE_MSG(memo == data, "[%s] memoized decode != input", dtype.c_str());
    auto it = enc.buffers.find(jit::buffer_key(0, "bp_offsets"));
    REQUIRE_MSG(it != enc.buffers.end() && it->second.ptr == reinterpret_cast<const void*>(d_off),
                "[%s] pre-bound bp_offsets must survive the launch (memoization contract)",
                dtype.c_str());
    enc.buffers.erase(jit::buffer_key(0, "bp_offsets"));  // leave the map as the other cases expect
    std::printf("PASS: %s bp_offsets-memoization\n", dtype.c_str());
  }
  return true;
}

// K3-delta: masked compacting decode of a delta->bitpack column
// (o_orderkey shape).  The mask is host-generated (arbitrary — in
// production it comes from other columns' K1 wave), CNT-equivalent on
// host, then mask_consume must equal plain decode + host filter.
template <typename Element>
bool run_delta_masked(const std::string& dtype, int arch)
{
  const std::int64_t n  = 5 * kChunk + 700;
  const std::int64_t nc = codegen::num_chunks_for(n);
  const rmm::cuda_stream_view stream{};

  // Orderkey-like: monotone with small steps (delta diffs bitpack tightly).
  std::vector<Element> data(static_cast<std::size_t>(n));
  std::int64_t v = 1000;
  for (std::int64_t i = 0; i < n; ++i) {
    std::uint64_t s = 0xBADD1Eull ^ (0xD1B54A32D192ED03ull * static_cast<std::uint64_t>(i + 1));
    v += static_cast<std::int64_t>(splitmix64(s) % 16ull);
    data[static_cast<std::size_t>(i)] = static_cast<Element>(v);
  }

  auto tree = jit::FusedTree::make(
    OpKind::Delta, {{"differences", jit::FusedTree::make(OpKind::Bitpack)}});

  GpuEncoded enc = codegen_test::gpu_encode_tree<Element>(*tree, dtype, data.data(), n, arch);
  // Delta root = node 0, bitpack differences child = node 1.
  if (!compact_packed_on_host(enc, nc, /*node_id=*/1)) return false;

  CUdeviceptr d_plain = enc.alloc(static_cast<std::size_t>(n) * sizeof(Element));
  REQUIRE_MSG(simpatico::launch_decode_fused_tree(
                *tree, enc.buffers, dtype.c_str(), n, reinterpret_cast<void*>(d_plain), stream),
              "[delta/%s] plain launch failed",
              dtype.c_str());
  std::vector<Element> plain(static_cast<std::size_t>(n));
  cudaMemcpy(plain.data(),
             reinterpret_cast<const void*>(d_plain),
             plain.size() * sizeof(Element),
             cudaMemcpyDeviceToHost);
  REQUIRE_MSG(plain == data, "[delta/%s] plain decode != original input", dtype.c_str());

  const std::vector<std::uint32_t> mask = make_host_mask(n, nc, 0x5EED, 37, /*zero_chunk=*/1);
  std::vector<std::uint32_t> chunk_offsets;
  const std::int64_t survivors = host_cnt(mask, nc, chunk_offsets);

  selection_mask sm;
  sm.words = reinterpret_cast<std::uint32_t*>(
    enc.upload_bytes(mask.data(), mask.size() * 4));
  sm.num_rows       = n;
  sm.chunk_offsets  = reinterpret_cast<std::uint32_t*>(
    enc.upload_bytes(chunk_offsets.data(), chunk_offsets.size() * 4));
  sm.survivor_count = survivors;

  CUdeviceptr d_out =
    enc.alloc(static_cast<std::size_t>(survivors > 0 ? survivors : 1) * sizeof(Element));
  REQUIRE_MSG(simpatico::launch_decode_fused_tree_mask_consume(
                *tree, enc.buffers, dtype.c_str(), n, sm, reinterpret_cast<void*>(d_out), stream),
              "[delta/%s] mask_consume launch failed",
              dtype.c_str());

  std::vector<Element> expect;
  expect.reserve(static_cast<std::size_t>(survivors));
  for (std::int64_t r = 0; r < n; ++r)
    if ((mask[static_cast<std::size_t>(r) / 32] >> (r % 32)) & 1u)
      expect.push_back(data[static_cast<std::size_t>(r)]);

  std::vector<Element> got(static_cast<std::size_t>(survivors));
  if (survivors > 0) {
    cudaMemcpy(got.data(), reinterpret_cast<const void*>(d_out),
               got.size() * sizeof(Element), cudaMemcpyDeviceToHost);
  }
  REQUIRE_MSG(got == expect,
              "[delta/%s] K3-delta compacted output != plain decode + host filter (%lld "
              "survivors)",
              dtype.c_str(),
              static_cast<long long>(survivors));
  std::printf("PASS: delta/%s (survivors=%lld of %lld)\n",
              dtype.c_str(),
              static_cast<long long>(survivors),
              static_cast<long long>(n));
  return true;
}

// K5: masked constant-width dictionary gather.  Codes are bitpacked int32
// (q1's l_returnflag/l_linestatus are width-1, 2-3 keys); the kernel must
// copy exactly the survivors' key bytes, compacted, in row order.
bool run_dict_gather(std::int32_t key_width, int arch)
{
  const std::int64_t n  = 3 * kChunk + 511;
  const std::int64_t nc = codegen::num_chunks_for(n);
  const std::int32_t num_keys = 3;
  const rmm::cuda_stream_view stream{};

  std::vector<std::int32_t> codes(static_cast<std::size_t>(n));
  for (std::int64_t i = 0; i < n; ++i) {
    std::uint64_t s = 0xD1C7ull ^ (0x9E3779B97F4A7C15ull * static_cast<std::uint64_t>(i + 1));
    // Chunk 1 constant (bits==0 short-circuit on the code leaf).
    codes[static_cast<std::size_t>(i)] =
      (i / kChunk == 1) ? 2 : static_cast<std::int32_t>(splitmix64(s) % num_keys);
  }

  auto tree      = jit::FusedTree::make(OpKind::Bitpack);
  GpuEncoded enc = codegen_test::gpu_encode_tree<std::int32_t>(
    *tree, "int32_t", codes.data(), n, arch);
  if (!compact_packed_on_host(enc, nc)) return false;

  // Key pool: key k = key_width copies of ('A' + k), so any wrong
  // code/rank/width shows up as a byte mismatch.
  std::vector<char> keys(static_cast<std::size_t>(num_keys) * key_width);
  for (std::int32_t k = 0; k < num_keys; ++k)
    for (std::int32_t b = 0; b < key_width; ++b)
      keys[static_cast<std::size_t>(k) * key_width + b] = static_cast<char>('A' + k);
  CUdeviceptr d_keys = enc.upload_bytes(keys.data(), keys.size());

  const std::vector<std::uint32_t> mask = make_host_mask(n, nc, 0xD1C7, 42, /*zero_chunk=*/2);
  std::vector<std::uint32_t> chunk_offsets;
  const std::int64_t survivors = host_cnt(mask, nc, chunk_offsets);

  selection_mask sm;
  sm.words          = reinterpret_cast<std::uint32_t*>(
    enc.upload_bytes(mask.data(), mask.size() * 4));
  sm.num_rows       = n;
  sm.chunk_offsets  = reinterpret_cast<std::uint32_t*>(
    enc.upload_bytes(chunk_offsets.data(), chunk_offsets.size() * 4));
  sm.survivor_count = survivors;

  const std::size_t out_bytes =
    static_cast<std::size_t>(survivors > 0 ? survivors : 1) * key_width;
  CUdeviceptr d_out = enc.alloc(out_bytes);
  cudaMemset(reinterpret_cast<void*>(d_out), 0xEE, out_bytes);

  REQUIRE_MSG(simpatico::launch_decode_fused_tree_mask_dict_gather(
                *tree,
                enc.buffers,
                "int32_t",
                n,
                sm,
                reinterpret_cast<const void*>(d_keys),
                key_width,
                reinterpret_cast<void*>(d_out),
                stream),
              "[dict/w%d] mask_dict_gather launch failed",
              key_width);

  std::vector<char> expect;
  expect.reserve(static_cast<std::size_t>(survivors) * key_width);
  for (std::int64_t r = 0; r < n; ++r) {
    if ((mask[static_cast<std::size_t>(r) / 32] >> (r % 32)) & 1u) {
      const std::int32_t code = codes[static_cast<std::size_t>(r)];
      expect.insert(expect.end(),
                    keys.begin() + static_cast<std::size_t>(code) * key_width,
                    keys.begin() + static_cast<std::size_t>(code + 1) * key_width);
    }
  }

  std::vector<char> got(static_cast<std::size_t>(survivors) * key_width);
  if (!got.empty()) {
    cudaMemcpy(got.data(), reinterpret_cast<const void*>(d_out), got.size(),
               cudaMemcpyDeviceToHost);
  }
  REQUIRE_MSG(got == expect,
              "[dict/w%d] K5 gathered chars != host reference (%lld survivors)",
              key_width,
              static_cast<long long>(survivors));
  std::printf("PASS: dict/w%d (survivors=%lld of %lld)\n",
              key_width,
              static_cast<long long>(survivors),
              static_cast<long long>(n));
  return true;
}

// K6: str_split masked survivor meta + fixed char copy.  `deep` = c_phone
// shape (offsets->delta->rle->bitpack, constant length 15); shallow =
// l_shipmode shape (offsets->bitpack, variable lengths 3..12).
bool run_str_split_masked(bool deep, int arch)
{
  const std::int64_t n     = 4 * kChunk + 300;  // string rows
  const std::int64_t n_off = n + 1;             // offsets elements
  const std::int64_t nc    = codegen::num_chunks_for(n);      // row chunks
  const rmm::cuda_stream_view stream{};
  const char* tag = deep ? "str-deep" : "str-shallow";

  // Host strings: offsets cumulative; chars pseudo-random bytes.
  std::vector<std::int32_t> offsets(static_cast<std::size_t>(n_off));
  offsets[0] = 0;
  for (std::int64_t r = 0; r < n; ++r) {
    std::uint64_t s = 0x57Ull ^ (0x9E3779B97F4A7C15ull * static_cast<std::uint64_t>(r + 1));
    const std::int32_t len =
      deep ? 15 : static_cast<std::int32_t>(3 + splitmix64(s) % 10ull);
    offsets[static_cast<std::size_t>(r) + 1] = offsets[static_cast<std::size_t>(r)] + len;
  }
  const std::int64_t chars_bytes = offsets[static_cast<std::size_t>(n)];
  std::vector<char> chars(static_cast<std::size_t>(chars_bytes));
  for (std::int64_t b = 0; b < chars_bytes; ++b) {
    std::uint64_t s = 0xC4A5ull ^ (0xD1B54A32D192ED03ull * static_cast<std::uint64_t>(b + 1));
    chars[static_cast<std::size_t>(b)] = static_cast<char>('a' + splitmix64(s) % 26ull);
  }

  auto tree = deep
                ? jit::FusedTree::make(
                    OpKind::Delta,
                    {{"differences",
                      jit::FusedTree::make(OpKind::Rle,
                                           {{"runs", jit::FusedTree::make(OpKind::Bitpack)},
                                            {"values", jit::FusedTree::make(OpKind::Bitpack)}})}})
                : jit::FusedTree::make(OpKind::Bitpack);

  GpuEncoded enc =
    codegen_test::gpu_encode_tree<std::int32_t>(*tree, "int32_t", offsets.data(), n_off, arch);
  const std::int64_t nc_off = codegen::num_chunks_for(n_off);
  if (deep) {
    // Preorder: delta=0 -> rle=1 -> {runs=2, values=3} (lex order).
    if (!compact_packed_on_host(enc, nc_off, /*node_id=*/2)) return false;
    if (!compact_packed_on_host(enc, nc_off, /*node_id=*/3)) return false;
  } else {
    if (!compact_packed_on_host(enc, nc_off, /*node_id=*/0)) return false;
  }

  // Plain offsets decode through the production launcher = cascade sanity.
  CUdeviceptr d_plain = enc.alloc(static_cast<std::size_t>(n_off) * sizeof(std::int32_t));
  REQUIRE_MSG(simpatico::launch_decode_fused_tree(
                *tree, enc.buffers, "int32_t", n_off, reinterpret_cast<void*>(d_plain), stream),
              "[%s] plain offsets decode failed",
              tag);
  std::vector<std::int32_t> plain(static_cast<std::size_t>(n_off));
  cudaMemcpy(plain.data(),
             reinterpret_cast<const void*>(d_plain),
             plain.size() * sizeof(std::int32_t),
             cudaMemcpyDeviceToHost);
  REQUIRE_MSG(plain == offsets, "[%s] plain offsets decode != input", tag);

  // Row-space mask; force the chunk-boundary rows of chunks 0 and 2 on so
  // the next-chunk first-offset peek is exercised (chunk 1 is all-zero).
  std::vector<std::uint32_t> mask = make_host_mask(n, nc, 0x57A7, 40, /*zero_chunk=*/1);
  mask[(0 * kChunk + 1023) / 32] |= (1u << ((0 * kChunk + 1023) % 32));
  mask[(2 * kChunk + 1023) / 32] |= (1u << ((2 * kChunk + 1023) % 32));
  std::vector<std::uint32_t> chunk_offsets;
  const std::int64_t survivors = host_cnt(mask, nc, chunk_offsets);

  selection_mask sm;
  sm.words          = reinterpret_cast<std::uint32_t*>(
    enc.upload_bytes(mask.data(), mask.size() * 4));
  sm.num_rows       = n;
  sm.chunk_offsets  = reinterpret_cast<std::uint32_t*>(
    enc.upload_bytes(chunk_offsets.data(), chunk_offsets.size() * 4));
  sm.survivor_count = survivors;

  CUdeviceptr d_src = enc.alloc(static_cast<std::size_t>(survivors) * sizeof(std::int64_t));
  CUdeviceptr d_len = enc.alloc(static_cast<std::size_t>(survivors) * sizeof(std::int32_t));
  REQUIRE_MSG(simpatico::launch_decode_fused_tree_str_split_meta(
                *tree,
                enc.buffers,
                "int32_t",
                n,
                sm,
                reinterpret_cast<std::int64_t*>(d_src),
                reinterpret_cast<std::int32_t*>(d_len),
                stream),
              "[%s] str_split_meta launch failed",
              tag);

  std::vector<std::int64_t> src_got(static_cast<std::size_t>(survivors));
  std::vector<std::int32_t> len_got(static_cast<std::size_t>(survivors));
  cudaMemcpy(src_got.data(), reinterpret_cast<const void*>(d_src), src_got.size() * 8,
             cudaMemcpyDeviceToHost);
  cudaMemcpy(len_got.data(), reinterpret_cast<const void*>(d_len), len_got.size() * 4,
             cudaMemcpyDeviceToHost);

  std::vector<std::int64_t> src_ref;
  std::vector<std::int32_t> len_ref;
  std::vector<std::int32_t> out_off(1, 0);
  std::vector<char> chars_ref;
  for (std::int64_t r = 0; r < n; ++r) {
    if ((mask[static_cast<std::size_t>(r) / 32] >> (r % 32)) & 1u) {
      const std::int32_t o0 = offsets[static_cast<std::size_t>(r)];
      const std::int32_t o1 = offsets[static_cast<std::size_t>(r) + 1];
      src_ref.push_back(o0);
      len_ref.push_back(o1 - o0);
      out_off.push_back(out_off.back() + (o1 - o0));
      chars_ref.insert(chars_ref.end(), chars.begin() + o0, chars.begin() + o1);
    }
  }
  REQUIRE_MSG(src_got == src_ref && len_got == len_ref,
              "[%s] K6 meta (src offsets / lengths) != host reference (%lld survivors)",
              tag,
              static_cast<long long>(survivors));

  // Phase 2: fixed char copy against the host gather.
  CUdeviceptr d_chars   = enc.upload_bytes(chars.data(), chars.size());
  CUdeviceptr d_out_off = enc.upload_bytes(out_off.data(), out_off.size() * 4);
  const std::size_t out_bytes = chars_ref.empty() ? 1 : chars_ref.size();
  CUdeviceptr d_out_chars     = enc.alloc(out_bytes);
  cudaMemset(reinterpret_cast<void*>(d_out_chars), 0xEE, out_bytes);
  REQUIRE_MSG(simpatico::launch_masked_char_copy(reinterpret_cast<const void*>(d_chars),
                                                 reinterpret_cast<const std::int64_t*>(d_src),
                                                 reinterpret_cast<const std::int32_t*>(d_out_off),
                                                 survivors,
                                                 reinterpret_cast<void*>(d_out_chars),
                                                 stream),
              "[%s] masked_char_copy launch failed",
              tag);
  std::vector<char> chars_got(chars_ref.size());
  if (!chars_got.empty()) {
    cudaMemcpy(chars_got.data(), reinterpret_cast<const void*>(d_out_chars), chars_got.size(),
               cudaMemcpyDeviceToHost);
  }
  REQUIRE_MSG(chars_got == chars_ref,
              "[%s] K6 gathered chars != host reference (%zu bytes)",
              tag,
              chars_ref.size());
  std::printf("PASS: %s (survivors=%lld of %lld, %zu chars)\n",
              tag,
              static_cast<long long>(survivors),
              static_cast<long long>(n),
              chars_ref.size());
  return true;
}

// K1m2: two-column pair predicate truth table (diffs in {-1,0,+1} so the
// equal-values boundary is dense), plus a constant-range AND and the
// chunk-geometry-mismatch rejection.
bool run_pair_mask(int arch)
{
  const std::int64_t n  = 3 * kChunk + 257;
  const std::int64_t nc = codegen::num_chunks_for(n);
  const rmm::cuda_stream_view stream{};

  std::vector<std::int32_t> a(static_cast<std::size_t>(n));
  std::vector<std::int32_t> b(static_cast<std::size_t>(n));
  for (std::int64_t i = 0; i < n; ++i) {
    std::uint64_t s = 0xAB12ull ^ (0x9E3779B97F4A7C15ull * static_cast<std::uint64_t>(i + 1));
    a[static_cast<std::size_t>(i)] = 8035 + static_cast<std::int32_t>(splitmix64(s) % 2526ull);
    b[static_cast<std::size_t>(i)] =
      a[static_cast<std::size_t>(i)] + static_cast<std::int32_t>(splitmix64(s) % 3ull) - 1;
  }

  auto tree_a = jit::FusedTree::make(OpKind::Bitpack);
  auto tree_b = jit::FusedTree::make(OpKind::Bitpack);
  GpuEncoded enc_a =
    codegen_test::gpu_encode_tree<std::int32_t>(*tree_a, "int32_t", a.data(), n, arch);
  GpuEncoded enc_b =
    codegen_test::gpu_encode_tree<std::int32_t>(*tree_b, "int32_t", b.data(), n, arch);
  if (!compact_packed_on_host(enc_a, nc)) return false;
  if (!compact_packed_on_host(enc_b, nc)) return false;

  const std::size_t nwords = static_cast<std::size_t>(nc) * kWordsPerChunk;
  const range_predicate no_range{INT64_MIN, INT64_MAX};
  const range_predicate a_range{8035 + 100, 8035 + 1200};

  struct Case {
    simpatico::pair_cmp op;
    const char* name;
    bool ranged;
  };
  const Case cases[] = {
    {simpatico::pair_cmp::lt, "lt", false},
    {simpatico::pair_cmp::le, "le", false},
    {simpatico::pair_cmp::gt, "gt", false},
    {simpatico::pair_cmp::ge, "ge", false},
    {simpatico::pair_cmp::lt, "lt+range_a", true},
  };
  for (const Case& c : cases) {
    CUdeviceptr d_mask = enc_a.alloc(nwords * 4);
    cudaMemset(reinterpret_cast<void*>(d_mask), 0xFF, nwords * 4);
    selection_mask sm;
    sm.words    = reinterpret_cast<std::uint32_t*>(d_mask);
    sm.num_rows = n;
    REQUIRE_MSG(simpatico::launch_decode_fused_tree_pair_mask_out(*tree_a,
                                                                  enc_a.buffers,
                                                                  "int32_t",
                                                                  *tree_b,
                                                                  enc_b.buffers,
                                                                  "int32_t",
                                                                  n,
                                                                  c.op,
                                                                  sm,
                                                                  stream,
                                                                  c.ranged ? a_range : no_range,
                                                                  no_range),
                "[pair/%s] launch failed",
                c.name);
    std::vector<std::uint32_t> got(nwords);
    cudaMemcpy(got.data(), reinterpret_cast<const void*>(d_mask), nwords * 4,
               cudaMemcpyDeviceToHost);

    std::vector<std::uint32_t> ref(nwords, 0u);
    for (std::int64_t r = 0; r < n; ++r) {
      const std::int64_t av = a[static_cast<std::size_t>(r)];
      const std::int64_t bv = b[static_cast<std::size_t>(r)];
      bool pass = (c.op == simpatico::pair_cmp::lt)   ? (av < bv)
                  : (c.op == simpatico::pair_cmp::le) ? (av <= bv)
                  : (c.op == simpatico::pair_cmp::gt) ? (av > bv)
                                                      : (av >= bv);
      if (c.ranged) pass = pass && av >= a_range.lo && av <= a_range.hi;
      if (pass) ref[static_cast<std::size_t>(r) / 32] |= (1u << (r % 32));
    }
    REQUIRE_MSG(got == ref, "[pair/%s] K1m2 mask != host truth table", c.name);
    std::printf("PASS: pair/%s\n", c.name);
  }

  // Chunk-geometry mismatch must be rejected before any launch.
  {
    jit::LabeledBuffers bad = enc_b.buffers;
    auto it                 = bad.find(jit::buffer_key(0, "chunk_count"));
    REQUIRE_MSG(it != bad.end(), "[pair] missing chunk_count in copy");
    it->second.length -= 1;
    CUdeviceptr d_mask = enc_a.alloc(nwords * 4);
    selection_mask sm;
    sm.words    = reinterpret_cast<std::uint32_t*>(d_mask);
    sm.num_rows = n;
    REQUIRE_MSG(!simpatico::launch_decode_fused_tree_pair_mask_out(*tree_a,
                                                                   enc_a.buffers,
                                                                   "int32_t",
                                                                   *tree_b,
                                                                   bad,
                                                                   "int32_t",
                                                                   n,
                                                                   simpatico::pair_cmp::lt,
                                                                   sm,
                                                                   stream),
                "[pair] chunk-geometry mismatch must be rejected");
    std::printf("PASS: pair/geometry-mismatch-rejected\n");
  }
  return true;
}

// ---------------------------------------------------------------------------
// --bench mode: time the PLAIN decode launcher (the production path, not the
// masked variants) per shape, against the memset write floor of the output
// buffer.  Measurement only — separates launch/host overhead (call vs kernel
// time) from kernel bandwidth so decode shapes can be classified as
// write-floor-bound, launch-bound, or kernel-bound.
// CSV: shape,n,kernel_ms,call_ms,memset_ms,kernel_GBps,call_GBps
//   kernel_ms = median cudaEvent bracket around the launcher call (GPU work:
//               bp_offsets transient scan + decode kernel),
//   call_ms   = median chrono wall of the full call (adds render/cache hash,
//               transient allocs, internal sync),
//   memset_ms = median event-timed cudaMemsetAsync of the output bytes.
// ---------------------------------------------------------------------------

double median_of(std::vector<double> v)
{
  std::sort(v.begin(), v.end());
  return v.empty() ? 0.0 : v[v.size() / 2];
}

// Per-chunk residual with exact bit width `b`: positions 0/1 of every chunk
// pin the max/min so chunk_bits == b deterministically.
std::int64_t bench_residual(std::int64_t i, int b)
{
  const std::uint64_t m = (std::uint64_t{1} << b) - 1;
  const std::int64_t p  = i & (kChunk - 1);
  if (p == 0) return static_cast<std::int64_t>(m);
  if (p == 1) return 0;
  return static_cast<std::int64_t>(
    (static_cast<std::uint64_t>(i) * 0x9E3779B97F4A7C15ull >> 13) & m);
}

template <typename Element>
bool bench_one(const char* shape, std::int64_t n, int bits, bool delta_shape, int arch)
{
  const rmm::cuda_stream_view stream{};
  const std::int64_t nc          = codegen::num_chunks_for(n);
  const std::size_t output_bytes = static_cast<std::size_t>(n) * sizeof(Element);

  std::vector<Element> data(static_cast<std::size_t>(n));
  if (delta_shape) {
    std::int64_t v = 1000;
    for (std::int64_t i = 0; i < n; ++i) {
      v += bench_residual(i, bits);
      data[static_cast<std::size_t>(i)] = static_cast<Element>(v);
    }
  } else {
    const std::int64_t base = (sizeof(Element) == 8) ? 3'000'000'000LL : 8035;
    for (std::int64_t i = 0; i < n; ++i)
      data[static_cast<std::size_t>(i)] = static_cast<Element>(base + bench_residual(i, bits));
  }

  auto tree = delta_shape
                ? jit::FusedTree::make(OpKind::Delta,
                                       {{"differences", jit::FusedTree::make(OpKind::Bitpack)}})
                : jit::FusedTree::make(OpKind::Bitpack);
  const std::string dtype = (sizeof(Element) == 8) ? "int64_t" : "int32_t";

  GpuEncoded enc = codegen_test::gpu_encode_tree<Element>(*tree, dtype, data.data(), n, arch);
  if (!compact_packed_on_host(enc, nc, /*node_id=*/delta_shape ? 1 : 0)) return false;

  CUdeviceptr d_out = enc.alloc(output_bytes);

  cudaEvent_t e0 = nullptr, e1 = nullptr;
  cudaEventCreate(&e0);
  cudaEventCreate(&e1);

  // Warmups (first one pays any cold NVRTC compile) + a spot check that the
  // bench measures a CORRECT decode.
  for (int w = 0; w < 3; ++w) {
    REQUIRE_MSG(simpatico::launch_decode_fused_tree(
                  *tree, enc.buffers, dtype.c_str(), n, reinterpret_cast<void*>(d_out), stream),
                "[bench %s n=%lld] warmup decode failed",
                shape,
                static_cast<long long>(n));
  }
  {
    const std::int64_t checks[] = {0, (n / 2) & ~std::int64_t{1023}, n - 1024};
    std::vector<Element> win(1024);
    for (std::int64_t off : checks) {
      cudaMemcpy(win.data(),
                 reinterpret_cast<const Element*>(d_out) + off,
                 win.size() * sizeof(Element),
                 cudaMemcpyDeviceToHost);
      for (std::size_t k = 0; k < win.size(); ++k) {
        REQUIRE_MSG(win[k] == data[static_cast<std::size_t>(off) + k],
                    "[bench %s n=%lld] spot check mismatch at row %lld",
                    shape,
                    static_cast<long long>(n),
                    static_cast<long long>(off + static_cast<std::int64_t>(k)));
      }
    }
  }

  constexpr int kReps = 20;
  std::vector<double> kernel_ms, call_ms, memset_ms;
  kernel_ms.reserve(kReps);
  call_ms.reserve(kReps);
  memset_ms.reserve(kReps);
  for (int r = 0; r < kReps; ++r) {
    cudaEventRecord(e0, stream.value());
    const auto t0 = std::chrono::steady_clock::now();
    const bool ok = simpatico::launch_decode_fused_tree(
      *tree, enc.buffers, dtype.c_str(), n, reinterpret_cast<void*>(d_out), stream);
    const auto t1 = std::chrono::steady_clock::now();
    cudaEventRecord(e1, stream.value());
    cudaEventSynchronize(e1);
    REQUIRE_MSG(ok, "[bench %s n=%lld] rep decode failed", shape, static_cast<long long>(n));
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, e0, e1);
    kernel_ms.push_back(ms);
    call_ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
  }
  for (int r = 0; r < kReps; ++r) {
    cudaEventRecord(e0, stream.value());
    cudaMemsetAsync(reinterpret_cast<void*>(d_out), 0, output_bytes, stream.value());
    cudaEventRecord(e1, stream.value());
    cudaEventSynchronize(e1);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, e0, e1);
    memset_ms.push_back(ms);
  }
  cudaEventDestroy(e0);
  cudaEventDestroy(e1);

  const double k_ms = median_of(kernel_ms);
  const double c_ms = median_of(call_ms);
  const double m_ms = median_of(memset_ms);
  const double gb   = static_cast<double>(output_bytes) / 1e9;
  std::printf("%s,%lld,%.4f,%.4f,%.4f,%.1f,%.1f\n",
              shape,
              static_cast<long long>(n),
              k_ms,
              c_ms,
              m_ms,
              k_ms > 0 ? gb / (k_ms * 1e-3) : 0.0,
              c_ms > 0 ? gb / (c_ms * 1e-3) : 0.0);
  std::fflush(stdout);
  return true;
}

int run_bench(int arch)
{
  std::printf("shape,n,kernel_ms,call_ms,memset_ms,kernel_GBps,call_GBps\n");
  const std::int64_t sizes[] = {170205184LL, std::int64_t{1} << 29};
  for (std::int64_t n : sizes) {
    bench_one<std::int64_t>("i64_13b", n, 13, /*delta=*/false, arch);
    bench_one<std::int64_t>("i64_4b", n, 4, /*delta=*/false, arch);
    bench_one<std::int64_t>("i64_24b", n, 24, /*delta=*/false, arch);
    bench_one<std::int32_t>("i32_12b", n, 12, /*delta=*/false, arch);
    bench_one<std::int32_t>("i32_2b", n, 2, /*delta=*/false, arch);
    bench_one<std::int64_t>("i64_delta_13b", n, 13, /*delta=*/true, arch);
  }
  return g_failures > 0 ? 1 : 0;
}

// Render-level contract checks (no kernel launches).
bool render_checks()
{
  auto bp = jit::FusedTree::make(OpKind::Bitpack);

  // 1. plain is byte-identical with and without the variant argument.
  const cdj::DecodeKernelSpec s3 = cdj::render(*bp, "int64_t", 8);
  const cdj::DecodeKernelSpec s4 = cdj::render(*bp, "int64_t", 8, cdj::DecodeVariant::plain);
  REQUIRE_MSG(s3.source == s4.source && s3.entry_symbol == s4.entry_symbol,
              "plain render changed by the variant default argument");

  // 2. mask variants: distinct symbols + sources, same input channels.
  const cdj::DecodeKernelSpec k1 = cdj::render(*bp, "int64_t", 8, cdj::DecodeVariant::mask_out);
  const cdj::DecodeKernelSpec k3 = cdj::render(*bp, "int64_t", 8, cdj::DecodeVariant::mask_consume);
  REQUIRE_MSG(k1.entry_symbol == s3.entry_symbol + "_mask_out",
              "mask_out entry symbol suffix wrong: %s",
              k1.entry_symbol.c_str());
  REQUIRE_MSG(k3.entry_symbol == s3.entry_symbol + "_mask_consume",
              "mask_consume entry symbol suffix wrong: %s",
              k3.entry_symbol.c_str());
  REQUIRE_MSG(k1.source != s3.source && k3.source != s3.source && k1.source != k3.source,
              "variant sources must be distinct (JIT-cache separation)");
  REQUIRE_MSG(k1.buffers.size() == s3.buffers.size() && k3.buffers.size() == s3.buffers.size(),
              "variants must not change the input-channel manifest");
  // Predicate constants must be parameters, not literals baked into source.
  REQUIRE_MSG(k1.source.find("pred_lo") != std::string::npos &&
                k1.source.find("pred_hi") != std::string::npos,
              "mask_out source must take pred_lo/pred_hi kernel params");

  // 3. Delta root: mask_out stays rejected (filter columns are bitpack);
  //    mask_consume renders (K3-delta) with its own symbol + source.
  auto delta = jit::FusedTree::make(
    OpKind::Delta, {{"differences", jit::FusedTree::make(OpKind::Bitpack)}});
  bool rejected = false;
  try {
    (void)cdj::render(*delta, "int64_t", 8, cdj::DecodeVariant::mask_out);
  } catch (const cdj::RenderError&) {
    rejected = true;
  }
  REQUIRE_MSG(rejected, "mask_out on a Delta root must throw RenderError");
  const cdj::DecodeKernelSpec dplain = cdj::render(*delta, "int64_t", 8);
  const cdj::DecodeKernelSpec dmask =
    cdj::render(*delta, "int64_t", 8, cdj::DecodeVariant::mask_consume);
  REQUIRE_MSG(dmask.entry_symbol == dplain.entry_symbol + "_mask_consume",
              "K3-delta entry symbol suffix wrong: %s",
              dmask.entry_symbol.c_str());
  REQUIRE_MSG(dmask.source != dplain.source && dmask.buffers.size() == dplain.buffers.size(),
              "K3-delta must differ in source only, not in the channel manifest");

  // 4. K5 dict gather: bitpack code leaf only; distinct symbol; key pool +
  //    width are kernel params (never literals in the source).
  const cdj::DecodeKernelSpec k5 = cdj::render(*bp, "int32_t", 8, cdj::DecodeVariant::mask_dict_gather);
  const cdj::DecodeKernelSpec p32 = cdj::render(*bp, "int32_t", 8);
  REQUIRE_MSG(k5.entry_symbol == p32.entry_symbol + "_mask_dict",
              "K5 entry symbol suffix wrong: %s",
              k5.entry_symbol.c_str());
  REQUIRE_MSG(k5.source.find("keys_chars") != std::string::npos &&
                k5.source.find("key_width") != std::string::npos,
              "K5 source must take keys_chars/key_width kernel params");
  REQUIRE_MSG(k5.buffers.size() == p32.buffers.size(),
              "K5 must not change the input-channel manifest");
  rejected = false;
  try {
    (void)cdj::render(*delta, "int64_t", 8, cdj::DecodeVariant::mask_dict_gather);
  } catch (const cdj::RenderError&) {
    rejected = true;
  }
  REQUIRE_MSG(rejected, "mask_dict_gather on a Delta root must throw RenderError");

  // 5. K4 index_consume: bitpack leaf only; own symbol; index list + offsets
  //    are kernel params; delta roots rejected (fall back to K3-delta).
  const cdj::DecodeKernelSpec k4 = cdj::render(*bp, "int64_t", 8, cdj::DecodeVariant::index_consume);
  REQUIRE_MSG(k4.entry_symbol == s3.entry_symbol + "_index_consume",
              "K4 entry symbol suffix wrong: %s",
              k4.entry_symbol.c_str());
  REQUIRE_MSG(k4.source.find("row_indices") != std::string::npos &&
                k4.source.find("chunk_offsets") != std::string::npos,
              "K4 source must take row_indices/chunk_offsets kernel params");
  REQUIRE_MSG(k4.buffers.size() == s3.buffers.size(),
              "K4 must not change the input-channel manifest");
  REQUIRE_MSG(k4.source != k3.source && k4.source != s3.source,
              "K4 source must be distinct (JIT-cache separation)");
  rejected = false;
  try {
    (void)cdj::render(*delta, "int64_t", 8, cdj::DecodeVariant::index_consume);
  } catch (const cdj::RenderError&) {
    rejected = true;
  }
  REQUIRE_MSG(rejected, "index_consume on a Delta root must throw RenderError");

  // 6. K6 str_split_meta: bitpack + delta roots render (with the next-chunk
  //    peek), rle root rejected; params not literals.
  const cdj::DecodeKernelSpec k6 = cdj::render(*bp, "int32_t", 8, cdj::DecodeVariant::str_split_meta);
  REQUIRE_MSG(k6.entry_symbol == p32.entry_symbol + "_str_meta",
              "K6 entry symbol suffix wrong: %s",
              k6.entry_symbol.c_str());
  REQUIRE_MSG(k6.source.find("len_out") != std::string::npos &&
                k6.source.find("next0") != std::string::npos,
              "K6 source must take len_out and emit the next-chunk peek");
  auto deep_offsets = jit::FusedTree::make(
    OpKind::Delta,
    {{"differences",
      jit::FusedTree::make(OpKind::Rle,
                           {{"runs", jit::FusedTree::make(OpKind::Bitpack)},
                            {"values", jit::FusedTree::make(OpKind::Bitpack)}})}});
  const cdj::DecodeKernelSpec k6d =
    cdj::render(*deep_offsets, "int32_t", 8, cdj::DecodeVariant::str_split_meta);
  REQUIRE_MSG(k6d.source.find("delta_first_0[chunk_id + 1]") != std::string::npos,
              "K6 delta root must peek delta_first[chunk_id+1]");
  rejected = false;
  try {
    auto rle_root = jit::FusedTree::make(OpKind::Rle,
                                         {{"runs", jit::FusedTree::make(OpKind::Bitpack)},
                                          {"values", jit::FusedTree::make(OpKind::Bitpack)}});
    (void)cdj::render(*rle_root, "int32_t", 8, cdj::DecodeVariant::str_split_meta);
  } catch (const cdj::RenderError&) {
    rejected = true;
  }
  REQUIRE_MSG(rejected, "str_split_meta on an Rle root must throw RenderError");

  // 7. Compositional mask_consume: a FOR-rooted cascade renders through the
  //    generic value_source seam (no hand-written variant needed).
  auto for_tree = jit::FusedTree::make(
    OpKind::For, {{"deltas", jit::FusedTree::make(OpKind::Bitpack)}});
  const cdj::DecodeKernelSpec kfor =
    cdj::render(*for_tree, "int64_t", 8, cdj::DecodeVariant::mask_consume);
  REQUIRE_MSG(kfor.entry_symbol.find("_mask_consume") != std::string::npos &&
                kfor.source.find("K3-generic") != std::string::npos,
              "FOR-rooted mask_consume must render via the generic seam");

  // 8. K1m2 pair mask: combined symbol, params-not-constants, both columns'
  //    channels distinct; self-pair and non-bitpack columns rejected.
  const cdj::DecodeKernelSpec kp = cdj::render_pair_mask(*bp, "int32_t", *for_tree->children.at("deltas"), "int32_t", 8);
  REQUIRE_MSG(kp.entry_symbol == "simpatico_decode_pair_mask_int32_t_int32_t",
              "pair mask entry symbol wrong: %s",
              kp.entry_symbol.c_str());
  REQUIRE_MSG(kp.source.find("cmp_op") != std::string::npos &&
                kp.source.find("lo_a") != std::string::npos &&
                kp.source.find("chunk_min_0") != std::string::npos &&
                kp.source.find("chunk_min_1") != std::string::npos,
              "pair mask must take op/bounds params and bind two node channels");
  REQUIRE_MSG(kp.buffers.size() == 2 * p32.buffers.size(),
              "pair mask manifest must carry both columns' channels");
  rejected = false;
  try {
    (void)cdj::render_pair_mask(*bp, "int32_t", *bp, "int32_t", 8);
  } catch (const cdj::RenderError&) {
    rejected = true;
  }
  REQUIRE_MSG(rejected, "pair mask with the same tree object twice must throw");
  rejected = false;
  try {
    (void)cdj::render_pair_mask(*delta, "int64_t", *bp, "int64_t", 8);
  } catch (const cdj::RenderError&) {
    rejected = true;
  }
  REQUIRE_MSG(rejected, "pair mask with a non-bitpack column must throw");

  std::printf("PASS: render contract checks\n");
  return true;
}

}  // namespace

int main(int argc, char** argv)
{
  if (cudaSetDevice(0) != cudaSuccess) {
    std::fprintf(stderr, "FAIL: cudaSetDevice(0) failed\n");
    return 1;
  }
  const int arch = jit::arch_cc_for_current_device();

  if (argc > 1 && std::string(argv[1]) == "--bench") {
    try {
      return run_bench(arch);
    } catch (const std::exception& e) {
      std::fprintf(stderr, "FAIL: bench: unhandled exception: %s\n", e.what());
      return 1;
    }
  }

  try {
    render_checks();
    // int32 date-like domain; int64 with a base above INT32_MAX so the widened
    // int64 compare path is exercised on genuinely 64-bit decoded values.
    run_roundtrip<std::int32_t>("int32_t", 8035, 2526, arch);
    run_roundtrip<std::int64_t>("int64_t", 3'000'000'000LL, 5052, arch);
    // Iteration 3: K3-delta (o_orderkey shape) and K5 dict gather (q1 shape).
    run_delta_masked<std::int64_t>("int64_t", arch);
    run_delta_masked<std::int32_t>("int32_t", arch);
    run_dict_gather(/*key_width=*/1, arch);  // l_returnflag / l_linestatus
    run_dict_gather(/*key_width=*/4, arch);  // constant-width generality
    // Iteration 5: K6 str_split gather (l_shipmode / c_phone shapes) and
    // K1m2 pair predicates (q12 shape).
    run_str_split_masked(/*deep=*/false, arch);
    run_str_split_masked(/*deep=*/true, arch);
    run_pair_mask(arch);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "FAIL: unhandled exception: %s\n", e.what());
    return 1;
  }

  if (g_failures > 0) {
    std::fprintf(stderr, "%d check(s) failed\n", g_failures);
    return 1;
  }
  std::printf("ALL PASS\n");
  return 0;
}
