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

    std::printf("PASS: %s/%s (survivors=%lld of %lld)\n",
                dtype.c_str(),
                pred_names[p],
                static_cast<long long>(survivors),
                static_cast<long long>(n));
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

  std::printf("PASS: render contract checks\n");
  return true;
}

}  // namespace

int main()
{
  if (cudaSetDevice(0) != cudaSuccess) {
    std::fprintf(stderr, "FAIL: cudaSetDevice(0) failed\n");
    return 1;
  }
  const int arch = jit::arch_cc_for_current_device();

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
