// Late-materialization sparse row decode (K8 / K5s + chunk-CSR row sets) —
// correctness roundtrip against plain decode + host-side gather.
//
// Pipeline under test (production entry points, not re-implementations):
//   gpu_encode_tree -> host compaction to the Compact packed layout ->
//   launch_decode_fused_tree (plain reference) ->
//   bucket_sorted_local_ids / row_set_to_local_indices / row_set_to_mask
//   (device CSR construction vs host reference) ->
//   launch_decode_fused_tree_sparse_rows (K8: bitpack random access AND the
//   delta staged-slab compositional path) ->
//   launch_decode_fused_tree_sparse_dict_gather (K5s) ->
//   sort_unique_global_ids (order-restoration ranks).
//
// Verified properties:
//   1. Sparse variants render with distinct entry symbols and a
//      chunk_list-derived chunk_id; the plain render remains byte-free of
//      any sparse artifact (JIT-cache separation).
//   2. Device chunk-CSR construction (bucket_sorted_local_ids) matches a
//      host-built CSR exactly: touched list, offsets, u16 positions; the
//      int32 expansion and mask expansion match host references bit-for-bit
//      (mask buffers pre-filled 0xFF to prove tail/untouched zeroing).
//   3. K8 compacted output == plain decode + host gather, ascending row
//      order, at ~0.1% / ~2% / ~15% densities plus an untouched-chunk-heavy
//      clustered set; bitpack (random access) and delta->bitpack (staged
//      slab) roots; a constant (bits==0) chunk is exercised by the data gen.
//   4. K5s chars == host key gather for constant-width keys.
//   5. sort_unique_global_ids: sorted-unique output + restore ranks
//      reproduce the original (dup-carrying, shuffled) list.
//
// K6s (sparse_str_meta) gets render checks here; its GPU roundtrip rides the
// str_split encode harness and is queued as a follow-up GPU job.
//
// GPU required (encode/decode kernels + NVRTC). Same standalone-main harness
// as the other tests in this directory.

#include "codegen/codegen_bridge.hpp"
#include "codegen/decode/jit/renderer.hpp"
#include "codegen/decode/latemat_launch.hpp"
#include "codegen/jit/fused_tree.hpp"
#include "codegen/jit/nvrtc_compiler.hpp"
#include "codegen/selection/row_set.hpp"
#include "codegen/selection/selection.hpp"
#include "codegen/selection/selection_capture.hpp"
#include "gpu_encode.hpp"

// Fast-path coverage: the extension-side materializer is compiled into this
// test target (see the target_sources line in CMakeLists.txt).
#include "api/simpatico_codegen.hpp"
#include "late_mat/late_materializer.hpp"
#include "late_mat/rowid_emission.hpp"

#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/mr/per_device_resource.hpp>

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace jit = codegen::jit;
namespace cdj = codegen::decode::jit;
using codegen::OpKind;
using codegen_test::GpuEncoded;
using sirius::codegen::chunk_row_set;
using sirius::codegen::owned_chunk_row_set;
using sirius::codegen::selection_mask;

namespace {

constexpr std::int64_t kChunk = codegen::kChunkSize;  // 1024
constexpr int kWordsPerChunk  = static_cast<int>(kChunk / 32);

int g_failures = 0;

#define REQUIRE_MSG(cond, ...)                    \
  do {                                            \
    if (!(cond)) {                                \
      std::fprintf(stderr, "FAIL: " __VA_ARGS__); \
      std::fprintf(stderr, "\n");                 \
      ++g_failures;                               \
      return false;                               \
    }                                             \
  } while (0)

std::uint64_t splitmix64(std::uint64_t& s)
{
  s += 0x9E3779B97F4A7C15ull;
  std::uint64_t z = s;
  z               = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
  z               = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
  return z ^ (z >> 31);
}

// Same chunk-shaped data as test_masked_decode_variants: chunk 2 constant
// (bits==0 short-circuit).
template <typename Element>
std::vector<Element> gen_data(std::int64_t n, std::int64_t base, std::int64_t range)
{
  std::vector<Element> v(static_cast<std::size_t>(n));
  for (std::int64_t i = 0; i < n; ++i) {
    std::uint64_t s = 0xC0FFEEull ^ (0xD1B54A32D192ED03ull * static_cast<std::uint64_t>(i));
    const std::int64_t r =
      static_cast<std::int64_t>(splitmix64(s) % static_cast<std::uint64_t>(range));
    v[static_cast<std::size_t>(i)] =
      static_cast<Element>((i / kChunk == 2) ? base + range / 3 : base + r);
  }
  return v;
}

// Pseudo-random sorted-unique batch-local row-id sets. `zero_chunk` is fully
// excluded (untouched-chunk path); `cluster_runs` selects runs of 64
// consecutive rows instead of independent rows.
std::vector<std::uint32_t> gen_row_ids(
  std::int64_t n, unsigned keep_permille, unsigned seed, std::int64_t zero_chunk, bool cluster_runs)
{
  std::vector<std::uint32_t> ids;
  for (std::int64_t r = 0; r < n; ++r) {
    if (r / kChunk == zero_chunk) continue;
    std::uint64_t s =
      seed ^ (0x9E3779B97F4A7C15ull * static_cast<std::uint64_t>((cluster_runs ? r / 64 : r) + 1));
    if (splitmix64(s) % 1000ull < keep_permille) ids.push_back(static_cast<std::uint32_t>(r));
  }
  return ids;
}

// Host-reference chunk-CSR.
struct HostCsr {
  std::vector<std::uint32_t> chunks;
  std::vector<std::uint32_t> offsets;  // T+1
  std::vector<std::uint16_t> in_chunk;
};
HostCsr host_csr(std::vector<std::uint32_t> const& ids)
{
  HostCsr csr;
  for (std::size_t i = 0; i < ids.size(); ++i) {
    const std::uint32_t c = ids[i] >> 10;
    if (csr.chunks.empty() || csr.chunks.back() != c) {
      csr.chunks.push_back(c);
      csr.offsets.push_back(static_cast<std::uint32_t>(i));
    }
    csr.in_chunk.push_back(static_cast<std::uint16_t>(ids[i] & 1023u));
  }
  csr.offsets.push_back(static_cast<std::uint32_t>(ids.size()));
  return csr;
}

// Same host packed-layout compaction as test_masked_decode_variants.
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
                 codegen_test::device_ptr<std::uint8_t>(enc, node_id, "chunk_bits"),
                 bits.size(),
                 cudaMemcpyDeviceToHost) != cudaSuccess ||
      cudaMemcpy(count.data(),
                 codegen_test::device_ptr<std::int32_t>(enc, node_id, "chunk_count"),
                 count.size() * 4,
                 cudaMemcpyDeviceToHost) != cudaSuccess) {
    REQUIRE_MSG(false, "compact_packed_on_host: D2H failed");
  }

  std::vector<std::uint32_t> dense;
  dense.reserve(total_words + 3);
  for (std::int64_t c = 0; c < nc; ++c) {
    const std::size_t live = (static_cast<std::size_t>(count[static_cast<std::size_t>(c)]) *
                                bits[static_cast<std::size_t>(c)] +
                              31) /
                             32;
    const std::uint32_t* src = over.data() + static_cast<std::size_t>(c) * stride;
    dense.insert(dense.end(), src, src + live);
  }
  dense.insert(dense.end(), 3, 0u);

  CUdeviceptr d = enc.upload_bytes(dense.data(), dense.size() * 4);
  enc.buffers[key_packed] =
    jit::LabeledBuffer{reinterpret_cast<const void*>(d), dense.size(), sizeof(std::uint32_t)};
  return true;
}

// Upload a host CSR into device buffers owned by `enc`; returns a view.
chunk_row_set upload_csr(GpuEncoded& enc, HostCsr const& csr, std::int64_t n)
{
  chunk_row_set set;
  set.num_touched   = static_cast<std::int64_t>(csr.chunks.size());
  set.num_survivors = static_cast<std::int64_t>(csr.in_chunk.size());
  set.num_rows      = n;
  if (set.num_survivors == 0) { return set; }
  set.chunk_ids =
    reinterpret_cast<std::uint32_t*>(enc.upload_bytes(csr.chunks.data(), csr.chunks.size() * 4));
  set.chunk_out_offsets =
    reinterpret_cast<std::uint32_t*>(enc.upload_bytes(csr.offsets.data(), csr.offsets.size() * 4));
  set.in_chunk_offsets = reinterpret_cast<std::uint16_t*>(
    enc.upload_bytes(csr.in_chunk.data(), csr.in_chunk.size() * 2));
  return set;
}

// ── 1. Render contract ───────────────────────────────────────────────────────
bool render_checks()
{
  auto bp                           = jit::FusedTree::make(OpKind::Bitpack);
  const cdj::DecodeKernelSpec plain = cdj::render(*bp, "int32_t", 8);
  const cdj::DecodeKernelSpec k8    = cdj::render(*bp, "int32_t", 8, cdj::kShapeSparseConsume);
  REQUIRE_MSG(k8.entry_symbol != plain.entry_symbol &&
                k8.entry_symbol.find("_sparse_index") != std::string::npos,
              "sparse_index_consume must get its own entry symbol");
  REQUIRE_MSG(k8.source.find("chunk_list[blockIdx.x]") != std::string::npos,
              "sparse variant must derive chunk_id from chunk_list");
  REQUIRE_MSG(plain.source.find("chunk_list") == std::string::npos &&
                plain.source.find("in_chunk_offsets") == std::string::npos,
              "plain render must stay byte-free of sparse artifacts");

  // Compositional delta root through the same variant (staged slab).
  auto delta =
    jit::FusedTree::make(OpKind::Delta, {{"differences", jit::FusedTree::make(OpKind::Bitpack)}});
  const cdj::DecodeKernelSpec k8d = cdj::render(*delta, "int64_t", 8, cdj::kShapeSparseConsume);
  REQUIRE_MSG(k8d.source.find("K8") != std::string::npos,
              "delta-rooted sparse_index_consume must render (compositional)");

  const cdj::DecodeKernelSpec k5s = cdj::render(*bp, "int32_t", 8, cdj::kShapeSparseDictGather);
  REQUIRE_MSG(k5s.entry_symbol.find("_sparse_dict") != std::string::npos &&
                k5s.source.find("keys_chars") != std::string::npos,
              "sparse_dict_gather render contract");
  bool rejected = false;
  try {
    (void)cdj::render(*delta, "int64_t", 8, cdj::kShapeSparseDictGather);
  } catch (const cdj::RenderError&) {
    rejected = true;
  }
  REQUIRE_MSG(rejected, "sparse_dict_gather on a Delta root must throw RenderError");

  const cdj::DecodeKernelSpec k6s = cdj::render(*bp, "int64_t", 8, cdj::kShapeSparseStrMeta);
  REQUIRE_MSG(k6s.entry_symbol.find("_sparse_str_meta") != std::string::npos &&
                k6s.source.find("len_out") != std::string::npos &&
                k6s.source.find("chunk_list[blockIdx.x]") != std::string::npos,
              "sparse_str_meta render contract");
  rejected = false;
  try {
    auto rle = jit::FusedTree::make(OpKind::Rle,
                                    {{"values", jit::FusedTree::make(OpKind::Bitpack)},
                                     {"counts", jit::FusedTree::make(OpKind::Bitpack)}});
    (void)cdj::render(*rle, "int32_t", 8, cdj::kShapeSparseStrMeta);
  } catch (const cdj::RenderError&) {
    rejected = true;
  }
  REQUIRE_MSG(rejected, "sparse_str_meta on an Rle root must throw RenderError");

  std::printf("PASS: sparse render contract checks\n");
  return true;
}

// ── 2+3. Row-set construction + K8 roundtrip ────────────────────────────────
template <typename Element>
bool run_sparse_roundtrip(
  const std::string& dtype, std::int64_t base, std::int64_t range, bool delta_root, int arch)
{
  const std::int64_t n  = 7 * kChunk + 700;  // partial tail chunk
  const std::int64_t nc = codegen::num_chunks_for(n);
  const rmm::cuda_stream_view stream{};
  auto mr = rmm::mr::get_current_device_resource_ref();

  const std::vector<Element> data = gen_data<Element>(n, base, range);
  auto tree                       = delta_root
                                      ? jit::FusedTree::make(OpKind::Delta,
                                                             {{"differences", jit::FusedTree::make(OpKind::Bitpack)}})
                                      : jit::FusedTree::make(OpKind::Bitpack);

  GpuEncoded enc = codegen_test::gpu_encode_tree<Element>(*tree, dtype, data.data(), n, arch);
  const std::int32_t bp_node = delta_root ? 1 : 0;
  if (!compact_packed_on_host(enc, nc, bp_node)) return false;

  // Plain reference through the production launcher.
  CUdeviceptr d_plain = enc.alloc(static_cast<std::size_t>(n) * sizeof(Element));
  REQUIRE_MSG(simpatico::launch_decode_fused_tree(
                *tree, enc.buffers, dtype.c_str(), n, reinterpret_cast<void*>(d_plain), stream),
              "[%s] plain decode failed",
              dtype.c_str());
  std::vector<Element> plain(static_cast<std::size_t>(n));
  cudaMemcpy(plain.data(),
             reinterpret_cast<const void*>(d_plain),
             plain.size() * sizeof(Element),
             cudaMemcpyDeviceToHost);
  REQUIRE_MSG(plain == data, "[%s] plain decode != input", dtype.c_str());

  struct Case {
    const char* name;
    unsigned keep_permille;
    bool cluster;
  };
  const Case cases[] = {{"sparse_0.1pct", 1, false},
                        {"mid_2pct", 20, false},
                        {"dense_15pct", 150, false},
                        {"clustered", 150, true}};
  for (auto const& tc : cases) {
    const std::vector<std::uint32_t> ids =
      gen_row_ids(n, tc.keep_permille, 0xBEEF ^ tc.keep_permille, /*zero_chunk=*/4, tc.cluster);
    if (ids.empty()) continue;
    const HostCsr ref_csr = host_csr(ids);

    // Device CSR construction must match the host CSR exactly.
    CUdeviceptr d_ids = enc.upload_bytes(ids.data(), ids.size() * 4);
    owned_chunk_row_set built =
      sirius::codegen::bucket_sorted_local_ids(reinterpret_cast<std::uint32_t const*>(d_ids),
                                               static_cast<std::int64_t>(ids.size()),
                                               n,
                                               stream,
                                               mr);
    REQUIRE_MSG(built.num_touched == static_cast<std::int64_t>(ref_csr.chunks.size()) &&
                  built.num_survivors == static_cast<std::int64_t>(ids.size()),
                "[%s/%s] bucket counts (T=%lld vs %zu)",
                dtype.c_str(),
                tc.name,
                static_cast<long long>(built.num_touched),
                ref_csr.chunks.size());
    std::vector<std::uint32_t> got_chunks(ref_csr.chunks.size());
    std::vector<std::uint32_t> got_offsets(ref_csr.offsets.size());
    std::vector<std::uint16_t> got_in(ref_csr.in_chunk.size());
    cudaMemcpy(
      got_chunks.data(), built.chunk_ids.data(), got_chunks.size() * 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(got_offsets.data(),
               built.chunk_out_offsets.data(),
               got_offsets.size() * 4,
               cudaMemcpyDeviceToHost);
    cudaMemcpy(
      got_in.data(), built.in_chunk_offsets.data(), got_in.size() * 2, cudaMemcpyDeviceToHost);
    REQUIRE_MSG(
      got_chunks == ref_csr.chunks && got_offsets == ref_csr.offsets && got_in == ref_csr.in_chunk,
      "[%s/%s] device CSR != host CSR",
      dtype.c_str(),
      tc.name);

    // int32 expansion and mask expansion vs host references.
    CUdeviceptr d_exp = enc.alloc(ids.size() * 4);
    sirius::codegen::row_set_to_local_indices(
      built.view(), reinterpret_cast<std::int32_t*>(d_exp), stream);
    std::vector<std::int32_t> got_exp(ids.size());
    cudaStreamSynchronize(stream.value());
    cudaMemcpy(
      got_exp.data(), reinterpret_cast<const void*>(d_exp), ids.size() * 4, cudaMemcpyDeviceToHost);
    bool exp_ok = true;
    for (std::size_t i = 0; i < ids.size(); ++i)
      exp_ok &= got_exp[i] == static_cast<std::int32_t>(ids[i]);
    REQUIRE_MSG(exp_ok, "[%s/%s] int32 expansion mismatch", dtype.c_str(), tc.name);

    const std::size_t nwords = static_cast<std::size_t>(nc) * kWordsPerChunk;
    CUdeviceptr d_mask       = enc.alloc(nwords * 4);
    cudaMemset(reinterpret_cast<void*>(d_mask), 0xFF, nwords * 4);  // prove zeroing
    CUdeviceptr d_choffs = enc.alloc(static_cast<std::size_t>(nc + 1) * 4);
    cudaMemset(reinterpret_cast<void*>(d_choffs), 0xFF, static_cast<std::size_t>(nc + 1) * 4);
    sirius::codegen::row_set_to_mask(built.view(),
                                     reinterpret_cast<std::uint32_t*>(d_mask),
                                     reinterpret_cast<std::uint32_t*>(d_choffs),
                                     stream,
                                     mr);
    cudaStreamSynchronize(stream.value());
    std::vector<std::uint32_t> ref_mask(nwords, 0u);
    for (auto id : ids)
      ref_mask[id / 32] |= 1u << (id % 32);
    std::vector<std::uint32_t> ref_choffs(static_cast<std::size_t>(nc) + 1, 0u);
    {
      std::uint32_t acc = 0;
      for (std::int64_t c = 0; c < nc; ++c) {
        ref_choffs[static_cast<std::size_t>(c)] = acc;
        for (int w = 0; w < kWordsPerChunk; ++w)
          acc += static_cast<std::uint32_t>(
            __builtin_popcount(ref_mask[static_cast<std::size_t>(c) * kWordsPerChunk + w]));
      }
      ref_choffs[static_cast<std::size_t>(nc)] = acc;
    }
    std::vector<std::uint32_t> got_mask(nwords);
    std::vector<std::uint32_t> got_choffs(static_cast<std::size_t>(nc) + 1);
    cudaMemcpy(
      got_mask.data(), reinterpret_cast<const void*>(d_mask), nwords * 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(got_choffs.data(),
               reinterpret_cast<const void*>(d_choffs),
               got_choffs.size() * 4,
               cudaMemcpyDeviceToHost);
    REQUIRE_MSG(got_mask == ref_mask && got_choffs == ref_choffs,
                "[%s/%s] mask expansion mismatch",
                dtype.c_str(),
                tc.name);

    // K8 sparse decode vs host gather of the plain reference.
    CUdeviceptr d_out = enc.alloc(ids.size() * sizeof(Element));
    cudaMemset(reinterpret_cast<void*>(d_out), 0xAB, ids.size() * sizeof(Element));
    REQUIRE_MSG(
      simpatico::launch_decode_fused_tree_sparse_rows(
        *tree, enc.buffers, dtype.c_str(), n, built.view(), reinterpret_cast<void*>(d_out), stream),
      "[%s/%s] sparse_rows launch failed",
      dtype.c_str(),
      tc.name);
    cudaStreamSynchronize(stream.value());  // launcher is stream-ordered by contract
    std::vector<Element> got(ids.size());
    cudaMemcpy(got.data(),
               reinterpret_cast<const void*>(d_out),
               got.size() * sizeof(Element),
               cudaMemcpyDeviceToHost);
    std::vector<Element> expect;
    expect.reserve(ids.size());
    for (auto id : ids)
      expect.push_back(data[id]);
    REQUIRE_MSG(got == expect,
                "[%s/%s] K8 output != host gather (%zu rows)",
                dtype.c_str(),
                tc.name,
                ids.size());
  }

  std::printf(
    "PASS: sparse roundtrip %s (%s root)\n", dtype.c_str(), delta_root ? "delta" : "bitpack");
  return true;
}

// ── 4. K5s dict gather ──────────────────────────────────────────────────────
bool run_sparse_dict_gather(std::int32_t key_width, int arch)
{
  const std::int64_t n  = 5 * kChunk + 321;
  const std::int64_t nc = codegen::num_chunks_for(n);
  const rmm::cuda_stream_view stream{};
  auto mr = rmm::mr::get_current_device_resource_ref();

  constexpr std::int32_t kKeys = 7;
  std::vector<char> keys(static_cast<std::size_t>(kKeys) * key_width);
  for (std::size_t i = 0; i < keys.size(); ++i)
    keys[i] = static_cast<char>('A' + (i % 23));

  const std::vector<std::int32_t> codes = gen_data<std::int32_t>(n, 0, kKeys);
  auto tree                             = jit::FusedTree::make(OpKind::Bitpack);
  GpuEncoded enc =
    codegen_test::gpu_encode_tree<std::int32_t>(*tree, "int32_t", codes.data(), n, arch);
  if (!compact_packed_on_host(enc, nc)) return false;
  CUdeviceptr d_keys = enc.upload_bytes(keys.data(), keys.size());

  const std::vector<std::uint32_t> ids = gen_row_ids(n, 37, 0xD1C7, /*zero_chunk=*/1, false);
  const HostCsr csr                    = host_csr(ids);
  chunk_row_set set                    = upload_csr(enc, csr, n);

  CUdeviceptr d_out = enc.alloc(ids.size() * static_cast<std::size_t>(key_width));
  cudaMemset(reinterpret_cast<void*>(d_out), 0xEE, ids.size() * key_width);
  REQUIRE_MSG(
    simpatico::launch_decode_fused_tree_sparse_dict_gather(*tree,
                                                           enc.buffers,
                                                           "int32_t",
                                                           n,
                                                           set,
                                                           reinterpret_cast<const void*>(d_keys),
                                                           key_width,
                                                           reinterpret_cast<void*>(d_out),
                                                           stream),
    "[w=%d] sparse_dict_gather launch failed",
    key_width);
  cudaStreamSynchronize(stream.value());
  std::vector<char> got(ids.size() * static_cast<std::size_t>(key_width));
  cudaMemcpy(got.data(), reinterpret_cast<const void*>(d_out), got.size(), cudaMemcpyDeviceToHost);
  std::vector<char> expect;
  expect.reserve(got.size());
  for (auto id : ids) {
    const char* k = keys.data() + codes[id] * key_width;
    expect.insert(expect.end(), k, k + key_width);
  }
  REQUIRE_MSG(got == expect, "[w=%d] K5s chars != host key gather", key_width);
  std::printf("PASS: sparse dict gather (width %d)\n", key_width);
  return true;
}

// ── 5. sort_unique_global_ids ───────────────────────────────────────────────
bool run_sort_unique()
{
  const rmm::cuda_stream_view stream{};
  auto mr = rmm::mr::get_current_device_resource_ref();

  // Shuffled, duplicate-carrying u64 list spanning >2^32 (the very case that
  // kills u32 global ids).
  std::vector<std::uint64_t> ids;
  std::uint64_t s = 0xFEED;
  for (int i = 0; i < 40000; ++i) {
    const std::uint64_t v = splitmix64(s) % (6'000'000'000ull);
    ids.push_back(v);
    if (i % 7 == 0) ids.push_back(v);  // duplicates
  }
  void* d_ids = nullptr;
  cudaMalloc(&d_ids, ids.size() * 8);
  cudaMemcpy(d_ids, ids.data(), ids.size() * 8, cudaMemcpyHostToDevice);

  auto res = sirius::codegen::sort_unique_global_ids(
    static_cast<std::uint64_t const*>(d_ids), static_cast<std::int64_t>(ids.size()), stream, mr);

  std::vector<std::uint64_t> ref_sorted(ids);
  std::sort(ref_sorted.begin(), ref_sorted.end());
  ref_sorted.erase(std::unique(ref_sorted.begin(), ref_sorted.end()), ref_sorted.end());
  // Async contract (sync surgery): unique count is device-resident; the ids
  // buffer is worst-case sized with the first unique_count entries valid.
  std::int32_t unique_count = 0;
  cudaStreamSynchronize(stream.value());
  cudaMemcpy(&unique_count, res.count_dev.data(), 4, cudaMemcpyDeviceToHost);
  REQUIRE_MSG(unique_count == static_cast<std::int32_t>(ref_sorted.size()),
              "unique_count %d != %zu",
              unique_count,
              ref_sorted.size());
  std::vector<std::uint64_t> got_unique(ref_sorted.size());
  std::vector<std::int32_t> got_rank(ids.size());
  cudaMemcpy(got_unique.data(), res.ids.data(), got_unique.size() * 8, cudaMemcpyDeviceToHost);
  cudaMemcpy(got_rank.data(), res.restore_rank.data(), got_rank.size() * 4, cudaMemcpyDeviceToHost);
  REQUIRE_MSG(got_unique == ref_sorted, "sorted-unique ids mismatch");
  bool rank_ok = true;
  for (std::size_t i = 0; i < ids.size(); ++i)
    rank_ok &= got_unique[static_cast<std::size_t>(got_rank[i])] == ids[i];
  REQUIRE_MSG(rank_ok, "restore ranks do not reproduce the original list");
  cudaFree(d_ids);
  std::printf("PASS: sort_unique_global_ids (restore ranks)\n");
  return true;
}

// ── 6. Raw-gather fast path + lazy canonicalization (materializer layer) ────
// The q9-nation +61 ms fix: single-batch uncompressed origins gather directly
// with the raw (unsorted, duplicated) u64 ids — no sort, no restore. A
// compressed column against the SAME prepared selection canonicalizes lazily
// and must return the identical row sequence. A 2-batch layout exercises the
// canonical (folded-sync) prepare end to end.
bool run_raw_fastpath()
{
  const rmm::cuda_stream_view stream{};
  auto mr              = rmm::mr::get_current_device_resource_ref();
  const std::int64_t n = 3 * kChunk + 421;

  // Host data + device cudf column (uncompressed origin).
  std::vector<std::int32_t> data(static_cast<std::size_t>(n));
  for (std::int64_t i = 0; i < n; ++i) {
    data[static_cast<std::size_t>(i)] = static_cast<std::int32_t>((i * 2654435761u) % 100000);
  }
  auto col = cudf::make_fixed_width_column(cudf::data_type{cudf::type_id::INT32},
                                           static_cast<cudf::size_type>(n),
                                           cudf::mask_state::UNALLOCATED,
                                           stream,
                                           mr);
  cudaMemcpy(
    col->mutable_view().head<void>(), data.data(), data.size() * 4, cudaMemcpyHostToDevice);

  // Unsorted, duplicate-carrying id list (gather semantics).
  std::vector<std::uint64_t> ids;
  std::uint64_t s = 0xACE;
  for (int i = 0; i < 20000; ++i) {
    const std::uint64_t v = splitmix64(s) % static_cast<std::uint64_t>(n);
    ids.push_back(v);
    if (i % 5 == 0) ids.push_back(v);  // duplicates
  }
  void* d_ids = nullptr;
  cudaMalloc(&d_ids, ids.size() * 8);
  cudaMemcpy(d_ids, ids.data(), ids.size() * 8, cudaMemcpyHostToDevice);

  std::vector<std::int32_t> expect;
  expect.reserve(ids.size());
  for (auto id : ids)
    expect.push_back(data[static_cast<std::size_t>(id)]);

  auto check = [&](std::unique_ptr<cudf::column> const& got, char const* what) {
    REQUIRE_MSG(got && got->size() == static_cast<cudf::size_type>(ids.size()),
                "%s: wrong output size",
                what);
    std::vector<std::int32_t> host(ids.size());
    cudaStreamSynchronize(stream.value());
    cudaMemcpy(host.data(), got->view().head<void>(), host.size() * 4, cudaMemcpyDeviceToHost);
    REQUIRE_MSG(host == expect, "%s: output != host gather (order/dups)", what);
    return true;
  };

  // (a) Single-batch layout: prepare must take the raw path (no device work).
  {
    auto layout = sirius::late_mat::pinned_table_layout::from_batch_rows({n}, 7);
    sirius::late_mat::row_id_list list{
      static_cast<std::uint64_t const*>(d_ids), static_cast<std::int64_t>(ids.size()), false};
    auto sel = sirius::late_mat::prepare_selection(layout, list, stream, mr);
    REQUIRE_MSG(sel->raw_ids != nullptr, "single-batch prepare must take the raw path");

    sirius::late_mat::pinned_column_view origin;
    origin.dtype          = cudf::data_type{cudf::type_id::INT32};
    origin.pin_generation = 7;
    origin.batches.push_back({nullptr, 0, col->view(), n});
    if (!check(sirius::late_mat::materialize(origin, *sel, stream, mr), "raw uncompressed"))
      return false;

    // (b) Compressed origin against the SAME prepared selection: lazy
    // canonicalization, identical row sequence (restore ranks).
    auto ct = simpatico::compress_with_plan(
      cudf::table_view{{col->view()}}, "input -> bitpack\n", stream, mr);
    sirius::late_mat::pinned_column_view corigin;
    corigin.dtype          = origin.dtype;
    corigin.pin_generation = 7;
    corigin.batches.push_back({&ct, 0, cudf::column_view{}, n});
    if (!check(sirius::late_mat::materialize(corigin, *sel, stream, mr),
               "raw->canonical compressed"))
      return false;
  }

  // (c) Two-batch layout: canonical prepare (folded-sync split) end to end.
  {
    const std::int64_t rows0 = 2 * kChunk;
    auto layout = sirius::late_mat::pinned_table_layout::from_batch_rows({rows0, n - rows0}, 9);
    sirius::late_mat::row_id_list list{
      static_cast<std::uint64_t const*>(d_ids), static_cast<std::int64_t>(ids.size()), false};
    auto sel = sirius::late_mat::prepare_selection(layout, list, stream, mr);
    REQUIRE_MSG(sel->raw_ids == nullptr, "multi-batch prepare must take the canonical path");

    auto v0 = cudf::slice(col->view(),
                          {0,
                           static_cast<cudf::size_type>(rows0),
                           static_cast<cudf::size_type>(rows0),
                           static_cast<cudf::size_type>(n)});
    sirius::late_mat::pinned_column_view origin;
    origin.dtype          = cudf::data_type{cudf::type_id::INT32};
    origin.pin_generation = 9;
    origin.batches.push_back({nullptr, 0, v0[0], rows0});
    origin.batches.push_back({nullptr, 0, v0[1], n - rows0});
    if (!check(sirius::late_mat::materialize(origin, *sel, stream, mr),
               "canonical 2-batch uncompressed"))
      return false;
  }

  cudaFree(d_ids);
  std::printf("PASS: raw-gather fast path + lazy canonicalization\n");
  return true;
}

// ── 7. Seam-ii capture helper contract (status-gated, tag-independent) ──────
// capture_scan_filter_selection must move+rebind the wave-1 buffers IFF
// status == applied — including the untagged membership/partial-coverage
// batches, which share the applied status — and must return empty WITHOUT
// touching the result on bailed_high_selectivity / refused / failed.
bool run_capture_contract()
{
  const rmm::cuda_stream_view stream{};
  auto mr = rmm::mr::get_current_device_resource_ref();
  using sirius::codegen::scan_filter_result;
  using sirius::codegen::scan_filter_status;

  auto make_result = [&](scan_filter_status st, bool with_indices) {
    scan_filter_result r;
    r.status         = st;
    r.applied        = st == scan_filter_status::applied;
    r.num_rows       = 2048;
    r.survivor_count = 100;
    r.mask_words     = rmm::device_buffer(64 * sizeof(std::uint32_t), stream, mr);
    r.chunk_offsets  = rmm::device_buffer(3 * sizeof(std::uint32_t), stream, mr);
    if (with_indices) {
      r.row_indices = rmm::device_buffer(100 * sizeof(std::int32_t), stream, mr);
    }
    return r;
  };

  {  // applied: buffers MOVE out (source emptied), set_stream-rebound, scalars kept
    auto r   = make_result(scan_filter_status::applied, true);
    auto cap = sirius::codegen::capture_scan_filter_selection(std::move(r), stream);
    REQUIRE_MSG(static_cast<bool>(cap), "applied result must capture");
    REQUIRE_MSG(cap.mask_words && cap.mask_words->size() == 64 * sizeof(std::uint32_t) &&
                  cap.chunk_offsets && cap.chunk_offsets->size() == 3 * sizeof(std::uint32_t) &&
                  cap.row_indices && cap.row_indices->size() == 100 * sizeof(std::int32_t),
                "captured buffers must carry the moved allocations");
    REQUIRE_MSG(cap.num_rows == 2048 && cap.survivor_count == 100,
                "captured scalars must mirror the result");
    REQUIRE_MSG(
      r.mask_words.size() == 0 && r.chunk_offsets.size() == 0 && r.row_indices.size() == 0,
      "applied capture must MOVE (source buffers emptied)");
    REQUIRE_MSG(r.num_rows == 2048 && r.survivor_count == 100,
                "capture must leave the result's scalar fields intact (DIAG contract)");
    REQUIRE_MSG(cap.mask_words->stream().value() == stream.value(),
                "moved buffers must be set_stream-rebound to the capture stream");
  }
  {  // applied without a TierB index list: row_indices stays null
    auto r   = make_result(scan_filter_status::applied, false);
    auto cap = sirius::codegen::capture_scan_filter_selection(std::move(r), stream);
    REQUIRE_MSG(static_cast<bool>(cap) && cap.row_indices == nullptr,
                "mask-only applied capture must leave row_indices null");
  }
  for (auto st : {scan_filter_status::bailed_high_selectivity,
                  scan_filter_status::refused,
                  scan_filter_status::failed}) {
    auto r   = make_result(st, true);
    auto cap = sirius::codegen::capture_scan_filter_selection(std::move(r), stream);
    REQUIRE_MSG(
      !static_cast<bool>(cap), "non-applied status %d must not capture", static_cast<int>(st));
    REQUIRE_MSG(
      r.mask_words.size() != 0 && r.chunk_offsets.size() != 0 && r.row_indices.size() != 0,
      "non-applied capture attempt must leave the result untouched (status %d)",
      static_cast<int>(st));
  }
  std::printf("PASS: seam-ii capture contract (status-gated)\n");
  return true;
}

// ── 8. Stored-dtype re-tag through the column-level materialize path ────────
// The q9 arm-C regression class: codecs run on integer STORAGE; the logical
// type (decimal scale, date/timestamp unit) is restored by apply_stored_dtype
// at the table-level decompress seams — the column-level late-mat path must
// re-tag too, or every compressed fixed-point/temporal column materializes
// scale-less (DECIMAL64(-2) values silently x100). This sweeps the reachable
// class, not just the q9 instance: decimal32/64 x bitpack (K8 random access),
// decimal64 x delta (K8 staged slab), date32 + timestamp_us (temporal
// re-tags), and decimal64 x identity (tier_b => the full-decode + gather
// fallback route). DECIMAL128 is not swept: the 16-byte width is outside the
// fused codecs' domain and identity-stored columns keep their original type.
template <typename Storage>
bool run_stored_dtype_case(
  char const* name, cudf::data_type stored, char const* dsl, std::int64_t base, std::int64_t range)
{
  const rmm::cuda_stream_view stream{};
  auto mr              = rmm::mr::get_current_device_resource_ref();
  const std::int64_t n = 2 * kChunk + 77;

  std::vector<Storage> raw(static_cast<std::size_t>(n));
  std::uint64_t s = 0x5CA1E ^ static_cast<std::uint64_t>(range);
  for (std::int64_t i = 0; i < n; ++i) {
    raw[static_cast<std::size_t>(i)] = static_cast<Storage>(
      base + static_cast<std::int64_t>(splitmix64(s) % static_cast<std::uint64_t>(range)));
  }
  auto col = cudf::make_fixed_width_column(
    stored, static_cast<cudf::size_type>(n), cudf::mask_state::UNALLOCATED, stream, mr);
  cudaMemcpy(col->mutable_view().head<void>(),
             raw.data(),
             raw.size() * sizeof(Storage),
             cudaMemcpyHostToDevice);
  auto ct = simpatico::compress_with_plan(cudf::table_view{{col->view()}}, dsl, stream, mr);

  // Sparse-ish (10%), unsorted, duplicated ids: raw prepare -> lazy canonical
  // -> sparse/full route per the column's tier.
  std::vector<std::uint64_t> ids;
  std::uint64_t s2 = 0xF00D;
  for (std::int64_t i = 0; i < n / 10; ++i) {
    const std::uint64_t v = splitmix64(s2) % static_cast<std::uint64_t>(n);
    ids.push_back(v);
    if (i % 4 == 0) ids.push_back(v);
  }
  void* d_ids = nullptr;
  cudaMalloc(&d_ids, ids.size() * 8);
  cudaMemcpy(d_ids, ids.data(), ids.size() * 8, cudaMemcpyHostToDevice);

  auto layout = sirius::late_mat::pinned_table_layout::from_batch_rows({n}, 3);
  sirius::late_mat::row_id_list list{
    static_cast<std::uint64_t const*>(d_ids), static_cast<std::int64_t>(ids.size()), false};
  auto sel = sirius::late_mat::prepare_selection(layout, list, stream, mr);

  sirius::late_mat::pinned_column_view origin;
  origin.dtype          = stored;
  origin.pin_generation = 3;
  origin.batches.push_back({&ct, 0, cudf::column_view{}, n});
  auto got = sirius::late_mat::materialize(origin, *sel, stream, mr);
  cudaStreamSynchronize(stream.value());

  REQUIRE_MSG(
    got && got->size() == static_cast<cudf::size_type>(ids.size()), "[%s] wrong output size", name);
  REQUIRE_MSG(got->type() == stored,
              "[%s] stored dtype NOT re-tagged (got id=%d scale=%d, want id=%d scale=%d) — "
              "the q9 arm-C class",
              name,
              static_cast<int>(got->type().id()),
              got->type().scale(),
              static_cast<int>(stored.id()),
              stored.scale());
  std::vector<Storage> host(ids.size());
  cudaMemcpy(
    host.data(), got->view().head<void>(), host.size() * sizeof(Storage), cudaMemcpyDeviceToHost);
  bool bits_ok = true;
  for (std::size_t i = 0; i < ids.size(); ++i) {
    bits_ok &= host[i] == raw[static_cast<std::size_t>(ids[i])];
  }
  REQUIRE_MSG(bits_ok, "[%s] payload bits != host gather", name);
  cudaFree(d_ids);
  std::printf("PASS: stored-dtype re-tag [%s]\n", name);
  return true;
}

bool run_stored_dtype_retag()
{
  using cudf::data_type;
  using cudf::type_id;
  bool ok = true;
  ok &= run_stored_dtype_case<std::int64_t>(
    "decimal64_bitpack", data_type{type_id::DECIMAL64, -2}, "input -> bitpack\n", 100000, 9000);
  ok &= run_stored_dtype_case<std::int64_t>("decimal64_delta",
                                            data_type{type_id::DECIMAL64, -2},
                                            "input -> delta -> differences\n",
                                            5000000,
                                            700);
  ok &= run_stored_dtype_case<std::int32_t>(
    "decimal32_bitpack", data_type{type_id::DECIMAL32, -2}, "input -> bitpack\n", 20000, 3000);
  ok &= run_stored_dtype_case<std::int32_t>(
    "date32_bitpack", data_type{type_id::TIMESTAMP_DAYS}, "input -> bitpack\n", 8035, 2526);
  ok &= run_stored_dtype_case<std::int64_t>("timestamp_us_delta",
                                            data_type{type_id::TIMESTAMP_MICROSECONDS},
                                            "input -> delta -> differences\n",
                                            694224000000000LL,
                                            86400000000LL);
  ok &= run_stored_dtype_case<std::int64_t>("decimal64_identity_tierb",
                                            data_type{type_id::DECIMAL64, -2},
                                            "input -> identity\n",
                                            100000,
                                            9000);
  return ok;
}

// Host mask + CNT helpers for the emission test (same shapes as
// test_masked_decode_variants' harness).
std::vector<std::uint32_t> make_host_mask(
  std::int64_t n, std::int64_t nc, unsigned seed, unsigned keep_pct, std::int64_t zero_chunk)
{
  std::vector<std::uint32_t> m(static_cast<std::size_t>(nc) * kWordsPerChunk, 0u);
  for (std::int64_t r = 0; r < n; ++r) {
    if (r / kChunk == zero_chunk) continue;
    std::uint64_t s = seed ^ (0x9E3779B97F4A7C15ull * static_cast<std::uint64_t>(r + 1));
    if (splitmix64(s) % 100ull < keep_pct) m[static_cast<std::size_t>(r) / 32] |= (1u << (r % 32));
  }
  return m;
}

std::int64_t host_cnt(std::vector<std::uint32_t> const& mask,
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

// ── 9. Rowid emission: dense/mask x u64/u32 + guards ────────────────────────
bool run_rowid_emission()
{
  const rmm::cuda_stream_view stream{};
  auto mr = rmm::mr::get_current_device_resource_ref();
  using sirius::late_mat::emit_rowid_column;
  using sirius::late_mat::rowid_emission_request;
  using sirius::late_mat::rowid_width;

  // Dense, both widths: out[k] == start + k.
  for (auto width : {rowid_width::u64, rowid_width::u32}) {
    rowid_emission_request req;
    req.range = {123456, 2000};
    req.width = width;
    auto col  = emit_rowid_column(req, 2000, stream, mr);
    cudaStreamSynchronize(stream.value());
    REQUIRE_MSG(col && col->size() == 2000, "dense emission size");
    if (width == rowid_width::u64) {
      REQUIRE_MSG(col->type().id() == cudf::type_id::UINT64, "dense u64 type");
      std::vector<std::uint64_t> got(2000);
      cudaMemcpy(got.data(), col->view().head<void>(), 2000 * 8, cudaMemcpyDeviceToHost);
      for (int k = 0; k < 2000; ++k)
        if (got[k] != 123456ull + k) { REQUIRE_MSG(false, "dense u64 value at %d", k); }
    } else {
      REQUIRE_MSG(col->type().id() == cudf::type_id::UINT32, "dense u32 type");
      std::vector<std::uint32_t> got(2000);
      cudaMemcpy(got.data(), col->view().head<void>(), 2000 * 4, cudaMemcpyDeviceToHost);
      for (int k = 0; k < 2000; ++k)
        if (got[k] != 123456u + k) { REQUIRE_MSG(false, "dense u32 value at %d", k); }
    }
  }

  // Mask form: survivors of a host mask, + base, both widths.
  {
    const std::int64_t n  = 2 * kChunk + 300;
    const std::int64_t nc = codegen::num_chunks_for(n);
    auto hmask            = make_host_mask(n, nc, 0xAB, 20, /*zero_chunk=*/1);
    std::vector<std::uint32_t> choffs(static_cast<std::size_t>(nc) + 1, 0);
    std::int64_t const survivors = host_cnt(hmask, nc, choffs);
    void* d_mask                 = nullptr;
    void* d_offs                 = nullptr;
    cudaMalloc(&d_mask, hmask.size() * 4);
    cudaMalloc(&d_offs, choffs.size() * 4);
    cudaMemcpy(d_mask, hmask.data(), hmask.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(d_offs, choffs.data(), choffs.size() * 4, cudaMemcpyHostToDevice);
    selection_mask mask{
      static_cast<std::uint32_t*>(d_mask), n, survivors, static_cast<std::uint32_t*>(d_offs)};
    std::vector<std::uint64_t> expect;
    for (std::int64_t r = 0; r < n; ++r)
      if ((hmask[static_cast<std::size_t>(r) / 32] >> (r % 32)) & 1u)
        expect.push_back(7777000ull + static_cast<std::uint64_t>(r));

    rowid_emission_request req;
    req.range = {7777000, n};
    req.mask  = &mask;
    for (auto width : {sirius::late_mat::rowid_width::u64, sirius::late_mat::rowid_width::u32}) {
      req.width = width;
      auto col  = emit_rowid_column(req, survivors, stream, mr);
      cudaStreamSynchronize(stream.value());
      REQUIRE_MSG(col && col->size() == static_cast<cudf::size_type>(survivors),
                  "mask emission size");
      bool ok = true;
      if (width == sirius::late_mat::rowid_width::u64) {
        std::vector<std::uint64_t> got(expect.size());
        cudaMemcpy(got.data(), col->view().head<void>(), got.size() * 8, cudaMemcpyDeviceToHost);
        ok = got == expect;
      } else {
        std::vector<std::uint32_t> got(expect.size());
        cudaMemcpy(got.data(), col->view().head<void>(), got.size() * 4, cudaMemcpyDeviceToHost);
        for (std::size_t i = 0; ok && i < got.size(); ++i)
          ok = got[i] == static_cast<std::uint32_t>(expect[i]);
      }
      REQUIRE_MSG(ok, "mask emission values (width %d)", static_cast<int>(width));
    }
    cudaFree(d_mask);
    cudaFree(d_offs);
  }

  // Guards: dense row mismatch throws; u32 span overflow throws.
  {
    bool threw = false;
    try {
      rowid_emission_request req;
      req.range = {0, 100};
      (void)emit_rowid_column(req, 99, stream, mr);
    } catch (std::exception const&) {
      threw = true;
    }
    REQUIRE_MSG(threw, "dense rows/range mismatch must throw");
    threw = false;
    try {
      rowid_emission_request req;
      req.range = {(std::int64_t{1} << 32) - 10, 100};  // crosses 2^32
      req.width = sirius::late_mat::rowid_width::u32;
      (void)emit_rowid_column(req, 100, stream, mr);
    } catch (std::exception const&) {
      threw = true;
    }
    REQUIRE_MSG(threw, "u32 overflow span must throw");
  }
  std::printf("PASS: rowid emission (dense/mask x u64/u32 + guards)\n");
  return true;
}

// ── 10. Multi-source fixed-width gather (the multi-batch raw-path kernel) ───
// Kernel-level coverage. The materializer's multi-batch raw dispatch reads the
// SIRIUS_EXP_LATE_MAT_V2 gate through a first-use-cached static, so a single
// process runs either the gated or ungated dispatch — this file tests the
// ungated (v1) semantics end to end and the gated path's kernel directly; a
// gate-on integration run needs a setenv-before-first-use main.
bool run_multi_source_gather()
{
  const rmm::cuda_stream_view stream{};
  // 3 batches with uneven sizes; ids unsorted, duplicated, hugging batch
  // boundaries.
  const std::vector<std::int64_t> rows = {1000, 1, 2048};
  std::vector<std::int64_t> starts     = {0, 1000, 1001, 3049};
  std::vector<std::vector<std::int64_t>> data(3);
  std::vector<void*> d_batches(3);
  std::vector<void const*> bases(3);
  for (int b = 0; b < 3; ++b) {
    data[b].resize(static_cast<std::size_t>(rows[b]));
    for (std::int64_t i = 0; i < rows[b]; ++i)
      data[b][static_cast<std::size_t>(i)] = (b + 1) * 1000000 + i;
    cudaMalloc(&d_batches[b], data[b].size() * 8);
    cudaMemcpy(d_batches[b], data[b].data(), data[b].size() * 8, cudaMemcpyHostToDevice);
    bases[b] = d_batches[b];
  }
  std::vector<std::uint64_t> ids = {0, 999, 1000, 1001, 3048, 500, 1000, 2048, 3048, 0};
  std::vector<std::int64_t> expect;
  for (auto id : ids) {
    int b = id < 1000 ? 0 : (id < 1001 ? 1 : 2);
    expect.push_back(data[b][static_cast<std::size_t>(id - starts[b])]);
  }
  void *d_bases = nullptr, *d_starts = nullptr, *d_ids = nullptr, *d_out = nullptr;
  cudaMalloc(&d_bases, bases.size() * sizeof(void*));
  cudaMalloc(&d_starts, starts.size() * 8);
  cudaMalloc(&d_ids, ids.size() * 8);
  cudaMalloc(&d_out, ids.size() * 8);
  cudaMemcpy(d_bases, bases.data(), bases.size() * sizeof(void*), cudaMemcpyHostToDevice);
  cudaMemcpy(d_starts, starts.data(), starts.size() * 8, cudaMemcpyHostToDevice);
  cudaMemcpy(d_ids, ids.data(), ids.size() * 8, cudaMemcpyHostToDevice);
  sirius::codegen::multi_source_gather_fixed(static_cast<void const* const*>(d_bases),
                                             static_cast<std::int64_t const*>(d_starts),
                                             3,
                                             /*elem_size=*/8,
                                             static_cast<std::uint64_t const*>(d_ids),
                                             static_cast<std::int64_t>(ids.size()),
                                             d_out,
                                             stream);
  cudaStreamSynchronize(stream.value());
  std::vector<std::int64_t> got(ids.size());
  cudaMemcpy(got.data(), d_out, got.size() * 8, cudaMemcpyDeviceToHost);
  REQUIRE_MSG(got == expect, "multi-source gather mismatch (8B)");
  for (auto p : {d_bases, d_starts, d_ids, d_out})
    cudaFree(p);
  for (auto p : d_batches)
    cudaFree(p);
  std::printf("PASS: multi-source fixed-width gather (boundary/dup/disorder)\n");
  return true;
}

// ── 11. Multi-origin output-group materialization (FD-GBR composition) ──────
// The v3 shape: groups keyed by ONE nominated rowid; key columns from N
// origin tables, each with its own rider rowid column at the output port.
// This test proves the claim that NO new machinery is needed: N origins = N
// INDEPENDENT prepare_selection+materialize pairs against the same port row
// set — different layouts, different id lists (row-aligned), same stream.
// Rider semantics under the FD proof: all rows of a group share the rider
// value, so the (arbitrary, duplicated) per-group representative ids here
// stand in for any group-representative choice.
bool run_multi_origin_groups()
{
  const rmm::cuda_stream_view stream{};
  auto mr = rmm::mr::get_current_device_resource_ref();

  // Origin A: tiny single-batch dimension (nation-like, 25 rows, INT32).
  const std::int64_t n_a = 25;
  std::vector<std::int32_t> a_data(n_a);
  for (int i = 0; i < n_a; ++i)
    a_data[i] = 900 + i;
  auto a_col = cudf::make_fixed_width_column(cudf::data_type{cudf::type_id::INT32},
                                             static_cast<cudf::size_type>(n_a),
                                             cudf::mask_state::UNALLOCATED,
                                             stream,
                                             mr);
  cudaMemcpy(
    a_col->mutable_view().head<void>(), a_data.data(), a_data.size() * 4, cudaMemcpyHostToDevice);

  // Origin B: larger single-batch fact-side table (customer-like, INT64).
  const std::int64_t n_b = 3 * kChunk + 11;
  std::vector<std::int64_t> b_data(static_cast<std::size_t>(n_b));
  for (std::int64_t i = 0; i < n_b; ++i)
    b_data[static_cast<std::size_t>(i)] = 5'000'000 + i * 3;
  auto b_col = cudf::make_fixed_width_column(cudf::data_type{cudf::type_id::INT64},
                                             static_cast<cudf::size_type>(n_b),
                                             cudf::mask_state::UNALLOCATED,
                                             stream,
                                             mr);
  cudaMemcpy(
    b_col->mutable_view().head<void>(), b_data.data(), b_data.size() * 8, cudaMemcpyHostToDevice);

  // One output-group port: G groups, each carrying a rider id per origin
  // (duplicated + unordered — group order is whatever the aggregate emitted).
  const std::size_t G = 700;
  std::vector<std::uint64_t> a_ids(G), b_ids(G);
  std::uint64_t s = 0x6B0;
  for (std::size_t g = 0; g < G; ++g) {
    a_ids[g] = splitmix64(s) % static_cast<std::uint64_t>(n_a);
    b_ids[g] = splitmix64(s) % static_cast<std::uint64_t>(n_b);
  }
  void *d_a = nullptr, *d_b = nullptr;
  cudaMalloc(&d_a, G * 8);
  cudaMalloc(&d_b, G * 8);
  cudaMemcpy(d_a, a_ids.data(), G * 8, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, b_ids.data(), G * 8, cudaMemcpyHostToDevice);

  // N independent prepare+materialize pairs, same stream, same port rows.
  auto layout_a = sirius::late_mat::pinned_table_layout::from_batch_rows({n_a}, 11);
  auto layout_b = sirius::late_mat::pinned_table_layout::from_batch_rows({n_b}, 12);
  auto sel_a    = sirius::late_mat::prepare_selection(
    layout_a,
    {static_cast<std::uint64_t const*>(d_a), static_cast<std::int64_t>(G), false},
    stream,
    mr);
  auto sel_b = sirius::late_mat::prepare_selection(
    layout_b,
    {static_cast<std::uint64_t const*>(d_b), static_cast<std::int64_t>(G), false},
    stream,
    mr);

  sirius::late_mat::pinned_column_view origin_a;
  origin_a.dtype          = cudf::data_type{cudf::type_id::INT32};
  origin_a.pin_generation = 11;
  origin_a.batches.push_back({nullptr, 0, a_col->view(), n_a});
  sirius::late_mat::pinned_column_view origin_b;
  origin_b.dtype          = cudf::data_type{cudf::type_id::INT64};
  origin_b.pin_generation = 12;
  origin_b.batches.push_back({nullptr, 0, b_col->view(), n_b});

  auto got_a = sirius::late_mat::materialize(origin_a, *sel_a, stream, mr);
  auto got_b = sirius::late_mat::materialize(origin_b, *sel_b, stream, mr);
  cudaStreamSynchronize(stream.value());

  REQUIRE_MSG(got_a && got_a->size() == static_cast<cudf::size_type>(G) && got_b &&
                got_b->size() == static_cast<cudf::size_type>(G),
              "multi-origin output sizes");
  std::vector<std::int32_t> ha(G);
  std::vector<std::int64_t> hb(G);
  cudaMemcpy(ha.data(), got_a->view().head<void>(), G * 4, cudaMemcpyDeviceToHost);
  cudaMemcpy(hb.data(), got_b->view().head<void>(), G * 8, cudaMemcpyDeviceToHost);
  bool ok = true;
  for (std::size_t g = 0; g < G; ++g) {
    ok &= ha[g] == a_data[static_cast<std::size_t>(a_ids[g])];
    ok &= hb[g] == b_data[static_cast<std::size_t>(b_ids[g])];
  }
  REQUIRE_MSG(ok, "multi-origin per-group values (row alignment across origins)");
  cudaFree(d_a);
  cudaFree(d_b);
  std::printf("PASS: multi-origin output-group materialization (2 origins, 1 port)\n");
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
    run_sparse_roundtrip<std::int32_t>("int32_t", 8035, 2526, /*delta_root=*/false, arch);
    run_sparse_roundtrip<std::int64_t>(
      "int64_t", 3'000'000'000LL, 5052, /*delta_root=*/false, arch);
    run_sparse_roundtrip<std::int64_t>("int64_t", 1'000'000LL, 997, /*delta_root=*/true, arch);
    run_sparse_dict_gather(/*key_width=*/1, arch);
    run_sparse_dict_gather(/*key_width=*/4, arch);
    run_sort_unique();
    run_raw_fastpath();
    run_capture_contract();
    run_stored_dtype_retag();
    run_rowid_emission();
    run_multi_source_gather();
    run_multi_origin_groups();
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
