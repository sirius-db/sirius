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

#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>

#include <cuda_runtime.h>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/mr/per_device_resource.hpp>

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
std::vector<std::uint32_t> gen_row_ids(std::int64_t n,
                                       unsigned keep_permille,
                                       unsigned seed,
                                       std::int64_t zero_chunk,
                                       bool cluster_runs)
{
  std::vector<std::uint32_t> ids;
  for (std::int64_t r = 0; r < n; ++r) {
    if (r / kChunk == zero_chunk) continue;
    std::uint64_t s = seed ^ (0x9E3779B97F4A7C15ull *
                              static_cast<std::uint64_t>((cluster_runs ? r / 64 : r) + 1));
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
    const std::size_t live =
      (static_cast<std::size_t>(count[static_cast<std::size_t>(c)]) *
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
  set.chunk_ids = reinterpret_cast<std::uint32_t*>(
    enc.upload_bytes(csr.chunks.data(), csr.chunks.size() * 4));
  set.chunk_out_offsets = reinterpret_cast<std::uint32_t*>(
    enc.upload_bytes(csr.offsets.data(), csr.offsets.size() * 4));
  set.in_chunk_offsets = reinterpret_cast<std::uint16_t*>(
    enc.upload_bytes(csr.in_chunk.data(), csr.in_chunk.size() * 2));
  return set;
}

// ── 1. Render contract ───────────────────────────────────────────────────────
bool render_checks()
{
  auto bp = jit::FusedTree::make(OpKind::Bitpack);
  const cdj::DecodeKernelSpec plain = cdj::render(*bp, "int32_t", 8);
  const cdj::DecodeKernelSpec k8 =
    cdj::render(*bp, "int32_t", 8, cdj::DecodeVariant::sparse_index_consume);
  REQUIRE_MSG(k8.entry_symbol != plain.entry_symbol &&
                k8.entry_symbol.find("_sparse_index") != std::string::npos,
              "sparse_index_consume must get its own entry symbol");
  REQUIRE_MSG(k8.source.find("chunk_list[blockIdx.x]") != std::string::npos,
              "sparse variant must derive chunk_id from chunk_list");
  REQUIRE_MSG(plain.source.find("chunk_list") == std::string::npos &&
                plain.source.find("in_chunk_offsets") == std::string::npos,
              "plain render must stay byte-free of sparse artifacts");

  // Compositional delta root through the same variant (staged slab).
  auto delta = jit::FusedTree::make(
    OpKind::Delta, {{"differences", jit::FusedTree::make(OpKind::Bitpack)}});
  const cdj::DecodeKernelSpec k8d =
    cdj::render(*delta, "int64_t", 8, cdj::DecodeVariant::sparse_index_consume);
  REQUIRE_MSG(k8d.source.find("K8") != std::string::npos,
              "delta-rooted sparse_index_consume must render (compositional)");

  const cdj::DecodeKernelSpec k5s =
    cdj::render(*bp, "int32_t", 8, cdj::DecodeVariant::sparse_dict_gather);
  REQUIRE_MSG(k5s.entry_symbol.find("_sparse_dict") != std::string::npos &&
                k5s.source.find("keys_chars") != std::string::npos,
              "sparse_dict_gather render contract");
  bool rejected = false;
  try {
    (void)cdj::render(*delta, "int64_t", 8, cdj::DecodeVariant::sparse_dict_gather);
  } catch (const cdj::RenderError&) {
    rejected = true;
  }
  REQUIRE_MSG(rejected, "sparse_dict_gather on a Delta root must throw RenderError");

  const cdj::DecodeKernelSpec k6s =
    cdj::render(*bp, "int64_t", 8, cdj::DecodeVariant::sparse_str_meta);
  REQUIRE_MSG(k6s.entry_symbol.find("_sparse_str_meta") != std::string::npos &&
                k6s.source.find("len_out") != std::string::npos &&
                k6s.source.find("chunk_list[blockIdx.x]") != std::string::npos,
              "sparse_str_meta render contract");
  rejected = false;
  try {
    auto rle = jit::FusedTree::make(
      OpKind::Rle, {{"values", jit::FusedTree::make(OpKind::Bitpack)},
                    {"counts", jit::FusedTree::make(OpKind::Bitpack)}});
    (void)cdj::render(*rle, "int32_t", 8, cdj::DecodeVariant::sparse_str_meta);
  } catch (const cdj::RenderError&) {
    rejected = true;
  }
  REQUIRE_MSG(rejected, "sparse_str_meta on an Rle root must throw RenderError");

  std::printf("PASS: sparse render contract checks\n");
  return true;
}

// ── 2+3. Row-set construction + K8 roundtrip ────────────────────────────────
template <typename Element>
bool run_sparse_roundtrip(const std::string& dtype,
                          std::int64_t base,
                          std::int64_t range,
                          bool delta_root,
                          int arch)
{
  const std::int64_t n  = 7 * kChunk + 700;  // partial tail chunk
  const std::int64_t nc = codegen::num_chunks_for(n);
  const rmm::cuda_stream_view stream{};
  auto mr = rmm::mr::get_current_device_resource_ref();

  const std::vector<Element> data = gen_data<Element>(n, base, range);
  auto tree =
    delta_root
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
              "[%s] plain decode failed", dtype.c_str());
  std::vector<Element> plain(static_cast<std::size_t>(n));
  cudaMemcpy(plain.data(), reinterpret_cast<const void*>(d_plain),
             plain.size() * sizeof(Element), cudaMemcpyDeviceToHost);
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
    owned_chunk_row_set built = sirius::codegen::bucket_sorted_local_ids(
      reinterpret_cast<std::uint32_t const*>(d_ids), static_cast<std::int64_t>(ids.size()), n,
      stream, mr);
    REQUIRE_MSG(built.num_touched == static_cast<std::int64_t>(ref_csr.chunks.size()) &&
                  built.num_survivors == static_cast<std::int64_t>(ids.size()),
                "[%s/%s] bucket counts (T=%lld vs %zu)", dtype.c_str(), tc.name,
                static_cast<long long>(built.num_touched), ref_csr.chunks.size());
    std::vector<std::uint32_t> got_chunks(ref_csr.chunks.size());
    std::vector<std::uint32_t> got_offsets(ref_csr.offsets.size());
    std::vector<std::uint16_t> got_in(ref_csr.in_chunk.size());
    cudaMemcpy(got_chunks.data(), built.chunk_ids.data(), got_chunks.size() * 4,
               cudaMemcpyDeviceToHost);
    cudaMemcpy(got_offsets.data(), built.chunk_out_offsets.data(), got_offsets.size() * 4,
               cudaMemcpyDeviceToHost);
    cudaMemcpy(got_in.data(), built.in_chunk_offsets.data(), got_in.size() * 2,
               cudaMemcpyDeviceToHost);
    REQUIRE_MSG(got_chunks == ref_csr.chunks && got_offsets == ref_csr.offsets &&
                  got_in == ref_csr.in_chunk,
                "[%s/%s] device CSR != host CSR", dtype.c_str(), tc.name);

    // int32 expansion and mask expansion vs host references.
    CUdeviceptr d_exp = enc.alloc(ids.size() * 4);
    sirius::codegen::row_set_to_local_indices(
      built.view(), reinterpret_cast<std::int32_t*>(d_exp), stream);
    std::vector<std::int32_t> got_exp(ids.size());
    cudaStreamSynchronize(stream.value());
    cudaMemcpy(got_exp.data(), reinterpret_cast<const void*>(d_exp), ids.size() * 4,
               cudaMemcpyDeviceToHost);
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
                                     reinterpret_cast<std::uint32_t*>(d_choffs), stream, mr);
    cudaStreamSynchronize(stream.value());
    std::vector<std::uint32_t> ref_mask(nwords, 0u);
    for (auto id : ids) ref_mask[id / 32] |= 1u << (id % 32);
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
    cudaMemcpy(got_mask.data(), reinterpret_cast<const void*>(d_mask), nwords * 4,
               cudaMemcpyDeviceToHost);
    cudaMemcpy(got_choffs.data(), reinterpret_cast<const void*>(d_choffs),
               got_choffs.size() * 4, cudaMemcpyDeviceToHost);
    REQUIRE_MSG(got_mask == ref_mask && got_choffs == ref_choffs,
                "[%s/%s] mask expansion mismatch", dtype.c_str(), tc.name);

    // K8 sparse decode vs host gather of the plain reference.
    CUdeviceptr d_out = enc.alloc(ids.size() * sizeof(Element));
    cudaMemset(reinterpret_cast<void*>(d_out), 0xAB, ids.size() * sizeof(Element));
    REQUIRE_MSG(simpatico::launch_decode_fused_tree_sparse_rows(
                  *tree, enc.buffers, dtype.c_str(), n, built.view(),
                  reinterpret_cast<void*>(d_out), stream),
                "[%s/%s] sparse_rows launch failed", dtype.c_str(), tc.name);
    cudaStreamSynchronize(stream.value());  // launcher is stream-ordered by contract
    std::vector<Element> got(ids.size());
    cudaMemcpy(got.data(), reinterpret_cast<const void*>(d_out), got.size() * sizeof(Element),
               cudaMemcpyDeviceToHost);
    std::vector<Element> expect;
    expect.reserve(ids.size());
    for (auto id : ids) expect.push_back(data[id]);
    REQUIRE_MSG(got == expect, "[%s/%s] K8 output != host gather (%zu rows)", dtype.c_str(),
                tc.name, ids.size());
  }

  std::printf("PASS: sparse roundtrip %s (%s root)\n", dtype.c_str(),
              delta_root ? "delta" : "bitpack");
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
  auto tree = jit::FusedTree::make(OpKind::Bitpack);
  GpuEncoded enc =
    codegen_test::gpu_encode_tree<std::int32_t>(*tree, "int32_t", codes.data(), n, arch);
  if (!compact_packed_on_host(enc, nc)) return false;
  CUdeviceptr d_keys = enc.upload_bytes(keys.data(), keys.size());

  const std::vector<std::uint32_t> ids = gen_row_ids(n, 37, 0xD1C7, /*zero_chunk=*/1, false);
  const HostCsr csr                    = host_csr(ids);
  chunk_row_set set                    = upload_csr(enc, csr, n);

  CUdeviceptr d_out = enc.alloc(ids.size() * static_cast<std::size_t>(key_width));
  cudaMemset(reinterpret_cast<void*>(d_out), 0xEE, ids.size() * key_width);
  REQUIRE_MSG(simpatico::launch_decode_fused_tree_sparse_dict_gather(
                *tree, enc.buffers, "int32_t", n, set,
                reinterpret_cast<const void*>(d_keys), key_width,
                reinterpret_cast<void*>(d_out), stream),
              "[w=%d] sparse_dict_gather launch failed", key_width);
  cudaStreamSynchronize(stream.value());
  std::vector<char> got(ids.size() * static_cast<std::size_t>(key_width));
  cudaMemcpy(got.data(), reinterpret_cast<const void*>(d_out), got.size(),
             cudaMemcpyDeviceToHost);
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
    static_cast<std::uint64_t const*>(d_ids), static_cast<std::int64_t>(ids.size()), stream,
    mr);

  std::vector<std::uint64_t> ref_sorted(ids);
  std::sort(ref_sorted.begin(), ref_sorted.end());
  ref_sorted.erase(std::unique(ref_sorted.begin(), ref_sorted.end()), ref_sorted.end());
  // Async contract (sync surgery): unique count is device-resident; the ids
  // buffer is worst-case sized with the first unique_count entries valid.
  std::int32_t unique_count = 0;
  cudaStreamSynchronize(stream.value());
  cudaMemcpy(&unique_count, res.count_dev.data(), 4, cudaMemcpyDeviceToHost);
  REQUIRE_MSG(unique_count == static_cast<std::int32_t>(ref_sorted.size()),
              "unique_count %d != %zu", unique_count, ref_sorted.size());
  std::vector<std::uint64_t> got_unique(ref_sorted.size());
  std::vector<std::int32_t> got_rank(ids.size());
  cudaMemcpy(got_unique.data(), res.ids.data(), got_unique.size() * 8, cudaMemcpyDeviceToHost);
  cudaMemcpy(got_rank.data(), res.restore_rank.data(), got_rank.size() * 4,
             cudaMemcpyDeviceToHost);
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
  auto mr = rmm::mr::get_current_device_resource_ref();
  const std::int64_t n = 3 * kChunk + 421;

  // Host data + device cudf column (uncompressed origin).
  std::vector<std::int32_t> data(static_cast<std::size_t>(n));
  for (std::int64_t i = 0; i < n; ++i) {
    data[static_cast<std::size_t>(i)] = static_cast<std::int32_t>((i * 2654435761u) % 100000);
  }
  auto col = cudf::make_fixed_width_column(cudf::data_type{cudf::type_id::INT32},
                                           static_cast<cudf::size_type>(n),
                                           cudf::mask_state::UNALLOCATED, stream, mr);
  cudaMemcpy(col->mutable_view().head<void>(), data.data(), data.size() * 4,
             cudaMemcpyHostToDevice);

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
  for (auto id : ids) expect.push_back(data[static_cast<std::size_t>(id)]);

  auto check = [&](std::unique_ptr<cudf::column> const& got, char const* what) {
    REQUIRE_MSG(got && got->size() == static_cast<cudf::size_type>(ids.size()),
                "%s: wrong output size", what);
    std::vector<std::int32_t> host(ids.size());
    cudaStreamSynchronize(stream.value());
    cudaMemcpy(host.data(), got->view().head<void>(), host.size() * 4, cudaMemcpyDeviceToHost);
    REQUIRE_MSG(host == expect, "%s: output != host gather (order/dups)", what);
    return true;
  };

  // (a) Single-batch layout: prepare must take the raw path (no device work).
  {
    auto layout = sirius::late_mat::pinned_table_layout::from_batch_rows({n}, 7);
    sirius::late_mat::row_id_list list{static_cast<std::uint64_t const*>(d_ids),
                                       static_cast<std::int64_t>(ids.size()), false};
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
    sirius::late_mat::row_id_list list{static_cast<std::uint64_t const*>(d_ids),
                                       static_cast<std::int64_t>(ids.size()), false};
    auto sel = sirius::late_mat::prepare_selection(layout, list, stream, mr);
    REQUIRE_MSG(sel->raw_ids == nullptr, "multi-batch prepare must take the canonical path");

    auto v0 = cudf::slice(col->view(), {0, static_cast<cudf::size_type>(rows0),
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
    REQUIRE_MSG(r.mask_words.size() == 0 && r.chunk_offsets.size() == 0 &&
                  r.row_indices.size() == 0,
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
  for (auto st : {scan_filter_status::bailed_high_selectivity, scan_filter_status::refused,
                  scan_filter_status::failed}) {
    auto r   = make_result(st, true);
    auto cap = sirius::codegen::capture_scan_filter_selection(std::move(r), stream);
    REQUIRE_MSG(!static_cast<bool>(cap), "non-applied status %d must not capture",
                static_cast<int>(st));
    REQUIRE_MSG(r.mask_words.size() != 0 && r.chunk_offsets.size() != 0 &&
                  r.row_indices.size() != 0,
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
bool run_stored_dtype_case(char const* name,
                           cudf::data_type stored,
                           char const* dsl,
                           std::int64_t base,
                           std::int64_t range)
{
  const rmm::cuda_stream_view stream{};
  auto mr = rmm::mr::get_current_device_resource_ref();
  const std::int64_t n = 2 * kChunk + 77;

  std::vector<Storage> raw(static_cast<std::size_t>(n));
  std::uint64_t s = 0x5CA1E ^ static_cast<std::uint64_t>(range);
  for (std::int64_t i = 0; i < n; ++i) {
    raw[static_cast<std::size_t>(i)] = static_cast<Storage>(
      base + static_cast<std::int64_t>(splitmix64(s) % static_cast<std::uint64_t>(range)));
  }
  auto col = cudf::make_fixed_width_column(stored, static_cast<cudf::size_type>(n),
                                           cudf::mask_state::UNALLOCATED, stream, mr);
  cudaMemcpy(col->mutable_view().head<void>(), raw.data(), raw.size() * sizeof(Storage),
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
  sirius::late_mat::row_id_list list{static_cast<std::uint64_t const*>(d_ids),
                                     static_cast<std::int64_t>(ids.size()), false};
  auto sel = sirius::late_mat::prepare_selection(layout, list, stream, mr);

  sirius::late_mat::pinned_column_view origin;
  origin.dtype          = stored;
  origin.pin_generation = 3;
  origin.batches.push_back({&ct, 0, cudf::column_view{}, n});
  auto got = sirius::late_mat::materialize(origin, *sel, stream, mr);
  cudaStreamSynchronize(stream.value());

  REQUIRE_MSG(got && got->size() == static_cast<cudf::size_type>(ids.size()),
              "[%s] wrong output size", name);
  REQUIRE_MSG(got->type() == stored,
              "[%s] stored dtype NOT re-tagged (got id=%d scale=%d, want id=%d scale=%d) — "
              "the q9 arm-C class",
              name, static_cast<int>(got->type().id()), got->type().scale(),
              static_cast<int>(stored.id()), stored.scale());
  std::vector<Storage> host(ids.size());
  cudaMemcpy(host.data(), got->view().head<void>(), host.size() * sizeof(Storage),
             cudaMemcpyDeviceToHost);
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
  ok &= run_stored_dtype_case<std::int64_t>(
    "decimal64_delta", data_type{type_id::DECIMAL64, -2}, "input -> delta -> differences\n",
    5000000, 700);
  ok &= run_stored_dtype_case<std::int32_t>(
    "decimal32_bitpack", data_type{type_id::DECIMAL32, -2}, "input -> bitpack\n", 20000, 3000);
  ok &= run_stored_dtype_case<std::int32_t>(
    "date32_bitpack", data_type{type_id::TIMESTAMP_DAYS}, "input -> bitpack\n", 8035, 2526);
  ok &= run_stored_dtype_case<std::int64_t>(
    "timestamp_us_delta", data_type{type_id::TIMESTAMP_MICROSECONDS},
    "input -> delta -> differences\n", 694224000000000LL, 86400000000LL);
  ok &= run_stored_dtype_case<std::int64_t>(
    "decimal64_identity_tierb", data_type{type_id::DECIMAL64, -2}, "input -> identity\n",
    100000, 9000);
  return ok;
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
    run_sparse_roundtrip<std::int64_t>("int64_t", 3'000'000'000LL, 5052, /*delta_root=*/false,
                                       arch);
    run_sparse_roundtrip<std::int64_t>("int64_t", 1'000'000LL, 997, /*delta_root=*/true, arch);
    run_sparse_dict_gather(/*key_width=*/1, arch);
    run_sparse_dict_gather(/*key_width=*/4, arch);
    run_sort_unique();
    run_raw_fastpath();
    run_capture_contract();
    run_stored_dtype_retag();
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
