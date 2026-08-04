// Late-materialization row-set / sparse-decode EDGE GEOMETRIES.
//
// Companion to test_late_mat_row_decode.cpp (which covers mid-density
// patterns on multi-chunk shapes); this file pins down the boundary
// geometries an engine integration can hit but the density sweeps do not:
//
//   1. Chunk-boundary row counts: n = 1023 / 1024 / 1025 / 2048 / 7*1024+1
//      (1-row tail chunk) for CSR construction, int32 expansion and mask
//      expansion — all vs host references, mask buffers pre-filled 0xFF to
//      prove tail/untouched-chunk zeroing at every n.
//   2. Degenerate selections at every geometry: empty (S=0), {row 0},
//      {row n-1} (last row of a partial tail), full density (S=n),
//      one-row-per-chunk at in-chunk position 1023 (max u16 offset in a full
//      chunk) and at the tail chunk's last row, alternating parity.
//   3. K8 sparse decode (bitpack root AND delta->bitpack staged-slab root)
//      at the same geometries: full-density CSR must equal the plain decode
//      byte-for-byte; a single survivor in the last row of a 1-row tail
//      chunk; one-per-chunk-last-row.
//   4. split_sorted_ids_by_batch: ids exactly at batch boundaries, an
//      empty (0-row) batch mid-layout, all ids in one batch, empty id list,
//      and the out-of-range contract prepare_selection relies on
//      (starts.back() < count iff an id lies past the table end).
//   5. global_slice_to_local roundtrip at a non-zero batch base.
//   6. sort_unique_global_ids: count=1, all-duplicates, already-sorted
//      input, and values straddling the exact 2^31 / 2^32 boundaries (the
//      u32-death case with the boundary values themselves present).
//
// GPU required (encode/decode kernels + NVRTC). Standalone-main harness,
// same pattern as the other tests in this directory. Additive only: no
// shipped code and no existing test is modified.

#include "codegen/codegen_bridge.hpp"
#include "codegen/decode/latemat_launch.hpp"
#include "codegen/jit/fused_tree.hpp"
#include "codegen/jit/nvrtc_compiler.hpp"
#include "codegen/selection/row_set.hpp"
#include "codegen/selection/selection.hpp"
#include "gpu_encode.hpp"

#include <cuda_runtime.h>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/mr/per_device_resource.hpp>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace jit = codegen::jit;
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

template <typename Element>
std::vector<Element> gen_data(std::int64_t n, std::int64_t base, std::int64_t range)
{
  std::vector<Element> v(static_cast<std::size_t>(n));
  for (std::int64_t i = 0; i < n; ++i) {
    std::uint64_t s = 0xC0FFEEull ^ (0xD1B54A32D192ED03ull * static_cast<std::uint64_t>(i));
    v[static_cast<std::size_t>(i)] = static_cast<Element>(
      base + static_cast<std::int64_t>(splitmix64(s) % static_cast<std::uint64_t>(range)));
  }
  return v;
}

// Host-reference chunk-CSR (same shape as test_late_mat_row_decode.cpp).
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

// Same host packed-layout compaction as the sibling tests.
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

// The degenerate selections of TEST-PLAN U1, per geometry.
std::vector<std::pair<std::string, std::vector<std::uint32_t>>> edge_selections(std::int64_t n)
{
  std::vector<std::pair<std::string, std::vector<std::uint32_t>>> out;
  out.emplace_back("empty", std::vector<std::uint32_t>{});
  out.emplace_back("row0", std::vector<std::uint32_t>{0u});
  out.emplace_back("last_row", std::vector<std::uint32_t>{static_cast<std::uint32_t>(n - 1)});
  {
    std::vector<std::uint32_t> full(static_cast<std::size_t>(n));
    for (std::int64_t i = 0; i < n; ++i) full[static_cast<std::size_t>(i)] =
      static_cast<std::uint32_t>(i);
    out.emplace_back("full_density", std::move(full));
  }
  {
    // Last row of every chunk: in-chunk position 1023 for full chunks, the
    // tail's true last row for a partial tail.
    std::vector<std::uint32_t> per_chunk;
    for (std::int64_t c = 0; c * kChunk < n; ++c) {
      const std::int64_t last = std::min(n, (c + 1) * kChunk) - 1;
      per_chunk.push_back(static_cast<std::uint32_t>(last));
    }
    out.emplace_back("chunk_last_rows", std::move(per_chunk));
  }
  {
    std::vector<std::uint32_t> parity;
    for (std::int64_t i = 0; i < n; i += 2) parity.push_back(static_cast<std::uint32_t>(i));
    out.emplace_back("even_rows", std::move(parity));
  }
  return out;
}

// ── 1+2. CSR construction / expansions at boundary geometries ───────────────
bool run_geometry_csr(std::int64_t n)
{
  const std::int64_t nc = codegen::num_chunks_for(n);
  const rmm::cuda_stream_view stream{};
  auto mr = rmm::mr::get_current_device_resource_ref();

  // Scratch device arena for uploads (reuse the encode harness allocator with
  // a trivial 1-value encode so we do not duplicate its RAII machinery).
  auto tree      = jit::FusedTree::make(OpKind::Bitpack);
  const std::vector<std::int32_t> dummy(1, 42);
  GpuEncoded enc = codegen_test::gpu_encode_tree<std::int32_t>(
    *tree, "int32_t", dummy.data(), 1, jit::arch_cc_for_current_device());

  for (auto const& [name, ids] : edge_selections(n)) {
    const HostCsr ref = host_csr(ids);

    const std::uint32_t* d_ids = nullptr;
    if (!ids.empty()) {
      d_ids = reinterpret_cast<const std::uint32_t*>(
        enc.upload_bytes(ids.data(), ids.size() * 4));
    }
    owned_chunk_row_set built;
    try {
      built = sirius::codegen::bucket_sorted_local_ids(
        d_ids, static_cast<std::int64_t>(ids.size()), n, stream, mr);
    } catch (const std::exception& e) {
      REQUIRE_MSG(false, "[n=%lld/%s] bucket_sorted_local_ids threw: %s",
                  static_cast<long long>(n), name.c_str(), e.what());
    }
    REQUIRE_MSG(built.num_rows == n, "[n=%lld/%s] num_rows not preserved",
                static_cast<long long>(n), name.c_str());
    REQUIRE_MSG(built.num_survivors == static_cast<std::int64_t>(ids.size()) &&
                  built.num_touched == static_cast<std::int64_t>(ref.chunks.size()),
                "[n=%lld/%s] counts S=%lld T=%lld vs host S=%zu T=%zu",
                static_cast<long long>(n), name.c_str(),
                static_cast<long long>(built.num_survivors),
                static_cast<long long>(built.num_touched), ids.size(), ref.chunks.size());
    if (!ids.empty()) {
      std::vector<std::uint32_t> got_chunks(ref.chunks.size());
      std::vector<std::uint32_t> got_offsets(ref.offsets.size());
      std::vector<std::uint16_t> got_in(ref.in_chunk.size());
      cudaMemcpy(got_chunks.data(), built.chunk_ids.data(), got_chunks.size() * 4,
                 cudaMemcpyDeviceToHost);
      cudaMemcpy(got_offsets.data(), built.chunk_out_offsets.data(), got_offsets.size() * 4,
                 cudaMemcpyDeviceToHost);
      cudaMemcpy(got_in.data(), built.in_chunk_offsets.data(), got_in.size() * 2,
                 cudaMemcpyDeviceToHost);
      REQUIRE_MSG(got_chunks == ref.chunks && got_offsets == ref.offsets &&
                    got_in == ref.in_chunk,
                  "[n=%lld/%s] device CSR != host CSR", static_cast<long long>(n),
                  name.c_str());

      // int32 expansion == the id list itself.
      CUdeviceptr d_exp = enc.alloc(ids.size() * 4);
      sirius::codegen::row_set_to_local_indices(
        built.view(), reinterpret_cast<std::int32_t*>(d_exp), stream);
      cudaStreamSynchronize(stream.value());
      std::vector<std::int32_t> got_exp(ids.size());
      cudaMemcpy(got_exp.data(), reinterpret_cast<const void*>(d_exp), ids.size() * 4,
                 cudaMemcpyDeviceToHost);
      bool exp_ok = true;
      for (std::size_t i = 0; i < ids.size(); ++i)
        exp_ok &= got_exp[i] == static_cast<std::int32_t>(ids[i]);
      REQUIRE_MSG(exp_ok, "[n=%lld/%s] int32 expansion mismatch", static_cast<long long>(n),
                  name.c_str());

      // Mask expansion: 0xFF prefill proves untouched-chunk AND tail zeroing —
      // load-bearing at n=1023/1025 (tail bits beyond n MUST be zero).
      const std::size_t nwords = static_cast<std::size_t>(nc) * kWordsPerChunk;
      CUdeviceptr d_mask       = enc.alloc(nwords * 4);
      cudaMemset(reinterpret_cast<void*>(d_mask), 0xFF, nwords * 4);
      CUdeviceptr d_choffs = enc.alloc(static_cast<std::size_t>(nc + 1) * 4);
      cudaMemset(reinterpret_cast<void*>(d_choffs), 0xFF, static_cast<std::size_t>(nc + 1) * 4);
      sirius::codegen::row_set_to_mask(built.view(),
                                       reinterpret_cast<std::uint32_t*>(d_mask),
                                       reinterpret_cast<std::uint32_t*>(d_choffs), stream, mr);
      cudaStreamSynchronize(stream.value());
      std::vector<std::uint32_t> ref_mask(nwords, 0u);
      for (auto id : ids) ref_mask[id / 32] |= 1u << (id % 32);
      std::vector<std::uint32_t> ref_choffs(static_cast<std::size_t>(nc) + 1, 0u);
      std::uint32_t acc = 0;
      for (std::int64_t c = 0; c < nc; ++c) {
        ref_choffs[static_cast<std::size_t>(c)] = acc;
        for (int w = 0; w < kWordsPerChunk; ++w)
          acc += static_cast<std::uint32_t>(
            __builtin_popcount(ref_mask[static_cast<std::size_t>(c) * kWordsPerChunk + w]));
      }
      ref_choffs[static_cast<std::size_t>(nc)] = acc;
      std::vector<std::uint32_t> got_mask(nwords);
      std::vector<std::uint32_t> got_choffs(static_cast<std::size_t>(nc) + 1);
      cudaMemcpy(got_mask.data(), reinterpret_cast<const void*>(d_mask), nwords * 4,
                 cudaMemcpyDeviceToHost);
      cudaMemcpy(got_choffs.data(), reinterpret_cast<const void*>(d_choffs),
                 got_choffs.size() * 4, cudaMemcpyDeviceToHost);
      REQUIRE_MSG(got_mask == ref_mask && got_choffs == ref_choffs,
                  "[n=%lld/%s] mask expansion mismatch (tail/untouched zeroing?)",
                  static_cast<long long>(n), name.c_str());
    }
  }
  std::printf("PASS: CSR/expansion edges at n=%lld\n", static_cast<long long>(n));
  return true;
}

// ── 3. K8 roundtrip at boundary geometries ──────────────────────────────────
template <typename Element>
bool run_k8_edges(std::int64_t n, bool delta_root, int arch)
{
  const std::int64_t nc = codegen::num_chunks_for(n);
  const rmm::cuda_stream_view stream{};
  auto mr = rmm::mr::get_current_device_resource_ref();

  const std::vector<Element> data = gen_data<Element>(n, 5000, 1789);
  auto tree =
    delta_root
      ? jit::FusedTree::make(OpKind::Delta,
                             {{"differences", jit::FusedTree::make(OpKind::Bitpack)}})
      : jit::FusedTree::make(OpKind::Bitpack);
  GpuEncoded enc =
    codegen_test::gpu_encode_tree<Element>(*tree, delta_root ? "int64_t" : "int32_t",
                                           data.data(), n, arch);
  const std::int32_t bp_node = delta_root ? 1 : 0;
  if (!compact_packed_on_host(enc, nc, bp_node)) return false;
  const char* dtype = delta_root ? "int64_t" : "int32_t";

  // Plain reference through the production launcher.
  CUdeviceptr d_plain = enc.alloc(static_cast<std::size_t>(n) * sizeof(Element));
  REQUIRE_MSG(simpatico::launch_decode_fused_tree(
                *tree, enc.buffers, dtype, n, reinterpret_cast<void*>(d_plain), stream),
              "[n=%lld] plain decode failed", static_cast<long long>(n));
  cudaStreamSynchronize(stream.value());
  std::vector<Element> plain(static_cast<std::size_t>(n));
  cudaMemcpy(plain.data(), reinterpret_cast<const void*>(d_plain),
             plain.size() * sizeof(Element), cudaMemcpyDeviceToHost);
  REQUIRE_MSG(plain == data, "[n=%lld] plain decode != input", static_cast<long long>(n));

  for (auto const& [name, ids] : edge_selections(n)) {
    if (ids.empty()) continue;  // launcher contract: S>0 (engine skips S=0 batches)
    const std::uint32_t* d_ids = reinterpret_cast<const std::uint32_t*>(
      enc.upload_bytes(ids.data(), ids.size() * 4));
    owned_chunk_row_set built = sirius::codegen::bucket_sorted_local_ids(
      d_ids, static_cast<std::int64_t>(ids.size()), n, stream, mr);

    CUdeviceptr d_out = enc.alloc(ids.size() * sizeof(Element));
    cudaMemset(reinterpret_cast<void*>(d_out), 0xAB, ids.size() * sizeof(Element));
    REQUIRE_MSG(simpatico::launch_decode_fused_tree_sparse_rows(
                  *tree, enc.buffers, dtype, n, built.view(),
                  reinterpret_cast<void*>(d_out), stream),
                "[n=%lld/%s/%s] sparse_rows launch failed", static_cast<long long>(n),
                delta_root ? "delta" : "bitpack", name.c_str());
    cudaStreamSynchronize(stream.value());
    std::vector<Element> got(ids.size());
    cudaMemcpy(got.data(), reinterpret_cast<const void*>(d_out), got.size() * sizeof(Element),
               cudaMemcpyDeviceToHost);
    std::vector<Element> expect;
    expect.reserve(ids.size());
    for (auto id : ids) expect.push_back(data[id]);
    REQUIRE_MSG(got == expect, "[n=%lld/%s/%s] K8 output != host gather (%zu rows)",
                static_cast<long long>(n), delta_root ? "delta" : "bitpack", name.c_str(),
                ids.size());
  }
  std::printf("PASS: K8 edges n=%lld (%s root)\n", static_cast<long long>(n),
              delta_root ? "delta" : "bitpack");
  return true;
}

// ── 4+5. Batch split + global->local at boundaries ──────────────────────────
bool run_batch_split()
{
  const rmm::cuda_stream_view stream{};
  auto mr = rmm::mr::get_current_device_resource_ref();

  // Layout: batch0 1000 rows, batch1 EMPTY, batch2 2048 rows, batch3 1025 rows.
  const std::vector<std::int64_t> row_start = {0, 1000, 1000, 3048, 4073};

  // Boundary-hugging sorted ids: batch firsts, batch lasts, one interior.
  const std::vector<std::uint64_t> ids = {0, 999, 1000, 2047, 3047, 3048, 4072};
  void* d_ids                          = nullptr;
  cudaMalloc(&d_ids, ids.size() * 8);
  cudaMemcpy(d_ids, ids.data(), ids.size() * 8, cudaMemcpyHostToDevice);

  auto starts = sirius::codegen::split_sorted_ids_by_batch(
    static_cast<std::uint64_t const*>(d_ids), static_cast<std::int64_t>(ids.size()),
    /*count_dev=*/nullptr, row_start, /*count_out=*/nullptr, stream, mr);
  const std::vector<std::int64_t> expect = {0, 2, 2, 5, 7};
  REQUIRE_MSG(starts == expect, "boundary split starts mismatch");

  // Empty id list.
  auto empty_starts = sirius::codegen::split_sorted_ids_by_batch(
    static_cast<std::uint64_t const*>(d_ids), 0, /*count_dev=*/nullptr, row_start,
    /*count_out=*/nullptr, stream, mr);
  REQUIRE_MSG(!empty_starts.empty() && empty_starts.back() == 0,
              "empty split must end at 0");

  // All ids in ONE batch (batch2).
  const std::vector<std::uint64_t> one_batch = {1000, 1500, 3047};
  cudaMemcpy(d_ids, one_batch.data(), one_batch.size() * 8, cudaMemcpyHostToDevice);
  starts = sirius::codegen::split_sorted_ids_by_batch(
    static_cast<std::uint64_t const*>(d_ids), 3, /*count_dev=*/nullptr, row_start,
    /*count_out=*/nullptr, stream, mr);
  REQUIRE_MSG((starts == std::vector<std::int64_t>{0, 0, 0, 3, 3}),
              "one-batch split starts mismatch");

  // Out-of-range id: starts.back() must fall short of count (prepare's check).
  const std::vector<std::uint64_t> oob = {0, 4073};
  cudaMemcpy(d_ids, oob.data(), oob.size() * 8, cudaMemcpyHostToDevice);
  starts = sirius::codegen::split_sorted_ids_by_batch(
    static_cast<std::uint64_t const*>(d_ids), 2, /*count_dev=*/nullptr, row_start,
    /*count_out=*/nullptr, stream, mr);
  REQUIRE_MSG(starts.back() == 1, "out-of-range id must be excluded by starts.back()");

  // global_slice_to_local at a non-zero base (batch3: base 3048, 1025 rows).
  const std::vector<std::uint64_t> g = {3048, 3049, 4072};
  cudaMemcpy(d_ids, g.data(), g.size() * 8, cudaMemcpyHostToDevice);
  void* d_local = nullptr;
  cudaMalloc(&d_local, g.size() * 4);
  sirius::codegen::global_slice_to_local(static_cast<std::uint64_t const*>(d_ids),
                                         static_cast<std::int64_t>(g.size()), 3048,
                                         static_cast<std::uint32_t*>(d_local), stream);
  cudaStreamSynchronize(stream.value());
  std::vector<std::uint32_t> got(g.size());
  cudaMemcpy(got.data(), d_local, got.size() * 4, cudaMemcpyDeviceToHost);
  REQUIRE_MSG((got == std::vector<std::uint32_t>{0, 1, 1024}),
              "global_slice_to_local mismatch");
  cudaFree(d_ids);
  cudaFree(d_local);
  std::printf("PASS: batch split + global->local boundaries\n");
  return true;
}

// ── 6. sort_unique_global_ids degenerate inputs + 2^31/2^32 straddles ───────
bool run_sort_unique_edges()
{
  const rmm::cuda_stream_view stream{};
  auto mr = rmm::mr::get_current_device_resource_ref();

  auto check = [&](std::vector<std::uint64_t> const& ids, const char* name) -> bool {
    void* d_ids = nullptr;
    cudaMalloc(&d_ids, ids.size() * 8);
    cudaMemcpy(d_ids, ids.data(), ids.size() * 8, cudaMemcpyHostToDevice);
    auto res = sirius::codegen::sort_unique_global_ids(
      static_cast<std::uint64_t const*>(d_ids), static_cast<std::int64_t>(ids.size()),
      stream, mr);
    std::vector<std::uint64_t> ref(ids);
    std::sort(ref.begin(), ref.end());
    ref.erase(std::unique(ref.begin(), ref.end()), ref.end());
    // Async contract (W3 sync-surgery rev): the unique count is device-
    // resident (count_dev); ids buffer is worst-case sized, valid prefix only.
    std::int32_t unique_count = 0;
    cudaStreamSynchronize(stream.value());
    cudaMemcpy(&unique_count, res.count_dev.data(), 4, cudaMemcpyDeviceToHost);
    REQUIRE_MSG(unique_count == static_cast<std::int32_t>(ref.size()),
                "[%s] unique_count %d != %zu", name, unique_count, ref.size());
    std::vector<std::uint64_t> got_u(ref.size());
    std::vector<std::int32_t> got_r(ids.size());
    cudaMemcpy(got_u.data(), res.ids.data(), got_u.size() * 8, cudaMemcpyDeviceToHost);
    cudaMemcpy(got_r.data(), res.restore_rank.data(), got_r.size() * 4,
               cudaMemcpyDeviceToHost);
    REQUIRE_MSG(got_u == ref, "[%s] sorted-unique mismatch", name);
    for (std::size_t i = 0; i < ids.size(); ++i) {
      REQUIRE_MSG(got_u[static_cast<std::size_t>(got_r[i])] == ids[i],
                  "[%s] restore rank %zu wrong", name, i);
    }
    cudaFree(d_ids);
    return true;
  };

  bool ok = true;
  ok &= check({7}, "single");
  ok &= check(std::vector<std::uint64_t>(1000, 42), "all_duplicates");
  {
    std::vector<std::uint64_t> sorted(5000);
    for (std::size_t i = 0; i < sorted.size(); ++i) sorted[i] = 3 * i;
    ok &= check(sorted, "already_sorted");
  }
  ok &= check({(1ull << 31) - 1, (1ull << 31), (1ull << 32) - 1, (1ull << 32),
               (1ull << 32) + 5, 0, (1ull << 31), (1ull << 32)},
              "pow2_straddle");
  if (ok) std::printf("PASS: sort_unique edge inputs\n");
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
    for (std::int64_t n : {std::int64_t{1023}, std::int64_t{1024}, std::int64_t{1025},
                           std::int64_t{2048}, std::int64_t{7 * 1024 + 1}}) {
      run_geometry_csr(n);
    }
    for (std::int64_t n : {std::int64_t{1023}, std::int64_t{1024}, std::int64_t{1025},
                           std::int64_t{7 * 1024 + 1}}) {
      run_k8_edges<std::int32_t>(n, /*delta_root=*/false, arch);
      run_k8_edges<std::int64_t>(n, /*delta_root=*/true, arch);
    }
    run_batch_split();
    run_sort_unique_edges();
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
