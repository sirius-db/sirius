// SPDX-License-Identifier: Apache-2.0
//
// Unit test for `codegen::encode::compact_bitpack_into` —
// the scan+gather primitive that densifies an OverAllocate-laid-out
// bitpack `packed` buffer into a caller-provided destination.
//
// Fixture flow:
//
//   1. Run the same synthetic int32 column (n=4321) the encode-side
//      JIT test uses through the renderer + plain-CUDA encode kernel
//      to materialise the OverAllocate `packed` buffer + per-chunk
//      `live_words` array on device.
//
//   2. Compact once into a device buffer and *decode-verify* it (the
//      dense Compact-layout bitpack stream must reconstruct the input
//      column).  Those verified dense bytes become the golden reference
//      for the remaining destinations — a true encode->compact->decode
//      roundtrip, no host reference encoder.
//
//   3. Call `compact_bitpack_into` with four different destination
//      memory classes, each independently exercised:
//        a) device (rmm/cudaMalloc-style, cudaMemcpyDeviceToDevice)
//        b) host pinned (cudaHostAlloc)
//        c) host pageable (plain new uint8_t[])
//        d) managed (cudaMallocManaged)
//
//      For each, the test passes cudaMemcpyDefault and lets the
//      orchestrator auto-detect via cudaPointerGetAttributes — this
//      is the contract callers (the rep's compact_into() member,
//      future Rust wrapper) rely on.
//
//   4. Verify byte-equality against the host-reference dense output
//      on all four destinations.
//
// What this test does NOT cover:
//   * The rep's compact_into() / compact() member fns (lives in
//     bindings/cudf-sys/src/bitpack_compact_glue.cpp).
//   * The lazy-compaction cache in `packed_view_for_export()`.
//   * End-to-end roundtrip through the file writer (covered by the
//     simulator-side integration tests).
//
// Why .cu and not .cpp?  The fixture needs the encode kernel pipeline
// (which calls cuLaunch / NVRTC) to wire the OverAllocate source; using
// .cu lets us follow the existing test_*.cu pattern (add_codegen_test) and
// inherit the codegen_kernels object link automatically — which is
// where compact.cu lives.

#include "codegen/encode/compact.hpp"
#include "codegen/encode/jit/plain_compile.hpp"
#include "codegen/encode/jit/renderer.hpp"
#include "codegen/jit/fused_tree.hpp"
#include "jit_decode.hpp"
#include "test_utils.hpp"

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/mr/per_device_resource.hpp>

#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace cc  = codegen;
namespace cje = codegen::encode::jit;
namespace cjj = codegen::jit;
namespace cce = codegen::encode;

namespace {

int report_fail(const char* tag, const char* what, const std::string& details = "")
{
  std::fprintf(stderr, "FAIL [%s]: %s\n", tag, what);
  if (!details.empty()) { std::fprintf(stderr, "--- details ---\n%s\n", details.c_str()); }
  return 1;
}

std::string cu_err_str(CUresult r)
{
  const char* s = nullptr;
  cuGetErrorString(r, &s);
  return s ? std::string(s) : ("CUresult=" + std::to_string((int)r));
}

#define CU_OR_FAIL(call, tag, what)                                            \
  do {                                                                         \
    CUresult _r = (call);                                                      \
    if (_r != CUDA_SUCCESS) { return report_fail(tag, what, cu_err_str(_r)); } \
  } while (0)

#define CUDA_OR_FAIL(call, tag, what)                                                 \
  do {                                                                                \
    cudaError_t _e = (call);                                                          \
    if (_e != cudaSuccess) { return report_fail(tag, what, cudaGetErrorString(_e)); } \
  } while (0)

// Same synth column used by test_encode_bitpack_jit so we can compare
// fixtures across tests when debugging.
std::vector<std::int32_t> synth_data(std::int64_t n)
{
  std::vector<std::int32_t> data(static_cast<std::size_t>(n));
  for (std::int64_t i = 0; i < n; ++i) {
    std::int32_t cid = static_cast<std::int32_t>(i / cc::kChunkSize);
    std::int32_t pos = static_cast<std::int32_t>(i % cc::kChunkSize);
    switch (cid % 4) {
      case 0: data[i] = (pos & 1) ? 100 : 101; break;
      case 1: data[i] = static_cast<std::int32_t>(200 + (pos % 200)); break;
      case 2: data[i] = static_cast<std::int32_t>(-12345 + pos * 13); break;
      case 3: data[i] = static_cast<std::int32_t>(pos * 7 - 50000); break;
    }
  }
  return data;
}

const cje::BufferSpec& find_spec(const cje::EncodeKernelSpec& spec,
                                 std::int32_t node_id,
                                 const char* field)
{
  for (const auto& b : spec.buffers) {
    if (b.node_id == node_id && b.field == field) return b;
  }
  std::fprintf(stderr, "FATAL: encoder spec missing field 'node=%d %s'\n", node_id, field);
  std::abort();
}

// Per-test-destination context holding the device-side OverAllocate
// source + the precomputed live_packed_bytes so each destination
// variant runs against an identical input.
struct CompactFixture {
  std::int32_t num_chunks       = 0;
  std::int32_t stride_words     = 0;
  CUdeviceptr d_packed          = 0;  // OverAllocate source
  CUdeviceptr d_live_words      = 0;
  std::size_t live_packed_bytes = 0;

  // Decode metadata kept so we can decode-verify the compacted stream.
  CUdeviceptr d_chunk_min   = 0;
  CUdeviceptr d_chunk_count = 0;
  CUdeviceptr d_chunk_bits  = 0;
  CUdeviceptr d_bp_offsets  = 0;  // exclusive prefix of live_words

  // Host golden: dense packed bytes obtained by compacting once and
  // decode-verifying against the original column.
  std::vector<std::uint8_t> golden_bytes;

  // RAII owner for the device buffers above (RMM pool); freed on teardown.
  std::vector<rmm::device_buffer> owned;
};

// Build the OverAllocate fixture by running [Bitpack] through the
// encode-side JIT, then sum live_words on the host to get
// live_packed_bytes (mirrors what the encode bridge would do at rep
// construction time).
//
// Returns 0 on success; populates `fx` with the device pointers and
// the host golden buffer.  Caller frees fx.d_packed / fx.d_live_words.
int build_fixture(CompactFixture& fx,
                  const std::vector<std::int32_t>& data,
                  std::int64_t n,
                  const cje::CompileOptions& opts)
{
  constexpr const char* kTag = "fixture";
  using Element              = std::int32_t;
  fx.num_chunks              = cc::num_chunks_for(n);
  fx.stride_words =
    static_cast<std::int32_t>(cc::kChunkSize * sizeof(Element) / sizeof(std::uint32_t));

  // Tree + render + compile.
  auto tree          = cjj::FusedTree::make(cc::OpKind::Bitpack);
  tree->fixed_stride = true;

  cje::EncodeKernelSpec spec;
  try {
    spec = cje::render(*tree, "int32_t", fx.num_chunks);
  } catch (const std::exception& e) {
    return report_fail(kTag, "render", e.what());
  }

  cje::CompiledKernel kernel;
  try {
    kernel = cje::compile_plain_kernel(spec.source, spec.entry_symbol, opts);
  } catch (const cje::CompileError& e) {
    return report_fail(
      kTag, "compile", std::string("log:\n") + e.log + "\n--- source ---\n" + e.source);
  } catch (const std::exception& e) {
    return report_fail(kTag, "compile", e.what());
  }

  // GPU encode → OverAllocate source.  Keep chunk_min/count/bits on
  // device (we decode-verify the compacted stream below).
  cjj::ensure_cuda_context();

  const auto& min_spec   = find_spec(spec, 0, "chunk_min");
  const auto& count_spec = find_spec(spec, 0, "chunk_count");
  const auto& bits_spec  = find_spec(spec, 0, "chunk_bits");
  const auto& pkd_spec   = find_spec(spec, 0, "packed");
  // The renderer no longer emits a per-chunk live_words array — it now
  // accumulates the total live-word count into a 16-shard atomicAdd
  // counter (`lw_shards`) and expects callers to reconstruct per-chunk
  // offsets from chunk_bits + chunk_count.  We allocate + zero the shard
  // slab for the kernel, then rebuild the per-chunk live_words below.
  const auto& lws_spec = find_spec(spec, 0, "lw_shards");

  // Device buffers from the RMM pool.  fx.owned keeps the fixture buffers
  // alive for the whole test; flat_buf is local (only the encode reads it).
  const rmm::cuda_stream_view sv{};
  cudaStream_t cs = sv.value();
  auto mr         = rmm::mr::get_current_device_resource_ref();
  auto fx_alloc   = [&](std::size_t bytes) -> CUdeviceptr {
    fx.owned.emplace_back(bytes, sv, mr);
    return reinterpret_cast<CUdeviceptr>(fx.owned.back().data());
  };

  rmm::device_buffer flat_buf(static_cast<std::size_t>(n) * sizeof(Element), sv, mr);
  CUdeviceptr d_flat                = reinterpret_cast<CUdeviceptr>(flat_buf.data());
  fx.d_chunk_min                    = fx_alloc(min_spec.length * min_spec.elem_size);
  fx.d_chunk_count                  = fx_alloc(count_spec.length * count_spec.elem_size);
  fx.d_chunk_bits                   = fx_alloc(bits_spec.length * bits_spec.elem_size);
  fx.d_packed                       = fx_alloc(pkd_spec.length * pkd_spec.elem_size);
  const std::size_t lw_shards_bytes = lws_spec.length * lws_spec.elem_size;
  CUdeviceptr d_lw_shards           = fx_alloc(lw_shards_bytes);
  // Per-chunk live_words is no longer produced by the encoder; we
  // reconstruct it below and keep a device copy for compact_bitpack_into.
  fx.d_live_words = fx_alloc(static_cast<std::size_t>(fx.num_chunks) * sizeof(std::int32_t));

  CUDA_OR_FAIL(cudaMemcpyAsync(reinterpret_cast<void*>(d_flat),
                               data.data(),
                               static_cast<std::size_t>(n) * sizeof(Element),
                               cudaMemcpyHostToDevice,
                               cs),
               kTag,
               "memcpy flat H2D");
  // block_bitpack-mirror kernel does NOT clear `packed` in-kernel (atomicOr
  // packing requires a zeroed slot) and accumulates into `lw_shards`; both
  // must be pre-zeroed by the caller.
  CUDA_OR_FAIL(cudaMemsetAsync(
                 reinterpret_cast<void*>(fx.d_packed), 0, pkd_spec.length * pkd_spec.elem_size, cs),
               kTag,
               "memset packed");
  CUDA_OR_FAIL(cudaMemsetAsync(reinterpret_cast<void*>(d_lw_shards), 0, lw_shards_bytes, cs),
               kTag,
               "memset lw_shards");
  // Pool allocs + H2D are stream-ordered; sync before the driver encode launch.
  CUDA_OR_FAIL(cudaStreamSynchronize(cs), kTag, "sync before encode launch");

  long long total_n = n;
  void* args[]      = {&d_flat,
                       &total_n,
                       &fx.d_chunk_min,
                       &fx.d_chunk_count,
                       &fx.d_chunk_bits,
                       &fx.d_packed,
                       &d_lw_shards};
  CU_OR_FAIL(cuLaunchKernel(kernel.func,
                            static_cast<unsigned>(fx.num_chunks),
                            1,
                            1,
                            static_cast<unsigned>(spec.block_x),
                            static_cast<unsigned>(spec.block_y),
                            static_cast<unsigned>(spec.block_z),
                            static_cast<unsigned>(spec.shared_bytes),
                            cs,
                            args,
                            nullptr),
             kTag,
             "cuLaunchKernel");
  CU_OR_FAIL(cuCtxSynchronize(), kTag, "cuCtxSynchronize");
  // flat_buf RAII-freed at function exit.

  // Reconstruct per-chunk live_words = ceil(count*bits/32) (empty chunk
  // -> 1 sentinel word, matching the kernel's empty-chunk branch) from
  // the encoder's chunk_count + chunk_bits headers — the renderer no
  // longer materialises live_words.  Then build the exclusive prefix a
  // Compact-layout Bitpack decode expects as bp_offsets.
  std::vector<std::int32_t> host_cnt(fx.num_chunks);
  std::vector<std::uint8_t> host_bits(fx.num_chunks);
  CUDA_OR_FAIL(cudaMemcpy(host_cnt.data(),
                          reinterpret_cast<const void*>(fx.d_chunk_count),
                          fx.num_chunks * sizeof(std::int32_t),
                          cudaMemcpyDeviceToHost),
               kTag,
               "memcpy chunk_count D2H");
  CUDA_OR_FAIL(cudaMemcpy(host_bits.data(),
                          reinterpret_cast<const void*>(fx.d_chunk_bits),
                          fx.num_chunks * sizeof(std::uint8_t),
                          cudaMemcpyDeviceToHost),
               kTag,
               "memcpy chunk_bits D2H");
  std::vector<std::int32_t> host_lw(fx.num_chunks);
  for (std::int32_t c = 0; c < fx.num_chunks; ++c) {
    const std::int64_t cnt  = host_cnt[c];
    const std::int64_t bits = host_bits[c];
    host_lw[c]              = (cnt <= 0) ? 1 : static_cast<std::int32_t>((cnt * bits + 31) / 32);
  }
  std::vector<std::int32_t> host_off(static_cast<std::size_t>(fx.num_chunks) + 1);
  std::int32_t acc = 0;
  for (std::int32_t c = 0; c < fx.num_chunks; ++c) {
    host_off[c] = acc;
    acc += host_lw[c];
  }
  host_off[fx.num_chunks] = acc;
  fx.live_packed_bytes    = static_cast<std::size_t>(acc) * sizeof(std::uint32_t);

  // Upload the reconstructed per-chunk live_words for compact_bitpack_into.
  CUDA_OR_FAIL(cudaMemcpy(reinterpret_cast<void*>(fx.d_live_words),
                          host_lw.data(),
                          fx.num_chunks * sizeof(std::int32_t),
                          cudaMemcpyHostToDevice),
               kTag,
               "memcpy live_words H2D");

  // Cross-check the reconstruction against the kernel's own sharded
  // counter: sum(lw_shards[shard*kShardStride]) must equal total words.
  constexpr std::int32_t kMaxBitsShards = 16;
  constexpr std::int32_t kShardStride   = 32;
  std::vector<std::uint32_t> host_shards(static_cast<std::size_t>(kMaxBitsShards) * kShardStride);
  CUDA_OR_FAIL(cudaMemcpy(host_shards.data(),
                          reinterpret_cast<const void*>(d_lw_shards),
                          lw_shards_bytes,
                          cudaMemcpyDeviceToHost),
               kTag,
               "memcpy lw_shards D2H");
  std::uint64_t shard_total = 0;
  for (std::int32_t s = 0; s < kMaxBitsShards; ++s) {
    shard_total += host_shards[static_cast<std::size_t>(s) * kShardStride];
  }
  if (shard_total != static_cast<std::uint64_t>(acc)) {
    return report_fail(
      kTag,
      "live_words reconstruction != lw_shards sum",
      "reconstructed=" + std::to_string(acc) + " lw_shards_sum=" + std::to_string(shard_total));
  }

  fx.d_bp_offsets = fx_alloc(host_off.size() * sizeof(std::int32_t));
  CUDA_OR_FAIL(cudaMemcpy(reinterpret_cast<void*>(fx.d_bp_offsets),
                          host_off.data(),
                          host_off.size() * sizeof(std::int32_t),
                          cudaMemcpyHostToDevice),
               kTag,
               "memcpy bp_offsets H2D");

  // Compact once into a device buffer and decode-verify it against
  // the original column.  The verified dense bytes become the golden
  // reference every destination variant is compared against.
  //
  // simpatico_bitunpack_one always loads three consecutive uint32 words, so
  // decoding the last element touches up to two words past live_packed_bytes.
  // Allocate + zero that trailing slack to match the contract in
  // bitpack_compressed_representation::compact_in_place.
  constexpr std::size_t kDecodeGatherSlackBytes = 2 * sizeof(std::uint32_t);
  rmm::device_buffer ref_buf(fx.live_packed_bytes + kDecodeGatherSlackBytes, sv, mr);
  cudaMemsetAsync(static_cast<std::uint8_t*>(ref_buf.data()) + fx.live_packed_bytes,
                  0,
                  kDecodeGatherSlackBytes,
                  cs);
  CUdeviceptr d_ref = reinterpret_cast<CUdeviceptr>(ref_buf.data());
  try {
    const std::size_t got =
      cce::compact_bitpack_into(reinterpret_cast<void*>(d_ref),
                                fx.live_packed_bytes,
                                fx.live_packed_bytes,
                                reinterpret_cast<const void*>(fx.d_packed),
                                reinterpret_cast<const void*>(fx.d_live_words),
                                fx.num_chunks,
                                fx.stride_words,
                                cudaMemcpyDeviceToDevice,
                                /*stream=*/nullptr);
    if (got != fx.live_packed_bytes) {
      return report_fail(
        kTag,
        "reference compact returned wrong size",
        "got=" + std::to_string(got) + " expected=" + std::to_string(fx.live_packed_bytes));
    }
  } catch (const std::exception& e) {
    return report_fail(kTag, "reference compact threw", e.what());
  }
  if (cudaError_t e = cudaDeviceSynchronize(); e != cudaSuccess) {
    return report_fail(kTag, "sync after reference compact", cudaGetErrorString(e));
  }

  // Decode-verify the Compact-layout stream via JIT decode.
  codegen_test::GpuEncoded dec_scratch;
  cjj::LabeledBuffers labeled;
  labeled[cjj::buffer_key(0, "chunk_min")]  = {reinterpret_cast<const void*>(fx.d_chunk_min),
                                               static_cast<std::size_t>(fx.num_chunks),
                                               sizeof(Element)};
  labeled[cjj::buffer_key(0, "chunk_bits")] = {reinterpret_cast<const void*>(fx.d_chunk_bits),
                                               static_cast<std::size_t>(fx.num_chunks),
                                               sizeof(std::uint8_t)};
  labeled[cjj::buffer_key(0, "packed")]     = {reinterpret_cast<const void*>(d_ref),
                                               fx.live_packed_bytes / sizeof(std::uint32_t),
                                               sizeof(std::uint32_t)};
  labeled[cjj::buffer_key(0, "bp_offsets")] = {reinterpret_cast<const void*>(fx.d_bp_offsets),
                                               static_cast<std::size_t>(fx.num_chunks) + 1,
                                               sizeof(std::int32_t)};

  auto dec_tree          = cjj::FusedTree::make(cc::OpKind::Bitpack);
  dec_tree->fixed_stride = false;
  try {
    auto recovered = codegen_test::jit_decode_tree<Element>(
      *dec_tree, "int32_t", n, labeled, dec_scratch, opts.arch_cc);
    for (std::int64_t i = 0; i < n; ++i) {
      if (recovered[static_cast<std::size_t>(i)] != data[static_cast<std::size_t>(i)]) {
        return report_fail(
          kTag, "compacted stream failed to roundtrip", "first mismatch at i=" + std::to_string(i));
      }
    }
  } catch (const std::exception& e) {
    return report_fail(kTag, "JIT decode verify", e.what());
  }

  // Stash the verified dense bytes as the golden reference.
  fx.golden_bytes.resize(fx.live_packed_bytes);
  cudaMemcpy(fx.golden_bytes.data(),
             reinterpret_cast<const void*>(d_ref),
             fx.live_packed_bytes,
             cudaMemcpyDeviceToHost);
  return 0;
}

void teardown_fixture(CompactFixture& fx)
{
  fx.owned.clear();  // RMM pool buffers RAII-freed
  fx.d_packed     = 0;
  fx.d_live_words = 0;
  fx.d_chunk_min = fx.d_chunk_count = fx.d_chunk_bits = fx.d_bp_offsets = 0;
}

// Per-destination test routine.  Allocates dst via the supplied
// allocator, runs compact_bitpack_into, syncs the stream, copies
// dst back to a host buffer (no-op for already-host dst), compares
// bytes against the golden dense words.
//
// On success returns 0.  On failure returns 1 and prints which
// destination type failed.
template <typename DstAlloc, typename DstFree, typename DstReadback>
int run_destination(const char* tag,
                    const CompactFixture& fx,
                    DstAlloc alloc_fn,
                    DstFree free_fn,
                    DstReadback readback_fn,
                    cudaStream_t stream)
{
  void* dst = nullptr;
  int rc    = alloc_fn(&dst, fx.live_packed_bytes);
  if (rc != 0)
    return report_fail(tag, "destination alloc failed", "cudaError=" + std::to_string(rc));

  try {
    const std::size_t n = cce::compact_bitpack_into(dst,
                                                    fx.live_packed_bytes,
                                                    fx.live_packed_bytes,
                                                    reinterpret_cast<const void*>(fx.d_packed),
                                                    reinterpret_cast<const void*>(fx.d_live_words),
                                                    fx.num_chunks,
                                                    fx.stride_words,
                                                    cudaMemcpyDefault,
                                                    stream);
    if (n != fx.live_packed_bytes) {
      free_fn(dst);
      return report_fail(
        tag,
        "compact_bitpack_into returned wrong size",
        "returned=" + std::to_string(n) + " expected=" + std::to_string(fx.live_packed_bytes));
    }
  } catch (const std::exception& e) {
    free_fn(dst);
    return report_fail(tag, "compact_bitpack_into threw", e.what());
  }

  cudaError_t e = cudaStreamSynchronize(stream);
  if (e != cudaSuccess) {
    free_fn(dst);
    return report_fail(tag, "cudaStreamSynchronize", cudaGetErrorString(e));
  }

  // Read destination back to a host buffer for comparison.
  std::vector<std::uint8_t> host_dst(fx.live_packed_bytes, 0);
  int read_rc = readback_fn(host_dst.data(), dst, fx.live_packed_bytes);
  free_fn(dst);
  if (read_rc != 0) {
    return report_fail(tag, "destination readback failed", "cudaError=" + std::to_string(read_rc));
  }

  // Compare against the decode-verified golden dense bytes.
  const auto* gold = fx.golden_bytes.data();
  for (std::size_t i = 0; i < fx.live_packed_bytes; ++i) {
    if (host_dst[i] != gold[i]) {
      char buf[256];
      std::snprintf(buf,
                    sizeof(buf),
                    "first mismatch at byte i=%zu (word=%zu byte_in_word=%zu): "
                    "dst=0x%02x gold=0x%02x",
                    i,
                    i / 4,
                    i % 4,
                    host_dst[i],
                    gold[i]);
      return report_fail(tag, "byte mismatch vs golden", buf);
    }
  }

  std::printf("  %-20s PASSED (%zu bytes compacted)\n", tag, fx.live_packed_bytes);
  return 0;
}

// Capacity-mismatch check: passing a too-small dst_capacity should
// throw std::invalid_argument without launching anything.  Verifies
// the orchestrator's bounds check fires correctly.
int run_capacity_check(const CompactFixture& fx, cudaStream_t stream)
{
  constexpr const char* kTag = "capacity-check";

  void* d_dst   = nullptr;
  cudaError_t e = cudaMallocAsync(&d_dst, fx.live_packed_bytes, stream);
  if (e != cudaSuccess) return report_fail(kTag, "cudaMallocAsync(d_dst)", cudaGetErrorString(e));
  cudaStreamSynchronize(stream);

  bool threw = false;
  try {
    cce::compact_bitpack_into(d_dst,
                              /*dst_capacity=*/fx.live_packed_bytes - 1,  // one byte short
                              fx.live_packed_bytes,
                              reinterpret_cast<const void*>(fx.d_packed),
                              reinterpret_cast<const void*>(fx.d_live_words),
                              fx.num_chunks,
                              fx.stride_words,
                              cudaMemcpyDeviceToDevice,
                              stream);
  } catch (const std::invalid_argument&) {
    threw = true;
  } catch (const std::exception& e) {
    cudaFreeAsync(d_dst, stream);
    return report_fail(kTag, "wrong exception type", std::string("got: ") + e.what());
  }

  cudaFreeAsync(d_dst, stream);
  cudaStreamSynchronize(stream);
  if (!threw) return report_fail(kTag, "no exception on undersized dst", "");
  std::printf("  %-20s PASSED (invalid_argument as expected)\n", kTag);
  return 0;
}

}  // namespace

int main()
{
  constexpr std::int64_t n = 4321;
  auto data                = synth_data(n);

  cjj::ensure_cuda_context();  // must precede detect_arch_cc (needs active context)

  cje::CompileOptions opts;
  opts.arch_cc = detect_arch_cc();

  std::printf("test_encode_bitpack_compact: n=%lld num_chunks=%d\n",
              static_cast<long long>(n),
              cc::num_chunks_for(n));

  CompactFixture fx{};
  if (int rc = build_fixture(fx, data, n, opts); rc != 0) {
    teardown_fixture(fx);
    return rc;
  }
  std::printf(
    "  fixture: live_packed_bytes=%zu (across %d chunks, "
    "stride_words=%d; compacted stream decode-verified)\n",
    fx.live_packed_bytes,
    fx.num_chunks,
    fx.stride_words);

  // One stream shared across every destination variant — exercises
  // the orchestrator's stream-ordered allocations end-to-end.
  cudaStream_t stream = nullptr;
  if (cudaError_t e = cudaStreamCreate(&stream); e != cudaSuccess) {
    teardown_fixture(fx);
    std::fprintf(stderr, "FATAL: cudaStreamCreate: %s\n", cudaGetErrorString(e));
    return 1;
  }

  // (a) Device (cudaMallocAsync) — D2D path.
  {
    auto alloc = [&](void** out, std::size_t bytes) -> int {
      return static_cast<int>(cudaMallocAsync(out, bytes, stream));
    };
    auto free_ = [&](void* p) {
      if (p) cudaFreeAsync(p, stream);
    };
    auto readback = [&](void* host, void* device, std::size_t bytes) -> int {
      return static_cast<int>(cudaMemcpyAsync(host, device, bytes, cudaMemcpyDeviceToHost, stream));
    };
    if (int rc = run_destination("device-async", fx, alloc, free_, readback, stream); rc != 0) {
      cudaStreamDestroy(stream);
      teardown_fixture(fx);
      return rc;
    }
  }

  // (b) Host pinned (cudaHostAlloc) — D2H path with registered host.
  {
    auto alloc = [](void** out, std::size_t bytes) -> int {
      return static_cast<int>(cudaHostAlloc(out, bytes, cudaHostAllocDefault));
    };
    auto free_ = [](void* p) {
      if (p) cudaFreeHost(p);
    };
    auto readback = [](void* host, void* src, std::size_t bytes) -> int {
      std::memcpy(host, src, bytes);
      return 0;
    };
    if (int rc = run_destination("host-pinned", fx, alloc, free_, readback, stream); rc != 0) {
      cudaStreamDestroy(stream);
      teardown_fixture(fx);
      return rc;
    }
  }

  // (c) Host pageable (plain new[]) — D2H path with unregistered host.
  {
    auto alloc = [](void** out, std::size_t bytes) -> int {
      *out = new (std::nothrow) std::uint8_t[bytes];
      return (*out == nullptr) ? 1 : 0;
    };
    auto free_    = [](void* p) { delete[] static_cast<std::uint8_t*>(p); };
    auto readback = [](void* host, void* src, std::size_t bytes) -> int {
      std::memcpy(host, src, bytes);
      return 0;
    };
    if (int rc = run_destination("host-pageable", fx, alloc, free_, readback, stream); rc != 0) {
      cudaStreamDestroy(stream);
      teardown_fixture(fx);
      return rc;
    }
  }

  // (d) Managed memory — D2D-classified path with unified migration.
  {
    auto alloc = [](void** out, std::size_t bytes) -> int {
      return static_cast<int>(cudaMallocManaged(out, bytes, cudaMemAttachGlobal));
    };
    auto free_ = [](void* p) {
      if (p) cudaFree(p);
    };
    auto readback = [&](void* host, void* managed, std::size_t bytes) -> int {
      // Managed memory: pull explicitly through D2H so we don't
      // rely on access-fault migration (host read after stream
      // sync is fine, but cudaMemcpyAsync is the closest
      // analogue to the device-side case).
      return static_cast<int>(
        cudaMemcpyAsync(host, managed, bytes, cudaMemcpyDeviceToHost, stream));
    };
    if (int rc = run_destination("managed", fx, alloc, free_, readback, stream); rc != 0) {
      cudaStreamDestroy(stream);
      teardown_fixture(fx);
      return rc;
    }
  }

  // (e) Capacity-check / negative path.
  if (int rc = run_capacity_check(fx, stream); rc != 0) {
    cudaStreamDestroy(stream);
    teardown_fixture(fx);
    return rc;
  }

  cudaStreamDestroy(stream);
  teardown_fixture(fx);
  std::printf("test_encode_bitpack_compact: OK (all destinations verified)\n");
  return 0;
}
