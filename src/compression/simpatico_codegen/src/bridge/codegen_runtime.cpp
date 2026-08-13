// SPDX-License-Identifier: Apache-2.0
//
// JIT backend for the fused codegen compress/decompress path. Decode entry
// points are called from plan/decompress.cpp; the encode entry point
// (encode_fused_subtree) from plan/compress.cpp.
//
// Decode pipeline: decompress.cpp builds the runtime jit::FusedTree and a
// jit::LabeledBuffers of the real device buffers from the stored reps, keyed by
// buffer_key(node_id, field). launch_decode_fused_tree then:
//   1. synthesize_decode_transients adds decode-only buffers the encoder does
//      not store: Bitpack's bp_offsets cumsum (CUB ExclusiveSum + 1-thread tail
//      patch, see offsets_cumsum.cu) and RLE scratch, into RMM-pool transients.
//   2. KernelCache::get_or_compile_plain resolves shape -> CUkernel (keyed
//      on a hash of the rendered CUDA source, so a shape hits the cache
//      across compress and decompress). First touch pays the nvrtc compile;
//      warm hits are cheap. CUfunction is derived per-device at launch time.
//   3. cuLaunchKernel binds the labeled buffers as flat per-field device
//      pointers in the kernel's parameter order, runs over the rendered
//      __global__ kernel, syncs the stream, and frees the transients.
//
// Restrictions: int32/int64 dtypes; fused leaf kinds bitpack/delta/rle/for/
// zigzag (+ the synthesized raw passthrough).
// Other kinds fall through (caller routes to the legacy operator path).
// Every stored bitpack rep is dense (compact_bitpack_packed runs before
// publish), so decode always uses the Compact gather over bp_offsets.
// Failures log to stderr and return nullptr / -1.

#include "codegen/decode/jit/renderer.hpp"
#include "codegen/decode/masked_launch.hpp"
#include "codegen/encode/jit/renderer.hpp"
#include "codegen/jit/fused_tree.hpp"
#include "codegen/jit/kernel_cache.hpp"
#include "codegen/jit/nvrtc_compiler.hpp"

#include <cuda.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <functional>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

// Heavyweight C++ types pulled in only by the encode bridge (rep
// construction, plan-DSL walk, rmm device buffers).  Keeping these
// includes here avoids re-parsing them for the decode-only TUs that
// already compile fine without them.
#include "codegen/bridge/fused_tree_build.hpp"
#include "codegen/codegen_bridge.hpp"
#include "codegen/plan/plan_dsl.hpp"
#include "codegen/plan/plan_interpreter.hpp"
#include "codegen/plan/plan_tree.hpp"
#include "codegen/plan/representation.hpp"

#include <cudf/column/column.hpp>
#include <cudf/types.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/mr/per_device_resource.hpp>

namespace cc  = codegen;
namespace cje = codegen::encode::jit;
namespace cdj = codegen::decode::jit;
namespace jit = codegen::jit;
namespace enc = codegen::encode;

// Device-side cumsum entry points implemented in ``offsets_cumsum.cu``
// (both the Bitpack handle build and the Rle encode finalise scans). Plain
// C linkage so this TU doesn't need thrust/cub/nvcc — the bridge dispatches.
extern "C" {
int simpatico_compute_bp_offsets_scan(const void* d_chunk_count_v,
                                      const void* d_chunk_bits_v,
                                      std::int32_t num_chunks,
                                      void* d_bp_offsets_v,
                                      void* d_temp_storage,
                                      std::size_t* d_temp_storage_bytes_inout,
                                      void* stream_v);
int simpatico_compute_bp_offsets_tail(const void* d_chunk_count_v,
                                      const void* d_chunk_bits_v,
                                      std::int32_t num_chunks,
                                      void* d_bp_offsets_v,
                                      void* stream_v);
int simpatico_compute_rle_offsets_inclusive_scan(void* d_rle_offsets_v,
                                                 std::int32_t length,
                                                 void* d_temp_storage,
                                                 std::size_t* d_temp_storage_bytes_inout,
                                                 void* stream_v);
int simpatico_compact_raw_values(const void* d_padded_v,
                                 void* d_compact_v,
                                 const void* d_offsets_v,
                                 std::int32_t num_chunks,
                                 std::int32_t chunk_size,
                                 std::int32_t elem_size,
                                 void* stream_v);
}

#define SIMPATICO_CUDA_CHECK(call, ret, ...)                     \
  do {                                                           \
    if (cudaError_t _cerr = (call); _cerr != cudaSuccess) {      \
      std::fprintf(stderr, "simpatico::codegen: " __VA_ARGS__);  \
      std::fprintf(stderr, ": %s\n", cudaGetErrorString(_cerr)); \
      return (ret);                                              \
    }                                                            \
  } while (0)

#define SIMPATICO_CU_CHECK(call, ret, ...)                      \
  do {                                                          \
    if (CUresult _dr = (call); _dr != CUDA_SUCCESS) {           \
      const char* _desc = nullptr;                              \
      cuGetErrorString(_dr, &_desc);                            \
      std::fprintf(stderr, "simpatico::codegen: " __VA_ARGS__); \
      std::fprintf(stderr, ": %s\n", _desc ? _desc : "?");      \
      return (ret);                                             \
    }                                                           \
  } while (0)

namespace {

// Raise MAX_DYNAMIC_SHARED_SIZE_BYTES if needed; return false and log on
// failure (skipping cuLaunchKernel avoids CUDA_ERROR_INVALID_HANDLE cascade).
bool maybe_raise_smem(CUfunction fn, int dynamic_bytes, const char* ctx)
{
  int static_smem = 0;
  cuFuncGetAttribute(&static_smem, CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, fn);
  if (static_smem + dynamic_bytes <= 48 * 1024) return true;
  if (CUresult r =
        cuFuncSetAttribute(fn, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, dynamic_bytes);
      r != CUDA_SUCCESS) {
    const char* desc = nullptr;
    cuGetErrorString(r, &desc);
    std::fprintf(stderr,
                 "simpatico::codegen: %s: "
                 "cuFuncSetAttribute(MAX_DYNAMIC_SHARED_SIZE_BYTES, %d) failed: %s\n",
                 ctx,
                 dynamic_bytes,
                 desc ? desc : "?");
    return false;
  }
  return true;
}

// Fill ``d_bp_off`` (int32[num_chunks+1]) with the exclusive-prefix scan of the
// per-chunk live-word counts (derived from chunk_bits × chunk_count) plus the
// [num_chunks] total sentinel, so bp_offsets[c+1]-bp_offsets[c] == live_words[c].
// ``alloc_scratch(bytes)`` supplies CUB temp storage that must stay live until
// this returns (the caller owns it). Shared by the encode-side compaction and
// the decode-side transient synthesis. Returns 0 on success, else the failing
// cuda/rc code.
int compute_bp_offsets(const void* cc_p,
                       const void* cb_p,
                       std::int32_t num_chunks,
                       void* d_bp_off,
                       const std::function<void*(std::size_t)>& alloc_scratch,
                       void* stream_v)
{
  std::size_t scratch_bytes = 0;
  if (int rc = simpatico_compute_bp_offsets_scan(
        cc_p, cb_p, num_chunks, d_bp_off, /*d_temp_storage=*/nullptr, &scratch_bytes, stream_v);
      rc != 0) {
    return rc;
  }
  void* d_scratch = scratch_bytes > 0 ? alloc_scratch(scratch_bytes) : nullptr;
  if (int rc = simpatico_compute_bp_offsets_scan(
        cc_p, cb_p, num_chunks, d_bp_off, d_scratch, &scratch_bytes, stream_v);
      rc != 0) {
    return rc;
  }
  return simpatico_compute_bp_offsets_tail(cc_p, cb_p, num_chunks, d_bp_off, stream_v);
}

// Render-side kernel compile: optionally dump the source (``dump_env`` names an
// env var holding a path), then compile-or-warm-cache the rendered source.
// Returns the cached kernel, or nullptr on any failure (logged with ``ctx`` as
// the message prefix). ``default_device`` is nvrtc's -default-device (decode
// needs it for rle_block.cuh's unannotated constexpr accessors).
const jit::CompiledKernel* compile_rendered(const std::string& source,
                                            const std::string& entry_symbol,
                                            bool default_device,
                                            const char* dump_env,
                                            const char* ctx)
{
  if (const char* dump = std::getenv(dump_env)) {
    if (FILE* fp = std::fopen(dump, "w")) {
      std::fwrite(source.data(), 1, source.size(), fp);
      std::fclose(fp);
    }
  }
  jit::CompileOptions opts;
  opts.arch_cc        = jit::arch_cc_for_current_device();
  opts.default_device = default_device;
  try {
    const jit::CompiledKernel* kernel =
      jit::KernelCache::instance().get_or_compile_plain(source, entry_symbol, opts);
    if (kernel == nullptr || kernel->kern == nullptr) {
      std::fprintf(stderr, "simpatico::codegen: %s: null kernel\n", ctx);
      return nullptr;
    }
    return kernel;
  } catch (const jit::CompileError& e) {
    std::fprintf(stderr,
                 "simpatico::codegen: %s: nvrtc rejected source: %s\n--- log ---\n%s\n",
                 ctx,
                 e.what(),
                 e.log.c_str());
    return nullptr;
  }
}

}  // namespace

namespace simpatico {
namespace {

// ---------------------------------------------------------------------------
// compact_bitpack_packed — densify a fused-encode ``packed`` OverAllocate
// buffer into a tight Compact UINT32 column.
//
// The fused encode kernel (renderer.cpp::emit_bitpack) emits ``packed`` in
// slot-strided OverAllocate layout: chunk c's live data sits at
// ``packed[c*stride_words .. +live_words[c])`` with the tail zeroed. That is
// correct and fast for the decoder (per-chunk stride reads) but wrong for
// anything that transit-emits dense bytes (file write, tail-codec input). The
// only caller is the bitpack branch of the encode bridge below, right after it
// builds the OverAllocate columns. It reuses the exact machinery
// ``synthesize_decode_transients`` runs for every bitpack column on decode:
//   1. Derive per-chunk offsets from chunk_bits x chunk_count via
//      simpatico_compute_bp_offsets_scan + _tail — an exclusive scan of the
//      live-word counts plus the [num_chunks] sentinel, so
//      bp_offsets[c+1] - bp_offsets[c] == live_words[c].
//   2. Gather the live words into a tight dense buffer with the shared
//      simpatico_compact_raw_values strided gather (elem_size = 4 = uint32
//      words), which reads each chunk's run length from consecutive offsets.
//
// Stream-ordered: the transient bp_offsets + CUB scratch are enqueued on the
// caller's stream and RAII-freed on it; the caller syncs before reading the
// dense bytes from another stream. ``live_packed_bytes`` is the precomputed
// tight byte count (always a multiple of 4).
// ---------------------------------------------------------------------------
std::unique_ptr<cudf::column> compact_bitpack_packed(cudf::column const& chunk_count,
                                                     cudf::column const& chunk_bits,
                                                     cudf::column const& packed_overalloc,
                                                     std::int32_t stride_words,
                                                     std::int64_t live_packed_bytes,
                                                     rmm::cuda_stream_view stream,
                                                     rmm::device_async_resource_ref mr)
{
  auto check_rc = [](int rc, const char* what) {
    if (rc != 0)
      throw std::runtime_error(std::string("compact_bitpack_packed: ") + what +
                               " failed (rc=" + std::to_string(rc) + ")");
  };

  const std::size_t live = static_cast<std::size_t>(live_packed_bytes);

  // simpatico_bitunpack_one loads three consecutive uint32 words (w0/w1/w2);
  // three guard words keep the last element's gather in-bounds.  They travel
  // as part of the stored payload so num_rows already accounts for them.
  constexpr std::size_t kGuardWords = 3;
  const std::size_t guard_bytes     = kGuardWords * sizeof(std::uint32_t);

  rmm::device_buffer dense(live + guard_bytes, stream, mr);
  if (live > 0) {
    const cudf::size_type num_chunks = chunk_count.size();
    const void* cc_p                 = chunk_count.view().head<void>();
    const void* cb_p                 = chunk_bits.view().head<void>();

    // Per-chunk destination offsets: exclusive scan of the live-word counts
    // derived from chunk_bits x chunk_count, plus the [num_chunks] sentinel —
    // the same derivation synthesize_decode_transients runs on decode, so
    // bp_offsets[c+1] - bp_offsets[c] == live_words[c]. The scan reads its input
    // through a fused transform iterator, so no live_words array is materialised
    // and there is no CUB raw-input over-read to pad against.
    rmm::device_buffer bp_offsets_buf(
      (static_cast<std::size_t>(num_chunks) + 1) * sizeof(std::int32_t), stream, mr);
    void* d_bp_off = bp_offsets_buf.data();

    // CUB scratch is kept alive here (freed async on ``stream`` at scope exit),
    // outliving the scan the shared helper enqueues.
    rmm::device_buffer scratch_buf;
    check_rc(compute_bp_offsets(
               cc_p,
               cb_p,
               static_cast<std::int32_t>(num_chunks),
               d_bp_off,
               [&](std::size_t bytes) {
                 scratch_buf = rmm::device_buffer(bytes, stream, mr);
                 return scratch_buf.data();
               },
               stream.value()),
             "bp_offsets");

    // Strided gather: copy live_words[c] = bp_offsets[c+1] - bp_offsets[c]
    // uint32 words from packed[c*stride_words ..] to dense[bp_offsets[c] ..].
    // The same shared gather the RLE raw-values path uses, with elem_size = 4
    // (uint32 words) and chunk_size = stride_words.
    check_rc(simpatico_compact_raw_values(packed_overalloc.view().head<void>(),
                                          dense.data(),
                                          d_bp_off,
                                          static_cast<std::int32_t>(num_chunks),
                                          stride_words,
                                          static_cast<std::int32_t>(sizeof(std::uint32_t)),
                                          stream.value()),
             "compact gather");
  }

  // Zero the guard words so the decode over-read returns deterministic
  // (masked-out) zeros rather than uninitialised memory.
  cudaMemsetAsync(static_cast<std::uint8_t*>(dense.data()) + live, 0, guard_bytes, stream.value());

  // packed is uint32 words; UINT32 (size = words) keeps a >2GB dense buffer
  // under cudf's 2^31-element cap.  The column size includes the guard words
  // so they travel through serialisation/deserialisation without any special
  // extra allocation at the read site.  `live` is always a multiple of 4.
  return std::make_unique<cudf::column>(cudf::data_type(cudf::type_id::UINT32),
                                        static_cast<cudf::size_type>(live / 4 + kGuardWords),
                                        std::move(dense),
                                        rmm::device_buffer(0, stream, mr),
                                        0);
}

}  // namespace
}  // namespace simpatico

namespace {

constexpr int kChunkSize = cc::kChunkSize;  // 1024

// DFS-preorder traversal with lex-sorted children — matches
// ``jit::assign_ids`` so the node visit order assigns ids in the
// same way ``build_root`` consumes them.
void dfs_nodes(const jit::FusedTree& node, std::vector<const jit::FusedTree*>& out)
{
  out.push_back(&node);
  for (auto const& [_unused, child] : node.children) {
    dfs_nodes(*child, out);
  }
}

// ---------------------------------------------------------------------------
// synthesize_decode_transients: computes decode-only buffers the encoder
// never stores, reading the already-bound real buffers out of `out` (keyed by
// DFS-preorder node_id). Used by the structural PlanTree decode binder
// (bind_fused_subtree in plan/decompress.cpp):
//   * Bitpack bp_offsets: exclusive cumsum of n_words[c] via CUB (device-side
//       scan + 1-thread tail patch), reading chunk_count/chunk_bits from `out`.
//   * Rle scatter-scan-gather transients (rle_scratch / rle_run_values):
//       registered as null placeholders (length only) — the rendered RLE decode
//       kernel sizes its grid from them but does not read their contents.
// ---------------------------------------------------------------------------
bool synthesize_decode_transients(const jit::FusedTree& tree,
                                  std::size_t element_size,
                                  const std::function<CUdeviceptr(std::size_t)>& alloc,
                                  void* stream_v,
                                  jit::LabeledBuffers& out,
                                  std::string* err)
{
  std::vector<const jit::FusedTree*> nodes;
  dfs_nodes(tree, nodes);

  for (std::int32_t node_id = 0; node_id < static_cast<std::int32_t>(nodes.size()); ++node_id) {
    const auto& node = *nodes[node_id];
    if (node.op == cc::OpKind::Bitpack) {
      auto cc_it = out.find(jit::buffer_key(node_id, "chunk_count"));
      auto cb_it = out.find(jit::buffer_key(node_id, "chunk_bits"));
      if (cc_it == out.end() || cb_it == out.end()) {
        if (err)
          *err = "synthesize_decode_transients: Bitpack node " + std::to_string(node_id) +
                 " missing chunk_count/chunk_bits";
        return false;
      }
      const std::int64_t num_chunks = static_cast<std::int64_t>(cc_it->second.length);
      // Every bitpack rep is dense, so decode uses the Compact gather:
      // synthesize the per-chunk ``bp_offsets`` on-device via a CUB scan of
      // chunk_bits × chunk_count.
      {
        const void* cc_p = cc_it->second.ptr;
        const void* cb_p = cb_it->second.ptr;
        const std::size_t bp_off_bytes =
          (static_cast<std::size_t>(num_chunks) + 1) * sizeof(std::int32_t);
        CUdeviceptr d_bp_off = alloc(bp_off_bytes);

        // Transients (bp_offsets + CUB scratch) both come from the caller's
        // ``alloc`` pool, so the shared helper's scratch outlives the scan.
        if (int rc = compute_bp_offsets(
              cc_p,
              cb_p,
              static_cast<std::int32_t>(num_chunks),
              reinterpret_cast<void*>(d_bp_off),
              [&](std::size_t bytes) { return reinterpret_cast<void*>(alloc(bytes)); },
              stream_v);
            rc != 0) {
          if (err) *err = "compute_bp_offsets failed (cudaError=" + std::to_string(rc) + ")";
          return false;
        }
        out[jit::buffer_key(node_id, "bp_offsets")] = {reinterpret_cast<const void*>(d_bp_off),
                                                       static_cast<std::size_t>(num_chunks) + 1,
                                                       sizeof(std::int32_t)};
      }
    } else if (node.op == cc::OpKind::Rle) {
      auto ro_it = out.find(jit::buffer_key(node_id, "rle_runs_offsets"));
      if (ro_it == out.end()) {
        if (err)
          *err = "synthesize_decode_transients: Rle node " + std::to_string(node_id) +
                 " missing rle_runs_offsets";
        return false;
      }
      const std::int64_t ro_l           = static_cast<std::int64_t>(ro_it->second.length);
      const std::int64_t rle_num_chunks = ro_l > 0 ? ro_l - 1 : 0;
      const std::size_t scratch_len     = static_cast<std::size_t>(rle_num_chunks) * kChunkSize;
      out[jit::buffer_key(node_id, "rle_scratch")] = {
        /*ptr=*/nullptr, scratch_len, sizeof(std::int32_t)};
      out[jit::buffer_key(node_id, "rle_run_values")] = {
        /*ptr=*/nullptr, scratch_len, element_size};
    }
  }
  return true;
}

const char* dtype_to_cxx(const char* dtype)
{
  if (!dtype) return nullptr;
  std::string s(dtype);
  if (s == "int32" || s == "int32_t") return "int32_t";
  if (s == "int64" || s == "int64_t") return "int64_t";
  // Float types are processed as same-width integers; the output column retains
  // the original float type_id so cudf reinterprets the bits correctly.
  if (s == "float32") return "int32_t";
  if (s == "float64") return "int64_t";
  // Byte types (string sub-channels: chars, null_mask, keys_chars) map to the
  // SIGNED int8_t: zigzag's shift = elem_size*8-1 relies on sign-extension.
  // The output column retains the original UINT8 type_id.
  if (s == "uint8" || s == "uint8_t" || s == "int8" || s == "int8_t") return "int8_t";
  // 16-bit signed/unsigned both process as int16_t; zigzag's shift = 15 relies
  // on sign-extension, and the output column keeps the original UINT16 type_id.
  if (s == "int16" || s == "int16_t" || s == "uint16" || s == "uint16_t") return "int16_t";
  // Unsigned 32/64-bit process as their same-width signed integer (bit-level
  // codecs roundtrip either way); the output column keeps the original UINT type.
  if (s == "uint32" || s == "uint32_t") return "int32_t";
  if (s == "uint64" || s == "uint64_t") return "int64_t";
  return nullptr;
}

// Trailing kernel arguments for the row-selecting decode variants
// (mask_out / mask_consume).  Defaults = plain decode, no extras.
// Predicate constants / mask pointers are kernel PARAMETERS so one NVRTC
// compile per (shape, dtype, variant) covers every literal.
struct VariantLaunchArgs {
  cdj::DecodeShape shape    = cdj::kShapePlain;
  std::int64_t pred_lo      = 0;  // mask_out: inclusive range, decoded int domain
  std::int64_t pred_hi      = 0;
  CUdeviceptr sel_mask      = 0;  // mask_consume/mask_dict_gather: selection-mask words (input)
  CUdeviceptr chunk_offsets = 0;  // mask_consume/mask_dict_gather: per-chunk survivor bases
  CUdeviceptr keys_chars    = 0;  // mask_dict_gather: constant-width key pool chars
  std::int32_t key_width    = 0;  // mask_dict_gather: bytes per key
  CUdeviceptr row_indices   = 0;  // index_consume: ascending global int32 row ids
  CUdeviceptr len_out       = 0;  // str_split_meta: per-survivor byte lengths (output)
};

// Per-variant launch preconditions, checked in ONE place with ONE diagnostic.
//
// Each entry says which VariantLaunchArgs slots must be populated and whether
// the variant's compare is integer-domain only.  Previously every public
// launcher hand-wrote its own null checks and its own fprintf listing its own
// fields; the differences between them were exactly this table.
struct VariantContract {
  bool sel_mask      = false;
  bool chunk_offsets = false;
  bool row_indices   = false;
  bool keys_chars    = false;  // implies key_width >= 1
  bool len_out       = false;
  bool out           = false;  // the (out, n) slot must be non-null
  // Floats decode as bit-reinterpreted same-width ints, so an integer-domain
  // range compare on them is meaningless — refuse rather than corrupt.
  bool rejects_float = false;
};

VariantContract variant_contract(cdj::DecodeShape shape)
{
  VariantContract c;
  // Enumerator: what it needs to walk the chunk's rows.
  switch (shape.enumerator) {
    case cdj::Enumerator::all_rows: break;
    case cdj::Enumerator::mask_bits: c.sel_mask = c.chunk_offsets = true; break;
    case cdj::Enumerator::index_list: c.row_indices = c.chunk_offsets = true; break;
  }
  // Consumer: what it needs to act on a row, and whether `out` is a column.
  switch (shape.consumer) {
    case cdj::Consumer::write_column: c.out = true; break;
    case cdj::Consumer::ballot_range:
      // The `out` slot carries the mask words, so it is checked as sel_mask.
      // Integer-domain compare only: floats decode bit-reinterpreted.
      c.sel_mask      = true;
      c.rejects_float = true;
      break;
    case cdj::Consumer::dict_gather: c.keys_chars = c.out = true; break;
    case cdj::Consumer::offsets_meta: c.len_out = c.out = true; break;
  }
  return c;
}

// Check `va` (plus the mask geometry and dtype) against the variant's contract.
// `rows` is the ROW-space count the mask describes — for str_split_meta that is
// the string row count, while the kernel's `n` is the offsets count.
bool check_variant_contract(const VariantLaunchArgs& va,
                            const ::sirius::codegen::selection_mask* mask,
                            std::int64_t rows,
                            const void* out,
                            const char* dtype,
                            const char* ctx)
{
  const VariantContract c = variant_contract(va.shape);
  const char* missing     = nullptr;
  if (c.sel_mask && va.sel_mask == 0)
    missing = "selection mask words";
  else if (c.chunk_offsets && va.chunk_offsets == 0)
    missing = "chunk_offsets (did the CNT wave run?)";
  else if (c.row_indices && va.row_indices == 0)
    missing = "row_indices (did the mask->indices wave run?)";
  else if (c.keys_chars && (va.keys_chars == 0 || va.key_width < 1))
    missing = "dictionary keys_chars/key_width";
  else if (c.len_out && va.len_out == 0)
    missing = "len_out";
  else if (c.out && out == nullptr)
    missing = "output buffer";
  else if (mask != nullptr && mask->num_rows != rows)
    missing = "mask.num_rows != row count";

  if (missing != nullptr) {
    std::fprintf(stderr,
                 "simpatico::codegen: %s: incomplete inputs — %s (mask.num_rows=%lld rows=%lld "
                 "words=%p chunk_offsets=%p row_indices=%p keys=%p key_width=%d len_out=%p "
                 "out=%p)\n",
                 ctx,
                 missing,
                 static_cast<long long>(mask != nullptr ? mask->num_rows : -1),
                 static_cast<long long>(rows),
                 reinterpret_cast<void*>(va.sel_mask),
                 reinterpret_cast<void*>(va.chunk_offsets),
                 reinterpret_cast<void*>(va.row_indices),
                 reinterpret_cast<void*>(va.keys_chars),
                 va.key_width,
                 reinterpret_cast<void*>(va.len_out),
                 out);
    return false;
  }
  if (c.rejects_float && dtype != nullptr &&
      (std::strcmp(dtype, "float32") == 0 || std::strcmp(dtype, "float64") == 0)) {
    std::fprintf(stderr,
                 "simpatico::codegen: %s: float dtype '%s' not supported for integer-domain "
                 "range predicates\n",
                 ctx,
                 dtype);
    return false;
  }
  return true;
}

// The invariant half of every masked launch. Bundled because each public
// launcher otherwise repeats the same six parameters, and because the contract
// check and the launch belong together: a launcher that skipped the check would
// bind a missing pointer and fault.
struct masked_launch {
  codegen::jit::FusedTree const& tree;
  codegen::jit::LabeledBuffers& labeled;
  char const* dtype;
  std::int64_t num_rows;  // ROW-space count the mask describes
  ::sirius::codegen::selection_mask const& mask;
  rmm::cuda_stream_view stream;
};

// Storage for the trailing kernel arguments, bound by TAG rather than by
// variant.  The ORDER comes from the renderer (DecodeKernelSpec::trailing),
// which emitted the matching declarations from the same table — so this side
// only has to say where each tag's value lives, and the two lists cannot
// disagree.  Must outlive the cuLaunchKernel call (args holds pointers into it).
struct TrailingArgStorage {
  std::int64_t pred_lo, pred_hi;
  CUdeviceptr sel_mask, chunk_offsets, keys_chars, row_indices, len_out;
  std::int32_t key_width;

  explicit TrailingArgStorage(const VariantLaunchArgs& va)
    : pred_lo(va.pred_lo),
      pred_hi(va.pred_hi),
      sel_mask(va.sel_mask),
      chunk_offsets(va.chunk_offsets),
      keys_chars(va.keys_chars),
      row_indices(va.row_indices),
      len_out(va.len_out),
      key_width(va.key_width)
  {
  }

  // Append `trailing`'s arguments, in the renderer's order, to `args`.
  // Returns false (and reports) if a tag has no storage — which can only mean
  // TrailingParam gained a member without a slot here.
  bool append(const std::vector<cdj::TrailingParam>& trailing,
              std::vector<void*>& args,
              const char* ctx)
  {
    void* slot[static_cast<std::size_t>(cdj::TrailingParam::kCount)] = {};
    using TP                                                         = cdj::TrailingParam;
    slot[static_cast<std::size_t>(TP::pred_lo)]                      = &pred_lo;
    slot[static_cast<std::size_t>(TP::pred_hi)]                      = &pred_hi;
    slot[static_cast<std::size_t>(TP::sel_mask)]                     = &sel_mask;
    slot[static_cast<std::size_t>(TP::chunk_offsets)]                = &chunk_offsets;
    slot[static_cast<std::size_t>(TP::keys_chars)]                   = &keys_chars;
    slot[static_cast<std::size_t>(TP::key_width)]                    = &key_width;
    slot[static_cast<std::size_t>(TP::row_indices)]                  = &row_indices;
    slot[static_cast<std::size_t>(TP::len_out)]                      = &len_out;

    for (const auto tag : trailing) {
      const auto idx = static_cast<std::size_t>(tag);
      if (idx >= static_cast<std::size_t>(TP::kCount) || slot[idx] == nullptr) {
        std::fprintf(stderr,
                     "simpatico::codegen: %s: trailing parameter tag %u has no argument slot — "
                     "TrailingParam and TrailingArgStorage are out of sync\n",
                     ctx,
                     static_cast<unsigned>(idx));
        return false;
      }
      args.push_back(slot[idx]);
    }
    return true;
  }
};

// Launch the plain-CUDA rendered decode kernel (symmetric to the
// encode renderer).  Returns 1 on success, -1 on failure.  Reuses the
// LabeledBuffers the caller already built; binds device pointers by
// (node_id, field) in the spec's parameter order, then out + n (+ the
// variant's trailing params).  For mask_out, ``out_ptr`` carries the
// selection-mask words pointer (no column output exists).
// Bind a rendered spec's buffers, append (out, n) + the trailing args, and
// launch, under the per-chunk metadata bounds guard below.
//
// `labeled` must already contain every channel the spec names.  `ctx` prefixes
// diagnostics.
int launch_rendered_spec(const cdj::DecodeKernelSpec& spec,
                         const jit::CompiledKernel& kernel,
                         const jit::LabeledBuffers& labeled,
                         std::int64_t num_rows,
                         std::int32_t num_chunks,
                         std::uintptr_t out_ptr,
                         std::uintptr_t stream_ptr,
                         const std::function<void(const char*)>& lap,
                         const VariantLaunchArgs& va,
                         const char* ctx)
{
  // Bind device pointers in the kernel's parameter order.
  std::vector<CUdeviceptr> dptrs;
  dptrs.reserve(spec.buffers.size());
  for (const auto& b : spec.buffers) {
    const std::string key = jit::buffer_key(b.node_id, b.field);
    auto it               = labeled.find(key);
    if (it == labeled.end()) {
      std::fprintf(
        stderr, "simpatico::codegen: %s: missing labeled buffer '%s'\n", ctx, key.c_str());
      return -1;
    }
    dptrs.push_back(reinterpret_cast<CUdeviceptr>(it->second.ptr));
    // Guard: a per-chunk metadata buffer is indexed by the (root) chunk_id in
    // [0, num_chunks), so it must hold at least num_chunks entries (+1 for the
    // exclusive-offset arrays). If the launch grid (derived from num_rows) exceeds
    // what the bound metadata describes, the kernel would read out of bounds and
    // fault the CUDA context. Fail cleanly instead — this catches any future drift
    // between a rep's num_rows and its serialized per-chunk channels.
    {
      const std::size_t len = it->second.length;
      const bool is_off     = (b.field == "rle_runs_offsets" || b.field == "bp_offsets");
      const bool is_perchk =
        (b.field == "chunk_min" || b.field == "chunk_bits" || b.field == "chunk_count" ||
         b.field == "references" || b.field == "offsets" || is_off);
      const std::size_t need = static_cast<std::size_t>(num_chunks) + (is_off ? 1u : 0u);
      if (is_perchk && len < need) {
        std::fprintf(stderr,
                     "simpatico::codegen: %s: metadata buffer '%s' (node %d) has %zu "
                     "entries but the grid needs >=%zu (num_rows=%lld num_chunks=%d, kernel=%s) — "
                     "num_rows/metadata mismatch; refusing to launch\n",
                     ctx,
                     b.field.c_str(),
                     b.node_id,
                     len,
                     need,
                     static_cast<long long>(num_rows),
                     num_chunks,
                     spec.entry_symbol.c_str());
        return -1;
      }
    }
  }

  CUdeviceptr d_out    = static_cast<CUdeviceptr>(out_ptr);
  std::int64_t total_n = num_rows;
  std::vector<void*> args;
  args.reserve(dptrs.size() + 8);
  for (auto& p : dptrs)
    args.push_back(&p);
  args.push_back(&d_out);
  args.push_back(&total_n);

  // Trailing kernel parameters, in the order the RENDERER emitted them
  // (spec.trailing) — storage outlives the launch below.
  TrailingArgStorage trailing_storage{va};
  if (!trailing_storage.append(spec.trailing, args, ctx)) { return -1; }

  if (!maybe_raise_smem(kernel.func_for_current_device(), static_cast<int>(spec.shared_bytes), ctx))
    return -1;

  CUstream stream   = reinterpret_cast<CUstream>(stream_ptr);
  CUfunction fn_dec = kernel.func_for_current_device();
  SIMPATICO_CU_CHECK(cuLaunchKernel(fn_dec,
                                    static_cast<unsigned>(num_chunks),
                                    1,
                                    1,
                                    static_cast<unsigned>(spec.block_x),
                                    1,
                                    1,
                                    static_cast<unsigned>(spec.shared_bytes),
                                    stream,
                                    args.data(),
                                    nullptr),
                     -1,
                     "cuLaunchKernel failed");
  lap("launch");

  SIMPATICO_CUDA_CHECK(
    cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(stream_ptr)), -1, "stream sync failed");
  lap("sync");
  return 1;
}

int run_rendered_decode(const jit::FusedTree& tree,
                        const char* cxx_dtype,
                        const jit::LabeledBuffers& labeled,
                        std::int64_t num_rows,
                        std::uintptr_t out_ptr,
                        std::uintptr_t stream_ptr,
                        const std::function<void(const char*)>& lap,
                        const VariantLaunchArgs& va = VariantLaunchArgs{})
{
  const std::int32_t num_chunks =
    static_cast<std::int32_t>(std::max<std::int64_t>(1, (num_rows + kChunkSize - 1) / kChunkSize));

  cdj::DecodeKernelSpec spec;
  try {
    spec = cdj::render(tree, cxx_dtype, num_chunks, va.shape);
  } catch (const cdj::RenderError& e) {
    std::fprintf(
      stderr, "simpatico::codegen: rendered decode: render rejected shape: %s\n", e.what());
    return -1;
  }
  lap("render");
  // The rendered decode source includes rle_block.cuh (→ tree.hpp), whose
  // unannotated constexpr accessors require nvrtc's -default-device.
  const jit::CompiledKernel* kernel = compile_rendered(spec.source,
                                                       spec.entry_symbol,
                                                       /*default_device=*/true,
                                                       "CODEGEN_JIT_DUMP_DECODE_SOURCE",
                                                       "rendered decode");
  if (kernel == nullptr) { return -1; }
  lap("compile");

  return launch_rendered_spec(
    spec, *kernel, labeled, num_rows, num_chunks, out_ptr, stream_ptr, lap, va, "rendered decode");
}

}  // namespace

namespace simpatico {

// Launch one prepared codegen-fused decode tree. The high-level
// ``decode_fused_subtree`` caller
// already holds the subtree structurally: it passes a fully-built ``FusedTree``
// (decode is Compact-only) and a
// ``LabeledBuffers`` of the real device buffers, keyed by DFS-preorder
// ``buffer_key(node_id, field)`` — exactly the ids ``run_rendered_decode``
// resolves against. This function only adds the decode-only transients the
// encoder never stores (Compact ``bp_offsets`` CUB scan, RLE scratch), compiles
// (or warm-cache hits) the rendered decode kernel, launches, and synchronises.
//
// All device transients come from the RMM async pool and are freed when this
// returns. ``labeled`` is mutated in place (transients are added).
//
// Returns 1 on success, -1 on any failure (logged to stderr).
namespace {

bool launch_decode_fused_tree_impl(codegen::jit::FusedTree const& tree,
                                   codegen::jit::LabeledBuffers& labeled,
                                   char const* dtype,
                                   std::int64_t num_rows,
                                   void* out,
                                   rmm::cuda_stream_view stream,
                                   VariantLaunchArgs const& va)
{
  try {
    const bool _timing = std::getenv("SIMPATICO_DECODE_TIMING") != nullptr;
    auto _t            = std::chrono::steady_clock::now();
    auto _lap          = [&](const char* what) {
      if (!_timing) return;
      auto now = std::chrono::steady_clock::now();
      std::fprintf(stderr,
                   "[decode] %-12s %7.1f us\n",
                   what,
                   std::chrono::duration<double, std::micro>(now - _t).count());
      _t = now;
    };

    const char* cxx_dtype = dtype_to_cxx(dtype);
    if (cxx_dtype == nullptr) {
      std::fprintf(stderr,
                   "simpatico::codegen: launch_decode_fused_tree: unsupported dtype '%s'\n",
                   dtype ? dtype : "(null)");
      return false;
    }
    const std::size_t elem_size = (std::strcmp(cxx_dtype, "int64_t") == 0)   ? 8u
                                  : (std::strcmp(cxx_dtype, "int16_t") == 0) ? 2u
                                  : (std::strcmp(cxx_dtype, "int8_t") == 0)  ? 1u
                                                                             : 4u;

    // Device transients (bp_offsets/scratch) from the RMM async pool, freed
    // async on return.
    rmm::device_async_resource_ref mr = rmm::mr::get_current_device_resource_ref();
    std::vector<rmm::device_buffer> transients;
    transients.reserve(8);
    auto alloc = [&](std::size_t bytes) -> CUdeviceptr {
      transients.emplace_back(bytes, stream, mr);
      return reinterpret_cast<CUdeviceptr>(transients.back().data());
    };

    std::string err;
    if (!synthesize_decode_transients(tree, elem_size, alloc, stream.value(), labeled, &err)) {
      std::fprintf(stderr,
                   "simpatico::codegen: launch_decode_fused_tree: transient synth failed: %s\n",
                   err.c_str());
      return false;
    }

    _lap("labeled");

    // run_rendered_decode is an internal helper with a uintptr_t ABI; the public
    // signature is C++-typed, so convert once here.
    return run_rendered_decode(
             tree,
             cxx_dtype,
             labeled,
             num_rows,
             reinterpret_cast<std::uintptr_t>(out),
             reinterpret_cast<std::uintptr_t>(stream.value()),
             [&](const char* what) { _lap(what); },
             va) == 1;
  } catch (const jit::CompileError& e) {
    std::fprintf(
      stderr,
      "simpatico::codegen: launch_decode_fused_tree: CompileError: %s\n--- log ---\n%s\n",
      e.what(),
      e.log.c_str());
    return false;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "simpatico::codegen: launch_decode_fused_tree: %s\n", e.what());
    return false;
  } catch (...) {
    std::fprintf(stderr, "simpatico::codegen: launch_decode_fused_tree: unknown exception\n");
    return false;
  }
}

}  // namespace

bool launch_decode_fused_tree(codegen::jit::FusedTree const& tree,
                              codegen::jit::LabeledBuffers& labeled,
                              char const* dtype,
                              std::int64_t num_rows,
                              void* out,
                              rmm::cuda_stream_view stream)
{
  return launch_decode_fused_tree_impl(
    tree, labeled, dtype, num_rows, out, stream, VariantLaunchArgs{});
}

namespace {

// Check the shape's contract, then launch it. `kernel_rows` is the domain the
// KERNEL runs over, which differs from the mask's row space only for the
// str_split metadata shape (offsets = rows + 1).
bool check_and_launch(masked_launch const& m,
                      VariantLaunchArgs const& va,
                      void* out,
                      char const* what,
                      std::int64_t kernel_rows)
{
  if (!check_variant_contract(va, &m.mask, m.num_rows, out, m.dtype, what)) { return false; }
  return launch_decode_fused_tree_impl(m.tree, m.labeled, m.dtype, kernel_rows, out, m.stream, va);
}

CUdeviceptr dev(void const* p) { return reinterpret_cast<CUdeviceptr>(const_cast<void*>(p)); }

}  // namespace

// Range ballot: fused decode + range predicate -> selection-mask words (masked_launch.hpp).
bool launch_decode_fused_tree_mask_out(codegen::jit::FusedTree const& tree,
                                       codegen::jit::LabeledBuffers& labeled,
                                       char const* dtype,
                                       std::int64_t num_rows,
                                       ::sirius::codegen::range_predicate pred,
                                       ::sirius::codegen::selection_mask& mask,
                                       rmm::cuda_stream_view stream)
{
  VariantLaunchArgs va;
  va.shape    = cdj::kShapeMaskOut;
  va.pred_lo  = pred.lo;
  va.pred_hi  = pred.hi;
  va.sel_mask = dev(mask.words);
  // The `out` slot carries the mask words: a ballot writes no column.
  return check_and_launch(
    {tree, labeled, dtype, num_rows, mask, stream}, va, mask.words, "range ballot", num_rows);
}

// Compacting value decode, either enumeration (masked_launch.hpp). The
// enumerator is chosen by the caller's `row_indices`: the shape, the contract
// and the trailing args all follow from it, so the two walks cannot drift.
bool launch_decode_fused_tree_compacted(codegen::jit::FusedTree const& tree,
                                        codegen::jit::LabeledBuffers& labeled,
                                        char const* dtype,
                                        std::int64_t num_rows,
                                        ::sirius::codegen::selection_mask const& mask,
                                        std::int32_t const* row_indices,
                                        void* out,
                                        rmm::cuda_stream_view stream)
{
  bool const by_index = row_indices != nullptr;
  VariantLaunchArgs va;
  va.shape         = by_index ? cdj::kShapeIndexConsume : cdj::kShapeMaskConsume;
  va.chunk_offsets = dev(mask.chunk_offsets);
  if (by_index) {
    va.row_indices = dev(row_indices);
  } else {
    va.sel_mask = dev(mask.words);
  }
  return check_and_launch(
    {tree, labeled, dtype, num_rows, mask, stream},
    va,
    out,
    by_index ? "compacted decode (index walk)" : "compacted decode (mask walk)",
    num_rows);
}

// str_split phase 1: survivor metadata (masked_launch.hpp). The kernel runs over
// the OFFSETS domain (rows + 1) while the mask stays row-space.
bool launch_decode_fused_tree_str_split_meta(codegen::jit::FusedTree const& tree,
                                             codegen::jit::LabeledBuffers& labeled,
                                             char const* dtype,
                                             std::int64_t num_string_rows,
                                             ::sirius::codegen::selection_mask const& mask,
                                             std::int64_t* src_offsets_out,
                                             std::int32_t* lengths_out,
                                             rmm::cuda_stream_view stream)
{
  VariantLaunchArgs va;
  va.shape         = cdj::kShapeStrSplitMeta;
  va.sel_mask      = dev(mask.words);
  va.chunk_offsets = dev(mask.chunk_offsets);
  va.len_out       = dev(lengths_out);
  return check_and_launch({tree, labeled, dtype, num_string_rows, mask, stream},
                          va,
                          src_offsets_out,
                          "str_split metadata",
                          num_string_rows + 1);
}

// Dictionary gather: constant-width key gather (masked_launch.hpp).
bool launch_decode_fused_tree_dict_gather(codegen::jit::FusedTree const& tree,
                                          codegen::jit::LabeledBuffers& labeled,
                                          char const* dtype,
                                          std::int64_t num_rows,
                                          ::sirius::codegen::selection_mask const& mask,
                                          void const* keys_chars,
                                          std::int32_t key_width,
                                          void* out_chars,
                                          rmm::cuda_stream_view stream)
{
  VariantLaunchArgs va;
  va.shape         = cdj::kShapeDictGather;
  va.sel_mask      = dev(mask.words);
  va.chunk_offsets = dev(mask.chunk_offsets);
  va.keys_chars    = dev(keys_chars);
  va.key_width     = key_width;
  return check_and_launch(
    {tree, labeled, dtype, num_rows, mask, stream}, va, out_chars, "dictionary gather", num_rows);
}

// str_split phase 2: fixed masked char-range copy (masked_launch.hpp).  The
// source comes from the renderer like every other kernel's — no tree behind it,
// so it takes no arguments — and compiles through the same JIT cache.
bool launch_masked_char_copy(void const* chars,
                             std::int64_t const* src_offsets,
                             std::int32_t const* out_offsets,
                             std::int64_t n_survivors,
                             void* out_chars,
                             rmm::cuda_stream_view stream)
{
  if (n_survivors <= 0) return true;  // empty selection: nothing to copy
  if (chars == nullptr || src_offsets == nullptr || out_offsets == nullptr ||
      out_chars == nullptr) {
    std::fprintf(stderr, "simpatico::codegen: launch_masked_char_copy: null input\n");
    return false;
  }
  const cdj::DecodeKernelSpec spec  = cdj::render_masked_char_copy();
  const jit::CompiledKernel* kernel = compile_rendered(spec.source,
                                                       spec.entry_symbol,
                                                       /*default_device=*/false,
                                                       "CODEGEN_JIT_DUMP_DECODE_SOURCE",
                                                       "masked char copy");
  if (kernel == nullptr) { return false; }

  CUdeviceptr d_chars  = reinterpret_cast<CUdeviceptr>(chars);
  CUdeviceptr d_src    = reinterpret_cast<CUdeviceptr>(src_offsets);
  CUdeviceptr d_offs   = reinterpret_cast<CUdeviceptr>(out_offsets);
  CUdeviceptr d_out    = reinterpret_cast<CUdeviceptr>(out_chars);
  long long n          = static_cast<long long>(n_survivors);
  void* args[]         = {&d_chars, &d_src, &d_offs, &n, &d_out};
  const unsigned block = static_cast<unsigned>(spec.block_x);
  const unsigned grid =
    static_cast<unsigned>(std::min<std::int64_t>(8192, (n_survivors + block - 1) / block));
  SIMPATICO_CU_CHECK(cuLaunchKernel(kernel->func_for_current_device(),
                                    grid,
                                    1,
                                    1,
                                    block,
                                    1,
                                    1,
                                    0,
                                    reinterpret_cast<CUstream>(stream.value()),
                                    args,
                                    nullptr),
                     false,
                     "masked char copy: cuLaunchKernel failed");
  SIMPATICO_CUDA_CHECK(
    cudaStreamSynchronize(stream.value()), false, "masked char copy: stream sync failed");
  return true;
}

}  // namespace simpatico

// =====================================================================
//                            ENCODE BRIDGE
// =====================================================================
//
// Counterpart to the make/launch/release decompress trio above. The compress
// driver (plan/compress.cpp) calls ``encode_fused_subtree`` (defined at the
// bottom of this section) for each plan node that roots a fusable region.
//
// Pipeline (per region)
// ---------------------
//   1. ``extract_fusable_subtree`` builds the fusable FusedTree out of
//      {Bitpack, Delta, Rle} PlanTree nodes rooted at ``start_node`` (via the
//      shared build_fused_tree). Bitpack is always a leaf; Delta/Rle must have
//      all required child ports followed by fusable nodes. Returns nullopt for
//      non-fusable nodes, in which case the compress walk handles the node with
//      the generic (non-fused) path.
//   2. ``cje::render`` walks the tree and emits the kernel +
//      ``EncodeBufferSpec`` list in DFS-preorder node_id order.
//   3. ``cje::compile_plain_kernel`` via the kernel cache.
//   4. ``rmm::device_buffer`` per EncodeBufferSpec; pointers passed to
//      ``cuLaunchKernel`` in walker-mandated order.
//   5. Per-node post-processing:
//        * Rle  : device-side inclusive scan of rle_runs_offsets
//                 (kernel writes raw nruns at idx+1 and 0 at idx 0).
//        * Bp   : DtoH live_words[N] sum on host → live_packed_bytes
//                 for the sparse rep ctor.
//   6. Construct the appropriate rep per node at its DSL path, move
//      ownership of the rmm::device_buffers into the cudf::columns.
//      Bitpack reps are born sparse (OverAllocate) then eagerly
//      compacted in place, so the writer / tail codecs see dense bytes.
//   7. Reps land in ``builder.leaves`` (path-keyed). Entropy tails / chained
//      steps on the region's boundary outputs are scheduled by the recursive
//      compress walk (plan/compress.cpp), not here.

// Extract the maximal fusable region rooted at ``start_node`` into a
// CodegenHead. Encode-internal helper (mirrors the decode binder's use of the
// shared build_fused_tree in plan/decompress.cpp); folded into the file so the
// encode/decode bridge exposes the high-level subtree operations and separate
// low-level launch operations
// pair. Not part of the public header.
static std::optional<simpatico::CodegenHead> extract_fusable_subtree(
  simpatico::PlanTree const& tree, simpatico::NodeId start_node, std::string* err)
{
  using simpatico::build_fused_tree;
  using simpatico::CodegenHead;

  auto fail = [&](const std::string& reason) -> std::optional<CodegenHead> {
    if (err) *err = reason;
    return std::nullopt;
  };

  if (start_node >= tree.nodes.size()) {
    return fail("extract_fusable_subtree: start_node out of range");
  }

  // Build the fused-region FusedTree with the SAME builder the decode binder
  // uses (bridge/fused_tree_build) — the two sides can never drift on structure
  // or node-id order. The encode kernel always emits the OverAllocate layout.
  auto built = build_fused_tree(tree, start_node);
  if (!built) { return fail("no fusable subtree rooted at node " + std::to_string(start_node)); }

  CodegenHead head;
  head.tree = built->tree;

  // Covered PlanTree nodes (DFS-preorder == jit node_id): the compress walk
  // marks these done and recurses into the boundary outputs (child edges that
  // leave the covered set). Synthesized Raw passthrough leaves have no PlanTree
  // op node of their own — their bytes belong to the parent rle's dropped
  // values channel — so they are skipped here.
  head.covered_nodes.reserve(built->preorder.size());
  for (auto const& origin : built->preorder) {
    if (!origin.is_raw_passthrough) head.covered_nodes.push_back(origin.plan_node);
  }
  // Full preorder origins (including raw passthroughs) for the encode launcher
  // to map jit node_id → PlanTree NodeId when keying reps.
  head.preorder = built->preorder;
  return head;
}

namespace {

// Look up a (node_id, field) in spec.buffers.  Linear scan — buffers
// list is tiny (≤5 per node, ≤dozens per tree).  Aborts (debug only)
// if the field is missing: it's a render/codegen contract violation
// that should never happen at runtime.
std::size_t find_buffer_idx(const std::vector<cje::EncodeBufferSpec>& bufs,
                            std::int32_t node_id,
                            std::string_view field)
{
  for (std::size_t i = 0; i < bufs.size(); ++i) {
    if (bufs[i].node_id == node_id && bufs[i].field == field) return i;
  }
  std::fprintf(stderr,
               "simpatico::codegen: encode bridge: BUG - missing buffer (nid=%d field=%.*s); "
               "EncodeBufferSpec list out of sync with walker emit_X — likely a recent walker "
               "change that didn't update the bridge's per-op buffer fan-out\n",
               node_id,
               static_cast<int>(field.size()),
               field.data());
  std::abort();
}

}  // namespace

// Shared encode implementation. ``head`` is the already-extracted fusable
// region; ``head.preorder`` carries the per-node PlanTree NodeId mapping so
// reps can be keyed by NodeId rather than DSL path.
// ``stream``/``mr`` may be null/default — callers pass explicit values for
// multi-stream column workers.
static int launch_encode_fused_tree_impl(const simpatico::CodegenHead& head,
                                         cudf::data_type original_type,
                                         const char* cxx_dtype,
                                         std::int64_t num_rows,
                                         std::uintptr_t data_ptr,
                                         simpatico::fused_leaf_builder* builder,
                                         rmm::cuda_stream_view stream,
                                         rmm::device_async_resource_ref mr)
{
  if (builder == nullptr || cxx_dtype == nullptr) { return -1; }

  try {
    const std::int32_t num_chunks = static_cast<std::int32_t>(
      std::max<std::int64_t>(1, (num_rows + kChunkSize - 1) / kChunkSize));

    // Render the kernel + buffer manifest.
    cje::EncodeKernelSpec spec;
    try {
      spec = cje::render(*head.tree, cxx_dtype, num_chunks);
    } catch (const cje::RenderError& e) {
      std::fprintf(stderr, "simpatico::codegen: cpp encode: render rejected shape: %s\n", e.what());
      return 0;  // walker can't emit this shape — try legacy
    }
    // Compile via the shared in-process cache.  Cold path runs nvrtc
    // (~300 ms); warm path is a hash lookup over (source, entry_symbol,
    // arch_cc).  Since spec.source is invariant in num_chunks (the
    // kernel reads chunk_id from blockIdx.x), every column with the
    // same fused-tree shape and dtype hits the cached compile —
    // including different files / different num_rows.  Cache owns the
    // CompiledKernel; we hold a non-owning pointer for the launch.
    const jit::CompiledKernel* kernel = compile_rendered(spec.source,
                                                         spec.entry_symbol,
                                                         /*default_device=*/false,
                                                         "CODEGEN_JIT_DUMP_ENCODE_SOURCE",
                                                         "cpp encode");
    if (kernel == nullptr) { return -1; }

    // Allocate one rmm::device_buffer per EncodeBufferSpec.  Buffers are moved
    // into cudf::columns during rep construction below — the vector
    // serves as the kernel-arg backing storage in the meantime.
    // ``stream`` is always supplied by the caller (encode_fused_subtree →
    // CompressWalk stream); no default-stream fallback.

    std::vector<rmm::device_buffer> bufs;
    bufs.reserve(spec.buffers.size());
    std::vector<CUdeviceptr> dev_ptrs(spec.buffers.size(), 0);
    for (std::size_t i = 0; i < spec.buffers.size(); ++i) {
      const std::size_t bytes = spec.buffers[i].length * spec.buffers[i].elem_size;
      bufs.emplace_back(bytes, stream, mr);
      dev_ptrs[i] = reinterpret_cast<CUdeviceptr>(bufs.back().data());
    }

    // Pre-zero buffers that need a clean slate before the kernel.
    // "packed": atomicOr requires a zeroed slate; skip if marked no_pre_zero
    //   (smem-accumulation path zeroes only the words it uses inline).
    // "lw_shards": always zero (accumulates live-word counts via atomicAdd).
    for (std::size_t i = 0; i < spec.buffers.size(); ++i) {
      const auto& f = spec.buffers[i].field;
      if (f == "lw_shards" || (f == "packed" && !spec.buffers[i].no_pre_zero)) {
        const std::size_t bytes = spec.buffers[i].length * spec.buffers[i].elem_size;
        if (bytes > 0)
          cudaMemsetAsync(reinterpret_cast<void*>(dev_ptrs[i]), 0, bytes, stream.value());
      }
    }

    // Build the kernel arg vector.  Order: (flat, n, then EncodeBufferSpecs
    // in walker-emit order — matches add_param order inside the
    // emit_X functions).
    CUdeviceptr d_flat   = static_cast<CUdeviceptr>(data_ptr);
    std::int64_t total_n = num_rows;
    std::vector<void*> args;
    args.reserve(2 + dev_ptrs.size());
    args.push_back(&d_flat);
    args.push_back(&total_n);
    for (auto& p : dev_ptrs)
      args.push_back(&p);

    if (!maybe_raise_smem(
          kernel->func_for_current_device(), static_cast<int>(spec.shared_bytes), "cpp encode"))
      return -1;
    SIMPATICO_CU_CHECK(cuLaunchKernel(kernel->func_for_current_device(),
                                      static_cast<unsigned>(num_chunks),
                                      1,
                                      1,
                                      static_cast<unsigned>(spec.block_x),
                                      1,
                                      1,
                                      static_cast<unsigned>(spec.shared_bytes),
                                      stream.value(),
                                      args.data(),
                                      nullptr),
                       -1,
                       "cpp encode: cuLaunchKernel failed");

    // Per-node post-processing + rep construction.
    //
    // No explicit sync here.  All post-encode GPU work (rle_runs_offsets
    // CUB scan, fill_stride_offsets, DtoH lw_shards) runs on the same
    // legacy default stream as the encode kernel, so stream ordering
    // guarantees correct sequencing without a redundant barrier.
    // Synchronisation happens exactly where it must:
    //   - Bitpack case: explicit cudaStreamSynchronize after DtoH lw_shards.
    //   - Raw case: blocking cudaMemcpy for total_runs (acts as final sync).

    // Walk nodes in preorder (index == jit node_id).  For each real op,
    // builder->leaves is keyed by the PlanTree NodeId; for Raw passthrough
    // nodes the rep lands in builder->raw_passthrough_leaves (parent_node key).
    for (std::int32_t node_id = 0; node_id < static_cast<std::int32_t>(head.preorder.size());
         ++node_id) {
      const simpatico::FusedNodeOrigin& origin = head.preorder[node_id];
      const jit::FusedTree& node               = *origin.node;

      switch (node.op) {
        case cc::OpKind::Bitpack: {
          const auto i_min  = find_buffer_idx(spec.buffers, node_id, "chunk_min");
          const auto i_cnt  = find_buffer_idx(spec.buffers, node_id, "chunk_count");
          const auto i_bits = find_buffer_idx(spec.buffers, node_id, "chunk_bits");
          const auto i_pkd  = find_buffer_idx(spec.buffers, node_id, "packed");
          const auto i_lws  = find_buffer_idx(spec.buffers, node_id, "lw_shards");

          // Sharded live-word sum: 16 shards × uint32, spaced 32 words apart
          // (128-byte stride keeps each shard on a separate L2 cache line).
          // DtoH only 16 × 32 × 4 = 2048 bytes (vs num_chunks × 4 bytes before).
          // The device buffer is NOT needed after this — live_packed_bytes is
          // all the caller requires; the compact_bitpack_packed pipeline reconstructs
          // per-chunk offsets from chunk_bits/chunk_count on demand.
          constexpr std::size_t kMaxBitsShards = 16;
          constexpr std::size_t kShardStride   = 32;
          std::vector<std::uint32_t> lw_shards(kMaxBitsShards * kShardStride, 0);
          SIMPATICO_CUDA_CHECK(
            cudaMemcpyAsync(lw_shards.data(),
                            reinterpret_cast<const void*>(dev_ptrs[i_lws]),
                            kMaxBitsShards * kShardStride * sizeof(std::uint32_t),
                            cudaMemcpyDeviceToHost,
                            stream.value()),
            -1,
            "cpp encode: lw_shards DtoH failed (nid=%d)",
            node_id);
          cudaStreamSynchronize(stream.value());
          std::int64_t live_packed_bytes = 0;
          for (std::size_t s = 0; s < kMaxBitsShards; ++s)
            live_packed_bytes += static_cast<std::int64_t>(lw_shards[s * kShardStride]);
          live_packed_bytes *= 4;  // uint32 words → bytes

          // ``stride_words`` from the renderer — the encode-side OverAllocate
          // slot stride. Passed to compact_bitpack_packed to gather the padded
          // ``packed`` scratch tight before the rep is published.
          const std::int32_t stride_words = static_cast<std::int32_t>(
            spec.buffers[i_pkd].length / static_cast<std::size_t>(num_chunks));
          // packed is uint32 words; a UINT32 column (size = word count) keeps a
          // >2GB packed buffer under cudf's 2^31-element cap. Guard the >8GB edge.
          if (spec.buffers[i_pkd].length > static_cast<std::size_t>(INT32_MAX)) {
            std::fprintf(stderr,
                         "simpatico::codegen: cpp encode: bitpack packed too large (%zu words)\n",
                         spec.buffers[i_pkd].length);
            return -1;
          }

          // Construct the rep.  chunk_min's dtype follows the bitpack
          // channel's element size (from the renderer's emit_bitpack), not
          // the column's original_type.  For the values channel this matches
          // original_type, but for the runs channel (int32 counts) it must
          // be INT32 regardless of the column dtype.
          const cudf::data_type bp_elem_type =
            (spec.buffers[i_min].elem_size == 8)   ? cudf::data_type(cudf::type_id::INT64)
            : (spec.buffers[i_min].elem_size == 2) ? cudf::data_type(cudf::type_id::INT16)
            : (spec.buffers[i_min].elem_size == 1) ? cudf::data_type(cudf::type_id::UINT8)
                                                   : cudf::data_type(cudf::type_id::INT32);
          auto mins_col = std::make_unique<cudf::column>(bp_elem_type,
                                                         static_cast<cudf::size_type>(num_chunks),
                                                         std::move(bufs[i_min]),
                                                         rmm::device_buffer(0, stream),
                                                         0);
          auto cnt_col  = std::make_unique<cudf::column>(cudf::data_type(cudf::type_id::INT32),
                                                        static_cast<cudf::size_type>(num_chunks),
                                                        std::move(bufs[i_cnt]),
                                                        rmm::device_buffer(0, stream),
                                                        0);
          auto bits_col = std::make_unique<cudf::column>(cudf::data_type(cudf::type_id::UINT8),
                                                         static_cast<cudf::size_type>(num_chunks),
                                                         std::move(bufs[i_bits]),
                                                         rmm::device_buffer(0, stream),
                                                         0);
          auto pkd_overalloc =
            std::make_unique<cudf::column>(cudf::data_type(cudf::type_id::UINT32),
                                           static_cast<cudf::size_type>(spec.buffers[i_pkd].length),
                                           std::move(bufs[i_pkd]),
                                           rmm::device_buffer(0, stream),
                                           0);

          // EAGERLY compact the OverAllocate ``packed`` so every fused subtree's
          // bitpack output is born Compact (dense). The meta columns
          // (chunk_min/count/bits) are stored as-is.
          //
          // FUTURE (plan: drop-overalloc — "sync+scan-halfway"): evaluate
          // doing a sync+scan partway through the encode to obtain the
          // per-chunk offsets + the exact output size, then packing tight
          // directly in the encode kernel, and measure whether that beats
          // this post-encode scan+gather.
          auto pkd_col = simpatico::compact_bitpack_packed(
            *cnt_col, *bits_col, *pkd_overalloc, stride_words, live_packed_bytes, stream, mr);
          // compact_bitpack_packed gathers on ``stream``; sync so the dense
          // bytes are safe to read from any stream before the pointer is exposed.
          SIMPATICO_CUDA_CHECK(cudaStreamSynchronize(stream.value()),
                               -1,
                               "cpp encode: bitpack eager-compact sync failed (nid=%d)",
                               node_id);
          auto rep = std::make_unique<simpatico::codegen_fused_representation>(
            simpatico::OpId::Bitpack, original_type, static_cast<cudf::size_type>(num_rows));
          rep->buffers.emplace_back("chunk_min", std::move(mins_col));
          rep->buffers.emplace_back("chunk_count", std::move(cnt_col));
          rep->buffers.emplace_back("chunk_bits", std::move(bits_col));
          rep->buffers.emplace_back("packed", std::move(pkd_col));
          builder->leaves.emplace(origin.plan_node, std::move(rep));
          break;
        }

        case cc::OpKind::Delta: {
          const auto i_first = find_buffer_idx(spec.buffers, node_id, "delta_first");
          // delta_first's width follows the rendered buffer, not a blanket
          // original_type — Delta may be nested under a channel with a
          // different width than the column (e.g. rle's always-int32
          // "runs"), same reasoning as the Raw-passthrough fix above.
          const std::int32_t first_elem_size =
            static_cast<std::int32_t>(spec.buffers[i_first].elem_size);
          const cudf::data_type first_elem_type =
            (first_elem_size == static_cast<std::int32_t>(cudf::size_of(original_type)))
              ? original_type
            : (first_elem_size == 8) ? cudf::data_type(cudf::type_id::INT64)
            : (first_elem_size == 1) ? cudf::data_type(cudf::type_id::UINT8)
                                     : cudf::data_type(cudf::type_id::INT32);
          auto first_col = std::make_unique<cudf::column>(first_elem_type,
                                                          static_cast<cudf::size_type>(num_chunks),
                                                          std::move(bufs[i_first]),
                                                          rmm::device_buffer(0, stream),
                                                          0);
          auto rep       = std::make_unique<simpatico::codegen_fused_representation>(
            simpatico::OpId::Delta, original_type, static_cast<cudf::size_type>(num_rows));
          rep->buffers.emplace_back("delta_first", std::move(first_col));
          builder->leaves.emplace(origin.plan_node, std::move(rep));
          break;
        }

        case cc::OpKind::Rle: {
          const auto i_off = find_buffer_idx(spec.buffers, node_id, "rle_runs_offsets");

          // Kernel pre-staged the buffer as [0, nruns0, nruns1, ...,
          // nruns_{N-1}] of length N+1.  Convert to the decoder's
          // exclusive-prefix layout [0, nruns0, nruns0+nruns1, ...,
          // sum(nruns)] via a single in-place CUB InclusiveSum on the
          // same stream as the encode kernel — no host roundtrip.
          // Decoder contract: the exclusive-prefix runs_offsets layout the
          // decode binder reads as the Rle ``rle_runs_offsets`` slot.
          const std::size_t off_count = static_cast<std::size_t>(num_chunks) + 1;

          std::size_t tmp_bytes = 0;
          if (int rc = simpatico_compute_rle_offsets_inclusive_scan(
                reinterpret_cast<void*>(dev_ptrs[i_off]),
                static_cast<std::int32_t>(off_count),
                /*d_temp_storage=*/nullptr,
                &tmp_bytes,
                /*stream=*/stream.value());
              rc != 0) {
            std::fprintf(stderr,
                         "simpatico::codegen: cpp encode: rle_runs_offsets scan probe "
                         "failed (nid=%d, cudaError=%d)\n",
                         node_id,
                         rc);
            return -1;
          }
          rmm::device_buffer scratch(tmp_bytes, stream, mr);
          if (int rc = simpatico_compute_rle_offsets_inclusive_scan(
                reinterpret_cast<void*>(dev_ptrs[i_off]),
                static_cast<std::int32_t>(off_count),
                tmp_bytes > 0 ? scratch.data() : nullptr,
                &tmp_bytes,
                /*stream=*/stream.value());
              rc != 0) {
            std::fprintf(stderr,
                         "simpatico::codegen: cpp encode: rle_runs_offsets scan run "
                         "failed (nid=%d, cudaError=%d)\n",
                         node_id,
                         rc);
            return -1;
          }

          auto off_col = std::make_unique<cudf::column>(cudf::data_type(cudf::type_id::INT32),
                                                        static_cast<cudf::size_type>(off_count),
                                                        std::move(bufs[i_off]),
                                                        rmm::device_buffer(0, stream),
                                                        0);
          auto rep     = std::make_unique<simpatico::codegen_fused_representation>(
            simpatico::OpId::Rle, original_type, static_cast<cudf::size_type>(num_rows));
          rep->buffers.emplace_back("rle_runs_offsets", std::move(off_col));
          builder->leaves.emplace(origin.plan_node, std::move(rep));
          break;
        }

        case cc::OpKind::For: {
          // `references` holds num_chunks per-chunk minimums, normally the
          // same dtype as the column — but its width follows the rendered
          // buffer, not a blanket original_type, since For may be nested
          // under a channel with a different width than the column (e.g.
          // rle's always-int32 "runs"), same reasoning as the
          // Raw-passthrough fix above. Buffer layout is flat:
          // references[chunk_id].
          const auto i_refs = find_buffer_idx(spec.buffers, node_id, "references");
          const std::int32_t refs_elem_size =
            static_cast<std::int32_t>(spec.buffers[i_refs].elem_size);
          const cudf::data_type refs_elem_type =
            (refs_elem_size == static_cast<std::int32_t>(cudf::size_of(original_type)))
              ? original_type
            : (refs_elem_size == 8) ? cudf::data_type(cudf::type_id::INT64)
            : (refs_elem_size == 1) ? cudf::data_type(cudf::type_id::UINT8)
                                    : cudf::data_type(cudf::type_id::INT32);
          auto refs_col = std::make_unique<cudf::column>(refs_elem_type,
                                                         static_cast<cudf::size_type>(num_chunks),
                                                         std::move(bufs[i_refs]),
                                                         rmm::device_buffer(0, stream),
                                                         0);
          auto rep      = std::make_unique<simpatico::codegen_fused_representation>(
            simpatico::OpId::For, original_type, static_cast<cudf::size_type>(num_rows));
          rep->buffers.emplace_back("references", std::move(refs_col));
          builder->leaves.emplace(origin.plan_node, std::move(rep));
          break;
        }

        case cc::OpKind::Zigzag: {
          // Transformer mode (fused child on the `zigzag` channel): ZigZag
          // rewrote the lane value inline and stored NOTHING — the child owns
          // all buffers, so there is no rep to build for this node.
          if (!node.children.empty()) break;
          // Leaf mode: `zigzag` holds the transformed stream in fixed-stride
          // OverAllocate layout: num_chunks * kChunkSize elements, element i of
          // chunk c at [c*kChunkSize + i].  Decode is layout-aware (reads
          // chunk_id*CHUNK + i), so no compaction is needed — the channel is
          // published as-is and routed to a downstream codec (or stored) like
          // any other rep channel.
          const auto i_zz = find_buffer_idx(spec.buffers, node_id, "zigzag");
          const std::size_t zz_rows =
            static_cast<std::size_t>(num_chunks) * static_cast<std::size_t>(kChunkSize);
          // Use the renderer's actual buffer elem_size to determine the column type.
          // When Zigzag sits on an RLE "runs" channel the kernel processes int32_t
          // run counts, so elem_size == 4 regardless of the column's element type.
          // Using original_type for INT16 columns would make ANS compress only the
          // low 2 bytes of each int32_t code, corrupting the run counts on decode.
          const std::int32_t zz_elem_size = static_cast<std::int32_t>(spec.buffers[i_zz].elem_size);
          const cudf::data_type zz_type =
            (zz_elem_size == static_cast<std::int32_t>(cudf::size_of(original_type)))
              ? original_type
            : (zz_elem_size == 8) ? cudf::data_type(cudf::type_id::INT64)
            : (zz_elem_size == 1) ? cudf::data_type(cudf::type_id::UINT8)
                                  : cudf::data_type(cudf::type_id::INT32);
          auto zz_col = std::make_unique<cudf::column>(zz_type,
                                                       static_cast<cudf::size_type>(zz_rows),
                                                       std::move(bufs[i_zz]),
                                                       rmm::device_buffer(0, stream),
                                                       0);
          auto rep    = std::make_unique<simpatico::codegen_fused_representation>(
            simpatico::OpId::Zigzag, zz_type, static_cast<cudf::size_type>(num_rows));
          rep->buffers.emplace_back("zigzag", std::move(zz_col));
          builder->leaves.emplace(origin.plan_node, std::move(rep));
          break;
        }

        case cc::OpKind::Raw: {
          // Synthesized Raw passthrough leaf (is_raw_passthrough == true).
          // No PlanTree op node of its own.  The rep is keyed by
          // origin.parent_node in builder->raw_passthrough_leaves; the compress
          // driver parks it in tree.nodes[parent_node].channels under the
          // output path for origin.parent_channel.
          //
          // Two layout families, selected by parent_op:
          //
          //   Fixed-stride ("for" and "delta"): the kernel wrote
          //     data[c*CHUNK + t] for each element t of chunk c.
          //     The offsets buffer holds c*CHUNK for chunk c — a simple stride.
          //     No compaction needed; data and offsets are copied as-is.
          //
          //   Compact ("rle", for both "values" and "runs"): the kernel wrote
          //     data in padded OverAllocate layout; rle_runs_offsets gives the
          //     per-chunk start + total element count, so we scatter → compact
          //     via simpatico_compact_raw_values.
          //
          // Element width/type follow the actual rendered buffer, not a
          // blanket original_type derived from the outer column alone: this
          // node's input may be a channel with a different width than the
          // column (e.g. delta/for nested under rle's always-int32 "runs"
          // channel renders 4-byte elements even when the column is i64/f64).
          // The renderer already resolves each node's own dtype correctly —
          // this mirrors the fix already applied to bitpack's chunk_min
          // above (spec.buffers[i_min].elem_size) instead of re-deriving
          // width from a "is this channel literally named 'runs'" special
          // case plus original_type.

          const bool is_fixed_stride = (origin.parent_op == "for" || origin.parent_op == "delta");

          const auto i_data            = find_buffer_idx(spec.buffers, node_id, "data");
          const std::int32_t elem_size = static_cast<std::int32_t>(spec.buffers[i_data].elem_size);
          const cudf::data_type data_elem_type =
            (elem_size == static_cast<std::int32_t>(cudf::size_of(original_type))) ? original_type
            : (elem_size == 8) ? cudf::data_type(cudf::type_id::INT64)
            : (elem_size == 1) ? cudf::data_type(cudf::type_id::UINT8)
                               : cudf::data_type(cudf::type_id::INT32);

          if (is_fixed_stride) {
            // ─── Fixed-stride Raw (FOR deltas / Delta differences) ─────────
            // The kernel wrote data[c*CHUNK + t] and offsets[c] = c*CHUNK.
            const auto i_offs = find_buffer_idx(spec.buffers, node_id, "offsets");

            auto data_col =
              std::make_unique<cudf::column>(data_elem_type,
                                             static_cast<cudf::size_type>(num_chunks * kChunkSize),
                                             std::move(bufs[i_data]),
                                             rmm::device_buffer(0, stream),
                                             0);
            auto offs_col =
              std::make_unique<cudf::column>(cudf::data_type(cudf::type_id::INT32),
                                             static_cast<cudf::size_type>(num_chunks + 1),
                                             std::move(bufs[i_offs]),
                                             rmm::device_buffer(0, stream),
                                             0);

            auto rep = std::make_unique<simpatico::codegen_fused_representation>(
              simpatico::OpId::Identity, original_type, static_cast<cudf::size_type>(num_rows));
            rep->buffers.emplace_back("data", std::move(data_col));
            rep->buffers.emplace_back("offsets", std::move(offs_col));
            builder->raw_passthrough_leaves.push_back(
              {origin.parent_node, origin.parent_channel, std::move(rep)});
            break;
          }

          {
            // ─── Compact Raw (RLE values / RLE runs) ──────────
            // Scatter padded OverAllocate → compact using rle_runs_offsets.
            //
            // Locate the parent RLE's node_id by scanning backward: the RLE
            // node could be at index >0 when there is a transformer (e.g.
            // Delta) in front of it in the same fused region.
            std::int32_t rle_node_id = -1;
            for (std::int32_t j = node_id - 1; j >= 0; --j) {
              const simpatico::FusedNodeOrigin& o = head.preorder[j];
              if (!o.is_raw_passthrough && o.node->op == cc::OpKind::Rle &&
                  o.plan_node == origin.parent_node) {
                rle_node_id = j;
                break;
              }
            }
            if (rle_node_id < 0) {
              std::fprintf(stderr,
                           "simpatico::codegen: encode bridge: BUG - cannot find parent RLE "
                           "node in preorder (nid=%d parent_node=%u); preorder may be malformed\n",
                           node_id,
                           static_cast<unsigned>(origin.parent_node));
              return -1;
            }
            const auto i_rle_off = find_buffer_idx(spec.buffers, rle_node_id, "rle_runs_offsets");
            std::int32_t total_runs = 0;
            {
              const CUdeviceptr d_tail =
                dev_ptrs[i_rle_off] + static_cast<CUdeviceptr>(num_chunks) * sizeof(std::int32_t);
              SIMPATICO_CUDA_CHECK(cudaMemcpyAsync(&total_runs,
                                                   reinterpret_cast<const void*>(d_tail),
                                                   sizeof(std::int32_t),
                                                   cudaMemcpyDeviceToHost,
                                                   stream.value()),
                                   -1,
                                   "RLE Raw compact (%s): total_runs DtoH failed",
                                   origin.parent_channel.c_str());
              cudaStreamSynchronize(stream.value());
            }

            rmm::device_buffer compact_buf(
              static_cast<std::size_t>(total_runs) * static_cast<std::size_t>(elem_size),
              stream,
              mr);
            if (total_runs > 0) {
              if (int rc =
                    simpatico_compact_raw_values(reinterpret_cast<const void*>(dev_ptrs[i_data]),
                                                 compact_buf.data(),
                                                 reinterpret_cast<const void*>(dev_ptrs[i_rle_off]),
                                                 num_chunks,
                                                 kChunkSize,
                                                 elem_size,
                                                 stream.value());
                  rc != 0) {
                std::fprintf(
                  stderr,
                  "simpatico::codegen: RLE Raw compact (%s): compact_raw_values failed (%d)\n",
                  origin.parent_channel.c_str(),
                  rc);
                return -1;
              }
              cudaStreamSynchronize(stream.value());
            }

            auto data_col = std::make_unique<cudf::column>(data_elem_type,
                                                           static_cast<cudf::size_type>(total_runs),
                                                           std::move(compact_buf),
                                                           rmm::device_buffer(0, stream),
                                                           0);

            const std::size_t offs_bytes =
              static_cast<std::size_t>(num_chunks + 1) * sizeof(std::int32_t);
            rmm::device_buffer offs_buf(offs_bytes, stream, mr);
            SIMPATICO_CUDA_CHECK(cudaMemcpyAsync(offs_buf.data(),
                                                 reinterpret_cast<const void*>(dev_ptrs[i_rle_off]),
                                                 offs_bytes,
                                                 cudaMemcpyDeviceToDevice,
                                                 stream.value()),
                                 -1,
                                 "RLE Raw compact (%s): offsets DtoD failed",
                                 origin.parent_channel.c_str());
            cudaStreamSynchronize(stream.value());
            auto offs_col =
              std::make_unique<cudf::column>(cudf::data_type(cudf::type_id::INT32),
                                             static_cast<cudf::size_type>(num_chunks + 1),
                                             std::move(offs_buf),
                                             rmm::device_buffer(0, stream),
                                             0);

            auto rep = std::make_unique<simpatico::codegen_fused_representation>(
              simpatico::OpId::Identity, original_type, static_cast<cudf::size_type>(num_rows));
            rep->buffers.emplace_back("data", std::move(data_col));
            rep->buffers.emplace_back("offsets", std::move(offs_col));
            builder->raw_passthrough_leaves.push_back(
              {origin.parent_node, origin.parent_channel, std::move(rep)});
          }
          break;
        }

        case cc::OpKind::None:
        default:
          std::fprintf(stderr,
                       "simpatico::codegen: cpp encode: BUG - unexpected op '%s' at node %d "
                       "(extract_fusable_subtree should have filtered)\n",
                       jit::op_kind_name(node.op).c_str(),
                       node_id);
          return -1;
      }
    }

    // The fused region's reps are now in builder.leaves (NodeId-keyed) and
    // builder.raw_passthrough_leaves. Entropy tails / chained steps on this
    // region's boundary outputs are scheduled by the recursive compress walk
    // (CompressWalk::emit_fused_step), not here.
    return 1;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "simpatico::codegen: cpp encode: exception: %s\n", e.what());
    return -1;
  } catch (...) {
    std::fprintf(stderr, "simpatico::codegen: cpp encode: unknown exception\n");
    return -1;
  }
}

namespace simpatico {

bool launch_encode_fused_tree(CodegenHead const& head,
                              cudf::column_view const& input_col,
                              rmm::cuda_stream_view stream,
                              rmm::device_async_resource_ref const& mr,
                              fused_leaf_builder& builder,
                              std::string* error_out)
{
  // Fixed-point columns are physically their storage integer (DECIMAL32 -> INT32,
  // DECIMAL64 -> INT64); the codecs run losslessly on those bits. We encode/decode
  // as the storage integer and restore the DECIMAL type (and scale) from the stored
  // column dtype at the end of decompress (see apply_stored_dtype).
  const char* dtype             = nullptr;
  cudf::type_id storage_type_id = input_col.type().id();
  switch (input_col.type().id()) {
    case cudf::type_id::INT8: dtype = "int8"; break;
    case cudf::type_id::UINT8: dtype = "uint8"; break;
    case cudf::type_id::INT16: dtype = "int16"; break;
    case cudf::type_id::UINT16: dtype = "uint16"; break;
    case cudf::type_id::INT32: dtype = "int32"; break;
    case cudf::type_id::INT64: dtype = "int64"; break;
    case cudf::type_id::UINT32: dtype = "uint32"; break;
    case cudf::type_id::UINT64: dtype = "uint64"; break;
    case cudf::type_id::FLOAT32: dtype = "float32"; break;
    case cudf::type_id::FLOAT64: dtype = "float64"; break;
    case cudf::type_id::DECIMAL32:
      dtype           = "int32";
      storage_type_id = cudf::type_id::INT32;
      break;
    case cudf::type_id::DECIMAL64:
      dtype           = "int64";
      storage_type_id = cudf::type_id::INT64;
      break;
    // Chrono columns are physically their integer storage (DATE32 = days as
    // int32, timestamps/durations as int64); like DECIMAL, encode/decode run
    // on the storage integer and the logical type is restored from the stored
    // column dtype at the end of decompress (apply_stored_dtype).
    case cudf::type_id::TIMESTAMP_DAYS:
    case cudf::type_id::DURATION_DAYS:
      dtype           = "int32";
      storage_type_id = cudf::type_id::INT32;
      break;
    case cudf::type_id::TIMESTAMP_SECONDS:
    case cudf::type_id::TIMESTAMP_MILLISECONDS:
    case cudf::type_id::TIMESTAMP_MICROSECONDS:
    case cudf::type_id::TIMESTAMP_NANOSECONDS:
    case cudf::type_id::DURATION_SECONDS:
    case cudf::type_id::DURATION_MILLISECONDS:
    case cudf::type_id::DURATION_MICROSECONDS:
    case cudf::type_id::DURATION_NANOSECONDS:
      dtype           = "int64";
      storage_type_id = cudf::type_id::INT64;
      break;
    default:
      if (error_out)
        *error_out = "non-fusable input dtype (cudf type_id=" +
                     std::to_string(static_cast<int>(input_col.type().id())) + ")";
      return false;
  }
  const char* cxx_dtype = dtype_to_cxx(dtype);
  if (cxx_dtype == nullptr) return false;  // shouldn't happen, but guard
  const cudf::data_type original_type{storage_type_id};

  std::uintptr_t data_ptr = reinterpret_cast<std::uintptr_t>(
    input_col.head<uint8_t>() + static_cast<std::size_t>(input_col.offset()) *
                                  static_cast<std::size_t>(cudf::size_of(input_col.type())));

  int rc = launch_encode_fused_tree_impl(
    head, original_type, cxx_dtype, input_col.size(), data_ptr, &builder, stream, mr);
  if (rc == 1) return true;
  if (rc == 0) {
    if (error_out) *error_out = "encode declined (non-fusable subtree shape)";
    return false;
  }
  if (error_out) *error_out = "launch_encode_fused_tree failed";
  return false;
}

bool encode_fused_subtree(PlanTree const& tree,
                          NodeId start_node,
                          cudf::column_view input_col,
                          rmm::cuda_stream_view stream,
                          rmm::device_async_resource_ref mr,
                          fused_leaf_builder& builder,
                          std::string* error_out,
                          CodegenHead* head_out)
{
  if (start_node >= tree.nodes.size()) return false;

  std::string err;
  auto head = extract_fusable_subtree(tree, start_node, &err);
  if (!head.has_value()) {
    if (error_out) *error_out = err;
    return false;  // non-fusable node: caller runs the generic path
  }
  if (head_out) *head_out = *head;

  if (launch_encode_fused_tree(*head, input_col, stream, mr, builder, error_out)) return true;
  if (error_out && error_out->empty()) *error_out = "encode_fused_subtree failed";
  return false;
}

}  // namespace simpatico
