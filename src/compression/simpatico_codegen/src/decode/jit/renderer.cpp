// Decode-side renderer — recursive composable walker (plain CUDA).
//
// Symmetric counterpart to `encode/jit/renderer.cpp`.  Emits one plain
// CUDA `__global__` per (tree shape, dtype) that reconstructs the flat
// output column from the per-op compressed channels.
//
// Fusion model (mirrors the encode walker op-for-op)
// ==================================================
//
//   * Bitpack  — LEAF value reader.  Exposes a closed-form C++
//     expression `simpatico_bp_at(packed + base, bits, min, __POS__)` that
//     yields the decoded value at any per-chunk position.  No shared
//     memory, no sync — the parent splices the expression inline.
//
//   * Delta    — INLINE transformer.  Decode delta is a prefix sum, so
//     unlike the closed-form encode delta it needs a block-wide scan.
//     But it does NOT need an intermediate shared-mem buffer: the diff
//     source (the child's closed-form read expression) is spliced
//     directly into the scan-input load.  One `cub::BlockScan`, zero
//     `kChunkSize` scratch buffers — this is the whole point of the
//     fusion.  When the child is itself a producer (Delta/Rle) the
//     child first materialises into a shared-mem slab and the diff
//     source becomes a load from that slab.
//
//   * Rle      — STAGE BOUNDARY (the only one).  Materialises its
//     `values` and `runs` children into shared-mem slabs (each via a
//     fused producer pass), syncs, then expands with the block-collective
//     `block_rle_decompress`.  This is exactly the encode-side
//     stage-boundary treatment, just reading instead of writing.
//
// Buffer / symbol naming mirrors the encode walker: each op suffixes
// its kernel-parameter and scratch names with its node_id (assigned in
// DFS-preorder, lex-sorted children — the same order
// `jit::assign_ids` and the decode binder (bind_fused_subtree in
// plan/decompress.cpp) use, so the launcher can bind by (node_id, field)).

#include "codegen/decode/jit/renderer.hpp"

#include "codegen/jit/fused_tree.hpp"
#include "codegen/jit/render_util.hpp"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace codegen::decode::jit {

namespace {

using ::codegen::jit::make_entry_symbol;
using ::codegen::jit::replace_all;
using ::codegen::jit::unsigned_counterpart;

// ---------------------------------------------------------------------
// Dtype table — element size per supported scalar type.
// ---------------------------------------------------------------------
std::size_t dtype_elem_size(const std::string& name)
{
  if (name == "int8_t") return 1;
  if (name == "int16_t") return 2;
  if (name == "int32_t") return 4;
  if (name == "int64_t") return 8;
  return 0;  // 0 => unsupported (caller throws)
}

// Exact-width unsigned type name. A signed element must pass through this
// BEFORE widening to unsigned_counterpart wherever the widened value is
// right-shifted (ZigZag's inverse): a direct int8_t -> uint32_t cast
// sign-extends garbage above the element's width. No-op for 4/8-byte
// elements (exact width == counterpart width).
const char* exact_unsigned(std::size_t elem_size)
{
  return (elem_size == 8)   ? "uint64_t"
         : (elem_size == 2) ? "uint16_t"
         : (elem_size == 1) ? "uint8_t"
                            : "uint32_t";
}

// Substitute a value source's __POS__ token with the given position expr.
std::string at_pos(const std::string& read_expr, const std::string& pos_expr)
{
  return replace_all(read_expr, "__POS__", pos_expr);
}

// ---------------------------------------------------------------------
// ValueSource — the parent-to-child contract (decode analog of the
// encode LaneInput).  `read_expr` is a closed-form C++ expression in
// the literal token `__POS__` producing the decoded element value at
// that per-chunk position.
// ---------------------------------------------------------------------
struct ValueSource {
  std::string elem_type;
  std::string read_expr;  // closed-form C++ expr with __POS__ placeholder
};

// ---------------------------------------------------------------------
// SharedMemAllocator — stack-discipline dynamic-shared budget tracker.
// Identical contract to the encode walker's allocator: alloc returns a
// byte offset into the single `workspace[]`; mark/release_to reuse bytes
// across siblings; peak_bytes() becomes DecodeKernelSpec::shared_bytes.
// ---------------------------------------------------------------------
class SharedMemAllocator {
 public:
  using Mark = std::size_t;
  std::size_t alloc(std::size_t elem_size, std::size_t count)
  {
    const std::size_t align = std::max<std::size_t>(elem_size, 4);
    cur_                    = (cur_ + align - 1) & ~(align - 1);
    const std::size_t off   = cur_;
    cur_ += elem_size * count;
    if (cur_ > peak_) peak_ = cur_;
    return off;
  }
  Mark mark() const noexcept { return cur_; }
  void release_to(Mark m) noexcept { cur_ = m; }
  std::size_t peak_bytes() const noexcept { return peak_; }

 private:
  std::size_t cur_  = 0;
  std::size_t peak_ = 0;
};

// ---------------------------------------------------------------------
// Trailing kernel parameters — the SINGLE source of truth for what follows
// (out, n) in a rendered kernel's signature.
//
// `decl` is the emitted C++ declaration; `tag` is what the launcher binds by
// (DecodeKernelSpec::trailing).  Adding a parameter here updates the rendered
// signature and the launcher's argument order together — they cannot drift.
// ---------------------------------------------------------------------
struct TrailingParamDecl {
  TrailingParam tag;
  const char* decl;
};

// Trailing parameters contributed by the ENUMERATOR (how rows are walked).
const std::vector<TrailingParamDecl>& enumerator_params(Enumerator e)
{
  static const std::vector<TrailingParamDecl> kNone{};
  static const std::vector<TrailingParamDecl> kMaskBits{
    {TrailingParam::sel_mask, "const uint32_t* __restrict__ sel_mask"},
    {TrailingParam::chunk_offsets, "const uint32_t* __restrict__ chunk_offsets"},
  };
  static const std::vector<TrailingParamDecl> kIndexList{
    {TrailingParam::row_indices, "const int32_t* __restrict__ row_indices"},
    {TrailingParam::chunk_offsets, "const uint32_t* __restrict__ chunk_offsets"},
  };
  switch (e) {
    case Enumerator::all_rows: return kNone;
    case Enumerator::mask_bits: return kMaskBits;
    case Enumerator::index_list: return kIndexList;
  }
  return kNone;
}

// Trailing parameters contributed by the CONSUMER (what happens per row).
// Predicate constants travel here as kernel arguments so one compile serves
// every literal.
const std::vector<TrailingParamDecl>& consumer_params(Consumer c)
{
  static const std::vector<TrailingParamDecl> kNone{};
  static const std::vector<TrailingParamDecl> kBallotRange{
    {TrailingParam::pred_lo, "int64_t pred_lo"},
    {TrailingParam::pred_hi, "int64_t pred_hi"},
  };
  static const std::vector<TrailingParamDecl> kDictGather{
    {TrailingParam::keys_chars, "const char* __restrict__ keys_chars"},
    {TrailingParam::key_width, "int32_t key_width"},
  };
  static const std::vector<TrailingParamDecl> kOffsetsMeta{
    {TrailingParam::len_out, "int32_t* __restrict__ len_out"},
  };
  switch (c) {
    case Consumer::write_column: return kNone;
    case Consumer::ballot_range: return kBallotRange;
    case Consumer::dict_gather: return kDictGather;
    case Consumer::offsets_meta: return kOffsetsMeta;
  }
  return kNone;
}

// Entry-symbol suffix per shape. The exact strings are load-bearing: they key
// the JIT cache, so changing one forces a recompile of that kernel everywhere.
std::string shape_symbol_suffix(DecodeShape shape)
{
  if (shape == kShapeMaskOut) return "_mask_out";
  if (shape == kShapeMaskConsume) return "_mask_consume";
  if (shape == kShapeDictGather) return "_mask_dict";
  if (shape == kShapeIndexConsume) return "_index_consume";
  if (shape == kShapeStrSplitMeta) return "_str_meta";
  return "";  // kShapePlain
}

// The kernel's trailing parameter list: enumerator's, then consumer's.
std::vector<TrailingParamDecl> shape_trailing_params(DecodeShape shape)
{
  std::vector<TrailingParamDecl> out = enumerator_params(shape.enumerator);
  auto const& cp                     = consumer_params(shape.consumer);
  out.insert(out.end(), cp.begin(), cp.end());
  return out;
}

// The `out` slot: type and name vary by variant (mask variants repurpose it for
// the mask words; dict gather writes chars; str_split_meta writes int64 source
// offsets).
std::string out_param_decl(Consumer c, const std::string& dtype)
{
  switch (c) {
    // The ballot consumers write no column: the slot carries the mask words.
    case Consumer::ballot_range: return "uint32_t* __restrict__ sel_mask";
    case Consumer::dict_gather: return "char* __restrict__ out";
    case Consumer::offsets_meta: return "int64_t* __restrict__ out";
    case Consumer::write_column: break;
  }
  return dtype + "* __restrict__ out";
}

// ---------------------------------------------------------------------
// Walker.
// ---------------------------------------------------------------------
class Walker {
 public:
  Walker(std::string root_dtype, DecodeShape shape) : dtype_(std::move(root_dtype)), shape_(shape)
  {
  }

  DecodeKernelSpec build(const ::codegen::jit::FusedTree& tree)
  {
    assign_ids(tree);
    switch (shape_.consumer) {
      case Consumer::write_column:
        switch (shape_.enumerator) {
          case Enumerator::all_rows:
            // Root producer writes the reconstructed chunk straight to out.
            emit_producer(tree, "(out + chunk_start)", "len", dtype_);
            break;
          case Enumerator::mask_bits:
            if (tree.op == ::codegen::OpKind::Delta) {
              emit_delta_mask_consume(tree);  // tuned register path, no slab
            } else {
              // Compositional seam: any value_source-supported root (Bitpack
              // leaf closed-form, FOR, Zigzag, RLE cascades, ...) stages the
              // chunk via the existing emitters and masked-stores from it.
              emit_generic_mask_consume(tree);
            }
            break;
          case Enumerator::index_list: emit_bitpack_index_consume(tree); break;
        }
        break;
      case Consumer::ballot_range:
        // One seam: value_source() routes a Bitpack leaf to the same
        // closed-form bitpack_value_source the dedicated emitter used, so the
        // emitted body is identical for every root shape.
        emit_generic_mask_out(tree);
        break;
      case Consumer::dict_gather: emit_bitpack_mask_dict_gather(tree); break;
      case Consumer::offsets_meta: emit_str_split_meta(tree); break;
    }
    return finalize(tree);
  }

 private:
  std::string dtype_;
  DecodeShape shape_        = kShapePlain;
  static constexpr int tbs_ = ::codegen::kTBSize;
  std::ostringstream params_;
  std::ostringstream body_;
  std::vector<DecodeBufferSpec> buffers_;
  SharedMemAllocator sm_;
  std::unordered_map<const ::codegen::jit::FusedTree*, std::int32_t> ids_;

  // DFS-preorder, lex-sorted children (std::map iteration) — must match
  // jit::assign_ids and the decode binder (bind_fused_subtree) so
  // (node_id, field) keys line up.
  void assign_ids(const ::codegen::jit::FusedTree& node)
  {
    ids_.emplace(&node, static_cast<std::int32_t>(ids_.size()));
    for (const auto& [k, child] : node.children) {
      (void)k;
      assign_ids(*child);
    }
  }
  std::int32_t id_of(const ::codegen::jit::FusedTree& node) const { return ids_.at(&node); }

  void add_param(const std::string& type, const std::string& name)
  {
    if (params_.tellp() > 0) params_ << ",\n";
    params_ << "    " << type << " " << name;
  }
  void add_buffer(std::int32_t node_id, const std::string& field, std::size_t elem_size)
  {
    buffers_.push_back(DecodeBufferSpec{node_id, field, elem_size});
  }

  // ---- Emit a producer: write `node`'s full output into dst[0,len). ----
  void emit_producer(const ::codegen::jit::FusedTree& node,
                     const std::string& dst,
                     const std::string& len,
                     const std::string& elem_type);

  // ---- Build a closed-form ValueSource for `node`. ----
  // Bitpack: pure expression.  Delta/Rle: materialise into a shared
  // slab (leaving it live for the caller to read) and return a slab
  // load expression.
  ValueSource value_source(const ::codegen::jit::FusedTree& node,
                           const std::string& elem_type,
                           const std::string& len);

  void emit_bitpack_producer(const ::codegen::jit::FusedTree& node,
                             const std::string& dst,
                             const std::string& len,
                             const std::string& elem_type);

  // ---- Row-selecting variants. ----
  // The ballot emits one __ballot_sync word per 32 consecutive rows; the consuming
  // variants stage the chunk's mask via a warp scan of per-word popcounts
  // (emit_selection_stage) and write compacted output.
  void emit_selection_stage();
  void emit_mask_survivor_loop(const std::string& sink);

  // How a Delta producer writes its reconstructed chunk.
  //   plain        — every row to dst[row], as the full-column decode does.
  //   mask_compact — survivors only, compacted to dst[rank].
  //                  Requires emit_selection_stage() to have run.
  // The reconstruction itself (striped load, block scan, transpose) is
  // identical either way, which is why both share emit_delta_producer.
  enum class DeltaStore { plain, mask_compact };
  void emit_delta_mask_consume(const ::codegen::jit::FusedTree& node);
  void emit_bitpack_mask_dict_gather(const ::codegen::jit::FusedTree& node);
  void emit_bitpack_index_consume(const ::codegen::jit::FusedTree& node);
  void emit_generic_mask_consume(const ::codegen::jit::FusedTree& node);
  void emit_generic_mask_out(const ::codegen::jit::FusedTree& node);
  void emit_str_split_meta(const ::codegen::jit::FusedTree& node);
  void emit_delta_producer(const ::codegen::jit::FusedTree& node,
                           const std::string& dst,
                           const std::string& len,
                           const std::string& elem_type,
                           DeltaStore store = DeltaStore::plain);
  void emit_rle_producer(const ::codegen::jit::FusedTree& node,
                         const std::string& dst,
                         const std::string& len,
                         const std::string& elem_type);
  void emit_for_producer(const ::codegen::jit::FusedTree& node,
                         const std::string& dst,
                         const std::string& len,
                         const std::string& elem_type);
  void emit_raw_producer(const ::codegen::jit::FusedTree& node,
                         const std::string& dst,
                         const std::string& len,
                         const std::string& elem_type);
  void emit_zigzag_producer(const ::codegen::jit::FusedTree& node,
                            const std::string& dst,
                            const std::string& len,
                            const std::string& elem_type);

  // Emit ZigZag scalar-load prelude + return its closed-form (inverse) read
  // expr — reads the stored `zigzag` channel and applies the inverse map
  // inline, so a ZigZag child of Delta/FOR/RLE needs no shared-mem slab.
  ValueSource zigzag_value_source_inline(const ::codegen::jit::FusedTree& node,
                                         const std::string& elem_type);

  // Emit Bitpack scalar-load prelude + return its closed-form read expr.
  ValueSource bitpack_value_source(const ::codegen::jit::FusedTree& node,
                                   const std::string& elem_type);

  // Emit Raw scalar-load prelude + return its closed-form read expr.
  // Unlike emit_raw_producer, this returns an expression without materialising
  // into a shared-memory slab — callers can inline it directly or wrap in a
  // device functor to skip the sh_values smem allocation.
  ValueSource raw_value_source_inline(const ::codegen::jit::FusedTree& node,
                                      const std::string& elem_type);

  DecodeKernelSpec finalize(const ::codegen::jit::FusedTree& tree)
  {
    DecodeKernelSpec spec;
    spec.entry_symbol = make_entry_symbol(tree, dtype_, "simpatico_decode_");
    spec.entry_symbol += shape_symbol_suffix(shape_);

    // The `out` slot's type and meaning vary by variant; everything after
    // (out, n) comes from ONE table that yields both the declaration text and
    // spec.trailing, so what the kernel declares and what the launcher pushes
    // are the same list (see TrailingParam).
    std::vector<std::string> decls;
    decls.push_back(out_param_decl(shape_.consumer, dtype_));
    decls.emplace_back("int64_t n");
    for (const auto& tp : shape_trailing_params(shape_)) {
      spec.trailing.push_back(tp.tag);
      decls.emplace_back(tp.decl);
    }

    std::ostringstream src;
    src << kPrelude;
    src << "\nextern \"C\" __global__\n"
        << "void " << spec.entry_symbol << "(\n";
    if (params_.tellp() > 0) src << params_.str() << ",\n";
    for (std::size_t i = 0; i < decls.size(); ++i) {
      src << "    " << decls[i] << (i + 1 < decls.size() ? ",\n" : ")\n");
    }
    src << "{\n"
        << "    constexpr int32_t CHUNK = " << ::codegen::kChunkSize << ";\n"
        << "    const int32_t chunk_id = static_cast<int32_t>(blockIdx.x);\n"
        << "    const int32_t tid      = static_cast<int32_t>(threadIdx.x);\n"
        << "    const int64_t chunk_start = static_cast<int64_t>(chunk_id) *\n"
        << "                                static_cast<int64_t>(CHUNK);\n"
        << "    const int32_t len = static_cast<int32_t>(\n"
        << "        (n - chunk_start) < static_cast<int64_t>(CHUNK)\n"
        << "            ? (n - chunk_start) : static_cast<int64_t>(CHUNK));\n"
        << "    if (len <= 0) return;\n"
        << "    extern __shared__ __align__(16) unsigned char workspace[];\n"
        << "    (void)workspace;\n"
        << "\n"
        << body_.str() << "}\n";

    spec.source       = src.str();
    spec.buffers      = std::move(buffers_);
    spec.block_x      = tbs_;
    spec.shared_bytes = static_cast<int>(sm_.peak_bytes());
    spec.note =
      "decode-walker-rendered; dynamic-shared workspace = " + std::to_string(sm_.peak_bytes()) +
      " bytes";
    return spec;
  }

  static constexpr const char* kPrelude = R"src(
#include "codegen/decode/rle_block.cuh"
#include <cub/block/block_scan.cuh>
#include <cub/block/block_exchange.cuh>
#include <cuda/std/type_traits>

namespace {

// 3-word gather: extract element `idx` from a bit-packed stream.
// Loads three consecutive uint32 words unconditionally — no
// loop, no data dependency between the loads, compiler pipelines
// all three.  Sufficient for any bit width in [1,64].
// (bit_in==0 → shift_l==32; uint64<<32 is well-defined in C++20.)
__device__ __forceinline__ uint64_t simpatico_bitunpack_one(
    const uint32_t* __restrict__ packed, int bits, int32_t idx) {
    const uint64_t bp      = static_cast<uint64_t>(static_cast<uint32_t>(idx))
                           * static_cast<uint32_t>(bits);
    const int32_t  word_in = static_cast<int32_t>(bp >> 5);
    const int32_t  bit_in  = static_cast<int32_t>(bp & 31);
    const int32_t  shift_l = 32 - bit_in;
    const uint64_t w0 = packed[word_in    ];
    const uint64_t w1 = packed[word_in + 1];
    const uint64_t w2 = packed[word_in + 2];
    const uint64_t lo = ((w0 >> bit_in) | (w1 << shift_l)) & 0xFFFFFFFFULL;
    const uint64_t hi = ((w1 >> bit_in) | (w2 << shift_l)) & 0xFFFFFFFFULL;
    const uint64_t stitched = (hi << 32) | lo;
    const uint64_t mask = (bits == 64) ? ~uint64_t{0} : ((uint64_t{1} << bits) - 1);
    return stitched & mask;
}

// Random-access bitpack read at per-chunk position `idx`, with the
// constant-chunk (bits==0) short-circuit returning the chunk minimum.
template <class T>
__device__ __forceinline__ T simpatico_bp_at(const uint32_t* packed_base,
                                          int32_t bits, T minv, int32_t idx) {
    using U = typename ::cuda::std::make_unsigned<T>::type;
    if (bits == 0) return minv;
    const uint64_t v = simpatico_bitunpack_one(packed_base, bits, idx);
    return static_cast<T>(static_cast<U>(minv) + static_cast<U>(v));
}

}  // namespace
)src";
};

// =====================================================================
// emit_producer — dispatch.
// =====================================================================
void Walker::emit_producer(const ::codegen::jit::FusedTree& node,
                           const std::string& dst,
                           const std::string& len,
                           const std::string& elem_type)
{
  switch (node.op) {
    case ::codegen::OpKind::Bitpack: emit_bitpack_producer(node, dst, len, elem_type); return;
    case ::codegen::OpKind::Delta: emit_delta_producer(node, dst, len, elem_type); return;
    case ::codegen::OpKind::Rle: emit_rle_producer(node, dst, len, elem_type); return;
    case ::codegen::OpKind::For: emit_for_producer(node, dst, len, elem_type); return;
    case ::codegen::OpKind::Raw: emit_raw_producer(node, dst, len, elem_type); return;
    case ::codegen::OpKind::Zigzag: emit_zigzag_producer(node, dst, len, elem_type); return;
    default:
      throw RenderError(std::string("decode render: invalid op '") +
                        ::codegen::jit::op_kind_name(node.op) + "'");
  }
}

// =====================================================================
// Bitpack — closed-form value source + scalar-load prelude.
// =====================================================================
ValueSource Walker::bitpack_value_source(const ::codegen::jit::FusedTree& node,
                                         const std::string& elem_type)
{
  const std::size_t esize = dtype_elem_size(elem_type);
  if (esize == 0) {
    throw RenderError("decode render: Bitpack op-local dtype '" + elem_type + "' not supported");
  }
  const std::int32_t id   = id_of(node);
  const std::string idstr = std::to_string(id);

  const std::string p_min  = "chunk_min_" + idstr;
  const std::string p_bits = "chunk_bits_" + idstr;
  const std::string p_pkd  = "packed_" + idstr;
  const std::string v_min  = "bpmin_" + idstr;
  const std::string v_bits = "bpbits_" + idstr;
  const std::string v_base = "bpbase_" + idstr;

  // Kernel params (order == buffers order == launcher arg order).
  add_param("const " + elem_type + "* __restrict__", p_min);
  add_param("const uint8_t* __restrict__", p_bits);
  add_param("const uint32_t* __restrict__", p_pkd);
  add_buffer(id, "chunk_min", esize);
  add_buffer(id, "chunk_bits", sizeof(std::uint8_t));
  add_buffer(id, "packed", sizeof(std::uint32_t));

  // Per-chunk scalar prelude (loaded once per block).
  body_ << "    // --- node " << id << ": Bitpack (" << elem_type << ") value source ---\n"
        << "    const int32_t " << v_bits << " = static_cast<int32_t>(" << p_bits
        << "[chunk_id]);\n"
        << "    const " << elem_type << " " << v_min << " = " << p_min << "[chunk_id];\n";

  // Decode reads the Compact per-chunk ``bp_offsets`` layout; every stored
  // bitpack rep is dense (the fused encode path compacts in place).
  {
    const std::string p_off = "bp_offsets_" + idstr;
    add_param("const int32_t* __restrict__", p_off);
    add_buffer(id, "bp_offsets", sizeof(std::int32_t));
    body_ << "    const int32_t " << v_base << " = " << p_off << "[chunk_id];\n";
  }

  ValueSource vs;
  vs.elem_type = elem_type;
  vs.read_expr =
    "simpatico_bp_at(" + p_pkd + " + " + v_base + ", " + v_bits + ", " + v_min + ", (__POS__))";
  return vs;
}

void Walker::emit_bitpack_producer(const ::codegen::jit::FusedTree& node,
                                   const std::string& dst,
                                   const std::string& len,
                                   const std::string& elem_type)
{
  if (!node.children.empty()) { throw RenderError("decode render: Bitpack must be a leaf"); }
  ValueSource vs          = bitpack_value_source(node, elem_type);
  const std::string idstr = std::to_string(id_of(node));
  // Coalesced strided write: thread t handles positions t, t+128, ...
  body_ << "    // --- node " << id_of(node) << ": Bitpack producer ---\n"
        << "    for (int32_t i = tid; i < (" << len << "); i += " << tbs_ << ") {\n"
        << "        (" << dst << ")[i] = " << at_pos(vs.read_expr, "i") << ";\n"
        << "    }\n";
}

// =====================================================================
// Mask variants — the ballot and mask-walk epilogues.
//
// Iteration mapping: i = j*128 + tid, so iteration j of warp w covers 32
// CONSECUTIVE rows [chunk_start + j*128 + w*32, +32) — one __ballot_sync
// per warp-iteration yields exactly one selection-mask uint32 (row r ->
// bit r%32 of word r/32), stored coalesced.  Mask layout is 32 words per
// 1024-row chunk (mask base = chunk_start/32); the last chunk's tail
// bits/words are written as zero, which the CNT/combine wave relies on.
// =====================================================================

// Shared mask-consuming stage (warp 0): load the chunk's 32 selection-mask
// words into shared memory, compute the exclusive per-word survivor prefix +
// chunk total via a warp scan of popcounts, and early-return zero-survivor
// chunks.  Leaves sel_words / sel_base / sel_total live for the caller's
// store loop.
void Walker::emit_selection_stage()
{
  body_ << "    __shared__ uint32_t sel_words[CHUNK / 32];\n"
        << "    __shared__ int32_t sel_base[CHUNK / 32];  // exclusive survivor prefix per word\n"
        << "    __shared__ int32_t sel_total;             // survivors in this chunk\n"
        << "    if (tid < CHUNK / 32) {\n"
        << "        const uint32_t w = sel_mask[(chunk_start >> 5) + tid];\n"
        << "        sel_words[tid] = w;\n"
        << "        const int32_t pc = __popc(w);\n"
        << "        int32_t x = pc;\n"
        << "        #pragma unroll\n"
        << "        for (int32_t o = 1; o < 32; o <<= 1) {\n"
        << "            const int32_t y = __shfl_up_sync(0xFFFFFFFFu, x, o);\n"
        << "            if (tid >= o) x += y;\n"
        << "        }\n"
        << "        sel_base[tid] = x - pc;\n"
        << "        if (tid == 31) sel_total = x;\n"
        << "    }\n"
        << "    __syncthreads();\n"
        << "    if (sel_total == 0) return;  // zero-survivor chunk: nothing to write\n";
}

// Emitted C++ expression for a row's survivor rank given `w` (its mask word,
// already loaded from sel_words) and `pos` (the row's in-chunk position).
static std::string rank_expr(const std::string& pos)
{
  return "sel_base[" + pos + " >> 5] + __popc(w & ((1u << (" + pos + " & 31)) - 1u))";
}

// The survivor loop shared by every mask-consuming variant (the value store,
// the dictionary gather, the str_split offsets meta).  Requires emit_selection_stage() to have run.
//
// Walks the chunk's rows, tests the mask bit, binds `rank` (the row's compacted
// slot within the chunk) and emits `sink` — the ONE line that differs between
// those variants.  In scope for `sink`: `i` (in-chunk row), `rank`, `out_base`
// (the chunk's base in the compacted output), `w` (the row's mask word).
void Walker::emit_mask_survivor_loop(const std::string& sink)
{
  body_ << "    const int64_t out_base = static_cast<int64_t>(chunk_offsets[chunk_id]);\n"
        << "    #pragma unroll\n"
        << "    for (int32_t j = 0; j < CHUNK / " << tbs_ << "; ++j) {\n"
        << "        const int32_t i = j * " << tbs_ << " + tid;\n"
        << "        const uint32_t w = sel_words[i >> 5];\n"
        << "        if ((w >> (i & 31)) & 1u) {\n"
        << "            const int32_t rank = " << rank_expr("i") << ";\n"
        << sink << "        }\n"
        << "    }\n";
}

// =====================================================================
// Delta mask_consume — the same reconstruction as the plain Delta
// producer, storing only survivors.  The prefix sum is inherently sequential
// within a chunk, so ALL rows are still decoded; the mask gates only the
// global STORE (row idx+1 lands at its compacted survivor slot).  Saves the
// full-width column write + the downstream gather, not the unpack compute.
// =====================================================================
void Walker::emit_delta_mask_consume(const ::codegen::jit::FusedTree& node)
{
  if (node.op != ::codegen::OpKind::Delta || node.children.size() != 1) {
    throw RenderError(
      "decode render: mask_consume Delta root must have exactly one "
      "'differences' child");
  }
  if (node.children.find("differences") == node.children.end()) {
    throw RenderError("decode render: mask_consume Delta root missing 'differences' child");
  }

  body_ << "    // --- node " << id_of(node) << ": Delta masked decode -> compacted output ---\n";
  emit_selection_stage();
  // out_base is declared by the masked store inside the producer's block.
  emit_delta_producer(node, "(out + out_base)", "len", dtype_, DeltaStore::mask_compact);
}

// =====================================================================
// Dictionary-code masked gather — the Bitpack leaf holds the dictionary
// CODES; for survivor rows only, decode the code and copy the key's bytes
// from a constant-width, null-free key pool straight into the compacted
// chars output.  Skips both the full-width INT32 code materialisation and
// the separate key-gather pass.  Offsets are analytic (j * key_width) and
// built by the caller (see masked_launch.hpp).
// =====================================================================
void Walker::emit_bitpack_mask_dict_gather(const ::codegen::jit::FusedTree& node)
{
  if (node.op != ::codegen::OpKind::Bitpack || !node.children.empty()) {
    throw RenderError(
      "decode render: mask_dict_gather variant requires a Bitpack code leaf root "
      "(got '" +
      std::string(::codegen::jit::op_kind_name(node.op)) + "')");
  }
  body_ << "    // --- node " << id_of(node) << ": Bitpack code masked dictionary gather ---\n"
        << "    (void)len;  // mask tail bits are zero, so selected rows are always < len\n";
  emit_selection_stage();

  ValueSource vs = bitpack_value_source(node, dtype_);
  emit_mask_survivor_loop("            const int64_t code = static_cast<int64_t>(" +
                          at_pos(vs.read_expr, "i") +
                          ");\n"
                          "            const char* k = keys_chars + code * key_width;\n"
                          "            char* o = out + (out_base + rank) * key_width;\n"
                          "            for (int32_t b = 0; b < key_width; ++b) o[b] = k[b];\n");
}

// =====================================================================
// Index-list-consuming decode — the low-selectivity sibling of the mask walk.
// Block c reads its slice of the ascending GLOBAL row-index list
// (row_indices[chunk_offsets[c] .. chunk_offsets[c+1])) and random-access
// decodes only those rows: out slot chunk_offsets[c]+k gets the value of
// row row_indices[chunk_offsets[c]+k].  No mask staging, no ballot — the
// per-block loop runs `cnt` iterations instead of 8 full 128-wide strips,
// so runtime scales with survivors (microbench: 0.30 vs the mask walk's
// 0.78 ms/payload at 1.9% selectivity; the mask walk wins again above the
// ~15% crossover
// — the caller picks from the survivor count).  Delta roots cannot
// row-skip and are rejected.
// =====================================================================
void Walker::emit_bitpack_index_consume(const ::codegen::jit::FusedTree& node)
{
  if (node.op != ::codegen::OpKind::Bitpack || !node.children.empty()) {
    throw RenderError(
      "decode render: index_consume variant requires a Bitpack leaf root "
      "(got '" +
      std::string(::codegen::jit::op_kind_name(node.op)) + "')");
  }
  body_ << "    // --- node " << id_of(node)
        << ": Bitpack index-list decode -> compacted output ---\n"
        << "    (void)len;  // listed rows are < n by construction (mask tail bits were zero)\n"
        << "    const int64_t out_base = static_cast<int64_t>(chunk_offsets[chunk_id]);\n"
        << "    const int32_t cnt = static_cast<int32_t>(\n"
        << "        static_cast<int64_t>(chunk_offsets[chunk_id + 1]) - out_base);\n"
        << "    if (cnt == 0) return;  // zero-survivor chunk: nothing to decode\n";

  // Per-chunk scalar prelude after the early return, then the survivor loop.
  ValueSource vs = bitpack_value_source(node, dtype_);
  body_ << "    const int32_t* idxs = row_indices + out_base;\n"
        << "    for (int32_t k = tid; k < cnt; k += " << tbs_ << ") {\n"
        << "        const int32_t i = static_cast<int32_t>(idxs[k] - chunk_start);  // in-chunk "
           "pos\n"
        << "        (out + out_base)[k] = " << at_pos(vs.read_expr, "i") << ";\n"
        << "    }\n";
}

// =====================================================================
// Generic mask_consume — the compositional seam.  Any root the existing
// value_source machinery supports (FOR/Zigzag transformers, RLE cascades,
// ...) is staged in full for the chunk (closed-form or shared-mem slab via
// the plain emitters) and masked-stored from there.  The tuned Bitpack
// (closed-form, no staging) and Delta (register-striped scan) roots keep
// their dedicated emitters; everything else routes here, so new depth
// combinations don't multiply hand-written variants.
// =====================================================================
void Walker::emit_generic_mask_consume(const ::codegen::jit::FusedTree& node)
{
  body_ << "    // --- node " << id_of(node) << ": " << ::codegen::jit::op_kind_name(node.op)
        << " masked decode -> compacted output (generic) ---\n"
        << "    (void)len;  // mask tail bits are zero, so selected rows are always < len\n";
  emit_selection_stage();

  const auto mark = sm_.mark();
  ValueSource vs  = value_source(node, dtype_, "len");
  emit_mask_survivor_loop("            (out + out_base)[rank] = " + at_pos(vs.read_expr, "i") +
                          ";\n");
  sm_.release_to(mark);
}

// =====================================================================
// str_split meta (phase 1) — masked survivor {source char offset,
// length} extraction from a string column's OFFSETS subtree.  The offsets
// cascade is reconstructed IN FULL per chunk through value_source (depth
// is fine: bitpack closed-form, delta->rle->bitpack slab, ...); only the
// per-survivor metadata is written.  Rows live in [0, n-1) — the offsets
// column has one trailing element — and row r needs offsets[r] and
// offsets[r+1]; the latter crosses into the next chunk for the chunk's
// last row, so the root must expose its next chunk's FIRST element as a
// per-chunk scalar peek:
//   * Bitpack leaf: bp_at(packed + bp_offsets[c+1], bits[c+1], min[c+1], 0)
//   * Delta root:   delta_first[c+1] (the stored per-chunk anchor)
// Other roots are rejected.  Raw chars are gathered in phase 2
// (launch_masked_char_copy) after the caller scans the lengths.
// =====================================================================
void Walker::emit_str_split_meta(const ::codegen::jit::FusedTree& node)
{
  const bool bitpack_root = node.op == ::codegen::OpKind::Bitpack && node.children.empty();
  const bool delta_root   = node.op == ::codegen::OpKind::Delta;
  if (!bitpack_root && !delta_root) {
    throw RenderError(
      "decode render: str_split_meta requires a Bitpack- or Delta-rooted "
      "offsets subtree (got '" +
      std::string(::codegen::jit::op_kind_name(node.op)) + "')");
  }

  body_ << "    // --- node " << id_of(node) << ": str_split offsets masked survivor meta ---\n"
        << "    const int64_t n_rows = n - 1;  // offsets count = string rows + 1\n"
        << "    if (chunk_start >= n_rows) return;  // offsets-tail chunk: no rows\n";
  emit_selection_stage();

  const auto mark = sm_.mark();
  ValueSource vs  = value_source(node, dtype_, "len");

  // Next chunk's first offset (only read when the chunk is full and its last
  // row survives; the guard also proves chunk_id+1 is in range then).
  const std::string idstr = std::to_string(id_of(node));
  body_ << "    " << dtype_ << " next0 = " << dtype_ << "{0};\n"
        << "    if ((static_cast<int64_t>(chunk_id) + 1) * static_cast<int64_t>(CHUNK) < n) {\n";
  if (bitpack_root) {
    body_ << "        next0 = simpatico_bp_at(packed_" << idstr << " + bp_offsets_" << idstr
          << "[chunk_id + 1],\n"
          << "                                static_cast<int32_t>(chunk_bits_" << idstr
          << "[chunk_id + 1]),\n"
          << "                                chunk_min_" << idstr << "[chunk_id + 1], 0);\n";
  } else {
    body_ << "        next0 = delta_first_" << idstr << "[chunk_id + 1];\n";
  }
  body_ << "    }\n";

  emit_mask_survivor_loop(
    "            const int64_t off_r = static_cast<int64_t>(" + at_pos(vs.read_expr, "i") +
    ");\n"
    "            const int64_t off_r1 = (i + 1 < len)\n"
    "                ? static_cast<int64_t>(" +
    at_pos(vs.read_expr, "(i + 1)") +
    ")\n"
    "                : static_cast<int64_t>(next0);\n"
    "            (out + out_base)[rank] = off_r;\n"
    "            len_out[out_base + rank] = static_cast<int32_t>(off_r1 - off_r);\n");
  sm_.release_to(mark);
}

// =====================================================================
// Generic mask_out — the compositional ballot for
// non-closed-form roots, primarily delta->bitpack shapes: the
// decode-evaluable form of min-max dynamic join filters.  Delta's prefix
// sum is sequential within a chunk, so the chunk is reconstructed IN FULL
// via value_source (staged to a shared-mem slab by the existing plain
// emitters), then the range predicate is balloted from the staged values.
// Reading i-aligned positions keeps the one-word-per-32-consecutive-rows
// mask layout intact (ballot lanes == mask bits).  No column output; same
// trailing params as the bitpack ballot (sel_mask in the out slot, pred_lo/
// pred_hi widened-int64 kernel parameters).
// =====================================================================
void Walker::emit_generic_mask_out(const ::codegen::jit::FusedTree& node)
{
  body_ << "    // --- node " << id_of(node) << ": " << ::codegen::jit::op_kind_name(node.op)
        << " fused range predicate -> selection mask (generic) ---\n";
  const auto mark = sm_.mark();
  ValueSource vs  = value_source(node, dtype_, "len");
  body_ << "    uint32_t* mask_words = sel_mask + (chunk_start >> 5);\n"
        << "    #pragma unroll\n"
        << "    for (int32_t j = 0; j < CHUNK / " << tbs_ << "; ++j) {\n"
        << "        const int32_t i = j * " << tbs_ << " + tid;\n"
        << "        bool pass = false;\n"
        << "        if (i < len) {\n"
        << "            const int64_t v = static_cast<int64_t>(" << at_pos(vs.read_expr, "i")
        << ");\n"
        << "            pass = (v >= pred_lo) && (v <= pred_hi);\n"
        << "        }\n"
        << "        const uint32_t ballot = __ballot_sync(0xFFFFFFFFu, pass);\n"
        << "        if ((tid & 31) == 0) mask_words[i >> 5] = ballot;\n"
        << "    }\n";
  sm_.release_to(mark);
}

// =====================================================================
// Delta — inline-fused block scan (no intermediate diff buffer).
// =====================================================================
void Walker::emit_delta_producer(const ::codegen::jit::FusedTree& node,
                                 const std::string& dst,
                                 const std::string& len,
                                 const std::string& elem_type,
                                 DeltaStore store)
{
  const bool masked = store == DeltaStore::mask_compact;
  if (node.children.size() != 1) {
    throw RenderError("decode render: Delta must have exactly one 'differences' child");
  }
  auto vit = node.children.find("differences");
  if (vit == node.children.end()) {
    throw RenderError("decode render: Delta missing 'differences' child");
  }
  const std::size_t esize = dtype_elem_size(elem_type);
  if (esize == 0) {
    throw RenderError("decode render: Delta op-local dtype '" + elem_type + "' not supported");
  }

  const std::int32_t id     = id_of(node);
  const std::string idstr   = std::to_string(id);
  const std::string p_first = "delta_first_" + idstr;
  add_param("const " + elem_type + "* __restrict__", p_first);
  add_buffer(id, "delta_first", esize);

  // Diff stream length = len - 1.  The child's value source reads diffs
  // at contiguous positions; for a closed-form (Bitpack) child this
  // fuses with no scratch, otherwise the child materialises to a slab.
  const std::string dlen  = "dlen_" + idstr;
  const std::string ndiff = "ndiff_" + idstr;
  const std::string items = "ditems_" + idstr;
  const std::string first = "dfirst_" + idstr;

  // The diff stream is read in STRIPED order (thread tid owns global
  // positions {j*TB + tid}) so the child's global loads (and the final
  // output writes) are coalesced across the warp — matching the
  // bitpack-alone access pattern.  cub::BlockExchange transposes
  // striped<->blocked so the InclusiveSum still sees true global order;
  // the exchange/scan temp storage is unioned (used sequentially).
  const std::string scan = "DScan_" + idstr;
  const std::string exch = "DExch_" + idstr;
  const std::string tmp  = "dtmp_" + idstr;
  if (!masked) {
    body_ << "    // --- node " << id << ": Delta (inline-fused scan, coalesced, " << elem_type
          << ") ---\n";
  }
  body_ << "    {\n"
        << "        constexpr int32_t TBS = " << tbs_ << ";\n"
        << "        constexpr int32_t IPT = CHUNK / TBS;\n"
        << "        const int32_t " << dlen << " = static_cast<int32_t>(" << len << ");\n"
        << "        const int32_t " << ndiff << " = (" << dlen << " > 0) ? (" << dlen
        << " - 1) : 0;\n";

  // Child value source (closed-form expr, or a materialised slab read).
  // Allocator mark: any slab the child needs is transient to this scan.
  const auto mark = sm_.mark();
  ValueSource child_vs =
    value_source(*vit->second, elem_type, "((" + len + ") > 0 ? ((" + len + ") - 1) : 0)");

  // Layout choice: striped (coalesced global access, needs BlockExchange) vs
  // blocked (no exchange, stride-IPT global access).
  //
  // Use STRIPED at root level (sm_.mark()==0, no outer smem pressure): the
  // 8 KB BlockExchange static smem is acceptable and coalescing matters.
  //
  // Use BLOCKED when nested inside an RLE (sm_.mark()>0): smem is already
  // occupied by the outer Rle's slabs; BlockExchange would push total >48 KB
  // and halve SM occupancy.  The diff stream is tiny (few inner runs × small
  // bit widths) so it stays L1-resident regardless of access pattern —
  // profiling confirmed DRAM<10%, bytes/sector=32 (L1-cached).
  //
  // The masked store is root-only (the mask is row-space), so it always takes
  // the striped branch — the nested/blocked case cannot arise there.
  const bool use_striped = masked || (sm_.mark() == 0);

  body_ << "        " << elem_type << " " << items << "[IPT];\n"
        << "        #pragma unroll\n"
        << "        for (int32_t j = 0; j < IPT; ++j) {\n";
  if (use_striped) {
    body_ << "            const int32_t idx = j * TBS + tid;  // striped\n";
  } else {
    body_ << "            const int32_t idx = tid * IPT + j;  // blocked\n";
  }
  body_ << "            " << items << "[j] = (idx < " << ndiff << ")\n"
        << "                ? static_cast<" << elem_type << ">("
        << at_pos(child_vs.read_expr, "idx") << ")\n"
        << "                : " << elem_type << "{0};\n"
        << "        }\n"
        << "        typedef cub::BlockScan<" << elem_type << ", TBS> " << scan << ";\n";

  if (use_striped) {
    // Striped path: transpose → scan → transpose back.  Full coalescing,
    // but BlockExchange<T,TBS,IPT> costs ~sizeof(T)*TBS*IPT static smem.
    body_ << "        typedef cub::BlockExchange<" << elem_type << ", TBS, IPT> " << exch << ";\n"
          << "        __shared__ union {\n"
          << "            typename " << exch << "::TempStorage ex;\n"
          << "            typename " << scan << "::TempStorage sc;\n"
          << "        } " << tmp << ";\n"
          << "        " << exch << "(" << tmp << ".ex).StripedToBlocked(" << items << ", " << items
          << ");\n"
          << "        __syncthreads();\n"
          << "        " << scan << "(" << tmp << ".sc).InclusiveSum(" << items << ", " << items
          << ");\n"
          << "        __syncthreads();\n"
          << "        " << exch << "(" << tmp << ".ex).BlockedToStriped(" << items << ", " << items
          << ");\n"
          << "        __syncthreads();\n";
  } else {
    // Blocked path: scan directly, no exchange.  BlockScan TempStorage is
    // tiny (~24 B for warp-scan algorithm); saves 8 KB static smem vs the
    // BlockExchange union, raising SM occupancy from 25% → ~35–50%.
    body_ << "        __shared__ typename " << scan << "::TempStorage " << tmp << ";\n"
          << "        " << scan << "(" << tmp << ").InclusiveSum(" << items << ", " << items
          << ");\n"
          << "        __syncthreads();\n";
  }

  // Reconstruction adds "first" (an arbitrary stored value) to a cumulative
  // diff sum; the sum can exceed the signed range (see emit_delta on the
  // encode side), so the add is done in the unsigned counterpart type to
  // avoid relying on signed-overflow wraparound (UB).
  const std::string dutype = unsigned_counterpart(esize);
  body_ << "        const " << elem_type << " " << first << " = " << p_first << "[chunk_id];\n";

  if (masked) {
    // Row 0 = first, row idx+1 = first + prefix[idx]; only survivor rows are
    // written, at out_base + rank. Row 0's rank is always 0 when its bit is
    // set (no lower bits in word 0).
    body_ << "        const int64_t out_base = static_cast<int64_t>(chunk_offsets[chunk_id]);\n"
          << "        if (tid == 0 && " << dlen << " > 0 && (sel_words[0] & 1u)) " << dst
          << "[0] = " << first << ";\n"
          << "        #pragma unroll\n"
          << "        for (int32_t j = 0; j < IPT; ++j) {\n"
          << "            const int32_t idx = j * TBS + tid;  // striped global pos\n"
          << "            const int32_t row = idx + 1;\n"
          << "            if (row < " << dlen << ") {\n"
          << "                const uint32_t w = sel_words[row >> 5];\n"
          << "                if ((w >> (row & 31)) & 1u) {\n"
          << "                    const int32_t rank = " << rank_expr("row") << ";\n"
          << "                    " << dst << "[rank] = static_cast<" << elem_type
          << ">(static_cast<" << dutype << ">(" << first << ") + static_cast<" << dutype << ">("
          << items << "[j]));\n"
          << "                }\n"
          << "            }\n"
          << "        }\n"
          << "    }\n";
    sm_.release_to(mark);
    return;
  }

  body_ << "        if (tid == 0 && " << dlen << " > 0) (" << dst << ")[0] = " << first << ";\n"
        << "        #pragma unroll\n"
        << "        for (int32_t j = 0; j < IPT; ++j) {\n";
  if (use_striped) {
    body_ << "            const int32_t idx = j * TBS + tid;  // striped global pos\n";
  } else {
    body_ << "            const int32_t idx = tid * IPT + j;  // blocked global pos\n";
  }
  body_ << "            if (idx + 1 < " << dlen << ") (" << dst << ")[idx + 1] = static_cast<"
        << elem_type << ">(static_cast<" << dutype << ">(" << first << ") + static_cast<" << dutype
        << ">(" << items << "[j]));\n"
        << "        }\n"
        << "        __syncthreads();\n"
        << "    }\n";

  sm_.release_to(mark);  // free any child slab now the scan has consumed it
}

// =====================================================================
// Rle — STAGE BOUNDARY.
// When the values child is a leaf (Bitpack or Raw), use a closed-form
// functor instead of materialising sh_values into shared memory, saving
// sizeof(T)*kChunkSize (8 KB for int64) of dynamic smem:
//   rle_runsbp:  16 KB → 8 KB   (~14 blocks/SM → ~25 blocks/SM)
//   nvcomp_def:  32 KB → 24 KB  (25% → ~38% occupancy)
// When both values and runs children are leaf Bitpacks AND we are inside a
// nested smem context (mark_pre > 0), skip sh_counts too (saves 4 KB more).
// =====================================================================
void Walker::emit_rle_producer(const ::codegen::jit::FusedTree& node,
                               const std::string& dst,
                               const std::string& len,
                               const std::string& elem_type)
{
  if (node.children.size() != 2) {
    throw RenderError("decode render: Rle must have 'runs' and 'values' children");
  }
  auto rit = node.children.find("runs");
  auto vit = node.children.find("values");
  if (rit == node.children.end() || vit == node.children.end()) {
    throw RenderError("decode render: Rle missing 'runs' or 'values' child");
  }
  const std::size_t esize = dtype_elem_size(elem_type);
  if (esize == 0) {
    throw RenderError("decode render: Rle op-local dtype '" + elem_type + "' not supported");
  }

  const std::int32_t id   = id_of(node);
  const std::string idstr = std::to_string(id);
  const std::string p_off = "rle_runs_offsets_" + idstr;
  add_param("const int32_t* __restrict__", p_off);
  add_buffer(id, "rle_runs_offsets", sizeof(std::int32_t));

  const std::string sh_values = "sh_values_" + idstr;
  const std::string sh_counts = "sh_counts_" + idstr;
  const std::string sh_starts = "sh_starts_" + idstr;
  const std::string nruns     = "nruns_" + idstr;
  const std::string vread     = "vread_" + idstr;

  const bool values_is_leaf =
    (vit->second->op == ::codegen::OpKind::Bitpack && vit->second->children.empty()) ||
    (vit->second->op == ::codegen::OpKind::Raw && vit->second->children.empty());

  const auto mark_pre = sm_.mark();

  body_ << "    // --- node " << id << ": Rle (stage boundary, " << elem_type
        << " values / int32_t runs) ---\n"
        << "    const int32_t " << nruns << " = " << p_off << "[chunk_id + 1] - " << p_off
        << "[chunk_id];\n";

  if (values_is_leaf) {
    // Closed-form path: get a direct read expression, no sh_values slab.
    ValueSource vs = value_source(*vit->second, elem_type, nruns);

    // When the runs child is also a Bitpack leaf AND we are inside a nested
    // smem context (mark_pre > 0), use a functor for counts too (saves sh_counts).
    // At root level (mark_pre == 0) the inline bitpack decode in the scatter
    // loop costs more than the prefetch-to-smem path, causing 12–16% regression.
    const bool counts_is_bp_leaf =
      (rit->second->op == ::codegen::OpKind::Bitpack && rit->second->children.empty()) &&
      (mark_pre > 0);

    if (!counts_is_bp_leaf) {
      const std::size_t off_counts = sm_.alloc(sizeof(std::int32_t), ::codegen::kChunkSize);
      body_ << "    int32_t* " << sh_counts << " = reinterpret_cast<int32_t*>(workspace + "
            << off_counts << ");\n";
    }
    const std::size_t off_starts = sm_.alloc(sizeof(std::int32_t), ::codegen::kChunkSize);
    body_ << "    int32_t* " << sh_starts << " = reinterpret_cast<int32_t*>(workspace + "
          << off_starts << ");\n";

    // Identity fast path: nruns == len → plain copy using the closed-form expr.
    body_ << "    if (" << nruns << " == static_cast<int32_t>(" << len << ")) {\n"
          << "        for (int32_t i = tid; i < " << nruns << "; i += " << tbs_ << ")\n"
          << "            (" << dst << ")[i] = " << at_pos(vs.read_expr, "i") << ";\n"
          << "        __syncthreads();\n"
          << "    } else {\n";

    // Emit the values struct functor (no sh_values, no smem roundtrip).
    const std::string vstruct    = "VRead" + idstr;
    const std::string v_child_id = std::to_string(id_of(*vit->second));
    body_ << "        __syncthreads();\n";
    if (vit->second->op == ::codegen::OpKind::Bitpack) {
      body_ << "        struct " << vstruct << " {\n"
            << "            const uint32_t* packed; int32_t bits;\n"
            << "            " << elem_type << " min; int32_t base;\n"
            << "            __device__ __forceinline__ " << elem_type
            << " operator()(int32_t idx) const noexcept {\n"
            << "                return simpatico_bp_at(packed + base, bits, min, idx);\n"
            << "            }\n"
            << "        } " << vread << " { packed_" << v_child_id << ", bpbits_" << v_child_id
            << ", bpmin_" << v_child_id << ", bpbase_" << v_child_id << " };\n";
    } else {
      // Raw leaf
      body_ << "        struct " << vstruct << " {\n"
            << "            const " << elem_type << "* data; int32_t base;\n"
            << "            __device__ __forceinline__ " << elem_type
            << " operator()(int32_t idx) const noexcept {\n"
            << "                return data[base + idx];\n"
            << "            }\n"
            << "        } " << vread << " { raw_data_" << v_child_id << ", raw_base_" << v_child_id
            << " };\n";
    }

    if (counts_is_bp_leaf) {
      // Runs child also a leaf — functor for counts, no sh_counts.
      const std::string cread      = "cread_" + idstr;
      const std::string cstruct    = "CntRead" + idstr;
      const std::string c_child_id = std::to_string(id_of(*rit->second));
      bitpack_value_source(*rit->second, "int32_t");  // emits kernel params
      body_ << "        struct " << cstruct << " {\n"
            << "            const uint32_t* packed; int32_t bits;\n"
            << "            int32_t min; int32_t base;\n"
            << "            __device__ __forceinline__ int32_t operator()(int32_t idx) const "
               "noexcept {\n"
            << "                return static_cast<int32_t>(simpatico_bp_at(packed + base, bits, "
               "min, idx));\n"
            << "            }\n"
            << "        } " << cread << " { packed_" << c_child_id << ", bpbits_" << c_child_id
            << ", bpmin_" << c_child_id << ", bpbase_" << c_child_id << " };\n"
            << "        ::codegen::block_rle_decompress_fv<" << elem_type << ", "
            << ::codegen::kChunkSize << ", " << tbs_ << ">(\n"
            << "            " << vread << ", " << cread << ",\n"
            << "            " << sh_starts << ",\n"
            << "            " << nruns << ", static_cast<int32_t>(" << len << "), (" << dst
            << "));\n";
    } else {
      // Runs child needs materialisation; use SmemCountsReader functor.
      emit_producer(*rit->second, sh_counts, nruns, "int32_t");
      body_ << "        __syncthreads();\n"
            << "        ::codegen::block_rle_decompress_fv<" << elem_type << ", "
            << ::codegen::kChunkSize << ", " << tbs_ << ">(\n"
            << "            " << vread << ",\n"
            << "            ::codegen::detail::SmemCountsReader{" << sh_counts << "},\n"
            << "            " << sh_starts << ",\n"
            << "            " << nruns << ", static_cast<int32_t>(" << len << "), (" << dst
            << "));\n";
    }
    body_ << "    }\n";
  } else {
    // Non-leaf values child (Delta or nested Rle subtree): materialise into
    // sh_values first, then use block_rle_decompress with pointer semantics.
    // sh_values must be allocated before the values child so the child can
    // write into it.  sh_counts and sh_starts are deferred until after the
    // values chain finishes, reusing those bytes (keeps the peak lower).
    const std::size_t off_values = sm_.alloc(esize, ::codegen::kChunkSize);

    body_ << "    " << elem_type << "* " << sh_values << " = reinterpret_cast<" << elem_type
          << "*>(workspace + " << off_values << ");\n";

    emit_producer(*vit->second, sh_values, nruns, elem_type);

    const std::size_t off_counts = sm_.alloc(sizeof(std::int32_t), ::codegen::kChunkSize);
    const std::size_t off_starts = sm_.alloc(sizeof(std::int32_t), ::codegen::kChunkSize);

    body_ << "    int32_t* " << sh_counts << " = reinterpret_cast<int32_t*>(workspace + "
          << off_counts << ");\n"
          << "    int32_t* " << sh_starts << " = reinterpret_cast<int32_t*>(workspace + "
          << off_starts << ");\n";

    body_ << "    __syncthreads();\n"
          << "    if (" << nruns << " == static_cast<int32_t>(" << len << ")) {\n"
          << "        for (int32_t i = tid; i < " << nruns << "; i += " << tbs_ << ")\n"
          << "            (" << dst << ")[i] = " << sh_values << "[i];\n"
          << "        __syncthreads();\n"
          << "    } else {\n";

    emit_producer(*rit->second, sh_counts, nruns, "int32_t");

    body_ << "        __syncthreads();\n"
          << "        ::codegen::block_rle_decompress<" << elem_type << ">(\n"
          << "            " << sh_values << ", " << sh_counts << ", " << sh_starts << ",\n"
          << "            " << nruns << ", static_cast<int32_t>(" << len << "), (" << dst << "));\n"
          << "    }\n";
  }

  sm_.release_to(mark_pre);
}

// =====================================================================
// FOR — semi-inline reverse transformer.
//
// Decode inverts the encode FOR step: given the compressed residuals
// (the `deltas` child) and the per-chunk reference stored in
// `references[chunk_id]`, reconstruct `original[i] = residual[i] + ref`.
//
// Two paths mirror the Delta producer's child handling:
//   * Closed-form child (Bitpack/Raw leaf): get a `value_source` expr
//     (no slab needed) and emit a strided loop that adds the reference.
//   * Producer child (Delta/Rle): materialise the child into a shared
//     slab first, then add the reference in a second strided loop.
// =====================================================================
void Walker::emit_for_producer(const ::codegen::jit::FusedTree& node,
                               const std::string& dst,
                               const std::string& len,
                               const std::string& elem_type)
{
  if (node.children.size() != 1) {
    throw RenderError(
      "decode render: FOR must have exactly one child named 'deltas' "
      "(got " +
      std::to_string(node.children.size()) + " children)");
  }
  auto dit = node.children.find("deltas");
  if (dit == node.children.end()) {
    throw RenderError("decode render: FOR missing 'deltas' child (got '" +
                      node.children.begin()->first + "' instead)");
  }
  const std::size_t esize = dtype_elem_size(elem_type);
  if (esize == 0) {
    throw RenderError("decode render: FOR op-local dtype '" + elem_type + "' not supported");
  }

  const std::int32_t id   = id_of(node);
  const std::string idstr = std::to_string(id);

  // Kernel parameter: references buffer (one entry per chunk).
  const std::string p_refs = "references_" + idstr;
  add_param("const " + elem_type + "* __restrict__", p_refs);
  add_buffer(id, "references", esize);

  const std::string v_ref = "for_ref_" + idstr;

  body_ << "    // --- node " << id << ": FOR (reverse, " << elem_type << ") ---\n"
        << "    const " << elem_type << " " << v_ref << " = " << p_refs << "[chunk_id];\n";

  const bool child_is_leaf =
    (dit->second->op == ::codegen::OpKind::Bitpack && dit->second->children.empty()) ||
    (dit->second->op == ::codegen::OpKind::Raw && dit->second->children.empty());

  // residual + reference can exceed the signed range (see emit_for on the
  // encode side); add in the unsigned counterpart type instead of relying
  // on signed-overflow wraparound (UB).
  const std::string futype = unsigned_counterpart(esize);

  if (child_is_leaf) {
    // Closed-form path: get an inline read expression for the residuals,
    // then add the per-chunk reference in a single strided loop.
    const auto mark      = sm_.mark();
    ValueSource child_vs = value_source(*dit->second, elem_type, len);
    body_ << "    for (int32_t i = tid; i < static_cast<int32_t>(" << len << "); i += " << tbs_
          << ") {\n"
          << "        (" << dst << ")[i] = static_cast<" << elem_type << ">(\n"
          << "            static_cast<" << futype << ">(" << at_pos(child_vs.read_expr, "i")
          << ") + static_cast<" << futype << ">(" << v_ref << "));\n"
          << "    }\n"
          << "    __syncthreads();\n";
    sm_.release_to(mark);
  } else {
    // Producer child (Delta/Rle): materialise residuals into a shared slab
    // first, then add the per-chunk reference in a second strided loop.
    const auto mark        = sm_.mark();
    const std::string slab = "for_slab_" + idstr;
    const std::size_t off  = sm_.alloc(esize, ::codegen::kChunkSize);

    body_ << "    " << elem_type << "* " << slab << " = reinterpret_cast<" << elem_type
          << "*>(workspace + " << off << ");\n";
    emit_producer(*dit->second, slab, len, elem_type);
    body_ << "    __syncthreads();\n"
          << "    for (int32_t i = tid; i < static_cast<int32_t>(" << len << "); i += " << tbs_
          << ") {\n"
          << "        (" << dst << ")[i] = static_cast<" << elem_type << ">(\n"
          << "            static_cast<" << futype << ">(" << slab << "[i]) + static_cast<" << futype
          << ">(" << v_ref << "));\n"
          << "    }\n"
          << "    __syncthreads();\n";
    sm_.release_to(mark);
  }
}

// =====================================================================
// raw_value_source_inline — closed-form expr for Raw leaf (no smem).
// =====================================================================
ValueSource Walker::raw_value_source_inline(const ::codegen::jit::FusedTree& node,
                                            const std::string& elem_type)
{
  if (!node.children.empty()) { throw RenderError("decode render: Raw must be a leaf"); }
  const std::size_t esize = dtype_elem_size(elem_type);
  if (esize == 0) {
    throw RenderError("decode render: raw_value_source_inline: unsupported dtype '" + elem_type +
                      "'");
  }
  const std::int32_t id    = id_of(node);
  const std::string idstr  = std::to_string(id);
  const std::string p_data = "raw_data_" + idstr;
  const std::string p_offs = "raw_offsets_" + idstr;
  const std::string v_base = "raw_base_" + idstr;

  add_param("const " + elem_type + "* __restrict__", p_data);
  add_param("const int32_t* __restrict__", p_offs);
  add_buffer(id, "data", esize);
  add_buffer(id, "offsets", sizeof(std::int32_t));

  body_ << "    // --- node " << id << ": Raw (inline, " << elem_type << ") ---\n"
        << "    const int32_t " << v_base << " = " << p_offs << "[chunk_id];\n";

  ValueSource vs;
  vs.elem_type = elem_type;
  vs.read_expr = p_data + "[" + v_base + " + (__POS__)]";
  return vs;
}

// =====================================================================
// zigzag_value_source_inline — closed-form inverse ZigZag for the leaf.
//
// Reads the stored `zigzag` channel at fixed stride (base chunk_start) and
// applies the inverse map inline:  n = (z >> 1) ^ -(z & 1)  computed in the
// unsigned domain to avoid signed-shift UB.  No shared memory — callers
// splice the expression directly (Bitpack-like).
// =====================================================================
ValueSource Walker::zigzag_value_source_inline(const ::codegen::jit::FusedTree& node,
                                               const std::string& elem_type)
{
  if (!node.children.empty()) { throw RenderError("decode render: Zigzag must be a leaf"); }
  const std::size_t esize = dtype_elem_size(elem_type);
  if (esize == 0) {
    throw RenderError("decode render: zigzag_value_source_inline: unsupported dtype '" + elem_type +
                      "'");
  }
  const std::string utype  = unsigned_counterpart(esize);
  const std::int32_t id    = id_of(node);
  const std::string idstr  = std::to_string(id);
  const std::string p_data = "zigzag_" + idstr;

  add_param("const " + elem_type + "* __restrict__", p_data);
  add_buffer(id, "zigzag", esize);

  // Stored element at this position (read once textually; compiler CSEs the
  // duplicate load).  base == chunk_start (chunk_id*CHUNK), known at entry.
  const std::string load = "static_cast<" + utype + ">(static_cast<" + exact_unsigned(esize) +
                           ">(" + p_data + "[chunk_start + (__POS__)]))";

  ValueSource vs;
  vs.elem_type = elem_type;
  vs.read_expr = "static_cast<" + elem_type + ">((" + load +
                 " >> 1) ^ "
                 "(static_cast<" +
                 utype + ">(0) - (" + load + " & static_cast<" + utype + ">(1))))";
  return vs;
}

// =====================================================================
// ZigZag — DUAL-MODE producer.
//
// Transformer mode (a fused op produced the ZigZag codes, e.g.
// `zigzag.zigzag -> bitpack`): reconstruct the child's values, then apply the
// inverse map  n = (z >> 1) ^ -(z & 1)  per element.  Two paths mirror FOR:
//   * Closed-form child (Bitpack/Raw leaf): splice its value_source expr.
//   * Producer child (Delta/Rle): materialise into a shared slab first.
// The code value is loaded into a local once so a non-trivial child read
// (e.g. simpatico_bp_at) is evaluated a single time per element.
//
// Leaf mode: reads the stored `zigzag` channel, applies the inverse map,
// writes `len` reconstructed elements into `dst`.
// =====================================================================
void Walker::emit_zigzag_producer(const ::codegen::jit::FusedTree& node,
                                  const std::string& dst,
                                  const std::string& len,
                                  const std::string& elem_type)
{
  if (!node.children.empty()) {
    if (node.children.size() != 1) {
      throw RenderError(
        "decode render: Zigzag transformer must have exactly one child "
        "named 'zigzag' (got " +
        std::to_string(node.children.size()) + " children)");
    }
    auto zit = node.children.find("zigzag");
    if (zit == node.children.end()) {
      throw RenderError("decode render: Zigzag missing 'zigzag' child (got '" +
                        node.children.begin()->first + "' instead)");
    }
    const std::size_t esize = dtype_elem_size(elem_type);
    if (esize == 0) {
      throw RenderError("decode render: Zigzag op-local dtype '" + elem_type + "' not supported");
    }
    const std::string utype = unsigned_counterpart(esize);
    const std::int32_t id   = id_of(node);
    const std::string idstr = std::to_string(id);

    const bool child_is_leaf =
      (zit->second->op == ::codegen::OpKind::Bitpack && zit->second->children.empty()) ||
      (zit->second->op == ::codegen::OpKind::Raw && zit->second->children.empty());

    body_ << "    // --- node " << id << ": Zigzag (reverse inline, " << elem_type << ") ---\n";

    // Emit the per-element inverse, given a textual code-value expression
    // `code_expr` (already position-substituted).
    auto emit_inverse_loop = [&](const std::string& code_expr) {
      // The exact-width unsigned inner cast prevents sign-extension garbage
      // above the element's width in the shift below (matters for int8_t).
      body_ << "    for (int32_t i = tid; i < static_cast<int32_t>(" << len << "); i += " << tbs_
            << ") {\n"
            << "        const " << utype << " _zc = static_cast<" << utype << ">(static_cast<"
            << exact_unsigned(esize) << ">(" << code_expr << "));\n"
            << "        (" << dst << ")[i] = static_cast<" << elem_type
            << ">((_zc >> 1) ^ (static_cast<" << utype << ">(0) - (_zc & static_cast<" << utype
            << ">(1))));\n"
            << "    }\n"
            << "    __syncthreads();\n";
    };

    if (child_is_leaf) {
      const auto mark = sm_.mark();
      ValueSource cvs = value_source(*zit->second, elem_type, len);
      emit_inverse_loop(at_pos(cvs.read_expr, "i"));
      sm_.release_to(mark);
    } else {
      const auto mark        = sm_.mark();
      const std::string slab = "zz_slab_" + idstr;
      const std::size_t off  = sm_.alloc(esize, ::codegen::kChunkSize);
      body_ << "    " << elem_type << "* " << slab << " = reinterpret_cast<" << elem_type
            << "*>(workspace + " << off << ");\n";
      emit_producer(*zit->second, slab, len, elem_type);
      body_ << "    __syncthreads();\n";
      emit_inverse_loop(slab + "[i]");
      sm_.release_to(mark);
    }
    return;
  }

  ValueSource vs = zigzag_value_source_inline(node, elem_type);
  body_ << "    // --- node " << id_of(node) << ": Zigzag producer (" << elem_type << ") ---\n"
        << "    for (int32_t i = tid; i < (" << len << "); i += " << tbs_ << ") {\n"
        << "        (" << dst << ")[i] = " << at_pos(vs.read_expr, "i") << ";\n"
        << "    }\n";
}

// =====================================================================
// value_source — closed-form expr (Bitpack / Raw / Zigzag leaf) or
// materialised-slab read (Delta / Rle / non-leaf).
// =====================================================================
ValueSource Walker::value_source(const ::codegen::jit::FusedTree& node,
                                 const std::string& elem_type,
                                 const std::string& len)
{
  if (node.op == ::codegen::OpKind::Bitpack) {
    if (!node.children.empty()) { throw RenderError("decode render: Bitpack must be a leaf"); }
    return bitpack_value_source(node, elem_type);
  }
  if (node.op == ::codegen::OpKind::Raw && node.children.empty()) {
    return raw_value_source_inline(node, elem_type);
  }
  if (node.op == ::codegen::OpKind::Zigzag && node.children.empty()) {
    return zigzag_value_source_inline(node, elem_type);
  }

  // Delta / Rle / non-leaf: materialise the subtree into a fresh shared slab,
  // leave it LIVE for the caller (the caller's allocator mark releases it).
  const std::size_t esize = dtype_elem_size(elem_type);
  if (esize == 0) {
    throw RenderError("decode render: value_source dtype '" + elem_type + "' not supported");
  }
  const std::int32_t id  = id_of(node);
  const std::string slab = "vbuf_" + std::to_string(id);
  const std::size_t off  = sm_.alloc(esize, ::codegen::kChunkSize);

  body_ << "    " << elem_type << "* " << slab << " = reinterpret_cast<" << elem_type
        << "*>(workspace + " << off << ");\n";
  emit_producer(node, slab, len, elem_type);
  body_ << "    __syncthreads();\n";

  ValueSource vs;
  vs.elem_type = elem_type;
  vs.read_expr = slab + "[(__POS__)]";
  return vs;
}

// =====================================================================
// Raw — passthrough values stored compactly via rle_runs_offsets.
//
// On the encode side `data[offsets[c] + r]` holds run `r` of chunk `c`.
// `offsets` is a copy of the parent Rle's `rle_runs_offsets` (exclusive
// prefix sum), so `offsets[chunk_id]` is the global start for this chunk.
// The producer just copies `nruns` values into the destination slab.
// =====================================================================
void Walker::emit_raw_producer(const ::codegen::jit::FusedTree& node,
                               const std::string& dst,
                               const std::string& len,
                               const std::string& elem_type)
{
  if (!node.children.empty()) { throw RenderError("decode render: Raw must be a leaf"); }
  const std::size_t esize = dtype_elem_size(elem_type);
  if (esize == 0) {
    throw RenderError("decode render: Raw op-local dtype '" + elem_type + "' not supported");
  }

  const std::int32_t id    = id_of(node);
  const std::string idstr  = std::to_string(id);
  const std::string p_data = "raw_data_" + idstr;
  const std::string p_offs = "raw_offsets_" + idstr;
  const std::string v_base = "raw_base_" + idstr;
  const std::string v_len  = "raw_len_" + idstr;

  add_param("const " + elem_type + "* __restrict__", p_data);
  add_param("const int32_t* __restrict__", p_offs);
  add_buffer(id, "data", esize);
  add_buffer(id, "offsets", sizeof(std::int32_t));

  body_ << "    // --- node " << id << ": Raw (compact passthrough, " << elem_type << ") ---\n"
        << "    {\n"
        << "        const int32_t " << v_base << " = " << p_offs << "[chunk_id];\n"
        << "        const int32_t " << v_len << " = static_cast<int32_t>(" << len << ");\n"
        << "        for (int32_t i = tid; i < " << v_len << "; i += " << tbs_ << ")\n"
        << "            (" << dst << ")[i] = " << p_data << "[" << v_base << " + i];\n"
        << "        __syncthreads();\n"
        << "    }\n";
}

}  // namespace

bool shape_is_supported(DecodeShape shape)
{
  // Shipped points of the product. The meaningful-but-unbuilt combinations —
  // index_list x dict_gather and index_list x offsets_meta, which would give
  // index-walk-speed dictionary and string decode below the crossover — belong here
  // the moment their emitters land; see DECODE_PUSHDOWN_PLAN.md section 7.
  return shape == kShapePlain || shape == kShapeMaskOut || shape == kShapeMaskConsume ||
         shape == kShapeIndexConsume || shape == kShapeDictGather || shape == kShapeStrSplitMeta;
}

DecodeKernelSpec render_masked_char_copy()
{
  DecodeKernelSpec spec;
  spec.entry_symbol = "simpatico_masked_char_copy";
  spec.block_x      = 256;
  spec.note         = "fixed masked char-range copy (str_split phase 2)";
  spec.source       = R"src(
extern "C" __global__ void simpatico_masked_char_copy(
    const unsigned char* __restrict__ chars,
    const long long* __restrict__ src_offsets,
    const int* __restrict__ out_offsets,
    long long n_survivors,
    unsigned char* __restrict__ out)
{
    const long long stride = static_cast<long long>(gridDim.x) * blockDim.x;
    for (long long j = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
         j < n_survivors; j += stride) {
        const long long s = src_offsets[j];
        const int o       = out_offsets[j];
        const int len     = out_offsets[j + 1] - o;
        for (int b = 0; b < len; ++b) out[o + b] = chars[s + b];
    }
}
)src";
  return spec;
}

// =====================================================================
// Public entry point.
// =====================================================================
DecodeKernelSpec render(const ::codegen::jit::FusedTree& tree,
                        const std::string& element_dtype,
                        std::int32_t num_chunks,
                        DecodeShape shape)
{
  if (element_dtype.empty()) {
    throw std::invalid_argument("decode render: element_dtype is empty");
  }
  if (!shape_is_supported(shape)) {
    throw RenderError("decode render: unsupported enumerator/consumer combination");
  }
  if (num_chunks < 1) { throw std::invalid_argument("decode render: num_chunks must be >= 1"); }
  if (dtype_elem_size(element_dtype) == 0) {
    throw RenderError("decode render: unsupported element_dtype '" + element_dtype +
                      "'. Supported: int32_t, int64_t");
  }
  Walker w(element_dtype, shape);
  return w.build(tree);
}

}  // namespace codegen::decode::jit
