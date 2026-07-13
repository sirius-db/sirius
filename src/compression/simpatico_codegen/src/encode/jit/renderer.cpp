// Encode-side renderer — recursive composable walker.
//
// A single tree-walk that emits one CUDA-C++ kernel for any FusedTree shape.
//
// Architecture
// ============
//
//  +-----------------+
//  | FusedTree       |
//  +-------+---------+
//          |
//          v
//  +-------+----------------------------------+
//  | Walker                                   |
//  |                                          |
//  |   emit_node(tree, LaneInput::root_global)|
//  |     +-> emit_delta (transformer)         |
//  |     +-> emit_rle   (stage boundary)      |
//  |     +-> emit_for   (transformer)         |
//  |     +-> emit_bitpack (LEAF; sink of      |
//  |          the fused cluster — tail codecs |
//  |          like bitcomp/ANS attach to the  |
//  |          `packed` channel downstream)    |
//  |     +-> emit_raw   (leaf, no encode)     |
//  |                                          |
//  |   build() assembles the source string    |
//  +-----------+------------------------------+
//              |
//              v
//      EncodeKernelSpec
//
// LaneInput contract
// ------------------
// Every op receives a `LaneInput` from its parent describing "how does
// lane T read its input value of element type E for this stage?".
// `read_expr` is a C++ expression containing the literal token
// `__LANE__` which the consumer substitutes with the per-lane index
// expression (typically `tid`, but conceivably `tid + offset` for a
// stride loop).  `length_expr` is the effective per-chunk length at
// THIS stage.
//
// Each op then either:
//   (a) Leaf (Bitpack, Raw): consumes the LaneInput and emits the
//       final per-lane work, no recursion.
//   (b) Inline transformer (Delta, FOR): returns a wrapped LaneInput
//       whose `read_expr` is a closed-form C++ expression in the
//       parent's `read_expr` (e.g. Delta: "(parent at LANE+1) -
//       (parent at LANE)").  No __syncthreads(), no shared mem.
//       Recurses into its sole input child with the new LaneInput.
//   (c) Stage boundary (Rle, deep Delta with register pressure):
//       materialises its output into a shared-mem slab, emits
//       __syncthreads(), returns a Shared-kind LaneInput pointing at
//       the slab.  Recurses into each child with the new LaneInput.
//
// Buffer + symbol naming
// ----------------------
// Each op suffixes its kernel-parameter names and buffer-field names
// with its node_id (assigned in DFS-preorder lex-sorted children) so
// the same op kind (Bitpack, etc.) can appear twice in a tree without
// collision.  The kernel parameter list order is the same order as
// the BufferSpecs in EncodeKernelSpec::buffers — that's the contract
// the launcher relies on for passing CUdeviceptr args via cuLaunchKernel.
//
// Failure handling
// ----------------
// Bitpack / Delta / Rle / For / Zigzag / Raw all render; Raw is a verbatim
// passthrough leaf synthesized for a delta/rle/for channel that isn't further
// fused.  An op the renderer can't emit throws RenderError with a clear
// diagnostic.

#include "codegen/encode/jit/renderer.hpp"

#include "codegen/tree.hpp"

#include <cctype>
#include <cstdint>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace codegen::encode::jit {

namespace {

// ---------------------------------------------------------------------
// Dtype table — one source of truth for the per-type info the walker
// needs (size, sentinel-max literal).  Extending dtype support is one
// table-entry edit.
// ---------------------------------------------------------------------
struct DtypeInfo {
  std::size_t elem_size;
  const char* max_literal;
  const char* min_literal;
};

const DtypeInfo* lookup_dtype(const std::string& name)
{
  static const std::unordered_map<std::string, DtypeInfo> table = {
    {"int8_t", {sizeof(std::int8_t), "INT8_MAX", "INT8_MIN"}},
    {"int32_t", {sizeof(std::int32_t), "INT32_MAX", "INT32_MIN"}},
    {"int64_t", {sizeof(std::int64_t), "INT64_MAX", "INT64_MIN"}},
  };
  auto it = table.find(name);
  return (it == table.end()) ? nullptr : &it->second;
}

// Same-width unsigned type name, for arithmetic that must wrap modulo 2^N
// instead of relying on signed overflow (UB).
const char* unsigned_counterpart(std::size_t elem_size)
{
  return (elem_size == 8) ? "uint64_t" : "uint32_t";
}

// ---------------------------------------------------------------------
// String utilities.
// ---------------------------------------------------------------------

std::string replace_all(std::string s, std::string_view needle, std::string_view repl)
{
  std::string out;
  out.reserve(s.size());
  std::size_t pos = 0;
  while (true) {
    auto found = s.find(needle, pos);
    if (found == std::string::npos) {
      out.append(s, pos, std::string::npos);
      break;
    }
    out.append(s, pos, found - pos);
    out.append(repl);
    pos = found + needle.size();
  }
  return out;
}

// Substitute the LaneInput's `__LANE__` token with the given lane
// expression.  The result is a valid C++ expression in `tid` (or
// whatever the caller passed) producing the input element value.
std::string at_lane(const std::string& read_expr, const std::string& lane_expr)
{
  return replace_all(read_expr, "__LANE__", lane_expr);
}

// Walk-order op tag for the entry symbol.  Kept short — the
// kernel-cache key is the source hash, not the symbol.
void append_op_segment(std::ostringstream& oss, const ::codegen::jit::FusedTree& node)
{
  switch (node.op) {
    case ::codegen::OpKind::Bitpack: oss << "bp"; break;
    case ::codegen::OpKind::Delta: oss << "dl"; break;
    case ::codegen::OpKind::Rle: oss << "rl"; break;
    case ::codegen::OpKind::Raw: oss << "rw"; break;
    case ::codegen::OpKind::For: oss << "fr"; break;
    case ::codegen::OpKind::Zigzag: oss << "zz"; break;
    default: oss << "un"; break;
  }
  for (const auto& [k, child] : node.children) {
    (void)k;
    oss << "_";
    append_op_segment(oss, *child);
  }
}

std::string make_entry_symbol(const ::codegen::jit::FusedTree& tree,
                              const std::string& element_dtype)
{
  std::ostringstream oss;
  oss << "simpatico_encode_";
  append_op_segment(oss, tree);
  oss << "_";
  for (char c : element_dtype) {
    oss << (std::isalnum(static_cast<unsigned char>(c)) || c == '_' ? c : '_');
  }
  return oss.str();
}

// ---------------------------------------------------------------------
// LaneInput — the parent-to-child contract.
// ---------------------------------------------------------------------
struct LaneInput {
  enum Kind { Global, Shared, Expr };
  Kind kind = Global;
  std::string elem_type;    // C++ element-type string
  std::string read_expr;    // C++ expression w/ __LANE__ placeholder
  std::string length_expr;  // C++ expression for per-chunk length
};

// ---------------------------------------------------------------------
// SharedMemAllocator — stack-discipline shared-memory budget tracker.
//
// Replaces per-op `__shared__ T arr[N];` declarations with offsets
// into a single `extern __shared__ alignas(16) unsigned char
// workspace[]` buffer.  The walker carves up the workspace at codegen
// time; the launcher passes the computed peak size as the
// `sharedMemBytes` argument to cuLaunchKernel.
//
// Why this exists
// ---------------
// Naive per-op static __shared__ decls hit ~76 KB for
// Rle{Rle{Bp,Bp}, Bp} and ~56 KB for i64 single-Rle.  The default
// per-block static shared limit is 48 KB on every supported arch;
// kernels over the limit either fail to LOAD (compile-time error in
// nvcc) or fail to LAUNCH.
//
// Two cooperating optimisations make the peak shrink dramatically:
//
//   (A) Stack-discipline reuse across siblings.  Each emit_X
//       allocates its slabs at `cur_`, uses them, then releases back
//       to a pre-saved mark.  The NEXT sibling (or recursion path)
//       starts allocating from the same `cur_` — same backing
//       bytes, different semantic content.  Sibling Bitpacks under
//       an Rle therefore reuse the same min/max scratch slots
//       instead of each owning their own.
//
//   (B) Intra-op staged release.  Inside emit_rle, the run-detection
//       scratch (sh_input, sh_run_idx, sh_start_pos, sh_nruns) dies
//       BEFORE the children execute (the children only read from
//       sh_values + sh_counts).  We release the scratch mid-body so
//       children's slabs reuse those bytes.  Inside emit_bitpack,
//       sh_min dies before sh_max is needed — same trick at a
//       finer grain.
//
// Per-op contract
// ---------------
// Every emit_X is required to leave `cur_` in the same state it was
// on entry.  Parents may rely on this when accounting for permanent
// vs transient slabs (e.g. emit_rle keeps sh_values+sh_counts alive
// across child recursion).
//
// Peak tracking
// -------------
// `peak_` is monotone — `release_to` only rewinds `cur_`, never
// `peak_`.  After the walk completes, `peak_bytes()` returns the
// max simultaneous live workspace usage, which becomes
// EncodeKernelSpec::shared_bytes.
//
// Alignment
// ---------
// Each `alloc(elem_size, count)` aligns `cur_` up to
// max(elem_size, 4) bytes — guarantees the reinterpret_cast<T*> the
// walker emits hits a properly aligned pointer for natural-width
// loads/stores.  Workspace base is `alignas(16)` so 8-byte slabs are
// also safe.
// ---------------------------------------------------------------------
class SharedMemAllocator {
 public:
  using Mark = std::size_t;

  // Allocate `count * elem_size` bytes, aligned to max(elem_size, 4).
  // Returns the BYTE OFFSET into the workspace; emitter consumes
  // it via `reinterpret_cast<T*>(workspace + offset)`.
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

  // Release back to a previously-saved mark.  Peak is NOT rewound.
  void release_to(Mark m) noexcept { cur_ = m; }

  std::size_t peak_bytes() const noexcept { return peak_; }

 private:
  std::size_t cur_  = 0;
  std::size_t peak_ = 0;
};

// ---------------------------------------------------------------------
// Walker — accumulates the kernel as we visit the tree.
// ---------------------------------------------------------------------
class Walker {
 public:
  Walker(std::string dtype, const DtypeInfo& dt, std::int32_t num_chunks)
    : dtype_(std::move(dtype)), dt_(dt), num_chunks_(num_chunks)
  {
  }

  EncodeKernelSpec build(const ::codegen::jit::FusedTree& tree)
  {
    // Root LaneInput: lane T reads its value from `flat[start + (T)]`,
    // length is the prelude-computed `len` (chunk's effective length).
    LaneInput root;
    root.kind        = LaneInput::Global;
    root.elem_type   = dtype_;
    root.read_expr   = "flat[start + (__LANE__)]";
    root.length_expr = "len";

    emit_node(tree, root);
    return finalize(tree);
  }

 private:
  // ----------------- Accumulators -----------------
  std::string dtype_;
  const DtypeInfo& dt_;
  std::int32_t num_chunks_;
  std::int32_t next_node_id_ = 0;
  std::ostringstream params_;  // post-(flat,n) params, joined with ",\n"
  std::ostringstream body_;    // kernel body lines
  std::vector<BufferSpec> buffers_;
  SharedMemAllocator sm_;

  std::int32_t take_id() { return next_node_id_++; }

  void add_param(const std::string& type, const std::string& name)
  {
    if (params_.tellp() > 0) params_ << ",\n";
    params_ << "    " << type << " " << name;
  }

  void add_buffer(std::int32_t node_id,
                  const std::string& field,
                  std::size_t elem_size,
                  std::size_t length,
                  bool no_pre_zero = false)
  {
    BufferSpec b;
    b.node_id     = node_id;
    b.field       = field;
    b.elem_size   = elem_size;
    b.length      = length;
    b.no_pre_zero = no_pre_zero;
    buffers_.push_back(std::move(b));
  }

  EncodeKernelSpec finalize(const ::codegen::jit::FusedTree& tree)
  {
    EncodeKernelSpec spec;
    spec.entry_symbol = make_entry_symbol(tree, dtype_);

    std::ostringstream src;
    src << kPrelude;
    src << "\n"
        << "extern \"C\" __global__\n"
        << "void " << spec.entry_symbol << "(\n"
        << "    const " << dtype_ << "* __restrict__ flat,\n"
        << "    int64_t                         n";
    if (params_.tellp() > 0) { src << ",\n" << params_.str(); }
    src << ")\n"
        << "{\n"
        << "    constexpr int32_t CHUNK = 1024;\n"
        << "\n"
        << "    const int32_t chunk_id = blockIdx.x;\n"
        << "    const int32_t tid      = threadIdx.x;\n"
        << "    const int64_t start    = static_cast<int64_t>(chunk_id) *\n"
        << "                             static_cast<int64_t>(CHUNK);\n"
        << "    const int32_t len      = static_cast<int32_t>(\n"
        << "        (n - start) < static_cast<int64_t>(CHUNK)\n"
        << "            ? (n - start)\n"
        << "            : static_cast<int64_t>(CHUNK));\n"
        << "\n"
        << "    // Single dynamic-shared workspace, carved up by\n"
        << "    // SharedMemAllocator at codegen time.  Size passed via\n"
        << "    // cuLaunchKernel's sharedMemBytes (EncodeKernelSpec::shared_bytes).\n"
        << "    extern __shared__ __align__(16) unsigned char workspace[];\n"
        << "\n"
        << body_.str() << "}\n";

    spec.source       = src.str();
    spec.buffers      = std::move(buffers_);
    spec.block_x      = 128;  // uniform block=128 (fused TBSize)
    spec.shared_bytes = static_cast<std::int32_t>(sm_.peak_bytes());
    spec.note =
      "walker-rendered, OverAllocate; "
      "dynamic-shared workspace = " +
      std::to_string(sm_.peak_bytes()) + " bytes";
    return spec;
  }

  // ----------------- Per-op emitters -----------------
  void emit_node(const ::codegen::jit::FusedTree& node, LaneInput in);
  void emit_delta(const ::codegen::jit::FusedTree& node, LaneInput in);
  void emit_for(const ::codegen::jit::FusedTree& node, LaneInput in);
  // use_smem: accumulate into a per-block __shared__ slab, then store to global.
  //   Eliminates the cudaMemsetAsync pre-zero requirement for the packed buffer.
  //   Valid only for leaf Bitpacks where bits * kChunkSize / 32 + 2 fits in the
  //   smem budget (runs int32, bits ≤ 10: 322 words × 4 = 1288 bytes;
  //   values int32, bits ≤ 32: 1026 words × 4 = 4 KB).
  void emit_bitpack(const ::codegen::jit::FusedTree& node, LaneInput in, bool use_smem = false);
  void emit_rle(const ::codegen::jit::FusedTree& node, LaneInput in);
  void emit_raw(const ::codegen::jit::FusedTree& node, LaneInput in);
  void emit_zigzag(const ::codegen::jit::FusedTree& node, LaneInput in);

  // ----------------- Source prelude (constant per kernel) -----------------
  static constexpr const char* kPrelude = R"src(
// nvrtc cannot see the host stdlib; CCCL provides the C++ standard
// surface for device code via <cuda/std/...>.  These typedefs let the
// walker emit signatures using bare `int32_t` / `uint32_t` / etc.
#include <cuda/std/cstdint>
#include <cuda/std/climits>
#include <cub/block/block_reduce.cuh>
#include <cub/block/block_scan.cuh>

struct SimpaticoMin { template <class T> __device__ __forceinline__ T operator()(const T& a, const T& b) const { return a < b ? a : b; } };
struct SimpaticoMax { template <class T> __device__ __forceinline__ T operator()(const T& a, const T& b) const { return a > b ? a : b; } };

using ::cuda::std::int8_t;
using ::cuda::std::int16_t;
using ::cuda::std::int32_t;
using ::cuda::std::int64_t;
using ::cuda::std::uint8_t;
using ::cuda::std::uint16_t;
using ::cuda::std::uint32_t;
using ::cuda::std::uint64_t;

namespace {

// __clzll-based bit-width.  Returns 0 for zero range (constant chunk —
// decode fast-path handles bits==0).  Constant chunks write zero packed
// bytes, matching nvcomp's convention and maximising ratio for sorted /
// low-cardinality data.  The decode prelude's `if (bits == 0) return minv`
// path handles reconstruction.
__device__ inline int simpatico_bit_width_u64(unsigned long long x) {
    return (x == 0ULL) ? 0 : (64 - __clzll(x));
}

}  // namespace
)src";
};

// =====================================================================
// emit_node — dispatch to the per-op emitter.
// =====================================================================
void Walker::emit_node(const ::codegen::jit::FusedTree& node, LaneInput in)
{
  switch (node.op) {
    case ::codegen::OpKind::Bitpack: emit_bitpack(node, std::move(in)); return;
    case ::codegen::OpKind::Delta: emit_delta(node, std::move(in)); return;
    case ::codegen::OpKind::Rle: emit_rle(node, std::move(in)); return;
    case ::codegen::OpKind::Raw: emit_raw(node, std::move(in)); return;
    case ::codegen::OpKind::For: emit_for(node, std::move(in)); return;
    case ::codegen::OpKind::Zigzag: emit_zigzag(node, std::move(in)); return;
    case ::codegen::OpKind::None:
    default:
      throw RenderError(std::string("render: invalid op kind '") +
                        ::codegen::jit::op_kind_name(node.op) + "'");
  }
}

// =====================================================================
// emit_delta — inline transformer.
//
// Delta reads its input at fixed lane positions, so its output is a
// closed-form C++ expression in __LANE__ (no shared-mem
// materialisation, no __syncthreads()).  We simply rewrite the
// LaneInput.read_expr to "(parent at LANE+1) - (parent at LANE)" and
// shorten the length by 1.  Children consume the new expression
// uniformly.
//
// Delta's only persistent output is `delta_first[c]` — the first
// element of chunk c, captured as a single-thread write.  Bound to
// node_id; the decode side reads it back via the matching manifest
// key.
//
// Edge cases
// ----------
//   * len == 0: write Element{0} sentinel into delta_first; the
//     subsequent Bitpack sees length_expr that evaluates to 0
//     (the new length is `(len > 0 ? len - 1 : 0)`) and takes its
//     own empty-chunk branch.
//   * len == 1: delta_first[c] = flat[start]; diff stream length is
//     zero — Bitpack sentinel-headers the chunk.
// =====================================================================
void Walker::emit_delta(const ::codegen::jit::FusedTree& node, LaneInput in)
{
  if (node.children.size() != 1) {
    throw RenderError(
      "render: Delta must have exactly one child named 'differences' "
      "(got " +
      std::to_string(node.children.size()) + " children)");
  }
  auto vit = node.children.find("differences");
  if (vit == node.children.end()) {
    throw RenderError("render: Delta missing 'differences' child (got '" +
                      node.children.begin()->first + "' instead)");
  }

  // Per-op dtype lookup — Delta's element size follows the
  // LaneInput, NOT the root column dtype.  Future-proofs for e.g.
  // Delta-under-Rle.runs (int32 counts) even though that's an
  // unusual chain.
  const DtypeInfo* op_dt = lookup_dtype(in.elem_type);
  if (op_dt == nullptr) {
    throw RenderError("render: Delta op-local dtype '" + in.elem_type + "' not in dtype table");
  }

  const std::int32_t id   = take_id();
  const std::string idstr = std::to_string(id);

  // Own output: delta_first[c].  One element per chunk.
  const std::string first_param = "delta_first_" + idstr;
  add_param(in.elem_type + "*", first_param);
  add_buffer(id, "delta_first", op_dt->elem_size, static_cast<std::size_t>(num_chunks_));

  // Single-thread body contribution: capture the first element (or
  // sentinel zero for an empty chunk).
  body_ << "    // --- node " << id << ": Delta (" << in.elem_type << ") ---\n"
        << "    if (tid == 0) {\n"
        << "        " << first_param << "[chunk_id] = ((" << in.length_expr << ") > 0)\n"
        << "            ? (" << at_lane(in.read_expr, "0") << ")\n"
        << "            : " << in.elem_type << "{0};\n"
        << "    }\n";

  // Rewrite LaneInput for the child: diff = parent[LANE+1] - parent[LANE],
  // length shrinks by 1 (or 0 if the parent length was 0).
  //
  // Composability note: this is the inline path — read_expr is a
  // closed-form C++ expression in __LANE__, spliced directly into the
  // scan input.  Kept as long as register pressure is fine; the
  // staged path flips to shared-mem materialisation when needed.
  //
  // The subtraction is done in the unsigned counterpart type: adjacent
  // values can differ by more than the signed range holds (e.g. delta over
  // raw float32 bit patterns), and the diff/reconstruct roundtrip relies on
  // wraparound. Signed overflow is UB in C++; unsigned wraparound is
  // well-defined, so this is not just cosmetic.
  const std::string dutype = unsigned_counterpart(op_dt->elem_size);
  LaneInput out;
  out.kind      = LaneInput::Expr;
  out.elem_type = in.elem_type;
  out.read_expr = "(static_cast<" + in.elem_type + ">(static_cast<" + dutype + ">(" +
                  at_lane(in.read_expr, "(__LANE__) + 1") + ") - static_cast<" + dutype + ">(" +
                  at_lane(in.read_expr, "(__LANE__)") + ")))";
  out.length_expr = "((" + in.length_expr + ") > 0 ? " + "((" + in.length_expr + ") - 1) : 0)";

  emit_node(*vit->second, std::move(out));
}

// =====================================================================
// emit_for — semi-inline transformer.
//
// FOR computes a per-chunk minimum (via CUB BlockReduce), stores it in
// `references[chunk_id]`, then rewrites the LaneInput so downstream ops
// see `value - chunk_min` (the residual).  Unlike Delta it needs a
// block-wide reduction and a __syncthreads() before the child can read
// the rewritten expression, but it does NOT change the element count or
// need a shared-mem slab for the residuals themselves — those are
// expressed as a closed-form C++ expression that the child splices in.
//
// `references` is a kernel output buffer (num_chunks × elem_size).  It
// is exposed by the encode bridge as `named_channels()["references"]` so
// the compress boundary loop can store it or route it to a downstream op
// (e.g. `for.references -> identity/snappy`), exactly like bitpack's
// `chunk_min`/`packed` channels.
//
// CUB BlockReduce TempStorage uses a static __shared__ declaration
// inside the body (matching emit_bitpack style) — not tracked by
// SharedMemAllocator since it is a fixed-size CUB scratch, not a
// logical data slab.  The broadcast cell `sh_for_min_N` is also static.
// =====================================================================
void Walker::emit_for(const ::codegen::jit::FusedTree& node, LaneInput in)
{
  if (node.children.size() != 1) {
    throw RenderError(
      "render: FOR must have exactly one child named 'deltas' "
      "(got " +
      std::to_string(node.children.size()) + " children)");
  }
  auto dit = node.children.find("deltas");
  if (dit == node.children.end()) {
    throw RenderError("render: FOR missing 'deltas' child (got '" + node.children.begin()->first +
                      "' instead)");
  }

  const DtypeInfo* op_dt = lookup_dtype(in.elem_type);
  if (op_dt == nullptr) {
    throw RenderError("render: FOR op-local dtype '" + in.elem_type + "' not in dtype table");
  }

  const std::int32_t id   = take_id();
  const std::string idstr = std::to_string(id);

  // Output buffer: references[num_chunks] — one per-chunk minimum.
  const std::string p_refs = "references_" + idstr;
  add_param(in.elem_type + "*", p_refs);
  add_buffer(id, "references", op_dt->elem_size, static_cast<std::size_t>(num_chunks_));

  // Local names for the CUB reduce and broadcast cell.
  const std::string sh_min = "sh_for_min_" + idstr;
  const std::string v_min  = "for_min_" + idstr;
  const std::string v_lmin = "for_lmin_" + idstr;

  // Body: per-lane local min → CUB BlockReduce → broadcast → store.
  body_ << "    // --- node " << id << ": FOR (semi-inline, " << in.elem_type << ") ---\n"
        << "    const int32_t for_len_" << idstr << " = static_cast<int32_t>(" << in.length_expr
        << ");\n"
        << "    " << in.elem_type << " " << v_lmin << " = static_cast<" << in.elem_type << ">("
        << op_dt->max_literal << ");\n"
        << "    for (int32_t i = tid; i < for_len_" << idstr << "; i += 128) {\n"
        << "        " << in.elem_type << " _fv = (" << at_lane(in.read_expr, "i") << ");\n"
        << "        if (_fv < " << v_lmin << ") " << v_lmin << " = _fv;\n"
        << "    }\n"
        << "    typedef cub::BlockReduce<" << in.elem_type << ", 128> ForBR_" << idstr << ";\n"
        << "    __shared__ typename ForBR_" << idstr << "::TempStorage for_br_ts_" << idstr << ";\n"
        << "    __shared__ " << in.elem_type << " " << sh_min << ";\n"
        << "    {\n"
        << "        " << in.elem_type << " _m = ForBR_" << idstr << "(for_br_ts_" << idstr
        << ").Reduce(" << v_lmin << ", SimpaticoMin());\n"
        << "        if (tid == 0) " << sh_min << " = _m;\n"
        << "    }\n"
        << "    __syncthreads();\n"
        << "    const " << in.elem_type << " " << v_min << " = " << sh_min << ";\n"
        << "    if (tid == 0) " << p_refs << "[chunk_id] = " << v_min << ";\n";

  // Rewrite LaneInput for the child: residual = parent_value - chunk_min.
  // Length is unchanged — FOR preserves the element count.
  //
  // Unsigned subtraction (see emit_delta): parent_value - chunk_min can
  // exceed the signed range even though chunk_min IS the block minimum,
  // when the values themselves span more than the signed range (e.g. raw
  // float32 bit patterns).
  const std::string futype = unsigned_counterpart(op_dt->elem_size);
  LaneInput out;
  out.kind      = LaneInput::Expr;
  out.elem_type = in.elem_type;
  out.read_expr = "(static_cast<" + in.elem_type + ">(static_cast<" + futype + ">(" +
                  at_lane(in.read_expr, "(__LANE__)") + ") - static_cast<" + futype + ">(" + v_min +
                  ")))";
  out.length_expr = in.length_expr;

  emit_node(*dit->second, std::move(out));
}

// =====================================================================
// emit_rle — STAGE-BOUNDARY transformer.
//
// The first op whose output positions are data-dependent: each chunk
// produces a variable number of runs.  Inline composition (Delta-style)
// is impossible because downstream lanes can't express their input as
// a closed-form C++ expression in __LANE__ — they have to wait for the
// run-detection scan to land in shared memory.
//
// Algorithm (per chunk = one CUDA block):
//
//   Stage 1a: load `len` elements from `in.read_expr` into
//             sh_input_N[CHUNK].
//   Stage 1b: each lane computes is_start (kept in a register; NOT
//             materialised to shared).
//   Stage 1c: seed sh_run_idx_N with is_start, then block-wide
//             Hillis-Steele inclusive scan.  log2(CHUNK)=10 passes,
//             two __syncthreads() per pass — slow but obvious.  CUB
//             BlockScan would shave passes; deferred.
//   Stage 1d: lane 0 reads sh_run_idx_N[len-1] → sh_nruns_N,
//             broadcasts to all lanes via shared cell.
//   Stage 1e: start lanes write sh_values_N[run_idx_excl] = input[t]
//             and sh_start_pos_N[run_idx_excl] = t.  Lane 0 writes
//             the start-pos sentinel at index nruns.
//   Stage 1f: lanes < nruns compute sh_counts_N[t] =
//             sh_start_pos_N[t+1] - sh_start_pos_N[t].
//   Stage 1g: lane 0 writes rle_runs_offsets[c+1] = nruns_N (and
//             only chunk-0's lane 0 zeroes rle_runs_offsets[0]).
//             Launcher post-cumsum will inclusive-scan
//             rle_runs_offsets[0..num_chunks] in-place to produce
//             the decoder-facing exclusive cumsum.
//
// Children (lex order: "runs" then "values") then run their bodies
// in the same kernel, reading from sh_counts_N / sh_values_N via a
// kind=Shared LaneInput.  The Bitpack early-out path is now
// `do { } while(0) + break` rather than `return`, so the runs
// child's empty-chunk sentinel doesn't kill the values child.
//
// Shared-mem footprint (per Rle, after stack-discipline reuse):
//
//   Phase 1 (stage 1, all 6 slabs live):
//     sh_values (op_dt)     + sh_counts  (i32)
//     sh_input  (op_dt)     + sh_run_idx (i32)
//     sh_start_pos (i32+4B) + sh_nruns   (4B)
//     i32 column: 4+4+4+4+4 = ~20 KB
//     i64 column: 8+4+8+4+4 = ~28 KB
//
//   Phase 2 (children recurse — transients released, only sh_values
//   + sh_counts kept alive over child code):
//     i32 column: ~8 KB  base + child peak
//     i64 column: ~12 KB base + child peak
//
// Compared to a naive layout (separate __shared__ decls):
//
//   i32 Rle{Bp,Bp}:        44 KB  ->  20 KB peak
//   i64 Rle{Bp,Bp}:        60 KB  ->  28 KB peak
//   i32 Rle{Rle{Bp,Bp},Bp}: 76 KB  ->  28 KB peak  (now fits!)
//
// All cases land well under the 48 KB default static-shared limit;
// no cudaFuncSetAttribute opt-in needed for these shapes.
// =====================================================================
void Walker::emit_rle(const ::codegen::jit::FusedTree& node, LaneInput in)
{
  if (node.children.size() != 2) {
    throw RenderError(
      "render: Rle must have exactly two children ('runs' and "
      "'values'); got " +
      std::to_string(node.children.size()));
  }
  auto rit = node.children.find("runs");
  auto vit = node.children.find("values");
  if (rit == node.children.end() || vit == node.children.end()) {
    throw RenderError("render: Rle missing 'runs' or 'values' child");
  }
  const DtypeInfo* op_dt = lookup_dtype(in.elem_type);
  if (op_dt == nullptr) {
    throw RenderError("render: Rle op-local dtype '" + in.elem_type + "' not in dtype table");
  }

  const std::int32_t id   = take_id();
  const std::string idstr = std::to_string(id);

  const std::string p_offsets   = "rle_runs_offsets_" + idstr;
  const std::string sh_startpos = "sh_start_pos_" + idstr;
  const std::string sh_values   = "sh_values_" + idstr;
  const std::string sh_counts   = "sh_counts_" + idstr;
  const std::string v_nruns     = "nruns_" + idstr;

  // Own output: rle_runs_offsets[num_chunks + 1] int32_t.  Filled
  // by the per-chunk kernel as raw nruns[c] at position c+1; the
  // launcher inclusive-scans in-place to obtain the decoder-facing
  // exclusive cumsum.
  add_param("int32_t*", p_offsets);
  add_buffer(
    id, "rle_runs_offsets", sizeof(std::int32_t), static_cast<std::size_t>(num_chunks_) + 1);

  // Shared-mem layout:
  //   sh_values  — permanent: lives until the values child finishes.
  //   sh_counts  — semi-permanent: written by phase 1, read by the RUNS
  //                child only.  Released after the runs child emits so
  //                the VALUES child (inner Rle) can reuse those bytes.
  //                For nested Rle shapes (e.g. nvcomp_def) this drops
  //                peak smem from 28 KB → 24 KB → +1 extra block/SM.
  //   sh_start_pos — transient: scratch for the count-gap computation,
  //                  released before any child.
  const auto mark_pre_rle      = sm_.mark();
  const std::size_t off_values = sm_.alloc(op_dt->elem_size, ::codegen::kChunkSize);
  const auto mark_after_values = sm_.mark();  // save point for deferred release
  const std::size_t off_counts = sm_.alloc(sizeof(std::int32_t), ::codegen::kChunkSize);
  const auto mark_post_perm    = sm_.mark();
  const std::size_t off_startp =
    sm_.alloc(sizeof(std::int32_t), static_cast<std::size_t>(::codegen::kChunkSize) + 1);

  body_ << "    // --- node " << id << ": Rle (stage-boundary, block=128 grid-stride, "
        << in.elem_type << " -> values=" << in.elem_type << ", runs=int32_t) ---\n"
        << "    " << in.elem_type << "* " << sh_values << " = reinterpret_cast<" << in.elem_type
        << "*>(workspace + " << off_values << ");\n"
        << "    int32_t* " << sh_counts << " = reinterpret_cast<int32_t*>(workspace + "
        << off_counts << ");\n"
        << "    int32_t* " << sh_startpos << " = reinterpret_cast<int32_t*>(workspace + "
        << off_startp << ");\n"
        << "    int32_t " << v_nruns << " = 0;\n"
        << "    {\n"
        << "        constexpr int32_t IPT = CHUNK / 128;\n"
        << "        const int32_t rle_n = static_cast<int32_t>(" << in.length_expr << ");\n"
        << "        // Run-start flags, BLOCKED layout: thread t owns idx = t*IPT + j.\n"
        << "        int32_t is_start[IPT];\n"
        << "        #pragma unroll\n"
        << "        for (int32_t j = 0; j < IPT; ++j) {\n"
        << "            const int32_t idx = tid * IPT + j;\n"
        << "            is_start[j] = (idx < rle_n && (idx == 0 ||\n"
        << "                (" << at_lane(in.read_expr, "idx") << ") != ("
        << at_lane(in.read_expr, "(idx - 1)") << "))) ? 1 : 0;\n"
        << "        }\n"
        << "        // Exclusive scan of run-start flags -> per-element run index + total.\n"
        << "        typedef cub::BlockScan<int32_t, 128> RleScan_" << idstr << ";\n"
        << "        __shared__ typename RleScan_" << idstr << "::TempStorage rle_ts_" << idstr
        << ";\n"
        << "        __shared__ int32_t rle_snruns_" << idstr << ";\n"
        << "        int32_t output_idx[IPT];\n"
        << "        int32_t total_runs = 0;\n"
        << "        RleScan_" << idstr << "(rle_ts_" << idstr
        << ").ExclusiveSum(is_start, output_idx, total_runs);\n"
        << "        // Start lanes scatter their value + start position.\n"
        << "        #pragma unroll\n"
        << "        for (int32_t j = 0; j < IPT; ++j) {\n"
        << "            const int32_t idx = tid * IPT + j;\n"
        << "            if (idx < rle_n && is_start[j]) {\n"
        << "                " << sh_values << "[output_idx[j]] = (" << at_lane(in.read_expr, "idx")
        << ");\n"
        << "                " << sh_startpos << "[output_idx[j]] = idx;\n"
        << "            }\n"
        << "        }\n"
        << "        if (tid == 0) {\n"
        << "            rle_snruns_" << idstr << " = total_runs;\n"
        << "            " << sh_startpos << "[total_runs] = rle_n;\n"
        << "            " << p_offsets << "[chunk_id + 1] = total_runs;\n"
        << "            if (chunk_id == 0) " << p_offsets << "[0] = 0;\n"
        << "        }\n"
        << "        __syncthreads();\n"
        << "        " << v_nruns << " = rle_snruns_" << idstr << ";\n"
        << "        // Counts = gaps between consecutive run start positions.\n"
        << "        for (int32_t k = tid; k < " << v_nruns << "; k += 128) {\n"
        << "            " << sh_counts << "[k] = " << sh_startpos << "[k + 1] - " << sh_startpos
        << "[k];\n"
        << "        }\n"
        << "        __syncthreads();\n"
        << "    }\n";

  // Release transient sh_start_pos.  Children re-allocate over those bytes.
  sm_.release_to(mark_post_perm);

  // Stage 2: recurse into children.  Lex order — "runs" before "values".
  LaneInput runs_in;
  runs_in.kind        = LaneInput::Shared;
  runs_in.elem_type   = "int32_t";
  runs_in.read_expr   = sh_counts + "[__LANE__]";
  runs_in.length_expr = v_nruns;
  // The runs child is always a Bitpack leaf (run counts are bounded:
  // bits ≤ ceil(log2(CHUNK)) = 10 → smem slab = 322 words × 4 = 1288 bytes).
  // Use the smem-accumulation path to eliminate the cudaMemsetAsync pre-zero
  // for the packed_runs buffer (~60–240 MB per call depending on column size).
  const bool runs_is_bp_leaf =
    rit->second->op == ::codegen::OpKind::Bitpack && rit->second->children.empty();
  if (runs_is_bp_leaf) {
    emit_bitpack(*rit->second, std::move(runs_in), /*use_smem=*/true);
  } else {
    emit_node(*rit->second, std::move(runs_in));
  }

  // sh_counts is no longer needed after the runs child emits.  Release
  // it so the values child (often a nested Rle) allocates there instead
  // of above it.  At runtime the runs child has already consumed
  // sh_counts; the values child will harmlessly overwrite those bytes.
  sm_.release_to(mark_after_values);

  LaneInput vals_in;
  vals_in.kind        = LaneInput::Shared;
  vals_in.elem_type   = in.elem_type;
  vals_in.read_expr   = sh_values + "[__LANE__]";
  vals_in.length_expr = v_nruns;
  // Use smem accumulation for int32 values Bitpack leaves.
  // int64 values (bits ≤ 64) would need 8 KB slab — too expensive for occupancy.
  const bool vals_is_bp_leaf_i32 = vit->second->op == ::codegen::OpKind::Bitpack &&
                                   vit->second->children.empty() && vals_in.elem_type == "int32_t";
  if (vals_is_bp_leaf_i32) {
    emit_bitpack(*vit->second, std::move(vals_in), /*use_smem=*/true);
  } else {
    emit_node(*vit->second, std::move(vals_in));
  }

  // Release permanent slabs.  emit_X contract: leave allocator in
  // the same state it was on entry.
  sm_.release_to(mark_pre_rle);
}

// =====================================================================
// emit_raw — LEAF (passthrough).
//
// Raw stores its input verbatim — no compression.  Today it only
// appears as a child of Rle (`RawRleLeaf` on the decode side), so the
// LaneInput it receives is the parent Rle's per-run shared slab
// (`sh_values`/`sh_counts`) with `length_expr == nruns`.
//
// Buffer layout — OverAllocate, mirroring Bitpack
// -----------------------------------------------
// The decode-side `RawRleLeaf::decode_at` reads
// `data[offsets[chunk_id] + run]`.  A compact `data` (indexed by the
// parent's exclusive run-prefix) would need each chunk's global run
// base, but that prefix is only known AFTER the launcher's post-kernel
// `rle_runs_offsets` cumsum — not at kernel time.  So we use the same
// trick Bitpack's `packed` uses: a fixed per-chunk stride of CHUNK
// elements.  Chunk c writes run r at `data[c*CHUNK + r]` and publishes
// `offsets[c] = c*CHUNK`.  nruns ≤ CHUNK always, so the slot never
// overflows; trailing slots are never read (decode masks to nruns).
// This makes `offsets` an independent fixed-stride array rather than a
// copy of `rle_runs_offsets`, but decode is layout-agnostic.
// =====================================================================
void Walker::emit_raw(const ::codegen::jit::FusedTree& node, LaneInput in)
{
  if (!node.children.empty()) {
    throw RenderError(
      "render: Raw must be a leaf (no children) — it is a "
      "verbatim passthrough sink");
  }
  const DtypeInfo* op_dt = lookup_dtype(in.elem_type);
  if (op_dt == nullptr) {
    throw RenderError("render: Raw op-local dtype '" + in.elem_type + "' not in dtype table");
  }

  const std::int32_t id    = take_id();
  const std::string idstr  = std::to_string(id);
  const std::string p_data = "raw_data_" + idstr;
  const std::string p_offs = "raw_offsets_" + idstr;
  const std::string rlen   = "raw_len_" + idstr;
  const std::string dbase  = "raw_base_" + idstr;

  // Buffer params + specs (order MUST match the launcher's arg vector).
  add_param(in.elem_type + "*", p_data);
  add_param("int32_t*", p_offs);

  const std::size_t nc = static_cast<std::size_t>(num_chunks_);
  add_buffer(id, "data", op_dt->elem_size, nc * static_cast<std::size_t>(::codegen::kChunkSize));
  add_buffer(id, "offsets", sizeof(std::int32_t), nc + 1);

  body_ << "    // --- node " << id << ": Raw (leaf passthrough, " << in.elem_type << ") ---\n"
        << "    {\n"
        << "        // OverAllocate: chunk c's runs occupy the fixed slot\n"
        << "        // [c*CHUNK, (c+1)*CHUNK); offsets[c] = c*CHUNK so the\n"
        << "        // decode-side RawRleLeaf reads data[offsets[c] + r].\n"
        << "        const int64_t " << dbase << " = static_cast<int64_t>(chunk_id) *\n"
        << "                             static_cast<int64_t>(CHUNK);\n"
        << "        if (tid == 0) {\n"
        << "            " << p_offs << "[chunk_id] = static_cast<int32_t>(" << dbase << ");\n"
        << "            if (chunk_id == static_cast<int32_t>(gridDim.x) - 1) {\n"
        << "                " << p_offs << "[chunk_id + 1] =\n"
        << "                    static_cast<int32_t>(" << dbase << " + CHUNK);\n"
        << "            }\n"
        << "        }\n"
        << "        const int32_t " << rlen << " = static_cast<int32_t>(" << in.length_expr
        << ");\n"
        << "        for (int32_t i = tid; i < " << rlen << "; i += 128) {\n"
        << "            " << p_data << "[" << dbase << " + i] = (" << at_lane(in.read_expr, "i")
        << ");\n"
        << "        }\n"
        << "    }\n";
}

// =====================================================================
// emit_zigzag — DUAL-MODE (inline transformer OR storing leaf).
//
// ZigZag is the closed-form per-lane bijection
//   z = (uv << 1) ^ (uv >> (W-1))   (arithmetic shift supplies the sign mask)
// which the decode side inverts with  n = (z >> 1) ^ -(z & 1).
//
// Transformer mode (a FUSABLE op consumes the `zigzag` channel, e.g.
// `zigzag.zigzag -> bitpack`):  ZigZag rewrites LaneInput.read_expr to the
// closed-form map and recurses into the child WITHOUT storing anything — a
// pure inline transform exactly like Delta/FOR splice their residual into the
// child's read expression.  The child stores its own buffers; ZigZag carries
// no per-chunk metadata.
//
// Leaf mode (no outgoing edge, or the edge feeds a NON-fused entropy coder
// such as `…zigzag -> ans`):  ZigZag STORES the transformed stream into the
// single output channel "zigzag" so a downstream non-fused op can entropy-code
// it.  Layout — fixed-stride OverAllocate, identical to FOR's Raw passthrough.
// Element i of chunk c lands at zigzag[c*CHUNK + i]; decode reads
// zigzag[chunk_id*CHUNK + i] with the chunk_id known at kernel entry, so no
// per-chunk offsets buffer is needed (the stride is the constant CHUNK).  The
// buffer is num_chunks*CHUNK elements; trailing slots of the final short chunk
// are pre-zeroed (default) and never read — keeping the entropy tail
// deterministic.  The channel is exposed via the rep's named_channels() and
// routed by the compress boundary loop exactly like FOR's `references` or
// Bitpack's `packed`.
// =====================================================================
void Walker::emit_zigzag(const ::codegen::jit::FusedTree& node, LaneInput in)
{
  const DtypeInfo* op_dt = lookup_dtype(in.elem_type);
  if (op_dt == nullptr) {
    throw RenderError("render: Zigzag op-local dtype '" + in.elem_type +
                      "' not in dtype table (ZigZag is defined for signed integers)");
  }

  // Unsigned counterpart + sign-bit shift width, derived from element size.
  const std::string utype = unsigned_counterpart(op_dt->elem_size);
  const int shift         = static_cast<int>(op_dt->elem_size) * 8 - 1;

  // ---- Transformer mode: rewrite read_expr inline, recurse, store nothing.
  if (!node.children.empty()) {
    if (node.children.size() != 1) {
      throw RenderError(
        "render: Zigzag transformer must have exactly one child named "
        "'zigzag' (got " +
        std::to_string(node.children.size()) + " children)");
    }
    auto zit = node.children.find("zigzag");
    if (zit == node.children.end()) {
      throw RenderError("render: Zigzag missing 'zigzag' child (got '" +
                        node.children.begin()->first + "' instead)");
    }
    // Consume a node id to keep the encode counter aligned with the
    // decode-side DFS-preorder ids (the child must be id+1), even though
    // ZigZag emits no params/buffers of its own.
    const std::int32_t id = take_id();
    const std::string v   = at_lane(in.read_expr, "(__LANE__)");
    body_ << "    // --- node " << id << ": Zigzag (inline transform, " << in.elem_type
          << ") ---\n";

    LaneInput out;
    out.kind      = LaneInput::Expr;
    out.elem_type = in.elem_type;
    out.read_expr = "(static_cast<" + in.elem_type + ">(" + "(static_cast<" + utype + ">(" + v +
                    ") << 1) ^ " + "static_cast<" + utype + ">((" + v + ") >> " +
                    std::to_string(shift) + ")))";
    out.length_expr = in.length_expr;
    emit_node(*zit->second, std::move(out));
    return;
  }

  // ---- Leaf mode: store the ZigZag-mapped stream to the `zigzag` channel.
  const std::int32_t id    = take_id();
  const std::string idstr  = std::to_string(id);
  const std::string p_data = "zigzag_" + idstr;
  const std::string zbase  = "zz_base_" + idstr;
  const std::string zlen   = "zz_len_" + idstr;

  add_param(in.elem_type + "*", p_data);
  const std::size_t nc = static_cast<std::size_t>(num_chunks_);
  add_buffer(id, "zigzag", op_dt->elem_size, nc * static_cast<std::size_t>(::codegen::kChunkSize));

  body_ << "    // --- node " << id << ": Zigzag (leaf store, " << in.elem_type << ") ---\n"
        << "    {\n"
        << "        const int64_t " << zbase << " = static_cast<int64_t>(chunk_id) *\n"
        << "                             static_cast<int64_t>(CHUNK);\n"
        << "        const int32_t " << zlen << " = static_cast<int32_t>(" << in.length_expr
        << ");\n"
        << "        for (int32_t i = tid; i < " << zlen << "; i += 128) {\n"
        << "            const " << in.elem_type << " _zv = (" << at_lane(in.read_expr, "i")
        << ");\n"
        << "            const " << utype << " _zu =\n"
        << "                (static_cast<" << utype << ">(_zv) << 1) ^\n"
        << "                static_cast<" << utype << ">(_zv >> " << shift << ");\n"
        << "            " << p_data << "[" << zbase << " + i] = static_cast<" << in.elem_type
        << ">(_zu);\n"
        << "        }\n"
        << "    }\n";
}

// =====================================================================
// emit_bitpack — LEAF.
//
// Bitpack is always the sink of the fused encode cluster.  Whatever
// chain of transformers preceded it (Delta, FOR, ...) has rewritten
// LaneInput.read_expr into a single C++ expression for "lane T's
// residual-input value"; we consume it here, run the reductions, and
// write the OverAllocate packed output.  Tail codecs (bitcomp/ANS)
// attach downstream to the `packed_<id>` channel.
//
// The kernel work per chunk:
//   1. Compute `bp<id>_len` from the LaneInput length expression.
//   2. Empty-chunk sentinel: header + one zero word in packed slot.
//   3. Per-lane v = (LANE < bp_len) ? read_expr@tid : ELEMENT_MAX
//      (inactive lanes contribute sentinel for the min reduction).
//   4. Block-wide signed min reduction -> chunk_min.
//   5. Per-lane residual = (int64) v - (int64) chunk_min; sentinel 0
//      for inactive lanes.
//   6. Block-wide max-residual reduction -> bits.
//   7. Single-thread header writes (chunk_min, chunk_count=bp_len,
//      chunk_bits, live_words).
//   8. Clear chunk's stride slot in packed (atomicOr correctness
//      pre-req), sync, then per-lane atomicOr pack at bit offset
//      LANE * bits (3-word straddle for 64-bit elements with
//      non-multiple-of-32 widths).
//
// Each Bitpack instance uses two __shared__ arrays for its
// reductions.  Naming includes node_id so two Bitpacks in the same
// tree (e.g. Rle.runs + Rle.values both Bitpack) won't collide.
// =====================================================================
void Walker::emit_bitpack(const ::codegen::jit::FusedTree& node, LaneInput in, bool use_smem)
{
  if (!node.children.empty()) {
    throw RenderError(
      "render: Bitpack must be a leaf (no children) — encode "
      "chain ends at Bitpack; tail codecs (bitcomp/ANS) attach "
      "to the `packed` channel downstream, not as fused children");
  }
  // Per-op dtype lookup — the LaneInput is the source of truth.  A
  // Bitpack under Rle's `runs` subtree sees in.elem_type=int32_t
  // even when the column dtype is int64_t.  Walker::dt_ is the root
  // column type and intentionally unused here.
  const DtypeInfo* op_dt = lookup_dtype(in.elem_type);
  if (op_dt == nullptr) {
    throw RenderError("render: Bitpack op-local dtype '" + in.elem_type + "' not in dtype table");
  }

  const std::int32_t id     = take_id();
  const std::string idstr   = std::to_string(id);
  const std::string p_min   = "chunk_min_" + idstr;
  const std::string p_cnt   = "chunk_count_" + idstr;
  const std::string p_bits  = "chunk_bits_" + idstr;
  const std::string p_pkd   = "packed_" + idstr;
  const std::string p_lws   = "lw_shards_" + idstr;  // sharded atomicAdd (16-shard DtoH)
  const std::string v_need3 = "need3_" + idstr;      // per-chunk: bits_v+31 > 64 possible
  const std::string sh_min  = "sh_min_" + idstr;
  const std::string sh_max  = "sh_max_" + idstr;
  const std::string bp_len  = "bp_len_" + idstr;
  const std::string v_var   = "v_" + idstr;
  const std::string cmin    = "cmin_" + idstr;
  const std::string resid   = "resid_" + idstr;
  const std::string max_r   = "max_resid_" + idstr;
  const std::string bits_v  = "bits_" + idstr;
  const std::string dst     = "dst_base_" + idstr;
  const std::string sw_var  = "stride_words_" + idstr;
  const std::string emax    = "elem_max_" + idstr;

  // Buffer params + specs (order MUST match the launcher's
  // cuLaunchKernel arg vector).
  add_param(in.elem_type + "*", p_min);
  add_param("int32_t*", p_cnt);
  add_param("uint8_t*", p_bits);
  add_param("uint32_t*", p_pkd);
  add_param("uint32_t*", p_lws);  // sharded live-word counter (16-shard)

  const std::size_t stride_words =
    static_cast<std::size_t>(::codegen::kChunkSize) * op_dt->elem_size / sizeof(std::uint32_t);
  const std::size_t nc = static_cast<std::size_t>(num_chunks_);
  // Sharded atomicAdd slab: kMaxBitsShards=16 shards × kStride=32 uint32s
  // (128-byte spacing keeps each shard on its own L2 cache line to avoid
  // serialisation).  Host post-kernel sums 16 entries.  Mirrors
  // block_bitpack's packed_counter from fused_block_primitives.cuh.
  constexpr std::size_t kMaxBitsShards = 16;
  constexpr std::size_t kShardStride   = 32;

  add_buffer(id, "chunk_min", op_dt->elem_size, nc);
  add_buffer(id, "chunk_count", sizeof(std::int32_t), nc);
  add_buffer(id, "chunk_bits", sizeof(std::uint8_t), nc);
  // When use_smem the kernel accumulates into a per-block shared slab and
  // then stores cleanly — no read-modify-write on the global buffer, so no
  // pre-zeroing is required.
  add_buffer(id,
             "packed",
             sizeof(std::uint32_t),
             nc * stride_words,
             /*no_pre_zero=*/use_smem);
  add_buffer(id, "lw_shards", sizeof(std::uint32_t), kMaxBitsShards * kShardStride);
  // Note: no per-chunk live_words buffer — live_packed_bytes is derived
  // from lw_shards (DtoH only 2 KB instead of num_chunks×4 B).
  // compact_in_place() reconstructs per-chunk offsets from chunk_bits+chunk_count.

  // Smem slab for use_smem path.
  // For runs (int32, bits ≤ 10):  322 words × 4 = 1288 bytes.
  // For values int32 (bits ≤ 32): 1026 words × 4 = 4 KB.
  // The smem path is only activated for int32, so use the dtype-derived bound:
  //   kChunkSize * elem_size / 4 + 2.
  const std::string smem_pkd = "smem_pkd_" + idstr;
  const int kSmemPkdWords    = static_cast<int>(::codegen::kChunkSize) *
                              static_cast<int>(op_dt->elem_size) /
                              static_cast<int>(sizeof(std::uint32_t)) +
                            2;

  body_ << "    // --- node " << id << ": Bitpack (cub@128, block_bitpack-mirror, " << in.elem_type
        << (use_smem ? ", smem-accumulate" : "") << ") ---\n"
        << "    do {\n"
        << "        constexpr int32_t " << sw_var << " = CHUNK * " << op_dt->elem_size << " / 4;\n"
        << "        constexpr " << in.elem_type << " " << emax << " = static_cast<" << in.elem_type
        << ">(" << op_dt->max_literal << ");\n"
        << "        uint32_t* " << dst << " = " << p_pkd
        << " + static_cast<int64_t>(chunk_id) * static_cast<int64_t>(" << sw_var << ");\n"
        << "        const int32_t " << bp_len << " = static_cast<int32_t>(" << in.length_expr
        << ");\n"
        << "        if (" << bp_len << " <= 0) {\n"
        << "            if (tid == 0) { " << p_min << "[chunk_id] = " << in.elem_type << "{0}; "
        << p_cnt << "[chunk_id] = 0; " << p_bits << "[chunk_id] = 1;\n"
        << "                atomicAdd(" << p_lws
        << " + (static_cast<uint32_t>(chunk_id) & 15u) * 32u, 1u); }\n"
        << "            break;\n"
        << "        }\n"
        // Pass 1: combined min+max in a single grid-stride loop.
        << "        " << in.elem_type << " _lmin = " << emax << ";\n"
        << "        " << in.elem_type << " _lmax = static_cast<" << in.elem_type << ">("
        << op_dt->min_literal << ");\n"
        << "        for (int32_t i = tid; i < " << bp_len << "; i += 128) {\n"
        << "            " << in.elem_type << " _vi = (" << at_lane(in.read_expr, "i") << ");\n"
        << "            if (_vi < _lmin) _lmin = _vi;\n"
        << "            if (_vi > _lmax) _lmax = _vi;\n"
        << "        }\n"
        << "        typedef cub::BlockReduce<" << in.elem_type << ", 128> BR_" << idstr << ";\n"
        << "        __shared__ typename BR_" << idstr << "::TempStorage br_ts_" << idstr << ";\n"
        << "        __shared__ " << in.elem_type << " " << sh_min << "_b;\n"
        << "        __shared__ " << in.elem_type << " " << sh_max << "_b;\n"
        << "        { " << in.elem_type << " _m = BR_" << idstr << "(br_ts_" << idstr
        << ").Reduce(_lmin, SimpaticoMin()); if (tid == 0) " << sh_min << "_b = _m; }\n"
        << "        __syncthreads();\n"
        << "        { " << in.elem_type << " _m = BR_" << idstr << "(br_ts_" << idstr
        << ").Reduce(_lmax, SimpaticoMax()); if (tid == 0) " << sh_max << "_b = _m; }\n"
        << "        __syncthreads();\n"
        << "        const " << in.elem_type << " " << cmin << " = " << sh_min << "_b;\n"
        << "        const uint64_t " << max_r << " = static_cast<uint64_t>(static_cast<int64_t>("
        << sh_max << "_b) - static_cast<int64_t>(" << cmin << "));\n"
        << "        const int32_t " << bits_v << " = simpatico_bit_width_u64("
        << "static_cast<unsigned long long>(" << max_r << "));\n"
        << "        if (tid == 0) {\n"
        << "            " << p_min << "[chunk_id] = " << cmin << ";\n"
        << "            " << p_cnt << "[chunk_id] = " << bp_len << ";\n"
        << "            " << p_bits << "[chunk_id] = static_cast<uint8_t>(" << bits_v << ");\n"
        << "            const uint32_t _pw = static_cast<uint32_t>((static_cast<int32_t>(" << bp_len
        << ") * " << bits_v << " + 31) / 32);\n"
        << "            atomicAdd(" << p_lws
        << " + (static_cast<uint32_t>(chunk_id) & 15u) * 32u, _pw);\n"
        << "        }\n";

  if (use_smem) {
    // Smem accumulation path: accumulate into per-block __shared__ slab,
    // then store to global.  Eliminates the cudaMemsetAsync pre-zero.
    body_ << "        __shared__ uint32_t " << smem_pkd << "[" << kSmemPkdWords
          << "];\n"
          // _nwords_live: actual packed words written to global (matches lw_shards count).
          // _nwords_slab: smem slab to zero (+ 2 guard words so the last element's
          //   atomicOr can safely write _tw+1 without smem OOB).
          << "        const int32_t _nwords_live_" << idstr << " = (static_cast<int32_t>(" << bp_len
          << ") * " << bits_v << " + 31) / 32;\n"
          << "        const int32_t _nwords_slab_" << idstr << " = _nwords_live_" << idstr
          << " + 2;\n"
          << "        for (int32_t _w = tid; _w < _nwords_slab_" << idstr << "; _w += 128)\n"
          << "            " << smem_pkd << "[_w] = 0u;\n"
          << "        __syncthreads();\n"
          // Pack into smem (SM-local atomicOr, no L2 cache contention).
          << "        for (int32_t i = tid; i < " << bp_len << "; i += 128) {\n"
          << "            uint64_t _rv = static_cast<uint64_t>(static_cast<int64_t>("
          << at_lane(in.read_expr, "i") << ") - static_cast<int64_t>(" << cmin << "));\n"
          << "            const int32_t _ibits = i * static_cast<int32_t>(" << bits_v << ");\n"
          << "            int32_t _tw = _ibits >> 5;\n"
          << "            int32_t _tb = _ibits & 31;\n"
          << "            int32_t _rem = " << bits_v << ", _sv = 0;\n"
          << "            while (_rem > 0) {\n"
          << "                const int32_t _chunk = (32 - _tb < _rem) ? (32 - _tb) : _rem;\n"
          << "                const uint32_t _mask = (_chunk == 32) ? 0xFFFFFFFFu : ((1u << "
             "_chunk) - 1u);\n"
          << "                atomicOr(&" << smem_pkd << "[_tw],\n"
          << "                    static_cast<uint32_t>((_rv >> _sv) & _mask) << _tb);\n"
          << "                _sv += _chunk; _rem -= _chunk; _tw++; _tb = 0;\n"
          << "            }\n"
          << "        }\n"
          << "        __syncthreads();\n"
          // Coalesced store from smem to global — live words only (guard words stay in smem).
          << "        for (int32_t _w = tid; _w < _nwords_live_" << idstr << "; _w += 128)\n"
          << "            " << dst << "[_w] = " << smem_pkd << "[_w];\n";
  } else {
    // Global atomicOr path (default): requires pre-zeroed packed buffer.
    // Pass 2: while-loop pack with one shared 32-bit multiply per element.
    //
    // Direction 1 kept: i * bits_v ≤ CHUNK*64 = 65536 fits in int32.
    // One 32-bit IMAD (_ibits) shared for both _tw and _tb replaces
    // the original two independent 64-bit multiplies (~4 vs ~8 cycles).
    body_ << "        for (int32_t i = tid; i < " << bp_len << "; i += 128) {\n"
          << "            uint64_t _rv = static_cast<uint64_t>(static_cast<int64_t>("
          << at_lane(in.read_expr, "i") << ") - static_cast<int64_t>(" << cmin << "));\n"
          << "            const int32_t _ibits = i * static_cast<int32_t>(" << bits_v << ");\n"
          << "            int32_t _tw = _ibits >> 5;\n"
          << "            int32_t _tb = _ibits & 31;\n"
          << "            int32_t _rem = " << bits_v << ", _sv = 0;\n"
          << "            while (_rem > 0) {\n"
          << "                const int32_t _chunk = (32 - _tb < _rem) ? (32 - _tb) : _rem;\n"
          << "                const uint32_t _mask = (_chunk == 32) ? 0xFFFFFFFFu\n"
          << "                                    : ((1u << _chunk) - 1u);\n"
          << "                atomicOr(&" << dst << "[_tw],\n"
          << "                    static_cast<uint32_t>((_rv >> _sv) & _mask) << _tb);\n"
          << "                _sv += _chunk; _rem -= _chunk; _tw++; _tb = 0;\n"
          << "            }\n"
          << "        }\n";
  }
  body_ << "    } while (0);\n";
}

}  // namespace

// =====================================================================
// Public entry point.
// =====================================================================
EncodeKernelSpec render(const ::codegen::jit::FusedTree& tree,
                        const std::string& element_dtype,
                        std::int32_t num_chunks)
{
  if (element_dtype.empty()) { throw std::invalid_argument("render: element_dtype is empty"); }
  if (num_chunks < 1) {
    throw std::invalid_argument("render: num_chunks must be >= 1 (got " +
                                std::to_string(num_chunks) + ")");
  }
  const DtypeInfo* dt = lookup_dtype(element_dtype);
  if (dt == nullptr) {
    throw RenderError("render: unsupported element_dtype '" + element_dtype +
                      "'.  Supported: int32_t, int64_t");
  }

  Walker w(element_dtype, *dt, num_chunks);
  return w.build(tree);
}

}  // namespace codegen::encode::jit
