// SPDX-License-Identifier: Apache-2.0
//
// C++-native .hpln write/read for compressed_table.
// See compressed_table_io.hpp for the on-disk layout.

#include "api/compressed_table_io.hpp"

#include "codegen/plan/chunk_subset.hpp"
#include "codegen/plan/operator_registry.hpp"
#include "codegen/plan/plan_interpreter.hpp"
#include "codegen/plan/representation.hpp"

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/traits.hpp>

#include <rmm/device_buffer.hpp>
#include <rmm/mr/per_device_resource.hpp>

#include <cuda_runtime.h>
#include <nvtx3/nvtx3.hpp>

#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace simpatico {
namespace {

// ---------------------------------------------------------------------------
// LE binary helpers — std::bit_cast (C++20); asserts little-endian platform.
// ---------------------------------------------------------------------------
static_assert(std::endian::native == std::endian::little, "LE-only serialization");

template <typename T>
static void push_le(std::vector<std::uint8_t>& v, T x)
{
  auto b = std::bit_cast<std::array<std::uint8_t, sizeof(T)>>(x);
  v.insert(v.end(), b.begin(), b.end());
}

static void push_str16(std::vector<std::uint8_t>& v, std::string const& s)
{
  push_le(v, static_cast<std::uint16_t>(s.size()));
  v.insert(v.end(), s.begin(), s.end());
}

struct Reader {
  const std::uint8_t* p = nullptr;
  std::size_t rem       = 0;

  bool read(void* dst, std::size_t n)
  {
    if (n > rem) return false;
    std::memcpy(dst, p, n);
    p += n;
    rem -= n;
    return true;
  }

  template <typename T>
  bool read_le(T& x)
  {
    std::array<std::uint8_t, sizeof(T)> b;
    if (!read(b.data(), sizeof(T))) return false;
    x = std::bit_cast<T>(b);
    return true;
  }

  bool read_str16(std::string& s)
  {
    std::uint16_t len;
    if (!read_le(len)) return false;
    if (len > rem) return false;
    s.assign(reinterpret_cast<const char*>(p), len);
    p += len;
    rem -= len;
    return true;
  }
};

// ---------------------------------------------------------------------------
// leaf_meta_v ↔ binary encoding
// meta_kind tags: 0=none 1=alp_rd 2=ans 3=bitcomp 4=cascaded 5=snappy 6=lz4 7=deflate
// ---------------------------------------------------------------------------
enum : std::uint8_t {
  META_NONE     = 0,
  META_ALP_RD   = 1,
  META_ANS      = 2,
  META_BITCOMP  = 3,
  META_CASCADED = 4,
  META_SNAPPY   = 5,
  META_LZ4      = 6,
  META_DEFLATE  = 7,
};

static void push_meta(std::vector<std::uint8_t>& v, leaf_meta_v const& m)
{
  struct Visitor {
    std::vector<std::uint8_t>& v;
    void operator()(leaf_meta::none const&) { push_le(v, META_NONE); }
    void operator()(leaf_meta::alp_rd const& a)
    {
      push_le(v, META_ALP_RD);
      push_le(v, a.right_bw);
    }
    void operator()(leaf_meta::ans const& a)
    {
      push_le(v, META_ANS);
      push_le(v, a.uncompressed_size);
      push_le(v, a.original_type_id);
    }
    void operator()(leaf_meta::bitcomp const& b)
    {
      push_le(v, META_BITCOMP);
      push_le(v, b.uncompressed_size);
      push_le(v, b.original_type_id);
      push_le(v, b.algorithm);
    }
    void operator()(leaf_meta::nvcomp_cascaded const& c)
    {
      push_le(v, META_CASCADED);
      push_le(v, c.uncompressed_size);
      push_le(v, c.original_type_id);
      push_le(v, c.num_deltas);
      push_le(v, c.num_RLEs);
      push_le(v, c.use_bp);
    }
    void operator()(leaf_meta::snappy const& s)
    {
      push_le(v, META_SNAPPY);
      push_le(v, s.uncompressed_size);
      push_le(v, s.original_type_id);
    }
    void operator()(leaf_meta::lz4 const& l)
    {
      push_le(v, META_LZ4);
      push_le(v, l.uncompressed_size);
      push_le(v, l.original_type_id);
    }
    void operator()(leaf_meta::deflate const& d)
    {
      push_le(v, META_DEFLATE);
      push_le(v, d.uncompressed_size);
      push_le(v, d.original_type_id);
    }
  };
  std::visit(Visitor{v}, m);
}

static bool read_meta(Reader& r, leaf_meta_v& out)
{
  std::uint8_t mk;
  if (!r.read_le(mk)) return false;
  switch (mk) {
    case META_NONE: out = leaf_meta::none{}; return true;
    case META_ALP_RD: {
      std::uint8_t rbw;
      if (!r.read_le(rbw)) return false;
      out = leaf_meta::alp_rd{rbw};
      return true;
    }
    case META_ANS: {
      leaf_meta::ans a;
      if (!r.read_le(a.uncompressed_size) || !r.read_le(a.original_type_id)) return false;
      out = a;
      return true;
    }
    case META_BITCOMP: {
      leaf_meta::bitcomp b;
      if (!r.read_le(b.uncompressed_size) || !r.read_le(b.original_type_id) ||
          !r.read_le(b.algorithm))
        return false;
      out = b;
      return true;
    }
    case META_CASCADED: {
      std::uint64_t us;
      std::int32_t ti, num_deltas, nr, bp;
      if (!r.read_le(us) || !r.read_le(ti) || !r.read_le(num_deltas) || !r.read_le(nr) ||
          !r.read_le(bp))
        return false;
      out = leaf_meta::nvcomp_cascaded{us, ti, num_deltas, nr, bp};
      return true;
    }
    case META_SNAPPY: {
      std::uint64_t us;
      std::int32_t ti;
      if (!r.read_le(us) || !r.read_le(ti)) return false;
      out = leaf_meta::snappy{us, ti};
      return true;
    }
    case META_LZ4: {
      std::uint64_t us;
      std::int32_t ti;
      if (!r.read_le(us) || !r.read_le(ti)) return false;
      out = leaf_meta::lz4{us, ti};
      return true;
    }
    case META_DEFLATE: {
      std::uint64_t us;
      std::int32_t ti;
      if (!r.read_le(us) || !r.read_le(ti)) return false;
      out = leaf_meta::deflate{us, ti};
      return true;
    }
    default: return false;
  }
}

// ---------------------------------------------------------------------------
// rep_from_leaf_desc: per-kind rep factory
// ---------------------------------------------------------------------------
// `fill(i, dst_device, size, stream)` copies buffer i's `size` bytes into the
// pre-allocated device pointer `dst_device`. It abstracts the byte source so the
// same reconstruction serves both the file reader (contiguous host payload) and
// the in-memory reader (a caller-owned, possibly multi-block pinned payload).
using leaf_buffer_fill =
  std::function<void(std::size_t i, void* dst_device, std::size_t size, rmm::cuda_stream_view)>;

// Bytes make_col allocates from leaf_mr for one enumerated buffer: the leaf column
// is sized to the DECODED element count (num_rows) times the element width — NOT the
// compressed byte count. Every buffer that a decode kernel over-reads carries the
// reachable tail inside its own num_rows, so this stays a plain product. Must stay in
// lockstep with make_col below; build_compressed_table_header records this so a slab
// caller reserves the right slice size (a decode kernel reaches the whole column).
static std::uint64_t leaf_alloc_bytes(std::uint8_t type_tag, std::uint64_t num_rows)
{
  cudf::data_type const dt = tag_to_dtype(type_tag);
  auto const width         = static_cast<std::uint64_t>(cudf::size_of(dt));
  return num_rows * width;
}

static std::unique_ptr<compressed_representation> rep_from_leaf_desc(
  leaf_desc const& ld,
  leaf_buffer_fill const& fill,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr,
  rmm::device_async_resource_ref leaf_mr,
  std::string* err)
{
  auto const& bufs = ld.buffers;

  // Helper: allocate a device column (from @p leaf_mr — the enumerated leaf
  // buffers, which a caller may want placed in a dedicated arena/slab) and fill it
  // from the i-th payload buffer. Any codec decode scratch allocated below the
  // leaf level goes through @p mr instead, so leaf placement stays exact.
  auto make_col = [&](std::size_t i) -> std::unique_ptr<cudf::column> {
    auto const& bd     = bufs[i];
    cudf::data_type dt = tag_to_dtype(bd.type_tag);
    auto col           = cudf::make_numeric_column(dt,
                                         static_cast<cudf::size_type>(bd.num_rows),
                                         cudf::mask_state::UNALLOCATED,
                                         stream,
                                         leaf_mr);
    if (bd.size_bytes > 0) {
      fill(i, col->mutable_view().head<void>(), static_cast<std::size_t>(bd.size_bytes), stream);
    }
    return col;
  };

  // Delta, Rle, For, Zigzag and Bitpack are codegen-only operators: encode always
  // produces a JIT codegen_fused_representation carrying the fused region's
  // manifest buffers, so their leaves are reconstructed unconditionally as a fused
  // rep. num_rows -- the node's own output length, round-tripped in leaf_desc --
  // drives the codegen decode grid (for Bitpack it is not derivable from any
  // buffer, since chunk_min/count/bits are per-chunk and packed is words).
  const cudf::size_type node_rows = static_cast<cudf::size_type>(ld.num_rows);
  auto make_fused_rep             = [&](OpId op_id) {
    auto rep =
      std::make_unique<codegen_fused_representation>(op_id, tag_to_dtype(ld.type_tag), node_rows);
    for (std::size_t i = 0; i < bufs.size(); ++i) {
      rep->buffers.emplace_back(bufs[i].name, make_col(i));
    }
    return std::unique_ptr<compressed_representation>(std::move(rep));
  };

  if (ld.kind == OpId::Delta || ld.kind == OpId::Rle || ld.kind == OpId::For ||
      ld.kind == OpId::Zigzag || ld.kind == OpId::Bitpack) {
    return make_fused_rep(ld.kind);
  }
  if (ld.kind == OpId::Identity) {
    bool is_fused = !(bufs.size() == 1 && bufs[0].name == "data");
    if (is_fused) return make_fused_rep(ld.kind);
  }

  // All other kinds (and non-fused identity): route through reconstruct_representation.
  if (ld.kind == OpId::Unknown) {
    if (err) *err = "rep_from_leaf_desc: unknown leaf kind in file";
    return nullptr;
  }
  std::string_view cname = op_info(ld.kind).name;
  if (cname.empty()) {
    if (err)
      *err = "rep_from_leaf_desc: unsupported kind " + std::to_string(static_cast<int>(ld.kind));
    return nullptr;
  }
  std::vector<std::string> names;
  std::vector<std::unique_ptr<cudf::column>> cols;
  names.reserve(bufs.size());
  cols.reserve(bufs.size());
  for (std::size_t i = 0; i < bufs.size(); ++i) {
    names.push_back(bufs[i].name);
    cols.push_back(make_col(i));
  }
  return reconstruct_representation(
    std::string(cname), names, std::move(cols), stream, mr, err, ld.meta);
}

// Payload layout identifier. A reader accepts only an exact match (parse_hpln_header),
// so this must advance whenever the meaning of the serialized bytes changes — including
// changes that keep every field in place but reinterpret one, which no structural check
// downstream could catch.
//
// 11: a Bitpack "packed" buffer counts its decode gather guard words in num_rows, so the
//     stored word count is what the reader allocates and the guard needs no read-side
//     reconstruction (see compact_bitpack_packed).
static constexpr std::uint8_t kVersion = 11;

// Serialize one node's structure (op, bitjoin params, edges, output names).
// Other ops carry their params in the op name, so only bitjoin needs attrs.
static void push_node(std::vector<std::uint8_t>& hdr, PlanNode const& node)
{
  push_str16(hdr, node.op);
  if (node.attrs.bitjoin.has_value()) {
    auto const& bj = *node.attrs.bitjoin;
    // Flags are read back as std::uint8_t — write them at that exact width.
    // (push_le deduces T from the argument, so a bare `1`/`0` would emit a
    // 4-byte int and desync the reader.)
    push_le(hdr, std::uint8_t{1});
    push_le(hdr, dtype_to_tag(bj.output_type));
    push_le(hdr, static_cast<std::uint16_t>(bj.inputs.size()));
    for (auto const& in : bj.inputs) {
      push_le(hdr, in.node);
      push_str16(hdr, in.channel);
      push_le(hdr, static_cast<std::uint8_t>(in.range.has_value() ? 1 : 0));
      if (in.range.has_value()) {
        push_le(hdr, in.range->first);
        push_le(hdr, in.range->second);
      }
    }
  } else {
    push_le(hdr, std::uint8_t{0});
  }
  push_le(hdr, static_cast<std::uint16_t>(node.children.size()));
  for (auto const& e : node.children) {
    push_str16(hdr, e.channel);
    push_le(hdr, e.child);
  }
  push_le(hdr, static_cast<std::uint16_t>(node.output_names.size()));
  for (auto const& n : node.output_names)
    push_str16(hdr, n);
}

// Inverse of push_node. output_paths mirror output_names as the node-local
// channel key; bitjoin input arity/ranges live on the bitjoin attrs, and each
// node's structural input_sources are rebuilt by compute_input_sources after
// the whole tree is read.
static bool read_node(Reader& r, PlanNode& node)
{
  if (!r.read_str16(node.op)) return false;
  std::uint8_t is_bitjoin;
  if (!r.read_le(is_bitjoin)) return false;
  if (is_bitjoin) {
    bitjoin_attrs bj;
    std::uint8_t out_tag;
    if (!r.read_le(out_tag)) return false;
    bj.output_type = tag_to_dtype(out_tag);
    std::uint16_t num_inputs;
    if (!r.read_le(num_inputs)) return false;
    bj.inputs.resize(num_inputs);
    for (auto& in : bj.inputs) {
      if (!r.read_le(in.node) || !r.read_str16(in.channel)) return false;
      std::uint8_t has_range;
      if (!r.read_le(has_range)) return false;
      if (has_range) {
        std::uint32_t hi, lo;
        if (!r.read_le(hi) || !r.read_le(lo)) return false;
        in.range = bit_range{hi, lo};
      }
    }
    node.attrs.bitjoin = std::move(bj);
  }
  std::uint16_t ne;
  if (!r.read_le(ne)) return false;
  node.children.resize(ne);
  for (auto& e : node.children) {
    if (!r.read_str16(e.channel) || !r.read_le(e.child)) return false;
  }
  std::uint16_t no;
  if (!r.read_le(no)) return false;
  node.output_names.resize(no);
  node.output_paths.resize(no);
  for (std::uint16_t i = 0; i < no; ++i) {
    if (!r.read_str16(node.output_names[i])) return false;
    node.output_paths[i] = node.output_names[i];
  }
  return true;
}

// Per-column parsed structure (header only; buffer bytes live in the payload
// region and are pulled in separately during reconstruction).
struct ColRecord {
  std::string name;
  std::uint8_t dtype_tag = 0;
  std::int32_t scale     = 0;  // fixed-point scale for the column dtype (0 otherwise)
  std::int64_t num_rows  = 0;
  PlanTree tree;
  std::vector<leaf_desc> leaf_descs;
  std::vector<std::vector<std::uint64_t>> buf_offsets;  // [leaf][buffer] -> payload offset
};

// Byte offsets, relative to where the Reader started, of the header fields that a chunk subset
// has to restate. Recorded during parsing so build_chunk_subset_header can PATCH a copy of the
// original header rather than re-emit one from the parsed records: a second writer would be free
// to drift from build_compressed_table_header and the divergence would only show up as wrong
// query results much later. Every field involved is fixed-width at a computable offset, so
// patching keeps the output structurally identical to a written header by construction.
struct HeaderFieldOffsets {
  struct Buffer {
    std::uint64_t size_bytes_at     = 0;  // uint64
    std::uint64_t payload_offset_at = 0;  // uint64
  };
  struct Leaf {
    std::uint64_t num_rows_at = 0;  // uint64
    std::vector<Buffer> buffers;
  };
  struct Column {
    std::uint64_t num_rows_at = 0;  // int64
    std::vector<Leaf> leaves;
  };
  std::vector<Column> columns;
};

// Parse the .hpln header from `r` into one ColRecord per column. On success
// `r` is left pointing just past the header (i.e. at the payload region for the
// concatenated file layout). Returns false and sets *err on any structural error.
static bool parse_hpln_header(Reader& r,
                              std::vector<ColRecord>& out,
                              std::string* err,
                              HeaderFieldOffsets* field_offsets = nullptr)
{
  nvtx3::scoped_range nvtx_range{"simpatico::io::parse_header"};
  // Field offsets are recorded relative to where this Reader started, which is where the header
  // begins in every caller, so they index straight into a copy of the header bytes.
  std::uint8_t const* const header_base = r.p;
  auto const field_at = [&] { return static_cast<std::uint64_t>(r.p - header_base); };
  auto bad            = [&](std::string const& m) {
    if (err) *err = m;
    return false;
  };

  std::uint8_t magic[4];
  if (!r.read(magic, 4)) return bad("truncated header");
  if (magic[0] != 'H' || magic[1] != 'P' || magic[2] != 'L' || magic[3] != 'N')
    return bad("not a HPLN file");
  std::uint8_t ver;
  if (!r.read_le(ver)) return bad("truncated header");
  if (ver != kVersion)
    return bad("unsupported version " + std::to_string(ver) + " (expected " +
               std::to_string(kVersion) + ")");

  std::uint16_t num_cols;
  if (!r.read_le(num_cols)) return bad("truncated header");
  out.clear();
  out.resize(num_cols);
  if (field_offsets) {
    field_offsets->columns.clear();
    field_offsets->columns.resize(num_cols);
  }

  for (std::uint16_t ci = 0; ci < num_cols; ++ci) {
    auto& cr = out[ci];
    if (!r.read_str16(cr.name)) return bad("truncated col name");
    if (!r.read_le(cr.dtype_tag)) return bad("truncated col dtype");
    if (!r.read_le(cr.scale)) return bad("truncated col scale");
    if (field_offsets) field_offsets->columns[ci].num_rows_at = field_at();
    if (!r.read_le(cr.num_rows)) return bad("truncated col num_rows");

    std::uint16_t nn;
    if (!r.read_le(nn)) return bad("truncated num_nodes");
    cr.tree.nodes.resize(nn);
    for (std::uint16_t ni = 0; ni < nn; ++ni) {
      if (!read_node(r, cr.tree.nodes[ni])) return bad("truncated plan node");
    }

    std::uint16_t nl;
    if (!r.read_le(nl)) return bad("truncated num_leaves");
    cr.leaf_descs.resize(nl);
    cr.buf_offsets.resize(nl);
    if (field_offsets) field_offsets->columns[ci].leaves.resize(nl);

    for (std::uint16_t li = 0; li < nl; ++li) {
      auto& ld = cr.leaf_descs[li];
      if (!r.read_le(ld.node_index)) return bad("truncated leaf node_index");
      if (!r.read_le(ld.slot)) return bad("truncated leaf slot");
      std::uint8_t k;
      if (!r.read_le(k)) return bad("truncated leaf kind");
      ld.kind = static_cast<OpId>(k);
      if (!r.read_le(ld.type_tag)) return bad("truncated leaf type_tag");
      if (field_offsets) field_offsets->columns[ci].leaves[li].num_rows_at = field_at();
      if (!r.read_le(ld.num_rows)) return bad("truncated leaf num_rows");
      if (!read_meta(r, ld.meta)) return bad("truncated/unknown leaf meta");

      std::uint8_t nb;
      if (!r.read_le(nb)) return bad("truncated num_bufs");
      ld.buffers.resize(nb);
      cr.buf_offsets[li].resize(nb);
      if (field_offsets) field_offsets->columns[ci].leaves[li].buffers.resize(nb);

      for (std::uint8_t bi = 0; bi < nb; ++bi) {
        auto& bd = ld.buffers[bi];
        if (!r.read_str16(bd.name)) return bad("truncated buf name");
        if (!r.read_le(bd.type_tag)) return bad("truncated buf type_tag");
        if (field_offsets) {
          field_offsets->columns[ci].leaves[li].buffers[bi].size_bytes_at = field_at();
        }
        if (!r.read_le(bd.size_bytes)) return bad("truncated buf size_bytes");
        if (field_offsets) {
          field_offsets->columns[ci].leaves[li].buffers[bi].payload_offset_at = field_at();
        }
        std::uint64_t poff;
        if (!r.read_le(poff)) return bad("truncated buf payload_offset");
        cr.buf_offsets[li][bi] = poff;
        bd.num_rows =
          (bd.size_bytes > 0 && bd.type_tag < 255)
            ? bd.size_bytes / static_cast<std::uint64_t>(cudf::size_of(tag_to_dtype(bd.type_tag)))
            : 0;
      }
    }
  }
  return true;
}

// Reconstruct a compressed_table from parsed column records, pulling each leaf
// buffer's bytes into device memory via `fetch(offset, size, dst_device, stream)`.
// `recs` is consumed (plan trees are moved into the result).
static compressed_table reconstruct_from_records(std::vector<ColRecord>& recs,
                                                 payload_fetch_fn const& fetch,
                                                 rmm::cuda_stream_view stream,
                                                 rmm::device_async_resource_ref mr,
                                                 rmm::device_async_resource_ref leaf_mr,
                                                 std::string* err)
{
  nvtx3::scoped_range nvtx_range{"simpatico::io::fetch_payload"};
  auto fail = [&](std::string const& m) -> compressed_table {
    if (err) *err = m;
    return {};
  };

  compressed_table result;
  result.columns.resize(recs.size());

  for (std::size_t ci = 0; ci < recs.size(); ++ci) {
    auto& cr      = recs[ci];
    auto& out_col = result.columns[ci];

    if (!cr.name.empty()) out_col.name = cr.name;
    auto const dtype_id = tag_to_dtype(cr.dtype_tag).id();
    out_col.dtype = (dtype_id == cudf::type_id::DECIMAL32 || dtype_id == cudf::type_id::DECIMAL64 ||
                     dtype_id == cudf::type_id::DECIMAL128)
                      ? cudf::data_type{dtype_id, cr.scale}
                      : cudf::data_type{dtype_id};
    out_col.num_rows = cr.num_rows;

    if (cr.tree.nodes.empty()) continue;  // column stored without a plan

    auto plan_tree = std::make_unique<PlanTree>();
    *plan_tree     = std::move(cr.tree);
    auto& nodes    = plan_tree->nodes;

    for (std::size_t li = 0; li < cr.leaf_descs.size(); ++li) {
      auto const& ld    = cr.leaf_descs[li];
      auto const& boffs = cr.buf_offsets[li];
      if (ld.node_index >= nodes.size())
        return fail("leaf node_index out of range in col " + std::to_string(ci));

      auto fill = [&](std::size_t bi, void* dst, std::size_t sz, rmm::cuda_stream_view s) {
        fetch(boffs[bi], sz, dst, s);
      };

      std::string rep_err;
      auto rep = rep_from_leaf_desc(ld, fill, stream, mr, leaf_mr, &rep_err);
      if (!rep) return fail("rep_from_leaf_desc (col " + std::to_string(ci) + "): " + rep_err);

      PlanNode& node = nodes[ld.node_index];
      if (ld.slot == kSelfRepSlot) {
        node.meta = rep->describe_meta();
        node.rep  = std::move(rep);
      } else if (ld.slot >= 0 && static_cast<std::size_t>(ld.slot) < node.output_paths.size()) {
        node.channels.emplace(node.output_paths[ld.slot], std::move(rep));
      } else {
        return fail("leaf slot out of range in col " + std::to_string(ci));
      }
    }

    compute_input_sources(*plan_tree);
    out_col.plan_tree = std::move(plan_tree);
  }

  return result;
}

}  // anonymous namespace

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

namespace {

constexpr std::uint16_t kChunkDirectoryVersion = 1;

}  // namespace

std::vector<std::uint8_t> pack_hpln_chunk_directory(std::span<const hpln_chunk_ref> chunks)
{
  std::vector<std::uint8_t> out;
  push_le(out, kChunkDirectoryVersion);
  push_le(out, static_cast<std::uint32_t>(chunks.size()));
  for (auto const& c : chunks) {
    push_le(out, c.header_offset);
    push_le(out, c.header_bytes);
    push_le(out, c.payload_offset);
    push_le(out, c.payload_bytes);
    push_le(out, c.num_rows);
  }
  return out;
}

std::string unpack_hpln_chunk_directory(std::span<const std::uint8_t> bytes,
                                        std::vector<hpln_chunk_ref>& out)
{
  out.clear();
  constexpr std::size_t kEntryBytes = 5 * sizeof(std::uint64_t);
  if (bytes.size() < 6) return "chunk directory: truncated preamble";
  std::uint16_t version = 0;
  std::uint32_t n       = 0;
  std::memcpy(&version, bytes.data(), sizeof(version));
  std::memcpy(&n, bytes.data() + 2, sizeof(n));
  if (version != kChunkDirectoryVersion) {
    return "chunk directory: unsupported version " + std::to_string(version);
  }
  if (bytes.size() < 6 + static_cast<std::size_t>(n) * kEntryBytes) {
    return "chunk directory: truncated entry table";
  }
  out.resize(n);
  for (std::uint32_t i = 0; i < n; ++i) {
    auto const* p = bytes.data() + 6 + static_cast<std::size_t>(i) * kEntryBytes;
    std::memcpy(&out[i].header_offset, p, 8);
    std::memcpy(&out[i].header_bytes, p + 8, 8);
    std::memcpy(&out[i].payload_offset, p + 16, 8);
    std::memcpy(&out[i].payload_bytes, p + 24, 8);
    std::memcpy(&out[i].num_rows, p + 32, 8);
  }
  return {};
}

std::string write_compressed_tables(std::span<compressed_table const* const> tables,
                                    std::string const& path,
                                    rmm::cuda_stream_view stream,
                                    std::span<const hpln_extra_segment> extra)
{
  nvtx3::scoped_range nvtx_range{"simpatico::io::write_table[file]"};
  if (tables.empty()) return "write_compressed_tables: no chunks to write";

  // Build every chunk's header first. They are written as one contiguous region ahead of any
  // payload, so all of them have to exist before the first byte goes out.
  std::vector<std::vector<std::uint8_t>> headers(tables.size());
  std::vector<std::vector<payload_buffer_ref>> buffers(tables.size());
  std::vector<std::uint64_t> payload_sizes(tables.size(), 0);
  for (std::size_t i = 0; i < tables.size(); ++i) {
    if (tables[i] == nullptr)
      return "write_compressed_tables: null chunk at index " + std::to_string(i);
    std::string err =
      build_compressed_table_header(*tables[i], headers[i], buffers[i], payload_sizes[i], stream);
    if (!err.empty()) return err;
  }

  std::ofstream f(path, std::ios::binary | std::ios::trunc);
  if (!f) return "failed to open '" + path + "' for writing";

  std::vector<hpln_chunk_ref> dir(tables.size());
  std::uint64_t at = 0;
  for (std::size_t i = 0; i < tables.size(); ++i) {
    dir[i].header_offset = at;
    dir[i].header_bytes  = headers[i].size();
    dir[i].num_rows      = tables[i]->num_rows();
    f.write(reinterpret_cast<const char*>(headers[i].data()),
            static_cast<std::streamsize>(headers[i].size()));
    at += headers[i].size();
  }
  auto const header_region_bytes = at;

  // One chunk's payload is gathered, written and released before the next is staged: a file of N
  // chunks would otherwise need every chunk's decompressed-to-host bytes resident at once.
  for (std::size_t i = 0; i < tables.size(); ++i) {
    std::vector<std::uint8_t> payload(static_cast<std::size_t>(payload_sizes[i]));
    for (auto const& b : buffers[i]) {
      if (b.size_bytes > 0 && b.device_ptr) {
        cudaMemcpyAsync(payload.data() + b.offset,
                        b.device_ptr,
                        static_cast<std::size_t>(b.size_bytes),
                        cudaMemcpyDeviceToHost,
                        stream.value());
      }
    }
    stream.synchronize();  // D→H copies must complete before the file write
    dir[i].payload_offset = at;
    dir[i].payload_bytes  = payload_sizes[i];
    f.write(reinterpret_cast<const char*>(payload.data()),
            static_cast<std::streamsize>(payload.size()));
    at += payload_sizes[i];
  }

  // Segment table. header and payload are always present and always first, and for a multi-chunk
  // file they span the whole of their respective regions -- so a reader that only wants metadata
  // still fetches exactly one range, and the directory subdivides it.
  std::vector<hpln_segment_ref> segs;
  segs.push_back({hpln_segment::header, 0, header_region_bytes});
  segs.push_back({hpln_segment::payload, header_region_bytes, at - header_region_bytes});
  auto const dir_bytes = pack_hpln_chunk_directory(dir);
  f.write(reinterpret_cast<const char*>(dir_bytes.data()),
          static_cast<std::streamsize>(dir_bytes.size()));
  segs.push_back({hpln_segment::chunk_directory, at, dir_bytes.size()});
  at += dir_bytes.size();
  for (auto const& e : extra) {
    if (e.bytes.empty()) continue;
    f.write(reinterpret_cast<const char*>(e.bytes.data()),
            static_cast<std::streamsize>(e.bytes.size()));
    segs.push_back({e.kind, at, e.bytes.size()});
    at += e.bytes.size();
  }

  std::vector<std::uint8_t> ps;
  push_le(ps, static_cast<std::uint16_t>(segs.size()));
  for (auto const& sg : segs) {
    push_le(ps, static_cast<std::uint16_t>(sg.kind));
    push_le(ps, sg.offset);
    push_le(ps, sg.bytes);
  }
  f.write(reinterpret_cast<const char*>(ps.data()), static_cast<std::streamsize>(ps.size()));

  // Fixed 16-byte trailer, magic LAST so a reader validates by reading the tail.
  std::vector<std::uint8_t> tr;
  push_le(tr, at);                                     // postscript offset (u64)
  push_le(tr, static_cast<std::uint16_t>(ps.size()));  // postscript length (u16)
  push_le(tr, kHplnFileVersion);                       // version (u16)
  tr.push_back('H');
  tr.push_back('P');
  tr.push_back('L');
  tr.push_back('N');
  f.write(reinterpret_cast<const char*>(tr.data()), static_cast<std::streamsize>(tr.size()));
  if (!f) return "write error on '" + path + "'";
  return {};
}

std::string write_compressed_table(compressed_table const& table,
                                   std::string const& path,
                                   rmm::cuda_stream_view stream,
                                   std::span<const hpln_extra_segment> extra)
{
  compressed_table const* one = &table;
  return write_compressed_tables({&one, 1}, path, stream, extra);
}

std::string read_hpln_postscript(std::span<const std::uint8_t> tail,
                                 std::uint64_t file_size,
                                 std::vector<hpln_segment_ref>& out,
                                 std::uint64_t* need_bytes)
{
  out.clear();
  if (need_bytes) *need_bytes = kHplnTrailerBytes;
  if (tail.size() < kHplnTrailerBytes || file_size < kHplnTrailerBytes) {
    return "read_hpln_postscript: need at least the trailer";
  }
  auto const* t = tail.data() + tail.size() - kHplnTrailerBytes;
  if (t[12] != 'H' || t[13] != 'P' || t[14] != 'L' || t[15] != 'N') {
    return "read_hpln_postscript: no trailer magic (a pre-v7 file: parse the header from the "
           "front)";
  }
  std::uint64_t ps_off = 0;
  std::memcpy(&ps_off, t, sizeof(ps_off));
  std::uint16_t ps_len = 0, version = 0;
  std::memcpy(&ps_len, t + 8, sizeof(ps_len));
  std::memcpy(&version, t + 10, sizeof(version));
  if (version != kHplnFileVersion) {
    return "read_hpln_postscript: unsupported file version " + std::to_string(version);
  }
  if (ps_off + ps_len + kHplnTrailerBytes > file_size) {
    return "read_hpln_postscript: postscript runs past the file";
  }
  // Is the postscript inside the tail we were given? If not, say exactly how much would do.
  auto const tail_start = file_size - tail.size();
  if (ps_off < tail_start) {
    if (need_bytes) *need_bytes = file_size - ps_off;
    return "read_hpln_postscript: tail too short for the postscript";
  }
  Reader r{tail.data() + (ps_off - tail_start), ps_len};
  std::uint16_t n = 0;
  if (!r.read_le(n)) return "read_hpln_postscript: truncated postscript";
  out.reserve(n);
  for (std::uint16_t i = 0; i < n; ++i) {
    std::uint16_t kind = 0;
    std::uint64_t off = 0, len = 0;
    if (!r.read_le(kind) || !r.read_le(off) || !r.read_le(len)) {
      return "read_hpln_postscript: truncated segment table";
    }
    if (off + len > file_size) return "read_hpln_postscript: segment runs past the file";
    out.push_back({static_cast<hpln_segment>(kind), off, len});
  }
  return {};
}

compressed_table read_compressed_table(std::string const& path,
                                       rmm::cuda_stream_view stream,
                                       rmm::device_async_resource_ref mr,
                                       std::string* error_out)
{
  nvtx3::scoped_range nvtx_range{"simpatico::io::read_table[file]"};
  auto fail = [&](std::string const& msg) -> compressed_table {
    if (error_out) *error_out = msg;
    return {};
  };

  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f) return fail("failed to open '" + path + "' for reading");
  auto file_size = static_cast<std::size_t>(f.tellg());
  f.seekg(0);
  std::vector<std::uint8_t> raw(file_size);
  f.read(reinterpret_cast<char*>(raw.data()), static_cast<std::streamsize>(file_size));
  if (!f) return fail("read error on '" + path + "'");

  Reader r{raw.data(), file_size};
  std::vector<ColRecord> col_records;
  if (!parse_hpln_header(r, col_records, error_out)) return {};

  // The payload region is whatever follows the header in the file.
  const std::uint8_t* payload_base = r.p;
  const std::size_t payload_total  = r.rem;

  // Bounds-check every buffer up front so a truncated file is reported here
  // rather than faulting mid-copy, then copy each buffer straight to device.
  for (auto const& cr : col_records) {
    for (std::size_t li = 0; li < cr.leaf_descs.size(); ++li) {
      for (std::size_t bi = 0; bi < cr.leaf_descs[li].buffers.size(); ++bi) {
        std::size_t sz = static_cast<std::size_t>(cr.leaf_descs[li].buffers[bi].size_bytes);
        std::uint64_t const off = cr.buf_offsets[li][bi];
        // Subtraction form: `off + sz` could overflow for a corrupt/hostile
        // file's huge declared offset and wrap below payload_total, passing the
        // check and then faulting in fetch(). Compare against the remaining space
        // instead so it can never overflow.
        if (sz > 0 && (off > payload_total || sz > payload_total - off))
          return fail("payload out of bounds");
      }
    }
  }

  payload_fetch_fn fetch =
    [&](std::uint64_t off, std::size_t sz, void* dst, rmm::cuda_stream_view s) {
      cudaMemcpyAsync(dst, payload_base + off, sz, cudaMemcpyHostToDevice, s.value());
    };

  return reconstruct_from_records(col_records, fetch, stream, mr, /*leaf_mr=*/mr, error_out);
}

// ---------------------------------------------------------------------------
// compressed_table::describe() — the leaf manifest the writer below walks.
// ---------------------------------------------------------------------------
namespace {

leaf_desc make_leaf_desc(std::uint32_t node_index,
                         std::int32_t slot,
                         OpId kind,
                         compressed_representation const* rep,
                         rmm::cuda_stream_view stream)
{
  leaf_desc d;
  d.node_index = node_index;
  d.slot       = slot;
  d.kind       = kind;
  d.type_tag   = dtype_to_tag(rep->decoded_type());
  // The node's own output length. Decode sizes the codegen kernel grid from this,
  // so a nested fused subtree (whose length is far below the column row count)
  // must round-trip its true length rather than inherit the column's.
  d.num_rows = rep->num_rows > 0 ? static_cast<std::uint64_t>(rep->num_rows) : 0;
  d.meta     = rep->describe_meta();
  for (auto const& ch : rep->named_channels(stream)) {
    leaf_buffer_desc bd;
    bd.name       = ch.name;
    bd.type_tag   = dtype_to_tag(ch.view.type());
    bd.num_rows   = static_cast<std::uint64_t>(ch.view.size());
    bd.size_bytes = static_cast<std::uint64_t>(ch.view.size()) *
                    static_cast<std::uint64_t>(cudf::size_of(ch.view.type()));
    bd.device_ptr = ch.view.head<void>();
    d.buffers.push_back(std::move(bd));
  }
  return d;
}

}  // namespace

// Walk each column's PlanTree; for every stored rep emit one leaf_desc. Two
// storage slots per PlanNode:
//   * node.rep      (the op's own representation)
//   * node.channels (path = the map key)
// rep->kind() is used for all rep types including codegen_fused_representation,
// which maps its fused op tag to a leaf kind (delta->Delta, rle->Rle,
// bitpack->Bitpack, for->For, zigzag->Zigzag, RawFused->Identity).
std::vector<std::vector<leaf_desc>> compressed_table::describe(rmm::cuda_stream_view stream) const
{
  std::vector<std::vector<leaf_desc>> result;
  result.reserve(columns.size());
  for (auto const& col : columns) {
    std::vector<leaf_desc> descs;
    if (!col.plan_tree) {
      result.push_back({});
      continue;
    }
    auto const& nodes = col.plan_tree->nodes;
    for (std::uint32_t ni = 0; ni < nodes.size(); ++ni) {
      auto const& node = nodes[ni];
      if (node.rep) {
        descs.push_back(make_leaf_desc(ni, kSelfRepSlot, node.rep->kind(), node.rep.get(), stream));
      }
      for (std::size_t k = 0; k < node.output_paths.size(); ++k) {
        auto it = node.channels.find(node.output_paths[k]);
        if (it != node.channels.end() && it->second) {
          descs.push_back(make_leaf_desc(
            ni, static_cast<std::int32_t>(k), it->second->kind(), it->second.get(), stream));
        }
      }
    }
    result.push_back(std::move(descs));
  }
  return result;
}

// ─── In-memory (pinned host) serialization ──────────────────────────────────

std::string build_compressed_table_header(compressed_table const& table,
                                          std::vector<std::uint8_t>& out_header,
                                          std::vector<payload_buffer_ref>& out_buffers,
                                          std::uint64_t& out_payload_bytes,
                                          rmm::cuda_stream_view stream)
{
  nvtx3::scoped_range nvtx_range{"simpatico::io::build_header"};
  auto const all_descs = table.describe(stream);

  out_header.clear();
  out_buffers.clear();

  auto& hdr = out_header;
  hdr.push_back('H');
  hdr.push_back('P');
  hdr.push_back('L');
  hdr.push_back('N');
  push_le(hdr, kVersion);
  push_le(hdr, static_cast<std::uint16_t>(table.columns.size()));

  std::uint64_t payload_offset = 0;

  static const PlanTree kEmptyTree;
  for (std::size_t ci = 0; ci < table.columns.size(); ++ci) {
    auto const& col   = table.columns[ci];
    auto const& descs = all_descs[ci];

    push_str16(hdr, col.name.value_or(std::string{}));
    if (dtype_to_tag(col.dtype) == 255) {
      return "write: column '" + col.name.value_or(std::to_string(ci)) +
             "' has an unsupported dtype (id " + std::to_string(static_cast<int>(col.dtype.id())) +
             ") with no serialization tag";
    }
    push_le(hdr, dtype_to_tag(col.dtype));
    push_le(hdr, col.dtype.scale());  // fixed-point scale (0 for non-decimal)
    push_le(hdr, col.num_rows);

    // Structural plan tree (identical layout to the file header, so the same
    // parser reconstructs it): the node array is the source of truth on read.
    PlanTree const& tree = col.plan_tree ? *col.plan_tree : kEmptyTree;
    push_le(hdr, static_cast<std::uint16_t>(tree.nodes.size()));
    for (auto const& node : tree.nodes)
      push_node(hdr, node);

    push_le(hdr, static_cast<std::uint16_t>(descs.size()));
    for (auto const& ld : descs) {
      if (ld.type_tag == 255) {
        return "write: leaf node " + std::to_string(ld.node_index) +
               " has an unsupported dtype with no serialization tag";
      }
      push_le(hdr, ld.node_index);
      push_le(hdr, ld.slot);
      push_le(hdr, static_cast<std::uint8_t>(ld.kind));
      push_le(hdr, ld.type_tag);
      push_le(hdr, ld.num_rows);
      push_meta(hdr, ld.meta);
      push_le(hdr, static_cast<std::uint8_t>(ld.buffers.size()));

      for (auto const& bd : ld.buffers) {
        if (bd.type_tag == 255) {
          return "write: leaf buffer '" + bd.name +
                 "' has an unsupported dtype with no serialization tag";
        }
        push_str16(hdr, bd.name);
        push_le(hdr, bd.type_tag);
        push_le(hdr, bd.size_bytes);
        push_le(hdr, payload_offset);

        // Record the buffer for the caller to stage out of device memory; no
        // bytes are copied here.
        out_buffers.push_back(payload_buffer_ref{payload_offset,
                                                 bd.device_ptr,
                                                 bd.size_bytes,
                                                 leaf_alloc_bytes(bd.type_tag, bd.num_rows),
                                                 ci});
        payload_offset += bd.size_bytes;
      }
    }
  }

  out_payload_bytes = payload_offset;
  return {};
}

compressed_table read_compressed_table_from_memory(
  std::span<const std::uint8_t> header,
  payload_fetch_fn const& fetch,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr,
  std::string* error_out,
  std::optional<rmm::device_async_resource_ref> leaf_mr)
{
  nvtx3::scoped_range nvtx_range{"simpatico::io::read_table[memory]"};
  Reader r{header.data(), header.size()};
  std::vector<ColRecord> col_records;
  if (!parse_hpln_header(r, col_records, error_out)) return {};
  // Leaf (enumerated) buffers come from leaf_mr when supplied, so a caller can place
  // them in a dedicated slab/arena; codec decode scratch always comes from mr.
  return reconstruct_from_records(col_records, fetch, stream, mr, leaf_mr.value_or(mr), error_out);
}

compressed_table read_compressed_table_subset_from_memory(
  std::span<const std::uint8_t> header,
  payload_fetch_fn const& fetch,
  std::span<const std::size_t> selected_columns,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr,
  std::string* error_out)
{
  nvtx3::scoped_range nvtx_range{"simpatico::io::read_table[memory,subset]"};
  Reader r{header.data(), header.size()};
  std::vector<ColRecord> col_records;
  if (!parse_hpln_header(r, col_records, error_out)) return {};
  // Keep only the requested columns' records; reconstruct_from_records fetches
  // payload buffers only for the records handed to it, so a subset read pulls
  // just those columns' bytes to the GPU (buffer offsets in the header are
  // absolute, so dropping columns does not disturb the survivors' fetches).
  std::vector<ColRecord> selected;
  selected.reserve(selected_columns.size());
  std::vector<bool> consumed(col_records.size(), false);
  for (auto idx : selected_columns) {
    if (idx >= col_records.size()) {
      if (error_out) {
        *error_out = "read_compressed_table_subset_from_memory: column index out of range";
      }
      return {};
    }
    if (!consumed[idx]) {
      selected.push_back(std::move(col_records[idx]));
      consumed[idx] = true;
      continue;
    }

    // ColRecord owns a PlanTree and is intentionally move-only. Reparse the
    // header for a repeated selection so each output column gets independent
    // ownership while retaining the same absolute payload offsets.
    Reader duplicate_reader{header.data(), header.size()};
    std::vector<ColRecord> duplicate_records;
    if (!parse_hpln_header(duplicate_reader, duplicate_records, error_out)) return {};
    selected.push_back(std::move(duplicate_records[idx]));
  }
  return reconstruct_from_records(selected, fetch, stream, mr, /*leaf_mr=*/mr, error_out);
}

// ─── Chunk-subset header synthesis ──────────────────────────────────────────

namespace {

// Rows per decode chunk. Mirrors codegen::kChunkSize, which the decode grid is sized from
// (leaf_desc::num_rows / kChunkSize); surviving chunk ids are in these units.
constexpr std::uint64_t kSubsetRowsPerChunk = 1024;

// Rows the surviving chunks of a @p total_rows sequence cover. Not simply
// survivors * kSubsetRowsPerChunk: the last chunk of a column is short.
std::uint64_t rows_over_chunks(std::uint64_t total_rows,
                               std::span<const std::uint32_t> surviving_chunks)
{
  if (total_rows == 0) return 0;
  auto const n_chunks = (total_rows + kSubsetRowsPerChunk - 1) / kSubsetRowsPerChunk;
  std::uint64_t rows  = 0;
  for (auto const chunk : surviving_chunks) {
    if (chunk >= n_chunks) continue;  // a survivor past the end addresses nothing
    auto const first = static_cast<std::uint64_t>(chunk) * kSubsetRowsPerChunk;
    rows += std::min(kSubsetRowsPerChunk, total_rows - first);
  }
  return rows;
}

// Element width of a serialized buffer, or 0 when the tag is not a fixed-width type (which the
// buffer layout arithmetic cannot address, so such a buffer is never subsetted).
std::uint64_t buffer_elem_size(std::uint8_t type_tag)
{
  if (type_tag == 255) return 0;
  auto const dt = tag_to_dtype(type_tag);
  if (!cudf::is_fixed_width(dt)) return 0;
  return static_cast<std::uint64_t>(cudf::size_of(dt));
}

}  // namespace

std::string describe_compressed_table_header(std::span<const std::uint8_t> header, hpln_schema& out)
{
  out = {};
  Reader r{header.data(), header.size()};
  std::vector<ColRecord> recs;
  std::string err;
  if (!parse_hpln_header(r, recs, &err)) {
    return err.empty() ? std::string("describe_compressed_table_header: malformed header") : err;
  }
  out.header_bytes = header.size() - r.rem;
  out.columns.reserve(recs.size());
  for (auto const& cr : recs) {
    hpln_column_desc d;
    d.name      = cr.name;
    d.dtype_tag = cr.dtype_tag;
    d.scale     = cr.scale;
    d.num_rows  = cr.num_rows;
    for (std::size_t li = 0; li < cr.leaf_descs.size(); ++li) {
      for (std::size_t bi = 0; bi < cr.leaf_descs[li].buffers.size(); ++bi) {
        auto const size = cr.leaf_descs[li].buffers[bi].size_bytes;
        d.compressed_bytes += size;
        out.payload_bytes = std::max(out.payload_bytes, cr.buf_offsets[li][bi] + size);
      }
    }
    out.columns.push_back(std::move(d));
  }
  return {};
}

std::string build_chunk_subset_header(std::span<const std::uint8_t> header,
                                      std::span<const std::uint32_t> surviving_chunks,
                                      payload_host_read_fn const& read_payload,
                                      std::vector<std::uint8_t>& out_header,
                                      std::vector<gather_range>& out_gather,
                                      std::uint64_t max_gap_bytes,
                                      std::uint64_t* out_payload_bytes,
                                      std::vector<std::uint8_t>* out_column_subsetted)
{
  nvtx3::scoped_range nvtx_range{"simpatico::io::build_chunk_subset_header"};
  out_header.clear();
  out_gather.clear();
  if (out_payload_bytes) *out_payload_bytes = 0;
  if (out_column_subsetted) out_column_subsetted->clear();

  for (std::size_t i = 1; i < surviving_chunks.size(); ++i) {
    if (surviving_chunks[i] <= surviving_chunks[i - 1]) {
      return "build_chunk_subset_header: surviving_chunks must be strictly ascending";
    }
  }
  if (surviving_chunks.empty()) {
    // A table with no rows at all is not something the reader can reconstruct, and a caller with
    // nothing surviving has no reason to read the pin. Refuse rather than emit a degenerate table.
    return "build_chunk_subset_header: no surviving chunks";
  }

  Reader r{header.data(), header.size()};
  std::vector<ColRecord> recs;
  HeaderFieldOffsets offsets;
  std::string parse_err;
  if (!parse_hpln_header(r, recs, &parse_err, &offsets)) {
    return parse_err.empty() ? std::string("build_chunk_subset_header: malformed header")
                             : parse_err;
  }
  if (out_column_subsetted) out_column_subsetted->assign(recs.size(), 0);

  // Copy only the bytes the parse consumed: a caller may hand us a span covering the whole file,
  // and the header the reader gets back must be exactly a header.
  auto const header_bytes = header.size() - r.rem;
  out_header.assign(header.begin(), header.begin() + static_cast<std::ptrdiff_t>(header_bytes));

  auto patch = [&](std::uint64_t at, auto value) -> bool {
    if (at + sizeof(value) > out_header.size()) return false;
    auto const bytes = std::bit_cast<std::array<std::uint8_t, sizeof(value)>>(value);
    std::memcpy(out_header.data() + at, bytes.data(), bytes.size());
    return true;
  };

  // Merge a copy into the previous one when it is contiguous on BOTH sides. That is what makes an
  // all-chunks-surviving subset collapse to a single range over the whole payload, i.e. degrade
  // exactly to the existing whole-table fetch.
  auto append_gather = [&](std::uint64_t src, std::uint64_t size, std::uint64_t dst) {
    if (size == 0) return;
    if (!out_gather.empty()) {
      auto& last = out_gather.back();
      if (src == last.src_offset + last.size && dst == last.dst_offset + last.size) {
        last.size += size;
        return;
      }
    }
    out_gather.push_back(gather_range{src, size, dst});
  };

  std::uint64_t dst_offset = 0;

  for (std::size_t ci = 0; ci < recs.size(); ++ci) {
    auto const& cr       = recs[ci];
    auto const& col_offs = offsets.columns[ci];
    auto const col_rows =
      cr.num_rows > 0 ? static_cast<std::uint64_t>(cr.num_rows) : std::uint64_t{0};

    // Which nodes are COLUMN STATE rather than rows. A dictionary's keys are the case that
    // matters: whatever compresses them produces buffers that look row-indexed on their own grid
    // and have nothing to do with the column's rows, so the whole subtree hanging off such a
    // channel is fetched whole and its length is not evidence that the column cannot be subsetted.
    // Marked by walking the edges from the root; the mark is inherited, since a dictionary's keys
    // stay column state however deeply they are then compressed.
    std::vector<bool> node_is_column_state(cr.tree.nodes.size(), false);
    {
      std::vector<std::uint32_t> pending;
      if (!cr.tree.nodes.empty()) { pending.push_back(0); }
      while (!pending.empty()) {
        auto const ni = pending.back();
        pending.pop_back();
        auto const& node = cr.tree.nodes[ni];
        auto const kind  = op_id_from_name(node.op).value_or(OpId::Unknown);
        for (auto const& edge : node.children) {
          auto const child = static_cast<std::size_t>(edge.child);
          if (child >= cr.tree.nodes.size() || child == ni) { continue; }
          node_is_column_state[child] =
            node_is_column_state[ni] || channel_is_column_state(kind, edge.channel);
          pending.push_back(static_cast<std::uint32_t>(child));
        }
      }
    }
    // A leaf is column state when its node is, or when it IS a column-state output channel of a
    // row-indexed node (a bare `input -> dictionary`, whose keys are stored as they are).
    auto const leaf_is_column_state = [&](leaf_desc const& ld) {
      auto const node = static_cast<std::size_t>(ld.node_index);
      if (node >= cr.tree.nodes.size()) { return false; }
      if (node_is_column_state[node]) { return true; }
      if (ld.slot < 0) { return false; }
      auto const& names = cr.tree.nodes[node].output_names;
      auto const slot   = static_cast<std::size_t>(ld.slot);
      if (slot >= names.size()) { return false; }
      return channel_is_column_state(
        op_id_from_name(cr.tree.nodes[node].op).value_or(OpId::Unknown), names[slot]);
    };

    // Plan every buffer of every leaf first. A column is served as a subset only if ALL of its
    // ROW-INDEXED buffers can be, because the leaves of one column decode together: a compacted
    // `packed` next to a full-length `chunk_count` is not a valid column, it is silently wrong
    // data. Column-state leaves and buffers are emitted whole and constrain nothing.
    std::vector<std::vector<buffer_subset>> plans(cr.leaf_descs.size());
    std::vector<std::vector<bool>> buffer_whole(cr.leaf_descs.size());
    std::vector<bool> leaf_whole(cr.leaf_descs.size(), false);
    bool subsettable = col_rows > 0;

    for (std::size_t li = 0; subsettable && li < cr.leaf_descs.size(); ++li) {
      auto const& ld = cr.leaf_descs[li];
      if (leaf_is_column_state(ld)) {
        leaf_whole[li] = true;
        continue;
      }
      if (!supports_chunk_subset(ld.kind)) {
        subsettable = false;  // whole_column or unclassified: fetch the column whole
        break;
      }
      // Chunk ids are the COLUMN's 1024-row chunks. A node whose own output length differs (the
      // values channel under an rle, say) chunks on a different grid, so survivor c does not name
      // the same rows there and nothing about this column can be subsetted.
      if (ld.num_rows != col_rows) {
        subsettable = false;
        break;
      }

      auto const n_chunks = (ld.num_rows + kSubsetRowsPerChunk - 1) / kSubsetRowsPerChunk;
      std::vector<std::uint32_t> all_chunks(n_chunks);
      for (std::uint64_t c = 0; c < n_chunks; ++c) {
        all_chunks[c] = static_cast<std::uint32_t>(c);
      }

      // A bulk_variable buffer (only bitpack's `packed`) is sized by this leaf's own per-chunk
      // metadata, whose VALUES live in the payload. Pull them host-side; without them
      // plan_buffer_subset refuses and the column is emitted whole.
      std::vector<std::int32_t> chunk_count;
      std::vector<std::uint8_t> chunk_bits;
      bool const needs_sizing =
        std::any_of(ld.buffers.begin(), ld.buffers.end(), [&](auto const& bd) {
          auto const layout = buffer_layout(ld.kind, bd.name);
          return layout && *layout == ChannelLayout::bulk_variable;
        });
      if (needs_sizing && read_payload) {
        auto load = [&](char const* name, void* dst, std::uint64_t want) {
          for (std::size_t bi = 0; bi < ld.buffers.size(); ++bi) {
            if (ld.buffers[bi].name != name) continue;
            if (ld.buffers[bi].size_bytes < want) return false;
            return read_payload(cr.buf_offsets[li][bi], want, dst);
          }
          return false;
        };
        chunk_count.resize(n_chunks);
        chunk_bits.resize(n_chunks);
        if (!load("chunk_count", chunk_count.data(), n_chunks * sizeof(std::int32_t)) ||
            !load("chunk_bits", chunk_bits.data(), n_chunks * sizeof(std::uint8_t))) {
          chunk_count.clear();
          chunk_bits.clear();
        }
      }
      chunk_sizing_metadata const sizing{chunk_count, chunk_bits};

      plans[li].resize(ld.buffers.size());
      buffer_whole[li].assign(ld.buffers.size(), false);
      for (std::size_t bi = 0; bi < ld.buffers.size(); ++bi) {
        auto const& bd = ld.buffers[bi];
        // A column-state buffer of an otherwise row-indexed leaf — the keys of a bare
        // `input -> dictionary`, which stores its keys and its per-row indices in one leaf. The
        // keys are fetched whole and stay valid for whichever rows the indices name.
        if (buffer_layout(ld.kind, bd.name) == ChannelLayout::column_state) {
          buffer_whole[li][bi] = true;
          continue;
        }
        auto const elem_size = buffer_elem_size(bd.type_tag);
        auto const plan      = plan_buffer_subset(ld.kind,
                                             bd.name,
                                             static_cast<std::size_t>(elem_size),
                                             ld.num_rows,
                                             kSubsetRowsPerChunk,
                                             surviving_chunks,
                                             sizing,
                                             max_gap_bytes);
        if (!plan) {
          subsettable = false;
          break;
        }
        // Self-check: with EVERY chunk surviving the model must reproduce the size the writer
        // declared. When it does not, our idea of this buffer's layout disagrees with what was
        // actually written, and a subset built on it would be wrong without faulting -- the reader
        // allocates whatever the header says and the decode reads whatever landed there. Emitting
        // the column whole is always correct, so fall back to that.
        auto const full = plan_buffer_subset(ld.kind,
                                             bd.name,
                                             static_cast<std::size_t>(elem_size),
                                             ld.num_rows,
                                             kSubsetRowsPerChunk,
                                             all_chunks,
                                             sizing,
                                             /*max_gap_bytes=*/0);
        if (!full || full->compacted_size != bd.size_bytes) {
          subsettable = false;
          break;
        }
        plans[li][bi] = *plan;
      }
    }

    if (!subsettable) {
      // Emitted whole: sizes and row counts stay exactly as written; only the payload offsets move
      // to keep the compacted payload dense.
      for (std::size_t li = 0; li < cr.leaf_descs.size(); ++li) {
        auto const& ld = cr.leaf_descs[li];
        for (std::size_t bi = 0; bi < ld.buffers.size(); ++bi) {
          auto const size = ld.buffers[bi].size_bytes;
          if (!patch(col_offs.leaves[li].buffers[bi].payload_offset_at, dst_offset)) {
            return "build_chunk_subset_header: header field offset out of range";
          }
          append_gather(cr.buf_offsets[li][bi], size, dst_offset);
          dst_offset += size;
        }
      }
      continue;
    }

    if (out_column_subsetted) (*out_column_subsetted)[ci] = 1;

    // The column's row count over the survivors. Distinct from any leaf's, and not a multiple of
    // the chunk size: the column's last chunk is short.
    if (!patch(col_offs.num_rows_at,
               static_cast<std::int64_t>(rows_over_chunks(col_rows, surviving_chunks)))) {
      return "build_chunk_subset_header: header field offset out of range";
    }

    for (std::size_t li = 0; li < cr.leaf_descs.size(); ++li) {
      auto const& ld = cr.leaf_descs[li];
      // A column-state leaf is not indexed by rows at all, so neither its own length nor its
      // buffers change; only where they sit in the compacted payload.
      if (leaf_whole[li]) {
        for (std::size_t bi = 0; bi < ld.buffers.size(); ++bi) {
          auto const size = ld.buffers[bi].size_bytes;
          if (!patch(col_offs.leaves[li].buffers[bi].payload_offset_at, dst_offset)) {
            return "build_chunk_subset_header: header field offset out of range";
          }
          append_gather(cr.buf_offsets[li][bi], size, dst_offset);
          dst_offset += size;
        }
        continue;
      }
      // leaf_desc::num_rows is the NODE's own output length, not the column's (it drives the
      // codegen decode grid), so it is recomputed over the survivors in its own right.
      if (!patch(col_offs.leaves[li].num_rows_at,
                 rows_over_chunks(ld.num_rows, surviving_chunks))) {
        return "build_chunk_subset_header: header field offset out of range";
      }

      for (std::size_t bi = 0; bi < ld.buffers.size(); ++bi) {
        auto const& bd  = ld.buffers[bi];
        auto const& bof = col_offs.leaves[li].buffers[bi];
        if (buffer_whole[li][bi]) {
          if (!patch(bof.payload_offset_at, dst_offset)) {
            return "build_chunk_subset_header: header field offset out of range";
          }
          append_gather(cr.buf_offsets[li][bi], bd.size_bytes, dst_offset);
          dst_offset += bd.size_bytes;
          continue;
        }
        auto const& sub = plans[li][bi];
        if (!patch(bof.size_bytes_at, sub.compacted_size) ||
            !patch(bof.payload_offset_at, dst_offset)) {
          return "build_chunk_subset_header: header field offset out of range";
        }

        std::uint64_t written = 0;
        for (auto const& range : sub.ranges) {
          append_gather(cr.buf_offsets[li][bi] + range.offset, range.size, dst_offset + written);
          written += range.size;
        }
        if (written < sub.compacted_size) {
          // The declared size can exceed the bytes the plan moves: a bitpack `packed` buffer
          // carries decode guard words past its last live word. Fill them from whatever follows in
          // the source -- the decode never reads them as values, and it makes an all-surviving
          // subset a byte-exact copy of the payload rather than one with holes in it.
          auto const src_pos = sub.ranges.empty() ? std::uint64_t{0} : sub.ranges.back().end();
          auto const avail   = bd.size_bytes > src_pos ? bd.size_bytes - src_pos : 0;
          append_gather(cr.buf_offsets[li][bi] + src_pos,
                        std::min(sub.compacted_size - written, avail),
                        dst_offset + written);
        }
        dst_offset += sub.compacted_size;
      }
    }
  }

  if (out_payload_bytes) *out_payload_bytes = dst_offset;
  return {};
}

}  // namespace simpatico
