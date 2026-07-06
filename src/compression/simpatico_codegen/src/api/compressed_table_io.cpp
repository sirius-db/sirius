// SPDX-License-Identifier: Apache-2.0
//
// C++-native .hpln v8 write/read for compressed_table.
// See compressed_table_io.hpp for the on-disk layout.

#include "api/compressed_table_io.hpp"

#include "codegen/plan/plan_interpreter.hpp"
#include "codegen/plan/representation.hpp"

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/types.hpp>

#include <rmm/device_buffer.hpp>
#include <rmm/mr/per_device_resource.hpp>

#include <cuda_runtime.h>

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
// LE binary write helpers
// ---------------------------------------------------------------------------
static void push_u8(std::vector<std::uint8_t>& v, std::uint8_t x) { v.push_back(x); }
static void push_u16le(std::vector<std::uint8_t>& v, std::uint16_t x)
{
  v.push_back(static_cast<std::uint8_t>(x));
  v.push_back(static_cast<std::uint8_t>(x >> 8));
}
static void push_u32le(std::vector<std::uint8_t>& v, std::uint32_t x)
{
  for (int i = 0; i < 4; ++i)
    v.push_back(static_cast<std::uint8_t>(x >> (8 * i)));
}
static void push_i32le(std::vector<std::uint8_t>& v, std::int32_t x)
{
  push_u32le(v, static_cast<std::uint32_t>(x));
}
static void push_u64le(std::vector<std::uint8_t>& v, std::uint64_t x)
{
  for (int i = 0; i < 8; ++i)
    v.push_back(static_cast<std::uint8_t>(x >> (8 * i)));
}
static void push_i64le(std::vector<std::uint8_t>& v, std::int64_t x)
{
  push_u64le(v, static_cast<std::uint64_t>(x));
}
static void push_str16(std::vector<std::uint8_t>& v, std::string const& s)
{
  push_u16le(v, static_cast<std::uint16_t>(s.size()));
  v.insert(v.end(), s.begin(), s.end());
}

// ---------------------------------------------------------------------------
// LE binary read helpers
// ---------------------------------------------------------------------------
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
  bool read_u8(std::uint8_t& x) { return read(&x, 1); }
  bool read_u16le(std::uint16_t& x)
  {
    std::uint8_t b[2];
    if (!read(b, 2)) return false;
    x = static_cast<std::uint16_t>(b[0]) | (static_cast<std::uint16_t>(b[1]) << 8);
    return true;
  }
  bool read_u32le(std::uint32_t& x)
  {
    std::uint8_t b[4];
    if (!read(b, 4)) return false;
    x = 0;
    for (int i = 0; i < 4; ++i)
      x |= static_cast<std::uint32_t>(b[i]) << (8 * i);
    return true;
  }
  bool read_i32le(std::int32_t& x)
  {
    std::uint32_t u;
    if (!read_u32le(u)) return false;
    x = static_cast<std::int32_t>(u);
    return true;
  }
  bool read_u64le(std::uint64_t& x)
  {
    std::uint8_t b[8];
    if (!read(b, 8)) return false;
    x = 0;
    for (int i = 0; i < 8; ++i)
      x |= static_cast<std::uint64_t>(b[i]) << (8 * i);
    return true;
  }
  bool read_i64le(std::int64_t& x)
  {
    std::uint64_t u;
    if (!read_u64le(u)) return false;
    x = static_cast<std::int64_t>(u);
    return true;
  }
  bool read_str16(std::string& s)
  {
    std::uint16_t len;
    if (!read_u16le(len)) return false;
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
    void operator()(leaf_meta::none const&) { push_u8(v, META_NONE); }
    void operator()(leaf_meta::alp_rd const& a)
    {
      push_u8(v, META_ALP_RD);
      push_u8(v, a.right_bw);
    }
    void operator()(leaf_meta::ans const& a)
    {
      push_u8(v, META_ANS);
      push_u64le(v, a.uncompressed_size);
      push_i32le(v, a.original_type_id);
    }
    void operator()(leaf_meta::bitcomp const& b)
    {
      push_u8(v, META_BITCOMP);
      push_u64le(v, b.uncompressed_size);
      push_i32le(v, b.original_type_id);
      push_i32le(v, b.algorithm);
    }
    void operator()(leaf_meta::nvcomp_cascaded const& c)
    {
      push_u8(v, META_CASCADED);
      push_u64le(v, c.uncompressed_size);
      push_i32le(v, c.original_type_id);
      push_i32le(v, c.num_deltas);
      push_i32le(v, c.num_RLEs);
      push_i32le(v, c.use_bp);
    }
    void operator()(leaf_meta::snappy const& s)
    {
      push_u8(v, META_SNAPPY);
      push_u64le(v, s.uncompressed_size);
      push_i32le(v, s.original_type_id);
    }
    void operator()(leaf_meta::lz4 const& l)
    {
      push_u8(v, META_LZ4);
      push_u64le(v, l.uncompressed_size);
      push_i32le(v, l.original_type_id);
    }
    void operator()(leaf_meta::deflate const& d)
    {
      push_u8(v, META_DEFLATE);
      push_u64le(v, d.uncompressed_size);
      push_i32le(v, d.original_type_id);
    }
  };
  std::visit(Visitor{v}, m);
}

static bool read_meta(Reader& r, leaf_meta_v& out)
{
  std::uint8_t mk;
  if (!r.read_u8(mk)) return false;
  switch (mk) {
    case META_NONE: out = leaf_meta::none{}; return true;
    case META_ALP_RD: {
      std::uint8_t rbw;
      if (!r.read_u8(rbw)) return false;
      out = leaf_meta::alp_rd{rbw};
      return true;
    }
    case META_ANS: {
      std::uint64_t us;
      std::int32_t ti;
      if (!r.read_u64le(us) || !r.read_i32le(ti)) return false;
      out = leaf_meta::ans{us, ti};
      return true;
    }
    case META_BITCOMP: {
      std::uint64_t us;
      std::int32_t ti, alg;
      if (!r.read_u64le(us) || !r.read_i32le(ti) || !r.read_i32le(alg)) return false;
      out = leaf_meta::bitcomp{us, ti, alg};
      return true;
    }
    case META_CASCADED: {
      std::uint64_t us;
      std::int32_t ti, nd, nr, bp;
      if (!r.read_u64le(us) || !r.read_i32le(ti) || !r.read_i32le(nd) || !r.read_i32le(nr) ||
          !r.read_i32le(bp))
        return false;
      out = leaf_meta::nvcomp_cascaded{us, ti, nd, nr, bp};
      return true;
    }
    case META_SNAPPY: {
      std::uint64_t us;
      std::int32_t ti;
      if (!r.read_u64le(us) || !r.read_i32le(ti)) return false;
      out = leaf_meta::snappy{us, ti};
      return true;
    }
    case META_LZ4: {
      std::uint64_t us;
      std::int32_t ti;
      if (!r.read_u64le(us) || !r.read_i32le(ti)) return false;
      out = leaf_meta::lz4{us, ti};
      return true;
    }
    case META_DEFLATE: {
      std::uint64_t us;
      std::int32_t ti;
      if (!r.read_u64le(us) || !r.read_i32le(ti)) return false;
      out = leaf_meta::deflate{us, ti};
      return true;
    }
    default: return false;
  }
}

// ---------------------------------------------------------------------------
// PlanLeafKind → compressor name (for reconstruct_representation)
// ---------------------------------------------------------------------------
static const char* leaf_kind_to_compressor(PlanLeafKind k)
{
  switch (k) {
    case PlanLeafKind::Identity: return "identity";
    case PlanLeafKind::Delta: return "delta";
    case PlanLeafKind::Rle: return "rle";
    case PlanLeafKind::Dictionary: return "dictionary";
    case PlanLeafKind::Bitpack: return "bitpack";
    case PlanLeafKind::For: return "for";
    case PlanLeafKind::Zigzag: return "zigzag";
    case PlanLeafKind::Alp: return "alp";
    case PlanLeafKind::AlpRd: return "alp_rd";
    case PlanLeafKind::Ans: return "ans";
    case PlanLeafKind::Bitcomp: return "bitcomp";
    case PlanLeafKind::NvcompCascaded: return "nvcomp_cascaded";
    case PlanLeafKind::Snappy: return "snappy";
    case PlanLeafKind::Lz4: return "lz4";
    case PlanLeafKind::Deflate: return "deflate";
    default: return nullptr;
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

static std::unique_ptr<compressed_representation> rep_from_leaf_desc(
  leaf_desc const& ld,
  cudf::size_type col_num_rows,
  leaf_buffer_fill const& fill,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr,
  std::string* err)
{
  auto const& bufs = ld.buffers;

  // Helper: allocate a device column and fill it from the i-th payload buffer.
  auto make_col = [&](std::size_t i) -> std::unique_ptr<cudf::column> {
    auto const& bd     = bufs[i];
    cudf::data_type dt = tag_to_dtype(bd.type_tag);
    auto col           = cudf::make_numeric_column(
      dt, static_cast<cudf::size_type>(bd.num_rows), cudf::mask_state::UNALLOCATED, stream);
    if (bd.size_bytes > 0) {
      fill(i, col->mutable_view().head<void>(), static_cast<std::size_t>(bd.size_bytes), stream);
    }
    return col;
  };

  // Delta, Rle, For and Zigzag are codegen-only operators: encode always
  // produces a JIT codegen_fused_representation carrying the fused region's
  // manifest buffers, so their leaves are reconstructed unconditionally as a
  // fused rep.
  auto make_fused_rep = [&](const char* kind_tag) {
    auto rep = std::make_unique<codegen_fused_representation>(
      kind_tag, tag_to_dtype(ld.type_tag), col_num_rows);
    for (std::size_t i = 0; i < bufs.size(); ++i) {
      rep->buffers.emplace_back(bufs[i].name, make_col(i));
    }
    return std::unique_ptr<compressed_representation>(std::move(rep));
  };

  if (ld.kind == PlanLeafKind::Delta) { return make_fused_rep("delta"); }
  if (ld.kind == PlanLeafKind::Rle) { return make_fused_rep("rle"); }
  if (ld.kind == PlanLeafKind::For) { return make_fused_rep("for"); }
  if (ld.kind == PlanLeafKind::Zigzag) { return make_fused_rep("zigzag"); }
  if (ld.kind == PlanLeafKind::Identity) {
    bool is_fused = !(bufs.size() == 1 && bufs[0].name == "data");
    if (is_fused) return make_fused_rep("RawFused");
  }

  // All other kinds (and non-fused identity): route through reconstruct_representation.
  const char* cname = leaf_kind_to_compressor(ld.kind);
  if (!cname) {
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
  return reconstruct_representation(cname, names, std::move(cols), stream, mr, err, ld.meta);
}

static constexpr std::uint8_t kVersion = 9;

// Serialize one node's structure (op, bitjoin params, edges, output names).
// Other ops carry their params in the op name, so only bitjoin needs attrs.
static void push_node(std::vector<std::uint8_t>& hdr, PlanNode const& node)
{
  push_str16(hdr, node.op);
  if (node.attrs.bitjoin.has_value()) {
    auto const& bj = *node.attrs.bitjoin;
    push_u8(hdr, 1);
    push_u8(hdr, dtype_to_tag(bj.output_type));
    push_u16le(hdr, static_cast<std::uint16_t>(bj.inputs.size()));
    for (auto const& in : bj.inputs) {
      push_u32le(hdr, in.node);
      push_str16(hdr, in.channel);
      push_u8(hdr, in.range.has_value() ? 1 : 0);
      if (in.range.has_value()) {
        push_u32le(hdr, in.range->first);
        push_u32le(hdr, in.range->second);
      }
    }
  } else {
    push_u8(hdr, 0);
  }
  push_u16le(hdr, static_cast<std::uint16_t>(node.children.size()));
  for (auto const& e : node.children) {
    push_str16(hdr, e.channel);
    push_u32le(hdr, e.child);
  }
  push_u16le(hdr, static_cast<std::uint16_t>(node.output_names.size()));
  for (auto const& n : node.output_names)
    push_str16(hdr, n);
}

// Inverse of push_node. output_paths mirror output_names as the node-local
// channel key; bitjoin input arity/ranges are mirrored into input_paths/
// input_ranges for layout resolution.
static bool read_node(Reader& r, PlanNode& node)
{
  if (!r.read_str16(node.op)) return false;
  std::uint8_t is_bitjoin;
  if (!r.read_u8(is_bitjoin)) return false;
  if (is_bitjoin) {
    bitjoin_attrs bj;
    std::uint8_t ot;
    if (!r.read_u8(ot)) return false;
    bj.output_type = tag_to_dtype(ot);
    std::uint16_t nin;
    if (!r.read_u16le(nin)) return false;
    bj.inputs.resize(nin);
    for (auto& in : bj.inputs) {
      if (!r.read_u32le(in.node) || !r.read_str16(in.channel)) return false;
      std::uint8_t has_range;
      if (!r.read_u8(has_range)) return false;
      if (has_range) {
        std::uint32_t hi, lo;
        if (!r.read_u32le(hi) || !r.read_u32le(lo)) return false;
        in.range = bit_range{hi, lo};
      }
    }
    for (auto const& in : bj.inputs) {
      node.input_paths.push_back(in.channel);
      node.input_ranges.push_back(in.range);
    }
    node.attrs.bitjoin = std::move(bj);
  }
  std::uint16_t ne;
  if (!r.read_u16le(ne)) return false;
  node.children.resize(ne);
  for (auto& e : node.children) {
    if (!r.read_str16(e.channel) || !r.read_u32le(e.child)) return false;
  }
  std::uint16_t no;
  if (!r.read_u16le(no)) return false;
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

// Parse the .hpln v8 header from `r` into one ColRecord per column. On success
// `r` is left pointing just past the header (i.e. at the payload region for the
// concatenated file layout). Returns false and sets *err on any structural error.
static bool parse_hpln_header(Reader& r, std::vector<ColRecord>& out, std::string* err)
{
  auto bad = [&](std::string const& m) {
    if (err) *err = m;
    return false;
  };

  std::uint8_t magic[4];
  if (!r.read(magic, 4)) return bad("truncated header");
  if (magic[0] != 'H' || magic[1] != 'P' || magic[2] != 'L' || magic[3] != 'N')
    return bad("not a HPLN file");
  std::uint8_t ver;
  if (!r.read_u8(ver)) return bad("truncated header");
  if (ver != kVersion) return bad("unsupported version " + std::to_string(ver) + " (expected 9)");

  std::uint16_t num_cols;
  if (!r.read_u16le(num_cols)) return bad("truncated header");
  out.clear();
  out.resize(num_cols);

  for (std::uint16_t ci = 0; ci < num_cols; ++ci) {
    auto& cr = out[ci];
    if (!r.read_str16(cr.name)) return bad("truncated col name");
    if (!r.read_u8(cr.dtype_tag)) return bad("truncated col dtype");
    if (!r.read_i32le(cr.scale)) return bad("truncated col scale");
    if (!r.read_i64le(cr.num_rows)) return bad("truncated col num_rows");

    std::uint16_t nn;
    if (!r.read_u16le(nn)) return bad("truncated num_nodes");
    cr.tree.nodes.resize(nn);
    for (std::uint16_t ni = 0; ni < nn; ++ni) {
      if (!read_node(r, cr.tree.nodes[ni])) return bad("truncated plan node");
    }

    std::uint16_t nl;
    if (!r.read_u16le(nl)) return bad("truncated num_leaves");
    cr.leaf_descs.resize(nl);
    cr.buf_offsets.resize(nl);

    for (std::uint16_t li = 0; li < nl; ++li) {
      auto& ld = cr.leaf_descs[li];
      if (!r.read_u32le(ld.node_index)) return bad("truncated leaf node_index");
      if (!r.read_i32le(ld.slot)) return bad("truncated leaf slot");
      std::uint8_t k;
      if (!r.read_u8(k)) return bad("truncated leaf kind");
      ld.kind = static_cast<PlanLeafKind>(k);
      if (!r.read_u8(ld.type_tag)) return bad("truncated leaf type_tag");
      if (!read_meta(r, ld.meta)) return bad("truncated/unknown leaf meta");

      std::uint8_t nb;
      if (!r.read_u8(nb)) return bad("truncated num_bufs");
      ld.buffers.resize(nb);
      cr.buf_offsets[li].resize(nb);

      for (std::uint8_t bi = 0; bi < nb; ++bi) {
        auto& bd = ld.buffers[bi];
        if (!r.read_str16(bd.name)) return bad("truncated buf name");
        if (!r.read_u8(bd.type_tag)) return bad("truncated buf type_tag");
        if (!r.read_u64le(bd.size_bytes)) return bad("truncated buf size_bytes");
        std::uint64_t poff;
        if (!r.read_u64le(poff)) return bad("truncated buf payload_offset");
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
                                                 std::string* err)
{
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

    auto compound  = std::make_unique<plan_compound>();
    compound->tree = std::move(cr.tree);
    auto& nodes    = compound->tree.nodes;

    for (std::size_t li = 0; li < cr.leaf_descs.size(); ++li) {
      auto const& ld    = cr.leaf_descs[li];
      auto const& boffs = cr.buf_offsets[li];
      if (ld.node_index >= nodes.size())
        return fail("leaf node_index out of range in col " + std::to_string(ci));

      auto fill = [&](std::size_t bi, void* dst, std::size_t sz, rmm::cuda_stream_view s) {
        fetch(boffs[bi], sz, dst, s);
      };

      std::string rep_err;
      auto rep = rep_from_leaf_desc(
        ld, static_cast<cudf::size_type>(cr.num_rows), fill, stream, mr, &rep_err);
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

    compute_input_sources(compound->tree);
    out_col.compound = std::move(compound);
  }

  return result;
}

}  // anonymous namespace

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

std::string write_compressed_table(compressed_table const& table,
                                   std::string const& path,
                                   rmm::cuda_stream_view stream)
{
  // Build the header + payload buffer list once (shared with the in-memory
  // writer), then gather the payload into one contiguous blob for the file.
  std::vector<std::uint8_t> hdr;
  std::vector<payload_buffer_ref> buffers;
  std::uint64_t payload_bytes = 0;
  std::string err = build_compressed_table_header(table, hdr, buffers, payload_bytes, stream);
  if (!err.empty()) return err;

  std::vector<std::uint8_t> payload(static_cast<std::size_t>(payload_bytes));
  for (auto const& b : buffers) {
    if (b.size_bytes > 0 && b.device_ptr) {
      cudaMemcpyAsync(payload.data() + b.offset,
                      b.device_ptr,
                      static_cast<std::size_t>(b.size_bytes),
                      cudaMemcpyDeviceToHost,
                      stream.value());
    }
  }
  stream.synchronize();  // D→H copies must complete before the file write

  std::ofstream f(path, std::ios::binary | std::ios::trunc);
  if (!f) return "failed to open '" + path + "' for writing";
  f.write(reinterpret_cast<const char*>(hdr.data()), static_cast<std::streamsize>(hdr.size()));
  f.write(reinterpret_cast<const char*>(payload.data()),
          static_cast<std::streamsize>(payload.size()));
  if (!f) return "write error on '" + path + "'";
  return {};
}

compressed_table read_compressed_table(std::string const& path,
                                       rmm::cuda_stream_view stream,
                                       rmm::device_async_resource_ref mr,
                                       std::string* error_out)
{
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
        if (sz > 0 && cr.buf_offsets[li][bi] + sz > payload_total)
          return fail("payload out of bounds");
      }
    }
  }

  payload_fetch_fn fetch =
    [&](std::uint64_t off, std::size_t sz, void* dst, rmm::cuda_stream_view s) {
      cudaMemcpyAsync(dst, payload_base + off, sz, cudaMemcpyHostToDevice, s.value());
    };

  return reconstruct_from_records(col_records, fetch, stream, mr, error_out);
}

// ─── In-memory (pinned host) serialization ──────────────────────────────────

std::string build_compressed_table_header(compressed_table const& table,
                                          std::vector<std::uint8_t>& out_header,
                                          std::vector<payload_buffer_ref>& out_buffers,
                                          std::uint64_t& out_payload_bytes,
                                          rmm::cuda_stream_view stream)
{
  auto const all_descs = table.describe(stream);

  out_header.clear();
  out_buffers.clear();

  auto& hdr = out_header;
  hdr.push_back('H');
  hdr.push_back('P');
  hdr.push_back('L');
  hdr.push_back('N');
  push_u8(hdr, kVersion);
  push_u16le(hdr, static_cast<std::uint16_t>(table.columns.size()));

  std::uint64_t payload_offset = 0;

  static const PlanTree kEmptyTree;
  for (std::size_t ci = 0; ci < table.columns.size(); ++ci) {
    auto const& col   = table.columns[ci];
    auto const& descs = all_descs[ci];

    push_str16(hdr, col.name.value_or(std::string{}));
    push_u8(hdr, dtype_to_tag(col.dtype));
    push_i32le(hdr, col.dtype.scale());  // fixed-point scale (0 for non-decimal)
    push_i64le(hdr, col.num_rows);

    // Structural plan tree (identical layout to the file header, so the same
    // parser reconstructs it): the node array is the source of truth on read.
    PlanTree const& tree = col.compound ? col.compound->tree : kEmptyTree;
    push_u16le(hdr, static_cast<std::uint16_t>(tree.nodes.size()));
    for (auto const& node : tree.nodes)
      push_node(hdr, node);

    push_u16le(hdr, static_cast<std::uint16_t>(descs.size()));
    for (auto const& ld : descs) {
      push_u32le(hdr, ld.node_index);
      push_i32le(hdr, ld.slot);
      push_u8(hdr, static_cast<std::uint8_t>(ld.kind));
      push_u8(hdr, ld.type_tag);
      push_meta(hdr, ld.meta);
      push_u8(hdr, static_cast<std::uint8_t>(ld.buffers.size()));

      for (auto const& bd : ld.buffers) {
        push_str16(hdr, bd.name);
        push_u8(hdr, bd.type_tag);
        push_u64le(hdr, bd.size_bytes);
        push_u64le(hdr, payload_offset);

        // Record the buffer for the caller to stage out of device memory; no
        // bytes are copied here.
        out_buffers.push_back(payload_buffer_ref{payload_offset, bd.device_ptr, bd.size_bytes});
        payload_offset += bd.size_bytes;
      }
    }
  }

  out_payload_bytes = payload_offset;
  return {};
}

compressed_table read_compressed_table_from_memory(std::span<const std::uint8_t> header,
                                                   payload_fetch_fn const& fetch,
                                                   rmm::cuda_stream_view stream,
                                                   rmm::device_async_resource_ref mr,
                                                   std::string* error_out)
{
  Reader r{header.data(), header.size()};
  std::vector<ColRecord> col_records;
  if (!parse_hpln_header(r, col_records, error_out)) return {};
  return reconstruct_from_records(col_records, fetch, stream, mr, error_out);
}

}  // namespace simpatico
