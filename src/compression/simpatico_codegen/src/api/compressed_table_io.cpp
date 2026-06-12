// SPDX-License-Identifier: Apache-2.0
//
// C++-native .hpln v6 write/read for compressed_table.
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
static void push_str32(std::vector<std::uint8_t>& v, std::string const& s)
{
  push_u32le(v, static_cast<std::uint32_t>(s.size()));
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
  bool read_str32(std::string& s)
  {
    std::uint32_t len;
    if (!read_u32le(len)) return false;
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
static std::unique_ptr<compressed_representation> rep_from_leaf_desc(
  leaf_desc const& ld,
  cudf::size_type col_num_rows,
  std::vector<std::vector<std::uint8_t>> const& payloads,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr,
  std::string* err)
{
  auto const& bufs = ld.buffers;

  // Helper: allocate a device column from the i-th payload host blob.
  auto make_col = [&](std::size_t i) -> std::unique_ptr<cudf::column> {
    auto const& bd     = bufs[i];
    auto const& data   = payloads[i];
    cudf::data_type dt = tag_to_dtype(bd.type_tag);
    auto col           = cudf::make_numeric_column(
      dt, static_cast<cudf::size_type>(bd.num_rows), cudf::mask_state::UNALLOCATED, stream);
    if (!data.empty()) {
      cudaMemcpyAsync(col->mutable_view().head<void>(),
                      data.data(),
                      data.size(),
                      cudaMemcpyHostToDevice,
                      stream.value());
    }
    return col;
  };

  // Delta and Rle can be either a non-fused C++ rep or a JIT codegen_fused_representation.
  // Detect fused reps by their channel names:
  //   - Non-fused delta has a single "differences" channel.
  //   - Non-fused rle has "values" + "runs" channels.
  //   - Fused reps carry all the manifest buffers for the entire fused region.
  auto make_fused_rep = [&](const char* kind_tag) {
    auto rep = std::make_unique<codegen_fused_representation>(
      kind_tag, tag_to_dtype(ld.type_tag), col_num_rows);
    for (std::size_t i = 0; i < bufs.size(); ++i) {
      rep->buffers.emplace_back(bufs[i].name, make_col(i));
    }
    return std::unique_ptr<compressed_representation>(std::move(rep));
  };

  if (ld.kind == PlanLeafKind::Delta) {
    bool is_fused = !(bufs.size() == 1 && bufs[0].name == "differences");
    if (is_fused) return make_fused_rep("delta");
  }
  if (ld.kind == PlanLeafKind::Rle) {
    bool is_fused = !(bufs.size() == 2 && bufs[0].name == "values" && bufs[1].name == "runs");
    if (is_fused) return make_fused_rep("rle");
  }
  if (ld.kind == PlanLeafKind::Identity) {
    bool is_fused = !(bufs.size() == 1 && bufs[0].name == "data");
    if (is_fused) return make_fused_rep("RawFused");
  }

  // All other kinds (and non-fused delta/rle/identity): route through reconstruct_representation.
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

static constexpr char kEndMarker[]     = "\n---END-HEADERS-V6---\n";
static constexpr std::uint8_t kVersion = 6;

}  // anonymous namespace

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

std::string write_compressed_table(compressed_table const& table, std::string const& path)
{
  auto const all_descs = table.describe();

  // Build human-readable DSL header.
  std::string dsl_section;
  for (std::size_t ci = 0; ci < table.columns.size(); ++ci) {
    if (ci > 0) dsl_section += "---\n";
    std::string const& dsl =
      table.columns[ci].compound ? table.columns[ci].compound->plan_dsl : std::string{};
    dsl_section += dsl;
    if (!dsl.empty() && dsl.back() != '\n') dsl_section += '\n';
  }

  std::vector<std::uint8_t> hdr;
  hdr.push_back('H');
  hdr.push_back('P');
  hdr.push_back('L');
  hdr.push_back('N');
  push_u8(hdr, kVersion);
  push_u16le(hdr, static_cast<std::uint16_t>(table.columns.size()));

  std::vector<std::uint8_t> payload;
  std::uint64_t payload_offset = 0;

  for (std::size_t ci = 0; ci < table.columns.size(); ++ci) {
    auto const& col   = table.columns[ci];
    auto const& descs = all_descs[ci];

    push_str16(hdr, col.name.value_or(std::string{}));
    push_u8(hdr, dtype_to_tag(col.dtype));
    push_i64le(hdr, col.num_rows);
    std::string const& dsl = col.compound ? col.compound->plan_dsl : std::string{};
    push_str32(hdr, dsl);
    push_u16le(hdr, static_cast<std::uint16_t>(descs.size()));

    for (auto const& ld : descs) {
      push_str16(hdr, ld.path);
      push_u8(hdr, static_cast<std::uint8_t>(ld.kind));
      push_u8(hdr, ld.type_tag);
      push_meta(hdr, ld.meta);
      push_u8(hdr, static_cast<std::uint8_t>(ld.buffers.size()));

      for (auto const& bd : ld.buffers) {
        push_str16(hdr, bd.name);
        push_u8(hdr, bd.type_tag);
        push_u64le(hdr, bd.size_bytes);
        push_u64le(hdr, payload_offset);

        std::size_t old_sz = payload.size();
        payload.resize(old_sz + static_cast<std::size_t>(bd.size_bytes));
        if (bd.size_bytes > 0 && bd.device_ptr) {
          cudaMemcpyAsync(payload.data() + old_sz,
                          bd.device_ptr,
                          static_cast<std::size_t>(bd.size_bytes),
                          cudaMemcpyDeviceToHost,
                          cudaStreamDefault);
        }
        payload_offset += bd.size_bytes;
      }
    }
  }

  // All async D→H copies above used cudaStreamDefault; sync before writing.
  cudaStreamSynchronize(cudaStreamDefault);

  std::ofstream f(path, std::ios::binary | std::ios::trunc);
  if (!f) return "failed to open '" + path + "' for writing";
  f.write(dsl_section.data(), static_cast<std::streamsize>(dsl_section.size()));
  f.write(kEndMarker, static_cast<std::streamsize>(std::strlen(kEndMarker)));
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

  std::size_t marker_len           = std::strlen(kEndMarker);
  const std::uint8_t* marker_bytes = reinterpret_cast<const std::uint8_t*>(kEndMarker);
  bool found_marker                = false;
  std::size_t bin_start            = 0;
  for (std::size_t i = 0; i + marker_len <= file_size; ++i) {
    if (std::memcmp(raw.data() + i, marker_bytes, marker_len) == 0) {
      bin_start    = i + marker_len;
      found_marker = true;
      break;
    }
  }
  if (!found_marker) return fail("v6 end marker not found in '" + path + "'");

  Reader r{raw.data() + bin_start, file_size - bin_start};

  std::uint8_t magic[4];
  if (!r.read(magic, 4)) return fail("truncated header");
  if (magic[0] != 'H' || magic[1] != 'P' || magic[2] != 'L' || magic[3] != 'N')
    return fail("not a HPLN file");
  std::uint8_t ver;
  if (!r.read_u8(ver)) return fail("truncated header");
  if (ver != kVersion) return fail("unsupported version " + std::to_string(ver) + " (expected 6)");

  std::uint16_t num_cols;
  if (!r.read_u16le(num_cols)) return fail("truncated header");

  struct ColRecord {
    std::string name;
    std::uint8_t dtype_tag = 0;
    std::int64_t num_rows  = 0;
    std::string plan_dsl;
    std::vector<leaf_desc> leaf_descs;
    std::vector<std::vector<std::uint64_t>> buf_offsets;
  };
  std::vector<ColRecord> col_records(num_cols);

  for (std::uint16_t ci = 0; ci < num_cols; ++ci) {
    auto& cr = col_records[ci];
    if (!r.read_str16(cr.name)) return fail("truncated col name");
    if (!r.read_u8(cr.dtype_tag)) return fail("truncated col dtype");
    if (!r.read_i64le(cr.num_rows)) return fail("truncated col num_rows");
    if (!r.read_str32(cr.plan_dsl)) return fail("truncated col plan_dsl");

    std::uint16_t nl;
    if (!r.read_u16le(nl)) return fail("truncated num_leaves");
    cr.leaf_descs.resize(nl);
    cr.buf_offsets.resize(nl);

    for (std::uint16_t li = 0; li < nl; ++li) {
      auto& ld = cr.leaf_descs[li];
      if (!r.read_str16(ld.path)) return fail("truncated leaf path");
      std::uint8_t k;
      if (!r.read_u8(k)) return fail("truncated leaf kind");
      ld.kind = static_cast<PlanLeafKind>(k);
      if (!r.read_u8(ld.type_tag)) return fail("truncated leaf type_tag");
      if (!read_meta(r, ld.meta)) return fail("truncated/unknown leaf meta");

      std::uint8_t nb;
      if (!r.read_u8(nb)) return fail("truncated num_bufs");
      ld.buffers.resize(nb);
      cr.buf_offsets[li].resize(nb);

      for (std::uint8_t bi = 0; bi < nb; ++bi) {
        auto& bd = ld.buffers[bi];
        if (!r.read_str16(bd.name)) return fail("truncated buf name");
        if (!r.read_u8(bd.type_tag)) return fail("truncated buf type_tag");
        if (!r.read_u64le(bd.size_bytes)) return fail("truncated buf size_bytes");
        std::uint64_t poff;
        if (!r.read_u64le(poff)) return fail("truncated buf payload_offset");
        cr.buf_offsets[li][bi] = poff;
        bd.num_rows =
          (bd.size_bytes > 0 && bd.type_tag < 255)
            ? bd.size_bytes / static_cast<std::uint64_t>(cudf::size_of(tag_to_dtype(bd.type_tag)))
            : 0;
      }
    }
  }

  const std::uint8_t* payload_base = r.p;
  std::size_t payload_total        = r.rem;

  compressed_table result;
  result.columns.resize(num_cols);

  for (std::uint16_t ci = 0; ci < num_cols; ++ci) {
    auto const& cr = col_records[ci];
    auto& out_col  = result.columns[ci];

    if (!cr.name.empty()) out_col.name = cr.name;
    out_col.dtype    = tag_to_dtype(cr.dtype_tag);
    out_col.num_rows = cr.num_rows;

    if (cr.plan_dsl.empty()) continue;

    std::unordered_map<std::string, std::unique_ptr<compressed_representation>> leaves;

    for (std::size_t li = 0; li < cr.leaf_descs.size(); ++li) {
      auto const& ld    = cr.leaf_descs[li];
      auto const& boffs = cr.buf_offsets[li];

      std::vector<std::vector<std::uint8_t>> host_bufs(ld.buffers.size());
      for (std::size_t bi = 0; bi < ld.buffers.size(); ++bi) {
        auto const& bd    = ld.buffers[bi];
        std::uint64_t off = boffs[bi];
        std::size_t sz    = static_cast<std::size_t>(bd.size_bytes);
        if (sz > 0) {
          if (off + sz > payload_total)
            return fail("payload out of bounds for leaf '" + ld.path + "'");
          host_bufs[bi].assign(payload_base + off, payload_base + off + sz);
        }
      }

      std::string rep_err;
      auto rep = rep_from_leaf_desc(
        ld, static_cast<cudf::size_type>(cr.num_rows), host_bufs, stream, mr, &rep_err);
      if (!rep) return fail("rep_from_leaf_desc '" + ld.path + "': " + rep_err);

      leaves.emplace(ld.path, std::move(rep));
    }

    std::string compound_err;
    auto compound = plan_compound_from_leaves(cr.plan_dsl, std::move(leaves), &compound_err);
    if (!compound)
      return fail("plan_compound_from_leaves col " + std::to_string(ci) + ": " + compound_err);

    out_col.compound = std::move(compound);
  }

  return result;
}

}  // namespace simpatico
