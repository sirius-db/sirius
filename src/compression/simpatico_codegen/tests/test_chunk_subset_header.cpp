// SPDX-License-Identifier: Apache-2.0
//
// Header synthesis for serving a subset of a compressed column's 1024-row chunks.
//
// The failure mode under test is silent: read_compressed_table_from_memory allocates whatever the
// header declares and the decode reads whatever landed there, so a size_bytes that is off by a
// word does not fault -- it returns neighbouring chunks' bits as values. So the assertions below
// are about exact bytes and exact sizes, and the round-trip checks decoded VALUES rather than
// merely that a decode succeeded.
//
// The load-bearing property is the first test: with every chunk surviving, the synthesized header
// must be byte-for-byte the original and the gather must be one range over the whole payload. That
// is what proves the subset path degrades exactly to the existing whole-table path rather than to
// something subtly different.

#include "api/compressed_table_io.hpp"
#include "api/simpatico_codegen.hpp"
#include "codegen/jit/fused_tree.hpp"
#include "codegen/plan/leaf_desc.hpp"
#include "test_utils.hpp"

#include <cudf/column/column_factories.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/table/table.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <rmm/mr/per_device_resource.hpp>

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr std::int32_t kChunkRows   = static_cast<std::int32_t>(codegen::kChunkSize);
constexpr std::int32_t kPartialRows = 37;
constexpr std::int32_t kNumRows     = 4 * kChunkRows + kPartialRows;

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

// Values whose per-chunk bit widths all differ, so a range that is off by one chunk shifts every
// later chunk's decode and cannot pass by luck.
std::vector<std::int32_t> fixture_values()
{
  std::vector<std::int32_t> host(static_cast<std::size_t>(kNumRows));
  for (std::int32_t i = 0; i < kNumRows; ++i) {
    auto const chunk = i / kChunkRows;
    auto const pos   = i % kChunkRows;
    switch (chunk) {
      case 0: host[i] = 5; break;                    // zero-bit chunk
      case 1: host[i] = 1000 + (pos & 1); break;     // one bit
      case 2: host[i] = 2000 + (pos % 1024); break;  // ten bits
      case 3: host[i] = 3000 + (pos % 64); break;    // six bits
      default: host[i] = 7000 + pos * 1000; break;   // wide partial chunk
    }
  }
  return host;
}

std::unique_ptr<cudf::table> make_table(std::vector<std::int32_t> const& host)
{
  auto col      = cudf::make_numeric_column(cudf::data_type{cudf::type_id::INT32},
                                       static_cast<cudf::size_type>(host.size()),
                                       cudf::mask_state::UNALLOCATED);
  auto const rc = cudaMemcpy(col->mutable_view().head<std::int32_t>(),
                             host.data(),
                             host.size() * sizeof(std::int32_t),
                             cudaMemcpyHostToDevice);
  if (rc != cudaSuccess) throw std::runtime_error("fixture HtoD copy failed");
  std::vector<std::unique_ptr<cudf::column>> cols;
  cols.push_back(std::move(col));
  return std::make_unique<cudf::table>(std::move(cols));
}

// A serialized table: header plus the payload staged into host memory, which is what a pin looks
// like to build_chunk_subset_header.
struct SerializedTable {
  std::vector<std::uint8_t> header;
  std::vector<std::uint8_t> payload;
};

SerializedTable serialize(simpatico::compressed_table const& table, rmm::cuda_stream_view stream)
{
  SerializedTable out;
  std::vector<simpatico::payload_buffer_ref> refs;
  std::uint64_t payload_bytes = 0;
  auto const err =
    simpatico::build_compressed_table_header(table, out.header, refs, payload_bytes, stream);
  expect(err.empty(), err.empty() ? "header build failed" : err.c_str());
  out.payload.resize(payload_bytes);
  for (auto const& ref : refs) {
    if (ref.size_bytes == 0) continue;
    auto const rc = cudaMemcpy(
      out.payload.data() + ref.offset, ref.device_ptr, ref.size_bytes, cudaMemcpyDeviceToHost);
    if (rc != cudaSuccess) throw std::runtime_error("payload DtoH copy failed");
  }
  return out;
}

// Reads the original payload host-side; this is what supplies chunk_count/chunk_bits values.
simpatico::payload_host_read_fn host_reader(SerializedTable const& src)
{
  return [&src](std::uint64_t offset, std::uint64_t size, void* dst) {
    if (offset + size > src.payload.size()) return false;
    std::memcpy(dst, src.payload.data() + offset, size);
    return true;
  };
}

// Materialize the compacted payload described by a gather list.
std::vector<std::uint8_t> apply_gather(SerializedTable const& src,
                                       std::vector<simpatico::gather_range> const& gather,
                                       std::uint64_t payload_bytes)
{
  std::vector<std::uint8_t> out(payload_bytes, 0);
  for (auto const& g : gather) {
    expect(g.src_offset + g.size <= src.payload.size(), "gather reads past the source payload");
    expect(g.dst_offset + g.size <= out.size(), "gather writes past the compacted payload");
    std::memcpy(out.data() + g.dst_offset, src.payload.data() + g.src_offset, g.size);
  }
  return out;
}

void check_gather_well_formed(std::vector<simpatico::gather_range> const& gather, char const* label)
{
  std::uint64_t prev_src_end = 0;
  std::uint64_t prev_dst_end = 0;
  for (std::size_t i = 0; i < gather.size(); ++i) {
    auto const& g = gather[i];
    expect(g.size > 0, (std::string(label) + ": empty gather range").c_str());
    if (i > 0) {
      expect(g.src_offset >= prev_src_end,
             (std::string(label) + ": gather ranges are not ascending / non-overlapping").c_str());
      expect(g.dst_offset >= prev_dst_end,
             (std::string(label) + ": gather destinations overlap").c_str());
    }
    prev_src_end = g.src_offset + g.size;
    prev_dst_end = g.dst_offset + g.size;
  }
}

std::vector<std::uint32_t> chunk_ids(std::initializer_list<std::uint32_t> ids) { return ids; }

std::vector<std::uint32_t> all_chunk_ids(std::int32_t rows)
{
  auto const n = static_cast<std::uint32_t>((rows + kChunkRows - 1) / kChunkRows);
  std::vector<std::uint32_t> out(n);
  std::iota(out.begin(), out.end(), 0u);
  return out;
}

// Sum of the declared buffer sizes in a header, obtained by decoding it with the reader itself:
// read_compressed_table_from_memory fetches exactly size_bytes per buffer, so recording the fetch
// calls is the honest way to observe what the header claims.
struct FetchLog {
  std::vector<std::pair<std::uint64_t, std::uint64_t>> calls;  // offset, size
  std::uint64_t total() const
  {
    std::uint64_t n = 0;
    for (auto const& c : calls)
      n += c.second;
    return n;
  }
};

simpatico::compressed_table decode_from(std::vector<std::uint8_t> const& header,
                                        std::vector<std::uint8_t> const& payload,
                                        FetchLog* log,
                                        rmm::cuda_stream_view stream)
{
  std::string err;
  auto fetch =
    [&](std::uint64_t offset, std::size_t size, void* dst_device, rmm::cuda_stream_view s) {
      if (log) log->calls.emplace_back(offset, size);
      expect(offset + size <= payload.size(), "reader fetched past the compacted payload");
      auto const rc = cudaMemcpyAsync(
        dst_device, payload.data() + offset, size, cudaMemcpyHostToDevice, s.value());
      if (rc != cudaSuccess) throw std::runtime_error("payload HtoD copy failed");
    };
  auto table = simpatico::read_compressed_table_from_memory(
    header, fetch, stream, rmm::mr::get_current_device_resource_ref(), &err);
  expect(err.empty(), err.empty() ? "read_compressed_table_from_memory failed" : err.c_str());
  return table;
}

std::vector<std::int32_t> decode_values(simpatico::compressed_table const& table,
                                        rmm::cuda_stream_view stream)
{
  auto out = simpatico::decompress(table, stream, rmm::mr::get_current_device_resource_ref());
  expect(out != nullptr && out->num_columns() == 1, "decompress produced no column");
  auto const view = out->view().column(0);
  std::vector<std::int32_t> host(static_cast<std::size_t>(view.size()));
  if (!host.empty()) {
    auto const rc = cudaMemcpy(host.data(),
                               view.head<std::int32_t>(),
                               host.size() * sizeof(std::int32_t),
                               cudaMemcpyDeviceToHost);
    if (rc != cudaSuccess) throw std::runtime_error("result DtoH copy failed");
  }
  return host;
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

constexpr char const* kBitpackPlan =
  "input -> bitpack -> chunk_min, chunk_count, chunk_bits, packed\n";

// The property everything else rests on. If this holds, a subset header is the same object the
// writer produces, only smaller.
void test_all_chunks_reproduces_the_original()
{
  auto const stream = cudf::get_default_stream();
  auto const host   = fixture_values();
  auto const input  = make_table(host);
  auto compressed   = simpatico::compress_with_plan(
    input->view(), kBitpackPlan, stream, rmm::mr::get_current_device_resource_ref());
  auto const src = serialize(compressed, stream);

  std::vector<std::uint8_t> header;
  std::vector<simpatico::gather_range> gather;
  std::uint64_t payload_bytes = 0;
  auto const survivors        = all_chunk_ids(kNumRows);
  auto const err              = simpatico::build_chunk_subset_header(
    src.header, survivors, host_reader(src), header, gather, /*max_gap_bytes=*/0, &payload_bytes);
  expect(err.empty(), err.empty() ? "subset header build failed" : err.c_str());

  expect(header == src.header, "every chunk surviving must reproduce the original header exactly");
  expect(payload_bytes == src.payload.size(), "compacted payload size differs from the original");
  expect(gather.size() == 1, "every chunk surviving must gather one range");
  expect(gather.size() == 1 && gather[0].src_offset == 0 && gather[0].dst_offset == 0 &&
           gather[0].size == src.payload.size(),
         "the single range must cover the whole payload");
}

void test_subset_sizes_and_gather()
{
  auto const stream = cudf::get_default_stream();
  auto const host   = fixture_values();
  auto const input  = make_table(host);
  auto compressed   = simpatico::compress_with_plan(
    input->view(), kBitpackPlan, stream, rmm::mr::get_current_device_resource_ref());
  auto const src = serialize(compressed, stream);

  // Chunks 1 and 3 only: non-adjacent, skipping both the zero-bit chunk and the short tail.
  auto const survivors = chunk_ids({1, 3});
  std::vector<std::uint8_t> header;
  std::vector<simpatico::gather_range> gather;
  std::uint64_t payload_bytes = 0;
  auto const err              = simpatico::build_chunk_subset_header(
    src.header, survivors, host_reader(src), header, gather, /*max_gap_bytes=*/0, &payload_bytes);
  expect(err.empty(), err.empty() ? "subset header build failed" : err.c_str());

  expect(header.size() == src.header.size(),
         "patching a header in place must not change its length");
  expect(header != src.header, "a genuine subset must restate sizes and offsets");
  expect(payload_bytes < src.payload.size(), "a subset payload must be smaller than the whole");
  check_gather_well_formed(gather, "subset");

  // The reader fetches exactly the declared sizes; their sum is the compacted payload minus the
  // bytes no range covers (nothing here -- the guard words are filled from the source).
  FetchLog log;
  auto const compacted = apply_gather(src, gather, payload_bytes);
  auto const table     = decode_from(header, compacted, &log, stream);
  expect(log.total() == payload_bytes,
         "declared buffer sizes must sum to the compacted payload size");

  std::uint64_t gathered = 0;
  for (auto const& g : gather)
    gathered += g.size;
  expect(gathered == payload_bytes, "gathered bytes must equal the compacted payload size");

  expect(table.columns.size() == 1, "subset table lost its column");
  expect(table.columns[0].num_rows == 2 * kChunkRows,
         "subset column num_rows must be the surviving chunks' rows");
}

// Round-trip: the compacted table must decode to exactly the surviving rows of the original.
void test_roundtrip_values(std::vector<std::uint32_t> const& survivors, char const* label)
{
  auto const stream = cudf::get_default_stream();
  auto const host   = fixture_values();
  auto const input  = make_table(host);
  auto compressed   = simpatico::compress_with_plan(
    input->view(), kBitpackPlan, stream, rmm::mr::get_current_device_resource_ref());
  auto const src = serialize(compressed, stream);

  std::vector<std::uint8_t> header;
  std::vector<simpatico::gather_range> gather;
  std::uint64_t payload_bytes = 0;
  auto const err              = simpatico::build_chunk_subset_header(
    src.header, survivors, host_reader(src), header, gather, /*max_gap_bytes=*/0, &payload_bytes);
  expect(err.empty(), err.empty() ? "subset header build failed" : err.c_str());
  check_gather_well_formed(gather, label);

  auto const compacted = apply_gather(src, gather, payload_bytes);
  auto const table     = decode_from(header, compacted, nullptr, stream);
  auto const got       = decode_values(table, stream);

  std::vector<std::int32_t> want;
  for (auto const chunk : survivors) {
    auto const first = static_cast<std::int32_t>(chunk) * kChunkRows;
    auto const last  = std::min(first + kChunkRows, kNumRows);
    for (std::int32_t i = first; i < last; ++i)
      want.push_back(host[i]);
  }
  expect(got.size() == want.size(),
         (std::string(label) + ": decoded row count is not the surviving rows").c_str());
  expect(got == want, (std::string(label) + ": decoded values are not the surviving rows").c_str());
}

// A column whose root buffer is whole_column (snappy) cannot be subsetted, so it must be emitted
// whole -- correct, merely less selective -- while a subsettable column beside it is compacted.
void test_whole_column_is_emitted_whole()
{
  auto const stream = cudf::get_default_stream();
  auto const host   = fixture_values();

  auto bitpacked = simpatico::compress_with_plan(
    make_table(host)->view(), kBitpackPlan, stream, rmm::mr::get_current_device_resource_ref());
  auto snappied = simpatico::compress_with_plan(make_table(host)->view(),
                                                "input -> snappy\n",
                                                stream,
                                                rmm::mr::get_current_device_resource_ref());

  simpatico::compressed_table both;
  both.columns.push_back(std::move(bitpacked.columns[0]));
  both.columns.push_back(std::move(snappied.columns[0]));
  auto const src = serialize(both, stream);

  // Control: the fixture round-trips unmodified, so a later failure is the subset's fault.
  expect(decode_from(src.header, src.payload, nullptr, stream).columns.size() == 2,
         "the mixed fixture does not round-trip unmodified");

  auto const survivors = chunk_ids({1, 3});
  std::vector<std::uint8_t> header;
  std::vector<simpatico::gather_range> gather;
  std::uint64_t payload_bytes = 0;
  std::vector<std::uint8_t> column_subsetted;
  auto const err = simpatico::build_chunk_subset_header(src.header,
                                                        survivors,
                                                        host_reader(src),
                                                        header,
                                                        gather,
                                                        /*max_gap_bytes=*/0,
                                                        &payload_bytes,
                                                        &column_subsetted);
  expect(err.empty(), err.empty() ? "subset header build failed" : err.c_str());
  check_gather_well_formed(gather, "mixed");

  // The per-column report is how a caller avoids assembling a compacted column and a whole one
  // into one table: their row counts differ, and nothing downstream would tell it apart from a
  // correct table until the values were wrong.
  expect(column_subsetted == std::vector<std::uint8_t>{1, 0},
         "the per-column subsetted report does not match what was emitted");

  auto const compacted = apply_gather(src, gather, payload_bytes);
  auto const table     = decode_from(header, compacted, nullptr, stream);
  expect(table.columns.size() == 2, "mixed table lost a column");
  expect(table.columns[0].num_rows == 2 * kChunkRows,
         "the bitpack column should have been subsetted");
  expect(table.columns[1].num_rows == kNumRows, "the snappy column should have been emitted whole");

  // And both still decode. They are decoded one at a time because the two columns no longer have
  // the same length -- which is the point: mixing a compacted column with a whole one is exactly
  // what "emit the unsubsettable column whole" produces.
  std::string sub_err;
  auto fetch = [&](std::uint64_t offset, std::size_t size, void* dst, rmm::cuda_stream_view s) {
    expect(offset + size <= compacted.size(), "reader fetched past the compacted payload");
    if (cudaMemcpyAsync(dst, compacted.data() + offset, size, cudaMemcpyHostToDevice, s.value()) !=
        cudaSuccess) {
      throw std::runtime_error("payload HtoD copy failed");
    }
  };

  std::vector<std::size_t> const first{0};
  auto subsetted = simpatico::read_compressed_table_subset_from_memory(
    header, fetch, first, stream, rmm::mr::get_current_device_resource_ref(), &sub_err);
  expect(sub_err.empty(), sub_err.empty() ? "column 0 read failed" : sub_err.c_str());
  std::vector<std::int32_t> want;
  for (auto const chunk : survivors) {
    auto const begin = static_cast<std::int32_t>(chunk) * kChunkRows;
    for (std::int32_t i = begin; i < std::min(begin + kChunkRows, kNumRows); ++i) {
      want.push_back(host[i]);
    }
  }
  expect(decode_values(subsetted, stream) == want,
         "the subsetted column did not decode to the surviving rows");

  std::vector<std::size_t> const second{1};
  auto whole = simpatico::read_compressed_table_subset_from_memory(
    header, fetch, second, stream, rmm::mr::get_current_device_resource_ref(), &sub_err);
  expect(sub_err.empty(), sub_err.empty() ? "column 1 read failed" : sub_err.c_str());
  expect(decode_values(whole, stream) == host,
         "the whole-emitted column did not decode to every row");
}

// ---------------------------------------------------------------------------
// A nullable column beside a subsettable one.
//
// The validity mask is a payload buffer that hangs off no leaf, so the per-leaf loops that
// relocate everything else never see it. If it is not relocated and re-pointed explicitly, the
// reader parses whatever leaf bytes landed at its stale offset AS a null mask: the values still
// decode correctly and nothing faults, only the wrong rows come back null. So the assertion here
// is on the null POSITIONS, not merely on the values or on a successful decode.
//
// The mask also indexes the column's original rows, so a nullable column must be emitted whole:
// survivor chunk c does not name bit c's row in it.
// ---------------------------------------------------------------------------
void test_nullable_column_is_emitted_whole_with_its_mask()
{
  auto const stream = cudf::get_default_stream();
  auto const host   = fixture_values();

  // Nulls spread across every chunk, at positions that differ per chunk, so a mask read at the
  // wrong offset cannot reproduce this pattern by luck.
  std::vector<bool> want_valid(static_cast<std::size_t>(kNumRows), true);
  auto nullable_tbl = make_table(host);
  {
    auto& col = nullable_tbl->get_column(0);
    col.set_null_mask(
      cudf::create_null_mask(
        kNumRows, cudf::mask_state::ALL_VALID, stream, rmm::mr::get_current_device_resource_ref()),
      0);
    cudf::size_type nulls = 0;
    for (std::int32_t i = 0; i < kNumRows; ++i) {
      if ((i % kChunkRows) % (7 + i / kChunkRows) != 0) continue;
      cudf::set_null_mask(col.mutable_view().null_mask(), i, i + 1, /*valid=*/false, stream);
      want_valid[static_cast<std::size_t>(i)] = false;
      ++nulls;
    }
    col.set_null_count(nulls);
    expect(nulls > 0, "the nullable fixture has no nulls");
  }

  auto nullable = simpatico::compress_with_plan(
    nullable_tbl->view(), kBitpackPlan, stream, rmm::mr::get_current_device_resource_ref());
  auto plain = simpatico::compress_with_plan(
    make_table(host)->view(), kBitpackPlan, stream, rmm::mr::get_current_device_resource_ref());

  simpatico::compressed_table both;
  both.columns.push_back(std::move(nullable.columns[0]));
  both.columns.push_back(std::move(plain.columns[0]));
  auto const src = serialize(both, stream);

  auto const survivors = chunk_ids({1, 3});
  std::vector<std::uint8_t> header;
  std::vector<simpatico::gather_range> gather;
  std::uint64_t payload_bytes = 0;
  std::vector<std::uint8_t> column_subsetted;
  auto const err = simpatico::build_chunk_subset_header(src.header,
                                                        survivors,
                                                        host_reader(src),
                                                        header,
                                                        gather,
                                                        /*max_gap_bytes=*/0,
                                                        &payload_bytes,
                                                        &column_subsetted);
  expect(err.empty(), err.empty() ? "subset header build failed" : err.c_str());
  check_gather_well_formed(gather, "nullable");

  // The nullable column is whole; the plain one beside it is still compacted, so carrying a mask
  // costs that column its fetch skip and nothing more.
  expect(column_subsetted == std::vector<std::uint8_t>{0, 1},
         "a nullable column must be emitted whole and its plain neighbour still subsetted");

  auto const compacted = apply_gather(src, gather, payload_bytes);

  std::string sub_err;
  auto fetch = [&](std::uint64_t offset, std::size_t size, void* dst, rmm::cuda_stream_view s) {
    expect(offset + size <= compacted.size(), "reader fetched past the compacted payload");
    if (cudaMemcpyAsync(dst, compacted.data() + offset, size, cudaMemcpyHostToDevice, s.value()) !=
        cudaSuccess) {
      throw std::runtime_error("payload HtoD copy failed");
    }
  };

  std::vector<std::size_t> const first{0};
  auto whole = simpatico::read_compressed_table_subset_from_memory(
    header, fetch, first, stream, rmm::mr::get_current_device_resource_ref(), &sub_err);
  expect(sub_err.empty(), sub_err.empty() ? "nullable column read failed" : sub_err.c_str());
  expect(decode_values(whole, stream) == host,
         "the nullable column did not decode to every row's value");

  auto decoded = simpatico::decompress(whole, stream, rmm::mr::get_current_device_resource_ref());
  expect(decoded != nullptr && decoded->num_columns() == 1,
         "nullable decompress produced no column");
  expect(host_validity_bits(decoded->view().column(0)) == want_valid,
         "the relocated validity mask marks the wrong rows null");

  std::vector<std::size_t> const second{1};
  auto subsetted = simpatico::read_compressed_table_subset_from_memory(
    header, fetch, second, stream, rmm::mr::get_current_device_resource_ref(), &sub_err);
  expect(sub_err.empty(), sub_err.empty() ? "plain column read failed" : sub_err.c_str());
  std::vector<std::int32_t> want;
  for (auto const chunk : survivors) {
    auto const begin = static_cast<std::int32_t>(chunk) * kChunkRows;
    for (std::int32_t i = begin; i < std::min(begin + kChunkRows, kNumRows); ++i) {
      want.push_back(host[i]);
    }
  }
  expect(decode_values(subsetted, stream) == want,
         "the plain column beside a nullable one did not decode to the surviving rows");
}

// ---------------------------------------------------------------------------
// Dictionary-encoded strings: the shape that blocks the scan-bound TPC-H queries.
//
// A dictionary's keys are column-wide STATE -- they stay valid for whatever subset of the indices
// is served -- while the indices are one entry per row. That is what lets a dictionary column be
// chunk-addressed at all, and it is a different fact from "this buffer is opaque" (snappy), which
// still cannot be. The three plans below are the ones TPC-H actually uses: bare, indices bitpacked
// (l_returnflag), and keys bitpacked as well (o_orderpriority).
// ---------------------------------------------------------------------------

// Low-cardinality strings whose value identifies the chunk AND the position within it, so a subset
// that is off by a chunk decodes to values from the wrong chunk rather than passing by luck.
std::vector<std::string> dictionary_fixture_values()
{
  static char const* const kKeys[] = {"AIR", "RAIL", "SHIP", "TRUCK", "MAIL", "FOB", "REG AIR"};
  std::vector<std::string> out(static_cast<std::size_t>(kNumRows));
  for (std::int32_t i = 0; i < kNumRows; ++i) {
    auto const chunk                 = i / kChunkRows;
    auto const pos                   = i % kChunkRows;
    out[static_cast<std::size_t>(i)] = kKeys[(chunk * 3 + pos) % std::size(kKeys)];
  }
  return out;
}

void test_dictionary_shape(char const* dsl, char const* label)
{
  auto const stream = cudf::get_default_stream();
  auto const host   = dictionary_fixture_values();
  auto const input  = make_strings_table(host, {}, stream);
  auto compressed   = simpatico::compress_with_plan(
    input->view(), dsl, stream, rmm::mr::get_current_device_resource_ref());
  auto const src = serialize(compressed, stream);

  // Every chunk surviving must reproduce the original header byte for byte: with the keys emitted
  // whole and the indices compacted to everything, the subset path has to be the existing path.
  {
    std::vector<std::uint8_t> header;
    std::vector<simpatico::gather_range> gather;
    std::uint64_t payload_bytes = 0;
    std::vector<std::uint8_t> column_subsetted;
    auto const err = simpatico::build_chunk_subset_header(src.header,
                                                          all_chunk_ids(kNumRows),
                                                          host_reader(src),
                                                          header,
                                                          gather,
                                                          /*max_gap_bytes=*/0,
                                                          &payload_bytes,
                                                          &column_subsetted);
    expect(err.empty(), err.empty() ? "subset header build failed" : err.c_str());
    expect(column_subsetted == std::vector<std::uint8_t>{1},
           (std::string(label) + ": the column was not subsetted at all").c_str());
    expect(header == src.header,
           (std::string(label) + ": all chunks surviving must reproduce the header").c_str());
    expect(payload_bytes == src.payload.size(),
           (std::string(label) + ": all chunks surviving must reproduce the payload size").c_str());
  }

  auto const survivors = chunk_ids({1, 3});
  std::vector<std::uint8_t> header;
  std::vector<simpatico::gather_range> gather;
  std::uint64_t payload_bytes = 0;
  auto const err              = simpatico::build_chunk_subset_header(
    src.header, survivors, host_reader(src), header, gather, /*max_gap_bytes=*/0, &payload_bytes);
  expect(err.empty(), err.empty() ? "subset header build failed" : err.c_str());
  check_gather_well_formed(gather, label);

  auto const compacted = apply_gather(src, gather, payload_bytes);
  auto const table     = decode_from(header, compacted, nullptr, stream);
  auto const decoded =
    simpatico::decompress(table, stream, rmm::mr::get_current_device_resource_ref());
  expect(decoded != nullptr && decoded->num_columns() == 1, "decompress produced no column");

  std::vector<std::string> want;
  for (auto const chunk : survivors) {
    auto const first = static_cast<std::int32_t>(chunk) * kChunkRows;
    auto const last  = std::min(first + kChunkRows, kNumRows);
    for (std::int32_t i = first; i < last; ++i) {
      want.push_back(host[static_cast<std::size_t>(i)]);
    }
  }
  auto const expected = make_strings_column(want, {}, stream);
  expect(decoded->view().column(0).size() == static_cast<cudf::size_type>(want.size()),
         (std::string(label) + ": decoded row count is not the surviving rows").c_str());
  expect(strings_equal(decoded->view().column(0), expected->view(), stream),
         (std::string(label) + ": decoded values are not the surviving rows").c_str());
}

void test_malformed_input()
{
  auto const stream = cudf::get_default_stream();
  auto const input  = make_table(fixture_values());
  auto compressed   = simpatico::compress_with_plan(
    input->view(), kBitpackPlan, stream, rmm::mr::get_current_device_resource_ref());
  auto const src = serialize(compressed, stream);

  std::vector<std::uint8_t> header;
  std::vector<simpatico::gather_range> gather;

  auto const unsorted = chunk_ids({3, 1});
  expect(
    !simpatico::build_chunk_subset_header(src.header, unsorted, host_reader(src), header, gather)
       .empty(),
    "unsorted survivors must be rejected");

  expect(
    !simpatico::build_chunk_subset_header(src.header, {}, host_reader(src), header, gather).empty(),
    "an empty survivor set must be rejected");

  std::vector<std::uint8_t> truncated(src.header.begin(), src.header.begin() + 8);
  expect(!simpatico::build_chunk_subset_header(
            truncated, all_chunk_ids(kNumRows), host_reader(src), header, gather)
            .empty(),
         "a truncated header must be rejected");
}

// Without a payload reader the bitpack `packed` buffer cannot be sized, so the column must fall
// back to whole rather than guess.
void test_missing_sizing_metadata_falls_back_to_whole()
{
  auto const stream = cudf::get_default_stream();
  auto const input  = make_table(fixture_values());
  auto compressed   = simpatico::compress_with_plan(
    input->view(), kBitpackPlan, stream, rmm::mr::get_current_device_resource_ref());
  auto const src = serialize(compressed, stream);

  std::vector<std::uint8_t> header;
  std::vector<simpatico::gather_range> gather;
  std::uint64_t payload_bytes = 0;
  auto const err              = simpatico::build_chunk_subset_header(
    src.header, chunk_ids({1, 3}), nullptr, header, gather, /*max_gap_bytes=*/0, &payload_bytes);
  expect(err.empty(), err.empty() ? "subset header build failed" : err.c_str());
  expect(header == src.header, "a column with no sizing metadata must be emitted whole");
  expect(payload_bytes == src.payload.size(), "a whole fallback must carry the whole payload");
}

}  // namespace

int main()
{
  try {
    test_all_chunks_reproduces_the_original();
    test_subset_sizes_and_gather();
    test_roundtrip_values(chunk_ids({1, 3}), "non-adjacent subset");
    test_roundtrip_values(chunk_ids({0}), "zero-bit chunk alone");
    test_roundtrip_values(chunk_ids({2, 3, 4}), "adjacent subset including the short tail");
    test_roundtrip_values(all_chunk_ids(kNumRows), "all chunks");
    test_whole_column_is_emitted_whole();
    test_nullable_column_is_emitted_whole_with_its_mask();
    test_dictionary_shape("input -> dictionary\n", "bare dictionary");
    test_dictionary_shape(
      "input -> dictionary -> keys_offsets, keys_chars, indices\n"
      "dictionary.indices -> bitpack -> chunk_min, chunk_count, chunk_bits, packed\n",
      "dictionary with bitpacked indices");
    test_dictionary_shape(
      "input -> dictionary -> keys_offsets, keys_chars, indices\n"
      "dictionary.keys_offsets -> bitpack -> chunk_min, chunk_count, chunk_bits, packed\n"
      "dictionary.indices -> bitpack -> chunk_min, chunk_count, chunk_bits, packed\n",
      "dictionary with bitpacked keys and indices");
    test_malformed_input();
    test_missing_sizing_metadata_falls_back_to_whole();
  } catch (std::exception const& e) {
    std::fprintf(stderr, "test_chunk_subset_header: exception: %s\n", e.what());
    return 1;
  }
  std::printf("test_chunk_subset_header: PASS\n");
  return 0;
}
