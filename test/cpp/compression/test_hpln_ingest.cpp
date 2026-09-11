/*
 * Copyright 2026, Sirius Contributors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// Ingesting a .hpln file as a pinned compressed chunk.
//
// The property under test is that ingest is a byte-for-byte staging step, not a re-encode: what
// lands in pinned memory must decode to exactly what was written. A wrong payload offset here
// would not fault -- the decode reads whatever landed in the buffer and returns neighbouring
// bytes as values -- so the assertions compare decoded VALUES, not just that a decode succeeded.

#include "catch.hpp"
#include "compression/compression_converters.hpp"
#include "compression/simpatico_file_ingest.hpp"
#include "operator/operator_test_utils.hpp"
#include "scan_manager/pinned_chunk_stats.hpp"

#include <cudf/column/column_factories.hpp>
#include <cudf/table/table.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <cuda_runtime_api.h>

#include <api/compressed_table_io.hpp>
#include <api/simpatico_codegen.hpp>
#include <cucascade/cudf/gpu_data_representation.hpp>
#include <duckdb/common/types/decimal.hpp>
#include <duckdb/planner/filter/constant_filter.hpp>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <memory>
#include <numeric>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace {

namespace fs = std::filesystem;

struct ingest_env {
  std::unique_ptr<sirius::memory::sirius_memory_reservation_manager> mgr;
  cucascade::memory::memory_space* gpu_space;
  cucascade::memory::memory_space* host_space;

  ingest_env()
    : mgr(sirius::test::operator_utils::initialize_memory_manager()),
      gpu_space(mgr->get_memory_space(cucascade::memory::Tier::GPU, 0)),
      host_space(mgr->get_memory_space(cucascade::memory::Tier::HOST, 0))
  {
  }
};

ingest_env& env()
{
  static ingest_env e;
  return e;
}

bool no_gpu()
{
  int count = 0;
  cudaGetDeviceCount(&count);
  if (count >= 1) { return false; }
  WARN("hpln ingest test requires a GPU — skipping");
  return true;
}

std::vector<std::int32_t> ramp(int n, int stride)
{
  std::vector<std::int32_t> v(static_cast<std::size_t>(n));
  for (int i = 0; i < n; ++i)
    v[static_cast<std::size_t>(i)] = i * stride + (i % 7);
  return v;
}

std::unique_ptr<cudf::column> int32_column(std::vector<std::int32_t> const& values)
{
  auto col = cudf::make_numeric_column(cudf::data_type{cudf::type_id::INT32},
                                       static_cast<cudf::size_type>(values.size()),
                                       cudf::mask_state::UNALLOCATED);
  REQUIRE(cudaMemcpy(col->mutable_view().head<std::int32_t>(),
                     values.data(),
                     values.size() * sizeof(std::int32_t),
                     cudaMemcpyHostToDevice) == cudaSuccess);
  return col;
}

std::vector<std::int32_t> read_back(cudf::column_view const& v)
{
  std::vector<std::int32_t> host(static_cast<std::size_t>(v.size()));
  REQUIRE(cudaMemcpy(host.data(),
                     v.head<std::int32_t>(),
                     host.size() * sizeof(std::int32_t),
                     cudaMemcpyDeviceToHost) == cudaSuccess);
  return host;
}

}  // namespace

TEST_CASE("hpln ingest - a written file stages into pinned memory and decodes to its values",
          "[compression][hpln_ingest]")
{
  if (no_gpu()) { return; }
  auto stream    = cudf::get_default_stream();
  auto const dir = fs::temp_directory_path() / ("sirius_hpln_" + std::to_string(::getpid()));
  fs::create_directories(dir);
  auto const path = (dir / "t.hpln").string();

  constexpr int kRows = 5000;
  auto const a = ramp(kRows, 3), b = ramp(kRows, 11);
  std::vector<std::unique_ptr<cudf::column>> cols;
  cols.push_back(int32_column(a));
  cols.push_back(int32_column(b));
  auto table = std::make_unique<cudf::table>(std::move(cols));

  auto ct = simpatico::compress_with_plan(
    table->view(),
    "input -> bitpack -> chunk_min, chunk_count, chunk_bits, packed\n---\n"
    "input -> delta -> differences\ndelta.differences -> bitpack\n",
    stream,
    rmm::mr::get_current_device_resource_ref());
  REQUIRE(simpatico::write_compressed_table(ct, path, stream).empty());

  auto ingested = sirius::read_hpln_into_pinned(path, *env().host_space);

  SECTION("the header describes what was written")
  {
    REQUIRE(ingested.schema.columns.size() == 2);
    for (auto const& c : ingested.schema.columns) {
      REQUIRE(c.num_rows == kRows);
      REQUIRE(c.compressed_bytes > 0);
    }
    // Ingest stages, it does not re-encode: the pinned payload is the file's payload.
    REQUIRE(ingested.blob->payload_bytes == ingested.schema.payload_bytes);
    REQUIRE(ingested.blob->header.size() == ingested.schema.header_bytes);
    // The file is header + payload + postscript + trailer, so it is strictly larger than the
    // data it carries; what must hold is that the data region fits inside it.
    REQUIRE(fs::file_size(path) >= ingested.schema.header_bytes + ingested.schema.payload_bytes +
                                     simpatico::kHplnTrailerBytes);
  }

  SECTION("it serves through the ordinary pinned-compressed path, values intact")
  {
    // No ingest-specific serve path: this is the same representation and the same converter a
    // pin builds, which is the whole point of staging into the pinned form.
    std::vector<std::string> names{"a", "b"};
    sirius::compressed_host_representation rep(
      *env().host_space,
      ingested.blob,
      names,
      static_cast<std::size_t>(ingested.schema.payload_bytes),
      static_cast<std::size_t>(kRows) * 2 * sizeof(std::int32_t),
      kRows);
    auto& registry = sirius::converter_registry::get();
    sirius::register_compression_converters(registry);
    auto gpu = registry.convert<cucascade::gpu_table_representation>(rep, env().gpu_space, stream);
    REQUIRE(gpu != nullptr);
    auto view = gpu->cast<cucascade::gpu_table_representation>().get_table_view();
    REQUIRE(view.num_columns() == 2);
    REQUIRE(view.num_rows() == kRows);
    REQUIRE(read_back(view.column(0)) == a);
    REQUIRE(read_back(view.column(1)) == b);
  }

  fs::remove_all(dir);
}

TEST_CASE("hpln ingest - zone maps survive the file and prune an ingested table",
          "[compression][hpln_ingest]")
{
  if (no_gpu()) { return; }
  auto stream    = cudf::get_default_stream();
  auto const dir = fs::temp_directory_path() / ("sirius_hpln_zm_" + std::to_string(::getpid()));
  fs::create_directories(dir);
  auto const path = (dir / "zm.hpln").string();

  // Values ascend, so each group of 1024 covers a narrow, known range -- which is what makes the
  // pruning assertion below exact rather than approximate.
  constexpr int kRows = 8192;
  std::vector<std::int32_t> v(kRows);
  for (int i = 0; i < kRows; ++i)
    v[static_cast<std::size_t>(i)] = i;
  std::vector<std::unique_ptr<cudf::column>> cols;
  cols.push_back(int32_column(v));
  auto table = std::make_unique<cudf::table>(std::move(cols));

  duckdb::vector<duckdb::LogicalType> types{duckdb::LogicalType(duckdb::LogicalTypeId::INTEGER)};
  auto const err =
    sirius::write_table_to_hpln(table->view(),
                                types,
                                {"v"},
                                "input -> bitpack -> chunk_min, chunk_count, chunk_bits, packed\n",
                                /*group_rows=*/1024,
                                path,
                                stream,
                                rmm::mr::get_current_device_resource_ref());
  REQUIRE(err.empty());

  auto ingested = sirius::read_hpln_into_pinned(path, *env().host_space);

  // The load-bearing claim: an ingested file prunes WITHOUT decoding anything. Nothing here
  // decompresses; the bounds came out of the file.
  REQUIRE_FALSE(ingested.group_bounds.empty());
  REQUIRE(ingested.group_bounds.group_rows() == 1024);
  REQUIRE(ingested.group_bounds.groups_in_chunk(0) == 8);

  auto const cell = ingested.group_bounds.cell(0, 0);
  REQUIRE(cell.size() == 8);
  for (std::size_t g = 0; g < 8; ++g) {
    REQUIRE(cell.valid[g] != 0);
    REQUIRE(cell.mins[g] == static_cast<std::int64_t>(g * 1024));
    REQUIRE(cell.maxs[g] == static_cast<std::int64_t>(g * 1024 + 1023));
  }

  // v < 2500 can only be in groups 0..2, so a filter evaluated against the file's own bounds
  // must drop the other five.
  auto filter  = duckdb::make_uniq<duckdb::ConstantFilter>(duckdb::ExpressionType::COMPARE_LESSTHAN,
                                                          duckdb::Value::INTEGER(2500));
  auto lowered = sirius::scan_manager::lowered_bound_filter::lower(*filter, cell.type);
  REQUIRE(lowered.has_value());
  std::vector<std::uint32_t> survivors;
  lowered->select_survivors(cell, survivors);
  REQUIRE(survivors == std::vector<std::uint32_t>{0, 1, 2});

  fs::remove_all(dir);
}

TEST_CASE("hpln ingest - the engine's logical types survive the file", "[compression][hpln_ingest]")
{
  if (no_gpu()) { return; }
  auto stream    = cudf::get_default_stream();
  auto const dir = fs::temp_directory_path() / ("sirius_hpln_lt_" + std::to_string(::getpid()));
  fs::create_directories(dir);
  auto const path = (dir / "lt.hpln").string();

  constexpr int kRows = 2048;
  std::vector<std::unique_ptr<cudf::column>> cols;
  cols.push_back(int32_column(ramp(kRows, 1)));
  cols.push_back(int32_column(ramp(kRows, 2)));
  auto table = std::make_unique<cudf::table>(std::move(cols));

  // DECIMAL is the case that motivates the segment: cuDF tracks scale but not precision, so
  // DECIMAL(12,2) and DECIMAL(18,2) are indistinguishable from the physical types alone.
  duckdb::vector<duckdb::LogicalType> types{duckdb::LogicalType(duckdb::LogicalTypeId::DATE),
                                            duckdb::LogicalType::DECIMAL(12, 2)};
  REQUIRE(sirius::write_table_to_hpln(
            table->view(),
            types,
            {"d", "amt"},
            "input -> bitpack -> chunk_min, chunk_count, chunk_bits, packed\n---\n"
            "input -> bitpack -> chunk_min, chunk_count, chunk_bits, packed\n",
            /*group_rows=*/0,
            path,
            stream,
            rmm::mr::get_current_device_resource_ref())
            .empty());

  auto const ingested = sirius::read_hpln_into_pinned(path, *env().host_space);
  REQUIRE(ingested.column_types.size() == 2);
  REQUIRE(ingested.column_types[0].id() == duckdb::LogicalTypeId::DATE);
  REQUIRE(ingested.column_types[1].id() == duckdb::LogicalTypeId::DECIMAL);
  REQUIRE(duckdb::DecimalType::GetWidth(ingested.column_types[1]) == 12);
  REQUIRE(duckdb::DecimalType::GetScale(ingested.column_types[1]) == 2);

  fs::remove_all(dir);
}

TEST_CASE("hpln ingest - a pre-trailer file still ingests", "[compression][hpln_ingest]")
{
  // The fallback path for files written before the trailer existed. Untested until now, which is
  // how a fallback quietly stops working.
  if (no_gpu()) { return; }
  auto stream    = cudf::get_default_stream();
  auto const dir = fs::temp_directory_path() / ("sirius_hpln_old_" + std::to_string(::getpid()));
  fs::create_directories(dir);
  auto const path = (dir / "old.hpln").string();

  constexpr int kRows = 3000;
  auto const v        = ramp(kRows, 3);
  std::vector<std::unique_ptr<cudf::column>> cols;
  cols.push_back(int32_column(v));
  auto table = std::make_unique<cudf::table>(std::move(cols));
  auto ct    = simpatico::compress_with_plan(
    table->view(), "input -> bitpack\n", stream, rmm::mr::get_current_device_resource_ref());

  // Write the pre-v7 shape by hand: header then payload, no postscript and no trailer.
  std::vector<std::uint8_t> hdr;
  std::vector<simpatico::payload_buffer_ref> refs;
  std::uint64_t payload_bytes = 0;
  REQUIRE(simpatico::build_compressed_table_header(ct, hdr, refs, payload_bytes, stream).empty());
  std::vector<std::uint8_t> payload(static_cast<std::size_t>(payload_bytes));
  for (auto const& b : refs) {
    if (b.size_bytes > 0 && b.device_ptr) {
      REQUIRE(cudaMemcpy(payload.data() + b.offset,
                         b.device_ptr,
                         static_cast<std::size_t>(b.size_bytes),
                         cudaMemcpyDeviceToHost) == cudaSuccess);
    }
  }
  {
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    f.write(reinterpret_cast<char const*>(hdr.data()), static_cast<std::streamsize>(hdr.size()));
    f.write(reinterpret_cast<char const*>(payload.data()),
            static_cast<std::streamsize>(payload.size()));
  }

  auto const ingested = sirius::read_hpln_into_pinned(path, *env().host_space);
  REQUIRE(ingested.schema.columns.size() == 1);
  REQUIRE(ingested.schema.columns[0].num_rows == kRows);
  REQUIRE(ingested.blob->payload_bytes == payload_bytes);
  // No segments, so no statistics and no declared types -- serve unpruned, do not fail.
  REQUIRE(ingested.group_bounds.empty());
  REQUIRE(ingested.column_types.empty());

  fs::remove_all(dir);
}

TEST_CASE("hpln schema - binds a file without touching the payload", "[compression][hpln_ingest]")
{
  // What a read_simpatico() bind needs: names and engine types, from the file, with no payload
  // staged and no GPU work. Everything downstream (the table function, an ingestible's
  // table_info, a pin) asks the same question.
  if (no_gpu()) { return; }
  auto stream    = cudf::get_default_stream();
  auto const dir = fs::temp_directory_path() / ("sirius_hpln_bind_" + std::to_string(::getpid()));
  fs::create_directories(dir);

  constexpr int kRows = 1024;
  std::vector<std::unique_ptr<cudf::column>> cols;
  cols.push_back(int32_column(ramp(kRows, 1)));
  cols.push_back(int32_column(ramp(kRows, 5)));
  auto table = std::make_unique<cudf::table>(std::move(cols));
  auto const plan =
    std::string("input -> bitpack -> chunk_min, chunk_count, chunk_bits, packed\n---\n") +
    "input -> bitpack -> chunk_min, chunk_count, chunk_bits, packed\n";

  SECTION("declared types win when the file carries them")
  {
    auto const path = (dir / "declared.hpln").string();
    duckdb::vector<duckdb::LogicalType> types{duckdb::LogicalType(duckdb::LogicalTypeId::DATE),
                                              duckdb::LogicalType::DECIMAL(12, 2)};
    REQUIRE(sirius::write_table_to_hpln(table->view(),
                                        types,
                                        {"d", "amt"},
                                        plan,
                                        0,
                                        path,
                                        stream,
                                        rmm::mr::get_current_device_resource_ref())
              .empty());

    auto const schema = sirius::read_hpln_schema(path);
    REQUIRE(schema.names == std::vector<std::string>{"d", "amt"});
    REQUIRE(schema.num_rows == kRows);
    REQUIRE(schema.types[0].id() == duckdb::LogicalTypeId::DATE);
    REQUIRE(duckdb::DecimalType::GetWidth(schema.types[1]) == 12);
    REQUIRE(duckdb::DecimalType::GetScale(schema.types[1]) == 2);
  }

  SECTION("a file with no declared types still binds, approximately")
  {
    // The physical types are all a pre-logical_types file has, so binding must degrade rather
    // than refuse -- and the degradation is visible: INT32 rather than the DATE it was written as.
    auto const path = (dir / "physical.hpln").string();
    REQUIRE(sirius::write_table_to_hpln(table->view(),
                                        {},
                                        {"d", "amt"},
                                        plan,
                                        0,
                                        path,
                                        stream,
                                        rmm::mr::get_current_device_resource_ref())
              .empty());

    auto const schema = sirius::read_hpln_schema(path);
    REQUIRE(schema.names.size() == 2);
    REQUIRE(schema.types[0].id() == duckdb::LogicalTypeId::INTEGER);
    REQUIRE(schema.types[1].id() == duckdb::LogicalTypeId::INTEGER);
  }

  fs::remove_all(dir);
}

TEST_CASE("hpln ingest - refuses a file that is not a readable .hpln", "[compression][hpln_ingest]")
{
  if (no_gpu()) { return; }
  auto const dir = fs::temp_directory_path() / ("sirius_hpln_bad_" + std::to_string(::getpid()));
  fs::create_directories(dir);

  // A partially ingested table must never become a pinned entry, so every one of these throws
  // rather than yielding a short or empty blob.
  auto const junk = (dir / "junk.hpln").string();
  {
    std::ofstream f(junk, std::ios::binary);
    f << "not a simpatico file at all";
  }
  REQUIRE_THROWS(sirius::read_hpln_into_pinned(junk, *env().host_space));

  REQUIRE_THROWS(sirius::read_hpln_into_pinned((dir / "missing.hpln").string(), *env().host_space));

  fs::remove_all(dir);
}

namespace {

constexpr int kMultiChunks       = 3;
constexpr int kMultiRowsPerChunk = 2000;

/// Write @p n_chunks two-column chunks whose key column encodes its chunk id, and return the
/// expected flattened columns. Chunk-identifying values are what make "which chunk did this
/// split read" an assertable question -- a wrong offset returns neighbouring bytes, not an error.
std::vector<std::vector<std::int32_t>> write_multi_chunk_file(std::string const& path, int n_chunks)
{
  auto stream = cudf::get_default_stream();
  std::vector<std::vector<std::int32_t>> expected(2);
  std::vector<std::unique_ptr<cudf::table>> tables;
  std::vector<cudf::table_view> views;
  for (int c = 0; c < n_chunks; ++c) {
    std::vector<std::int32_t> keys(kMultiRowsPerChunk), vals(kMultiRowsPerChunk);
    for (int i = 0; i < kMultiRowsPerChunk; ++i) {
      keys[static_cast<std::size_t>(i)] = c * 1000000 + i;
      vals[static_cast<std::size_t>(i)] = i * 3 + c;
    }
    expected[0].insert(expected[0].end(), keys.begin(), keys.end());
    expected[1].insert(expected[1].end(), vals.begin(), vals.end());
    std::vector<std::unique_ptr<cudf::column>> cols;
    cols.push_back(int32_column(keys));
    cols.push_back(int32_column(vals));
    tables.push_back(std::make_unique<cudf::table>(std::move(cols)));
    views.push_back(tables.back()->view());
  }
  duckdb::vector<duckdb::LogicalType> types{duckdb::LogicalType(duckdb::LogicalTypeId::INTEGER),
                                            duckdb::LogicalType(duckdb::LogicalTypeId::INTEGER)};
  auto const leaf = std::string("input -> bitpack -> chunk_min, chunk_count, chunk_bits, packed\n");
  REQUIRE(sirius::write_tables_to_hpln(views,
                                       types,
                                       {"k", "v"},
                                       leaf + "---\n" + leaf,
                                       /*group_rows=*/1024,
                                       path,
                                       stream,
                                       rmm::mr::get_current_device_resource_ref())
            .empty());
  return expected;
}

/// Read the segment table of a written file, as a remote reader would: one tail read.
std::vector<simpatico::hpln_segment_ref> segments_of(std::string const& path)
{
  auto const size = static_cast<std::uint64_t>(fs::file_size(path));
  std::vector<std::uint8_t> tail(std::min<std::uint64_t>(size, 64u << 10));
  std::ifstream f(path, std::ios::binary);
  f.seekg(static_cast<std::streamoff>(size - tail.size()));
  f.read(reinterpret_cast<char*>(tail.data()), static_cast<std::streamsize>(tail.size()));
  std::vector<simpatico::hpln_segment_ref> segs;
  REQUIRE(simpatico::read_hpln_postscript(tail, size, segs, nullptr).empty());
  return segs;
}

/// Flip one bit at @p offset, in place. Corruption a reader cannot distinguish from data.
void flip_byte(std::string const& path, std::uint64_t offset)
{
  std::fstream f(path, std::ios::binary | std::ios::in | std::ios::out);
  REQUIRE(f);
  f.seekg(static_cast<std::streamoff>(offset));
  char byte = 0;
  f.read(&byte, 1);
  byte ^= 0x01;
  f.seekp(static_cast<std::streamoff>(offset));
  f.write(&byte, 1);
  REQUIRE(f);
}

std::optional<simpatico::hpln_segment_ref> segment_of(std::string const& path,
                                                      simpatico::hpln_segment kind)
{
  for (auto const& sg : segments_of(path)) {
    if (sg.kind == kind) { return sg; }
  }
  return std::nullopt;
}

/// Make @p path look like a file written before checksums existed: the postscript's entry count
/// is decremented so its last segment (the checksum table) is never seen. The bytes stay where
/// they are, which is what a reader of an older file would find -- segments it does not know
/// about are simply not in the table.
void drop_checksum_segment(std::string const& path)
{
  auto const size = static_cast<std::uint64_t>(fs::file_size(path));
  std::fstream f(path, std::ios::binary | std::ios::in | std::ios::out);
  REQUIRE(f);
  f.seekg(static_cast<std::streamoff>(size - simpatico::kHplnTrailerBytes));
  std::uint64_t ps_off = 0;
  f.read(reinterpret_cast<char*>(&ps_off), sizeof(ps_off));
  f.seekg(static_cast<std::streamoff>(ps_off));
  std::uint16_t n = 0;
  f.read(reinterpret_cast<char*>(&n), sizeof(n));
  REQUIRE(n > 1);
  n -= 1;
  f.seekp(static_cast<std::streamoff>(ps_off));
  f.write(reinterpret_cast<char const*>(&n), sizeof(n));
  REQUIRE(f);
}

}  // namespace

TEST_CASE("hpln checksums - the CRC is CRC32C, and the same split any way",
          "[compression][hpln_ingest][hpln_checksums]")
{
  // The standard's check value. Writer and reader share one implementation, so a file would
  // self-verify even with a wrong polynomial or a byte-swapped hardware path -- this is what says
  // the bytes on disk mean what the format claims they mean.
  std::string_view const check = "123456789";
  REQUIRE(simpatico::hpln_crc32c(
            {reinterpret_cast<std::uint8_t const*>(check.data()), check.size()}) == 0xE3069283u);

  // A staged payload is checksummed block by block, so the running form has to equal the one-shot
  // form for ANY split -- including one that does not land on the hardware path's 8-byte stride.
  std::vector<std::uint8_t> data(4099);
  for (std::size_t i = 0; i < data.size(); ++i) {
    data[i] = static_cast<std::uint8_t>((i * 37) ^ (i >> 5));
  }
  auto const whole = simpatico::hpln_crc32c(data);
  for (std::size_t cut :
       {std::size_t{1}, std::size_t{7}, std::size_t{8}, std::size_t{1024}, std::size_t{4098}}) {
    auto crc = simpatico::hpln_crc32c({data.data(), cut});
    crc      = simpatico::hpln_crc32c({data.data() + cut, data.size() - cut}, crc);
    REQUIRE(crc == whole);
  }
}

TEST_CASE("hpln checksums - a corrupt payload is an error, not wrong values",
          "[compression][hpln_ingest][hpln_checksums]")
{
  if (no_gpu()) { return; }
  auto const dir = fs::temp_directory_path() / ("sirius_hpln_crc_" + std::to_string(::getpid()));
  fs::create_directories(dir);
  auto const path = (dir / "t.hpln").string();
  write_multi_chunk_file(path, kMultiChunks);

  auto const payload = segment_of(path, simpatico::hpln_segment::payload);
  REQUIRE(payload.has_value());
  REQUIRE(segment_of(path, simpatico::hpln_segment::checksums).has_value());

  // A byte inside chunk 1's payload. Nothing structural: it decodes, and without a checksum it
  // decodes to a value that is simply wrong -- which is the failure this feature exists to turn
  // into an error.
  std::vector<simpatico::hpln_chunk_ref> chunks;
  {
    auto const dir_seg = segment_of(path, simpatico::hpln_segment::chunk_directory);
    REQUIRE(dir_seg.has_value());
    std::vector<std::uint8_t> bytes(static_cast<std::size_t>(dir_seg->bytes));
    std::ifstream f(path, std::ios::binary);
    f.seekg(static_cast<std::streamoff>(dir_seg->offset));
    f.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(dir_seg->bytes));
    REQUIRE(simpatico::unpack_hpln_chunk_directory(bytes, chunks).empty());
  }
  flip_byte(path, chunks[1].payload_offset + chunks[1].payload_bytes / 2);

  std::vector<std::size_t> const want{0, 1};
  sirius::hpln_open_options verifying;
  verifying.verify_payload = true;
  REQUIRE_THROWS_WITH(
    sirius::read_hpln_chunks_into_pinned(path, *env().host_space, want, verifying),
    Catch::Matchers::Contains("payload of chunk 1") && Catch::Matchers::Contains("CRC32C"));

  // The policy, stated as a test rather than as a comment: verification costs a pass over every
  // byte read, so it is off by default and the same corrupt file stages without complaint. The
  // metadata checks below are the ones that are always on.
  sirius::hpln_open_options unverified;
  unverified.verify_payload = false;
  REQUIRE_NOTHROW(sirius::read_hpln_chunks_into_pinned(path, *env().host_space, want, unverified));

  // Chunk 0 is untouched, so the check is per chunk rather than over the whole payload region --
  // a file-wide checksum would condemn every chunk for one bad byte.
  std::vector<std::size_t> const clean{0};
  REQUIRE_NOTHROW(sirius::read_hpln_chunks_into_pinned(path, *env().host_space, clean, verifying));

  fs::remove_all(dir);
}

TEST_CASE("hpln checksums - corrupt metadata is refused without being asked",
          "[compression][hpln_ingest][hpln_checksums]")
{
  if (no_gpu()) { return; }
  auto const dir =
    fs::temp_directory_path() / ("sirius_hpln_crc_meta_" + std::to_string(::getpid()));
  fs::create_directories(dir);

  // Metadata is small and decides WHERE the payload is read from, so it is verified on every
  // open, whatever verify_payload says. Each of these is a separate file: the first failure
  // aborts the open, so corrupting them together would only ever exercise one.
  auto corrupt_segment = [&](char const* name, simpatico::hpln_segment kind, char const* expected) {
    auto const path = (dir / (std::string(name) + ".hpln")).string();
    write_multi_chunk_file(path, kMultiChunks);
    auto const sg = segment_of(path, kind);
    REQUIRE(sg.has_value());
    REQUIRE(sg->bytes > 0);
    flip_byte(path, sg->offset + sg->bytes / 2);
    REQUIRE_THROWS_WITH(sirius::read_hpln_schema(path), Catch::Matchers::Contains(expected));
  };

  corrupt_segment("hdr", simpatico::hpln_segment::header, "header of chunk");
  corrupt_segment("zm", simpatico::hpln_segment::zone_maps, "zone_maps segment");
  corrupt_segment("lt", simpatico::hpln_segment::logical_types, "logical_types segment");
  corrupt_segment("dir", simpatico::hpln_segment::chunk_directory, "chunk_directory segment");

  fs::remove_all(dir);
}

TEST_CASE("hpln checksums - a file carrying none still reads",
          "[compression][hpln_ingest][hpln_checksums]")
{
  if (no_gpu()) { return; }
  auto const dir =
    fs::temp_directory_path() / ("sirius_hpln_crc_old_" + std::to_string(::getpid()));
  fs::create_directories(dir);
  auto const path     = (dir / "t.hpln").string();
  auto const expected = write_multi_chunk_file(path, kMultiChunks);

  // Every file written before the segment existed looks like this. Refusing it -- or refusing to
  // verify anything else because one segment is missing -- would be a version bump in disguise.
  drop_checksum_segment(path);
  REQUIRE_FALSE(segment_of(path, simpatico::hpln_segment::checksums).has_value());

  auto const schema = sirius::read_hpln_schema(path);
  REQUIRE(schema.num_rows == static_cast<std::int64_t>(kMultiChunks) * kMultiRowsPerChunk);
  REQUIRE(schema.chunk_rows.size() == static_cast<std::size_t>(kMultiChunks));
  REQUIRE_FALSE(schema.group_bounds.empty());

  std::vector<std::size_t> const want{0, 1, 2};
  sirius::hpln_open_options verifying;
  verifying.verify_payload = true;
  auto const ingested =
    sirius::read_hpln_chunks_into_pinned(path, *env().host_space, want, verifying);
  REQUIRE(ingested.size() == want.size());
  // ... and a corrupt payload in such a file is exactly what it always was: unnoticed. Stated so
  // that the compatibility story is not mistaken for coverage.
  static_cast<void>(expected);

  fs::remove_all(dir);
}

TEST_CASE("hpln container - a multi-chunk file keeps its metadata segregated from its payload",
          "[compression][hpln_ingest][simpatico_multichunk]")
{
  if (no_gpu()) { return; }
  auto const dir = fs::temp_directory_path() / ("sirius_hpln_multi_" + std::to_string(::getpid()));
  fs::create_directories(dir);
  auto const path = (dir / "t.hpln").string();
  write_multi_chunk_file(path, kMultiChunks);

  auto const segs = segments_of(path);
  std::vector<simpatico::hpln_chunk_ref> chunks;
  bool found_directory = false;
  for (auto const& sg : segs) {
    if (sg.kind != simpatico::hpln_segment::chunk_directory) { continue; }
    std::vector<std::uint8_t> bytes(static_cast<std::size_t>(sg.bytes));
    std::ifstream f(path, std::ios::binary);
    f.seekg(static_cast<std::streamoff>(sg.offset));
    f.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(sg.bytes));
    REQUIRE(simpatico::unpack_hpln_chunk_directory(bytes, chunks).empty());
    found_directory = true;
  }
  REQUIRE(found_directory);
  REQUIRE(chunks.size() == static_cast<std::size_t>(kMultiChunks));

  // The claim of CHUNK_SKIPPING_PLAN.md 7.5: every chunk's metadata is reachable in ONE
  // sequential read, never interleaved with bulk data. Interleaved [hdr][pay][hdr][pay] would
  // satisfy the directory just as well and would cost a seek per chunk, so the layout is what is
  // asserted -- but SEGREGATION is the property, not "headers first". The streaming writer emits
  // payloads as chunks close and the metadata at the end, because a writer that must place the
  // header region first cannot start writing until every chunk exists, which caps a file at what
  // fits in GPU memory. Metadata at the tail is if anything better: one tail read now covers the
  // trailer, the directory, the zone maps AND every chunk header.
  for (std::size_t i = 0; i < chunks.size(); ++i) {
    REQUIRE(chunks[i].num_rows == kMultiRowsPerChunk);
    REQUIRE(chunks[i].header_bytes > 0);
    REQUIRE(chunks[i].payload_bytes > 0);
    // No header overlaps any payload, whichever side of it they fall.
    for (auto const& other : chunks) {
      bool const disjoint =
        chunks[i].header_offset + chunks[i].header_bytes <= other.payload_offset ||
        other.payload_offset + other.payload_bytes <= chunks[i].header_offset;
      REQUIRE(disjoint);
    }
    if (i + 1 < chunks.size()) {
      REQUIRE(chunks[i].header_offset + chunks[i].header_bytes == chunks[i + 1].header_offset);
      REQUIRE(chunks[i].payload_offset + chunks[i].payload_bytes == chunks[i + 1].payload_offset);
    }
  }

  // And the metadata sits at the tail, past every payload byte -- what makes the single tail read
  // cover it.
  REQUIRE(chunks.front().header_offset >=
          chunks.back().payload_offset + chunks.back().payload_bytes);

  // The header and payload segments still bound their whole regions, so a reader that wants all
  // the metadata fetches one range and subdivides it with the directory.
  for (auto const& sg : segs) {
    if (sg.kind == simpatico::hpln_segment::header) {
      REQUIRE(sg.offset == chunks.front().header_offset);
      REQUIRE(sg.offset + sg.bytes == chunks.back().header_offset + chunks.back().header_bytes);
    } else if (sg.kind == simpatico::hpln_segment::payload) {
      REQUIRE(sg.offset == chunks.front().payload_offset);
      REQUIRE(sg.offset + sg.bytes == chunks.back().payload_offset + chunks.back().payload_bytes);
    }
  }

  fs::remove_all(dir);
}

TEST_CASE("hpln container - a named chunk stages only its own bytes",
          "[compression][hpln_ingest][simpatico_multichunk]")
{
  if (no_gpu()) { return; }
  auto stream = cudf::get_default_stream();
  auto const dir =
    fs::temp_directory_path() / ("sirius_hpln_chunkrd_" + std::to_string(::getpid()));
  fs::create_directories(dir);
  auto const path     = (dir / "t.hpln").string();
  auto const expected = write_multi_chunk_file(path, kMultiChunks);

  // Out of order and with a repeat: the reader must serve exactly what was asked for, positionally
  // -- this is the contract the coalescer's batches depend on.
  std::vector<std::size_t> const want{2, 0, 2};
  auto ingested = sirius::read_hpln_chunks_into_pinned(path, *env().host_space, want);
  REQUIRE(ingested.size() == want.size());

  for (std::size_t i = 0; i < want.size(); ++i) {
    REQUIRE(ingested[i].num_rows == kMultiRowsPerChunk);
    auto const& blob = *ingested[i].blob;
    simpatico::payload_fetch_fn fetch =
      [&blob](std::uint64_t off, std::size_t sz, void* dst, rmm::cuda_stream_view s) {
        sirius::copy_pinned_blocks_to_device(*blob.payload, off, dst, sz, s);
      };
    std::string err;
    auto const ct = simpatico::read_compressed_table_from_memory(
      blob.header, fetch, stream, rmm::mr::get_current_device_resource_ref(), &err);
    REQUIRE(err.empty());
    auto table = ct.decompress(stream, rmm::mr::get_current_device_resource_ref());
    stream.synchronize();
    REQUIRE(table->num_rows() == kMultiRowsPerChunk);
    auto const keys   = read_back(table->view().column(0));
    auto const offset = static_cast<std::ptrdiff_t>(want[i] * kMultiRowsPerChunk);
    REQUIRE(keys == std::vector<std::int32_t>(expected[0].begin() + offset,
                                              expected[0].begin() + offset + kMultiRowsPerChunk));
  }

  REQUIRE_THROWS(sirius::read_hpln_chunks_into_pinned(
    path, *env().host_space, std::vector<std::size_t>{kMultiChunks}));

  fs::remove_all(dir);
}

TEST_CASE("hpln container - a file whose chunks disagree about the schema is refused",
          "[compression][hpln_ingest][simpatico_multichunk]")
{
  if (no_gpu()) { return; }
  auto stream = cudf::get_default_stream();
  auto const dir =
    fs::temp_directory_path() / ("sirius_hpln_mismatch_" + std::to_string(::getpid()));
  fs::create_directories(dir);
  auto const path = (dir / "bad.hpln").string();

  // Two chunks, one two columns wide and one three. Nothing below the container layer notices:
  // each chunk parses, each decodes, and a scan would discover the disagreement only when a batch
  // failed to concatenate -- or, with matching widths and different columns, not at all.
  constexpr int kRows = 1024;
  auto const leaf = std::string("input -> bitpack -> chunk_min, chunk_count, chunk_bits, packed\n");
  std::vector<std::unique_ptr<cudf::column>> narrow_cols;
  narrow_cols.push_back(int32_column(ramp(kRows, 3)));
  narrow_cols.push_back(int32_column(ramp(kRows, 5)));
  cudf::table narrow(std::move(narrow_cols));
  std::vector<std::unique_ptr<cudf::column>> wide_cols;
  wide_cols.push_back(int32_column(ramp(kRows, 3)));
  wide_cols.push_back(int32_column(ramp(kRows, 5)));
  wide_cols.push_back(int32_column(ramp(kRows, 7)));
  cudf::table wide(std::move(wide_cols));

  auto mr = rmm::mr::get_current_device_resource_ref();
  auto ct_narrow =
    simpatico::compress_with_plan(narrow.view(), leaf + "---\n" + leaf, stream, mr, {"k", "v"});
  auto ct_wide = simpatico::compress_with_plan(
    wide.view(), leaf + "---\n" + leaf + "---\n" + leaf, stream, mr, {"k", "v", "w"});
  std::vector<simpatico::compressed_table const*> refs{&ct_narrow, &ct_wide};
  REQUIRE(simpatico::write_compressed_tables(refs, path, stream).empty());

  // Refused at BIND, before a query commits to running: a file whose chunks disagree has no one
  // schema to bind to, and discovering that mid-scan is a failed batch at best.
  REQUIRE_THROWS(sirius::read_hpln_schema(path));
  REQUIRE_THROWS(
    sirius::read_hpln_chunks_into_pinned(path, *env().host_space, std::vector<std::size_t>{0}));

  fs::remove_all(dir);
}

TEST_CASE("hpln container - a multi-chunk file's zone maps cover every chunk, in chunk order",
          "[compression][hpln_ingest][simpatico_multichunk]")
{
  if (no_gpu()) { return; }
  auto const dir =
    fs::temp_directory_path() / ("sirius_hpln_multizm_" + std::to_string(::getpid()));
  fs::create_directories(dir);
  auto const path = (dir / "t.hpln").string();
  write_multi_chunk_file(path, kMultiChunks);

  // Pruning is milestone C, but the statistics have to be written correctly NOW: a zone-map
  // segment that carried only chunk 0's bounds would look fine until a pruner indexed it by
  // chunk id and dropped live data.
  auto const segs = segments_of(path);
  sirius::scan_manager::group_bounds_arena arena;
  for (auto const& sg : segs) {
    if (sg.kind != simpatico::hpln_segment::zone_maps) { continue; }
    std::vector<std::uint8_t> bytes(static_cast<std::size_t>(sg.bytes));
    std::ifstream f(path, std::ios::binary);
    f.seekg(static_cast<std::streamoff>(sg.offset));
    f.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(sg.bytes));
    std::string err;
    arena = sirius::scan_manager::group_bounds_arena::unpack(bytes, &err);
    REQUIRE(err.empty());
  }
  REQUIRE(arena.chunk_count() == static_cast<std::size_t>(kMultiChunks));
  REQUIRE(arena.column_count() == 2);
  for (std::size_t c = 0; c < static_cast<std::size_t>(kMultiChunks); ++c) {
    auto const bounds = arena.cell(0, c);
    REQUIRE(bounds.size() == kMultiRowsPerChunk / 1024 + 1);
    // Chunk c's key column starts at c*1'000'000, so the bounds identify which chunk they
    // describe: a sidecar written chunk-major but read chunk-minor would fail here.
    REQUIRE(bounds.mins[0] == static_cast<std::int64_t>(c) * 1000000);
    REQUIRE(bounds.maxs[bounds.size() - 1] ==
            static_cast<std::int64_t>(c) * 1000000 + kMultiRowsPerChunk - 1);
  }

  fs::remove_all(dir);
}
