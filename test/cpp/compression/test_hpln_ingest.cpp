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
#include <duckdb/planner/filter/constant_filter.hpp>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <memory>
#include <numeric>
#include <string>
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
