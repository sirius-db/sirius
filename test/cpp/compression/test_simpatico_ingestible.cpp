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

// A .hpln file driven through the gpu_ingestible interface the scan operator uses.
//
// The interesting claims are that the walk terminates (a file yields one split per chunk, each
// exactly once, and then reports itself done) and that a split decodes to the values that were
// written. The second is the one worth being careful about: a wrong payload offset does not
// fault, it returns neighbouring bytes as data, so every assertion below compares decoded VALUES
// against what the writer was given. For the multi-chunk cases that extends to WHICH chunk was
// read: the key column encodes its chunk id, because a chunk emitted twice, skipped or decoded
// out of order all leave the row count looking right.

#include "catch.hpp"
#include "compression/simpatico_file_ingest.hpp"
#include "op/scan/simpatico_gpu_ingestible.hpp"
#include "operator/operator_test_utils.hpp"

#include <cudf/column/column_factories.hpp>
#include <cudf/table/table.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <cuda_runtime_api.h>

#include <duckdb/common/types/decimal.hpp>

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

namespace {

namespace fs = std::filesystem;

struct ingestible_env {
  std::unique_ptr<sirius::memory::sirius_memory_reservation_manager> mgr;
  cucascade::memory::memory_space* gpu_space;
  cucascade::memory::memory_space* host_space;

  ingestible_env()
    : mgr(sirius::test::operator_utils::initialize_memory_manager()),
      gpu_space(mgr->get_memory_space(cucascade::memory::Tier::GPU, 0)),
      host_space(mgr->get_memory_space(cucascade::memory::Tier::HOST, 0))
  {
  }
};

ingestible_env& env()
{
  static ingestible_env e;
  return e;
}

bool no_gpu()
{
  int count = 0;
  cudaGetDeviceCount(&count);
  if (count >= 1) { return false; }
  WARN("simpatico ingestible test requires a GPU — skipping");
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

constexpr int kRows = 6000;

/// A three-column file: DATE, INTEGER, DECIMAL(12,2). The declared types are not all derivable
/// from the physical ones, which is what makes the bind assertions meaningful.
std::string write_fixture(std::string const& path, std::vector<std::vector<std::int32_t>>& values)
{
  auto stream = cudf::get_default_stream();
  values      = {ramp(kRows, 3), ramp(kRows, 11), ramp(kRows, 5)};
  std::vector<std::unique_ptr<cudf::column>> cols;
  for (auto const& v : values)
    cols.push_back(int32_column(v));
  auto table = std::make_unique<cudf::table>(std::move(cols));

  duckdb::vector<duckdb::LogicalType> types{duckdb::LogicalType(duckdb::LogicalTypeId::DATE),
                                            duckdb::LogicalType(duckdb::LogicalTypeId::INTEGER),
                                            duckdb::LogicalType::DECIMAL(12, 2)};
  auto const leaf = std::string("input -> bitpack -> chunk_min, chunk_count, chunk_bits, packed\n");
  return sirius::write_table_to_hpln(table->view(),
                                     types,
                                     {"d", "n", "amt"},
                                     leaf + "---\n" + leaf + "---\n" + leaf,
                                     /*group_rows=*/1024,
                                     path,
                                     stream,
                                     rmm::mr::get_current_device_resource_ref());
}

constexpr int kChunks       = 4;
constexpr int kRowsPerChunk = 1500;

/// A multi-chunk file whose chunks are distinguishable BY VALUE: chunk c holds c*1'000'000 + i in
/// its key column. A chunk emitted twice, skipped, or decoded out of order all leave a plausible
/// row count, so the values are what the assertions compare.
///
/// @p out_expected receives the two columns flattened in chunk order, which is what a scan over
/// the file must produce.
std::string write_multi_fixture(std::string const& path,
                                int n_chunks,
                                std::vector<std::vector<std::int32_t>>& out_expected)
{
  auto stream = cudf::get_default_stream();
  out_expected.assign(2, {});
  std::vector<std::unique_ptr<cudf::table>> chunks;
  std::vector<cudf::table_view> views;
  for (int c = 0; c < n_chunks; ++c) {
    std::vector<std::int32_t> keys(kRowsPerChunk);
    std::vector<std::int32_t> vals(kRowsPerChunk);
    for (int i = 0; i < kRowsPerChunk; ++i) {
      keys[static_cast<std::size_t>(i)] = c * 1000000 + i;
      vals[static_cast<std::size_t>(i)] = i * 7 + c;
    }
    out_expected[0].insert(out_expected[0].end(), keys.begin(), keys.end());
    out_expected[1].insert(out_expected[1].end(), vals.begin(), vals.end());
    std::vector<std::unique_ptr<cudf::column>> cols;
    cols.push_back(int32_column(keys));
    cols.push_back(int32_column(vals));
    chunks.push_back(std::make_unique<cudf::table>(std::move(cols)));
    views.push_back(chunks.back()->view());
  }

  duckdb::vector<duckdb::LogicalType> types{duckdb::LogicalType(duckdb::LogicalTypeId::INTEGER),
                                            duckdb::LogicalType(duckdb::LogicalTypeId::INTEGER)};
  auto const leaf = std::string("input -> bitpack -> chunk_min, chunk_count, chunk_bits, packed\n");
  return sirius::write_tables_to_hpln(views,
                                      types,
                                      {"k", "v"},
                                      leaf + "---\n" + leaf,
                                      /*group_rows=*/1024,
                                      path,
                                      stream,
                                      rmm::mr::get_current_device_resource_ref());
}

/// Drive the split provider to exhaustion, as sirius_scan_manager's driver loop does.
std::vector<std::unique_ptr<sirius::op::scan::scan_info>> collect_splits(
  sirius::op::scan::simpatico_gpu_ingestible& ingestible)
{
  auto coalescer = ingestible.create_batch_coalescer();
  std::vector<std::unique_ptr<sirius::op::scan::scan_info>> batches;
  while (!ingestible.has_processed_all_metadata()) {
    auto task = ingestible.next_split_provider(
      [](std::string_view) -> std::shared_ptr<sirius::io::sirius_ioctx> { return nullptr; });
    if (!task) { break; }
    for (auto& b : coalescer->push(task())) {
      batches.push_back(std::move(b));
    }
  }
  for (auto& b : coalescer->flush()) {
    batches.push_back(std::move(b));
  }
  return batches;
}

}  // namespace

TEST_CASE("simpatico ingestible - binds a file to names, types and a cache identity",
          "[compression][simpatico_ingestible]")
{
  if (no_gpu()) { return; }
  auto const dir =
    fs::temp_directory_path() / ("sirius_simp_ing_bind_" + std::to_string(::getpid()));
  fs::create_directories(dir);
  auto const path = (dir / "t.hpln").string();
  std::vector<std::vector<std::int32_t>> values;
  REQUIRE(write_fixture(path, values).empty());

  auto ingestible = sirius::op::scan::make_ingestible(
    sirius::op::scan::bind_simpatico_file(path, *env().host_space));
  auto const& info = ingestible->table_info();

  REQUIRE(info.column_names().size() == 3);
  REQUIRE(std::vector<std::string>(info.column_names().begin(), info.column_names().end()) ==
          std::vector<std::string>{"d", "n", "amt"});
  // file_paths() is what the pinned-entry cache matches on, so it must be the .hpln itself and
  // nothing derived from it.
  REQUIRE(info.file_paths().size() == 1);
  REQUIRE(info.file_paths()[0] == path);

  // Every column of the file is materialized, in file order.
  REQUIRE(ingestible->materialized_column_order() == std::vector<std::size_t>{0, 1, 2});

  // The declared types come back, including the DECIMAL precision the physical types cannot carry.
  auto const& bound = static_cast<sirius::op::scan::simpatico_ingestible_table_info const&>(info);
  REQUIRE(bound.num_rows == kRows);
  REQUIRE(bound.types[0].id() == duckdb::LogicalTypeId::DATE);
  REQUIRE(bound.types[1].id() == duckdb::LogicalTypeId::INTEGER);
  REQUIRE(duckdb::DecimalType::GetWidth(bound.types[2]) == 12);

  fs::remove_all(dir);
}

TEST_CASE("simpatico ingestible - the walk emits one split and then terminates",
          "[compression][simpatico_ingestible]")
{
  if (no_gpu()) { return; }
  auto const dir =
    fs::temp_directory_path() / ("sirius_simp_ing_walk_" + std::to_string(::getpid()));
  fs::create_directories(dir);
  auto const path = (dir / "t.hpln").string();
  std::vector<std::vector<std::int32_t>> values;
  REQUIRE(write_fixture(path, values).empty());

  auto ingestible = sirius::op::scan::make_ingestible(
    sirius::op::scan::bind_simpatico_file(path, *env().host_space));

  // A walk that never reports itself done hangs the driver loop, and one that emits no split
  // leaves the pipeline waiting for a completion signal that never fires.
  REQUIRE_FALSE(ingestible->has_processed_all_metadata());
  auto batches = collect_splits(*ingestible);
  REQUIRE(batches.size() == 1);
  REQUIRE(ingestible->has_processed_all_metadata());
  // The claim is one-shot: a second caller gets nothing rather than a duplicate of the file.
  REQUIRE(ingestible->next_split_provider(
            [](std::string_view) -> std::shared_ptr<sirius::io::sirius_ioctx> {
              return nullptr;
            }) == nullptr);

  auto const* split = dynamic_cast<sirius::op::scan::simpatico_scan_info const*>(batches[0].get());
  REQUIRE(split != nullptr);
  REQUIRE(split->path == path);
  // Three fixed-width 4-byte columns: the reservation estimate is exact here, which is the only
  // case where it can be.
  REQUIRE(split->estimated_bytes() == static_cast<std::size_t>(kRows) * 3 * sizeof(std::int32_t));

  fs::remove_all(dir);
}

TEST_CASE("simpatico ingestible - a split materializes to the values that were written",
          "[compression][simpatico_ingestible]")
{
  if (no_gpu()) { return; }
  auto stream = cudf::get_default_stream();
  auto const dir =
    fs::temp_directory_path() / ("sirius_simp_ing_mat_" + std::to_string(::getpid()));
  fs::create_directories(dir);
  auto const path = (dir / "t.hpln").string();
  std::vector<std::vector<std::int32_t>> values;
  REQUIRE(write_fixture(path, values).empty());

  auto ingestible = sirius::op::scan::make_ingestible(
    sirius::op::scan::bind_simpatico_file(path, *env().host_space));
  auto batches = collect_splits(*ingestible);
  REQUIRE(batches.size() == 1);

  auto materialized = ingestible->materialize_metadata_to_table(
    *batches[0], *env().gpu_space, stream, false, nullptr);
  // Nothing was filtered or projected during the decode; saying otherwise would make the scan
  // operator skip post_filter_and_project.
  REQUIRE(materialized.state == sirius::op::scan::filter_state::UNFILTERED);
  REQUIRE(materialized.predicate_columns.empty());
  REQUIRE(materialized.table.view().num_columns() == 3);
  REQUIRE(materialized.table.view().num_rows() == kRows);

  SECTION("post_filter_and_project passes the whole table through")
  {
    auto table = ingestible->post_filter_and_project(
      std::move(materialized), *env().gpu_space, stream, false, nullptr, nullptr, {});
    stream.synchronize();
    REQUIRE(table->num_columns() == 3);
    REQUIRE(table->num_rows() == kRows);
    for (std::size_t c = 0; c < 3; ++c) {
      REQUIRE(read_back(table->view().column(static_cast<cudf::size_type>(c))) == values[c]);
    }
  }

  SECTION("an elided position is dropped, and the rest keep their values")
  {
    // The caller overwrites the elided output itself (a late-materialization rowid), so the copy
    // must not pay for it -- and must not shift the columns that remain.
    std::vector<std::size_t> const elided{1};
    auto table = ingestible->post_filter_and_project(
      std::move(materialized), *env().gpu_space, stream, false, nullptr, nullptr, elided);
    stream.synchronize();
    REQUIRE(table->num_columns() == 2);
    REQUIRE(read_back(table->view().column(0)) == values[0]);
    REQUIRE(read_back(table->view().column(1)) == values[2]);
  }

  fs::remove_all(dir);
}

TEST_CASE("simpatico ingestible - a projection decodes only the columns it names",
          "[compression][simpatico_ingestible]")
{
  if (no_gpu()) { return; }
  auto stream = cudf::get_default_stream();
  auto const dir =
    fs::temp_directory_path() / ("sirius_simp_ing_proj_" + std::to_string(::getpid()));
  fs::create_directories(dir);
  auto const path = (dir / "t.hpln").string();
  std::vector<std::vector<std::int32_t>> values;
  REQUIRE(write_fixture(path, values).empty());

  // Out of file order and skipping a column: the plan projects by output POSITION, so emission
  // order has to be the order asked for rather than the file's.
  auto info        = sirius::op::scan::bind_simpatico_file(path, *env().host_space);
  info->column_ids = {2, 0};
  auto ingestible  = sirius::op::scan::make_ingestible(std::move(info));

  REQUIRE(ingestible->materialized_column_order() == std::vector<std::size_t>{2, 0});

  auto batches = collect_splits(*ingestible);
  REQUIRE(batches.size() == 1);
  // Two columns' worth of decode, not three -- the reservation must not be sized for the file.
  REQUIRE(batches[0]->estimated_bytes() ==
          static_cast<std::size_t>(kRows) * 2 * sizeof(std::int32_t));

  auto materialized = ingestible->materialize_metadata_to_table(
    *batches[0], *env().gpu_space, stream, false, nullptr);
  REQUIRE(materialized.table.view().num_columns() == 2);
  auto table = ingestible->post_filter_and_project(
    std::move(materialized), *env().gpu_space, stream, false, nullptr, nullptr, {});
  stream.synchronize();
  REQUIRE(table->num_columns() == 2);
  REQUIRE(read_back(table->view().column(0)) == values[2]);
  REQUIRE(read_back(table->view().column(1)) == values[0]);

  fs::remove_all(dir);
}

TEST_CASE("simpatico ingestible - refuses what it cannot serve",
          "[compression][simpatico_ingestible]")
{
  if (no_gpu()) { return; }
  auto const dir =
    fs::temp_directory_path() / ("sirius_simp_ing_bad_" + std::to_string(::getpid()));
  fs::create_directories(dir);

  auto const junk = (dir / "junk.hpln").string();
  {
    std::ofstream f(junk, std::ios::binary);
    f << "not a simpatico file at all";
  }
  REQUIRE_THROWS(sirius::op::scan::bind_simpatico_file(junk, *env().host_space));
  REQUIRE_THROWS(
    sirius::op::scan::bind_simpatico_file((dir / "missing.hpln").string(), *env().host_space));

  // A staging space is not optional: an ingestible built without one would throw on its first
  // split instead, long after the query committed to running on the GPU.
  auto const path = (dir / "t.hpln").string();
  std::vector<std::vector<std::int32_t>> values;
  REQUIRE(write_fixture(path, values).empty());
  auto info        = sirius::op::scan::bind_simpatico_file(path, *env().host_space);
  info->host_space = nullptr;
  REQUIRE_THROWS(sirius::op::scan::make_ingestible(std::move(info)));

  // simpatico::decompress does not bounds-check its selection: an out-of-range column index would
  // return a neighbouring column's buffers as data rather than fail.
  auto out_of_range        = sirius::op::scan::bind_simpatico_file(path, *env().host_space);
  out_of_range->column_ids = {0, 3};
  REQUIRE_THROWS(sirius::op::scan::make_ingestible(std::move(out_of_range)));

  auto no_columns        = sirius::op::scan::bind_simpatico_file(path, *env().host_space);
  no_columns->column_ids = {};
  REQUIRE_THROWS(sirius::op::scan::make_ingestible(std::move(no_columns)));

  fs::remove_all(dir);
}

TEST_CASE("simpatico ingestible - a multi-chunk file binds to the sum of its chunks",
          "[compression][simpatico_ingestible][simpatico_multichunk]")
{
  if (no_gpu()) { return; }
  auto const dir =
    fs::temp_directory_path() / ("sirius_simp_multi_bind_" + std::to_string(::getpid()));
  fs::create_directories(dir);
  auto const path = (dir / "t.hpln").string();
  std::vector<std::vector<std::int32_t>> expected;
  REQUIRE(write_multi_fixture(path, kChunks, expected).empty());

  auto info = sirius::op::scan::bind_simpatico_file(path, *env().host_space);
  // The optimizer is given the whole file's cardinality, not one chunk's, and the walk is sized
  // by the per-chunk counts -- both come from parsing every chunk header at bind.
  REQUIRE(info->chunk_rows.size() == static_cast<std::size_t>(kChunks));
  for (auto const rows : info->chunk_rows) {
    REQUIRE(rows == kRowsPerChunk);
  }
  REQUIRE(info->num_rows == static_cast<std::int64_t>(kChunks) * kRowsPerChunk);
  REQUIRE(info->names == std::vector<std::string>{"k", "v"});

  fs::remove_all(dir);
}

TEST_CASE("simpatico ingestible - the walk emits one split per chunk, each exactly once",
          "[compression][simpatico_ingestible][simpatico_multichunk]")
{
  if (no_gpu()) { return; }
  auto const dir =
    fs::temp_directory_path() / ("sirius_simp_multi_walk_" + std::to_string(::getpid()));
  fs::create_directories(dir);
  auto const path = (dir / "t.hpln").string();
  std::vector<std::vector<std::int32_t>> expected;
  REQUIRE(write_multi_fixture(path, kChunks, expected).empty());

  auto info = sirius::op::scan::bind_simpatico_file(path, *env().host_space);
  // A budget no chunk fits in still emits every chunk -- a chunk cannot be split -- so each one
  // reaches the driver as its own batch and the walk is observable on its own.
  info->approximate_batch_size = 1;
  auto ingestible              = sirius::op::scan::make_ingestible(std::move(info));

  REQUIRE_FALSE(ingestible->has_processed_all_metadata());
  auto batches = collect_splits(*ingestible);
  REQUIRE(batches.size() == static_cast<std::size_t>(kChunks));
  REQUIRE(ingestible->has_processed_all_metadata());
  REQUIRE(ingestible->next_split_provider(
            [](std::string_view) -> std::shared_ptr<sirius::io::sirius_ioctx> {
              return nullptr;
            }) == nullptr);

  // Every chunk once and no other: a cursor that handed one out twice would still produce four
  // splits and the right total row count.
  std::vector<std::size_t> seen;
  for (auto const& b : batches) {
    auto const* split = dynamic_cast<sirius::op::scan::simpatico_scan_info const*>(b.get());
    REQUIRE(split != nullptr);
    REQUIRE(split->path == path);
    REQUIRE(split->chunk_ids.size() == 1);
    REQUIRE(split->num_rows == kRowsPerChunk);
    // Two fixed-width 4-byte columns of ONE chunk, not of the file.
    REQUIRE(split->estimated_bytes() ==
            static_cast<std::size_t>(kRowsPerChunk) * 2 * sizeof(std::int32_t));
    seen.push_back(split->chunk_ids.front());
  }
  std::sort(seen.begin(), seen.end());
  REQUIRE(seen == std::vector<std::size_t>{0, 1, 2, 3});

  fs::remove_all(dir);
}

TEST_CASE("simpatico ingestible - the coalescer bundles chunks up to its byte budget",
          "[compression][simpatico_ingestible][simpatico_multichunk]")
{
  if (no_gpu()) { return; }
  auto const dir =
    fs::temp_directory_path() / ("sirius_simp_multi_coal_" + std::to_string(::getpid()));
  fs::create_directories(dir);
  auto const path = (dir / "t.hpln").string();
  std::vector<std::vector<std::int32_t>> expected;
  REQUIRE(write_multi_fixture(path, kChunks, expected).empty());

  auto const chunk_bytes = static_cast<std::size_t>(kRowsPerChunk) * 2 * sizeof(std::int32_t);

  SECTION("a budget of two chunks yields two batches of two")
  {
    auto info                    = sirius::op::scan::bind_simpatico_file(path, *env().host_space);
    info->approximate_batch_size = 2 * chunk_bytes;
    auto ingestible              = sirius::op::scan::make_ingestible(std::move(info));
    auto batches                 = collect_splits(*ingestible);
    REQUIRE(batches.size() == 2);
    for (std::size_t b = 0; b < batches.size(); ++b) {
      auto const* split =
        dynamic_cast<sirius::op::scan::simpatico_scan_info const*>(batches[b].get());
      REQUIRE(split != nullptr);
      REQUIRE(split->chunk_ids ==
              std::vector<std::size_t>{2 * b, 2 * b + 1});  // ascending, and contiguous
      REQUIRE(split->num_rows == 2 * kRowsPerChunk);
      REQUIRE(split->estimated_bytes() == 2 * chunk_bytes);
    }
  }

  SECTION("a budget larger than the file yields one batch of everything")
  {
    auto info                    = sirius::op::scan::bind_simpatico_file(path, *env().host_space);
    info->approximate_batch_size = 1u << 30;
    auto ingestible              = sirius::op::scan::make_ingestible(std::move(info));
    auto batches                 = collect_splits(*ingestible);
    REQUIRE(batches.size() == 1);
    auto const* split =
      dynamic_cast<sirius::op::scan::simpatico_scan_info const*>(batches.front().get());
    REQUIRE(split->chunk_ids == std::vector<std::size_t>{0, 1, 2, 3});
    REQUIRE(split->num_rows == static_cast<std::int64_t>(kChunks) * kRowsPerChunk);
  }

  fs::remove_all(dir);
}

TEST_CASE("simpatico ingestible - a multi-chunk batch decodes every chunk, in file order",
          "[compression][simpatico_ingestible][simpatico_multichunk]")
{
  if (no_gpu()) { return; }
  auto stream = cudf::get_default_stream();
  auto const dir =
    fs::temp_directory_path() / ("sirius_simp_multi_mat_" + std::to_string(::getpid()));
  fs::create_directories(dir);
  auto const path = (dir / "t.hpln").string();
  std::vector<std::vector<std::int32_t>> expected;
  REQUIRE(write_multi_fixture(path, kChunks, expected).empty());

  SECTION("one batch holding every chunk concatenates them in chunk-id order")
  {
    auto info                    = sirius::op::scan::bind_simpatico_file(path, *env().host_space);
    info->approximate_batch_size = 1u << 30;
    auto ingestible              = sirius::op::scan::make_ingestible(std::move(info));
    auto batches                 = collect_splits(*ingestible);
    REQUIRE(batches.size() == 1);

    auto materialized = ingestible->materialize_metadata_to_table(
      *batches.front(), *env().gpu_space, stream, false, nullptr);
    REQUIRE(materialized.table.view().num_rows() == kChunks * kRowsPerChunk);
    auto table = ingestible->post_filter_and_project(
      std::move(materialized), *env().gpu_space, stream, false, nullptr, nullptr, {});
    stream.synchronize();
    // Compared elementwise rather than by sum: a sum survives two chunks swapped, and the key
    // column is built so that a swap, a repeat or a gap all show up here.
    REQUIRE(read_back(table->view().column(0)) == expected[0]);
    REQUIRE(read_back(table->view().column(1)) == expected[1]);
  }

  SECTION("one batch per chunk decodes each chunk's own rows")
  {
    auto info                    = sirius::op::scan::bind_simpatico_file(path, *env().host_space);
    info->approximate_batch_size = 1;
    auto ingestible              = sirius::op::scan::make_ingestible(std::move(info));
    auto batches                 = collect_splits(*ingestible);
    REQUIRE(batches.size() == static_cast<std::size_t>(kChunks));

    for (auto const& b : batches) {
      auto const* split = dynamic_cast<sirius::op::scan::simpatico_scan_info const*>(b.get());
      auto const chunk  = split->chunk_ids.front();
      auto materialized =
        ingestible->materialize_metadata_to_table(*b, *env().gpu_space, stream, false, nullptr);
      auto table = ingestible->post_filter_and_project(
        std::move(materialized), *env().gpu_space, stream, false, nullptr, nullptr, {});
      stream.synchronize();
      REQUIRE(table->num_rows() == kRowsPerChunk);
      auto const keys = read_back(table->view().column(0));
      // The whole point of the key encoding: this identifies WHICH chunk was read, so a split
      // that fetched a neighbour's payload fails here rather than returning plausible numbers.
      REQUIRE(keys.front() == static_cast<std::int32_t>(chunk) * 1000000);
      REQUIRE(keys.back() == static_cast<std::int32_t>(chunk) * 1000000 + kRowsPerChunk - 1);
      REQUIRE(keys ==
              std::vector<std::int32_t>(
                expected[0].begin() + static_cast<std::ptrdiff_t>(chunk * kRowsPerChunk),
                expected[0].begin() + static_cast<std::ptrdiff_t>((chunk + 1) * kRowsPerChunk)));
    }
  }

  fs::remove_all(dir);
}

TEST_CASE("simpatico ingestible - a projection over a multi-chunk file keeps its column order",
          "[compression][simpatico_ingestible][simpatico_multichunk]")
{
  if (no_gpu()) { return; }
  auto stream = cudf::get_default_stream();
  auto const dir =
    fs::temp_directory_path() / ("sirius_simp_multi_proj_" + std::to_string(::getpid()));
  fs::create_directories(dir);
  auto const path = (dir / "t.hpln").string();
  std::vector<std::vector<std::int32_t>> expected;
  REQUIRE(write_multi_fixture(path, kChunks, expected).empty());

  // Reversed projection over several chunks: the per-chunk decode and the concatenation both have
  // to agree about column order, and only one of them getting it right is a silent column swap.
  auto info                    = sirius::op::scan::bind_simpatico_file(path, *env().host_space);
  info->column_ids             = {1, 0};
  info->approximate_batch_size = 1u << 30;
  auto ingestible              = sirius::op::scan::make_ingestible(std::move(info));
  auto batches                 = collect_splits(*ingestible);
  REQUIRE(batches.size() == 1);

  auto materialized = ingestible->materialize_metadata_to_table(
    *batches.front(), *env().gpu_space, stream, false, nullptr);
  auto table = ingestible->post_filter_and_project(
    std::move(materialized), *env().gpu_space, stream, false, nullptr, nullptr, {});
  stream.synchronize();
  REQUIRE(table->num_columns() == 2);
  REQUIRE(read_back(table->view().column(0)) == expected[1]);
  REQUIRE(read_back(table->view().column(1)) == expected[0]);

  fs::remove_all(dir);
}

TEST_CASE("simpatico ingestible - a single-chunk reader refuses a multi-chunk file",
          "[compression][simpatico_ingestible][simpatico_multichunk]")
{
  if (no_gpu()) { return; }
  auto const dir =
    fs::temp_directory_path() / ("sirius_simp_multi_pin_" + std::to_string(::getpid()));
  fs::create_directories(dir);
  auto const path = (dir / "t.hpln").string();
  std::vector<std::vector<std::int32_t>> expected;
  REQUIRE(write_multi_fixture(path, kChunks, expected).empty());

  // read_hpln_into_pinned yields one blob, so serving a multi-chunk file through it would drop
  // every chunk but the first -- silently, and with a row count that looks like a small table.
  REQUIRE_THROWS(sirius::read_hpln_into_pinned(path, *env().host_space));

  // A single-chunk file still goes through it, so the pin path is unaffected by chunking.
  auto const one = (dir / "one.hpln").string();
  std::vector<std::vector<std::int32_t>> one_expected;
  REQUIRE(write_multi_fixture(one, 1, one_expected).empty());
  auto ingested = sirius::read_hpln_into_pinned(one, *env().host_space);
  REQUIRE(ingested.schema.columns.size() == 2);
  REQUIRE(ingested.schema.columns.front().num_rows == kRowsPerChunk);

  fs::remove_all(dir);
}
