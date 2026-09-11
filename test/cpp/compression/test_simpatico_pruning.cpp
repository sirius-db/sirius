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

// A .hpln scan pruned by the zone maps the file already carries.
//
// A scan that prunes nothing still returns the right rows, so "the results are correct" proves
// nothing here. Every case below therefore asserts on WHAT WAS READ -- which chunks became splits,
// how many 1024-row decode chunks survived inside them, and whether the subset header was actually
// used -- as well as on the values, whose key column encodes its chunk id.
//
// The dangerous direction is the other one: a chunk wrongly dropped is silent data loss, not a
// failure. So each pruning case is paired with an assertion that the rows that MUST survive are
// all present, and there is a case whose predicate matches exactly one row of the last chunk.

#include "catch.hpp"
#include "compression/simpatico_file_ingest.hpp"
#include "helper/type_conversions.hpp"
#include "op/scan/simpatico_gpu_ingestible.hpp"
#include "operator/operator_test_utils.hpp"

#include <cudf/column/column_factories.hpp>
#include <cudf/table/table.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <cuda_runtime_api.h>

#include <duckdb/planner/filter/conjunction_filter.hpp>
#include <duckdb/planner/filter/constant_filter.hpp>
#include <duckdb/planner/table_filter.hpp>

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace {

namespace fs = std::filesystem;

namespace scan = sirius::op::scan;

struct pruning_env {
  std::unique_ptr<sirius::memory::sirius_memory_reservation_manager> mgr;
  cucascade::memory::memory_space* gpu_space;
  cucascade::memory::memory_space* host_space;

  pruning_env()
    : mgr(sirius::test::operator_utils::initialize_memory_manager()),
      gpu_space(mgr->get_memory_space(cucascade::memory::Tier::GPU, 0)),
      host_space(mgr->get_memory_space(cucascade::memory::Tier::HOST, 0))
  {
  }
};

pruning_env& env()
{
  static pruning_env e;
  return e;
}

bool no_gpu()
{
  int count = 0;
  cudaGetDeviceCount(&count);
  if (count >= 1) { return false; }
  WARN("simpatico pruning test requires a GPU — skipping");
  return true;
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

constexpr int kChunks       = 4;
constexpr int kRowsPerChunk = 1500;  // two 1024-row decode chunks, and two 1024-row groups
constexpr int kGroupRows    = 1024;

/// Chunk c holds keys c*1'000'000 + i, ascending, so a chunk's zone-map bounds are exactly
/// [c*1'000'000, c*1'000'000 + 1499] and its two groups are [.., +1023] and [+1024, +1499].
/// That makes both which chunk and which GROUP survived a predicate a matter of arithmetic
/// rather than of inspection.
std::string write_fixture(std::string const& path)
{
  auto stream = cudf::get_default_stream();
  std::vector<std::unique_ptr<cudf::table>> chunks;
  std::vector<cudf::table_view> views;
  for (int c = 0; c < kChunks; ++c) {
    std::vector<std::int32_t> keys(kRowsPerChunk);
    std::vector<std::int32_t> vals(kRowsPerChunk);
    for (int i = 0; i < kRowsPerChunk; ++i) {
      keys[static_cast<std::size_t>(i)] = c * 1000000 + i;
      vals[static_cast<std::size_t>(i)] = i * 7 + c;
    }
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
                                      kGroupRows,
                                      path,
                                      stream,
                                      rmm::mr::get_current_device_resource_ref());
}

duckdb::unique_ptr<duckdb::TableFilter> ge(std::int32_t v)
{
  return duckdb::make_uniq<duckdb::ConstantFilter>(
    duckdb::ExpressionType::COMPARE_GREATERTHANOREQUALTO, duckdb::Value::INTEGER(v));
}

duckdb::unique_ptr<duckdb::TableFilter> lt(std::int32_t v)
{
  return duckdb::make_uniq<duckdb::ConstantFilter>(duckdb::ExpressionType::COMPARE_LESSTHAN,
                                                   duckdb::Value::INTEGER(v));
}

duckdb::unique_ptr<duckdb::TableFilter> both(duckdb::unique_ptr<duckdb::TableFilter> a,
                                             duckdb::unique_ptr<duckdb::TableFilter> b)
{
  auto conj = duckdb::make_uniq<duckdb::ConjunctionAndFilter>();
  conj->child_filters.push_back(std::move(a));
  conj->child_filters.push_back(std::move(b));
  return conj;
}

/// Bind @p path and attach @p filter on column 0 ("k"), the way the plan generator does.
std::unique_ptr<scan::simpatico_ingestible_table_info> bind_with_filter(
  std::string const& path, duckdb::unique_ptr<duckdb::TableFilter> filter)
{
  auto info = scan::bind_simpatico_file(path, *env().host_space);
  info->duckdb_column_ids.emplace_back(0);
  info->duckdb_column_ids.emplace_back(1);
  info->returned_types = sirius::from_duckdb_vec(info->types);
  if (filter) {
    info->table_filters             = duckdb::make_uniq<duckdb::TableFilterSet>();
    info->table_filters->filters[0] = std::move(filter);
  }
  return info;
}

/// Drive the split provider to exhaustion, as sirius_scan_manager's driver loop does.
///
/// @p reverse feeds the coalescer its splits back to front. Several dispatcher threads claim
/// chunks concurrently, so the coalescer genuinely sees them out of order; a single-threaded loop
/// never does, and would pass whether or not the coalescer reorders anything.
std::vector<std::unique_ptr<scan::scan_info>> collect_splits(scan::simpatico_gpu_ingestible& ing,
                                                             bool reverse = false)
{
  std::vector<std::unique_ptr<scan::scan_info>> splits;
  while (!ing.has_processed_all_metadata()) {
    auto task = ing.next_split_provider(
      [](std::string_view) -> std::shared_ptr<sirius::io::sirius_ioctx> { return nullptr; });
    if (!task) { break; }
    splits.push_back(task());
  }
  if (reverse) { std::reverse(splits.begin(), splits.end()); }

  auto coalescer = ing.create_batch_coalescer();
  std::vector<std::unique_ptr<scan::scan_info>> batches;
  for (auto& split : splits) {
    for (auto& b : coalescer->push(std::move(split))) {
      batches.push_back(std::move(b));
    }
  }
  for (auto& b : coalescer->flush()) {
    batches.push_back(std::move(b));
  }
  return batches;
}

/// Decode every batch and return the key column's values, concatenated in emission order.
std::vector<std::int32_t> decode_keys(scan::simpatico_gpu_ingestible& ing,
                                      std::vector<std::unique_ptr<scan::scan_info>> const& batches,
                                      bool apply_filter)
{
  auto stream = cudf::get_default_stream();
  std::vector<std::int32_t> keys;
  for (auto const& b : batches) {
    auto materialized =
      ing.materialize_metadata_to_table(*b, *env().gpu_space, stream, false, nullptr);
    if (!apply_filter) {
      stream.synchronize();
      auto const got = read_back(materialized.table.view().column(0));
      keys.insert(keys.end(), got.begin(), got.end());
      continue;
    }
    auto table = ing.post_filter_and_project(
      std::move(materialized), *env().gpu_space, stream, false, nullptr, nullptr, {});
    stream.synchronize();
    auto const got = read_back(table->view().column(0));
    keys.insert(keys.end(), got.begin(), got.end());
  }
  return keys;
}

struct fixture_dir {
  fs::path dir;
  std::string path;
  explicit fixture_dir(std::string const& tag)
  {
    dir =
      fs::temp_directory_path() / ("sirius_simp_prune_" + tag + "_" + std::to_string(::getpid()));
    fs::create_directories(dir);
    path = (dir / "t.hpln").string();
    REQUIRE(write_fixture(path).empty());
  }
  ~fixture_dir() { fs::remove_all(dir); }
};

}  // namespace

TEST_CASE("simpatico pruning - the file's zone maps reach the walk",
          "[compression][simpatico_pruning]")
{
  if (no_gpu()) { return; }
  fixture_dir fx("bind");
  auto const info = scan::bind_simpatico_file(fx.path, *env().host_space);
  // Without these the scan has nothing to prune WITH, and everything below would pass by serving
  // the whole file.
  REQUIRE_FALSE(info->group_bounds.empty());
  REQUIRE(info->group_bounds.chunk_count() == static_cast<std::size_t>(kChunks));
  REQUIRE(info->group_bounds.group_rows() == static_cast<std::size_t>(kGroupRows));
  REQUIRE(info->group_bounds.column_count() == 2);
}

TEST_CASE("simpatico pruning - no filter prunes nothing", "[compression][simpatico_pruning]")
{
  if (no_gpu()) { return; }
  fixture_dir fx("nofilter");
  auto ing = scan::make_ingestible(bind_with_filter(fx.path, nullptr));

  // The control the other cases are read against: the machinery must be inert without a predicate.
  REQUIRE(ing->pruning().chunks_total == static_cast<std::size_t>(kChunks));
  REQUIRE(ing->pruning().chunks_pruned == 0);
  REQUIRE(ing->pruning().decode_chunks_pruned == 0);
  auto const batches = collect_splits(*ing);
  auto const keys    = decode_keys(*ing, batches, /*apply_filter=*/false);
  REQUIRE(keys.size() == static_cast<std::size_t>(kChunks) * kRowsPerChunk);
  REQUIRE(keys.front() == 0);
  REQUIRE(keys.back() == 3 * 1000000 + kRowsPerChunk - 1);
}

TEST_CASE("simpatico pruning - IS NULL on a never-null column empties the file",
          "[compression][simpatico_pruning]")
{
  if (no_gpu()) { return; }
  fixture_dir fx("isnull");
  auto info = bind_with_filter(fx.path, nullptr);
  // What the chunk headers record, not what the zone maps do: this holds for a file with no
  // `zone_maps` segment at all.
  REQUIRE(info->column_has_nulls.size() == 2);
  REQUIRE_FALSE(info->column_has_nulls[1]);
  // `WHERE v IS NULL`, as the scan's pushdown_complex_filter hook harvests it -- DuckDB never
  // lowers a standalone IS NULL into a TableFilter, so it cannot arrive as one.
  info->is_null_columns = {1};

  auto ing = scan::make_ingestible(std::move(info));
  // Everything but the sentinel chunk, which exists so the pipeline still sees one batch.
  REQUIRE(ing->pruning().chunks_pruned == static_cast<std::size_t>(kChunks) - 1);
  REQUIRE(ing->pruning().decode_chunks_pruned ==
          ing->pruning().decode_chunks_total -
            (static_cast<std::size_t>(kRowsPerChunk) + 1023) / 1024);

  auto const batches = collect_splits(*ing);
  auto const keys    = decode_keys(*ing, batches, /*apply_filter=*/false);
  // Only chunk 0 was read. Its rows are emitted unfiltered: the IS NULL predicate was never taken
  // out of the plan, so the filter above the scan is what empties them.
  REQUIRE(keys.size() == static_cast<std::size_t>(kRowsPerChunk));
  REQUIRE(keys.front() == 0);
}

TEST_CASE("simpatico pruning - IS NULL on a column that has nulls prunes nothing",
          "[compression][simpatico_pruning]")
{
  if (no_gpu()) { return; }
  fixture_dir fx("isnull_neg");
  auto info             = bind_with_filter(fx.path, nullptr);
  info->is_null_columns = {1};
  // The negative control for the case above: the same predicate on a column the file says CAN
  // hold a null must read every chunk. A group's bounds say nothing about which of its rows are
  // null, so there is nothing finer to decide on.
  info->column_has_nulls[1] = true;

  auto ing = scan::make_ingestible(std::move(info));
  REQUIRE(ing->pruning().chunks_pruned == 0);
  REQUIRE(ing->pruning().decode_chunks_pruned == 0);
  auto const batches = collect_splits(*ing);
  REQUIRE(decode_keys(*ing, batches, /*apply_filter=*/false).size() ==
          static_cast<std::size_t>(kChunks) * kRowsPerChunk);
}

TEST_CASE("simpatico pruning - a chunk that cannot match is never read",
          "[compression][simpatico_pruning]")
{
  if (no_gpu()) { return; }
  fixture_dir fx("whole");
  // Chunks 0 and 1 top out at 1'001'499, so neither can hold a key >= 2'000'000.
  auto ing = scan::make_ingestible(bind_with_filter(fx.path, ge(2000000)));

  REQUIRE(ing->pruning().chunks_pruned == 2);
  // Both of the pruned chunks' decode chunks are dropped whole: 2 chunks x ceil(1500/1024).
  REQUIRE(ing->pruning().decode_chunks_pruned == 4);

  auto const batches = collect_splits(*ing);
  // Unfiltered so the assertion is about what was DECODED, not about what the predicate kept:
  // chunks 2 and 3 in file order, entire, and nothing from 0 or 1.
  auto const keys = decode_keys(*ing, batches, /*apply_filter=*/false);
  REQUIRE(keys.size() == 2 * static_cast<std::size_t>(kRowsPerChunk));
  REQUIRE(keys.front() == 2000000);
  REQUIRE(keys[kRowsPerChunk - 1] == 2000000 + kRowsPerChunk - 1);
  REQUIRE(keys[kRowsPerChunk] == 3000000);
  REQUIRE(keys.back() == 3000000 + kRowsPerChunk - 1);
  // Nothing was subsetted, so no subset header was synthesized and none could have been refused.
  REQUIRE(ing->subset_refusals() == 0);
}

TEST_CASE("simpatico pruning - a chunk with one matching row is kept",
          "[compression][simpatico_pruning]")
{
  if (no_gpu()) { return; }
  fixture_dir fx("onerow");
  // The worst failure this code can have is dropping a chunk that had to survive, which is silent:
  // the query returns fewer rows and nothing faults. Exactly one row of the file matches.
  auto const only = 3 * 1000000 + kRowsPerChunk - 1;
  auto ing        = scan::make_ingestible(bind_with_filter(fx.path, ge(only)));

  REQUIRE(ing->pruning().chunks_pruned == 3);
  auto const batches = collect_splits(*ing);
  REQUIRE(batches.size() == 1);
  auto const keys = decode_keys(*ing, batches, /*apply_filter=*/true);
  REQUIRE(keys == std::vector<std::int32_t>{only});
}

TEST_CASE("simpatico pruning - a surviving chunk decodes only its surviving decode chunks",
          "[compression][simpatico_pruning]")
{
  if (no_gpu()) { return; }
  fixture_dir fx("subset");
  // k in [1024, 1'000'000) leaves only chunk 0, and inside it only the second group -- rows
  // 1024..1499, i.e. decode chunk 1 of 2. Decode chunk 0 must never be fetched.
  auto ing = scan::make_ingestible(bind_with_filter(fx.path, both(ge(1024), lt(1000000))));

  auto const stats = ing->pruning();
  REQUIRE(stats.chunks_pruned == 3);
  // Three whole chunks (2 decode chunks each) plus decode chunk 0 of the survivor.
  REQUIRE(stats.decode_chunks_total == 8);
  REQUIRE(stats.decode_chunks_pruned == 7);

  auto const batches = collect_splits(*ing);
  REQUIRE(batches.size() == 1);
  // The reservation is sized by the rows the subset will produce, not by the chunk it came from.
  REQUIRE(batches[0]->estimated_bytes() ==
          static_cast<std::size_t>(kRowsPerChunk - 1024) * 2 * sizeof(std::int32_t));

  // Unfiltered: this is the assertion that the SUBSET decoded the right rows. A subset path that
  // was silently bypassed would return all 1500 rows here and still pass the filtered check below.
  auto const decoded = decode_keys(*ing, batches, /*apply_filter=*/false);
  REQUIRE(decoded.size() == static_cast<std::size_t>(kRowsPerChunk - 1024));
  REQUIRE(decoded.front() == 1024);
  REQUIRE(decoded.back() == kRowsPerChunk - 1);
  // Every column this scan reads was chunk-addressable, so the subset was used rather than
  // refused; a refusal here would serve the chunk whole and skip nothing.
  REQUIRE(ing->subset_refusals() == 0);
}

TEST_CASE("simpatico pruning - a batch mixes a subsetted chunk with whole ones",
          "[compression][simpatico_pruning]")
{
  if (no_gpu()) { return; }
  fixture_dir fx("mixed");
  // k >= 1024 narrows chunk 0 to its second group and leaves chunks 1-3 entirely, so one batch
  // carries both shapes. The surviving-decode-chunk list belongs to its own chunk: a list that
  // drifted onto a neighbour would decode the wrong rows of it and still return a plausible count.
  auto ing = scan::make_ingestible(bind_with_filter(fx.path, ge(1024)));

  REQUIRE(ing->pruning().chunks_pruned == 0);
  REQUIRE(ing->pruning().decode_chunks_pruned == 1);

  // Out of order on purpose: the batch's rows must come back in FILE order regardless, and each
  // chunk's surviving-decode-chunk list must have travelled with it.
  auto const keys =
    decode_keys(*ing, collect_splits(*ing, /*reverse=*/true), /*apply_filter=*/false);
  std::vector<std::int32_t> expected;
  for (int i = 1024; i < kRowsPerChunk; ++i) {
    expected.push_back(i);
  }
  for (int c = 1; c < kChunks; ++c) {
    for (int i = 0; i < kRowsPerChunk; ++i) {
      expected.push_back(c * 1000000 + i);
    }
  }
  REQUIRE(keys == expected);
  REQUIRE(ing->subset_refusals() == 0);
}

TEST_CASE("simpatico pruning - the predicate is still applied after the decode",
          "[compression][simpatico_pruning]")
{
  if (no_gpu()) { return; }
  fixture_dir fx("residual");
  // Zone maps BOUND rows, they do not test them: group 1 of chunk 0 covers 1024..1499 and survives
  // k >= 1200 whole, so 176 of its rows must be removed after the decode. DuckDB deleted this
  // predicate from the plan, so if the scan does not apply it nothing else will.
  auto ing = scan::make_ingestible(bind_with_filter(fx.path, both(ge(1200), lt(1000000))));

  REQUIRE(ing->pruning().chunks_pruned == 3);
  // What the decode produced: the whole surviving group, predicate not yet applied.
  auto const decoded = decode_keys(*ing, collect_splits(*ing), /*apply_filter=*/false);
  REQUIRE(decoded.size() == static_cast<std::size_t>(kRowsPerChunk - 1024));

  // What the scan emits: the same batch with the predicate applied.
  auto ing2       = scan::make_ingestible(bind_with_filter(fx.path, both(ge(1200), lt(1000000))));
  auto const kept = decode_keys(*ing2, collect_splits(*ing2), /*apply_filter=*/true);
  REQUIRE(kept.size() == static_cast<std::size_t>(kRowsPerChunk - 1200));
  REQUIRE(kept.front() == 1200);
  REQUIRE(kept.back() == kRowsPerChunk - 1);
}

TEST_CASE("simpatico pruning - an all-pruned file still emits a split",
          "[compression][simpatico_pruning]")
{
  if (no_gpu()) { return; }
  fixture_dir fx("allpruned");
  // Zero splits means zero tasks, and the pipeline waits for a completion that never fires. The
  // sentinel chunk keeps it alive; the post-decode filter empties it.
  auto ing = scan::make_ingestible(bind_with_filter(fx.path, ge(9000000)));

  REQUIRE(ing->pruning().chunks_pruned == static_cast<std::size_t>(kChunks) - 1);
  auto const batches = collect_splits(*ing);
  REQUIRE(batches.size() == 1);
  REQUIRE(decode_keys(*ing, batches, /*apply_filter=*/true).empty());
}
