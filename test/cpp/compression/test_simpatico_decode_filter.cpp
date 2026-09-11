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

// A .hpln scan that filters DURING the decode, on top of the pruning the zone maps already do.
//
// The two layers compete on the same column and complement on different ones: a group survives if
// ANY of its rows could match, so after pruning on a clustered column its survivors are nearly all
// matches and the decode abandons compaction as unprofitable. What decode-time filtering adds is
// the predicate on a column the file is NOT clustered by, which the zone maps cannot narrow at
// all. Every case below therefore carries TWO predicates -- one on the clustered key, one on an
// unclustered value -- and asserts that both layers acted: chunks dropped before any read, and the
// surviving chunks arriving already compacted to the rows that match.
//
// The assertions are on what came back BEFORE post_filter_and_project runs, because that is the
// only way to tell the decode filtered from the scan filtering afterwards: both produce the same
// answer, and only one of them is the feature.

#include "catch.hpp"
#include "compression/decompression_pushdown_policy.hpp"
#include "compression/simpatico_file_ingest.hpp"
#include "helper/type_conversions.hpp"
#include "op/scan/simpatico_gpu_ingestible.hpp"
#include "operator/operator_test_utils.hpp"

#include <cudf/column/column_factories.hpp>
#include <cudf/table/table.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <cuda_runtime_api.h>

#include <duckdb/planner/filter/constant_filter.hpp>
#include <duckdb/planner/table_filter.hpp>

#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace {

namespace fs   = std::filesystem;
namespace scan = sirius::op::scan;

struct decode_filter_env {
  std::unique_ptr<sirius::memory::sirius_memory_reservation_manager> mgr;
  cucascade::memory::memory_space* gpu_space;
  cucascade::memory::memory_space* host_space;

  decode_filter_env()
    : mgr(sirius::test::operator_utils::initialize_memory_manager()),
      gpu_space(mgr->get_memory_space(cucascade::memory::Tier::GPU, 0)),
      host_space(mgr->get_memory_space(cucascade::memory::Tier::HOST, 0))
  {
  }
};

decode_filter_env& env()
{
  static decode_filter_env e;
  return e;
}

bool no_gpu()
{
  int count = 0;
  cudaGetDeviceCount(&count);
  if (count >= 1) { return false; }
  WARN("simpatico decode-filter test requires a GPU — skipping");
  return true;
}

/// Turn the experimental gate on, and refuse to run if it did not take.
///
/// The gate is read live (not cached), so setting it here is enough regardless of what any other
/// test did first. Asserting it afterwards is what makes every case below actually about the
/// feature: without the assertion a cached-off gate would route them through the OLD path and they
/// would pass while proving nothing.
void require_pushdown_gate()
{
  ::setenv("SIRIUS_EXP_FUSED_SCAN_FILTER", "1", /*overwrite=*/1);
  REQUIRE(sirius::decompression_pushdown_enabled());
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

/// The unclustered column. 919 is coprime with 1000, so each 1000 consecutive rows are a
/// permutation of 0..999: every group's bounds span nearly the whole domain and no predicate on
/// `v` can prune anything. That is the point -- it leaves the win entirely to the decode.
std::int32_t value_at(int row) { return (row * 919) % 1000; }

/// Chunk c holds keys c*1'000'000 + i, ascending, so a chunk's zone-map bounds are exactly
/// [c*1'000'000, c*1'000'000 + 1499] -- which chunk and which GROUP a key predicate leaves is
/// arithmetic rather than inspection.
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
      vals[static_cast<std::size_t>(i)] = value_at(i);
    }
    std::vector<std::unique_ptr<cudf::column>> cols;
    cols.push_back(int32_column(keys));
    cols.push_back(int32_column(vals));
    chunks.push_back(std::make_unique<cudf::table>(std::move(cols)));
    views.push_back(chunks.back()->view());
  }
  duckdb::vector<duckdb::LogicalType> types{duckdb::LogicalType(duckdb::LogicalTypeId::INTEGER),
                                            duckdb::LogicalType(duckdb::LogicalTypeId::INTEGER)};
  // A bitpack root is what lets a column both PRODUCE the decode's row mask and come back
  // compacted; a plan the decode cannot mask on would decline every case here for the wrong
  // reason.
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

/// Bind @p path with a filter on "k" (column 0) and one on "v" (column 1), the way the plan
/// generator does. Either may be null.
std::unique_ptr<scan::simpatico_ingestible_table_info> bind_with_filters(
  std::string const& path,
  duckdb::unique_ptr<duckdb::TableFilter> on_key,
  duckdb::unique_ptr<duckdb::TableFilter> on_value)
{
  auto info = scan::bind_simpatico_file(path, *env().host_space);
  info->duckdb_column_ids.emplace_back(0);
  info->duckdb_column_ids.emplace_back(1);
  info->returned_types = sirius::from_duckdb_vec(info->types);
  if (on_key || on_value) {
    info->table_filters = duckdb::make_uniq<duckdb::TableFilterSet>();
    if (on_key) { info->table_filters->filters[0] = std::move(on_key); }
    if (on_value) { info->table_filters->filters[1] = std::move(on_value); }
  }
  return info;
}

/// Drive the split provider to exhaustion and coalesce, as the scan manager's driver loop does.
std::vector<std::unique_ptr<scan::scan_info>> collect_splits(scan::simpatico_gpu_ingestible& ing)
{
  std::vector<std::unique_ptr<scan::scan_info>> splits;
  while (!ing.has_processed_all_metadata()) {
    auto task = ing.next_split_provider(
      [](std::string_view) -> std::shared_ptr<sirius::io::sirius_ioctx> { return nullptr; });
    if (!task) { break; }
    splits.push_back(task());
  }
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

/// (k, v) pairs of one scan, and how the source described them.
struct scan_result {
  std::vector<std::int32_t> keys;
  std::vector<std::int32_t> values;
  /// True iff EVERY batch came back tagged as already row-filtered.
  bool all_row_filtered = true;
};

/// Decode every batch, optionally running the post-decode filter afterwards.
///
/// With @p apply_filter false this is what the DECODE produced, which is the assertion that
/// matters: a scan whose decode-time filtering was silently bypassed returns every row of the
/// surviving chunks here and still answers correctly once the post-decode filter runs.
scan_result run_scan(scan::simpatico_gpu_ingestible& ing,
                     std::vector<std::unique_ptr<scan::scan_info>> const& batches,
                     bool apply_filter)
{
  auto stream = cudf::get_default_stream();
  scan_result out;
  for (auto const& b : batches) {
    auto materialized =
      ing.materialize_metadata_to_table(*b, *env().gpu_space, stream, false, nullptr);
    out.all_row_filtered =
      out.all_row_filtered && materialized.state == scan::filter_state::ROW_FILTERED;
    if (!apply_filter) {
      stream.synchronize();
      auto const keys   = read_back(materialized.table.view().column(0));
      auto const values = read_back(materialized.table.view().column(1));
      out.keys.insert(out.keys.end(), keys.begin(), keys.end());
      out.values.insert(out.values.end(), values.begin(), values.end());
      continue;
    }
    auto table = ing.post_filter_and_project(
      std::move(materialized), *env().gpu_space, stream, false, nullptr, nullptr, {});
    stream.synchronize();
    auto const keys   = read_back(table->view().column(0));
    auto const values = read_back(table->view().column(1));
    out.keys.insert(out.keys.end(), keys.begin(), keys.end());
    out.values.insert(out.values.end(), values.begin(), values.end());
  }
  return out;
}

/// Every (k, v) of the file satisfying `k >= key_lo AND v < value_hi`, in file order.
scan_result expected_rows(std::int32_t key_lo, std::int32_t value_hi)
{
  scan_result out;
  for (int c = 0; c < kChunks; ++c) {
    for (int i = 0; i < kRowsPerChunk; ++i) {
      auto const k = c * 1000000 + i;
      auto const v = value_at(i);
      if (k >= key_lo && v < value_hi) {
        out.keys.push_back(k);
        out.values.push_back(v);
      }
    }
  }
  return out;
}

struct fixture_dir {
  fs::path dir;
  std::string path;
  explicit fixture_dir(std::string const& tag)
  {
    dir =
      fs::temp_directory_path() / ("sirius_simp_decfilt_" + tag + "_" + std::to_string(::getpid()));
    fs::create_directories(dir);
    path = (dir / "t.hpln").string();
    REQUIRE(write_fixture(path).empty());
  }
  ~fixture_dir() { fs::remove_all(dir); }
};

}  // namespace

TEST_CASE("simpatico decode filter - pruning and decode-time filtering both act",
          "[compression][simpatico_decode_filter]")
{
  if (no_gpu()) { return; }
  require_pushdown_gate();
  fixture_dir fx("both");
  // `k >= 2'000'000` is the layer the zone maps own: chunks 0 and 1 top out at 1'001'499 and are
  // never read. `v < 50` is the layer they cannot touch -- every group of every chunk spans the
  // whole value domain -- and it is what the decode carries, keeping about 5% of the rows, well
  // under the selectivity ceiling at which compaction stops paying.
  auto ing = scan::make_ingestible(bind_with_filters(fx.path, ge(2000000), lt(50)));

  REQUIRE(ing->pruning().chunks_pruned == 2);
  REQUIRE(ing->pruning().decode_chunks_pruned == 4);

  auto const batches  = collect_splits(*ing);
  auto const expected = expected_rows(2000000, 50);
  REQUIRE(expected.keys.size() > 0);

  // What the DECODE produced, with no post-decode filter run at all. Both predicates have already
  // been applied to it: the key range the decode carried as well as the value range.
  auto const decoded = run_scan(*ing, batches, /*apply_filter=*/false);
  REQUIRE(decoded.keys == expected.keys);
  REQUIRE(decoded.values == expected.values);
  REQUIRE(decoded.all_row_filtered);

  // Both surviving chunks were offered to the decode and both came back carrying the whole filter,
  // so the post-decode filter is skipped for them.
  auto const pushdown = ing->decode_filtering();
  REQUIRE(pushdown.chunks_offered == 2);
  REQUIRE(pushdown.chunks_row_filtered == 2);
  REQUIRE(pushdown.chunks_unprofitable == 0);
}

TEST_CASE("simpatico decode filter - a row-filtered batch is not filtered again",
          "[compression][simpatico_decode_filter]")
{
  if (no_gpu()) { return; }
  require_pushdown_gate();
  fixture_dir fx("skip");
  auto ing = scan::make_ingestible(bind_with_filters(fx.path, ge(2000000), lt(50)));

  // The answer must be identical whether or not post_filter_and_project has a filter left to run.
  // It skips it here (the batch is tagged ROW_FILTERED), and the rows still have to be exactly the
  // query's -- a batch wrongly tagged would return rows the predicate excludes.
  auto const filtered = run_scan(*ing, collect_splits(*ing), /*apply_filter=*/true);
  auto const expected = expected_rows(2000000, 50);
  REQUIRE(filtered.keys == expected.keys);
  REQUIRE(filtered.values == expected.values);
  REQUIRE(ing->decode_filtering().chunks_row_filtered == 2);
}

TEST_CASE("simpatico decode filter - a subsetted chunk is filtered during its decode",
          "[compression][simpatico_decode_filter]")
{
  if (no_gpu()) { return; }
  require_pushdown_gate();
  fixture_dir fx("subset");
  // `k >= 2'001'024` drops chunks 0 and 1 outright and narrows chunk 2 to its second group, so
  // that chunk decodes from a SYNTHESIZED SUBSET header describing 476 rows rather than 1500.
  // for_chunk narrows the request against that subset's compression plans, and the decode's own
  // planning reads the same trees -- this is the case where the table the decode sees is not the
  // chunk the file holds. Chunk 3 travels whole in the same scan, so both shapes are covered.
  auto ing = scan::make_ingestible(bind_with_filters(fx.path, ge(2001024), lt(50)));

  REQUIRE(ing->pruning().chunks_pruned == 2);
  REQUIRE(ing->pruning().decode_chunks_pruned == 5);  // 2 whole chunks, plus one of chunk 2's

  auto const decoded  = run_scan(*ing, collect_splits(*ing), /*apply_filter=*/false);
  auto const expected = expected_rows(2001024, 50);
  REQUIRE(expected.keys.size() > 0);
  REQUIRE(decoded.keys == expected.keys);
  REQUIRE(decoded.values == expected.values);
  REQUIRE(decoded.all_row_filtered);
  // The subset header was used rather than refused, so the decode really did run against a
  // compacted table.
  REQUIRE(ing->subset_refusals() == 0);
  REQUIRE(ing->decode_filtering().chunks_row_filtered == 2);
}

TEST_CASE("simpatico decode filter - the post-decode filter still applies when the decode declines",
          "[compression][simpatico_decode_filter]")
{
  if (no_gpu()) { return; }
  require_pushdown_gate();
  fixture_dir fx("unprofitable");
  // `v < 900` keeps ~90% of the rows: far above the ceiling at which compacting pays for itself,
  // so the decode measures the survivors and hands back ordinary full-width columns. That is a
  // normal outcome, not a failure -- but it makes the post-decode filter mandatory, because
  // nothing above the scan re-checks a predicate DuckDB deleted from the plan.
  auto ing = scan::make_ingestible(bind_with_filters(fx.path, ge(2000000), lt(900)));

  auto const batches = collect_splits(*ing);
  auto const decoded = run_scan(*ing, batches, /*apply_filter=*/false);
  REQUIRE_FALSE(decoded.all_row_filtered);
  // Nothing was dropped by the decode: every row of the two surviving chunks came back.
  REQUIRE(decoded.keys.size() == 2 * static_cast<std::size_t>(kRowsPerChunk));

  auto const pushdown = ing->decode_filtering();
  REQUIRE(pushdown.chunks_unprofitable >= 1);
  REQUIRE(pushdown.chunks_row_filtered == 0);
  // Selectivity barely varies between one file's chunks, so the first chunk to measure it
  // unprofitable ends the attempt for the rest: the second chunk is never offered.
  REQUIRE(pushdown.chunks_offered == 1);

  // The scan is still responsible for the whole predicate.
  auto ing2           = scan::make_ingestible(bind_with_filters(fx.path, ge(2000000), lt(900)));
  auto const filtered = run_scan(*ing2, collect_splits(*ing2), /*apply_filter=*/true);
  auto const expected = expected_rows(2000000, 900);
  REQUIRE(filtered.keys == expected.keys);
  REQUIRE(filtered.values == expected.values);
}

TEST_CASE("simpatico decode filter - no filter asks the decode for nothing",
          "[compression][simpatico_decode_filter]")
{
  if (no_gpu()) { return; }
  require_pushdown_gate();
  fixture_dir fx("nofilter");
  // The control the cases above are read against: with no predicate there is no request, no batch
  // may claim to be row-filtered, and the decode path is the one that existed before any of this.
  auto ing = scan::make_ingestible(bind_with_filters(fx.path, nullptr, nullptr));

  auto const decoded = run_scan(*ing, collect_splits(*ing), /*apply_filter=*/false);
  REQUIRE(decoded.keys.size() == static_cast<std::size_t>(kChunks) * kRowsPerChunk);
  REQUIRE_FALSE(decoded.all_row_filtered);
  REQUIRE(ing->decode_filtering().chunks_offered == 0);
}
