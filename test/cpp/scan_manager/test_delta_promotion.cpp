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

// Unit tests for the promotion sink, the pure contiguity ratchet, and the
// entry apply (against hand-built pinned entries — no scan manager). The GPU
// capture hook is covered in test_duckdb_native_host_backed_decode.cpp; the
// end-to-end pipeline in test_pin_table_mvcc_promotion.cpp.

#include "operator/operator_test_utils.hpp"

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/types.hpp>

#include <rmm/cuda_stream.hpp>

#include <catch.hpp>
#include <scan_manager/delta_promotion.hpp>
#include <scan_manager/sirius_scan_manager.hpp>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

using sirius::scan_manager::pinned_entry;
using sirius::scan_manager::promotion_captured_slice;
using sirius::scan_manager::promotion_sink;
using sirius::scan_manager::select_promotion_prefix;

namespace {

promotion_captured_slice make_slice(std::size_t first_rowid, std::size_t row_count)
{
  promotion_captured_slice s;
  s.first_rowid       = first_rowid;
  s.row_count         = row_count;
  s.row_group_indices = {static_cast<duckdb::idx_t>(first_rowid)};
  return s;
}

std::vector<std::size_t> first_rowids(std::vector<promotion_captured_slice> const& slices)
{
  std::vector<std::size_t> out;
  out.reserve(slices.size());
  for (auto const& s : slices) {
    out.push_back(s.first_rowid);
  }
  return out;
}

struct apply_env {
  std::unique_ptr<sirius::memory::sirius_memory_reservation_manager> mgr;
  apply_env() : mgr(sirius::test::operator_utils::initialize_memory_manager()) {}
};

apply_env& env()
{
  static apply_env e;
  return e;
}

std::shared_ptr<cudf::column> make_i32_column(std::size_t n,
                                              std::int32_t start,
                                              cucascade::memory::memory_space& space,
                                              rmm::cuda_stream_view stream)
{
  auto col = cudf::make_numeric_column(cudf::data_type{cudf::type_id::INT32},
                                       static_cast<cudf::size_type>(n),
                                       cudf::mask_state::UNALLOCATED,
                                       stream,
                                       space.get_default_allocator());
  std::vector<std::int32_t> host(n);
  for (std::size_t i = 0; i < n; ++i) {
    host[i] = start + static_cast<std::int32_t>(i);
  }
  cudaMemcpyAsync(col->mutable_view().data<std::int32_t>(),
                  host.data(),
                  n * sizeof(std::int32_t),
                  cudaMemcpyHostToDevice,
                  stream.value());
  stream.synchronize();
  return std::shared_ptr<cudf::column>(std::move(col));
}

/// One-chunk GPU entry over columns {"a","b"} with @p base_rows rows and mvcc
/// metadata: the smallest well-formed promotion target.
pinned_entry make_gpu_entry(std::size_t base_rows,
                            cucascade::memory::memory_space& space,
                            rmm::cuda_stream_view stream)
{
  pinned_entry entry;
  entry.tier                    = cucascade::memory::Tier::GPU;
  entry.memory_space            = &space;
  entry.cache_info.catalog_name = "memory";
  entry.cache_info.schema_name  = "main";
  entry.cache_info.table_name   = "t";
  entry.cache_info.column_ids.push_back(duckdb::ColumnIndex(0));
  entry.cache_info.column_ids.push_back(duckdb::ColumnIndex(1));
  entry.cache_info.names = {"a", "b"};
  entry.data_batches_by_column["a"].push_back(make_i32_column(base_rows, 0, space, stream));
  entry.data_batches_by_column["b"].push_back(make_i32_column(base_rows, 1000, space, stream));
  entry.chunk_memory_spaces.push_back(&space);
  entry.num_rows     = base_rows;
  entry.mvcc         = std::make_unique<sirius::scan_manager::duckdb_mvcc_metadata>();
  entry.mvcc->v_base = 1;
  entry.mvcc->base_row_count_per_chunk = {base_rows};
  return entry;
}

promotion_captured_slice make_gpu_slice(std::size_t first_rowid,
                                        std::size_t rows,
                                        cucascade::memory::memory_space& space,
                                        rmm::cuda_stream_view stream)
{
  auto slice         = make_slice(first_rowid, rows);
  slice.column_names = {"a", "b"};
  slice.columns.push_back(
    make_i32_column(rows, static_cast<std::int32_t>(first_rowid), space, stream));
  slice.columns.push_back(
    make_i32_column(rows, static_cast<std::int32_t>(1000 + first_rowid), space, stream));
  slice.space = &space;
  return slice;
}

}  // namespace

TEST_CASE("select_promotion_prefix: a contiguous run from n_cache is selected in rowid order",
          "[delta_promotion][scan_manager]")
{
  std::vector<promotion_captured_slice> slices;
  slices.push_back(make_slice(100, 10));
  slices.push_back(make_slice(110, 20));
  slices.push_back(make_slice(130, 5));

  std::vector<promotion_captured_slice> dropped;
  auto selected = select_promotion_prefix(std::move(slices), 100, dropped);

  REQUIRE(first_rowids(selected) == std::vector<std::size_t>{100, 110, 130});
  REQUIRE(dropped.empty());
}

TEST_CASE("select_promotion_prefix: a gap stops the ratchet with no holes",
          "[delta_promotion][scan_manager]")
{
  std::vector<promotion_captured_slice> slices;
  slices.push_back(make_slice(100, 10));
  slices.push_back(make_slice(120, 10));  // gap at 110
  slices.push_back(make_slice(130, 10));

  std::vector<promotion_captured_slice> dropped;
  auto selected = select_promotion_prefix(std::move(slices), 100, dropped);

  REQUIRE(first_rowids(selected) == std::vector<std::size_t>{100});
  REQUIRE(first_rowids(dropped) == std::vector<std::size_t>{120, 130});
}

TEST_CASE("select_promotion_prefix: nothing at n_cache promotes nothing",
          "[delta_promotion][scan_manager]")
{
  std::vector<promotion_captured_slice> slices;
  slices.push_back(make_slice(110, 10));  // starts above n_cache: a gap at the front

  std::vector<promotion_captured_slice> dropped;
  auto selected = select_promotion_prefix(std::move(slices), 100, dropped);

  REQUIRE(selected.empty());
  REQUIRE(first_rowids(dropped) == std::vector<std::size_t>{110});
}

TEST_CASE("select_promotion_prefix: out-of-order input is sorted before selecting",
          "[delta_promotion][scan_manager]")
{
  std::vector<promotion_captured_slice> slices;
  slices.push_back(make_slice(130, 5));
  slices.push_back(make_slice(100, 10));
  slices.push_back(make_slice(110, 20));

  std::vector<promotion_captured_slice> dropped;
  auto selected = select_promotion_prefix(std::move(slices), 100, dropped);

  REQUIRE(first_rowids(selected) == std::vector<std::size_t>{100, 110, 130});
  REQUIRE(dropped.empty());
}

TEST_CASE("select_promotion_prefix: empty input is a no-op", "[delta_promotion][scan_manager]")
{
  std::vector<promotion_captured_slice> dropped;
  auto selected = select_promotion_prefix({}, 100, dropped);
  REQUIRE(selected.empty());
  REQUIRE(dropped.empty());
}

TEST_CASE("promotion_sink: first-op-wins dedup keys on (entry, first row group)",
          "[delta_promotion][scan_manager]")
{
  promotion_sink sink;
  REQUIRE(sink.try_begin_capture("t", 5));        // first claim wins
  REQUIRE_FALSE(sink.try_begin_capture("t", 5));  // a self-join re-decode loses
  REQUIRE(sink.try_begin_capture("t", 6));        // a different row group is its own claim
  REQUIRE(sink.try_begin_capture("u", 5));        // a different entry is its own claim
}

TEST_CASE("promotion_sink: add groups slices by entry and take_all drains",
          "[delta_promotion][scan_manager]")
{
  promotion_sink sink;
  REQUIRE(sink.empty());
  sink.add("t", make_slice(100, 10));
  sink.add("t", make_slice(110, 10));
  sink.add("u", make_slice(200, 10));
  REQUIRE_FALSE(sink.empty());

  auto drained = sink.take_all();
  REQUIRE(sink.empty());  // take_all clears
  REQUIRE(drained.size() == 2);
  REQUIRE(first_rowids(drained.at("t").slices) == std::vector<std::size_t>{100, 110});
  REQUIRE(first_rowids(drained.at("u").slices) == std::vector<std::size_t>{200});
}

TEST_CASE("promotion_sink: a recorded skip is retained without creating slices",
          "[delta_promotion][scan_manager]")
{
  promotion_sink sink;
  sink.record_skip("t", "reservation-failed");
  REQUIRE_FALSE(sink.empty());  // a skip-only entry still needs draining to fold into stats

  auto drained = sink.take_all();
  REQUIRE(drained.at("t").slices.empty());
  REQUIRE(drained.at("t").last_skip_reason == "reservation-failed");
}

TEST_CASE("apply_promotion_to_entry: contiguous slices append in lock-step",
          "[delta_promotion][scan_manager]")
{
  auto& e     = env();
  auto* space = e.mgr->get_memory_space(cucascade::memory::Tier::GPU, 0);
  REQUIRE(space != nullptr);
  rmm::cuda_stream stream;

  auto entry = make_gpu_entry(/*base_rows=*/100, *space, stream.view());

  promotion_sink::entry_capture capture;
  capture.slices.push_back(make_gpu_slice(100, 40, *space, stream.view()));
  capture.slices.push_back(make_gpu_slice(140, 60, *space, stream.view()));

  sirius::scan_manager::apply_promotion_to_entry(entry, std::move(capture), "t");

  REQUIRE(entry.mvcc->base_row_count_per_chunk == std::vector<std::size_t>{100, 40, 60});
  REQUIRE(entry.mvcc->n_cache() == 200);
  REQUIRE(entry.num_rows == 200);
  REQUIRE(entry.data_batches_by_column.at("a").size() == 3);
  REQUIRE(entry.data_batches_by_column.at("b").size() == 3);
  REQUIRE(entry.chunk_memory_spaces.size() == 3);
  REQUIRE(entry.data_batches_by_column.at("a")[1]->size() == 40);
  REQUIRE(entry.data_batches_by_column.at("a")[2]->size() == 60);

  // The grown entry still passes the serve-time validator.
  std::vector<std::size_t> const all_cols{0, 1};
  REQUIRE_NOTHROW(sirius::scan_manager::validate_pinned_entry_for_serving(entry, all_cols));

  auto const& stats = entry.mvcc->promotion;
  REQUIRE(stats.promoted_chunks == 2);
  REQUIRE(stats.promoted_rows == 100);
  REQUIRE(stats.promotion_queries == 1);
  REQUIRE(stats.dropped_slices == 0);
}

TEST_CASE("apply_promotion_to_entry: a gap drops the tail and records the reason",
          "[delta_promotion][scan_manager]")
{
  auto& e     = env();
  auto* space = e.mgr->get_memory_space(cucascade::memory::Tier::GPU, 0);
  REQUIRE(space != nullptr);
  rmm::cuda_stream stream;

  auto entry = make_gpu_entry(/*base_rows=*/100, *space, stream.view());

  promotion_sink::entry_capture capture;
  capture.slices.push_back(make_gpu_slice(100, 40, *space, stream.view()));
  capture.slices.push_back(make_gpu_slice(150, 10, *space, stream.view()));  // gap at 140

  sirius::scan_manager::apply_promotion_to_entry(entry, std::move(capture), "t");

  REQUIRE(entry.mvcc->base_row_count_per_chunk == std::vector<std::size_t>{100, 40});
  REQUIRE(entry.num_rows == 140);
  REQUIRE(entry.mvcc->promotion.promoted_chunks == 1);
  REQUIRE(entry.mvcc->promotion.dropped_slices == 1);
  REQUIRE(entry.mvcc->promotion.last_skip_reason == "not-contiguous");
}

TEST_CASE("apply_promotion_to_entry: a layout mismatch drops everything and mutates nothing",
          "[delta_promotion][scan_manager]")
{
  auto& e     = env();
  auto* space = e.mgr->get_memory_space(cucascade::memory::Tier::GPU, 0);
  REQUIRE(space != nullptr);
  rmm::cuda_stream stream;

  auto entry = make_gpu_entry(/*base_rows=*/100, *space, stream.view());

  promotion_sink::entry_capture capture;
  auto slice         = make_gpu_slice(100, 40, *space, stream.view());
  slice.column_names = {"a"};  // entry caches {"a","b"}
  slice.columns.pop_back();
  capture.slices.push_back(std::move(slice));

  sirius::scan_manager::apply_promotion_to_entry(entry, std::move(capture), "t");

  REQUIRE(entry.mvcc->base_row_count_per_chunk == std::vector<std::size_t>{100});
  REQUIRE(entry.num_rows == 100);
  REQUIRE(entry.data_batches_by_column.at("a").size() == 1);
  REQUIRE(entry.mvcc->promotion.promoted_chunks == 0);
  REQUIRE(entry.mvcc->promotion.dropped_slices == 1);
  REQUIRE(entry.mvcc->promotion.last_skip_reason == "entry-layout-mismatch");
}

TEST_CASE("apply_promotion_to_entry: a skip-only capture folds the reason into stats",
          "[delta_promotion][scan_manager]")
{
  auto& e     = env();
  auto* space = e.mgr->get_memory_space(cucascade::memory::Tier::GPU, 0);
  REQUIRE(space != nullptr);
  rmm::cuda_stream stream;

  auto entry = make_gpu_entry(/*base_rows=*/100, *space, stream.view());

  promotion_sink::entry_capture capture;
  capture.last_skip_reason = "reservation-failed";

  sirius::scan_manager::apply_promotion_to_entry(entry, std::move(capture), "t");

  REQUIRE(entry.mvcc->base_row_count_per_chunk == std::vector<std::size_t>{100});
  REQUIRE(entry.mvcc->promotion.last_skip_reason == "reservation-failed");
  REQUIRE(entry.mvcc->promotion.promoted_chunks == 0);
}

TEST_CASE("apply_promotion_to_entry: consecutive queries ratchet from the grown base",
          "[delta_promotion][scan_manager]")
{
  auto& e     = env();
  auto* space = e.mgr->get_memory_space(cucascade::memory::Tier::GPU, 0);
  REQUIRE(space != nullptr);
  rmm::cuda_stream stream;

  auto entry = make_gpu_entry(/*base_rows=*/100, *space, stream.view());

  promotion_sink::entry_capture first;
  first.slices.push_back(make_gpu_slice(100, 40, *space, stream.view()));
  sirius::scan_manager::apply_promotion_to_entry(entry, std::move(first), "t");
  REQUIRE(entry.mvcc->n_cache() == 140);

  // The next query's capture starts where the grown base ends.
  promotion_sink::entry_capture second;
  second.slices.push_back(make_gpu_slice(140, 25, *space, stream.view()));
  sirius::scan_manager::apply_promotion_to_entry(entry, std::move(second), "t");

  REQUIRE(entry.mvcc->n_cache() == 165);
  REQUIRE(entry.mvcc->base_row_count_per_chunk == std::vector<std::size_t>{100, 40, 25});
  REQUIRE(entry.mvcc->promotion.promoted_chunks == 2);
  REQUIRE(entry.mvcc->promotion.promotion_queries == 2);
}
