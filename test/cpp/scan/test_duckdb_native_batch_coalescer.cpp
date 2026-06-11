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

#include <catch.hpp>
#include <op/scan/duckdb_native_batch_coalescer.hpp>

#include <cstddef>
#include <vector>

using namespace sirius;
using namespace sirius::op::scan;

namespace {

logical_type int_type() { return logical_type::make(type_id::INTEGER); }
logical_type varchar_type() { return logical_type::make(type_id::VARCHAR); }

duckdb_row_group_metadata make_rg(std::size_t index,
                                  std::size_t rows,
                                  std::size_t decoded_bytes,
                                  std::vector<std::size_t> varchar_bytes = {})
{
  duckdb_row_group_metadata rg;
  rg.row_group_index       = index;
  rg.row_group_start       = index * 100;  // arbitrary, monotone
  rg.row_count             = rows;
  rg.decoded_bytes_budget  = decoded_bytes;
  rg.varchar_bytes_per_col = std::move(varchar_bytes);
  return rg;
}

/// Push every row group and drain ready batches, then the tail. Returns batch
/// sizes in emission order.
std::vector<std::size_t> coalesce_sizes(batch_coalescer& c,
                                        std::vector<duckdb_row_group_metadata> rgs)
{
  std::vector<std::size_t> sizes;
  for (auto& rg : rgs) {
    c.push(std::move(rg));
    while (c.has_ready()) {
      sizes.push_back(c.pop_ready().size());
    }
  }
  auto tail = c.flush();
  if (!tail.empty()) { sizes.push_back(tail.size()); }
  return sizes;
}

}  // namespace

TEST_CASE("batch_coalescer with no caps yields a single batch", "[scan][batch_coalescer]")
{
  batch_coalescer c(/*approximate_batch_size=*/0, {int_type()});
  std::vector<duckdb_row_group_metadata> rgs;
  for (std::size_t i = 0; i < 5; ++i) {
    rgs.push_back(make_rg(i, /*rows=*/100, /*bytes=*/1000));
  }
  auto sizes = coalesce_sizes(c, std::move(rgs));
  REQUIRE(sizes == std::vector<std::size_t>{5});  // one batch holds all five
}

TEST_CASE("batch_coalescer splits on the total decoded-byte cap with one tail",
          "[scan][batch_coalescer]")
{
  // cap=1000, each row group 400 bytes: a batch closes at 2 row groups (800),
  // since a third (1200) exceeds the cap. Five row groups -> [2, 2, 1].
  batch_coalescer c(/*approximate_batch_size=*/1000, {int_type()});
  std::vector<duckdb_row_group_metadata> rgs;
  for (std::size_t i = 0; i < 5; ++i) {
    rgs.push_back(make_rg(i, /*rows=*/100, /*bytes=*/400));
  }
  auto sizes = coalesce_sizes(c, std::move(rgs));
  REQUIRE(sizes == std::vector<std::size_t>{2, 2, 1});
  // Only the final batch is below the 2-row-group fill.
  REQUIRE(sizes.back() == 1);
}

TEST_CASE("batch_coalescer keeps an over-cap row group as its own singleton batch",
          "[scan][batch_coalescer]")
{
  // Every row group (400 bytes) exceeds the cap (100) on its own, so each lands
  // alone: three row groups -> [1, 1, 1].
  batch_coalescer c(/*approximate_batch_size=*/100, {int_type()});
  std::vector<duckdb_row_group_metadata> rgs;
  for (std::size_t i = 0; i < 3; ++i) {
    rgs.push_back(make_rg(i, /*rows=*/100, /*bytes=*/400));
  }
  auto sizes = coalesce_sizes(c, std::move(rgs));
  REQUIRE(sizes == std::vector<std::size_t>{1, 1, 1});
}

TEST_CASE("batch_coalescer splits on the per-varchar-column cudf int32 cap",
          "[scan][batch_coalescer]")
{
  // No total cap; one varchar column. Two row groups at 1.5e9 chars each sum to
  // 3e9 >= kCudfInt32StringsThreshold (2^31-1), so they cannot share a batch.
  // Three row groups -> [1, 1, 1].
  batch_coalescer c(/*approximate_batch_size=*/0, {varchar_type()});
  constexpr std::size_t kBig = 1'500'000'000ULL;
  REQUIRE(kBig < kCudfInt32StringsThreshold);
  REQUIRE(2 * kBig >= kCudfInt32StringsThreshold);

  std::vector<duckdb_row_group_metadata> rgs;
  for (std::size_t i = 0; i < 3; ++i) {
    rgs.push_back(make_rg(i, /*rows=*/100, /*bytes=*/0, /*varchar_bytes=*/{kBig}));
  }
  auto sizes = coalesce_sizes(c, std::move(rgs));
  REQUIRE(sizes == std::vector<std::size_t>{1, 1, 1});
}

TEST_CASE("batch_coalescer drops empty row groups", "[scan][batch_coalescer]")
{
  batch_coalescer c(/*approximate_batch_size=*/0, {int_type()});
  std::vector<duckdb_row_group_metadata> rgs;
  rgs.push_back(make_rg(0, /*rows=*/100, /*bytes=*/1000));
  rgs.push_back(make_rg(1, /*rows=*/0, /*bytes=*/0));  // empty: dropped
  rgs.push_back(make_rg(2, /*rows=*/100, /*bytes=*/1000));
  auto sizes = coalesce_sizes(c, std::move(rgs));
  REQUIRE(sizes == std::vector<std::size_t>{2});  // two non-empty row groups, one batch
}

TEST_CASE("batch_coalescer with only empty row groups yields no batches", "[scan][batch_coalescer]")
{
  batch_coalescer c(/*approximate_batch_size=*/0, {int_type()});
  std::vector<duckdb_row_group_metadata> rgs;
  rgs.push_back(make_rg(0, /*rows=*/0, /*bytes=*/0));
  rgs.push_back(make_rg(1, /*rows=*/0, /*bytes=*/0));
  auto sizes = coalesce_sizes(c, std::move(rgs));
  REQUIRE(sizes.empty());
}

TEST_CASE("batch_coalescer preserves total row-group count across the split",
          "[scan][batch_coalescer]")
{
  batch_coalescer c(/*approximate_batch_size=*/1000, {int_type()});
  std::vector<duckdb_row_group_metadata> rgs;
  for (std::size_t i = 0; i < 17; ++i) {
    rgs.push_back(make_rg(i, /*rows=*/100, /*bytes=*/300));
  }
  auto sizes        = coalesce_sizes(c, std::move(rgs));
  std::size_t total = 0;
  for (auto s : sizes) {
    total += s;
  }
  REQUIRE(total == 17);        // no row group lost or duplicated
  REQUIRE(sizes.size() >= 2);  // actually split
}
