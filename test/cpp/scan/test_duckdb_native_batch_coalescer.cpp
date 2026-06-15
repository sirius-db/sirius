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
#include <op/scan/batch_coalescer.hpp>
#include <op/scan/coalescing_unit.hpp>

#include <cstddef>
#include <vector>

using namespace sirius;
using namespace sirius::op::scan;

namespace {

logical_type int_type() { return logical_type::make(type_id::INTEGER); }
logical_type varchar_type() { return logical_type::make(type_id::VARCHAR); }

coalescing_unit make_unit(std::size_t rows,
                          std::size_t decoded_bytes,
                          std::vector<std::size_t> varchar_bytes = {})
{
  coalescing_unit u;
  u.row_count             = rows;
  u.decoded_bytes         = decoded_bytes;
  u.varchar_bytes_per_col = std::move(varchar_bytes);
  // payload stays null: the coalescer only moves it, never dereferences it.
  return u;
}

/// Push every unit and drain ready batches, then the tail. Returns batch sizes
/// in emission order.
std::vector<std::size_t> coalesce_sizes(batch_coalescer& c, std::vector<coalescing_unit> units)
{
  std::vector<std::size_t> sizes;
  for (auto& u : units) {
    c.push(std::move(u));
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
  std::vector<coalescing_unit> units;
  for (std::size_t i = 0; i < 5; ++i) {
    units.push_back(make_unit(/*rows=*/100, /*bytes=*/1000));
  }
  auto sizes = coalesce_sizes(c, std::move(units));
  REQUIRE(sizes == std::vector<std::size_t>{5});  // one batch holds all five
}

TEST_CASE("batch_coalescer splits on the total decoded-byte cap with one tail",
          "[scan][batch_coalescer]")
{
  // cap=1000, each unit 400 bytes: a batch closes at 2 units (800), since a
  // third (1200) exceeds the cap. Five units -> [2, 2, 1].
  batch_coalescer c(/*approximate_batch_size=*/1000, {int_type()});
  std::vector<coalescing_unit> units;
  for (std::size_t i = 0; i < 5; ++i) {
    units.push_back(make_unit(/*rows=*/100, /*bytes=*/400));
  }
  auto sizes = coalesce_sizes(c, std::move(units));
  REQUIRE(sizes == std::vector<std::size_t>{2, 2, 1});
  // Only the final batch is below the 2-unit fill.
  REQUIRE(sizes.back() == 1);
}

TEST_CASE("batch_coalescer keeps an over-cap unit as its own singleton batch",
          "[scan][batch_coalescer]")
{
  // Every unit (400 bytes) exceeds the cap (100) on its own, so each lands
  // alone: three units -> [1, 1, 1].
  batch_coalescer c(/*approximate_batch_size=*/100, {int_type()});
  std::vector<coalescing_unit> units;
  for (std::size_t i = 0; i < 3; ++i) {
    units.push_back(make_unit(/*rows=*/100, /*bytes=*/400));
  }
  auto sizes = coalesce_sizes(c, std::move(units));
  REQUIRE(sizes == std::vector<std::size_t>{1, 1, 1});
}

TEST_CASE("batch_coalescer splits on the per-varchar-column cudf int32 cap",
          "[scan][batch_coalescer]")
{
  // No total cap; one varchar column. Two units at 1.5e9 chars each sum to
  // 3e9 >= kCudfInt32StringsThreshold (2^31-1), so they cannot share a batch.
  // Three units -> [1, 1, 1].
  batch_coalescer c(/*approximate_batch_size=*/0, {varchar_type()});
  constexpr std::size_t kBig = 1'500'000'000ULL;
  REQUIRE(kBig < kCudfInt32StringsThreshold);
  REQUIRE(2 * kBig >= kCudfInt32StringsThreshold);

  std::vector<coalescing_unit> units;
  for (std::size_t i = 0; i < 3; ++i) {
    units.push_back(make_unit(/*rows=*/100, /*bytes=*/0, /*varchar_bytes=*/{kBig}));
  }
  auto sizes = coalesce_sizes(c, std::move(units));
  REQUIRE(sizes == std::vector<std::size_t>{1, 1, 1});
}

TEST_CASE("batch_coalescer drops empty units", "[scan][batch_coalescer]")
{
  batch_coalescer c(/*approximate_batch_size=*/0, {int_type()});
  std::vector<coalescing_unit> units;
  units.push_back(make_unit(/*rows=*/100, /*bytes=*/1000));
  units.push_back(make_unit(/*rows=*/0, /*bytes=*/0));  // empty: dropped
  units.push_back(make_unit(/*rows=*/100, /*bytes=*/1000));
  auto sizes = coalesce_sizes(c, std::move(units));
  REQUIRE(sizes == std::vector<std::size_t>{2});  // two non-empty units, one batch
}

TEST_CASE("batch_coalescer with only empty units yields no batches", "[scan][batch_coalescer]")
{
  batch_coalescer c(/*approximate_batch_size=*/0, {int_type()});
  std::vector<coalescing_unit> units;
  units.push_back(make_unit(/*rows=*/0, /*bytes=*/0));
  units.push_back(make_unit(/*rows=*/0, /*bytes=*/0));
  auto sizes = coalesce_sizes(c, std::move(units));
  REQUIRE(sizes.empty());
}

TEST_CASE("batch_coalescer preserves total unit count across the split", "[scan][batch_coalescer]")
{
  batch_coalescer c(/*approximate_batch_size=*/1000, {int_type()});
  std::vector<coalescing_unit> units;
  for (std::size_t i = 0; i < 17; ++i) {
    units.push_back(make_unit(/*rows=*/100, /*bytes=*/300));
  }
  auto sizes        = coalesce_sizes(c, std::move(units));
  std::size_t total = 0;
  for (auto s : sizes) {
    total += s;
  }
  REQUIRE(total == 17);        // no unit lost or duplicated
  REQUIRE(sizes.size() >= 2);  // actually split
}

TEST_CASE("batch_coalescer never merges units with different group_keys", "[scan][batch_coalescer]")
{
  // No byte cap: only the group_key can force a boundary. The first unit lands
  // alone (its key differs from the next two); the last two share a key and
  // coalesce. This is the partition-affinity guarantee parquet relies on.
  batch_coalescer c(/*approximate_batch_size=*/0, {int_type()});
  std::vector<coalescing_unit> units;

  auto a      = make_unit(/*rows=*/100, /*bytes=*/1000);
  a.group_key = {"year=2020", "month=01"};
  auto b      = make_unit(/*rows=*/100, /*bytes=*/1000);
  b.group_key = {"year=2020", "month=02"};
  auto d      = make_unit(/*rows=*/100, /*bytes=*/1000);
  d.group_key = {"year=2020", "month=02"};
  units.push_back(std::move(a));
  units.push_back(std::move(b));
  units.push_back(std::move(d));

  auto sizes = coalesce_sizes(c, std::move(units));
  REQUIRE(sizes == std::vector<std::size_t>{1, 2});  // {01} | {02, 02}
}

TEST_CASE("batch_coalescer ignores group boundaries when keys are empty", "[scan][batch_coalescer]")
{
  // Unpartitioned scans (native, unpartitioned parquet) leave group_key empty;
  // an empty key must never trigger a boundary, or every batch would collapse to
  // a singleton.
  batch_coalescer c(/*approximate_batch_size=*/0, {int_type()});
  std::vector<coalescing_unit> units;
  for (std::size_t i = 0; i < 4; ++i) {
    units.push_back(make_unit(/*rows=*/100, /*bytes=*/1000));  // group_key default-empty
  }
  auto sizes = coalesce_sizes(c, std::move(units));
  REQUIRE(sizes == std::vector<std::size_t>{4});  // one batch holds all four
}
