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

// Unit tests for the promotion sink and the pure contiguity ratchet. No GPU:
// slices carry only their rowid extents. The GPU capture hook and the entry
// apply are covered by test_pin_table_mvcc_promotion.cpp.

#include <catch.hpp>
#include <scan_manager/delta_promotion.hpp>

#include <string>
#include <vector>

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
