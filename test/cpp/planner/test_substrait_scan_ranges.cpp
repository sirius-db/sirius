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

// extract_scan_byte_ranges and scan_byte_ranges_state on hand-built Substrait plans. No file is
// opened: these cover which plans may carry ranges at all. Row-group ownership is in
// test/cpp/scan/test_parquet_byte_range.cpp; end-to-end reads in
// test/cpp/exec/test_sirius_ffi_byte_ranges.cpp.

#include "planner/substrait_scan_ranges.hpp"
#include "sirius/exception.hpp"

#include <catch.hpp>
#include <substrait/plan.pb.h>

#include <cstdint>
#include <limits>
#include <string>
#include <utility>
#include <vector>

using sirius::planner::extract_scan_byte_ranges;
using sirius::planner::scan_byte_range;
using sirius::planner::scan_byte_ranges_state;

namespace {

struct item {
  std::string path;
  std::uint64_t start  = 0;
  std::uint64_t length = 0;
};

void fill_read(substrait::Rel& rel, std::vector<item> const& items)
{
  auto* files = rel.mutable_read()->mutable_local_files();
  for (auto const& it : items) {
    auto* file = files->add_items();
    file->set_uri_file(it.path);
    file->mutable_parquet();
    file->set_start(it.start);
    file->set_length(it.length);
  }
}

std::string bytes(substrait::Plan const& plan)
{
  std::string out;
  REQUIRE(plan.SerializeToString(&out));
  return out;
}

/// A plan whose root is one LocalFiles read of `items`.
std::string single_read_plan(std::vector<item> const& items)
{
  substrait::Plan plan;
  fill_read(*plan.add_relations()->mutable_root()->mutable_input(), items);
  return bytes(plan);
}

}  // namespace

TEST_CASE("one read's ranges of one file are returned together", "[substrait_scan_ranges]")
{
  auto const ranges = extract_scan_byte_ranges(
    single_read_plan({{"/d/f.parquet", 0, 100}, {"/d/f.parquet", 300, 50}, {"/d/g.parquet"}}));
  REQUIRE(ranges.size() == 1);
  REQUIRE(ranges.at("/d/f.parquet") == std::vector<scan_byte_range>{{0, 100}, {300, 50}});
}

TEST_CASE("a ranged file read by two ReadRels is refused before binding", "[substrait_scan_ranges]")
{
  // Ranges are claimed per path after DuckDB binds the plan. With two reads of the file the
  // optimizer may drop one (the survivor then claims both reads' ranges) or merge both into one
  // shared subplan (one claim, rows returned twice), so neither read's ranges are attributable.
  SECTION("an unreferenced relation and the root read different ranges")
  {
    substrait::Plan plan;
    fill_read(*plan.add_relations()->mutable_rel(), {{"/d/f.parquet", 500, 500}});
    fill_read(*plan.add_relations()->mutable_root()->mutable_input(), {{"/d/f.parquet", 0, 500}});
    REQUIRE_THROWS_WITH(extract_scan_byte_ranges(bytes(plan)),
                        Catch::Matchers::ContainsSubstring("reads it 2 times"));
  }

  SECTION("both sides of a join, one under a filter")
  {
    substrait::Plan plan;
    auto* join = plan.add_relations()->mutable_root()->mutable_input()->mutable_join();
    fill_read(*join->mutable_left(), {{"/d/f.parquet", 0, 500}});
    fill_read(*join->mutable_right()->mutable_filter()->mutable_input(),
              {{"/d/f.parquet", 500, 500}});
    REQUIRE_THROWS_AS(extract_scan_byte_ranges(bytes(plan)), sirius::invalid_input_exception);
  }

  SECTION("a whole-file read of the same file counts too")
  {
    substrait::Plan plan;
    auto* join = plan.add_relations()->mutable_root()->mutable_input()->mutable_join();
    fill_read(*join->mutable_left(), {{"/d/f.parquet"}});
    fill_read(*join->mutable_right(), {{"file:///d/f.parquet", 0, 500}});
    REQUIRE_THROWS_AS(extract_scan_byte_ranges(bytes(plan)), sirius::invalid_input_exception);
  }

  SECTION("two whole-file reads of one file carry no ranges and pass")
  {
    substrait::Plan plan;
    auto* join = plan.add_relations()->mutable_root()->mutable_input()->mutable_join();
    fill_read(*join->mutable_left(), {{"/d/f.parquet"}});
    fill_read(*join->mutable_right(), {{"/d/f.parquet"}});
    REQUIRE(extract_scan_byte_ranges(bytes(plan)).empty());
  }
}

TEST_CASE("one read naming a file both whole and by range is refused", "[substrait_scan_ranges]")
{
  // Before: the (0,0) item was skipped, so the file kept only (10,5), which owns no row group,
  // and the read returned no rows instead of the whole file.
  REQUIRE_THROWS_WITH(
    extract_scan_byte_ranges(single_read_plan({{"/d/f.parquet", 0, 0}, {"/d/f.parquet", 10, 5}})),
    Catch::Matchers::ContainsSubstring("both whole and by byte range"));
}

TEST_CASE("an overflowing range is refused at extraction", "[substrait_scan_ranges]")
{
  REQUIRE_THROWS_WITH(extract_scan_byte_ranges(single_read_plan(
                        {{"/d/f.parquet", 4, std::numeric_limits<std::uint64_t>::max()}})),
                      Catch::Matchers::ContainsSubstring("overflows"));
}

TEST_CASE("ranges on a remote path are refused whatever scan function reads them",
          "[substrait_scan_ranges]")
{
  REQUIRE_THROWS_WITH(
    extract_scan_byte_ranges(single_read_plan({{"s3://bucket/f.parquet", 0, 100}})),
    Catch::Matchers::ContainsSubstring("only supported on local parquet files"));
  // A whole-file remote read carries no range and is not this function's concern.
  REQUIRE(extract_scan_byte_ranges(single_read_plan({{"s3://bucket/f.parquet"}})).empty());
}

TEST_CASE("plans without ranges are accepted whatever rel types they use",
          "[substrait_scan_ranges]")
{
  // Before: the walk threw on any rel type outside its switch, so a CTE reference or a window
  // failed here even though the plan had no ranges and DuckDB's consumer accepts both.
  substrait::Plan plan;
  fill_read(*plan.add_relations()->mutable_rel(), {{"/d/f.parquet"}});
  auto* window = plan.add_relations()->mutable_root()->mutable_input()->mutable_window();
  window->mutable_input()->mutable_reference()->set_subtree_ordinal(0);
  REQUIRE(extract_scan_byte_ranges(bytes(plan)).empty());
}

TEST_CASE("a ranged read under any rel type is found", "[substrait_scan_ranges]")
{
  substrait::Plan plan;
  auto* window = plan.add_relations()->mutable_root()->mutable_input()->mutable_window();
  fill_read(*window->mutable_input(), {{"/d/f.parquet", 0, 100}});
  REQUIRE(extract_scan_byte_ranges(bytes(plan)).at("/d/f.parquet") ==
          std::vector<scan_byte_range>{{0, 100}});
}

TEST_CASE("a ranged item must name a uri_file", "[substrait_scan_ranges]")
{
  substrait::Plan plan;
  auto* file = plan.add_relations()
                 ->mutable_root()
                 ->mutable_input()
                 ->mutable_read()
                 ->mutable_local_files()
                 ->add_items();
  file->set_uri_path("/d/");
  file->mutable_parquet();
  file->set_start(0);
  file->set_length(100);
  REQUIRE_THROWS_AS(extract_scan_byte_ranges(bytes(plan)), sirius::invalid_input_exception);
}

TEST_CASE("scan_byte_ranges_state: canonical paths, single claim, loud leftovers",
          "[substrait_scan_ranges]")
{
  auto const extracted = extract_scan_byte_ranges(
    single_read_plan({{"file:///d/f.parquet", 0, 100}, {"file:/d/g.parquet", 0, 100}}));
  scan_byte_ranges_state state(extracted);

  // Every `file:` spelling and the plain path name one file.
  REQUIRE(state.has("/d/f.parquet"));
  REQUIRE(state.has("file://localhost/d/f.parquet"));
  REQUIRE_FALSE(state.has("/d/other.parquet"));

  REQUIRE(state.claim("/d/f.parquet") == std::vector<scan_byte_range>{{0, 100}});
  REQUIRE_THROWS_AS(state.claim("file:/d/f.parquet"), sirius::invalid_input_exception);

  // g's ranges were never claimed: executing would read g whole on every split.
  REQUIRE_THROWS_WITH(state.assert_all_consumed(),
                      Catch::Matchers::ContainsSubstring("/d/g.parquet"));
  REQUIRE(state.claim("/d/g.parquet").size() == 1);
  REQUIRE_NOTHROW(state.assert_all_consumed());
}
