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

#include "helper/natural_sort.hpp"

#include <catch.hpp>

#include <algorithm>
#include <string>
#include <vector>

TEST_CASE("natural_less orders numbered files by number", "[helper][natural_sort]")
{
  std::vector<std::string> files{"/d/lineitem/part.10.parquet",
                                 "/d/lineitem/part.2.parquet",
                                 "/d/lineitem/part.1.parquet",
                                 "/d/lineitem/part.0.parquet",
                                 "/d/lineitem/part.59.parquet",
                                 "/d/lineitem/part.19.parquet"};
  std::sort(files.begin(), files.end(), [](auto const& lhs, auto const& rhs) {
    return sirius::natural_less(lhs, rhs);
  });
  REQUIRE(files == std::vector<std::string>{"/d/lineitem/part.0.parquet",
                                            "/d/lineitem/part.1.parquet",
                                            "/d/lineitem/part.2.parquet",
                                            "/d/lineitem/part.10.parquet",
                                            "/d/lineitem/part.19.parquet",
                                            "/d/lineitem/part.59.parquet"});
}

TEST_CASE("natural_less handles text, prefixes, and leading zeros", "[helper][natural_sort]")
{
  using sirius::natural_less;
  CHECK(natural_less("a", "b"));
  CHECK_FALSE(natural_less("b", "a"));
  CHECK(natural_less("part", "part.1"));        // a prefix sorts first
  CHECK(natural_less("part.9", "part.10"));     // numeric, not lexicographic
  CHECK(natural_less("part.2.a", "part.2.b"));  // text after an equal number
  CHECK(natural_less("x99y", "x100y"));
  CHECK_FALSE(natural_less("part.007", "part.007"));  // irreflexive
  // Equal numeric value: plain lexicographic order breaks the tie, in both directions consistently.
  CHECK(natural_less("part.07", "part.7"));
  CHECK_FALSE(natural_less("part.7", "part.07"));
  CHECK(natural_less("part.0", "part.00") != natural_less("part.00", "part.0"));
}
