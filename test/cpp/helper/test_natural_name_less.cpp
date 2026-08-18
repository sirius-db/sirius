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

#include "catch.hpp"
#include "helper/utils.hpp"

#include <algorithm>
#include <string>
#include <vector>

TEST_CASE("natural_name_less orders numbered parts numerically", "[natural_file_order]")
{
  using sirius::utils::natural_name_less;
  REQUIRE(natural_name_less("part.2.parquet", "part.10.parquet"));
  REQUIRE_FALSE(natural_name_less("part.10.parquet", "part.2.parquet"));
  REQUIRE(natural_name_less("part.9.parquet", "part.10.parquet"));
  REQUIRE(natural_name_less("part.0.parquet", "part.1.parquet"));
  REQUIRE(natural_name_less("a/part.19.parquet", "a/part.20.parquet"));
  REQUIRE(natural_name_less("part.2", "part.2.parquet"));  // prefix first
  REQUIRE(natural_name_less("part.2", "part.02"));         // equal value: fewer leading zeros first
  REQUIRE_FALSE(natural_name_less("part.02", "part.2"));
  REQUIRE_FALSE(natural_name_less("abc", "abc"));  // irreflexive
  std::vector<std::string> names{"part.10", "part.2", "part.1", "part.0", "part.11"};
  std::sort(names.begin(), names.end(), [](const std::string& l, const std::string& r) {
    return natural_name_less(l, r);
  });
  REQUIRE(names == std::vector<std::string>{"part.0", "part.1", "part.2", "part.10", "part.11"});
}
