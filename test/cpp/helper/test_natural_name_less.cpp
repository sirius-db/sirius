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
#include "op/scan/parquet_gpu_ingestible.hpp"

#include <algorithm>
#include <filesystem>
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
  REQUIRE(natural_name_less("part.2", "part.2.parquet"));
  REQUIRE(natural_name_less("part.2", "part.02"));
  REQUIRE_FALSE(natural_name_less("part.02", "part.2"));
  REQUIRE_FALSE(natural_name_less("abc", "abc"));
  std::vector<std::string> names{"part.10", "part.2", "part.1", "part.0", "part.11"};
  std::sort(names.begin(), names.end(), [](const std::string& l, const std::string& r) {
    return natural_name_less(l, r);
  });
  REQUIRE(names == std::vector<std::string>{"part.0", "part.1", "part.2", "part.10", "part.11"});
}

TEST_CASE("natural_name_less orders high bytes as unsigned", "[natural_file_order]")
{
  using sirius::utils::natural_name_less;
  auto const named = [](unsigned char byte) {
    return std::string{"part."} + static_cast<char>(byte);
  };
  std::string const ascii  = named(0x7f);
  std::string const high   = named(0x80);
  std::string const higher = named(0xff);
  REQUIRE(natural_name_less(ascii, high));
  REQUIRE_FALSE(natural_name_less(high, ascii));
  REQUIRE(natural_name_less(high, higher));
  REQUIRE_FALSE(natural_name_less(higher, high));
  REQUIRE(natural_name_less("part.a", high));
}

TEST_CASE("natural_name_less is a strict weak ordering", "[natural_file_order]")
{
  using sirius::utils::natural_name_less;
  std::vector<std::string> universe{""};
  std::string const alphabet = "01.a";
  for (int len = 1; len <= 3; ++len) {
    std::vector<std::string> next;
    for (auto const& word : universe) {
      if (word.size() + 1 != static_cast<std::size_t>(len)) { continue; }
      for (char c : alphabet) {
        next.push_back(word + c);
      }
    }
    universe.insert(universe.end(), next.begin(), next.end());
  }

  auto const equiv = [](const std::string& a, const std::string& b) {
    return !natural_name_less(a, b) && !natural_name_less(b, a);
  };

  std::vector<std::string> irreflexive;
  std::vector<std::string> asymmetric;
  std::vector<std::string> transitive;
  std::vector<std::string> incomparable;
  for (auto const& a : universe) {
    if (natural_name_less(a, a)) { irreflexive.push_back("'" + a + "'"); }
    for (auto const& b : universe) {
      bool const ab = natural_name_less(a, b);
      if (ab && natural_name_less(b, a)) { asymmetric.push_back("'" + a + "' '" + b + "'"); }
      for (auto const& c : universe) {
        if (ab && natural_name_less(b, c) && !natural_name_less(a, c)) {
          transitive.push_back("'" + a + "' '" + b + "' '" + c + "'");
        }
        if (equiv(a, b) && equiv(b, c) && !equiv(a, c)) {
          incomparable.push_back("'" + a + "' '" + b + "' '" + c + "'");
        }
      }
    }
  }

  auto const first = [](const std::vector<std::string>& violations) {
    return violations.empty() ? std::string{"none"} : violations.front();
  };
  INFO("universe size " << universe.size());
  INFO("irreflexive violation: " << first(irreflexive));
  INFO("asymmetry violation: " << first(asymmetric));
  INFO("transitivity violation: " << first(transitive));
  INFO("incomparability-transitivity violation: " << first(incomparable));
  REQUIRE(irreflexive.empty());
  REQUIRE(asymmetric.empty());
  REQUIRE(transitive.empty());
  REQUIRE(incomparable.empty());
}

TEST_CASE("pin file ordering is opt-in", "[natural_file_order]")
{
  std::vector<std::string> paths{"part.10.parquet", "part.2.parquet"};
  auto const original = paths;

  sirius::op::scan::order_pin_file_paths(paths, false);

  REQUIRE(paths == original);
}

TEST_CASE("pin file ordering uses canonical natural keys and retains raw paths",
          "[natural_file_order]")
{
  namespace fs   = std::filesystem;
  auto const cwd = fs::current_path();

  auto const relative_two = std::string{"./part.2.parquet"};
  auto const absolute_ten = (cwd / "part.10.parquet").string();
  std::vector<std::string> mixed_paths{absolute_ten, relative_two};

  sirius::op::scan::order_pin_file_paths(mixed_paths, true);

  REQUIRE(mixed_paths == std::vector<std::string>{relative_two, absolute_ten});

  std::vector<std::string> relative_paths{"./part.10.parquet", "./part.2.parquet"};
  std::vector<std::string> absolute_paths{(cwd / "part.10.parquet").string(),
                                          (cwd / "part.2.parquet").string()};
  sirius::op::scan::order_pin_file_paths(relative_paths, true);
  sirius::op::scan::order_pin_file_paths(absolute_paths, true);

  sirius::op::scan::canonicalize_scan_file_paths(relative_paths);
  sirius::op::scan::canonicalize_scan_file_paths(absolute_paths);
  REQUIRE(relative_paths == absolute_paths);
}
