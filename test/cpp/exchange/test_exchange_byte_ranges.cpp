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

// How a distributed scan splits a table's files across ranks by byte range.

#include "catch.hpp"
#include "planner/distributed_plan.hpp"

#include <cstdint>
#include <vector>

using sirius::planner::byte_ranges_for_rank;
using sirius::planner::file_byte_range;

namespace {

/// Every byte of every file is owned by exactly one rank, and each rank's ranges come in file
/// order, one per file.
void require_exact_tiling(std::vector<std::uint64_t> const& files, std::size_t world)
{
  std::vector<std::vector<int>> owners(files.size());
  for (std::size_t f = 0; f < files.size(); ++f) {
    owners[f].assign(files[f], 0);
  }
  for (std::size_t rank = 0; rank < world; ++rank) {
    auto const mine   = byte_ranges_for_rank(files, rank, world);
    std::size_t prior = 0;
    for (std::size_t i = 0; i < mine.size(); ++i) {
      auto const& r = mine[i];
      REQUIRE(r.length > 0);
      REQUIRE(r.start + r.length <= files[r.file]);
      if (i > 0) { REQUIRE(r.file > prior); }
      prior = r.file;
      for (auto b = r.start; b < r.start + r.length; ++b) {
        ++owners[r.file][b];
      }
    }
  }
  for (auto const& file : owners) {
    for (auto const count : file) {
      REQUIRE(count == 1);
    }
  }
}

}  // namespace

TEST_CASE("byte_ranges_for_rank tiles every file exactly once", "[exchange][byte_range]")
{
  require_exact_tiling({100}, 1);
  require_exact_tiling({100}, 2);
  require_exact_tiling({101}, 3);
  require_exact_tiling({10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10}, 2);
  require_exact_tiling({7, 300, 1, 55, 1000, 2}, 4);
  require_exact_tiling({5, 5}, 8);
}

TEST_CASE("byte_ranges_for_rank splits one file between two ranks", "[exchange][byte_range]")
{
  std::vector<std::uint64_t> const files{1000};
  CHECK(byte_ranges_for_rank(files, 0, 2) == std::vector<file_byte_range>{{0, 0, 500}});
  CHECK(byte_ranges_for_rank(files, 1, 2) == std::vector<file_byte_range>{{0, 500, 500}});
}

TEST_CASE("byte_ranges_for_rank balances unevenly sized files by bytes", "[exchange][byte_range]")
{
  // Whole-file striping would give rank 0 files 0 and 2 (700 bytes) and rank 1 only file 1.
  std::vector<std::uint64_t> const files{600, 200, 100};
  CHECK(byte_ranges_for_rank(files, 0, 2) == std::vector<file_byte_range>{{0, 0, 450}});
  CHECK(byte_ranges_for_rank(files, 1, 2) ==
        std::vector<file_byte_range>{{0, 450, 150}, {1, 0, 200}, {2, 0, 100}});
}
