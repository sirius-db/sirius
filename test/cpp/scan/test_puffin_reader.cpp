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

// Unit tests for the Puffin deletion-vector reader.
//
// Every iceberg fixture in this repo stores relative paths, so none of them can tell whether
// this reader handles the URIs that Apache writers actually put in manifests. It opens the file
// directly rather than through sirius_ioctx, so the scheme stripping done at the datasource
// boundary does not reach it — and a failure here does not surface as an error: the table
// declines to CPU and the V3 path merely appears never to engage.
//
// No GPU: file IO and parsing only.

#include "op/scan/puffin_reader.hpp"

#include <catch.hpp>

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

using namespace sirius::op::scan;

namespace {

namespace fs = std::filesystem;

// The deletion vector of the iceberg_v3_deletion_vector fixture. The blob sits at offset 4
// because regenerate_puffin.py wraps it in a real Puffin container (4-byte leading magic); both
// numbers come from that fixture's manifest, and the script rewrites the manifest whenever it
// moves them.
constexpr int64_t kBlobOffset = 4;
constexpr int64_t kBlobSize   = 44;

std::string fixture_puffin_path()
{
  return "test/cpp/integration/data/iceberg_v3_deletion_vector/data/"
         "dv-00000-0-d7e8f9a0-0007-0007-0007-000000000002-00001.puffin";
}

}  // namespace

TEST_CASE("puffin reader reads the fixture deletion vector", "[scan][iceberg]")
{
  auto const path = fixture_puffin_path();
  // Same cwd requirement as the iceberg integration cases: fixture paths are repo-relative.
  if (!fs::exists(path)) {
    FAIL("puffin fixture not found; these tests must run with the repo root as cwd, cwd is "
         << fs::current_path().string());
  }

  auto const positions = read_deletion_vector(path, kBlobOffset, kBlobSize);

  // The fixture deletes 2 of its 5 rows — the same 3 survivors the integration case asserts.
  REQUIRE(positions.size() == 2);
  REQUIRE(std::is_sorted(positions.begin(), positions.end()));
}

TEST_CASE("puffin reader accepts a file:// URI", "[scan][iceberg]")
{
  auto const path = fixture_puffin_path();
  if (!fs::exists(path)) {
    FAIL("puffin fixture not found; these tests must run with the repo root as cwd, cwd is "
         << fs::current_path().string());
  }

  auto const expected = read_deletion_vector(path, kBlobOffset, kBlobSize);

  // What an Apache writer records: an absolute file:// URI. Before the reader stripped the
  // scheme, std::ifstream simply failed to open this and the whole table declined to CPU.
  auto const uri = "file://" + fs::absolute(path).string();
  REQUIRE(read_deletion_vector(uri, kBlobOffset, kBlobSize) == expected);
}

TEST_CASE("puffin reader rejects a non-Puffin file", "[scan][iceberg]")
{
  // A parquet data file is not a Puffin container. Pointing the reader at one must fail rather
  // than return an empty position list, which would read as "this data file has no deletes".
  auto const not_puffin = std::string(
    "test/cpp/integration/data/iceberg_v3_deletion_vector/data/"
    "00000-0-d7e8f9a0-0007-0007-0007-000000000001-00001.parquet");
  if (!fs::exists(not_puffin)) {
    FAIL("fixture not found; these tests must run with the repo root as cwd, cwd is "
         << fs::current_path().string());
  }

  REQUIRE_THROWS(read_deletion_vector(not_puffin, kBlobOffset, kBlobSize));
}
