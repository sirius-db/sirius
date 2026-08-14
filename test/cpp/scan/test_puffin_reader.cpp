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

// What the fixture's manifest entry says about that blob. The reader now checks all of it
// against the Puffin footer, so each field below is one thing a corrupt manifest could get wrong.
constexpr int64_t kCardinality = 2;

std::string fixture_puffin_path()
{
  return "test/cpp/integration/data/iceberg_v3_deletion_vector/data/"
         "dv-00000-0-d7e8f9a0-0007-0007-0007-000000000002-00001.puffin";
}

std::string fixture_data_file_path()
{
  return "test/cpp/integration/data/iceberg_v3_deletion_vector/data/"
         "00000-0-d7e8f9a0-0007-0007-0007-000000000001-00001.parquet";
}

/// Fixture paths are repo-relative, so a wrong cwd otherwise surfaces as a confusing parse error.
std::string require_fixture(std::string path)
{
  if (!fs::exists(path)) {
    FAIL("fixture '" << path
                     << "' not found; these tests must run with the repo root as cwd, cwd is "
                     << fs::current_path().string());
  }
  return path;
}

/// The manifest entry as the fixture actually writes it. Each test below changes exactly one
/// field, so a failure names the check that fired rather than "something threw".
DeletionVectorRef fixture_ref()
{
  return {.puffin_path           = require_fixture(fixture_puffin_path()),
          .content_offset        = kBlobOffset,
          .content_size_in_bytes = kBlobSize,
          .referenced_data_file  = fixture_data_file_path(),
          .record_count          = kCardinality};
}

}  // namespace

TEST_CASE("puffin reader reads the fixture deletion vector", "[scan][iceberg]")
{
  auto const positions = read_deletion_vector(fixture_ref());

  // The fixture deletes 2 of its 5 rows — the same 3 survivors the integration case asserts.
  REQUIRE(positions.size() == 2);
  REQUIRE(std::is_sorted(positions.begin(), positions.end()));
}

TEST_CASE("puffin reader accepts a file:// URI", "[scan][iceberg]")
{
  auto const expected = read_deletion_vector(fixture_ref());

  // What an Apache writer records: an absolute file:// URI. Before the reader stripped the
  // scheme, std::ifstream simply failed to open this and the whole table declined to CPU.
  auto ref        = fixture_ref();
  ref.puffin_path = "file://" + fs::absolute(fixture_puffin_path()).string();
  REQUIRE(read_deletion_vector(ref) == expected);
}

TEST_CASE("puffin reader accepts a file:// URI on the referenced data file", "[scan][iceberg]")
{
  // The manifest and the Puffin footer are free to disagree about the SCHEME while naming the
  // same file. Comparing them literally would refuse tables that are entirely well formed.
  //
  // Only the scheme: the comparison is textual otherwise, so a manifest and a footer that
  // disagreed about absolute versus relative form would still be refused. Iceberg writers put
  // the same location string in both, so that shape does not arise in a table anyone wrote.
  auto const expected = read_deletion_vector(fixture_ref());

  auto ref                 = fixture_ref();
  ref.referenced_data_file = "file://" + fixture_data_file_path();
  REQUIRE(read_deletion_vector(ref) == expected);
}

TEST_CASE("puffin reader rejects a non-Puffin file", "[scan][iceberg]")
{
  // A parquet data file is not a Puffin container. Pointing the reader at one must fail rather
  // than return an empty position list, which would read as "this data file has no deletes".
  auto ref        = fixture_ref();
  ref.puffin_path = require_fixture(fixture_data_file_path());

  REQUIRE_THROWS(read_deletion_vector(ref));
}

//===----------------------------------------------------------------------===//
// Footer-descriptor validation.
//
// The Iceberg spec requires content_offset/content_size_in_bytes to match the Puffin footer's
// blob descriptor exactly. Each case corrupts one field of an otherwise valid entry; every one of
// them decodes a well-formed vector and would be accepted on magic and CRC alone.
//===----------------------------------------------------------------------===//

TEST_CASE("puffin reader rejects an entry naming a different data file", "[scan][iceberg]")
{
  // The misbinding case: the blob is intact and its position count is whatever the manifest
  // claims, but the footer says it deletes from some other data file. Nothing else can catch it.
  auto ref                 = fixture_ref();
  ref.referenced_data_file = "some/other/data/file.parquet";

  REQUIRE_THROWS(read_deletion_vector(ref));
}

TEST_CASE("puffin reader rejects an offset with no blob", "[scan][iceberg]")
{
  // Off by one byte: still inside a real Puffin container, still passes the container magic
  // check, but no descriptor claims it. Reading on would decode whatever happened to be there.
  auto ref           = fixture_ref();
  ref.content_offset = kBlobOffset + 1;

  REQUIRE_THROWS(read_deletion_vector(ref));
}

TEST_CASE("puffin reader rejects a length the footer does not agree with", "[scan][iceberg]")
{
  auto ref                  = fixture_ref();
  ref.content_size_in_bytes = kBlobSize - 1;

  REQUIRE_THROWS(read_deletion_vector(ref));
}

TEST_CASE("puffin reader rejects a cardinality the footer does not agree with", "[scan][iceberg]")
{
  auto ref         = fixture_ref();
  ref.record_count = kCardinality + 1;

  REQUIRE_THROWS(read_deletion_vector(ref));
}
