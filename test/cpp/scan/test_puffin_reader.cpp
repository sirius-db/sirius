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
#include <fstream>
#include <limits>
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

//===----------------------------------------------------------------------===//
// Synthesized Puffin files.
//
// The checked-in fixture cannot express these cases: it is a well-formed vector written by a real
// writer, and what is under test here is what happens when a manifest or a blob says something no
// real writer would say. Each builder below produces a container that is valid in every respect
// EXCEPT the one being tested, so a throw names that one thing.
//===----------------------------------------------------------------------===//

uint32_t crc32_of(std::vector<uint8_t> const& data)
{
  // CRC-32/ISO-HDLC, the same one the reader computes; written out rather than shared so the test
  // does not agree with the reader by construction.
  uint32_t crc = 0xFFFFFFFFu;
  for (uint8_t byte : data) {
    crc ^= byte;
    for (int bit = 0; bit < 8; ++bit) {
      crc = (crc >> 1) ^ (0xEDB88320u & (~(crc & 1u) + 1u));
    }
  }
  return crc ^ 0xFFFFFFFFu;
}

void push_u16_le(std::vector<uint8_t>& out, uint16_t v)
{
  out.push_back(static_cast<uint8_t>(v));
  out.push_back(static_cast<uint8_t>(v >> 8));
}

void push_u32_le(std::vector<uint8_t>& out, uint32_t v)
{
  for (int i = 0; i < 4; ++i) {
    out.push_back(static_cast<uint8_t>(v >> (8 * i)));
  }
}

void push_u32_be(std::vector<uint8_t>& out, uint32_t v)
{
  for (int i = 3; i >= 0; --i) {
    out.push_back(static_cast<uint8_t>(v >> (8 * i)));
  }
}

/// A deletion-vector-v1 blob holding exactly one position, filed under 64-bit bitmap key @p key.
/// The low 32 bits are a Roaring array container, which is what a one-position vector really is.
std::vector<uint8_t> build_dv_blob(int32_t key)
{
  constexpr uint32_t kSerialCookieNoRunContainer = 12346;

  std::vector<uint8_t> roaring32;
  push_u32_le(roaring32, kSerialCookieNoRunContainer);
  push_u32_le(roaring32, 1);  // one container
  push_u16_le(roaring32, 0);  // container key (high 16 bits of the low word)
  push_u16_le(roaring32, 0);  // cardinality - 1, so: one value
  push_u32_le(roaring32, 0);  // offset header, one entry; the reader skips it
  push_u16_le(roaring32, 5);  // the value itself -- row 5 of the referenced file

  // checksummed region = DV magic + [8B num_bitmaps][4B key][roaring32]
  std::vector<uint8_t> checksummed = {0xD1, 0xD3, 0x39, 0x64};
  for (int i = 0; i < 8; ++i) {
    checksummed.push_back(i == 0 ? 1 : 0);  // num_bitmaps = 1, little-endian
  }
  push_u32_le(checksummed, static_cast<uint32_t>(key));
  checksummed.insert(checksummed.end(), roaring32.begin(), roaring32.end());

  std::vector<uint8_t> blob;
  push_u32_be(blob, static_cast<uint32_t>(checksummed.size()));
  blob.insert(blob.end(), checksummed.begin(), checksummed.end());
  push_u32_be(blob, crc32_of(checksummed));
  return blob;
}

/// Wraps @p blob in a real Puffin container whose footer agrees with it. @p fields_json is spelled
/// out because it is the one descriptor field with no other way to get it wrong.
std::string write_puffin(std::vector<uint8_t> const& blob,
                         std::string const& name,
                         int64_t cardinality,
                         std::string const& fields_json = "[]")
{
  static constexpr char kMagic[4] = {'P', 'F', 'A', '1'};
  auto const offset               = static_cast<int64_t>(sizeof(kMagic));

  std::string const footer_payload =
    std::string(R"({"blobs":[{"type":"deletion-vector-v1","fields":)") + fields_json +
    R"(,"snapshot-id":-1,"sequence-number":-1,"offset":)" + std::to_string(offset) +
    R"(,"length":)" + std::to_string(blob.size()) + R"(,"properties":{"referenced-data-file":")" +
    fixture_data_file_path() + R"(","cardinality":")" + std::to_string(cardinality) +
    R"("}}],"properties":{}})";

  std::vector<uint8_t> out(std::begin(kMagic), std::end(kMagic));
  out.insert(out.end(), blob.begin(), blob.end());
  out.insert(out.end(), std::begin(kMagic), std::end(kMagic));
  out.insert(out.end(), footer_payload.begin(), footer_payload.end());
  push_u32_le(out, static_cast<uint32_t>(footer_payload.size()));
  push_u32_le(out, 0);  // flags: bit 0 clear = uncompressed footer
  out.insert(out.end(), std::begin(kMagic), std::end(kMagic));

  auto const path = fs::temp_directory_path() / ("sirius_puffin_" + name + ".puffin");
  std::ofstream f(path, std::ios::binary | std::ios::trunc);
  f.write(reinterpret_cast<char const*>(out.data()), static_cast<std::streamsize>(out.size()));
  f.close();
  return path.string();
}

/// A manifest entry pointing at a synthesized file, correct in every field.
DeletionVectorRef synthetic_ref(std::string const& path, std::vector<uint8_t> const& blob)
{
  return {.puffin_path           = path,
          .content_offset        = 4,
          .content_size_in_bytes = static_cast<int64_t>(blob.size()),
          .referenced_data_file  = fixture_data_file_path(),
          .record_count          = 1};
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

//===----------------------------------------------------------------------===//
// Bounds on what a manifest entry may ask the reader to do.
//
// The manifest's record_count and the Puffin footer's cardinality are cross-checked against each
// other, but the TABLE writes both, so neither can be what bounds the decode: a Roaring run
// container is 4 bytes on disk and up to 65,536 positions in memory.
//===----------------------------------------------------------------------===//

TEST_CASE("puffin reader rejects a bitmap key with bit 31 set", "[scan][iceberg]")
{
  // First: the same synthesized blob under key 0 must READ. Without this the case below proves
  // only that the builder produces something unreadable.
  auto const good_blob = build_dv_blob(0);
  auto const good_path = write_puffin(good_blob, "key_zero", 1);
  auto const positions = read_deletion_vector(synthetic_ref(good_path, good_blob));
  REQUIRE(positions == std::vector<int64_t>{5});

  // Now the same vector under INT32_MIN. Widening the key to uint32_t and shifting it left by 32
  // sets bit 63, so every position under it is negative -- and positional_delete_filter searches a
  // sorted list against non-negative row offsets, so it matches nothing and the deletes vanish.
  // The count is still 1, so the record_count cross-check cannot see it.
  auto const bad_blob = build_dv_blob(std::numeric_limits<int32_t>::min());
  auto const bad_path = write_puffin(bad_blob, "key_int32_min", 1);
  REQUIRE_THROWS(read_deletion_vector(synthetic_ref(bad_path, bad_blob)));
}

TEST_CASE("puffin reader rejects an entry with no record_count", "[scan][iceberg]")
{
  // record_count is a required manifest field. Absent, the footer-to-manifest and
  // decoded-to-manifest checks both compare against nothing and the expansion is unbounded.
  auto const blob = build_dv_blob(0);
  auto const path = write_puffin(blob, "no_record_count", 1);

  auto ref         = synthetic_ref(path, blob);
  ref.record_count = -1;
  REQUIRE_THROWS(read_deletion_vector(ref));
}

TEST_CASE("puffin reader rejects a record_count above the materialization ceiling",
          "[scan][iceberg]")
{
  // A vector declaring more positions than the reader will materialize is DECLINED (the table
  // falls back to DuckDB, which streams) rather than being allowed to size a plan-time allocation.
  // The footer agrees with the manifest here, so the two cardinality checks both pass.
  auto const blob = build_dv_blob(0);
  auto const path = write_puffin(blob, "huge_record_count", kMaxDeletionVectorPositions + 1);

  auto ref         = synthetic_ref(path, blob);
  ref.record_count = kMaxDeletionVectorPositions + 1;
  REQUIRE_THROWS(read_deletion_vector(ref));
}

TEST_CASE("puffin reader rejects a descriptor whose 'fields' is not a list of integers",
          "[scan][iceberg]")
{
  auto const blob = build_dv_blob(0);

  auto const object_fields = write_puffin(blob, "fields_object", 1, "{}");
  REQUIRE_THROWS(read_deletion_vector(synthetic_ref(object_fields, blob)));

  auto const string_element = write_puffin(blob, "fields_string", 1, R"(["1"])");
  REQUIRE_THROWS(read_deletion_vector(synthetic_ref(string_element, blob)));

  // The spec requires the key to exist and to be a list of field ids; it does not require the list
  // to be empty, so a populated one must still read.
  auto const populated = write_puffin(blob, "fields_populated", 1, "[1,2]");
  REQUIRE(read_deletion_vector(synthetic_ref(populated, blob)) == std::vector<int64_t>{5});
}
