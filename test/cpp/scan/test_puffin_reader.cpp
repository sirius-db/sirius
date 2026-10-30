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
// directly rather than through ioctx, so the scheme stripping done at the datasource
// boundary does not reach it — and a failure here does not surface as an error: the table
// declines to CPU and the V3 path merely appears never to engage.
//
// No GPU: file IO and parsing only.

#include "io/io_errors.hpp"
#include "op/scan/iceberg_delete_set.hpp"
#include "op/scan/puffin_reader.hpp"
#include "scan_manager/preparation_test_support.hpp"
#include "utils/puffin_test_utils.hpp"

#include <catch.hpp>
#include <roaring/roaring.hh>

#include <algorithm>
#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <new>
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

using vector_puffin_source = sirius::test::memory_puffin_source;

class identity_changing_puffin_source final : public puffin_byte_source {
 public:
  identity_changing_puffin_source(std::streamoff size, std::string path)
    : size_(size), path_(std::move(path))
  {
  }
  bool seek(std::streamoff offset) override
  {
    if (offset < 0 || offset > size_) return false;
    offset_ = offset;
    return true;
  }
  bool read(std::span<std::byte>) override
  {
    throw sirius::io::object_changed_error(path_, "\"v1\"", "\"v2\"");
  }
  [[nodiscard]] std::streamoff size() const override { return size_; }

 private:
  std::streamoff size_;
  std::string path_;
  std::streamoff offset_ = 0;
};

class cancelling_puffin_source final : public puffin_byte_source {
 public:
  explicit cancelling_puffin_source(std::streamoff size) : size_(size) {}
  bool seek(std::streamoff offset) override
  {
    if (offset < 0 || offset > size_) return false;
    offset_ = offset;
    return true;
  }
  bool read(std::span<std::byte>) override
  {
    throw std::system_error(std::make_error_code(std::errc::operation_canceled));
  }
  [[nodiscard]] std::streamoff size() const override { return size_; }

 private:
  std::streamoff size_;
  std::streamoff offset_ = 0;
};

class throwing_puffin_source final : public puffin_byte_source {
 public:
  explicit throwing_puffin_source(std::streamoff size) : size_(size) {}
  bool seek(std::streamoff offset) override
  {
    if (offset < 0 || offset > size_) return false;
    offset_ = offset;
    return true;
  }
  bool read(std::span<std::byte>) override
  {
    throw std::runtime_error("injected footer read failure");
  }
  [[nodiscard]] std::streamoff size() const override { return size_; }

 private:
  std::streamoff size_;
  std::streamoff offset_ = 0;
};

class bad_alloc_puffin_source final : public puffin_byte_source {
 public:
  explicit bad_alloc_puffin_source(std::streamoff size) : size_(size) {}
  bool seek(std::streamoff offset) override
  {
    if (offset < 0 || offset > size_) return false;
    offset_ = offset;
    return true;
  }
  bool read(std::span<std::byte>) override { throw std::bad_alloc{}; }
  [[nodiscard]] std::streamoff size() const override { return size_; }

 private:
  std::streamoff size_;
  std::streamoff offset_ = 0;
};

class footer_magic_short_source final : public puffin_byte_source {
 public:
  explicit footer_magic_short_source(std::vector<std::byte> bytes) : bytes_(std::move(bytes)) {}
  bool seek(std::streamoff offset) override
  {
    if (offset < 0 || static_cast<std::uint64_t>(offset) > bytes_.size()) return false;
    offset_ = offset;
    return true;
  }
  bool read(std::span<std::byte> destination) override
  {
    ++read_count_;
    if (read_count_ == 4) return false;
    if (offset_ < 0 || static_cast<std::uint64_t>(offset_) > bytes_.size() ||
        destination.size() > bytes_.size() - static_cast<std::uint64_t>(offset_))
      return false;
    std::copy_n(bytes_.data() + offset_, destination.size(), destination.data());
    offset_ += static_cast<std::streamoff>(destination.size());
    return true;
  }
  [[nodiscard]] std::streamoff size() const override
  {
    return static_cast<std::streamoff>(bytes_.size());
  }

 private:
  std::vector<std::byte> bytes_;
  std::streamoff offset_ = 0;
  int read_count_        = 0;
};

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

std::vector<int64_t> read_for_test(DeletionVectorRef ref,
                                   bool charged,
                                   puffin_byte_source* supplied_source = nullptr)
{
  if (!charged)
    return supplied_source ? read_deletion_vector(ref, *supplied_source)
                           : read_deletion_vector(ref);
  using namespace sirius::scan_manager;
  test::test_reservation_provider provider;
  auto path = ref.puffin_path;
  if (path.starts_with("file://")) path.erase(0, 7);
  std::error_code error;
  auto size              = fs::file_size(path, error);
  ref.file_size_in_bytes = error ? 0 : size;
  auto positions =
    static_cast<uint64_t>(std::clamp<int64_t>(ref.record_count, 0, kMaxDeletionVectorPositions)) *
    8;
  auto blob            = static_cast<uint64_t>(std::max<int64_t>(ref.content_size_in_bytes, 0));
  auto parser          = static_cast<uint64_t>(ref.file_size_in_bytes) * 14 + 512;
  provider.grant_bytes = positions + blob + parser;
  preparation_ledger ledger(provider);
  std::array scans{scan_envelope{{{{blob, parser, 0}, positions}}, {}, true}};
  REQUIRE(ledger.admit(scans, memory_space_id(cucascade::memory::Tier::HOST, 0)).deferred);
  ledger.register_unit({1, 1}, positions);
  auto permit = ledger.acquire_permit({1, 1});
  REQUIRE(permit);
  auto allocator = ledger.allocator(permit);
  auto result    = supplied_source
                     ? read_deletion_vector_charged(DeletionVectorRefView{ref.puffin_path,
                                                                       ref.content_offset,
                                                                       ref.content_size_in_bytes,
                                                                       ref.referenced_data_file,
                                                                       ref.file_size_in_bytes,
                                                                       ref.record_count},
                                                 allocator,
                                                 ref.referenced_data_file,
                                                 std::make_shared<int>(0),
                                                 1,
                                                 *supplied_source)
                     : read_deletion_vector_charged(ref, allocator);
  return {result->positions.begin(), result->positions.end()};
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

/// A deletion-vector-v1 blob whose single container is a BITSET holding @p set_bits set bits,
/// while its descriptive header DECLARES @p declared_cardinality values.
///
/// Roaring picks a bitset container above 4096 values, and the portable format stores that
/// container's cardinality as a number rather than deriving it from the bits. So the two can
/// disagree, and nothing in the container's own bytes is malformed when they do -- which is why
/// `roaring_bitmap_portable_deserialize_size` and `readSafe` both accept this and the CRC is
/// perfectly valid over it.
std::vector<uint8_t> build_dv_blob_bitset(uint32_t declared_cardinality, uint32_t set_bits)
{
  constexpr uint32_t kSerialCookieNoRunContainer = 12346;
  constexpr size_t kBitsetBytes                  = 8192;

  std::vector<uint8_t> roaring32;
  push_u32_le(roaring32, kSerialCookieNoRunContainer);
  push_u32_le(roaring32, 1);  // one container
  push_u16_le(roaring32, 0);  // container key (high 16 bits of the low word)
  push_u16_le(roaring32, static_cast<uint16_t>(declared_cardinality - 1));
  // Offset header: where the container's bytes start, counted from the cookie. cookie + size +
  // one descriptive pair + one offset = 16.
  push_u32_le(roaring32, 16);

  std::vector<uint8_t> bitset(kBitsetBytes, 0);
  if (set_bits == kBitsetBytes * 8) {
    std::fill(bitset.begin(), bitset.end(), 0xFF);
  } else {
    // Bit 5 alone, so the container is legibly a single position and nothing else.
    REQUIRE(set_bits == 1);
    bitset[0] = 0x20;
  }
  roaring32.insert(roaring32.end(), bitset.begin(), bitset.end());

  std::vector<uint8_t> checksummed = {0xD1, 0xD3, 0x39, 0x64};
  for (int i = 0; i < 8; ++i) {
    checksummed.push_back(i == 0 ? 1 : 0);  // num_bitmaps = 1, little-endian
  }
  push_u32_le(checksummed, 0);  // bitmap key 0
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
  constexpr int64_t offset = 4;

  std::string const footer_payload =
    std::string(R"({"blobs":[{"type":"deletion-vector-v1","fields":)") + fields_json +
    R"(,"snapshot-id":-1,"sequence-number":-1,"offset":)" + std::to_string(offset) +
    R"(,"length":)" + std::to_string(blob.size()) + R"(,"properties":{"referenced-data-file":")" +
    fixture_data_file_path() + R"(","cardinality":")" + std::to_string(cardinality) +
    R"("}}],"properties":{}})";

  auto out = sirius::test::puffin_container(std::as_bytes(std::span{blob}), footer_payload);

  auto const path = fs::temp_directory_path() / ("sirius_puffin_" + name + ".puffin");
  std::ofstream f(path, std::ios::binary | std::ios::trunc);
  f.write(reinterpret_cast<char const*>(out.data()), static_cast<std::streamsize>(out.size()));
  f.close();
  return path.string();
}

/// A manifest entry pointing at a synthesized file, correct in every field.
DeletionVectorRef synthetic_ref(std::string const& path,
                                std::vector<uint8_t> const& blob,
                                int64_t record_count = 1)
{
  return {.puffin_path           = path,
          .content_offset        = 4,
          .content_size_in_bytes = static_cast<int64_t>(blob.size()),
          .referenced_data_file  = fixture_data_file_path(),
          .record_count          = record_count};
}

}  // namespace

TEST_CASE("puffin reader reads the fixture deletion vector", "[scan][iceberg]")
{
  auto charged = GENERATE(false, true);
  auto read    = [charged](DeletionVectorRef const& ref) { return read_for_test(ref, charged); };
  auto const positions = read(fixture_ref());

  // The fixture deletes 2 of its 5 rows — the same 3 survivors the integration case asserts.
  REQUIRE(positions.size() == 2);
  REQUIRE(std::is_sorted(positions.begin(), positions.end()));
}

TEST_CASE("charged Puffin reader accepts an injected byte source",
          "[scan][iceberg][charged_reader]")
{
  auto ref = fixture_ref();
  std::ifstream input(ref.puffin_path, std::ios::binary | std::ios::ate);
  REQUIRE(input);
  auto const size = input.tellg();
  REQUIRE(size >= 0);
  std::vector<std::byte> bytes(static_cast<std::size_t>(size));
  input.seekg(0);
  input.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
  REQUIRE(input);

  vector_puffin_source source(std::move(bytes));
  CHECK(read_for_test(ref, true, &source) == read_deletion_vector(ref));
}

TEST_CASE("puffin reader accepts a file:// URI", "[scan][iceberg]")
{
  auto charged = GENERATE(false, true);
  auto read    = [charged](DeletionVectorRef const& ref) { return read_for_test(ref, charged); };
  auto const expected = read(fixture_ref());

  // What an Apache writer records: an absolute file:// URI. Before the reader stripped the
  // scheme, std::ifstream simply failed to open this and the whole table declined to CPU.
  auto ref        = fixture_ref();
  ref.puffin_path = "file://" + fs::absolute(fixture_puffin_path()).string();
  REQUIRE(read(ref) == expected);
}

TEST_CASE("puffin reader accepts a file:// URI on the referenced data file", "[scan][iceberg]")
{
  auto charged = GENERATE(false, true);
  auto read    = [charged](DeletionVectorRef const& ref) { return read_for_test(ref, charged); };
  // The manifest and the Puffin footer are free to disagree about the SCHEME while naming the
  // same file. Comparing them literally would refuse tables that are entirely well formed.
  //
  // Only the scheme: the comparison is textual otherwise, so a manifest and a footer that
  // disagreed about absolute versus relative form would still be refused. Iceberg writers put
  // the same location string in both, so that shape does not arise in a table anyone wrote.
  auto const expected = read(fixture_ref());

  auto ref                 = fixture_ref();
  ref.referenced_data_file = "file://" + fixture_data_file_path();
  REQUIRE(read(ref) == expected);
}

TEST_CASE("puffin reader rejects a non-Puffin file", "[scan][iceberg]")
{
  auto charged = GENERATE(false, true);
  auto read    = [charged](DeletionVectorRef const& ref) { return read_for_test(ref, charged); };
  // A parquet data file is not a Puffin container. Pointing the reader at one must fail rather
  // than return an empty position list, which would read as "this data file has no deletes".
  auto ref        = fixture_ref();
  ref.puffin_path = require_fixture(fixture_data_file_path());

  REQUIRE_THROWS(read(ref));
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
  auto charged = GENERATE(false, true);
  auto read    = [charged](DeletionVectorRef const& ref) { return read_for_test(ref, charged); };
  // The misbinding case: the blob is intact and its position count is whatever the manifest
  // claims, but the footer says it deletes from some other data file. Nothing else can catch it.
  auto ref                 = fixture_ref();
  ref.referenced_data_file = "some/other/data/file.parquet";

  REQUIRE_THROWS(read(ref));
}

TEST_CASE("puffin reader rejects an offset with no blob", "[scan][iceberg]")
{
  auto charged = GENERATE(false, true);
  auto read    = [charged](DeletionVectorRef const& ref) { return read_for_test(ref, charged); };
  // Off by one byte: still inside a real Puffin container, still passes the container magic
  // check, but no descriptor claims it. Reading on would decode whatever happened to be there.
  auto ref           = fixture_ref();
  ref.content_offset = kBlobOffset + 1;

  REQUIRE_THROWS(read(ref));
}

TEST_CASE("puffin reader rejects a length the footer does not agree with", "[scan][iceberg]")
{
  auto charged = GENERATE(false, true);
  auto read    = [charged](DeletionVectorRef const& ref) { return read_for_test(ref, charged); };
  auto ref     = fixture_ref();
  ref.content_size_in_bytes = kBlobSize - 1;

  REQUIRE_THROWS(read(ref));
}

TEST_CASE("puffin reader rejects a cardinality the footer does not agree with", "[scan][iceberg]")
{
  auto charged = GENERATE(false, true);
  auto read    = [charged](DeletionVectorRef const& ref) { return read_for_test(ref, charged); };
  auto ref     = fixture_ref();
  ref.record_count = kCardinality + 1;

  REQUIRE_THROWS(read(ref));
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
  auto charged = GENERATE(false, true);
  auto read    = [charged](DeletionVectorRef const& ref) { return read_for_test(ref, charged); };
  // First: the same synthesized blob under key 0 must READ. Without this the case below proves
  // only that the builder produces something unreadable.
  auto const good_blob = build_dv_blob(0);
  auto const good_path = write_puffin(good_blob, "key_zero", 1);
  auto const positions = read(synthetic_ref(good_path, good_blob));
  REQUIRE(positions == std::vector<int64_t>{5});

  // Now the same vector under INT32_MIN. Widening the key to uint32_t and shifting it left by 32
  // sets bit 63, so every position under it is negative -- and positional_delete_filter searches a
  // sorted list against non-negative row offsets, so it matches nothing and the deletes vanish.
  // The count is still 1, so the record_count cross-check cannot see it.
  auto const bad_blob = build_dv_blob(std::numeric_limits<int32_t>::min());
  auto const bad_path = write_puffin(bad_blob, "key_int32_min", 1);
  REQUIRE_THROWS(read(synthetic_ref(bad_path, bad_blob)));
}

TEST_CASE("puffin reader rejects a bitset declaring fewer values than it holds", "[scan][iceberg]")
{
  auto charged = GENERATE(false, true);
  auto read    = [charged](DeletionVectorRef const& ref) { return read_for_test(ref, charged); };
  // Declared 4098, actually all 65,536 bits set. `bitset_container_read` trusts the declared
  // number, so cardinality() answers 4098 and the destination is sized to 4098 -- then
  // toUint32Array(), which takes no capacity argument, emits 65,536 values into it. The
  // record_count cross-check only runs on what survives that write.
  constexpr uint32_t kDeclared = 4098;
  auto const blob              = build_dv_blob_bitset(kDeclared, 8192 * 8);
  auto const path              = write_puffin(blob, "bitset_under_declared", kDeclared);

  REQUIRE_THROWS(read(synthetic_ref(path, blob, kDeclared)));
}

TEST_CASE("puffin reader rejects a bitset declaring more values than it holds", "[scan][iceberg]")
{
  auto charged = GENERATE(false, true);
  auto read    = [charged](DeletionVectorRef const& ref) { return read_for_test(ref, charged); };
  // The mirror image, and the one that returns a WRONG ANSWER rather than corrupting memory:
  // declared 4098 with a single bit set. resize() leaves 4097 zero-initialized tail entries,
  // extraction writes only the value 5, and every remaining zero reads as delete position 0. The
  // list is still sorted and its size still equals record_count, so nothing downstream objects and
  // row 0 is dropped from a table that keeps it.
  constexpr uint32_t kDeclared = 4098;
  auto const blob              = build_dv_blob_bitset(kDeclared, 1);
  auto const path              = write_puffin(blob, "bitset_over_declared", kDeclared);

  REQUIRE_THROWS(read(synthetic_ref(path, blob, kDeclared)));
}

TEST_CASE("puffin reader rejects an entry with no record_count", "[scan][iceberg]")
{
  auto charged = GENERATE(false, true);
  auto read    = [charged](DeletionVectorRef const& ref) { return read_for_test(ref, charged); };
  // record_count is a required manifest field. Absent, the footer-to-manifest and
  // decoded-to-manifest checks both compare against nothing and the expansion is unbounded.
  auto const blob = build_dv_blob(0);
  auto const path = write_puffin(blob, "no_record_count", 1);

  auto ref         = synthetic_ref(path, blob);
  ref.record_count = -1;
  REQUIRE_THROWS(read(ref));
}

TEST_CASE("puffin reader rejects a record_count above the materialization ceiling",
          "[scan][iceberg]")
{
  auto charged = GENERATE(false, true);
  auto read    = [charged](DeletionVectorRef const& ref) { return read_for_test(ref, charged); };
  // A vector declaring more positions than the reader will materialize is DECLINED (the table
  // falls back to DuckDB, which streams) rather than being allowed to size a plan-time allocation.
  // The footer agrees with the manifest here, so the two cardinality checks both pass.
  auto const blob = build_dv_blob(0);
  auto const path = write_puffin(blob, "huge_record_count", kMaxDeletionVectorPositions + 1);

  auto ref         = synthetic_ref(path, blob);
  ref.record_count = kMaxDeletionVectorPositions + 1;
  REQUIRE_THROWS(read(ref));
}

TEST_CASE("puffin reader rejects a descriptor whose 'fields' is not a list of integers",
          "[scan][iceberg]")
{
  auto charged    = GENERATE(false, true);
  auto read       = [charged](DeletionVectorRef const& ref) { return read_for_test(ref, charged); };
  auto const blob = build_dv_blob(0);

  auto const object_fields = write_puffin(blob, "fields_object", 1, "{}");
  REQUIRE_THROWS(read(synthetic_ref(object_fields, blob)));

  auto const string_element = write_puffin(blob, "fields_string", 1, R"(["1"])");
  REQUIRE_THROWS(read(synthetic_ref(string_element, blob)));

  // The spec requires the key to exist and to be a list of field ids; it does not require the list
  // to be empty, so a populated one must still read.
  auto const populated = write_puffin(blob, "fields_populated", 1, "[1,2]");
  REQUIRE(read(synthetic_ref(populated, blob)) == std::vector<int64_t>{5});
}

TEST_CASE(
  "Charged Puffin results retain backing until release and free temporary memory before returning "
  "its permit",
  "[scan_preparation][iceberg][charged_reader]")
{
  using namespace sirius::scan_manager;
  test::test_reservation_provider provider;
  provider.grant_bytes = 1024 * 1024;
  preparation_ledger ledger(provider);
  auto ref               = fixture_ref();
  ref.file_size_in_bytes = fs::file_size(ref.puffin_path);
  std::array scans{scan_envelope{{{{44, 64 * 1024, 0}, 16}}, {}, true}};
  REQUIRE(ledger.admit(scans, memory_space_id(cucascade::memory::Tier::HOST, 0)).deferred);
  ledger.register_unit({1, 1}, 16);
  auto permit = ledger.acquire_permit({1, 1});
  REQUIRE(permit);
  auto allocator = ledger.allocator(permit);
  auto set       = read_deletion_vector_charged(ref, allocator);
  REQUIRE(set);
  CHECK(std::vector<int64_t>(set->positions.begin(), set->positions.end()) ==
        read_deletion_vector(ref));
  CHECK(ledger.charged_bytes({1, 1}) == 16);
  CHECK(set->source_dv_count == 1);
  permit.release();
  CHECK(ledger.permits_in_flight() == 0);
  CHECK(provider.seen->allocated_bytes == 16);
  set.reset();
  CHECK(ledger.charged_bytes({1, 1}) == 0);
  CHECK(provider.seen->allocated_bytes == 0);
}

TEST_CASE("Charged Roaring expansion agrees with CRoaring for arrays, bitsets and runs",
          "[scan_preparation][iceberg][charged_reader]")
{
  auto shape = GENERATE(0, 1, 2, 3);
  roaring::Roaring bitmap;
  if (shape == 0)
    for (uint32_t i : {1u, 5u, 65537u, 0xfffffffeu})
      bitmap.add(i);
  else if (shape == 1)
    for (uint32_t i = 0; i < 65536; i += 2)
      bitmap.add(i);
  else {
    bitmap.addRange(0, 65536);
    if (shape == 3)
      for (uint32_t i : {65537u, 131073u, 196609u})
        bitmap.add(i);
    REQUIRE(bitmap.runOptimize());
  }
  std::vector<uint8_t> serial(bitmap.getSizeInBytes());
  REQUIRE(bitmap.write(reinterpret_cast<char*>(serial.data())) == serial.size());
  std::vector<uint8_t> checksummed{0xD1, 0xD3, 0x39, 0x64};
  for (int i = 0; i < 8; ++i)
    checksummed.push_back(i == 0 ? 1 : 0);
  push_u32_le(checksummed, 1);  // nonzero 64-bit high key
  checksummed.insert(checksummed.end(), serial.begin(), serial.end());
  std::vector<uint8_t> blob;
  push_u32_be(blob, checksummed.size());
  blob.insert(blob.end(), checksummed.begin(), checksummed.end());
  push_u32_be(blob, crc32_of(checksummed));
  auto path = write_puffin(blob, "charged_roaring_" + std::to_string(shape), bitmap.cardinality());
  auto ref  = synthetic_ref(path, blob, bitmap.cardinality());
  CHECK(read_for_test(ref, true) == read_deletion_vector(ref));
}

TEST_CASE("Charged Puffin failures distinguish corrupt input from missing files",
          "[scan_preparation][iceberg][charged_reader][failure]")
{
  auto corrupt = GENERATE(false, true);
  auto ref     = fixture_ref();
  if (corrupt)
    ref.referenced_data_file = "wrong.parquet";
  else
    ref.puffin_path += ".missing";
  try {
    read_for_test(ref, true);
    FAIL("invalid Puffin input was accepted");
  } catch (std::exception const& error) {
    auto failure = sirius::transparent::classify_failure(std::current_exception());
    CHECK(failure.cause == (corrupt ? sirius::transparent::late_failure_cause::physical_input
                                    : sirius::transparent::late_failure_cause::reader_io));
    if (corrupt) {
      auto const* physical =
        dynamic_cast<sirius::op::scan::unsupported_physical_input const*>(&error);
      REQUIRE(physical);
      CHECK(physical->reason == sirius::op::scan::verdict_reason::iceberg_delete_corrupt);
      CHECK(std::string(error.what()).find("referenced-data-file") != std::string::npos);
    } else
      CHECK(std::string(error.what()).find("Cannot open file") != std::string::npos);
  }
}

TEST_CASE("Charged Puffin preserves an opened-object identity failure",
          "[scan_preparation][iceberg][charged_reader][failure]")
{
  auto const ref = fixture_ref();
  std::error_code error;
  auto const size = fs::file_size(ref.puffin_path, error);
  REQUIRE_FALSE(error);
  identity_changing_puffin_source source(static_cast<std::streamoff>(size), ref.puffin_path);
  try {
    read_for_test(ref, true, &source);
    FAIL("identity change was not reported");
  } catch (sirius::io::object_changed_error const& failure) {
    CHECK(failure.object_path() == ref.puffin_path);
    CHECK(failure.expected_tag() == "\"v1\"");
    CHECK(failure.observed_tag() == "\"v2\"");
    auto classified = sirius::transparent::classify_failure(std::current_exception());
    CHECK(classified.cause == sirius::transparent::late_failure_cause::object_changed);
  }
}

TEST_CASE("Charged Puffin preserves cancellation instead of classifying it as corrupt input",
          "[scan_preparation][iceberg][charged_reader][failure]")
{
  auto const ref = fixture_ref();
  std::error_code error;
  auto const size = fs::file_size(ref.puffin_path, error);
  REQUIRE_FALSE(error);
  cancelling_puffin_source source(static_cast<std::streamoff>(size));
  CHECK_THROWS_AS(read_for_test(ref, true, &source), std::system_error);
  try {
    read_for_test(ref, true, &source);
  } catch (...) {
    CHECK(std::current_exception() != nullptr);
    CHECK(sirius::transparent::classify_failure(std::current_exception()).cause ==
          sirius::transparent::late_failure_cause::other);
  }
}

TEST_CASE("Charged Puffin classifies thrown magic reads as reader I/O",
          "[scan_preparation][iceberg][charged_reader][failure]")
{
  auto const ref = fixture_ref();
  std::error_code error;
  auto const size = fs::file_size(ref.puffin_path, error);
  REQUIRE_FALSE(error);
  throwing_puffin_source source(static_cast<std::streamoff>(size));
  try {
    read_for_test(ref, true, &source);
    FAIL("injected footer read failure was not reported");
  } catch (sirius::transparent::classified_execution_error const& failure) {
    CHECK(failure.cause == sirius::transparent::late_failure_cause::reader_io);
    CHECK(std::string(failure.what()).find("injected footer read failure") != std::string::npos);
  }
}

TEST_CASE("Charged Puffin preserves allocation failures from magic reads",
          "[scan_preparation][iceberg][charged_reader][failure]")
{
  auto const ref = fixture_ref();
  std::error_code error;
  auto const size = fs::file_size(ref.puffin_path, error);
  REQUIRE_FALSE(error);
  bad_alloc_puffin_source source(static_cast<std::streamoff>(size));
  CHECK_THROWS_AS(read_for_test(ref, true, &source), std::bad_alloc);
}

TEST_CASE("Charged Puffin reports a short footer magic read as missing magic",
          "[scan_preparation][iceberg][charged_reader][failure]")
{
  auto const ref = fixture_ref();
  std::ifstream input(ref.puffin_path, std::ios::binary);
  REQUIRE(input);
  std::vector<char> raw((std::istreambuf_iterator<char>(input)), {});
  std::vector<std::byte> bytes(raw.size());
  std::memcpy(bytes.data(), raw.data(), raw.size());
  footer_magic_short_source source(std::move(bytes));
  try {
    read_for_test(ref, true, &source);
    FAIL("short footer magic read was accepted");
  } catch (sirius::op::scan::unsupported_physical_input const& failure) {
    CHECK(failure.reason == sirius::op::scan::verdict_reason::iceberg_delete_corrupt);
    CHECK(std::string(failure.what()).find("Missing footer magic") != std::string::npos);
  }
}

TEST_CASE("Charged Puffin matches legacy Roaring validation and diagnostics",
          "[scan_preparation][iceberg][charged_reader]")
{
  auto scenario = GENERATE(std::string("bitset_under"),
                           std::string("bitset_over"),
                           std::string("array_order"),
                           std::string("key_order"),
                           std::string("run_empty"),
                           std::string("run_overflow"),
                           std::string("run_overlap"),
                           std::string("run_adjacent"),
                           std::string("bitmap_budget"),
                           std::string("late_corruption"),
                           std::string("run_header"));
  CAPTURE(scenario);
  int64_t declared = 1;
  std::vector<uint8_t> blob;
  if (scenario.starts_with("bitset")) {
    declared = 4098;
    blob     = build_dv_blob_bitset(declared, scenario == "bitset_under" ? 65536 : 1);
  } else {
    std::vector<uint8_t> bitmap;
    if (scenario.starts_with("run")) {
      push_u32_le(bitmap, 12347);
      bitmap.push_back(1);
      push_u16_le(bitmap, 0);
      push_u16_le(bitmap, 0);
      push_u16_le(bitmap, scenario == "run_empty" ? 0 : scenario == "run_header" ? 1 : 2);
      if (scenario != "run_empty") {
        push_u16_le(bitmap, scenario == "run_overflow" ? 65535 : 1);
        push_u16_le(bitmap, 1);
        if (scenario != "run_header") {
          push_u16_le(bitmap, scenario == "run_adjacent" ? 3 : 2);
          push_u16_le(bitmap, 0);
        } else {
          declared = 2;
        }
      }
    } else {
      push_u32_le(bitmap, 12346);
      push_u32_le(bitmap, 2);
      push_u16_le(bitmap, 0);
      push_u16_le(bitmap, 1);
      push_u16_le(bitmap, scenario == "key_order" ? 0 : 1);
      push_u16_le(bitmap, 1);
      push_u32_le(bitmap, 24);
      push_u32_le(bitmap, 28);
      push_u16_le(bitmap, 5);
      push_u16_le(bitmap, scenario == "array_order" ? 4 : 6);
      push_u16_le(bitmap, 5);
      push_u16_le(bitmap, scenario == "late_corruption" ? 4 : 6);
      declared = scenario == "bitmap_budget" || scenario == "late_corruption" ? 3 : 4;
    }
    std::vector<uint8_t> checksummed = {0xD1, 0xD3, 0x39, 0x64};
    for (int i = 0; i < 8; ++i)
      checksummed.push_back(i == 0 ? 1 : 0);
    push_u32_le(checksummed, 0);
    checksummed.insert(checksummed.end(), bitmap.begin(), bitmap.end());
    push_u32_be(blob, static_cast<uint32_t>(checksummed.size()));
    blob.insert(blob.end(), checksummed.begin(), checksummed.end());
    push_u32_be(blob, crc32_of(checksummed));
  }
  auto path  = write_puffin(blob, "diagnostic_" + scenario, declared);
  auto ref   = synthetic_ref(path, blob, declared);
  auto error = [&](bool charged) {
    try {
      (void)read_for_test(ref, charged);
    } catch (std::runtime_error const& e) {
      return std::string(e.what());
    }
    return std::string{};
  };
  if (scenario == "run_header") {
    CHECK(read_for_test(ref, false) == std::vector<int64_t>{1, 2});
    CHECK(read_for_test(ref, true) == read_for_test(ref, false));
  } else {
    auto legacy = error(false);
    REQUIRE_FALSE(legacy.empty());
    CHECK(error(true) == legacy);
  }
}

TEST_CASE("Neutral DV payload retains only actual decoded positions without a manifest",
          "[scan_preparation][dv_payload]")
{
  using namespace sirius::scan_manager;
  auto shape = GENERATE(0, 1, 2, 3, 4);
  roaring::Roaring bitmap;
  if (shape == 0)
    for (auto v : {1u, 5u, 65537u, 0xfffffffeu})
      bitmap.add(v);
  if (shape == 1)
    for (uint32_t v = 0; v < 65536; v += 2)
      bitmap.add(v);
  if (shape == 2) {
    bitmap.addRange(0, 65536);
    REQUIRE(bitmap.runOptimize());
  }
  if (shape == 4) bitmap.add(5);
  std::vector<uint8_t> serial(bitmap.getSizeInBytes());
  REQUIRE(bitmap.write(reinterpret_cast<char*>(serial.data())) == serial.size());
  std::vector<uint8_t> checksummed{0xD1, 0xD3, 0x39, 0x64};
  for (int i = 0; i < 8; ++i)
    checksummed.push_back(i == 0 ? (shape == 4 ? 2 : 1) : 0);
  for (int i = 0; i < (shape == 4 ? 2 : 1); ++i) {
    push_u32_le(checksummed, 1);
    checksummed.insert(checksummed.end(), serial.begin(), serial.end());
  }
  std::vector<uint8_t> blob;
  push_u32_be(blob, checksummed.size());
  blob.insert(blob.end(), checksummed.begin(), checksummed.end());
  push_u32_be(blob, crc32_of(checksummed));
  test::test_reservation_provider provider;
  provider.grant_bytes = 1024 * 1024;
  preparation_ledger ledger(provider);
  auto bytes = bitmap.cardinality() * sizeof(int64_t) * (shape == 4 ? 2 : 1);
  std::array scans{scan_envelope{{{{0, 0, 0}, bytes}}, {}, true}};
  REQUIRE(ledger.admit(scans, memory_space_id(cucascade::memory::Tier::HOST, 0)).deferred);
  ledger.register_unit({1, 1}, bytes);
  auto permit = ledger.acquire_permit({1, 1});
  REQUIRE(permit);
  auto allocator = ledger.allocator(permit);
  auto result    = decode_deletion_vector_payload(blob, allocator);
  auto positions = result->positions();
  CHECK(positions.size() == bitmap.cardinality());
  size_t index = 0;
  for (auto low : bitmap)
    REQUIRE(positions[index++] == ((int64_t{1} << 32) | low));
  CHECK(ledger.charged_bytes({1, 1}) == bytes);
  CHECK(provider.seen->allocations == (bytes ? 1 : 0));
  permit.release();
  CHECK(provider.seen->allocated_bytes == bytes);
  result.reset();
  CHECK(provider.seen->allocated_bytes == 0);
  CHECK(ledger.charged_bytes({1, 1}) == 0);
}

TEST_CASE("Neutral DV payload validates corruption before allocating positions",
          "[scan_preparation][dv_payload][failure]")
{
  using namespace sirius::scan_manager;
  auto mode = GENERATE("short",
                       "magic",
                       "crc",
                       "length",
                       "trailing",
                       "inner_trailing",
                       "negative_key",
                       "bitmap",
                       "negative_count");
  INFO(mode);
  auto blob = build_dv_blob(0);
  if (std::string_view(mode) == "short") blob.resize(11);
  if (std::string_view(mode) == "magic") blob[4] ^= 1;
  if (std::string_view(mode) == "crc") blob.back() ^= 1;
  if (std::string_view(mode) == "length") blob[0] = 0x7f;
  if (std::string_view(mode) == "trailing") blob.push_back(0);
  if (std::string_view(mode) == "negative_key") blob = build_dv_blob(-1);
  if (std::string_view(mode) == "bitmap") blob = build_dv_blob_bitset(4098, 1);
  if (std::string_view(mode) == "inner_trailing" || std::string_view(mode) == "negative_count") {
    std::vector<uint8_t> data(blob.begin() + 4, blob.end() - 4);
    if (std::string_view(mode) == "inner_trailing")
      data.push_back(0);
    else
      data[11] = 0x80;
    blob.clear();
    push_u32_be(blob, data.size());
    blob.insert(blob.end(), data.begin(), data.end());
    push_u32_be(blob, crc32_of(data));
  }
  test::test_reservation_provider provider;
  preparation_ledger ledger(provider);
  std::array scans{scan_envelope{{{{0, 0, 0}, 16}}, {}, true}};
  REQUIRE(ledger.admit(scans, memory_space_id(cucascade::memory::Tier::HOST, 0)).deferred);
  ledger.register_unit({1, 1}, 16);
  auto permit = ledger.acquire_permit({1, 1});
  REQUIRE(permit);
  auto allocator = ledger.allocator(permit);
  CHECK_THROWS_AS(decode_deletion_vector_payload(blob, allocator), std::runtime_error);
  CHECK(provider.seen->allocations == 0);
  CHECK(ledger.charged_bytes({1, 1}) == 0);
}

TEST_CASE("Neutral DV payload obeys independent limits and retained grant",
          "[scan_preparation][dv_payload][failure]")
{
  using namespace sirius::scan_manager;
  test::test_reservation_provider provider;
  preparation_ledger ledger(provider);
  std::array scans{scan_envelope{{{{0, 0, 0}, 8}}, {}, true}};
  REQUIRE(ledger.admit(scans, memory_space_id(cucascade::memory::Tier::HOST, 0)).deferred);
  ledger.register_unit({1, 1}, 8);
  auto permit = ledger.acquire_permit({1, 1});
  REQUIRE(permit);
  auto allocator = ledger.allocator(permit);
  auto blob      = build_dv_blob(0);
  CHECK_THROWS_AS(decode_deletion_vector_payload(blob, allocator, {blob.size() - 1, 1}),
                  preparation_resource_error);
  CHECK_THROWS_MATCHES(
    decode_deletion_vector_payload(blob, allocator, {blob.size(), 0}),
    preparation_resource_error,
    Catch::Matchers::Message("roaring: decoded positions exceed the configured limit of 0"));
  CHECK_THROWS_AS(
    decode_deletion_vector_payload(blob, allocator, {kMaxDeletionVectorPayloadBytes + 1, 1}),
    std::invalid_argument);
  CHECK_THROWS_AS(
    decode_deletion_vector_payload(blob, allocator, {blob.size(), kMaxDeletionVectorPositions + 1}),
    std::invalid_argument);
  CHECK(provider.seen->allocations == 0);
  auto result = decode_deletion_vector_payload(blob, allocator, {blob.size(), 1});
  CHECK(result->positions().size() == 1);
  CHECK_THROWS_AS(decode_deletion_vector_payload(blob, allocator), preparation_resource_error);
  CHECK(ledger.charged_bytes({1, 1}) == 8);
  result.reset();
  provider.seen->fail_allocation = true;
  CHECK_THROWS_AS(decode_deletion_vector_payload(blob, allocator), preparation_resource_error);
  CHECK(ledger.charged_bytes({1, 1}) == 0);
}

namespace {
std::vector<std::byte> metadata_container(std::string const& footer, uint32_t flags = 0)
{
  std::array<std::byte, 128> opaque{};
  return sirius::test::puffin_container(opaque, footer, flags);
}
std::string metadata_descriptor(std::string extra = {}, int64_t offset = 4, int64_t length = 20)
{
  return "{\"type\":\"deletion-vector-v1\",\"fields\":[1,2],\"snapshot-id\":-1,\"sequence-number\":"
         "-1,\"offset\":" +
         std::to_string(offset) + ",\"length\":" + std::to_string(length) + extra + "}";
}
class metadata_only_source final : public puffin_byte_source {
 public:
  explicit metadata_only_source(std::vector<std::byte> bytes) : inner_(std::move(bytes)) {}
  bool seek(std::streamoff offset) override
  {
    offset_ = offset;
    return inner_.seek(offset);
  }
  bool read(std::span<std::byte> output) override
  {
    if (offset_ < 132 && offset_ + static_cast<std::streamoff>(output.size()) > 4) ++data_reads;
    auto result = inner_.read(output);
    offset_ += output.size();
    return result;
  }
  std::streamoff size() const override { return inner_.size(); }
  size_t data_reads = 0;

 private:
  vector_puffin_source inner_;
  std::streamoff offset_ = 0;
};
}  // namespace

TEST_CASE("Neutral Puffin metadata preserves blob properties and charged ownership",
          "[scan_preparation][puffin_metadata]")
{
  using namespace sirius::scan_manager;
  test::test_reservation_provider provider;
  provider.grant_bytes = 1024 * 1024;
  preparation_ledger ledger(provider);
  std::array scans{scan_envelope{{{{0, 0, 0}, 512 * 1024}}, {}, true}};
  REQUIRE(ledger.admit(scans, memory_space_id(cucascade::memory::Tier::HOST, 0)).deferred);
  ledger.register_unit({1, 1}, 512 * 1024);
  auto permit = ledger.acquire_permit({1, 1});
  REQUIRE(permit);
  auto allocator = ledger.allocator(permit);
  std::shared_ptr<puffin_container_metadata const> metadata;
  {
    auto footer =
      std::string(
        R"({"properties":{"created-by":"test","empty":"","":"empty-key","nul":"a\u0000b"},"blobs":[)") +
      metadata_descriptor(
        R"(,"properties":{"cardinality":"not-an-integer","custom-snapshot":"7","empty":""})") +
      "," +
      R"({"type":"unknown-blob","offset":40,"length":20,"fields":[],"snapshot-id":12,"sequence-number":34,"compression-codec":"zstd"}]})";
    metadata_only_source source(metadata_container(footer));
    metadata = read_puffin_metadata(source, allocator);
    CHECK(source.data_reads == 0);
    CHECK(metadata->file_size() == source.size());
    CHECK(metadata->footer_start() == 132);
  }
  REQUIRE(metadata->blob_count() == 2);
  CHECK(metadata->blob(0).type == "deletion-vector-v1");
  CHECK(metadata->blob(0).offset == 4);
  CHECK(metadata->blob(0).length == 20);
  CHECK(metadata->blob(0).snapshot_id == -1);
  CHECK_FALSE(metadata->blob(0).compression_codec);
  CHECK(metadata->field_count(0) == 2);
  CHECK(metadata->field(0, 1) == 2);
  CHECK(metadata->property(0, "cardinality") == "not-an-integer");
  CHECK(metadata->property(0, "custom-snapshot") == "7");
  CHECK(metadata->property(0, "empty") == "");
  CHECK_FALSE(metadata->property(0, "absent"));
  CHECK_FALSE(metadata->property(1, "cardinality"));
  CHECK(metadata->blob(1).snapshot_id == 12);
  CHECK(metadata->blob(1).sequence_number == 34);
  CHECK(metadata->blob(1).compression_codec == "zstd");
  CHECK(metadata->file_property("created-by") == "test");
  CHECK(metadata->file_property({}) == "empty-key");
  CHECK(metadata->file_property("nul") == std::string_view("a\0b", 3));
  CHECK_THROWS_AS(metadata->blob(2), std::out_of_range);
  CHECK_THROWS_AS(metadata->field(0, 2), std::out_of_range);
  auto charged = ledger.charged_bytes({1, 1});
  REQUIRE(charged > 0);
  permit.release();
  CHECK(provider.seen->allocated_bytes == charged);
  metadata.reset();
  CHECK(provider.seen->allocated_bytes == 0);
  CHECK(ledger.charged_bytes({1, 1}) == 0);
}

TEST_CASE("Neutral Puffin metadata rejects malformed structure and releases its backing",
          "[scan_preparation][puffin_metadata][failure]")
{
  using namespace sirius::scan_manager;
  auto mode = GENERATE("json",
                       "root",
                       "blobs",
                       "duplicate_root",
                       "duplicate_blob",
                       "duplicate_property",
                       "offset",
                       "length",
                       "overlap",
                       "repeat_offset",
                       "integer_overflow",
                       "fields",
                       "codec",
                       "properties",
                       "property_value",
                       "trailing_magic",
                       "leading_magic",
                       "footer_magic",
                       "compressed",
                       "flags",
                       "footer_length");
  INFO(mode);
  std::string m          = mode;
  std::string descriptor = metadata_descriptor();
  if (m == "duplicate_blob") descriptor = metadata_descriptor(",\"offset\":40");
  if (m == "duplicate_property")
    descriptor = metadata_descriptor(R"(,"properties":{"x":"a","x":"b"})");
  if (m == "offset") descriptor = metadata_descriptor({}, 3);
  if (m == "length") descriptor = metadata_descriptor({}, 4, INT64_MAX);
  if (m == "integer_overflow")
    descriptor =
      R"({"type":"x","offset":18446744073709551615,"length":0,"fields":[],"snapshot-id":-1,"sequence-number":-1})";
  if (m == "fields")
    descriptor =
      R"({"type":"x","offset":4,"length":0,"fields":["1"],"snapshot-id":-1,"sequence-number":-1})";
  if (m == "codec") descriptor = metadata_descriptor(",\"compression-codec\":1");
  if (m == "properties") descriptor = metadata_descriptor(",\"properties\":[]");
  if (m == "property_value") descriptor = metadata_descriptor(R"(,"properties":{"cardinality":1})");
  auto footer = "{\"blobs\":[" + descriptor +
                ((m == "overlap" || m == "repeat_offset")
                   ? "," + metadata_descriptor({}, m == "overlap" ? 10 : 4)
                   : "") +
                "]}";
  if (m == "json") footer = "{";
  if (m == "root") footer = "[]";
  if (m == "blobs") footer = "{\"blobs\":{}}";
  if (m == "duplicate_root") footer = "{\"blobs\":[],\"blobs\":[]}";
  auto bytes = metadata_container(footer, m == "compressed" ? 1 : m == "flags" ? 256 : 0);
  if (m == "trailing_magic") bytes.back() = std::byte{0};
  if (m == "leading_magic") bytes.front() = std::byte{0};
  if (m == "footer_magic") bytes[132] = std::byte{0};
  if (m == "footer_length") bytes[bytes.size() - 9] = std::byte{0x7f};
  metadata_only_source source(std::move(bytes));
  test::test_reservation_provider provider;
  provider.grant_bytes = 1024 * 1024;
  preparation_ledger ledger(provider);
  std::array scans{scan_envelope{{{{0, 0, 0}, 512 * 1024}}, {}, true}};
  REQUIRE(ledger.admit(scans, memory_space_id(cucascade::memory::Tier::HOST, 0)).deferred);
  ledger.register_unit({1, 1}, 512 * 1024);
  auto permit = ledger.acquire_permit({1, 1});
  REQUIRE(permit);
  auto allocator = ledger.allocator(permit);
  CHECK_THROWS_AS(read_puffin_metadata(source, allocator), std::runtime_error);
  CHECK(source.data_reads == 0);
  CHECK(ledger.charged_bytes({1, 1}) == 0);
  CHECK(provider.seen->allocated_bytes == 0);
}

TEST_CASE("Neutral Puffin metadata has independent limits and preserves source failures",
          "[scan_preparation][puffin_metadata][failure]")
{
  using namespace sirius::scan_manager;
  test::test_reservation_provider provider;
  provider.grant_bytes = 1024 * 1024;
  preparation_ledger ledger(provider);
  std::array scans{scan_envelope{{{{0, 0, 0}, 512 * 1024}}, {}, true}};
  REQUIRE(ledger.admit(scans, memory_space_id(cucascade::memory::Tier::HOST, 0)).deferred);
  ledger.register_unit({1, 1}, 512 * 1024);
  auto permit = ledger.acquire_permit({1, 1});
  REQUIRE(permit);
  auto allocator = ledger.allocator(permit);
  auto footer    = "{\"blobs\":[" + metadata_descriptor() + "]}";
  vector_puffin_source source(metadata_container(footer));
  CHECK_THROWS_AS(read_puffin_metadata(source, allocator, {8 * 1024 * 1024 + 1, 1}),
                  std::invalid_argument);
  CHECK_THROWS_AS(read_puffin_metadata(source, allocator, {1000, 4097}), std::invalid_argument);
  CHECK_THROWS_AS(read_puffin_metadata(source, allocator, {1, 1}), preparation_resource_error);
  CHECK(provider.seen->allocations == 0);
  CHECK_THROWS_AS(read_puffin_metadata(source, allocator, {1000, 0}), preparation_resource_error);
  CHECK(provider.seen->allocated_bytes == 0);
  provider.seen->fail_allocation = true;
  CHECK_THROWS_AS(read_puffin_metadata(source, allocator), preparation_resource_error);
  provider.seen->fail_allocation = false;
  CHECK(ledger.charged_bytes({1, 1}) == 0);
  identity_changing_puffin_source changed(source.size(), "object");
  CHECK_THROWS_AS(read_puffin_metadata(changed, allocator), sirius::io::object_changed_error);
  cancelling_puffin_source cancelled(source.size());
  CHECK_THROWS_AS(read_puffin_metadata(cancelled, allocator), std::system_error);
  std::string many_keys = "{";
  for (int i = 0; i < 129; ++i) {
    if (i) many_keys += ",";
    many_keys += "\"k" + std::to_string(i) + "\":\"v\"";
  }
  many_keys += "}";
  vector_puffin_source keys(metadata_container(many_keys));
  CHECK_THROWS_AS(read_puffin_metadata(keys, allocator), preparation_resource_error);
  auto many_fields   = metadata_descriptor();
  std::string fields = "[0";
  for (int i = 0; i < 4096; ++i)
    fields += ",0";
  fields += "]";
  many_fields.replace(many_fields.find("[1,2]"), 5, fields);
  vector_puffin_source fields_source(metadata_container("{\"blobs\":[" + many_fields + "]}"));
  CHECK_THROWS_AS(read_puffin_metadata(fields_source, allocator), preparation_resource_error);
  CHECK(provider.seen->allocated_bytes == 0);
  vector_puffin_source empty(metadata_container("{\"blobs\":[]}"));
  auto metadata = read_puffin_metadata(empty, allocator, {1000, 0});
  CHECK(metadata->blob_count() == 0);
  CHECK_FALSE(metadata->file_property("absent"));
}
