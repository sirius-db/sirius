/*
 * Copyright 2025, Sirius Contributors.
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

#include <io/uri_parser.hpp>
#include <log/logging.hpp>
#include <op/scan/puffin_reader.hpp>

// Vendored by DuckDB core and already on this target's include path; duckdb_static bundles its
// objects, so reading the Puffin footer costs no new dependency.
#include "yyjson.hpp"

// CRoaring: the portable-Roaring reader. duckdb-iceberg decodes this same deletion-vector blob with
// the same two calls (`roaring_bitmap_portable_deserialize_size` + `Roaring::readSafe`), so the GPU
// path and DuckDB's own reader agree on a bitmap by construction rather than by our re-derivation.
#include <roaring/roaring.h>
#include <roaring/roaring.hh>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

// The readers below memcpy into the target type, so a big-endian host would decode garbage.
static_assert(__BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__,
              "puffin_reader decodes little-endian fields by direct memcpy");

namespace sirius::op::scan {

namespace {

uint32_t read_u32_le(const uint8_t* p)
{
  uint32_t v;
  std::memcpy(&v, p, 4);
  return v;
}

int64_t read_i64_le(const uint8_t* p)
{
  int64_t v;
  std::memcpy(&v, p, 8);
  return v;
}

int32_t read_i32_le(const uint8_t* p)
{
  int32_t v;
  std::memcpy(&v, p, 4);
  return v;
}

uint32_t read_u32_be(const uint8_t* p)
{
  return (static_cast<uint32_t>(p[0]) << 24) | (static_cast<uint32_t>(p[1]) << 16) |
         (static_cast<uint32_t>(p[2]) << 8) | static_cast<uint32_t>(p[3]);
}

// CRC-32, same polynomial as DuckDB's iceberg extension.
uint32_t crc32_table[256];
std::once_flag crc32_init_flag;

void init_crc32_table()
{
  for (uint32_t i = 0; i < 256; ++i) {
    uint32_t c = i;
    for (int j = 0; j < 8; ++j) {
      c = (c & 1) ? (0xEDB88320u ^ (c >> 1)) : (c >> 1);
    }
    crc32_table[i] = c;
  }
}

uint32_t compute_crc32(const uint8_t* data, size_t length)
{
  std::call_once(crc32_init_flag, init_crc32_table);
  uint32_t crc = 0xFFFFFFFFu;
  for (size_t i = 0; i < length; ++i) {
    crc = crc32_table[(crc ^ data[i]) & 0xFF] ^ (crc >> 8);
  }
  return crc ^ 0xFFFFFFFFu;
}

/// Decodes one 32-bit portable-Roaring bitmap out of @p data and appends its values to @p out,
/// returning the bytes consumed.
///
/// `roaring_bitmap_portable_deserialize_size` reports how many bytes a VALID bitmap occupies within
/// @p data_len and returns 0 otherwise, so the read is bounded by the buffer rather than by numbers
/// the file supplied -- there is no decode ceiling to tune here because the library cannot be made
/// to over-read or over-expand. The cardinality is known before any position is materialized, so
/// @p max_out is enforced BEFORE the allocation rather than inside an expansion loop.
size_t deserialize_roaring32(const uint8_t* data,
                             size_t data_len,
                             std::vector<uint32_t>& out,
                             size_t max_out)
{
  size_t const bitmap_size =
    roaring::api::roaring_bitmap_portable_deserialize_size(reinterpret_cast<const char*>(data),
                                                           data_len);
  if (bitmap_size == 0) {
    throw std::runtime_error(
      "roaring: no valid portable-Roaring bitmap in the remaining " + std::to_string(data_len) +
      " bytes; the deletion vector is truncated or corrupt");
  }

  roaring::Roaring bitmap =
    roaring::Roaring::readSafe(reinterpret_cast<const char*>(data), bitmap_size);

  uint64_t const cardinality = bitmap.cardinality();
  if (cardinality > max_out || out.size() > max_out - cardinality) {
    throw std::runtime_error(
      "roaring: decoded positions exceed the " + std::to_string(max_out) +
      " this deletion vector declares; the blob is corrupt or was crafted to expand");
  }

  size_t const base = out.size();
  out.resize(base + static_cast<size_t>(cardinality));
  bitmap.toUint32Array(out.data() + base);
  return bitmap_size;
}

using YyjsonDoc =
  std::unique_ptr<duckdb_yyjson::yyjson_doc, decltype(&duckdb_yyjson::yyjson_doc_free)>;

/// Puffin properties are a string->string map, so numbers arrive quoted.
std::string property_or_empty(duckdb_yyjson::yyjson_val* properties, char const* key)
{
  if (properties == nullptr) { return {}; }
  auto* val       = duckdb_yyjson::yyjson_obj_get(properties, key);
  auto const* str = duckdb_yyjson::yyjson_get_str(val);
  return str == nullptr ? std::string{} : std::string{str};
}

/// Reads the footer and returns the blob descriptor whose `offset` equals @p content_offset,
/// checking every property the spec fixes for `deletion-vector-v1`.
///
/// The manifest and the footer are written by the same commit but are separate structures, so
/// agreement between them is what proves the entry points at *its own* vector. The blob's magic
/// and CRC only prove it is a well-formed vector — a wrong offset landing on a different valid
/// vector passes both, and passes the cardinality check too whenever the two happen to be the
/// same size.
///
/// Returns the offset of the footer's leading magic, i.e. the first byte past the blob region. The
/// caller bounds the blob read against it: a manifest and footer are free to agree on a size that
/// the FILE cannot hold, and believing them is a value-initialized allocation of whatever they say.
std::streamoff validate_footer_descriptor(std::ifstream& f,
                                std::streamoff file_size,
                                DeletionVectorRef const& ref,
                                char const (&puffin_magic)[4])
{
  // Footer = Magic | Payload | PayloadSize(4, LE) | Flags(4) | Magic
  static constexpr std::streamoff kFooterTail = 12;  // PayloadSize + Flags + trailing Magic
  if (file_size < kFooterTail + 8) {
    throw std::runtime_error("[puffin] File too small to hold a footer: " + ref.puffin_path);
  }

  f.seekg(file_size - kFooterTail);
  uint8_t tail[kFooterTail];
  f.read(reinterpret_cast<char*>(tail), kFooterTail);
  if (!f) { throw std::runtime_error("[puffin] Cannot read footer tail of " + ref.puffin_path); }

  auto const payload_size = static_cast<int32_t>(read_u32_le(tail));
  if (payload_size < 0 || static_cast<std::streamoff>(payload_size) + kFooterTail + 4 > file_size) {
    throw std::runtime_error("[puffin] Footer payload size " + std::to_string(payload_size) +
                             " does not fit in " + ref.puffin_path);
  }

  // Bit 0 of the flags means the payload is LZ4-compressed. Nothing here can decompress it, and
  // guessing would be worse than declining.
  if ((tail[4] & 0x01u) != 0) {
    throw std::runtime_error("[puffin] Footer of " + ref.puffin_path +
                             " is compressed, so its blob descriptors cannot be checked");
  }

  auto const footer_start = file_size - kFooterTail - payload_size - 4;
  f.seekg(footer_start);
  char magic[4];
  f.read(magic, 4);
  if (!f || std::memcmp(magic, puffin_magic, 4) != 0) {
    throw std::runtime_error("[puffin] Missing footer magic in " + ref.puffin_path);
  }

  std::string payload(static_cast<size_t>(payload_size), '\0');
  f.read(payload.data(), payload_size);
  if (!f) { throw std::runtime_error("[puffin] Cannot read footer payload of " + ref.puffin_path); }

  YyjsonDoc doc(duckdb_yyjson::yyjson_read(payload.data(), payload.size(), 0),
                &duckdb_yyjson::yyjson_doc_free);
  if (!doc) {
    throw std::runtime_error("[puffin] Footer of " + ref.puffin_path + " is not valid JSON");
  }

  auto* blobs =
    duckdb_yyjson::yyjson_obj_get(duckdb_yyjson::yyjson_doc_get_root(doc.get()), "blobs");
  if (blobs == nullptr || !duckdb_yyjson::yyjson_is_arr(blobs)) {
    throw std::runtime_error("[puffin] Footer of " + ref.puffin_path + " has no 'blobs' array");
  }

  duckdb_yyjson::yyjson_val* descriptor = nullptr;
  size_t idx                            = 0;
  size_t max                            = 0;
  duckdb_yyjson::yyjson_val* blob       = nullptr;
  yyjson_arr_foreach(blobs, idx, max, blob)
  {
    auto* offset = duckdb_yyjson::yyjson_obj_get(blob, "offset");
    if (offset != nullptr && duckdb_yyjson::yyjson_get_sint(offset) == ref.content_offset) {
      descriptor = blob;
      break;
    }
  }
  if (descriptor == nullptr) {
    throw std::runtime_error("[puffin] No blob at offset " + std::to_string(ref.content_offset) +
                             " in the footer of " + ref.puffin_path +
                             "; the manifest entry points into the middle of the file");
  }

  auto const require = [&ref](char const* what, std::string const& got, std::string const& want) {
    if (got != want) {
      throw std::runtime_error("[puffin] Blob at offset " + std::to_string(ref.content_offset) +
                               " in " + ref.puffin_path + " has " + what + " '" + got +
                               "', but its manifest entry requires '" + want + "'");
    }
  };

  auto const* type =
    duckdb_yyjson::yyjson_get_str(duckdb_yyjson::yyjson_obj_get(descriptor, "type"));
  require("type", type == nullptr ? std::string{} : std::string{type}, "deletion-vector-v1");

  auto const length =
    duckdb_yyjson::yyjson_get_sint(duckdb_yyjson::yyjson_obj_get(descriptor, "length"));
  if (length != ref.content_size_in_bytes) {
    throw std::runtime_error("[puffin] Blob at offset " + std::to_string(ref.content_offset) +
                             " in " + ref.puffin_path + " is " + std::to_string(length) +
                             " bytes, but its manifest entry records " +
                             std::to_string(ref.content_size_in_bytes));
  }

  // "Snapshot ID and sequence number are not known at the time the Puffin file is created", so the
  // spec fixes both at -1. A real value means the file was not written as a deletion vector.
  for (auto const* field : {"snapshot-id", "sequence-number"}) {
    auto* val = duckdb_yyjson::yyjson_obj_get(descriptor, field);
    if (val == nullptr) {
      throw std::runtime_error("[puffin] Blob descriptor in " + ref.puffin_path + " has no '" +
                               field + "'");
    }
    if (duckdb_yyjson::yyjson_get_sint(val) != -1) {
      throw std::runtime_error("[puffin] Blob descriptor in " + ref.puffin_path + " has " + field +
                               "=" + std::to_string(duckdb_yyjson::yyjson_get_sint(val)) +
                               ", but deletion-vector-v1 fixes it at -1");
    }
  }

  // Required for every blob. Puffin constrains it to a list of field ids; a deletion vector puts
  // no meaning in the contents, but a descriptor that spells it as anything other than an array of
  // integers is not a Puffin blob descriptor and should not be read as one.
  auto* fields = duckdb_yyjson::yyjson_obj_get(descriptor, "fields");
  if (fields == nullptr) {
    throw std::runtime_error("[puffin] Blob descriptor in " + ref.puffin_path + " has no 'fields'");
  }
  if (!duckdb_yyjson::yyjson_is_arr(fields)) {
    throw std::runtime_error("[puffin] Blob descriptor in " + ref.puffin_path +
                             " spells 'fields' as something other than a JSON array");
  }
  size_t const n_fields = duckdb_yyjson::yyjson_arr_size(fields);
  for (size_t i = 0; i < n_fields; ++i) {
    if (!duckdb_yyjson::yyjson_is_int(duckdb_yyjson::yyjson_arr_get(fields, i))) {
      throw std::runtime_error("[puffin] Blob descriptor in " + ref.puffin_path +
                               " has a non-integer element at 'fields'[" + std::to_string(i) + "]");
    }
  }

  // An explicit JSON null is how some writers spell "absent", so only a real codec is a problem.
  auto* codec = duckdb_yyjson::yyjson_obj_get(descriptor, "compression-codec");
  if (codec != nullptr && !duckdb_yyjson::yyjson_is_null(codec)) {
    throw std::runtime_error("[puffin] Blob descriptor in " + ref.puffin_path +
                             " sets compression-codec, but deletion-vector-v1 is never compressed");
  }

  auto* properties      = duckdb_yyjson::yyjson_obj_get(descriptor, "properties");
  auto const referenced = property_or_empty(properties, "referenced-data-file");
  if (referenced.empty()) {
    throw std::runtime_error("[puffin] Blob descriptor in " + ref.puffin_path +
                             " has no referenced-data-file property");
  }
  // Compare on the bare path: the manifest and the footer are free to disagree about the URI
  // scheme, and rejecting on `file://` alone would refuse tables that are entirely well formed.
  require("referenced-data-file",
          sirius::io::strip_file_scheme(referenced),
          sirius::io::strip_file_scheme(ref.referenced_data_file));

  auto const cardinality = property_or_empty(properties, "cardinality");
  if (cardinality.empty()) {
    throw std::runtime_error("[puffin] Blob descriptor in " + ref.puffin_path +
                             " has no cardinality property");
  }
  if (cardinality != std::to_string(ref.record_count)) {
    throw std::runtime_error("[puffin] Blob at offset " + std::to_string(ref.content_offset) +
                             " in " + ref.puffin_path + " declares cardinality " + cardinality +
                             ", but its manifest entry records " +
                             std::to_string(ref.record_count));
  }

  return footer_start;
}

}  // anonymous namespace

std::vector<int64_t> read_deletion_vector(DeletionVectorRef const& ref)
{
  auto const& puffin_path          = ref.puffin_path;
  auto const content_offset        = ref.content_offset;
  auto const content_size_in_bytes = ref.content_size_in_bytes;
  auto const record_count          = ref.record_count;

  if (content_offset < 0 || content_size_in_bytes <= 0) {
    throw std::runtime_error(
      "[puffin] Invalid offset/size: offset=" + std::to_string(content_offset) +
      " size=" + std::to_string(content_size_in_bytes));
  }

  // Bound the decode BEFORE opening the file, and on a constant rather than on anything the table
  // wrote. `record_count` is a required manifest field; absent, both cardinality cross-checks
  // below are vacuous and the expansion is unbounded.
  if (record_count < 0) {
    throw std::runtime_error(
      "[puffin] Manifest entry for the deletion vector at offset " +
      std::to_string(content_offset) + " in " + puffin_path +
      " carries no record_count, so neither its footer cardinality nor its decoded position count "
      "can be checked against anything, and nothing bounds how far the blob may expand");
  }
  if (record_count > kMaxDeletionVectorPositions) {
    throw std::runtime_error(
      "[puffin] Deletion vector at offset " + std::to_string(content_offset) + " in " +
      puffin_path + " declares " + std::to_string(record_count) + " deleted positions, above the " +
      std::to_string(kMaxDeletionVectorPositions) + " this reader will materialize while planning");
  }

  // Apache manifests record URIs; this reader bypasses sirius_ioctx, so nothing else strips them.
  auto const local_path = sirius::io::strip_file_scheme(puffin_path);

  std::ifstream f(local_path, std::ios::binary);
  if (!f) {
    throw std::runtime_error("[puffin] Cannot open file: " + local_path +
                             (local_path == puffin_path ? "" : " (from '" + puffin_path + "')"));
  }

  // The blob's own magic and CRC below cannot catch a bare blob written with no container, so a
  // wrong offset would silently yield wrong deletes. Check the framing first.
  static constexpr char kPuffinMagic[4] = {'P', 'F', 'A', '1'};
  char magic[4];
  f.read(magic, 4);
  if (!f || std::memcmp(magic, kPuffinMagic, 4) != 0) {
    throw std::runtime_error("[puffin] Not a Puffin file (bad leading magic): " + puffin_path);
  }
  f.seekg(0, std::ios::end);
  auto const file_size = static_cast<std::streamoff>(f.tellg());
  f.seekg(file_size - 4);
  f.read(magic, 4);
  if (!f || std::memcmp(magic, kPuffinMagic, 4) != 0) {
    throw std::runtime_error("[puffin] Not a Puffin file (bad trailing magic): " + puffin_path);
  }

  auto const footer_start = validate_footer_descriptor(f, file_size, ref, kPuffinMagic);

  // The blob must lie entirely between the leading magic and the footer. Both bounds are compared
  // by SUBTRACTION against a length the file actually has: `content_offset + content_size` is a
  // sum of two manifest-supplied numbers and can wrap, and a descriptor that merely agrees with
  // the manifest proves nothing about the file -- a few-KB Puffin whose manifest and footer both
  // declare a 40 GiB blob would otherwise allocate 40 GiB here and fail on the read afterwards.
  static constexpr std::streamoff kLeadingMagic = 4;
  if (content_offset < kLeadingMagic || content_offset > footer_start ||
      content_size_in_bytes > footer_start - content_offset) {
    throw std::runtime_error(
      "[puffin] Deletion vector at offset " + std::to_string(content_offset) + " size " +
      std::to_string(content_size_in_bytes) + " does not fit between the leading magic and the "
      "footer (which starts at " + std::to_string(footer_start) + ") of " + puffin_path);
  }

  f.seekg(content_offset);
  if (!f) {
    throw std::runtime_error("[puffin] Cannot seek to offset " + std::to_string(content_offset) +
                             " in " + puffin_path);
  }

  std::vector<uint8_t> blob(static_cast<size_t>(content_size_in_bytes));
  f.read(reinterpret_cast<char*>(blob.data()), content_size_in_bytes);
  if (!f) {
    throw std::runtime_error("[puffin] Failed to read " + std::to_string(content_size_in_bytes) +
                             " bytes from " + puffin_path);
  }

  // deletion-vector-v1: [4B BE combined_length][4B magic][roaring_vector][4B BE CRC-32]
  auto const blob_size = blob.size();
  if (blob_size < 12) {
    throw std::runtime_error("[puffin] Blob too small (" + std::to_string(blob_size) +
                             " bytes) to be a deletion-vector-v1");
  }

  const uint8_t* p = blob.data();

  uint32_t combined_length = read_u32_be(p);
  p += 4;

  static constexpr uint8_t DV_MAGIC[4] = {0xD1, 0xD3, 0x39, 0x64};
  if (std::memcmp(p, DV_MAGIC, 4) != 0) {
    throw std::runtime_error("[puffin] Deletion vector magic mismatch in " + puffin_path);
  }

  const uint8_t* checksummed_start = p;
  p += 4;  // skip magic

  // combined_length covers magic + roaring_vector; the CRC follows it.
  //
  // Not redundant with the `roaring_len < 8` check below: that one runs on `checksummed_len - 4`,
  // which WRAPS for lengths 0..3 and sails past it.
  size_t checksummed_len = combined_length;
  if (checksummed_len < 12) {
    throw std::runtime_error("[puffin] combined_length=" + std::to_string(combined_length) +
                             " is too short to hold a deletion vector's magic and bitmap count");
  }
  if (4 + checksummed_len + 4 > blob_size) {
    throw std::runtime_error(
      "[puffin] Blob size mismatch: combined_length=" + std::to_string(combined_length) +
      " blob_size=" + std::to_string(blob_size));
  }

  const uint8_t* crc_ptr = checksummed_start + checksummed_len;
  uint32_t stored_crc    = read_u32_be(crc_ptr);
  uint32_t computed_crc  = compute_crc32(checksummed_start, checksummed_len);
  if (stored_crc != computed_crc) {
    throw std::runtime_error("[puffin] CRC-32 mismatch in " + puffin_path +
                             ": stored=" + std::to_string(stored_crc) +
                             " computed=" + std::to_string(computed_crc));
  }

  // Parse the 64-bit Roaring vector.
  // Layout: [8B LE num_bitmaps] { [4B LE key] [32-bit Roaring bitmap] } × N
  size_t roaring_len = checksummed_len - 4;  // minus magic
  if (roaring_len < 8) { throw std::runtime_error("[puffin] Roaring vector too small"); }

  int64_t num_bitmaps = read_i64_le(p);
  p += 8;
  roaring_len -= 8;

  // Negative would skip the loop and return an empty list, i.e. "no deleted rows". CRC-covered
  // already, so this is defence in depth.
  if (num_bitmaps < 0) {
    throw std::runtime_error("[puffin] Negative bitmap count (" + std::to_string(num_bitmaps) +
                             ") in " + puffin_path);
  }

  // Validated against kMaxDeletionVectorPositions above, so this is a bounded budget and not a
  // number the table chose. The decoded count is separately required to EQUAL it.
  size_t const max_positions = static_cast<size_t>(record_count);

  std::vector<int64_t> positions;

  for (int64_t bm = 0; bm < num_bitmaps; ++bm) {
    if (roaring_len < 4) { throw std::runtime_error("[puffin] Truncated bitmap key"); }
    int32_t key = read_i32_le(p);
    p += 4;
    roaring_len -= 4;

    // A negative key widens to [0x80000000, 0xffffffff] and the shift then sets bit 63, so every
    // position assembled under it is negative. positional_delete_filter binary-searches a sorted
    // list against non-negative row offsets, so those entries match nothing and their deletes are
    // silently dropped -- and the record_count check cannot see it, because the COUNT is right.
    // Puffin positions are non-negative 64-bit, so such a key is out of range by construction.
    if (key < 0) {
      throw std::runtime_error("[puffin] Bitmap key " + std::to_string(key) + " in " + puffin_path +
                               " has bit 31 set, which would encode a row position at or above "
                               "2^63; deletion-vector positions are non-negative");
    }

    int64_t high = static_cast<int64_t>(static_cast<uint32_t>(key)) << 32;

    std::vector<uint32_t> low_positions;
    // Shared across bitmaps, so a blob cannot beat the ceiling by splitting across keys.
    size_t const budget = max_positions - positions.size();
    size_t consumed     = deserialize_roaring32(p, roaring_len, low_positions, budget);
    p += consumed;
    roaring_len -= consumed;

    for (uint32_t low : low_positions) {
      positions.push_back(high | static_cast<int64_t>(low));
    }
  }

  std::sort(positions.begin(), positions.end());

  SIRIUS_LOG_INFO("[puffin] Read deletion vector from '{}': {} deleted position(s).",
                  puffin_path,
                  positions.size());

  return positions;
}

}  // namespace sirius::op::scan
