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

uint16_t read_u16_le(const uint8_t* p)
{
  uint16_t v;
  std::memcpy(&v, p, 2);
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

uint64_t read_u64_le(const uint8_t* p)
{
  uint64_t v;
  std::memcpy(&v, p, 8);
  return v;
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

// Roaring portable format, 32-bit: https://github.com/RoaringBitmap/RoaringFormatSpec
// Appends the set values to @p out and returns the bytes consumed.
constexpr uint32_t SERIAL_COOKIE_NO_RUNCONTAINER = 12346;
constexpr uint32_t SERIAL_COOKIE                 = 12347;

/// Checks read as `remaining(...) < need` rather than `p + need > p_end` because `need` comes
/// from the file, and forming a pointer past the end is UB even if only compared.
size_t remaining(const uint8_t* p, const uint8_t* p_end) { return static_cast<size_t>(p_end - p); }

/// Decoded output is unbounded relative to input size: a 4-byte run entry expands to 65,536
/// positions and there can be 65,536 containers, so a sub-megabyte blob can demand billions of
/// them. @p max_out is enforced INSIDE the expansion loops -- checking afterwards is checking
/// after the allocation that already killed the process. It cannot be derived from @p data_len,
/// because a legitimate run container really is that dense.
size_t deserialize_roaring32(const uint8_t* data,
                             size_t data_len,
                             std::vector<uint32_t>& out,
                             size_t max_out)
{
  if (data_len < 4) { throw std::runtime_error("roaring: buffer too small for cookie"); }

  auto const admit = [&out, max_out](size_t n) {
    if (n > max_out || out.size() > max_out - n) {
      throw std::runtime_error(
        "roaring: decoded positions exceed the " + std::to_string(max_out) +
        " this deletion vector declares; the blob is corrupt or was crafted to expand");
    }
  };

  const uint8_t* p     = data;
  const uint8_t* p_end = data + data_len;

  uint32_t cookie = read_u32_le(p);

  int num_containers        = 0;
  bool has_run_containers   = false;
  size_t run_bitmap_bytes   = 0;
  const uint8_t* run_bitmap = nullptr;

  if ((cookie & 0xFFFF) == SERIAL_COOKIE) {
    // Run-optimized format: lower 16 bits = cookie, upper 16 bits = (num_containers - 1)
    num_containers     = static_cast<int>((cookie >> 16) + 1);
    has_run_containers = true;
    p += 4;

    // Run bitmap: ceil(num_containers / 8) bytes
    run_bitmap_bytes = static_cast<size_t>((num_containers + 7) / 8);
    if (remaining(p, p_end) < run_bitmap_bytes) {
      throw std::runtime_error("roaring: truncated run bitmap");
    }
    run_bitmap = p;
    p += run_bitmap_bytes;
  } else if (cookie == SERIAL_COOKIE_NO_RUNCONTAINER) {
    p += 4;
    if (remaining(p, p_end) < 4u) {
      throw std::runtime_error("roaring: truncated container count");
    }
    num_containers = static_cast<int>(read_u32_le(p));
    p += 4;
  } else {
    throw std::runtime_error("roaring: unknown cookie " + std::to_string(cookie));
  }

  if (num_containers == 0) { return static_cast<size_t>(p - data); }

  // Read key-cardinality pairs: [key(u16), cardinality_minus_1(u16)] × num_containers
  size_t descriptor_bytes = static_cast<size_t>(num_containers) * 4;
  if (remaining(p, p_end) < descriptor_bytes) {
    throw std::runtime_error("roaring: truncated descriptors");
  }

  struct ContainerDesc {
    uint16_t key;
    uint32_t cardinality;
    int type;  // 0=array, 1=bitmap, 2=run
  };
  std::vector<ContainerDesc> containers(static_cast<size_t>(num_containers));

  for (int i = 0; i < num_containers; ++i) {
    containers[i].key         = read_u16_le(p);
    containers[i].cardinality = static_cast<uint32_t>(read_u16_le(p + 2)) + 1;
    p += 4;
  }

  // Determine container types
  for (int i = 0; i < num_containers; ++i) {
    if (has_run_containers && (run_bitmap[i / 8] & (1u << (i % 8)))) {
      containers[i].type = 2;  // run
    } else if (containers[i].cardinality <= 4096) {
      containers[i].type = 0;  // array
    } else {
      containers[i].type = 1;  // bitmap
    }
  }

  // Offset header (num_containers × 4 bytes):
  // - SERIAL_COOKIE_NO_RUNCONTAINER: ALWAYS present (per Roaring spec §2)
  // - SERIAL_COOKIE (run-optimized): present only when num_containers >= 4
  // We read containers sequentially so just skip the offset bytes.
  bool has_offsets =
    (cookie == SERIAL_COOKIE_NO_RUNCONTAINER) || (has_run_containers && num_containers >= 4);
  if (has_offsets) {
    size_t offset_bytes = static_cast<size_t>(num_containers) * 4;
    if (remaining(p, p_end) < offset_bytes) {
      throw std::runtime_error("roaring: truncated offset header");
    }
    p += offset_bytes;
  }

  // Read container data
  for (int i = 0; i < num_containers; ++i) {
    uint32_t high = static_cast<uint32_t>(containers[i].key) << 16;

    if (containers[i].type == 0) {
      // Array container: cardinality × uint16 sorted values
      size_t nbytes = static_cast<size_t>(containers[i].cardinality) * 2;
      if (remaining(p, p_end) < nbytes) {
        throw std::runtime_error("roaring: truncated array container");
      }
      admit(containers[i].cardinality);
      for (uint32_t j = 0; j < containers[i].cardinality; ++j) {
        out.push_back(high | read_u16_le(p));
        p += 2;
      }
    } else if (containers[i].type == 1) {
      // Bitmap container: 1024 × uint64 = 8192 bytes
      if (remaining(p, p_end) < 8192u) {
        throw std::runtime_error("roaring: truncated bitmap container");
      }
      for (int w = 0; w < 1024; ++w) {
        uint64_t word = read_u64_le(p + w * 8);
        admit(static_cast<size_t>(__builtin_popcountll(word)));
        while (word != 0) {
          int bit = __builtin_ctzll(word);
          out.push_back(high | static_cast<uint32_t>(w * 64 + bit));
          word &= word - 1;  // clear lowest set bit
        }
      }
      p += 8192;
    } else {
      // Run container: num_runs(u16), then num_runs × (start_u16, length_u16)
      if (remaining(p, p_end) < 2u) {
        throw std::runtime_error("roaring: truncated run container header");
      }
      uint16_t num_runs = read_u16_le(p);
      p += 2;
      size_t run_bytes = static_cast<size_t>(num_runs) * 4;
      if (remaining(p, p_end) < run_bytes) {
        throw std::runtime_error("roaring: truncated run container");
      }
      for (uint16_t r = 0; r < num_runs; ++r) {
        uint16_t start  = read_u16_le(p);
        uint16_t length = read_u16_le(p + 2);
        p += 4;
        admit(static_cast<size_t>(length) + 1);
        for (uint32_t v = start; v <= static_cast<uint32_t>(start) + length; ++v) {
          out.push_back(high | v);
        }
      }
    }
  }

  return static_cast<size_t>(p - data);
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
void validate_footer_descriptor(std::ifstream& f,
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

  f.seekg(file_size - kFooterTail - payload_size - 4);
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

  // Required for every blob, but its contents are unconstrained for a deletion vector.
  if (duckdb_yyjson::yyjson_obj_get(descriptor, "fields") == nullptr) {
    throw std::runtime_error("[puffin] Blob descriptor in " + ref.puffin_path + " has no 'fields'");
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
  if (ref.record_count >= 0 && cardinality != std::to_string(ref.record_count)) {
    throw std::runtime_error("[puffin] Blob at offset " + std::to_string(ref.content_offset) +
                             " in " + ref.puffin_path + " declares cardinality " + cardinality +
                             ", but its manifest entry records " +
                             std::to_string(ref.record_count));
  }
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

  validate_footer_descriptor(f, file_size, ref, kPuffinMagic);

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

  // The decoded count is already required to equal record_count, so anything beyond it is corrupt
  // by definition. Absent that, an absolute cap on what may be materialized while planning.
  static constexpr size_t kMaxPositionsWithoutRecordCount = 32u * 1024u * 1024u;
  size_t const max_positions =
    record_count >= 0 ? static_cast<size_t>(record_count) : kMaxPositionsWithoutRecordCount;

  std::vector<int64_t> positions;

  for (int64_t bm = 0; bm < num_bitmaps; ++bm) {
    if (roaring_len < 4) { throw std::runtime_error("[puffin] Truncated bitmap key"); }
    int32_t key = read_i32_le(p);
    p += 4;
    roaring_len -= 4;

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
