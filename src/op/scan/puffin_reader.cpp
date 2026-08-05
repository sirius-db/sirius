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

#include <log/logging.hpp>
#include <op/scan/puffin_reader.hpp>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

namespace sirius::op::scan {

namespace {

//===----------------------------------------------------------------------===//
// Endian helpers
//===----------------------------------------------------------------------===//

static uint32_t read_u32_le(const uint8_t* p)
{
  uint32_t v;
  std::memcpy(&v, p, 4);
  return v;
}

static uint16_t read_u16_le(const uint8_t* p)
{
  uint16_t v;
  std::memcpy(&v, p, 2);
  return v;
}

static int64_t read_i64_le(const uint8_t* p)
{
  int64_t v;
  std::memcpy(&v, p, 8);
  return v;
}

static int32_t read_i32_le(const uint8_t* p)
{
  int32_t v;
  std::memcpy(&v, p, 4);
  return v;
}

static uint32_t read_u32_be(const uint8_t* p)
{
  return (static_cast<uint32_t>(p[0]) << 24) | (static_cast<uint32_t>(p[1]) << 16) |
         (static_cast<uint32_t>(p[2]) << 8) | static_cast<uint32_t>(p[3]);
}

static uint64_t read_u64_le(const uint8_t* p)
{
  uint64_t v;
  std::memcpy(&v, p, 8);
  return v;
}

//===----------------------------------------------------------------------===//
// CRC-32 (same polynomial as DuckDB's iceberg extension)
//===----------------------------------------------------------------------===//

static uint32_t crc32_table[256];
static std::once_flag crc32_init_flag;

static void init_crc32_table()
{
  for (uint32_t i = 0; i < 256; ++i) {
    uint32_t c = i;
    for (int j = 0; j < 8; ++j) {
      c = (c & 1) ? (0xEDB88320u ^ (c >> 1)) : (c >> 1);
    }
    crc32_table[i] = c;
  }
}

static uint32_t compute_crc32(const uint8_t* data, size_t length)
{
  std::call_once(crc32_init_flag, init_crc32_table);
  uint32_t crc = 0xFFFFFFFFu;
  for (size_t i = 0; i < length; ++i) {
    crc = crc32_table[(crc ^ data[i]) & 0xFF] ^ (crc >> 8);
  }
  return crc ^ 0xFFFFFFFFu;
}

//===----------------------------------------------------------------------===//
// Roaring portable format deserializer (32-bit)
//
// Reference: https://github.com/RoaringBitmap/RoaringFormatSpec
//
// Produces a list of set uint32 values from a single serialized Roaring bitmap.
// Returns the number of bytes consumed from the input.
//===----------------------------------------------------------------------===//

// Cookie values that identify the serialization format.
static constexpr uint32_t SERIAL_COOKIE_NO_RUNCONTAINER = 12346;
static constexpr uint32_t SERIAL_COOKIE                 = 12347;

static size_t deserialize_roaring32(const uint8_t* data,
                                    size_t data_len,
                                    std::vector<uint32_t>& out)
{
  if (data_len < 4) { throw std::runtime_error("roaring: buffer too small for cookie"); }

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
    if (p + run_bitmap_bytes > p_end) { throw std::runtime_error("roaring: truncated run bitmap"); }
    run_bitmap = p;
    p += run_bitmap_bytes;
  } else if (cookie == SERIAL_COOKIE_NO_RUNCONTAINER) {
    p += 4;
    if (p + 4 > p_end) { throw std::runtime_error("roaring: truncated container count"); }
    num_containers = static_cast<int>(read_u32_le(p));
    p += 4;
  } else {
    throw std::runtime_error("roaring: unknown cookie " + std::to_string(cookie));
  }

  if (num_containers == 0) { return static_cast<size_t>(p - data); }

  // Read key-cardinality pairs: [key(u16), cardinality_minus_1(u16)] × num_containers
  size_t descriptor_bytes = static_cast<size_t>(num_containers) * 4;
  if (p + descriptor_bytes > p_end) { throw std::runtime_error("roaring: truncated descriptors"); }

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
    if (p + offset_bytes > p_end) { throw std::runtime_error("roaring: truncated offset header"); }
    p += offset_bytes;
  }

  // Read container data
  for (int i = 0; i < num_containers; ++i) {
    uint32_t high = static_cast<uint32_t>(containers[i].key) << 16;

    if (containers[i].type == 0) {
      // Array container: cardinality × uint16 sorted values
      size_t nbytes = static_cast<size_t>(containers[i].cardinality) * 2;
      if (p + nbytes > p_end) { throw std::runtime_error("roaring: truncated array container"); }
      for (uint32_t j = 0; j < containers[i].cardinality; ++j) {
        out.push_back(high | read_u16_le(p));
        p += 2;
      }
    } else if (containers[i].type == 1) {
      // Bitmap container: 1024 × uint64 = 8192 bytes
      if (p + 8192 > p_end) { throw std::runtime_error("roaring: truncated bitmap container"); }
      for (int w = 0; w < 1024; ++w) {
        uint64_t word = read_u64_le(p + w * 8);
        while (word != 0) {
          int bit = __builtin_ctzll(word);
          out.push_back(high | static_cast<uint32_t>(w * 64 + bit));
          word &= word - 1;  // clear lowest set bit
        }
      }
      p += 8192;
    } else {
      // Run container: num_runs(u16), then num_runs × (start_u16, length_u16)
      if (p + 2 > p_end) { throw std::runtime_error("roaring: truncated run container header"); }
      uint16_t num_runs = read_u16_le(p);
      p += 2;
      size_t run_bytes = static_cast<size_t>(num_runs) * 4;
      if (p + run_bytes > p_end) { throw std::runtime_error("roaring: truncated run container"); }
      for (uint16_t r = 0; r < num_runs; ++r) {
        uint16_t start  = read_u16_le(p);
        uint16_t length = read_u16_le(p + 2);
        p += 4;
        for (uint32_t v = start; v <= static_cast<uint32_t>(start) + length; ++v) {
          out.push_back(high | v);
        }
      }
    }
  }

  return static_cast<size_t>(p - data);
}

}  // anonymous namespace

//===----------------------------------------------------------------------===//
// Public API
//===----------------------------------------------------------------------===//

std::vector<int64_t> read_deletion_vector(std::string const& puffin_path,
                                          int64_t content_offset,
                                          int64_t content_size_in_bytes)
{
  if (content_offset < 0 || content_size_in_bytes <= 0) {
    throw std::runtime_error(
      "[puffin] Invalid offset/size: offset=" + std::to_string(content_offset) +
      " size=" + std::to_string(content_size_in_bytes));
  }

  std::ifstream f(puffin_path, std::ios::binary);
  if (!f) { throw std::runtime_error("[puffin] Cannot open file: " + puffin_path); }

  // Validate the container before trusting an offset into it. The blob's own magic and CRC
  // (below) are not enough: a bare deletion-vector blob written with no container passes all of
  // them, because none of them look at the file as a whole. Reading a blob at an offset into a
  // file whose framing was never checked turns a wrong offset into wrong deletes, not an error.
  static constexpr char kPuffinMagic[4] = {'P', 'F', 'A', '1'};
  char magic[4];
  f.read(magic, 4);
  if (!f || std::memcmp(magic, kPuffinMagic, 4) != 0) {
    throw std::runtime_error("[puffin] Not a Puffin file (bad leading magic): " + puffin_path);
  }
  f.seekg(-4, std::ios::end);
  f.read(magic, 4);
  if (!f || std::memcmp(magic, kPuffinMagic, 4) != 0) {
    throw std::runtime_error("[puffin] Not a Puffin file (bad trailing magic): " + puffin_path);
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

  // Parse deletion-vector-v1 blob format.
  // Layout: [4B BE combined_length] [4B magic] [roaring_vector] [4B BE CRC-32]
  auto const blob_size = blob.size();
  if (blob_size < 12) {
    throw std::runtime_error("[puffin] Blob too small (" + std::to_string(blob_size) +
                             " bytes) to be a deletion-vector-v1");
  }

  const uint8_t* p = blob.data();

  uint32_t combined_length = read_u32_be(p);
  p += 4;

  // Verify magic: 0xD1D33964
  static constexpr uint8_t DV_MAGIC[4] = {0xD1, 0xD3, 0x39, 0x64};
  if (std::memcmp(p, DV_MAGIC, 4) != 0) {
    throw std::runtime_error("[puffin] Deletion vector magic mismatch in " + puffin_path);
  }

  const uint8_t* checksummed_start = p;
  p += 4;  // skip magic

  // CRC-32 covers magic + roaring_vector (combined_length bytes total).
  // After the checksummed region: 4-byte BE CRC-32.
  size_t checksummed_len = combined_length;
  if (4 + checksummed_len + 4 > blob_size) {
    throw std::runtime_error(
      "[puffin] Blob size mismatch: combined_length=" + std::to_string(combined_length) +
      " blob_size=" + std::to_string(blob_size));
  }

  // Verify CRC-32
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

  std::vector<int64_t> positions;

  for (int64_t bm = 0; bm < num_bitmaps; ++bm) {
    if (roaring_len < 4) { throw std::runtime_error("[puffin] Truncated bitmap key"); }
    int32_t key = read_i32_le(p);
    p += 4;
    roaring_len -= 4;

    int64_t high = static_cast<int64_t>(static_cast<uint32_t>(key)) << 32;

    std::vector<uint32_t> low_positions;
    size_t consumed = deserialize_roaring32(p, roaring_len, low_positions);
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
