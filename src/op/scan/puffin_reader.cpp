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

#include "op/scan/puffin_reader.hpp"

#include "op/scan/iceberg_delete_set.hpp"

#include <io/uri_parser.hpp>
#include <log/logging.hpp>

// Vendored by DuckDB core and already on this target's include path; duckdb_static bundles its
// objects, so reading the Puffin footer costs no new dependency.
#include "yyjson.hpp"

// CRoaring: the portable-Roaring reader. The legacy path uses the same two calls as
// duckdb-iceberg (`roaring_bitmap_portable_deserialize_size` + `Roaring::readSafe`).
// The charged path shares the size check and expands containers without allocating a bitmap.
#include <roaring/roaring.h>
#include <roaring/roaring.hh>

#include <algorithm>
#include <bit>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <memory>
#include <mutex>
#include <span>
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
  size_t const bitmap_size = roaring::api::roaring_bitmap_portable_deserialize_size(
    reinterpret_cast<const char*>(data), data_len);
  if (bitmap_size == 0) {
    throw std::runtime_error("roaring: no valid portable-Roaring bitmap in the remaining " +
                             std::to_string(data_len) +
                             " bytes; the deletion vector is truncated or corrupt");
  }

  roaring::Roaring bitmap =
    roaring::Roaring::readSafe(reinterpret_cast<const char*>(data), bitmap_size);

  // readSafe bounds the READ; it does not check the bitmap's internal invariants. A bitset
  // container carries its cardinality as a stored number and `bitset_container_read` copies it
  // without recomputing the popcount, so a declared cardinality that disagrees with the bits is
  // accepted here and only discovered by what reads it. Both directions are live:
  // over-declaring leaves a zero-filled tail that materializes as delete position 0 and removes a
  // row the table keeps, and under-declaring sizes the destination below what `toUint32Array`
  // emits -- and that call takes no output-capacity argument, so it writes past the end. The
  // record_count cross-check runs after both.
  //
  // The Puffin CRC does not establish trust: the file supplies it alongside the bytes it covers.
  const char* validation_failure = nullptr;
  if (!roaring::api::roaring_bitmap_internal_validate(&bitmap.roaring, &validation_failure)) {
    throw std::runtime_error(
      "roaring: inconsistent bitmap in the deletion vector: " +
      std::string(validation_failure == nullptr ? "unknown reason" : validation_failure));
  }

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

// The deferred reader walks portable containers in place. CRoaring's size check is
// allocation-free; values go straight into the charged int64 output, without a bitmap
// object or uint32 staging vector. All container lengths/cardinalities are revalidated.
size_t decode_roaring_charged(
  uint8_t const* data, size_t len, int64_t high, std::span<int64_t> output, size_t& count)
{
  auto const consumed = roaring::api::roaring_bitmap_portable_deserialize_size(
    reinterpret_cast<char const*>(data), len);
  if (!consumed)
    throw std::runtime_error("roaring: no valid portable-Roaring bitmap in the remaining " +
                             std::to_string(len) +
                             " bytes; the deletion vector is truncated or corrupt");
  auto fail = [](char const* reason) {
    throw std::runtime_error(std::string("roaring: inconsistent bitmap in the deletion vector: ") +
                             reason);
  };
  auto u16 = [](uint8_t const* p) {
    uint16_t v;
    std::memcpy(&v, p, 2);
    return v;
  };
  auto const* end = data + consumed;
  auto const* p   = data;
  auto take       = [&](size_t bytes) {
    if (bytes > static_cast<size_t>(end - p)) fail("unknown reason");
    auto result = p;
    p += bytes;
    return result;
  };
  auto cookie          = read_u32_le(take(4));
  bool runs            = (cookie & 0xffff) == 12347;
  uint32_t containers  = runs ? (cookie >> 16) + 1 : read_u32_le(take(4));
  auto const* run_bits = runs ? take((containers + 7) / 8) : nullptr;
  auto const* headers  = take(static_cast<size_t>(containers) * 4);
  if (!runs || containers >= 4) take(static_cast<size_t>(containers) * 4);
  auto const* payload = p;

  // Match CRoaring's validation order before checking the output budget: keys, all
  // containers, then bitmap cardinality. Keep validation allocation-free.
  for (uint32_t i = 1; i < containers; ++i)
    if (u16(headers + i * 4) <= u16(headers + (i - 1) * 4)) fail("keys not strictly increasing");

  auto walk = [&](bool materialize) {
    p            = payload;
    size_t total = 0;
    for (uint32_t i = 0; i < containers; ++i) {
      auto key         = u16(headers + i * 4);
      auto cardinality = uint32_t(u16(headers + i * 4 + 2)) + 1;
      auto append      = [&](uint32_t low) { output[count++] = high | (int64_t(key) << 16) | low; };
      if (run_bits && (run_bits[i / 8] & (1u << (i % 8)))) {
        auto n = u16(take(2));
        if (!n) fail("zero run count");
        uint32_t previous_end = 0;
        for (uint32_t j = 0; j < n; ++j) {
          auto const* run = take(4);
          uint32_t first = u16(run), end = first + u16(run + 2) + 1;
          if (end > 65536) fail("run start + length too large");
          if (first < previous_end) fail("run start less than last end");
          if (first == previous_end && previous_end)
            fail("run start equal to last end, should have combined");
          previous_end = end;
          total += end - first;
          if (materialize)
            for (uint32_t v = first; v < end; ++v)
              append(v);
        }
        // CRoaring derives run cardinality from the runs, not the stored header.
      } else if (cardinality <= 4096) {
        auto const* values = take(cardinality * 2);
        for (uint32_t j = 0; j < cardinality; ++j) {
          auto value = u16(values + j * 2);
          if (j && value <= u16(values + (j - 1) * 2))
            fail("array elements not strictly increasing");
          if (materialize) append(value);
        }
        total += cardinality;
      } else {
        auto const* words = take(8192);
        uint32_t actual   = 0;
        for (uint32_t j = 0; j < 1024; ++j) {
          uint64_t word;
          std::memcpy(&word, words + j * 8, 8);
          actual += std::popcount(word);
          if (materialize)
            while (word) {
              append(j * 64 + std::countr_zero(word));
              word &= word - 1;
            }
        }
        if (actual != cardinality) fail("cardinality is incorrect");
        total += actual;
      }
    }
    return total;
  };
  auto const cardinality = walk(false);
  auto const remaining   = output.size() - count;
  if (cardinality > remaining)
    throw std::runtime_error(
      "roaring: decoded positions exceed the " + std::to_string(remaining) +
      " this deletion vector declares; the blob is corrupt or was crafted to expand");
  walk(true);
  return consumed;
}

using YyjsonDoc =
  std::unique_ptr<duckdb_yyjson::yyjson_doc, decltype(&duckdb_yyjson::yyjson_doc_free)>;

/// Puffin properties are a string->string map, so numbers arrive quoted.
std::string_view property_or_empty(duckdb_yyjson::yyjson_val* properties, char const* key)
{
  if (properties == nullptr) { return {}; }
  auto* val       = duckdb_yyjson::yyjson_obj_get(properties, key);
  auto const* str = duckdb_yyjson::yyjson_get_str(val);
  return str == nullptr ? std::string_view{} : std::string_view{str};
}

void read_puffin_bytes(std::ifstream& file,
                       char* destination,
                       std::streamsize bytes,
                       puffin_read_statistics* stats)
{
  if (stats) {
    ++stats->requests;
    stats->bytes_requested += bytes;
  }
  file.read(destination, bytes);
  if (stats) {
    stats->bytes_returned += file.gcount();
    stats->failures += !file;
  }
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
template <typename Ref>
std::streamoff validate_footer_descriptor(std::ifstream& f,
                                          std::streamoff file_size,
                                          Ref const& ref,
                                          char const (&puffin_magic)[4],
                                          sirius::scan_manager::charging_allocator* allocator,
                                          puffin_read_statistics* stats)
{
  // Footer = Magic | Payload | PayloadSize(4, LE) | Flags(4) | Magic
  static constexpr std::streamoff kFooterTail = 12;  // PayloadSize + Flags + trailing Magic
  if (file_size < kFooterTail + 8) {
    throw std::runtime_error("[puffin] File too small to hold a footer: " + ref.puffin_path);
  }

  f.seekg(file_size - kFooterTail);
  uint8_t tail[kFooterTail];
  read_puffin_bytes(f, reinterpret_cast<char*>(tail), kFooterTail, stats);
  if (!f) {
    throw sirius::transparent::classified_execution_error(
      sirius::transparent::late_failure_cause::reader_io,
      "[puffin] Cannot read footer tail of " + ref.puffin_path);
  }

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
  read_puffin_bytes(f, magic, 4, stats);
  if (!f || std::memcmp(magic, puffin_magic, 4) != 0) {
    throw std::runtime_error("[puffin] Missing footer magic in " + ref.puffin_path);
  }

  std::string legacy_payload;
  sirius::scan_manager::charged_block footer;
  duckdb_yyjson::yyjson_alc pool;
  auto const size = static_cast<size_t>(payload_size);
  char* payload;
  if (allocator) {
    // One allocation includes the JSON text and yyjson's documented maximum parser usage.
    auto const text_bytes   = (size + 15) / 16 * 16;
    auto const parser_bytes = duckdb_yyjson::yyjson_read_max_memory_usage(size, 0);
    if (!parser_bytes)
      throw sirius::scan_manager::preparation_resource_error("Puffin JSON envelope overflow", true);
    footer  = allocator->allocate(text_bytes + parser_bytes);
    payload = reinterpret_cast<char*>(footer.data());
    if (!duckdb_yyjson::yyjson_alc_pool_init(&pool, payload + text_bytes, parser_bytes))
      throw sirius::scan_manager::preparation_resource_error(
        "Puffin JSON pool initialization failed", false);
  } else {
    legacy_payload.resize(size);
    payload = legacy_payload.data();
  }
  read_puffin_bytes(f, payload, payload_size, stats);
  if (!f) {
    throw sirius::transparent::classified_execution_error(
      sirius::transparent::late_failure_cause::reader_io,
      "[puffin] Cannot read footer payload of " + ref.puffin_path);
  }

  duckdb_yyjson::yyjson_read_err parse_error{};
  YyjsonDoc doc(
    duckdb_yyjson::yyjson_read_opts(payload, size, 0, allocator ? &pool : nullptr, &parse_error),
    &duckdb_yyjson::yyjson_doc_free);
  if (!doc) {
    if (allocator && parse_error.code == duckdb_yyjson::YYJSON_READ_ERROR_MEMORY_ALLOCATION)
      throw sirius::scan_manager::preparation_resource_error("Puffin JSON pool exhausted", true);
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

  auto const require = [&ref](char const* what, std::string_view got, std::string_view want) {
    if (got != want) {
      throw std::runtime_error("[puffin] Blob at offset " + std::to_string(ref.content_offset) +
                               " in " + ref.puffin_path + " has " + what + " '" + std::string(got) +
                               "', but its manifest entry requires '" + std::string(want) + "'");
    }
  };

  auto const* type =
    duckdb_yyjson::yyjson_get_str(duckdb_yyjson::yyjson_obj_get(descriptor, "type"));
  require(
    "type", type == nullptr ? std::string_view{} : std::string_view{type}, "deletion-vector-v1");

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
          referenced.starts_with("file://") ? referenced.substr(7) : referenced,
          std::string_view(ref.referenced_data_file).starts_with("file://")
            ? std::string_view(ref.referenced_data_file).substr(7)
            : std::string_view(ref.referenced_data_file));

  auto const cardinality = property_or_empty(properties, "cardinality");
  if (cardinality.empty()) {
    throw std::runtime_error("[puffin] Blob descriptor in " + ref.puffin_path +
                             " has no cardinality property");
  }
  if (cardinality != std::to_string(ref.record_count)) {
    throw std::runtime_error("[puffin] Blob at offset " + std::to_string(ref.content_offset) +
                             " in " + ref.puffin_path + " declares cardinality " +
                             std::string(cardinality) + ", but its manifest entry records " +
                             std::to_string(ref.record_count));
  }

  return footer_start;
}

}  // anonymous namespace

namespace {
struct decoded_positions {
  std::vector<int64_t> legacy;
  sirius::scan_manager::charged_block backing;
  size_t count = 0;
};
template <typename Ref>
decoded_positions read_deletion_vector_impl(Ref const& ref,
                                            sirius::scan_manager::charging_allocator* allocator,
                                            physical_check_counters const* counters)
{
  struct read_report {
    physical_check_counters const* counters;
    std::string const& file;
    bool charged;
    puffin_read_statistics statistics;
    ~read_report()
    {
      if (!counters) return;
      try {
        counters->puffin_reads_for_testing(file, charged, statistics);
      } catch (...) {
        // Test diagnostics must not replace a reader error during stack unwinding.
      }
    }
  } report{
    counters && counters->track_units && counters->puffin_reads_for_testing ? counters : nullptr,
    ref.puffin_path,
    allocator != nullptr,
    {}};
  auto* stats                      = report.counters ? &report.statistics : nullptr;
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

  // Apache manifests record URIs; this reader bypasses ioctx, so nothing else strips them.
  std::string_view local_view(puffin_path);
  if (local_view.starts_with("file://")) local_view.remove_prefix(7);
  if (stats) ++stats->opens;
  std::ifstream f(local_view.data(), std::ios::binary);
  if (!f) {
    if (stats) ++stats->failures;
    throw sirius::transparent::classified_execution_error(
      sirius::transparent::late_failure_cause::reader_io,
      "[puffin] Cannot open file: " + std::string(local_view) +
        (local_view == puffin_path ? "" : " (from '" + puffin_path + "')"));
  }

  // The blob's own magic and CRC below cannot catch a bare blob written with no container, so a
  // wrong offset would silently yield wrong deletes. Check the framing first.
  static constexpr char kPuffinMagic[4] = {'P', 'F', 'A', '1'};
  char magic[4];
  read_puffin_bytes(f, magic, 4, stats);
  if (!f || std::memcmp(magic, kPuffinMagic, 4) != 0) {
    throw std::runtime_error("[puffin] Not a Puffin file (bad leading magic): " + puffin_path);
  }
  f.seekg(0, std::ios::end);
  auto const file_size = static_cast<std::streamoff>(f.tellg());
  f.seekg(file_size - 4);
  read_puffin_bytes(f, magic, 4, stats);
  if (!f || std::memcmp(magic, kPuffinMagic, 4) != 0) {
    throw std::runtime_error("[puffin] Not a Puffin file (bad trailing magic): " + puffin_path);
  }

  if (allocator && (ref.file_size_in_bytes < 0 || file_size > ref.file_size_in_bytes))
    throw sirius::scan_manager::preparation_resource_error(
      "Puffin container exceeds the lowering-time envelope", true);
  auto const footer_start =
    validate_footer_descriptor(f, file_size, ref, kPuffinMagic, allocator, stats);

  // The blob must lie entirely between the leading magic and the footer. Both bounds are compared
  // by SUBTRACTION against a length the file actually has: `content_offset + content_size` is a
  // sum of two manifest-supplied numbers and can wrap, and a descriptor that merely agrees with
  // the manifest proves nothing about the file -- a few-KB Puffin whose manifest and footer both
  // declare a 40 GiB blob would otherwise allocate 40 GiB here and fail on the read afterwards.
  static constexpr std::streamoff kLeadingMagic = 4;
  if (content_offset < kLeadingMagic || content_offset > footer_start ||
      content_size_in_bytes > footer_start - content_offset) {
    throw std::runtime_error("[puffin] Deletion vector at offset " +
                             std::to_string(content_offset) + " size " +
                             std::to_string(content_size_in_bytes) +
                             " does not fit between the leading magic and the "
                             "footer (which starts at " +
                             std::to_string(footer_start) + ") of " + puffin_path);
  }

  f.seekg(content_offset);
  if (!f) {
    throw sirius::transparent::classified_execution_error(
      sirius::transparent::late_failure_cause::reader_io,
      "[puffin] Cannot seek to offset " + std::to_string(content_offset) + " in " + puffin_path);
  }

  std::vector<uint8_t> legacy_blob;
  sirius::scan_manager::charged_block charged_blob;
  std::span<uint8_t> blob;
  if (allocator) {
    charged_blob = allocator->allocate(content_size_in_bytes);
    blob         = {reinterpret_cast<uint8_t*>(charged_blob.data()),
                    static_cast<size_t>(content_size_in_bytes)};
  } else {
    legacy_blob.resize(content_size_in_bytes);
    blob = legacy_blob;
  }
  read_puffin_bytes(f, reinterpret_cast<char*>(blob.data()), content_size_in_bytes, stats);
  if (!f) {
    throw sirius::transparent::classified_execution_error(
      sirius::transparent::late_failure_cause::reader_io,
      "[puffin] Failed to read " + std::to_string(content_size_in_bytes) + " bytes from " +
        puffin_path);
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

  decoded_positions result;
  auto& positions = result.legacy;
  if (allocator) result.backing = allocator->allocate_retained(max_positions * sizeof(int64_t));

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

    size_t consumed;
    if (allocator) {
      auto output =
        std::span<int64_t>(reinterpret_cast<int64_t*>(result.backing.data()), max_positions);
      consumed = decode_roaring_charged(p, roaring_len, high, output, result.count);
    } else {
      std::vector<uint32_t> low_positions;
      // Shared across bitmaps, so a blob cannot beat the ceiling by splitting across keys.
      size_t const budget = max_positions - positions.size();
      consumed            = deserialize_roaring32(p, roaring_len, low_positions, budget);
      for (uint32_t low : low_positions)
        positions.push_back(high | static_cast<int64_t>(low));
    }
    p += consumed;
    roaring_len -= consumed;
  }

  if (allocator) {
    auto* output = reinterpret_cast<int64_t*>(result.backing.data());
    if (result.count) std::sort(output, output + result.count);
  } else {
    std::sort(positions.begin(), positions.end());
    result.count = positions.size();
  }

  SIRIUS_LOG_INFO(
    "[puffin] Read deletion vector from '{}': {} deleted position(s).", puffin_path, result.count);

  return result;
}
}  // namespace

std::vector<int64_t> read_deletion_vector(DeletionVectorRef const& ref,
                                          physical_check_counters const* counters)
{
  return read_deletion_vector_impl(ref, nullptr, counters).legacy;
}

namespace {
template <typename Ref>
void check_count(Ref const& ref, size_t count)
{
  if (count != static_cast<size_t>(ref.record_count))
    throw std::runtime_error("[iceberg] Deletion vector for data file '" +
                             ref.referenced_data_file + "' in '" + ref.puffin_path + "' decoded " +
                             std::to_string(count) + " positions, but its manifest entry records " +
                             std::to_string(ref.record_count) + "; the blob at offset " +
                             std::to_string(ref.content_offset) +
                             " is not the one this entry describes");
}
template <typename Ref>
decoded_positions read_charged_checked(Ref const& ref,
                                       sirius::scan_manager::charging_allocator& allocator,
                                       scan_contract_id contract,
                                       physical_check_counters const* counters)
{
  try {
    auto result = read_deletion_vector_impl(ref, &allocator, counters);
    check_count(ref, result.count);
    return result;
  } catch (sirius::transparent::classified_execution_error const&) {
    throw;
  } catch (std::runtime_error const& error) {
    throw unsupported_physical_input(
      contract, ref.puffin_path, verdict_reason::iceberg_delete_corrupt, error.what());
  }
}
}  // namespace
std::shared_ptr<iceberg_delete_set const> read_deletion_vector_charged(
  DeletionVectorRef const& ref,
  sirius::scan_manager::charging_allocator& allocator,
  physical_check_counters const* counters)
{
  auto result     = read_charged_checked(ref, allocator, 0, counters);
  auto* positions = reinterpret_cast<int64_t*>(result.backing.data());
  // Repeated high keys are permitted by the legacy reader. Deduplicate only after
  // checking the encoded count, so the immutable result has set semantics.
  if (result.count) result.count = std::unique(positions, positions + result.count) - positions;
  return std::make_shared<iceberg_delete_set const>(
    ref.referenced_data_file, std::move(result.backing), result.count, 1);
}

std::shared_ptr<iceberg_delete_set const> read_deletion_vector_charged(
  DeletionVectorRefView const& ref,
  sirius::scan_manager::charging_allocator& allocator,
  std::string_view canonical_path,
  std::shared_ptr<void const> path_owner,
  scan_contract_id contract,
  physical_check_counters const* counters)
{
  auto result     = read_charged_checked(ref, allocator, contract, counters);
  auto* positions = reinterpret_cast<int64_t*>(result.backing.data());
  if (result.count) result.count = std::unique(positions, positions + result.count) - positions;
  return std::make_shared<iceberg_delete_set const>(
    canonical_path, std::move(result.backing), result.count, 1, std::move(path_owner));
}

}  // namespace sirius::op::scan
