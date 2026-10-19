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

#include <io/io_errors.hpp>
#include <io/sirius_datasource.hpp>
#include <log/logging.hpp>

// Vendored by DuckDB core and already on this target's include path; duckdb_static bundles its
// objects, so reading the Puffin footer costs no new dependency.
#include "yyjson.hpp"

// CRoaring: the portable-Roaring reader. The legacy path uses the same two calls as
// duckdb-iceberg (`roaring_bitmap_portable_deserialize_size` + `Roaring::readSafe`).
// The charged path shares the size check and expands containers without allocating a bitmap.
#include <roaring/roaring.h>

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
#include <system_error>
#include <vector>

// The readers below memcpy into the target type, so a big-endian host would decode garbage.
static_assert(__BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__,
              "puffin_reader decodes little-endian fields by direct memcpy");

namespace sirius::op::scan {

namespace {

class local_puffin_source final : public puffin_byte_source {
 public:
  explicit local_puffin_source(std::string_view path)
    : file_(std::string(path), std::ios::binary), path_(path)
  {
    if (file_) {
      file_.seekg(0, std::ios::end);
      size_ = static_cast<std::streamoff>(file_.tellg());
      file_.seekg(0);
    }
  }
  bool seek(std::streamoff offset) override
  {
    file_.clear();
    file_.seekg(offset);
    if (file_.bad()) {
      throw transparent::classified_execution_error(transparent::late_failure_cause::reader_io,
                                                    "[puffin] Cannot seek local file: " + path_);
    }
    return static_cast<bool>(file_);
  }
  bool read(std::span<std::byte> destination) override
  {
    file_.read(reinterpret_cast<char*>(destination.data()),
               static_cast<std::streamsize>(destination.size()));
    if (file_.bad()) {
      throw transparent::classified_execution_error(transparent::late_failure_cause::reader_io,
                                                    "[puffin] Cannot read local file: " + path_);
    }
    return static_cast<bool>(file_);
  }
  [[nodiscard]] std::streamoff size() const override { return size_; }
  explicit operator bool() const noexcept { return static_cast<bool>(file_); }

 private:
  std::ifstream file_;
  std::string path_;
  std::streamoff size_{-1};
};

class datasource_puffin_source final : public puffin_byte_source {
 public:
  explicit datasource_puffin_source(std::unique_ptr<io::sirius_datasource> source)
    : source_(std::move(source))
  {
  }
  bool seek(std::streamoff offset) override
  {
    if (offset < 0 || static_cast<std::uint64_t>(offset) > source_->size()) return false;
    offset_ = offset;
    return true;
  }
  bool read(std::span<std::byte> destination) override
  {
    auto const size = source_->size();
    if (offset_ < 0 || static_cast<std::uint64_t>(offset_) > size ||
        destination.size() > size - static_cast<std::uint64_t>(offset_))
      return false;
    std::size_t read;
    try {
      read = source_->host_read(static_cast<std::size_t>(offset_),
                                destination.size(),
                                reinterpret_cast<std::uint8_t*>(destination.data()));
    } catch (transparent::classified_execution_error const&) {
      throw;
    } catch (duckdb::InterruptException const&) {
      throw;
    } catch (io::object_changed_error const&) {
      // Preserve the typed version failure until the charged scan wrapper,
      // which records it and classifies the terminal outcome.
      throw;
    } catch (std::bad_alloc const&) {
      throw;
    } catch (std::system_error const& error) {
      if (error.code() == std::errc::operation_canceled) throw;
      throw transparent::classified_execution_error(transparent::late_failure_cause::reader_io,
                                                    error.what());
    } catch (std::exception const& error) {
      throw transparent::classified_execution_error(transparent::late_failure_cause::reader_io,
                                                    error.what());
    }
    if (read != destination.size()) return false;
    offset_ += static_cast<std::streamoff>(read);
    return true;
  }
  [[nodiscard]] std::streamoff size() const override
  {
    return static_cast<std::streamoff>(source_->size());
  }

 private:
  std::unique_ptr<io::sirius_datasource> source_;
  std::streamoff offset_ = 0;
};

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

/// Walk one 32-bit portable-Roaring bitmap, validating its containers before
/// materializing values through @p emit. Both eager and charged readers use this
/// walker so they share the same format and corruption checks.
template <typename Emit>
size_t walk_roaring32(const uint8_t* data, size_t len, size_t max_out, Emit&& emit)
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
    if (bytes > static_cast<size_t>(end - p)) fail("container exceeds bitmap bounds");
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

  for (uint32_t i = 1; i < containers; ++i)
    if (u16(headers + i * 4) <= u16(headers + (i - 1) * 4)) fail("keys not strictly increasing");

  auto walk = [&](bool materialize) {
    p            = payload;
    size_t total = 0;
    for (uint32_t i = 0; i < containers; ++i) {
      auto key         = u16(headers + i * 4);
      auto cardinality = uint32_t(u16(headers + i * 4 + 2)) + 1;
      auto append      = [&](uint32_t low) {
        if (total >= max_out) fail("decoded positions exceed the declared budget");
        emit((uint32_t(key) << 16) | low);
        ++total;
      };
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
          if (materialize)
            for (uint32_t v = first; v < end; ++v)
              append(v);
          else
            total += end - first;
        }
      } else if (cardinality <= 4096) {
        auto const* values = take(cardinality * 2);
        for (uint32_t j = 0; j < cardinality; ++j) {
          auto value = u16(values + j * 2);
          if (j && value <= u16(values + (j - 1) * 2))
            fail("array elements not strictly increasing");
          if (materialize) append(value);
        }
        if (!materialize) total += cardinality;
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
        if (!materialize) total += actual;
      }
    }
    return total;
  };

  auto const cardinality = walk(false);
  if (cardinality > max_out)
    throw std::runtime_error(
      "roaring: decoded positions exceed the " + std::to_string(max_out) +
      " this deletion vector declares; the blob is corrupt or was crafted to expand");
  walk(true);
  return consumed;
}

size_t deserialize_roaring32(const uint8_t* data,
                             size_t data_len,
                             std::vector<uint32_t>& out,
                             size_t max_out)
{
  auto const base = out.size();
  if (base > max_out)
    throw std::runtime_error("roaring: decoded positions exceed the declared budget");
  auto const consumed =
    walk_roaring32(data, data_len, max_out - base, [&](uint32_t value) { out.push_back(value); });
  return consumed;
}

size_t decode_roaring_charged(
  const uint8_t* data, size_t len, int64_t high, std::span<int64_t> output, size_t& count)
{
  auto const consumed = walk_roaring32(
    data, len, output.size() - count, [&](uint32_t value) { output[count++] = high | value; });
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

// Pure descriptor parser: the caller owns payload and optional parser arena.
// No file handles or byte transport enter this function.
template <typename Ref>
void parse_footer_descriptor(std::span<char> payload_bytes,
                             Ref const& ref,
                             duckdb_yyjson::yyjson_alc* pool)
{
  auto* payload   = payload_bytes.data();
  auto const size = payload_bytes.size();
  duckdb_yyjson::yyjson_read_err parse_error{};
  YyjsonDoc doc(duckdb_yyjson::yyjson_read_opts(payload, size, 0, pool, &parse_error),
                &duckdb_yyjson::yyjson_doc_free);
  if (!doc) {
    if (pool && parse_error.code == duckdb_yyjson::YYJSON_READ_ERROR_MEMORY_ALLOCATION)
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
    if (offset != nullptr && duckdb_yyjson::yyjson_is_int(offset) &&
        duckdb_yyjson::yyjson_get_sint(offset) == ref.content_offset) {
      descriptor = blob;
      break;
    }
  }
  if (descriptor == nullptr) {
    throw std::runtime_error("[puffin] No blob at offset " + std::to_string(ref.content_offset) +
                             " in the footer of " + ref.puffin_path +
                             "; the manifest entry points into the middle of the file");
  }

  auto const required_sint = [&](char const* field) -> int64_t {
    auto* value = duckdb_yyjson::yyjson_obj_get(descriptor, field);
    if (value == nullptr || !duckdb_yyjson::yyjson_is_int(value)) {
      throw std::runtime_error("[puffin] Blob descriptor in " + ref.puffin_path +
                               " has no integer '" + field + "'");
    }
    return duckdb_yyjson::yyjson_get_sint(value);
  };

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

  auto const length = required_sint("length");
  if (length != ref.content_size_in_bytes) {
    throw std::runtime_error("[puffin] Blob at offset " + std::to_string(ref.content_offset) +
                             " in " + ref.puffin_path + " is " + std::to_string(length) +
                             " bytes, but its manifest entry records " +
                             std::to_string(ref.content_size_in_bytes));
  }

  // "Snapshot ID and sequence number are not known at the time the Puffin file is created", so the
  // spec fixes both at -1. A real value means the file was not written as a deletion vector.
  for (auto const* field : {"snapshot-id", "sequence-number"}) {
    auto const value = required_sint(field);
    if (value != -1) {
      throw std::runtime_error("[puffin] Blob descriptor in " + ref.puffin_path + " has " + field +
                               "=" + std::to_string(value) +
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
}

/// Bounds discovered from a Puffin footer tail and its leading footer magic.
struct puffin_footer_bounds {
  std::streamoff start;
  std::size_t payload_size;
};

bool read_puffin_bytes(puffin_byte_source& source,
                       std::span<std::byte> destination,
                       std::string_view path)
{
  try {
    return source.read(destination);
  } catch (sirius::transparent::classified_execution_error const&) {
    throw;
  } catch (io::object_changed_error const&) {
    throw;
  } catch (std::bad_alloc const&) {
    // Allocation failures are resource failures, not reader I/O failures.
    throw;
  } catch (std::system_error const& error) {
    if (error.code() == std::errc::operation_canceled) throw;
    throw sirius::transparent::classified_execution_error(
      sirius::transparent::late_failure_cause::reader_io,
      "[puffin] Cannot read " + std::string(path) + ": " + error.what());
  } catch (std::exception const& error) {
    throw sirius::transparent::classified_execution_error(
      sirius::transparent::late_failure_cause::reader_io,
      "[puffin] Cannot read " + std::string(path) + ": " + error.what());
  }
}

template <typename Ref>
puffin_footer_bounds read_footer_bounds(puffin_byte_source& f,
                                        std::streamoff file_size,
                                        Ref const& ref,
                                        char const (&puffin_magic)[4])
{
  static constexpr std::streamoff kFooterTail = 12;
  if (file_size < kFooterTail + 8) {
    throw std::runtime_error("[puffin] File too small to hold a footer: " + ref.puffin_path);
  }
  if (!f.seek(file_size - kFooterTail))
    throw sirius::transparent::classified_execution_error(
      sirius::transparent::late_failure_cause::reader_io,
      "[puffin] Cannot seek to footer tail of " + ref.puffin_path);
  uint8_t tail[kFooterTail];
  if (!read_puffin_bytes(
        f, std::as_writable_bytes(std::span{tail}), "footer tail of " + ref.puffin_path))
    throw sirius::transparent::classified_execution_error(
      sirius::transparent::late_failure_cause::reader_io,
      "[puffin] Cannot read footer tail of " + ref.puffin_path);
  auto const payload_size = static_cast<int32_t>(read_u32_le(tail));
  if (payload_size < 0 || static_cast<std::streamoff>(payload_size) + kFooterTail + 4 > file_size)
    throw std::runtime_error("[puffin] Footer payload size " + std::to_string(payload_size) +
                             " does not fit in " + ref.puffin_path);
  if ((tail[4] & 0x01u) != 0)
    throw std::runtime_error("[puffin] Footer of " + ref.puffin_path +
                             " is compressed, so its blob descriptors cannot be checked");
  auto const footer_start = file_size - kFooterTail - payload_size - 4;
  if (!f.seek(footer_start))
    throw sirius::transparent::classified_execution_error(
      sirius::transparent::late_failure_cause::reader_io,
      "[puffin] Cannot seek to footer magic of " + ref.puffin_path);
  char magic[4];
  if (!read_puffin_bytes(
        f, std::as_writable_bytes(std::span{magic}), "footer magic of " + ref.puffin_path))
    throw std::runtime_error("[puffin] Missing footer magic in " + ref.puffin_path);
  if (std::memcmp(magic, puffin_magic, 4) != 0)
    throw std::runtime_error("[puffin] Missing footer magic in " + ref.puffin_path);
  return {footer_start, static_cast<std::size_t>(payload_size)};
}

/// Reads the footer payload and validates the blob descriptor whose `offset`
/// equals @p content_offset, including all deletion-vector-v1 properties.
template <typename Ref>
std::streamoff validate_footer_descriptor(
  puffin_byte_source& f,
  std::streamoff file_size,
  Ref const& ref,
  char const (&puffin_magic)[4],
  sirius::scan_manager::charging_allocator* allocator = nullptr)
{
  auto const bounds       = read_footer_bounds(f, file_size, ref, puffin_magic);
  auto const footer_start = bounds.start;
  auto const payload_size = bounds.payload_size;

  std::string legacy_payload;
  sirius::scan_manager::charged_block footer;
  duckdb_yyjson::yyjson_alc pool;
  auto const size = payload_size;
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
  if (!read_puffin_bytes(f,
                         std::as_writable_bytes(std::span{payload, size}),
                         "footer payload of " + ref.puffin_path)) {
    throw sirius::transparent::classified_execution_error(
      sirius::transparent::late_failure_cause::reader_io,
      "[puffin] Cannot read footer payload of " + ref.puffin_path);
  }

  parse_footer_descriptor(std::span<char>{payload, size}, ref, allocator ? &pool : nullptr);
  return footer_start;
}

}  // anonymous namespace

namespace {
struct decoded_positions {
  std::vector<int64_t> legacy;
  sirius::scan_manager::charged_block backing;
  size_t count = 0;
};
// Pure DV blob parser. Framing/descriptor and resource-envelope checks have
// already completed before the byte source allocates and fills this span.
template <typename Ref>
decoded_positions parse_deletion_vector_blob(std::span<uint8_t const> blob,
                                             Ref const& ref,
                                             sirius::scan_manager::charging_allocator* allocator)
{
  auto const& puffin_path = ref.puffin_path;
  auto const record_count = ref.record_count;
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

template <typename Ref>
std::streamoff validate_puffin_container(puffin_byte_source& source,
                                         Ref const& ref,
                                         sirius::scan_manager::charging_allocator* allocator)
{
  static constexpr char kPuffinMagic[4] = {'P', 'F', 'A', '1'};
  char magic[4];
  if (!source.seek(0) ||
      !read_puffin_bytes(
        source, std::as_writable_bytes(std::span{magic}), "leading magic of " + ref.puffin_path) ||
      std::memcmp(magic, kPuffinMagic, 4) != 0)
    throw std::runtime_error("[puffin] Not a Puffin file (bad leading magic): " + ref.puffin_path);
  auto const file_size = source.size();
  if (file_size < 4 || !source.seek(file_size - 4) ||
      !read_puffin_bytes(
        source, std::as_writable_bytes(std::span{magic}), "trailing magic of " + ref.puffin_path) ||
      std::memcmp(magic, kPuffinMagic, 4) != 0)
    throw std::runtime_error("[puffin] Not a Puffin file (bad trailing magic): " + ref.puffin_path);
  if (allocator && (ref.file_size_in_bytes < 0 || file_size > ref.file_size_in_bytes))
    throw sirius::scan_manager::preparation_resource_error(
      "Puffin container exceeds the lowering-time envelope", true);
  return validate_footer_descriptor(source, file_size, ref, kPuffinMagic, allocator);
}

template <typename Ref>
decoded_positions read_deletion_vector_impl(Ref const& ref,
                                            sirius::scan_manager::charging_allocator* allocator,
                                            puffin_byte_source* supplied_source = nullptr)
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

  std::string_view local_view(puffin_path);
  if (local_view.starts_with("file://")) local_view.remove_prefix(7);
  std::unique_ptr<local_puffin_source> local_source;
  if (supplied_source == nullptr) local_source = std::make_unique<local_puffin_source>(local_view);
  auto* source = supplied_source ? supplied_source : local_source.get();
  if (source == nullptr || (supplied_source == nullptr && !*local_source)) {
    throw sirius::transparent::classified_execution_error(
      sirius::transparent::late_failure_cause::reader_io,
      "[puffin] Cannot open file: " + std::string(local_view) +
        (local_view == puffin_path ? "" : " (from '" + puffin_path + "')"));
  }

  auto const footer_start = validate_puffin_container(*source, ref, allocator);

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

  if (!source->seek(content_offset)) {
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
  if (!source->read(std::as_writable_bytes(blob))) {
    throw sirius::transparent::classified_execution_error(
      sirius::transparent::late_failure_cause::reader_io,
      "[puffin] Failed to read " + std::to_string(content_size_in_bytes) + " bytes from " +
        puffin_path);
  }

  return parse_deletion_vector_blob(std::span<uint8_t const>{blob}, ref, allocator);
}
}  // namespace

std::vector<int64_t> read_deletion_vector(DeletionVectorRef const& ref)
{
  return read_deletion_vector_impl(ref, nullptr).legacy;
}

std::unique_ptr<puffin_byte_source> make_puffin_byte_source(
  std::unique_ptr<io::sirius_datasource> source)
{
  if (!source) return nullptr;
  return std::make_unique<datasource_puffin_source>(std::move(source));
}

std::vector<int64_t> read_deletion_vector(DeletionVectorRef const& ref, puffin_byte_source& source)
{
  return read_deletion_vector_impl(ref, nullptr, &source).legacy;
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
                                       puffin_byte_source* source = nullptr)
{
  try {
    auto result = read_deletion_vector_impl(ref, &allocator, source);
    check_count(ref, result.count);
    return result;
  } catch (sirius::transparent::classified_execution_error const&) {
    throw;
  } catch (io::object_changed_error const&) {
    // Preserve the typed version failure until the charged scan wrapper.
    throw;
  } catch (std::system_error const& error) {
    // Preserve cancellation; other system errors are reader I/O failures.
    if (error.code() == std::errc::operation_canceled) throw;
    throw sirius::transparent::classified_execution_error(
      sirius::transparent::late_failure_cause::reader_io, error.what());
  } catch (std::runtime_error const& error) {
    throw unsupported_physical_input(
      contract, ref.puffin_path, verdict_reason::iceberg_delete_corrupt, error.what());
  }
}
}  // namespace
std::shared_ptr<iceberg_delete_set const> read_deletion_vector_charged(
  DeletionVectorRef const& ref, sirius::scan_manager::charging_allocator& allocator)
{
  auto result     = read_charged_checked(ref, allocator, 0);
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
  scan_contract_id contract)
{
  auto result     = read_charged_checked(ref, allocator, contract);
  auto* positions = reinterpret_cast<int64_t*>(result.backing.data());
  if (result.count) result.count = std::unique(positions, positions + result.count) - positions;
  return std::make_shared<iceberg_delete_set const>(
    canonical_path, std::move(result.backing), result.count, 1, std::move(path_owner));
}

std::shared_ptr<iceberg_delete_set const> read_deletion_vector_charged(
  DeletionVectorRefView const& ref,
  sirius::scan_manager::charging_allocator& allocator,
  std::string_view canonical_path,
  std::shared_ptr<void const> path_owner,
  scan_contract_id contract,
  puffin_byte_source& source)
{
  auto result     = read_charged_checked(ref, allocator, contract, &source);
  auto* positions = reinterpret_cast<int64_t*>(result.backing.data());
  if (result.count) result.count = std::unique(positions, positions + result.count) - positions;
  return std::make_shared<iceberg_delete_set const>(
    canonical_path, std::move(result.backing), result.count, 1, std::move(path_owner));
}

}  // namespace sirius::op::scan
