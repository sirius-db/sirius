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

/**
 * Minimal Avro Object Container File reader for Iceberg manifest-list and
 * manifest files.  Supports "null" (uncompressed) and "deflate" codecs.
 *
 * Avro binary encoding reference: https://avro.apache.org/docs/current/spec.html
 *
 * The reader is intentionally narrow: it only handles the types that appear in
 * the Iceberg manifest schemas (null, boolean, int, long, bytes, string, union,
 * array, map, record).  Any other Avro feature (enum, fixed, …) will throw.
 */

#include <miniz.hpp>
#include <op/scan/iceberg_avro_reader.hpp>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace sirius::op::scan {

//===----------------------------------------------------------------------===//
// Avro binary primitives
//===----------------------------------------------------------------------===//

namespace {

// ---------------------------------------------------------------------------
// Deflate decompression using DuckDB's bundled miniz
// ---------------------------------------------------------------------------

static std::vector<uint8_t> inflate_deflate(const uint8_t* data, size_t size)
{
  std::vector<uint8_t> out(size * 4);  // initial guess
  duckdb_miniz::mz_stream strm{};
  // -15 = raw deflate (no zlib/gzip header)
  int ret = duckdb_miniz::mz_inflateInit2(&strm, -15);
  if (ret != duckdb_miniz::MZ_OK) { throw std::runtime_error("avro: mz_inflateInit2 failed"); }
  strm.next_in   = data;
  strm.avail_in  = static_cast<unsigned int>(size);
  strm.next_out  = out.data();
  strm.avail_out = static_cast<unsigned int>(out.size());

  while (true) {
    ret = duckdb_miniz::mz_inflate(&strm, duckdb_miniz::MZ_FINISH);
    if (ret == duckdb_miniz::MZ_STREAM_END) break;
    if (ret == duckdb_miniz::MZ_BUF_ERROR) {
      // Output buffer too small — grow and retry.
      size_t used = out.size() - strm.avail_out;
      out.resize(out.size() * 2);
      strm.next_out  = out.data() + used;
      strm.avail_out = static_cast<unsigned int>(out.size() - used);
    } else if (ret != duckdb_miniz::MZ_OK) {
      duckdb_miniz::mz_inflateEnd(&strm);
      throw std::runtime_error("avro: deflate decompression failed (mz_inflate error " +
                               std::to_string(ret) + ")");
    }
  }
  out.resize(strm.total_out);
  duckdb_miniz::mz_inflateEnd(&strm);
  return out;
}

// ---------------------------------------------------------------------------
// Zigzag decoding (variable-length int / long)
// ---------------------------------------------------------------------------

static int64_t read_vlong(const uint8_t*& p, const uint8_t* end)
{
  uint64_t n = 0;
  int shift  = 0;
  while (p < end && (*p & 0x80u)) {
    n |= uint64_t(*p & 0x7Fu) << shift;
    shift += 7;
    ++p;
  }
  if (p >= end) { throw std::runtime_error("avro: truncated variable-length integer"); }
  n |= uint64_t(*p++) << shift;
  return int64_t((n >> 1) ^ -(n & 1));
}

static int32_t read_vint(const uint8_t*& p, const uint8_t* end)
{
  return int32_t(read_vlong(p, end));
}

// ---------------------------------------------------------------------------
// Bytes / string  (length-prefixed)
// ---------------------------------------------------------------------------

static std::string read_bytes_val(const uint8_t*& p, const uint8_t* end)
{
  int64_t len = read_vlong(p, end);
  if (len < 0 || static_cast<int64_t>(end - p) < len) {
    throw std::runtime_error("avro: invalid bytes/string length");
  }
  std::string s(reinterpret_cast<const char*>(p), static_cast<std::size_t>(len));
  p += len;
  return s;
}

// Skip bytes/string without materialising
static void skip_bytes_val(const uint8_t*& p, const uint8_t* end)
{
  int64_t len = read_vlong(p, end);
  if (len < 0 || static_cast<int64_t>(end - p) < len) {
    throw std::runtime_error("avro: invalid bytes/string length during skip");
  }
  p += len;
}

// ---------------------------------------------------------------------------
// Avro schema: simplified type representation
// ---------------------------------------------------------------------------

// Type tags for the subset of Avro types used in Iceberg manifest schemas.
enum class AvroKind {
  Null,
  Boolean,
  Int,
  Long,
  Float,
  Double,
  Bytes,
  String,
  Union,
  Array,
  Map,
  Record,
};

struct AvroField;  // forward declaration

struct AvroType {
  AvroKind kind = AvroKind::Null;
  std::vector<AvroType> union_branches;  // AvroKind::Union
  std::vector<AvroField> record_fields;  // AvroKind::Record
  // Array and map item types are stored as the first element of union_branches
  // (reused as a single-element container to avoid extra allocation).
  const AvroType& item_type() const { return union_branches[0]; }
};

struct AvroField {
  std::string name;
  AvroType type;
};

// ---------------------------------------------------------------------------
// JSON schema parser
//
// The embedded Avro schema is a JSON object.  We implement a minimal recursive
// descent parser sufficient for the Iceberg manifest schemas.  It handles:
//   primitive strings ("null", "int", …), union arrays ([…]), objects ({…}).
// ---------------------------------------------------------------------------

struct JsonParser {
  const char* p;
  const char* end;

  explicit JsonParser(std::string_view s) : p(s.data()), end(s.data() + s.size()) {}

  void skip_ws()
  {
    while (p < end && (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r'))
      ++p;
  }

  char peek()
  {
    skip_ws();
    if (p >= end) throw std::runtime_error("avro schema: unexpected end of JSON");
    return *p;
  }

  char consume()
  {
    skip_ws();
    if (p >= end) throw std::runtime_error("avro schema: unexpected end of JSON");
    return *p++;
  }

  void expect(char c)
  {
    char got = consume();
    if (got != c) {
      throw std::runtime_error(std::string("avro schema: expected '") + c + "' got '" + got + "'");
    }
  }

  std::string read_string_val()
  {
    expect('"');
    std::string out;
    while (p < end) {
      char c = *p++;
      if (c == '"') return out;
      if (c == '\\' && p < end) {
        // Handle basic escape sequences
        char esc = *p++;
        switch (esc) {
          case '"': out += '"'; break;
          case '\\': out += '\\'; break;
          case '/': out += '/'; break;
          case 'n': out += '\n'; break;
          case 't': out += '\t'; break;
          default: out += esc; break;
        }
      } else {
        out += c;
      }
    }
    throw std::runtime_error("avro schema: unterminated string");
  }

  void skip_object()
  {
    expect('{');
    if (peek() == '}') {
      ++p;
      return;
    }
    do {
      skip_ws();
      if (peek() == '}') break;
      read_string_val();  // key
      expect(':');
      skip_value();
    } while (peek() == ',' && ++p);
    expect('}');
  }

  void skip_array_literal()
  {
    expect('[');
    if (peek() == ']') {
      ++p;
      return;
    }
    do {
      skip_ws();
      if (peek() == ']') break;
      skip_value();
    } while (peek() == ',' && ++p);
    expect(']');
  }

  void skip_number()
  {
    // Read sign, digits, decimal, exponent
    while (p < end && ((*p >= '0' && *p <= '9') || *p == '-' || *p == '+' || *p == '.' ||
                       *p == 'e' || *p == 'E')) {
      ++p;
    }
  }

  void skip_value()
  {
    skip_ws();
    if (p >= end) return;
    char c = *p;
    if (c == '"') {
      read_string_val();
    } else if (c == '{') {
      skip_object();
    } else if (c == '[') {
      skip_array_literal();
    } else if (c == 't') {
      p += 4;  // true
    } else if (c == 'f') {
      p += 5;  // false
    } else if (c == 'n') {
      p += 4;  // null
    } else {
      skip_number();
    }
  }

  // Parse an Avro type definition (string, array, or object)
  AvroType parse_type()
  {
    skip_ws();
    char c = peek();
    AvroType t;
    if (c == '"') {
      std::string s = read_string_val();
      if (s == "null") {
        t.kind = AvroKind::Null;
      } else if (s == "boolean") {
        t.kind = AvroKind::Boolean;
      } else if (s == "int") {
        t.kind = AvroKind::Int;
      } else if (s == "long") {
        t.kind = AvroKind::Long;
      } else if (s == "float") {
        t.kind = AvroKind::Float;
      } else if (s == "double") {
        t.kind = AvroKind::Double;
      } else if (s == "bytes") {
        t.kind = AvroKind::Bytes;
      } else if (s == "string") {
        t.kind = AvroKind::String;
      } else {
        // Named type reference — treat as record placeholder (skip-only)
        t.kind = AvroKind::Record;
      }
    } else if (c == '[') {
      // Union
      t.kind = AvroKind::Union;
      ++p;  // consume '['
      skip_ws();
      while (peek() != ']') {
        t.union_branches.push_back(parse_type());
        skip_ws();
        if (peek() == ',') ++p;
      }
      ++p;  // consume ']'
    } else if (c == '{') {
      // Complex type object
      expect('{');
      std::string type_name;
      // Read keys until we find "type"
      while (peek() != '}') {
        std::string key = read_string_val();
        expect(':');
        if (key == "type") {
          type_name = read_string_val();
        } else if (key == "fields" && type_name == "record") {
          // Parse fields array inline
          t.kind = AvroKind::Record;
          expect('[');
          skip_ws();
          while (peek() != ']') {
            t.record_fields.push_back(parse_field());
            skip_ws();
            if (peek() == ',') ++p;
          }
          ++p;  // consume ']'
        } else if (key == "items") {
          // Array item type
          t.kind        = AvroKind::Array;
          AvroType item = parse_type();
          t.union_branches.push_back(std::move(item));
        } else if (key == "values") {
          // Map value type
          t.kind        = AvroKind::Map;
          AvroType item = parse_type();
          t.union_branches.push_back(std::move(item));
        } else {
          skip_value();
        }
        skip_ws();
        if (peek() == ',') ++p;
      }
      expect('}');
      // If we got a type_name but didn't set kind yet
      if (t.kind == AvroKind::Null) {
        if (type_name == "record") {
          t.kind = AvroKind::Record;
        } else if (type_name == "array") {
          t.kind = AvroKind::Array;
        } else if (type_name == "map") {
          t.kind = AvroKind::Map;
        }
      }
    }
    return t;
  }

  AvroField parse_field()
  {
    expect('{');
    AvroField f;
    while (peek() != '}') {
      std::string key = read_string_val();
      expect(':');
      if (key == "name") {
        f.name = read_string_val();
      } else if (key == "type") {
        f.type = parse_type();
      } else {
        skip_value();
      }
      skip_ws();
      if (peek() == ',') ++p;
    }
    expect('}');
    return f;
  }

  AvroType parse_schema() { return parse_type(); }
};

// ---------------------------------------------------------------------------
// Avro binary: skip / read a value based on its AvroType
// ---------------------------------------------------------------------------

// Forward declarations
static void skip_avro_value(const AvroType& t, const uint8_t*& p, const uint8_t* end);
static int64_t skip_avro_block_items(const uint8_t*& p, const uint8_t* end);

static void skip_avro_value(const AvroType& t, const uint8_t*& p, const uint8_t* end)
{
  switch (t.kind) {
    case AvroKind::Null: break;
    case AvroKind::Boolean:
      if (p >= end) throw std::runtime_error("avro: truncated boolean");
      ++p;
      break;
    case AvroKind::Int: read_vint(p, end); break;
    case AvroKind::Long: read_vlong(p, end); break;
    case AvroKind::Float:
      if (end - p < 4) throw std::runtime_error("avro: truncated float");
      p += 4;
      break;
    case AvroKind::Double:
      if (end - p < 8) throw std::runtime_error("avro: truncated double");
      p += 8;
      break;
    case AvroKind::Bytes:
    case AvroKind::String: skip_bytes_val(p, end); break;
    case AvroKind::Union: {
      int32_t idx = read_vint(p, end);
      if (idx < 0 || static_cast<std::size_t>(idx) >= t.union_branches.size()) {
        throw std::runtime_error("avro: union index out of range");
      }
      skip_avro_value(t.union_branches[idx], p, end);
      break;
    }
    case AvroKind::Record:
      for (auto const& f : t.record_fields) {
        skip_avro_value(f.type, p, end);
      }
      break;
    case AvroKind::Array: {
      int64_t count;
      while ((count = read_vlong(p, end)) != 0) {
        if (count < 0) {
          // Negative count means the block byte-count follows; skip directly.
          int64_t byte_count = read_vlong(p, end);
          if (byte_count < 0 || static_cast<int64_t>(end - p) < byte_count) {
            throw std::runtime_error("avro: invalid array block byte count");
          }
          p += byte_count;
        } else {
          for (int64_t i = 0; i < count; ++i) {
            skip_avro_value(t.item_type(), p, end);
          }
        }
      }
      break;
    }
    case AvroKind::Map: {
      int64_t count;
      while ((count = read_vlong(p, end)) != 0) {
        if (count < 0) {
          int64_t byte_count = read_vlong(p, end);
          if (byte_count < 0 || static_cast<int64_t>(end - p) < byte_count) {
            throw std::runtime_error("avro: invalid map block byte count");
          }
          p += byte_count;
        } else {
          for (int64_t i = 0; i < count; ++i) {
            skip_bytes_val(p, end);  // key (always string in avro maps)
            skip_avro_value(t.item_type(), p, end);
          }
        }
      }
      break;
    }
  }
}

// ---------------------------------------------------------------------------
// Avro Object Container File (OCF) header
// ---------------------------------------------------------------------------

struct AvroHeader {
  AvroType schema;
  std::string codec;  // "null", "deflate", "snappy", …
  std::array<uint8_t, 16> sync_marker{};
};

/// Manages Avro OCF block decoding: reads count + byte_count, decompresses if
/// deflate, and provides pointers to the block's row data. After processing
/// rows, call advance_source() to update the source pointer past the block.
struct AvroBlockDecoder {
  int64_t count       = 0;
  const uint8_t* data = nullptr;  ///< Start of (decompressed) row data.
  const uint8_t* end  = nullptr;  ///< End of row data.

  /// Decode the next block from the source. Returns false if no more blocks.
  bool next(const uint8_t*& p, const uint8_t* src_end, std::string const& codec)
  {
    if (p >= src_end) return false;
    count = read_vlong(p, src_end);
    if (count == 0) return false;
    if (count < 0) count = -count;
    int64_t byte_count = read_vlong(p, src_end);

    if (codec == "deflate") {
      storage_ = inflate_deflate(p, static_cast<size_t>(byte_count));
      data     = storage_.data();
      end      = data + storage_.size();
      p += byte_count;
    } else {
      data = p;
      end  = p + byte_count;
    }
    return true;
  }

  /// After processing all rows, advance the source pointer past this block.
  /// Must be called before reading the sync marker.
  void advance_source(const uint8_t*& p, std::string const& codec) const
  {
    if (codec != "deflate") { p = data + (end - data); }
  }

  /// Verify and consume the 16-byte sync marker after a block.
  static void verify_sync(const uint8_t*& p, const uint8_t* src_end, AvroHeader const& hdr)
  {
    if (src_end - p < 16) { throw std::runtime_error("avro: truncated sync marker"); }
    if (std::memcmp(p, hdr.sync_marker.data(), 16) != 0) {
      throw std::runtime_error("avro: sync marker mismatch");
    }
    p += 16;
  }

 private:
  std::vector<uint8_t> storage_;  ///< Owns decompressed bytes if deflate.
};

static AvroHeader parse_avro_header(const uint8_t*& p, const uint8_t* end)
{
  // 4-byte magic
  static const uint8_t kMagic[4] = {0x4F, 0x62, 0x6A, 0x01};  // "Obj\x01"
  if (end - p < 4 || std::memcmp(p, kMagic, 4) != 0) {
    throw std::runtime_error("avro: not an Avro Object Container File");
  }
  p += 4;

  AvroHeader header;
  std::string schema_json;

  // Read file metadata (avro map of string→bytes)
  // Map blocks: {count, (key, value)*}* followed by 0-count terminator
  int64_t map_count;
  while ((map_count = read_vlong(p, end)) != 0) {
    if (map_count < 0) {
      // Negative count means block has a byte count; skip it
      read_vlong(p, end);
      map_count = -map_count;
    }
    for (int64_t i = 0; i < map_count; ++i) {
      std::string key = read_bytes_val(p, end);
      std::string val = read_bytes_val(p, end);
      if (key == "avro.schema") {
        schema_json = std::move(val);
      } else if (key == "avro.codec") {
        header.codec = std::move(val);
      }
    }
  }

  if (schema_json.empty()) { throw std::runtime_error("avro: missing avro.schema metadata"); }
  if (header.codec.empty()) { header.codec = "null"; }
  if (header.codec != "null" && header.codec != "deflate") {
    throw std::runtime_error("avro: unsupported codec '" + header.codec + "'");
  }

  // Parse schema
  JsonParser jp(schema_json);
  header.schema = jp.parse_schema();

  // 16-byte sync marker
  if (end - p < 16) { throw std::runtime_error("avro: file truncated at sync marker"); }
  std::memcpy(header.sync_marker.data(), p, 16);
  p += 16;

  return header;
}

// ---------------------------------------------------------------------------
// Manifest-list reader
// ---------------------------------------------------------------------------

// Find a field index by name in a record type (-1 if not found).
static int find_field(const AvroType& rec, const std::string& name)
{
  for (int i = 0; i < static_cast<int>(rec.record_fields.size()); ++i) {
    if (rec.record_fields[i].name == name) return i;
  }
  return -1;
}

// Read a single field value from a record, skipping everything before field_idx.
// p must be positioned at the start of the record.
// After this call, p is at the beginning of the requested field's encoded value.
static void advance_to_field(int field_idx,
                             const AvroType& rec_type,
                             const uint8_t*& p,
                             const uint8_t* end)
{
  for (int i = 0; i < field_idx; ++i) {
    skip_avro_value(rec_type.record_fields[i].type, p, end);
  }
}

}  // anonymous namespace

//===----------------------------------------------------------------------===//
// Public API
//===----------------------------------------------------------------------===//

std::vector<std::pair<std::string, int>> read_iceberg_manifest_list(const std::string& path)
{
  // Load the entire file into memory (manifest lists are typically < 1 MB)
  std::ifstream f(path, std::ios::binary);
  if (!f) { throw std::runtime_error("avro: cannot open manifest list: " + path); }
  std::vector<uint8_t> buf(std::istreambuf_iterator<char>(f), {});

  const uint8_t* p   = buf.data();
  const uint8_t* end = buf.data() + buf.size();

  AvroHeader hdr = parse_avro_header(p, end);

  // Locate fields in the schema
  int mp_idx = find_field(hdr.schema, "manifest_path");
  int ct_idx = find_field(hdr.schema, "content");
  if (mp_idx < 0) { throw std::runtime_error("avro: manifest_path field not found"); }
  if (ct_idx < 0) { throw std::runtime_error("avro: content field not found"); }

  int first_idx  = std::min(mp_idx, ct_idx);
  int second_idx = std::max(mp_idx, ct_idx);
  bool mp_first  = (mp_idx < ct_idx);

  std::vector<std::pair<std::string, int>> result;
  std::array<uint8_t, 16> block_sync{};

  // Data blocks
  // OCF data blocks: count (zigzag long), byte_count (zigzag long, ALWAYS present),
  // records..., sync_marker.  The byte_count is present for both positive and negative
  // counts (unlike embedded array/map blocks where it only appears for negative counts).
  AvroBlockDecoder block;
  while (block.next(p, end, hdr.codec)) {
    for (int64_t row = 0; row < block.count; ++row) {
      advance_to_field(first_idx, hdr.schema, block.data, block.end);

      std::string mp_val;
      int ct_val = 0;

      if (mp_first) {
        mp_val = read_bytes_val(block.data, block.end);
        for (int i = first_idx + 1; i < second_idx; ++i) {
          skip_avro_value(hdr.schema.record_fields[i].type, block.data, block.end);
        }
        ct_val = read_vint(block.data, block.end);
      } else {
        ct_val = read_vint(block.data, block.end);
        for (int i = first_idx + 1; i < second_idx; ++i) {
          skip_avro_value(hdr.schema.record_fields[i].type, block.data, block.end);
        }
        mp_val = read_bytes_val(block.data, block.end);
      }

      for (std::size_t i = second_idx + 1; i < hdr.schema.record_fields.size(); ++i) {
        skip_avro_value(hdr.schema.record_fields[i].type, block.data, block.end);
      }

      result.emplace_back(std::move(mp_val), ct_val);
    }
    block.advance_source(p, hdr.codec);
    AvroBlockDecoder::verify_sync(p, end, hdr);
  }

  return result;
}

// ---------------------------------------------------------------------------
// Helpers for reading optional (union-wrapped) fields
// ---------------------------------------------------------------------------

// Read a union value and return the string if non-null, else empty.
static std::string read_optional_string(const AvroType& type, const uint8_t*& p, const uint8_t* end)
{
  if (type.kind == AvroKind::String) { return read_bytes_val(p, end); }
  if (type.kind == AvroKind::Union) {
    int32_t branch = read_vint(p, end);
    if (branch < 0 || static_cast<std::size_t>(branch) >= type.union_branches.size()) {
      throw std::runtime_error("avro: union index out of range");
    }
    if (type.union_branches[branch].kind == AvroKind::Null) { return {}; }
    return read_bytes_val(p, end);
  }
  skip_avro_value(type, p, end);
  return {};
}

// Read a union value and return the long if non-null, else -1.
static int64_t read_optional_long(const AvroType& type, const uint8_t*& p, const uint8_t* end)
{
  if (type.kind == AvroKind::Long) { return read_vlong(p, end); }
  if (type.kind == AvroKind::Union) {
    int32_t branch = read_vint(p, end);
    if (branch < 0 || static_cast<std::size_t>(branch) >= type.union_branches.size()) {
      throw std::runtime_error("avro: union index out of range");
    }
    if (type.union_branches[branch].kind == AvroKind::Null) { return -1; }
    return read_vlong(p, end);
  }
  skip_avro_value(type, p, end);
  return -1;
}

// ---------------------------------------------------------------------------
// Manifest reader — delete entries with V3 fields
// ---------------------------------------------------------------------------

std::vector<IcebergDeleteFileEntry> read_iceberg_manifest_entries(const std::string& path,
                                                                  int target_content)
{
  std::ifstream f(path, std::ios::binary);
  if (!f) { throw std::runtime_error("avro: cannot open manifest: " + path); }
  std::vector<uint8_t> buf(std::istreambuf_iterator<char>(f), {});

  const uint8_t* p   = buf.data();
  const uint8_t* end = buf.data() + buf.size();

  AvroHeader hdr = parse_avro_header(p, end);

  // Top-level sequence_number field (union [null, long]).
  int seq_idx = find_field(hdr.schema, "sequence_number");

  int df_idx = find_field(hdr.schema, "data_file");
  if (df_idx < 0) { throw std::runtime_error("avro: data_file field not found in manifest"); }

  const AvroType& df_type = hdr.schema.record_fields[df_idx].type;

  // Locate all fields we care about inside data_file.
  // Required fields:
  int content_idx    = find_field(df_type, "content");
  int filepath_idx   = find_field(df_type, "file_path");
  int fileformat_idx = find_field(df_type, "file_format");
  if (content_idx < 0) { throw std::runtime_error("avro: data_file.content not found"); }
  if (filepath_idx < 0) { throw std::runtime_error("avro: data_file.file_path not found"); }
  if (fileformat_idx < 0) { throw std::runtime_error("avro: data_file.file_format not found"); }

  // Optional V3 fields (may be absent in V2 manifests):
  int ref_data_file_idx  = find_field(df_type, "referenced_data_file");
  int content_offset_idx = find_field(df_type, "content_offset");
  int content_size_idx   = find_field(df_type, "content_size_in_bytes");

  // Build a set of field indices we want to read.
  struct FieldSlot {
    int idx;
    enum Tag { CONTENT, FILE_PATH, FILE_FORMAT, REF_DATA_FILE, CONTENT_OFFSET, CONTENT_SIZE };
    Tag tag;
  };
  std::vector<FieldSlot> wanted;
  wanted.push_back({content_idx, FieldSlot::CONTENT});
  wanted.push_back({filepath_idx, FieldSlot::FILE_PATH});
  wanted.push_back({fileformat_idx, FieldSlot::FILE_FORMAT});
  if (ref_data_file_idx >= 0) { wanted.push_back({ref_data_file_idx, FieldSlot::REF_DATA_FILE}); }
  if (content_offset_idx >= 0) {
    wanted.push_back({content_offset_idx, FieldSlot::CONTENT_OFFSET});
  }
  if (content_size_idx >= 0) { wanted.push_back({content_size_idx, FieldSlot::CONTENT_SIZE}); }
  // Sort by field index so we can read in a single forward pass.
  std::sort(
    wanted.begin(), wanted.end(), [](auto const& a, auto const& b) { return a.idx < b.idx; });

  int const num_df_fields = static_cast<int>(df_type.record_fields.size());

  std::vector<IcebergDeleteFileEntry> result;

  AvroBlockDecoder block;
  while (block.next(p, end, hdr.codec)) {
    for (int64_t row = 0; row < block.count; ++row) {
      // Read sequence_number from top-level before advancing to data_file.
      int64_t entry_seq = 0;
      if (seq_idx >= 0 && seq_idx < df_idx) {
        advance_to_field(seq_idx, hdr.schema, block.data, block.end);
        entry_seq =
          read_optional_long(hdr.schema.record_fields[seq_idx].type, block.data, block.end);
        for (int i = seq_idx + 1; i < df_idx; ++i) {
          skip_avro_value(hdr.schema.record_fields[i].type, block.data, block.end);
        }
      } else {
        advance_to_field(df_idx, hdr.schema, block.data, block.end);
      }

      IcebergDeleteFileEntry entry;
      entry.sequence_number = entry_seq;
      int cursor            = 0;
      for (auto const& slot : wanted) {
        for (int i = cursor; i < slot.idx; ++i) {
          skip_avro_value(df_type.record_fields[i].type, block.data, block.end);
        }
        auto const& field_type = df_type.record_fields[slot.idx].type;
        switch (slot.tag) {
          case FieldSlot::CONTENT: entry.content = read_vint(block.data, block.end); break;
          case FieldSlot::FILE_PATH: entry.file_path = read_bytes_val(block.data, block.end); break;
          case FieldSlot::FILE_FORMAT:
            entry.file_format = read_bytes_val(block.data, block.end);
            std::transform(entry.file_format.begin(),
                           entry.file_format.end(),
                           entry.file_format.begin(),
                           ::tolower);
            break;
          case FieldSlot::REF_DATA_FILE:
            entry.referenced_data_file = read_optional_string(field_type, block.data, block.end);
            break;
          case FieldSlot::CONTENT_OFFSET:
            entry.content_offset = read_optional_long(field_type, block.data, block.end);
            break;
          case FieldSlot::CONTENT_SIZE:
            entry.content_size_in_bytes = read_optional_long(field_type, block.data, block.end);
            break;
        }
        cursor = slot.idx + 1;
      }

      for (int i = cursor; i < num_df_fields; ++i) {
        skip_avro_value(df_type.record_fields[i].type, block.data, block.end);
      }

      if (target_content < 0 || entry.content == target_content) {
        result.push_back(std::move(entry));
      }

      for (std::size_t i = df_idx + 1; i < hdr.schema.record_fields.size(); ++i) {
        skip_avro_value(hdr.schema.record_fields[i].type, block.data, block.end);
      }
    }
    block.advance_source(p, hdr.codec);
    AvroBlockDecoder::verify_sync(p, end, hdr);
  }

  return result;
}

}  // namespace sirius::op::scan
