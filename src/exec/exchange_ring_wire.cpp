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

#include "exec/exchange_ring_wire.hpp"

#include "sirius/exception.hpp"

#include <cudf/null_mask.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/traits.hpp>

#include <rmm/aligned.hpp>

#include <algorithm>
#include <array>
#include <bit>
#include <cstring>
#include <limits>
#include <string_view>

namespace sirius::exec::ring {
namespace {

static_assert(std::endian::native == std::endian::little, "the ring header is little-endian");

using cucascade::memory::column_metadata;

// "SXR1" | u8 kind | u64 rows | u64 stream bytes | kind-specific body, as native integers:
//   columns:    u32 n | n x column
//   compressed: u64 uncompressed | u32 header len | header | u32 n | n x (u32 len | utf-8 name)
// column: i32 type | i32 rows | i32 nulls | i32 scale | u8 has mask | u64 mask offset |
//         u64 mask size | u8 has data | u64 data offset | u64 data size | u32 n | n x column
constexpr std::array<char, 4> magic{'S', 'X', 'R', '1'};
constexpr std::size_t min_column_bytes = 4 * 4 + 2 * 1 + 4 * 8 + 4;
// STRING is the deepest column sent: one level of children.
constexpr int max_depth = 1;

void require(bool ok, std::string_view what)
{
  if (!ok) { throw sirius::invalid_input_exception("ring header: {}", what); }
}

template <typename T>
void put(std::vector<std::uint8_t>& out, T value)
{
  auto const* bytes = reinterpret_cast<std::uint8_t const*>(&value);
  out.insert(out.end(), bytes, bytes + sizeof(T));
}

template <typename T>
T take(std::span<std::uint8_t const>& in)
{
  require(in.size() >= sizeof(T), "truncated");
  T value;
  std::memcpy(&value, in.data(), sizeof(T));
  in = in.subspan(sizeof(T));
  return value;
}

std::span<std::uint8_t const> take_bytes(std::span<std::uint8_t const>& in, std::size_t n)
{
  require(in.size() >= n, "truncated");
  auto const head = in.first(n);
  in              = in.subspan(n);
  return head;
}

void put_column(std::vector<std::uint8_t>& out, column_metadata const& c)
{
  put(out, c.type_id);
  put(out, c.num_rows);
  put(out, c.null_count);
  put(out, c.scale);
  put(out, static_cast<std::uint8_t>(c.has_null_mask));
  put(out, static_cast<std::uint64_t>(c.null_mask_offset));
  put(out, static_cast<std::uint64_t>(c.null_mask_size));
  put(out, static_cast<std::uint8_t>(c.has_data));
  put(out, static_cast<std::uint64_t>(c.data_offset));
  put(out, static_cast<std::uint64_t>(c.data_size));
  put(out, static_cast<std::uint32_t>(c.children.size()));
  for (auto const& child : c.children) {
    put_column(out, child);
  }
}

bool take_flag(std::span<std::uint8_t const>& in)
{
  auto const flag = take<std::uint8_t>(in);
  require(flag <= 1, "flag is not 0 or 1");
  return flag == 1;
}

column_metadata take_column(std::span<std::uint8_t const>& in, int depth)
{
  column_metadata c{};
  c.type_id          = take<std::int32_t>(in);
  c.num_rows         = take<std::int32_t>(in);
  c.null_count       = take<std::int32_t>(in);
  c.scale            = take<std::int32_t>(in);
  c.has_null_mask    = take_flag(in);
  c.null_mask_offset = take<std::uint64_t>(in);
  c.null_mask_size   = take<std::uint64_t>(in);
  c.has_data         = take_flag(in);
  c.data_offset      = take<std::uint64_t>(in);
  c.data_size        = take<std::uint64_t>(in);
  auto const n       = take<std::uint32_t>(in);
  require(n == 0 || depth < max_depth, "columns nest too deep");
  // Bounded by the bytes left before reserving, so a forged count cannot force a huge allocation.
  require(n <= in.size() / min_column_bytes, "child count out of range");
  c.children.reserve(n);
  for (std::uint32_t i = 0; i < n; ++i) {
    c.children.push_back(take_column(in, depth + 1));
  }
  return c;
}

bool is_string(cudf::type_id id) { return id == cudf::type_id::STRING; }

bool is_fixed_width(cudf::type_id id)
{
  // EMPTY is checked first: is_fixed_width throws on it, since the type dispatcher omits it.
  return id != cudf::type_id::EMPTY && cudf::is_fixed_width(cudf::data_type{id});
}

/// A buffer of @p size at @p offset that fits in a stream of @p stream bytes.
void require_inside(std::size_t offset, std::size_t size, std::uint64_t stream_bytes)
{
  require(offset <= stream_bytes && size <= stream_bytes - offset, "buffer outside the stream");
}

/// Checks one flat buffer (a mask or data) of @p c; @p need is the bytes its type requires.
void check_buffer(bool has,
                  std::size_t offset,
                  std::size_t size,
                  std::size_t need,
                  std::uint64_t stream_bytes,
                  std::string_view what)
{
  if (!has) {
    require(need == 0, what);
    return;
  }
  require(size >= need, what);
  require_inside(offset, size, stream_bytes);
}

void check_column(column_metadata const& c, std::uint64_t rows, std::uint64_t stream_bytes)
{
  require(c.type_id >= 0 && c.type_id < static_cast<std::int32_t>(cudf::type_id::NUM_TYPE_IDS),
          "type id out of range");
  auto const id = static_cast<cudf::type_id>(c.type_id);
  require(is_string(id) || is_fixed_width(id), "type is neither fixed-width nor STRING");
  require(c.num_rows >= 0 && static_cast<std::uint64_t>(c.num_rows) == rows,
          "column rows differ from the batch's");
  require(c.null_count >= 0 && c.null_count <= c.num_rows, "null count out of range");
  require(c.null_count == 0 || c.has_null_mask, "nulls without a mask");
  require(cudf::is_fixed_point(cudf::data_type{id}) || c.scale == 0,
          "scale on a type that is not fixed-point");
  auto const n = static_cast<cudf::size_type>(c.num_rows);
  check_buffer(c.has_null_mask,
               c.null_mask_offset,
               c.null_mask_size,
               c.has_null_mask ? cudf::num_bitmask_words(n) * sizeof(cudf::bitmask_type) : 0,
               stream_bytes,
               "null mask too small");
  if (is_string(id)) {
    require(c.children.size() == 1, "a STRING column has one offsets child");
    auto const& offsets = c.children.front();
    require(offsets.type_id == static_cast<std::int32_t>(cudf::type_id::INT32) ||
              offsets.type_id == static_cast<std::int32_t>(cudf::type_id::INT64),
            "string offsets are not INT32 or INT64");
    require(static_cast<std::uint64_t>(offsets.num_rows) == rows + 1,
            "string offsets are not rows + 1");
    require(!offsets.has_null_mask && offsets.null_count == 0 && offsets.children.empty(),
            "string offsets have a mask or children");
    auto const width = cudf::size_of(cudf::data_type{static_cast<cudf::type_id>(offsets.type_id)});
    check_buffer(offsets.has_data,
                 offsets.data_offset,
                 offsets.data_size,
                 (rows + 1) * width,
                 stream_bytes,
                 "string offsets too small");
    if (c.has_data) { require_inside(c.data_offset, c.data_size, stream_bytes); }
    require(offsets.type_id == static_cast<std::int32_t>(cudf::type_id::INT64) ||
              c.data_size <= static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max()),
            "chars overflow INT32 offsets");
  } else {
    require(c.children.empty(), "a fixed-width column has children");
    check_buffer(c.has_data,
                 c.data_offset,
                 c.data_size,
                 rows * cudf::size_of(cudf::data_type{id}),
                 stream_bytes,
                 "data too small");
  }
}

std::size_t align(std::size_t offset) { return rmm::align_up(offset, stream_alignment); }

void walk(column_metadata const& c, std::vector<stream_buffer>& out)
{
  if (c.has_null_mask && c.null_mask_size > 0) {
    out.push_back({c.null_mask_offset, c.null_mask_size});
  }
  if (c.has_data && c.data_size > 0) { out.push_back({c.data_offset, c.data_size}); }
  for (auto const& child : c.children) {
    walk(child, out);
  }
}

}  // namespace

std::vector<std::uint8_t> encode_header(wire_batch const& batch)
{
  std::vector<std::uint8_t> out(magic.begin(), magic.end());
  put(out, static_cast<std::uint8_t>(batch.kind));
  put(out, batch.rows);
  put(out, batch.stream_bytes);
  if (batch.kind == batch_kind::compressed) {
    put(out, batch.uncompressed_bytes);
    put(out, static_cast<std::uint32_t>(batch.blob_header.size()));
    out.insert(out.end(), batch.blob_header.begin(), batch.blob_header.end());
    put(out, static_cast<std::uint32_t>(batch.blob_columns.size()));
    for (auto const& name : batch.blob_columns) {
      put(out, static_cast<std::uint32_t>(name.size()));
      out.insert(out.end(), name.begin(), name.end());
    }
    return out;
  }
  put(out, static_cast<std::uint32_t>(batch.columns.size()));
  for (auto const& c : batch.columns) {
    put_column(out, c);
  }
  return out;
}

wire_batch decode_header(std::span<std::uint8_t const> in)
{
  require(take<std::array<char, 4>>(in) == magic, "bad magic");
  wire_batch batch;
  auto const kind = take<std::uint8_t>(in);
  require(kind >= 1 && kind <= 3, "unknown batch kind");
  batch.kind         = static_cast<batch_kind>(kind);
  batch.rows         = take<std::uint64_t>(in);
  batch.stream_bytes = take<std::uint64_t>(in);
  // rows + 1 string offsets must fit in a cudf::size_type.
  require(batch.rows >= 1 &&
            batch.rows < static_cast<std::uint64_t>(std::numeric_limits<cudf::size_type>::max()),
          "row count out of range");
  if (batch.kind == batch_kind::compressed) {
    batch.uncompressed_bytes = take<std::uint64_t>(in);
    auto const header        = take_bytes(in, take<std::uint32_t>(in));
    batch.blob_header.assign(header.begin(), header.end());
    auto const n = take<std::uint32_t>(in);
    require(n <= in.size() / sizeof(std::uint32_t), "column name count out of range");
    batch.blob_columns.reserve(n);
    for (std::uint32_t i = 0; i < n; ++i) {
      auto const name = take_bytes(in, take<std::uint32_t>(in));
      batch.blob_columns.emplace_back(name.begin(), name.end());
    }
    require(batch.stream_bytes > 0, "a compressed batch has no payload");
    require(in.empty(), "trailing bytes");
    return batch;
  }
  auto const n = take<std::uint32_t>(in);
  require(n >= 1 && n <= in.size() / min_column_bytes, "column count out of range");
  batch.columns.reserve(n);
  for (std::uint32_t i = 0; i < n; ++i) {
    batch.columns.push_back(take_column(in, 0));
    check_column(batch.columns.back(), batch.rows, batch.stream_bytes);
  }
  require(in.empty(), "trailing bytes");
  return batch;
}

std::vector<stream_buffer> stream_map(std::vector<column_metadata> const& columns)
{
  std::vector<stream_buffer> out;
  for (auto const& c : columns) {
    walk(c, out);
  }
  return out;
}

std::vector<stream_piece> pieces_in(std::span<stream_buffer const> buffers,
                                    std::size_t begin,
                                    std::size_t end)
{
  std::vector<stream_piece> out;
  if (begin >= end) { return out; }
  for (std::size_t i = 0; i < buffers.size(); ++i) {
    auto const& b   = buffers[i];
    auto const from = std::max(begin, b.offset);
    auto const to   = std::min(end, b.offset + b.size);
    if (from >= to) { continue; }
    out.push_back({i, from - b.offset, from - begin, to - from});
  }
  std::sort(out.begin(), out.end(), [](stream_piece const& a, stream_piece const& b) {
    return a.range_offset < b.range_offset;
  });
  return out;
}

bool is_sliced(cudf::table_view const& table)
{
  return std::any_of(table.begin(), table.end(), [](cudf::column_view const& c) {
    return c.offset() != 0 ||
           std::any_of(c.child_begin(), c.child_end(), [](cudf::column_view const& child) {
             return child.offset() != 0;
           });
  });
}

std::vector<column_metadata> plan_stream(cudf::table_view const& table,
                                         rmm::cuda_stream_view stream,
                                         std::size_t& stream_bytes)
{
  if (is_sliced(table)) {
    throw sirius::invalid_input_exception("ring exchange cannot lay out a sliced table");
  }
  std::size_t cursor = 0;
  auto place         = [&](std::size_t size) {
    cursor           = align(cursor);
    auto const where = cursor;
    cursor += size;
    return where;
  };
  std::vector<column_metadata> columns;
  columns.reserve(table.num_columns());
  for (auto const& col : table) {
    auto const id = col.type().id();
    if (!is_string(id) && !is_fixed_width(id)) {
      throw sirius::invalid_input_exception("ring exchange cannot send type id {}",
                                            static_cast<std::int32_t>(id));
    }
    column_metadata c{};
    c.type_id    = static_cast<std::int32_t>(id);
    c.num_rows   = col.size();
    c.null_count = col.null_count();
    c.scale      = cudf::is_fixed_point(col.type()) ? col.type().scale() : 0;
    if (col.nullable()) {
      c.has_null_mask    = true;
      c.null_mask_size   = cudf::bitmask_allocation_size_bytes(col.size());
      c.null_mask_offset = place(c.null_mask_size);
    }
    if (is_string(id)) {
      cudf::strings_column_view const strings{col};
      auto const chars = col.size() > 0 ? static_cast<std::size_t>(strings.chars_size(stream)) : 0;
      if (chars > 0) {
        c.has_data    = true;
        c.data_size   = chars;
        c.data_offset = place(chars);
      }
      auto const offsets = strings.offsets();
      column_metadata o{};
      o.type_id     = static_cast<std::int32_t>(offsets.type().id());
      o.num_rows    = col.size() + 1;
      o.has_data    = true;
      o.data_size   = static_cast<std::size_t>(col.size() + 1) * cudf::size_of(offsets.type());
      o.data_offset = place(o.data_size);
      c.children.push_back(o);
    } else if (col.size() > 0) {
      c.has_data    = true;
      c.data_size   = static_cast<std::size_t>(col.size()) * cudf::size_of(col.type());
      c.data_offset = place(c.data_size);
    }
    columns.push_back(std::move(c));
  }
  stream_bytes = cursor;
  return columns;
}

std::vector<void const*> stream_sources(cudf::table_view const& table,
                                        std::vector<column_metadata> const& columns)
{
  std::vector<void const*> out;
  for (cudf::size_type i = 0; i < table.num_columns(); ++i) {
    auto const& col = table.column(i);
    auto const& c   = columns.at(static_cast<std::size_t>(i));
    if (c.has_null_mask && c.null_mask_size > 0) { out.push_back(col.null_mask()); }
    if (c.has_data && c.data_size > 0) { out.push_back(col.head()); }
    if (!c.children.empty()) {
      auto const& o = c.children.front();
      if (o.has_data && o.data_size > 0) {
        out.push_back(cudf::strings_column_view{col}.offsets().head());
      }
    }
  }
  return out;
}

}  // namespace sirius::exec::ring
