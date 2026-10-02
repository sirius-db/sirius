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

#include "exec/exchange_direct.hpp"

#include "sirius/exception.hpp"

#include <cudf/null_mask.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/traits.hpp>

#include <array>
#include <bit>
#include <cstring>
#include <limits>
#include <string_view>

namespace sirius::exec {
namespace {

// "SXD1" | i32 rows | u32 ncols | per column { i32 type_id | i32 scale | i32 null_count |
// u8 has_mask | STRING only: i32 offsets type_id | u64 chars }, copied as native integers.
static_assert(std::endian::native == std::endian::little, "the direct layout is little-endian");

constexpr std::array<char, 4> magic{'S', 'X', 'D', '1'};
constexpr std::size_t min_column_bytes = 13;
constexpr std::size_t alignment        = 256;

void require(bool ok, std::string_view what)
{
  if (!ok) { throw sirius::invalid_input_exception("direct layout: {}", what); }
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

bool is_string(cudf::data_type type) { return type.id() == cudf::type_id::STRING; }

bool is_sendable(cudf::data_type type)
{
  // EMPTY is checked first: is_fixed_width throws on it, since the type dispatcher omits it.
  return is_string(type) || (type.id() != cudf::type_id::EMPTY && cudf::is_fixed_width(type));
}

}  // namespace

std::vector<std::uint8_t> encode_layout(direct_layout const& layout)
{
  std::vector<std::uint8_t> out(magic.begin(), magic.end());
  put(out, layout.rows);
  put(out, static_cast<std::uint32_t>(layout.columns.size()));
  for (auto const& c : layout.columns) {
    put(out, static_cast<std::int32_t>(c.type.id()));
    put(out, c.type.scale());
    put(out, c.null_count);
    put(out, static_cast<std::uint8_t>(c.has_mask));
    if (is_string(c.type)) {
      put(out, static_cast<std::int32_t>(c.offsets));
      put(out, c.chars);
    }
  }
  return out;
}

direct_layout decode_layout(std::span<std::uint8_t const> in)
{
  require(take<std::array<char, 4>>(in) == magic, "bad magic");
  direct_layout layout{take<cudf::size_type>(in), {}};
  // rows + 1 string offsets must fit in a cudf::size_type.
  require(layout.rows >= 1 && layout.rows < std::numeric_limits<cudf::size_type>::max(),
          "row count out of range");
  auto const ncols = take<std::uint32_t>(in);
  // Bounded by the bytes left before reserving, so a forged count cannot force a huge allocation.
  require(ncols >= 1 && ncols <= in.size() / min_column_bytes, "column count out of range");
  layout.columns.reserve(ncols);
  for (std::uint32_t i = 0; i < ncols; ++i) {
    auto const id = take<std::int32_t>(in);
    require(id >= 0 && id < static_cast<std::int32_t>(cudf::type_id::NUM_TYPE_IDS),
            "type id out of range");
    cudf::data_type type{static_cast<cudf::type_id>(id)};
    require(is_sendable(type), "type is neither fixed-width nor STRING");
    auto const scale = take<std::int32_t>(in);
    if (cudf::is_fixed_point(type)) {
      type = cudf::data_type{type.id(), scale};
    } else {
      require(scale == 0, "scale on a type that is not fixed-point");
    }
    direct_column c{type, take<cudf::size_type>(in), false};
    auto const has_mask = take<std::uint8_t>(in);
    require(has_mask <= 1, "mask flag is not 0 or 1");
    c.has_mask = has_mask == 1;
    require(c.null_count >= 0 && c.null_count <= layout.rows, "null count out of range");
    require(c.null_count == 0 || c.has_mask, "nulls without a mask");
    if (is_string(type)) {
      c.offsets = static_cast<cudf::type_id>(take<std::int32_t>(in));
      c.chars   = take<std::uint64_t>(in);
      require(c.offsets == cudf::type_id::INT32 || c.offsets == cudf::type_id::INT64,
              "string offsets are not INT32 or INT64");
      require(c.offsets == cudf::type_id::INT64 ||
                c.chars <= static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max()),
              "chars overflow INT32 offsets");
    }
    layout.columns.push_back(c);
  }
  require(in.empty(), "trailing bytes");
  return layout;
}

std::vector<direct_buffer> plan_buffers(direct_layout const& layout, std::size_t limit)
{
  std::vector<direct_buffer> plan;
  std::size_t total = 0;
  auto const add    = [&](std::size_t column, std::size_t wire, std::size_t alloc) {
    // Counted in whole alignment units, so neither rounding alloc up nor the sum can wrap.
    auto const units = alloc / alignment + (alloc % alignment != 0);
    if (units > (limit - total) / alignment) {
      throw sirius::invalid_input_exception("direct layout: buffers exceed {} bytes", limit);
    }
    total += units * alignment;
    plan.push_back({column, wire, alloc});
  };
  auto const rows = static_cast<std::size_t>(layout.rows);
  for (std::size_t i = 0; i < layout.columns.size(); ++i) {
    auto const& c = layout.columns[i];
    if (c.has_mask) {
      add(i,
          cudf::num_bitmask_words(layout.rows) * sizeof(cudf::bitmask_type),
          cudf::bitmask_allocation_size_bytes(layout.rows));
    }
    if (is_string(c.type)) {
      auto const offsets = (rows + 1) * cudf::size_of(cudf::data_type{c.offsets});
      add(i, c.chars, c.chars);
      add(i, offsets, offsets);
    } else {
      auto const data = rows * cudf::size_of(c.type);
      add(i, data, data);
    }
  }
  return plan;
}

direct_export describe_table(cudf::table_view const& table, rmm::cuda_stream_view stream)
{
  direct_export out{{table.num_rows(), {}}, {}};
  for (auto const& c : table) {
    if (!is_sendable(c.type())) {
      throw sirius::invalid_input_exception("direct exchange cannot send type id {}",
                                            static_cast<std::int32_t>(c.type().id()));
    }
    direct_column d{c.type(), c.null_count(), c.nullable()};
    if (d.has_mask) { out.buffers.push_back(c.null_mask()); }
    out.buffers.push_back(c.head());
    if (is_string(c.type())) {
      cudf::strings_column_view const strings{c};
      d.offsets = strings.offsets().type().id();
      d.chars   = strings.chars_size(stream);
      out.buffers.push_back(strings.offsets().head());
    }
    out.layout.columns.push_back(d);
  }
  return out;
}

}  // namespace sirius::exec
