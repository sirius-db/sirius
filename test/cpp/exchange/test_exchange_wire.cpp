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

// The exchange's wire format: headers, and how a stream maps to column buffers.

#include "catch.hpp"
#include "exchange/wire.hpp"
#include "operator/operator_test_utils.hpp"
#include "sirius/exception.hpp"

#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/table/table.hpp>

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

using namespace sirius::exchange;
using cucascade::memory::column_metadata;

namespace {

namespace utils = sirius::test::operator_utils;

constexpr std::uint64_t rows = 1000;

column_metadata int64_column(std::size_t data_offset, bool nullable = false, std::size_t mask = 0)
{
  column_metadata c{};
  c.type_id  = static_cast<std::int32_t>(cudf::type_id::INT64);
  c.num_rows = static_cast<std::int32_t>(rows);
  if (nullable) {
    c.has_null_mask    = true;
    c.null_mask_offset = mask;
    c.null_mask_size   = cudf::bitmask_allocation_size_bytes(rows);
    c.null_count       = 3;
  }
  c.has_data    = true;
  c.data_offset = data_offset;
  c.data_size   = rows * 8;
  return c;
}

column_metadata string_column(std::size_t chars_offset, std::size_t chars, std::size_t offsets_at)
{
  column_metadata c{};
  c.type_id     = static_cast<std::int32_t>(cudf::type_id::STRING);
  c.num_rows    = static_cast<std::int32_t>(rows);
  c.has_data    = chars > 0;
  c.data_offset = chars > 0 ? chars_offset : 0;
  c.data_size   = chars;
  column_metadata o{};
  o.type_id     = static_cast<std::int32_t>(cudf::type_id::INT32);
  o.num_rows    = static_cast<std::int32_t>(rows + 1);
  o.has_data    = true;
  o.data_offset = offsets_at;
  o.data_size   = (rows + 1) * 4;
  c.children.push_back(o);
  return c;
}

batch_header columns_batch()
{
  batch_header b;
  b.rows         = rows;
  b.columns      = {int64_column(256, true, 0), string_column(8448, 5000, 13568)};
  b.stream_bytes = 13568 + (rows + 1) * 4;
  return b;
}

bool same(column_metadata const& a, column_metadata const& b)
{
  if (a.type_id != b.type_id || a.num_rows != b.num_rows || a.null_count != b.null_count ||
      a.scale != b.scale || a.has_null_mask != b.has_null_mask ||
      a.null_mask_offset != b.null_mask_offset || a.null_mask_size != b.null_mask_size ||
      a.has_data != b.has_data || a.data_offset != b.data_offset || a.data_size != b.data_size ||
      a.children.size() != b.children.size()) {
    return false;
  }
  for (std::size_t i = 0; i < a.children.size(); ++i) {
    if (!same(a.children[i], b.children[i])) { return false; }
  }
  return true;
}

template <typename T>
void poke(std::vector<std::uint8_t>& bytes, std::size_t at, T value)
{
  std::memcpy(bytes.data() + at, &value, sizeof(T));
}

// Header offsets: magic 4, rows 8, stream bytes 8, then the body.
constexpr std::size_t rows_at   = 4;
constexpr std::size_t stream_at = 12;
constexpr std::size_t body_at   = 20;
// Within the first column, after the u32 column count.
constexpr std::size_t first_type_at = body_at + 4;

}  // namespace

TEST_CASE("exchange headers round-trip", "[exchange][wire]")
{
  auto const b       = columns_batch();
  auto const decoded = decode_header(encode_header(b));
  CHECK(decoded.rows == b.rows);
  CHECK(decoded.stream_bytes == b.stream_bytes);
  REQUIRE(decoded.columns.size() == 2);
  CHECK(same(decoded.columns[0], b.columns[0]));
  CHECK(same(decoded.columns[1], b.columns[1]));
}

TEST_CASE("exchange headers refuse what they cannot rebuild", "[exchange][wire]")
{
  auto const good   = encode_header(columns_batch());
  auto const refuse = [](std::vector<std::uint8_t> const& bytes, std::string const& why) {
    CHECK_THROWS_WITH(decode_header(bytes), Catch::Matchers::ContainsSubstring(why));
  };
  SECTION("magic")
  {
    auto bytes = good;
    bytes[0]   = 'X';
    refuse(bytes, "bad magic");
  }
  SECTION("truncated")
  {
    // Cut anywhere, a header is refused, mostly as truncated.
    for (std::size_t cut = 0; cut < good.size(); ++cut) {
      CHECK_THROWS_AS(decode_header({good.data(), cut}), sirius::invalid_input_exception);
    }
    refuse({good.begin(), good.begin() + 3}, "truncated");
  }
  SECTION("trailing bytes")
  {
    auto bytes = good;
    bytes.push_back(0);
    refuse(bytes, "trailing bytes");
  }
  SECTION("no rows")
  {
    auto bytes = good;
    poke<std::uint64_t>(bytes, rows_at, 0);
    refuse(bytes, "row count out of range");
  }
  SECTION("a buffer outside the stream")
  {
    auto bytes = good;
    poke<std::uint64_t>(bytes, stream_at, 13568);
    refuse(bytes, "outside the stream");
  }
  SECTION("a type the exchange cannot send")
  {
    auto bytes = good;
    poke<std::int32_t>(bytes, first_type_at, static_cast<std::int32_t>(cudf::type_id::LIST));
    refuse(bytes, "neither fixed-width nor STRING");
  }
  SECTION("a forged column count")
  {
    auto bytes = good;
    poke<std::uint32_t>(bytes, body_at, 0xffff'ffffu);
    refuse(bytes, "column count out of range");
  }
  SECTION("column rows that differ from the batch's")
  {
    auto b                = columns_batch();
    b.columns[0].num_rows = 999;
    refuse(encode_header(b), "column rows differ");
  }
  SECTION("data too small for its rows")
  {
    auto b                 = columns_batch();
    b.columns[0].data_size = rows * 8 - 1;
    refuse(encode_header(b), "data too small");
  }
  SECTION("a mask too small for its rows")
  {
    auto b                      = columns_batch();
    b.columns[0].null_mask_size = 4;
    refuse(encode_header(b), "null mask too small");
  }
  SECTION("nulls without a mask")
  {
    auto b                     = columns_batch();
    b.columns[1].null_count    = 1;
    b.columns[1].has_null_mask = false;
    refuse(encode_header(b), "nulls without a mask");
  }
  SECTION("STRING without its offsets")
  {
    auto b = columns_batch();
    b.columns[1].children.clear();
    refuse(encode_header(b), "one offsets child");
  }
  SECTION("STRING offsets of the wrong length")
  {
    auto b                            = columns_batch();
    b.columns[1].children[0].num_rows = static_cast<std::int32_t>(rows);
    refuse(encode_header(b), "rows + 1");
  }
  SECTION("columns nested too deep")
  {
    auto b = columns_batch();
    b.columns[1].children[0].children.push_back(int64_column(0));
    refuse(encode_header(b), "nest too deep");
  }
}

TEST_CASE("stream_map walks masks, data, then children", "[exchange][wire]")
{
  auto const buffers = stream_map(columns_batch().columns);
  REQUIRE(buffers.size() == 4);
  CHECK(buffers[0] == stream_buffer{0, cudf::bitmask_allocation_size_bytes(rows)});
  CHECK(buffers[1] == stream_buffer{256, rows * 8});
  CHECK(buffers[2] == stream_buffer{8448, 5000});
  CHECK(buffers[3] == stream_buffer{13568, (rows + 1) * 4});

  // An empty STRING column's chars are not a buffer.
  auto const empty = stream_map({string_column(0, 0, 256)});
  REQUIRE(empty.size() == 1);
  CHECK(empty[0] == stream_buffer{256, (rows + 1) * 4});
}

TEST_CASE("pieces_in cuts buffers at frame edges and skips padding", "[exchange][wire]")
{
  std::vector<stream_buffer> const buffers{{0, 100}, {256, 1000}, {1280, 10}};
  // A frame [0, 512): all of buffer 0, the first 256 bytes of buffer 1.
  auto const first = pieces_in(buffers, 0, 512);
  REQUIRE(first.size() == 2);
  CHECK(first[0] == stream_piece{0, 0, 0, 100});
  CHECK(first[1] == stream_piece{1, 0, 256, 256});
  // [512, 1024): the middle of buffer 1.
  auto const second = pieces_in(buffers, 512, 1024);
  REQUIRE(second.size() == 1);
  CHECK(second[0] == stream_piece{1, 256, 0, 512});
  // [1024, 1290): the rest of buffer 1, padding, buffer 2.
  auto const third = pieces_in(buffers, 1024, 1290);
  REQUIRE(third.size() == 2);
  CHECK(third[0] == stream_piece{1, 768, 0, 232});
  CHECK(third[1] == stream_piece{2, 0, 256, 10});
  CHECK(pieces_in(buffers, 100, 256).empty());
  CHECK(pieces_in(buffers, 5, 5).empty());

  // Every byte of every buffer lands in exactly one piece, whatever the frame size.
  for (std::size_t frame : {1u, 7u, 64u, 256u, 300u, 4096u}) {
    std::size_t covered = 0;
    for (std::size_t at = 0; at < 1290; at += frame) {
      for (auto const& p : pieces_in(buffers, at, std::min<std::size_t>(at + frame, 1290))) {
        covered += p.size;
      }
    }
    CHECK(covered == 1110);
  }
}

TEST_CASE("plan_stream lays a table out aligned, in walk order", "[exchange][wire]")
{
  auto stream = utils::default_stream();
  auto mr     = cudf::get_current_device_resource_ref();
  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(cudf::make_fixed_width_column(
    cudf::data_type{cudf::type_id::INT32}, 10, cudf::mask_state::ALL_VALID, stream, mr));
  columns.push_back(
    utils::make_string_column({"a", "bb", "", "dddd", "e", "", "", "ff", "g", ""}, stream, mr));
  columns.push_back(cudf::make_fixed_width_column(
    cudf::data_type{cudf::type_id::DECIMAL64, -2}, 10, cudf::mask_state::UNALLOCATED, stream, mr));
  cudf::table const table(std::move(columns));

  std::size_t bytes = 0;
  auto const meta   = plan_stream(table.view(), stream, bytes);
  REQUIRE(meta.size() == 3);
  CHECK(meta[0].has_null_mask);
  CHECK(meta[0].null_mask_offset == 0);
  CHECK(meta[0].data_offset == 256);
  CHECK(meta[0].data_size == 40);
  CHECK(meta[1].data_size == 11);
  CHECK(meta[1].children.size() == 1);
  CHECK(meta[1].children[0].num_rows == 11);
  CHECK(meta[2].scale == -2);
  CHECK_FALSE(meta[2].has_null_mask);
  for (auto const& b : stream_map(meta)) {
    CHECK(b.offset % stream_alignment == 0);
    CHECK(b.offset + b.size <= bytes);
  }
  CHECK(stream_sources(table.view(), meta).size() == stream_map(meta).size());

  auto const sliced = cudf::slice(table.view().column(0), {2, 5}).front();
  CHECK_THROWS_WITH(plan_stream(cudf::table_view{{sliced}}, stream, bytes),
                    Catch::Matchers::ContainsSubstring("sliced"));
}
