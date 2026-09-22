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

#include "scan/decode_test_utils.hpp"
#include "scan/dict_fsst_fixture.hpp"
#include "utils/sirius_test_env.hpp"
#include "utils/transparent_execution_test_utils.hpp"

#include <cudf/column/column.hpp>
#include <cudf/strings/strings_column_view.hpp>

#include <rmm/cuda_stream.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/mr/cuda_async_memory_resource.hpp>

#include <cuda/scan/gpu_decode_strings.cuh>

#include <catch.hpp>

namespace {

using namespace sirius::test::dict_fsst;

void check_decode(segment const& seg,
                  std::vector<std::optional<std::string>> const& expected,
                  uint32_t skip = 0)
{
  REQUIRE(seg.rows > skip);
  auto const count = seg.rows - skip;
  rmm::cuda_stream stream;
  rmm::mr::cuda_async_memory_resource mr;
  rmm::device_buffer bytes(seg.bytes.data(), seg.bytes.size(), stream.view());
  uint32_t max_length = 0;
  bool has_nulls      = false;
  for (uint64_t row = seg.start + skip; row < seg.start + seg.rows; ++row) {
    REQUIRE(row < expected.size());
    if (expected[row]) {
      max_length = std::max(max_length, static_cast<uint32_t>(expected[row]->size()));
    } else {
      has_nulls = true;
    }
  }
  sirius::cuda::scan::gpu_string_segment_desc desc{static_cast<uint8_t const*>(bytes.data()),
                                                   static_cast<uint32_t>(bytes.size()),
                                                   0,
                                                   count,
                                                   skip,
                                                   max_length};
  sirius::cuda::scan::gpu_string_column_decode_input input;
  input.total_rows = count;
  input.has_nulls  = has_nulls;
  input.data.push_back({duckdb::CompressionType::COMPRESSION_DICT_FSST, {desc}});
  rmm::device_buffer validity;
  std::vector<uint8_t> zeros;
  if (seg.all_null_validity) {
    zeros.assign((count + 7) / 8, 0);
    validity = rmm::device_buffer(zeros.data(), zeros.size(), stream.view());
    input.validity.push_back({duckdb::CompressionType::COMPRESSION_UNCOMPRESSED,
                              {{static_cast<uint8_t const*>(validity.data()),
                                static_cast<uint32_t>(validity.size()),
                                0,
                                count}}});
  }
  auto column = sirius::cuda::scan::gpu_decode_strings_column(input, stream.view(), mr);
  cudf::strings_column_view strings(column->view());
  auto offsets = sirius::test::decode::download<int32_t>(
    strings.offsets().data<int32_t>(), count + 1, stream.value());
  std::vector<uint8_t> chars;
  if (strings.chars_size(stream) != 0) {
    chars = sirius::test::decode::download<uint8_t>(
      strings.chars_begin(stream), strings.chars_size(stream), stream.value());
  }
  std::vector<cudf::bitmask_type> mask;
  if (column->nullable()) {
    mask = sirius::test::decode::download<cudf::bitmask_type>(
      column->view().null_mask(), (count + 31) / 32, stream.value());
  }
  for (uint32_t row = 0; row < count; ++row) {
    CAPTURE(seg.start, row, skip, seg.mode, seg.dict_count, seg.lengths_width);
    auto const& value = expected[seg.start + skip + row];
    bool const valid  = mask.empty() || ((mask[row / 32] >> (row % 32)) & 1u);
    REQUIRE(valid == value.has_value());
    if (!valid) { continue; }
    REQUIRE(offsets[row + 1] >= offsets[row]);
    REQUIRE(static_cast<size_t>(offsets[row + 1]) <= chars.size());
    auto actual = offsets[row] == offsets[row + 1]
                    ? std::string{}
                    : std::string(reinterpret_cast<char const*>(chars.data() + offsets[row]),
                                  offsets[row + 1] - offsets[row]);
    REQUIRE(actual == *value);
  }
}

void check_sql(fixture& disk, std::vector<std::optional<std::string>> expected)
{
  auto con = sirius::test::g_integration_env->make_connection();
  query(con, "ATTACH '" + disk.path + "' AS strings_fixture");
  struct detach {
    duckdb::Connection& con;
    ~detach() { con.Query("DETACH strings_fixture"); }
  } guard{con};
  query(con, "SET SESSION gpu_execution = true");
  auto before = sirius::test::get_transparent_execution_stats(con);
  auto result = query(con, "SELECT s FROM strings_fixture.t");
  auto after  = sirius::test::get_transparent_execution_stats(con);
  sirius::test::require_transparent_execution_delta(before, after, 1, 0, 1);
  std::vector<std::optional<std::string>> actual;
  for (duckdb::idx_t row = 0; row < result->RowCount(); ++row) {
    auto value = result->GetValue(0, row);
    actual.push_back(value.IsNull() ? std::nullopt : std::optional<std::string>(value.ToString()));
  }
  std::sort(actual.begin(), actual.end());
  std::sort(expected.begin(), expected.end());
  REQUIRE(actual == expected);
}

void verify_shape(
  size_t shape_index, uint8_t mode, bool shifted, bool multiple_segments = false, uint32_t skip = 0)
{
  auto const input = candidate_shapes().at(shape_index);
  fixture disk;
  disk.create(input);
  REQUIRE(std::filesystem::file_size(disk.path) < 5 * 1024 * 1024);
  auto segments = disk.read_segments();
  REQUIRE_FALSE(segments.empty());
  if (multiple_segments) { REQUIRE(segments.size() > 1); }
  auto expected = disk.cpu_rows();
  REQUIRE(expected.size() == input.rows);
  size_t covered = 0;
  for (auto const& seg : segments) {
    INFO("mode=" << unsigned(seg.mode) << " count=" << seg.dict_count
                 << " slw=" << unsigned(seg.lengths_width) << " diw=" << unsigned(seg.indices_width)
                 << " old=" << seg.old_indices_offset() << " new=" << seg.indices_offset());
    REQUIRE(seg.mode == mode);
    static uint32_t const counts[] = {101, 101, 0, 31, 1, 2, 31};
    static uint8_t const lengths[] = {3, 6, 6, 8, 0, 0, 8};
    static uint8_t const indices[] = {7, 7, 0, 5, 0, 1, 5};
    REQUIRE(seg.dict_count == (mode == 2 ? seg.rows + 1 : counts[shape_index]));
    REQUIRE(seg.lengths_width == lengths[shape_index]);
    REQUIRE(seg.indices_width == indices[shape_index]);
    if (mode == 2) {
      REQUIRE(seg.indices_width == 0);
    } else {
      REQUIRE((seg.old_indices_offset() != seg.indices_offset()) == shifted);
    }
    check_decode(seg, expected, skip);
    ++covered;
  }
  REQUIRE(covered == segments.size());
  check_sql(disk, std::move(expected));
}

}  // namespace

TEST_CASE("DuckDB-written DICT_FSST mode 0 decodes shifted lengths padding",
          "[integration][scan][decode][strings][dict_fsst]")
{
  verify_shape(0, 0, true);
}

TEST_CASE("DuckDB-written DICT_FSST mode 1 decodes shifted lengths padding",
          "[integration][scan][decode][strings][dict_fsst]")
{
  verify_shape(1, 1, true);
}

TEST_CASE("DuckDB-written DICT_FSST FSST_ONLY decodes without dictionary indices",
          "[integration][scan][decode][strings][dict_fsst]")
{
  verify_shape(2, 2, false);
}

TEST_CASE("DuckDB-written DICT_FSST non-shifting alignment boundary stays correct",
          "[integration][scan][decode][strings][dict_fsst]")
{
  verify_shape(3, 0, false);
}

TEST_CASE("DuckDB-written DICT_FSST all-NULL dictionary preserves nullness",
          "[integration][scan][decode][strings][dict_fsst]")
{
  verify_shape(4, 0, false);
}

TEST_CASE("DuckDB-written DICT_FSST zero-width lengths distinguish NULL and empty",
          "[integration][scan][decode][strings][dict_fsst]")
{
  verify_shape(5, 0, false);
}

TEST_CASE("DuckDB-written DICT_FSST spans multiple segments",
          "[integration][scan][decode][strings][dict_fsst]")
{
  verify_shape(6, 0, false, true);
}

TEST_CASE("DuckDB-written DICT_FSST decodes a slice after the segment head",
          "[integration][scan][decode][strings][dict_fsst]")
{
  verify_shape(3, 0, false, false, 33);
}
