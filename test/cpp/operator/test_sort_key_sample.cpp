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

#include "operator_test_utils.hpp"

#include <cudf/utilities/default_stream.hpp>

#include <catch.hpp>
#include <data/data_batch_utils.hpp>
#include <op/sort_key_sample.hpp>

#include <algorithm>
#include <numeric>
#include <vector>

using namespace sirius::op;
using namespace sirius::test::operator_utils;

namespace {

struct run {
  std::int32_t first;
  std::size_t rows;
};

std::shared_ptr<cucascade::data_batch> sorted_batch(cucascade::memory::memory_space& space,
                                                    run const& values)
{
  std::vector<std::int32_t> column(values.rows);
  std::iota(column.begin(), column.end(), values.first);
  return make_numeric_batch<std::int32_t>(space, column, cudf::type_id::INT32);
}

void add(sort_key_sample& sample,
         cucascade::data_batch& batch,
         cucascade::memory::memory_space& space)
{
  sample.add(batch.get_batch_id(),
             sirius::get_cudf_table_view(batch),
             batch.to_read_only().get_data()->get_size_in_bytes(),
             space,
             cudf::get_default_stream());
}

std::vector<std::int32_t> kept(sort_key_sample const& sample,
                               cucascade::memory::memory_space& space)
{
  auto const keys = sample.on_device(space, cudf::get_default_stream());
  std::vector<std::int32_t> values;
  for (auto const& table : keys.tables) {
    auto const part = copy_column_to_host<std::int32_t>(table.column(0));
    values.insert(values.end(), part.begin(), part.end());
  }
  CHECK(values.size() == keys.key_rows);
  std::sort(values.begin(), values.end());
  return values;
}

std::size_t smallest_stride(std::vector<run> const& runs, std::size_t max_key_rows)
{
  std::size_t stride = 1;
  auto const total   = [&] {
    std::size_t rows = 0;
    for (auto const& r : runs) {
      rows += sort_key_sample::key_row_count(r.rows, stride);
    }
    return rows;
  };
  while (total() > max_key_rows) {
    stride *= 2;
  }
  return stride;
}

std::vector<std::int32_t> rows_at(std::vector<run> const& runs, std::size_t stride)
{
  std::vector<std::int32_t> values;
  for (auto const& r : runs) {
    for (auto row = sort_key_sample::first_row(stride); row < r.rows; row += stride) {
      values.push_back(r.first + static_cast<std::int32_t>(row));
    }
  }
  std::sort(values.begin(), values.end());
  return values;
}

}  // namespace

TEST_CASE("Sort key rows kept at a stride are among those kept at smaller strides",
          "[sort_key_sample]")
{
  for (std::size_t stride = 1; stride <= 4096; stride *= 2) {
    auto const first = sort_key_sample::first_row(stride);
    CHECK(first < stride);
    CHECK(sort_key_sample::first_row(stride * 2) % stride == first);
    CHECK(sort_key_sample::key_row_count(0, stride) == 0);
    CHECK(sort_key_sample::key_row_count(first, stride) == 0);
    CHECK(sort_key_sample::key_row_count(first + 1, stride) == 1);
    CHECK(sort_key_sample::key_row_count(first + stride, stride) == 1);
    CHECK(sort_key_sample::key_row_count(first + stride + 1, stride) == 2);
  }
  CHECK_THROWS_AS(sort_key_sample(0), std::invalid_argument);
}

TEST_CASE("Sort key sample keeps every key row of a small input", "[sort_key_sample]")
{
  auto* space = get_default_gpu_space();
  REQUIRE(space != nullptr);
  std::vector<run> const runs{{0, 100}, {100, 50}};

  sort_key_sample sample(1024);
  std::size_t bytes = 0;
  for (auto const& r : runs) {
    auto batch = sorted_batch(*space, r);
    bytes += batch->to_read_only().get_data()->get_size_in_bytes();
    add(sample, *batch, *space);
  }
  CHECK(sample.stride() == 1);
  CHECK(sample.seen().batches == 2);
  CHECK(sample.seen().rows == 150);
  CHECK(sample.seen().bytes == bytes);
  CHECK(kept(sample, *space) == rows_at(runs, 1));
}

TEST_CASE("Sort key sample doubles the stride until the kept rows fit, whatever the order",
          "[sort_key_sample]")
{
  auto* space = get_default_gpu_space();
  REQUIRE(space != nullptr);
  constexpr std::size_t max_key_rows = 256;
  std::vector<run> const runs{{0, 1000}, {10000, 300}, {20000, 2500}, {30000, 3}};
  auto const stride = smallest_stride(runs, max_key_rows);
  REQUIRE(stride > 1);

  std::vector<std::shared_ptr<cucascade::data_batch>> batches;
  for (auto const& r : runs) {
    batches.push_back(sorted_batch(*space, r));
  }
  sort_key_sample forward(max_key_rows);
  for (auto const& batch : batches) {
    add(forward, *batch, *space);
  }
  sort_key_sample backward(max_key_rows);
  for (auto it = batches.rbegin(); it != batches.rend(); ++it) {
    add(backward, **it, *space);
  }

  auto const expected = rows_at(runs, stride);
  CHECK(expected.size() <= max_key_rows);
  CHECK(forward.stride() == stride);
  CHECK(backward.stride() == stride);
  CHECK(kept(forward, *space) == expected);
  CHECK(kept(backward, *space) == expected);
  CHECK(forward.on_device(*space, cudf::get_default_stream()).stride == stride);
}

TEST_CASE("Sort key sample holds at most its cap of key rows in host memory", "[sort_key_sample]")
{
  auto* space = get_default_gpu_space();
  REQUIRE(space != nullptr);
  constexpr std::size_t max_key_rows = 64;
  std::vector<run> runs;
  for (std::int32_t i = 0; i < 16; ++i) {
    runs.push_back({i * 10000, 1024});
  }

  // The first batches are taken at smaller strides than the final one.
  sort_key_sample sample(max_key_rows);
  for (auto const& r : runs) {
    auto batch = sorted_batch(*space, r);
    add(sample, *batch, *space);
    CHECK(sample.host_key_rows() <= max_key_rows);
  }

  auto const stride = smallest_stride(runs, max_key_rows);
  REQUIRE(stride > 16);
  CHECK(sample.stride() == stride);
  CHECK(kept(sample, *space) == rows_at(runs, stride));
}

TEST_CASE("Sort key sample counts a batch that is added again once", "[sort_key_sample]")
{
  auto* space = get_default_gpu_space();
  REQUIRE(space != nullptr);
  std::vector<run> const runs{{0, 500}, {1000, 700}};

  sort_key_sample sample(128);
  auto first  = sorted_batch(*space, runs[0]);
  auto second = sorted_batch(*space, runs[1]);
  add(sample, *first, *space);
  add(sample, *second, *space);
  add(sample, *first, *space);

  CHECK(sample.seen().batches == 2);
  CHECK(sample.seen().rows == 1200);
  CHECK(kept(sample, *space) == rows_at(runs, smallest_stride(runs, 128)));
}

TEST_CASE("Sort key sample keeps nothing once closed", "[sort_key_sample]")
{
  auto* space = get_default_gpu_space();
  REQUIRE(space != nullptr);
  std::vector<run> const runs{{0, 64}};

  sort_key_sample sample;
  auto first = sorted_batch(*space, runs[0]);
  add(sample, *first, *space);
  sample.close();
  auto late = sorted_batch(*space, run{100, 64});
  add(sample, *late, *space);

  CHECK(sample.seen().batches == 1);
  CHECK(kept(sample, *space) == rows_at(runs, 1));
  sample.clear();
  CHECK(sample.on_device(*space, cudf::get_default_stream()).tables.empty());
}
