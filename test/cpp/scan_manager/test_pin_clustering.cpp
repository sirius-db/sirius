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

// Pin-time clustering: the sort that gives a pinned chunk's zone maps something to prune with.
//
// TPC-H as generated prunes 0.00% of chunks because every chunk spans the whole key range, and
// sorting the FILES does not fix it (the scan coalescer interleaves row groups). So the sort has
// to happen at pin time, per chunk. What must hold is narrow but load-bearing: every column moves
// together with the key, because the pin stores the result and every id handed out afterwards is
// positional against it.

#include "catch.hpp"
#include "pin_table.hpp"

#include <cudf/column/column_factories.hpp>
#include <cudf/table/table.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <cuda_runtime_api.h>

#include <cstdint>
#include <memory>
#include <vector>

namespace {

std::unique_ptr<cudf::column> int32_column(std::vector<std::int32_t> const& values)
{
  auto col = cudf::make_numeric_column(cudf::data_type{cudf::type_id::INT32},
                                       static_cast<cudf::size_type>(values.size()),
                                       cudf::mask_state::UNALLOCATED);
  REQUIRE(cudaMemcpy(col->mutable_view().head<std::int32_t>(),
                     values.data(),
                     values.size() * sizeof(std::int32_t),
                     cudaMemcpyHostToDevice) == cudaSuccess);
  return col;
}

std::vector<std::int32_t> read_column(cudf::column_view const& view)
{
  std::vector<std::int32_t> host(static_cast<std::size_t>(view.size()));
  if (!host.empty()) {
    REQUIRE(cudaMemcpy(host.data(),
                       view.head<std::int32_t>(),
                       host.size() * sizeof(std::int32_t),
                       cudaMemcpyDeviceToHost) == cudaSuccess);
  }
  return host;
}

std::unique_ptr<cudf::table> make_table(std::vector<std::vector<std::int32_t>> const& columns)
{
  std::vector<std::unique_ptr<cudf::column>> cols;
  cols.reserve(columns.size());
  for (auto const& values : columns) {
    cols.push_back(int32_column(values));
  }
  return std::make_unique<cudf::table>(std::move(cols));
}

std::unique_ptr<cudf::table> cluster(std::unique_ptr<cudf::table> table,
                                     std::vector<std::size_t> const& keys)
{
  return sirius::cluster_pin_chunk(
    std::move(table), keys, cudf::get_default_stream(), cudf::get_current_device_resource_ref());
}

}  // namespace

TEST_CASE("cluster_pin_chunk sorts a chunk on its key columns", "[pin_table][clustering]")
{
  SECTION("every column moves with the key")
  {
    // The payload column is the key's row index, so a payload that did not follow the gather is
    // visible as a value paired with the wrong key rather than as a wrong row count.
    auto table = make_table({{50, 10, 40, 20, 30}, {0, 1, 2, 3, 4}});
    auto out   = cluster(std::move(table), {0});

    REQUIRE(read_column(out->view().column(0)) == std::vector<std::int32_t>{10, 20, 30, 40, 50});
    REQUIRE(read_column(out->view().column(1)) == std::vector<std::int32_t>{1, 3, 4, 2, 0});
  }

  SECTION("a second key breaks ties within the first")
  {
    auto table = make_table({{2, 1, 2, 1}, {20, 30, 10, 40}});
    auto out   = cluster(std::move(table), {0, 1});

    REQUIRE(read_column(out->view().column(0)) == std::vector<std::int32_t>{1, 1, 2, 2});
    REQUIRE(read_column(out->view().column(1)) == std::vector<std::int32_t>{30, 40, 10, 20});
  }

  SECTION("no keys leaves the chunk in source order")
  {
    // The pin default. Reordering rows nobody asked to reorder would silently change what every
    // positional id downstream refers to.
    auto table = make_table({{50, 10, 40}});
    auto out   = cluster(std::move(table), {});
    REQUIRE(read_column(out->view().column(0)) == std::vector<std::int32_t>{50, 10, 40});
  }

  SECTION("a key past the chunk's width is skipped rather than fatal")
  {
    // The caller resolved key names against the pinned column list, so this means the chunk is
    // narrower than the pin declared. Pinning in source order is always correct.
    auto table = make_table({{50, 10, 40}});
    auto out   = cluster(std::move(table), {7});
    REQUIRE(read_column(out->view().column(0)) == std::vector<std::int32_t>{50, 10, 40});
  }

  SECTION("an empty chunk is returned as it came")
  {
    auto table = make_table({{}});
    auto out   = cluster(std::move(table), {0});
    REQUIRE(out->num_rows() == 0);
  }
}
