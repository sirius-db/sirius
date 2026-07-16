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

// test
#include <catch.hpp>

// sirius
#include <scan_manager/sirius_scan_manager.hpp>
#include <vss/vector_search_internal.hpp>

// cudf
#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/table/table.hpp>
#include <cudf/types.hpp>

#include <rmm/device_buffer.hpp>

#include <cuda_runtime_api.h>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace {

using sirius::vss::make_empty_vss_output;

std::shared_ptr<cudf::column> make_int32_chunk(std::vector<int32_t> const& values)
{
  auto col = cudf::make_numeric_column(cudf::data_type{cudf::type_id::INT32},
                                       static_cast<cudf::size_type>(values.size()),
                                       cudf::mask_state::UNALLOCATED);
  cudaMemcpy(col->mutable_view().data<int32_t>(),
             values.data(),
             sizeof(int32_t) * values.size(),
             cudaMemcpyHostToDevice);
  return std::shared_ptr<cudf::column>(std::move(col));
}

// A Sirius-style ARRAY<FLOAT>[dim] chunk (cudf LIST with a FLOAT32 child), so the
// preserved LIST type can be asserted on the empty output.
std::shared_ptr<cudf::column> make_float_list_chunk(cudf::size_type n_rows, cudf::size_type dim)
{
  auto child = cudf::make_numeric_column(
    cudf::data_type{cudf::type_id::FLOAT32}, n_rows * dim, cudf::mask_state::UNALLOCATED);
  std::vector<int32_t> offsets(n_rows + 1);
  for (cudf::size_type i = 0; i <= n_rows; ++i) {
    offsets[i] = i * dim;
  }
  auto offsets_col = cudf::make_numeric_column(
    cudf::data_type{cudf::type_id::INT32}, n_rows + 1, cudf::mask_state::UNALLOCATED);
  cudaMemcpy(offsets_col->mutable_view().data<int32_t>(),
             offsets.data(),
             sizeof(int32_t) * offsets.size(),
             cudaMemcpyHostToDevice);
  auto col = cudf::make_lists_column(
    n_rows, std::move(offsets_col), std::move(child), 0, rmm::device_buffer{});
  return std::shared_ptr<cudf::column>(std::move(col));
}

}  // namespace

TEST_CASE("make_empty_vss_output mirrors pin column types with a trailing distance", "[vss]")
{
  sirius::scan_manager::pinned_entry pin;
  pin.data_batches_by_column["id"]  = {make_int32_chunk({1, 2, 3})};
  pin.data_batches_by_column["vec"] = {make_float_list_chunk(3, 4)};

  SECTION("selected columns keep their types; FLOAT32 distance appended; zero rows")
  {
    auto out = make_empty_vss_output(pin, {"id", "vec"});
    REQUIRE(out->num_columns() == 3);
    REQUIRE(out->num_rows() == 0);
    REQUIRE(out->get_column(0).type().id() == cudf::type_id::INT32);
    REQUIRE(out->get_column(1).type().id() == cudf::type_id::LIST);
    REQUIRE(out->get_column(2).type().id() == cudf::type_id::FLOAT32);
  }

  SECTION("output column order is honored, distance always last")
  {
    auto out = make_empty_vss_output(pin, {"vec", "id"});
    REQUIRE(out->num_columns() == 3);
    REQUIRE(out->get_column(0).type().id() == cudf::type_id::LIST);
    REQUIRE(out->get_column(1).type().id() == cudf::type_id::INT32);
    REQUIRE(out->get_column(2).type().id() == cudf::type_id::FLOAT32);
  }

  SECTION("no output columns yields just the distance column")
  {
    auto out = make_empty_vss_output(pin, {});
    REQUIRE(out->num_columns() == 1);
    REQUIRE(out->num_rows() == 0);
    REQUIRE(out->get_column(0).type().id() == cudf::type_id::FLOAT32);
  }
}

TEST_CASE("make_empty_vss_output rejects a missing or empty output column", "[vss]")
{
  sirius::scan_manager::pinned_entry pin;
  pin.data_batches_by_column["id"]    = {make_int32_chunk({1, 2, 3})};
  pin.data_batches_by_column["empty"] = {};

  SECTION("absent column name throws") { REQUIRE_THROWS(make_empty_vss_output(pin, {"nope"})); }

  SECTION("present but chunkless column throws")
  {
    REQUIRE_THROWS(make_empty_vss_output(pin, {"empty"}));
  }
}
