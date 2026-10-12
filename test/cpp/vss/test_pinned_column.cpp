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
#include "operator/operator_test_utils.hpp"
#include "utils/sirius_test_env.hpp"

#include <catch.hpp>

// sirius
#include <scan_manager/sirius_scan_manager.hpp>
#include <sirius/exception.hpp>
#include <vss/pinned_column.hpp>

// cudf
#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/types.hpp>

#include <cuda_runtime_api.h>

#include <cstdint>
#include <memory>
#include <vector>

namespace {

namespace ou = sirius::test::operator_utils;
using sirius::vss::pinned_column_chunk_views;

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

}  // namespace

TEST_CASE("pinned_column_chunk_views returns borrowed chunks in pin order", "[vss]")
{
  auto* space = ou::get_default_gpu_space();

  auto c0 = make_int32_chunk({0, 1, 2});
  auto c1 = make_int32_chunk({3, 4});

  sirius::scan_manager::pinned_entry pin;
  pin.data_batches_by_column["vec"] = {c0, c1};
  pin.chunk_memory_spaces           = {space, space};

  auto views = pinned_column_chunk_views(pin, "vec", *space);

  REQUIRE(views.size() == 2);
  // Views borrow the pinned chunks (no copy): same device pointer and length, in
  // the order they were pinned.
  REQUIRE(views[0].size() == 3);
  REQUIRE(views[1].size() == 2);
  REQUIRE(views[0].data<int32_t>() == c0->view().data<int32_t>());
  REQUIRE(views[1].data<int32_t>() == c1->view().data<int32_t>());
}

TEST_CASE("pinned_column_chunk_views rejects a missing or empty column", "[vss]")
{
  auto* space = ou::get_default_gpu_space();

  sirius::scan_manager::pinned_entry pin;
  pin.data_batches_by_column["vec"]   = {make_int32_chunk({1, 2, 3})};
  pin.data_batches_by_column["empty"] = {};
  pin.chunk_memory_spaces             = {space};

  SECTION("absent column name throws")
  {
    REQUIRE_THROWS_AS(pinned_column_chunk_views(pin, "nope", *space), sirius::internal_exception);
  }

  SECTION("present but chunkless column throws")
  {
    REQUIRE_THROWS_AS(pinned_column_chunk_views(pin, "empty", *space), sirius::internal_exception);
  }
}

TEST_CASE("pinned_column_chunk_views rejects chunks on a different GPU", "[vss][multi_gpu]")
{
  if (!sirius::test::has_gpus(2)) { return; }
  auto* space = ou::get_default_gpu_space();

  static auto mgr2 = ou::initialize_memory_manager(/*n_gpus=*/2);
  auto* other      = const_cast<cucascade::memory::memory_space*>(
    mgr2->get_memory_space(cucascade::memory::Tier::GPU, 1));
  REQUIRE(other != nullptr);
  REQUIRE(other->get_device_id() != space->get_device_id());

  sirius::scan_manager::pinned_entry pin;
  pin.data_batches_by_column["vec"] = {make_int32_chunk({1, 2, 3})};
  pin.chunk_memory_spaces           = {other};  // chunk on a different device than space

  REQUIRE_THROWS_AS(pinned_column_chunk_views(pin, "vec", *space), sirius::internal_exception);
}
