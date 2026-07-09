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

#include <catch.hpp>

// sirius
#include <scan_manager/sirius_scan_manager.hpp>
#include <vss/pinned_column.hpp>

// cudf
#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <cuda_runtime_api.h>

#include <cucascade/memory/memory_space.hpp>

#include <cstdint>
#include <memory>
#include <vector>

namespace {

std::shared_ptr<cudf::column> make_int_col(std::vector<int32_t> const& v)
{
  auto col = cudf::make_numeric_column(cudf::data_type{cudf::type_id::INT32},
                                       static_cast<cudf::size_type>(v.size()),
                                       cudf::mask_state::UNALLOCATED);
  cudaMemcpy(col->mutable_view().data<int32_t>(),
             v.data(),
             sizeof(int32_t) * v.size(),
             cudaMemcpyHostToDevice);
  return col;
}

std::vector<int32_t> to_host_i32(cudf::column_view const& col)
{
  std::vector<int32_t> host(col.size());
  cudaMemcpy(
    host.data(), col.data<int32_t>(), sizeof(int32_t) * host.size(), cudaMemcpyDeviceToHost);
  return host;
}

}  // namespace

TEST_CASE("concat_pinned_column returns a single chunk's values", "[vss]")
{
  auto& space = *sirius::test::operator_utils::get_default_gpu_space();
  auto stream = cudf::get_default_stream();

  sirius::scan_manager::pinned_entry pin;
  pin.tier                        = cucascade::memory::Tier::GPU;
  pin.data_batches_by_column["v"] = {make_int_col({10, 20, 30})};
  pin.num_rows                    = 3;

  auto out = sirius::vss::concat_pinned_column(pin, "v", space, stream);
  stream.synchronize();

  REQUIRE(out->size() == 3);
  REQUIRE(to_host_i32(out->view()) == std::vector<int32_t>{10, 20, 30});
}

TEST_CASE("concat_pinned_column concatenates chunks in pin order", "[vss]")
{
  auto& space = *sirius::test::operator_utils::get_default_gpu_space();
  auto stream = cudf::get_default_stream();

  // The pin/index-build order must be preserved: the ANN search gathers rows by
  // index into exactly this dataset, so a reordered concat would corrupt results.
  sirius::scan_manager::pinned_entry pin;
  pin.tier                        = cucascade::memory::Tier::GPU;
  pin.data_batches_by_column["v"] = {make_int_col({1, 2, 3}), make_int_col({4, 5})};
  pin.num_rows                    = 5;

  auto out = sirius::vss::concat_pinned_column(pin, "v", space, stream);
  stream.synchronize();

  REQUIRE(out->size() == 5);
  REQUIRE(to_host_i32(out->view()) == std::vector<int32_t>{1, 2, 3, 4, 5});
}

TEST_CASE("concat_pinned_column throws when the column is absent", "[vss]")
{
  auto& space = *sirius::test::operator_utils::get_default_gpu_space();
  auto stream = cudf::get_default_stream();

  sirius::scan_manager::pinned_entry pin;
  pin.tier                        = cucascade::memory::Tier::GPU;
  pin.data_batches_by_column["v"] = {make_int_col({1, 2, 3})};

  REQUIRE_THROWS(sirius::vss::concat_pinned_column(pin, "missing", space, stream));
}

// NOTE: the multi-GPU rejection branch (a chunk resident on a different device
// than `space`) needs a two-GPU fixture and is covered alongside the multi-GPU
// pin-table integration tests, not here.
