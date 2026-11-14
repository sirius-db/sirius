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

// A range sort computes its partition boundaries once, on one GPU, and every SORT_PARTITION task
// reads them on the GPU that holds its batch. These tests cover sort_boundaries, which gives each
// GPU its own copy, and the SORT_SAMPLE -> SORT_PARTITION hand-off across two GPUs.

#include "data/data_batch_utils.hpp"
#include "helper/type_conversions.hpp"
#include "operator_test_utils.hpp"
#include "utils/sirius_test_env.hpp"

#include <cudf/strings/strings_column_view.hpp>
#include <cudf/table/table_view.hpp>

#include <rmm/cuda_device.hpp>

#include <cuda_runtime.h>

#include <catch.hpp>
#include <duckdb/planner/expression/bound_reference_expression.hpp>
#include <op/sirius_physical_sort_partition.hpp>
#include <op/sirius_physical_sort_sample.hpp>
#include <op/sort_boundaries.hpp>
#include <sirius_config.hpp>

#include <cstdint>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

using namespace sirius::op;
using namespace sirius::test::operator_utils;
using cucascade::memory::Tier;

namespace {

std::vector<std::int32_t> read_int32(cudf::column_view const& col)
{
  std::vector<std::int32_t> host(static_cast<std::size_t>(col.size()));
  if (!host.empty()) {
    REQUIRE(cudaMemcpy(host.data(),
                       col.data<std::int32_t>(),
                       host.size() * sizeof(std::int32_t),
                       cudaMemcpyDeviceToHost) == cudaSuccess);
  }
  return host;
}

std::vector<std::string> read_strings(cudf::column_view const& col)
{
  cudf::strings_column_view const scv(col);
  std::vector<std::int32_t> offsets(static_cast<std::size_t>(scv.size()) + 1);
  REQUIRE(cudaMemcpy(offsets.data(),
                     scv.offsets().data<std::int32_t>() + scv.offset(),
                     offsets.size() * sizeof(std::int32_t),
                     cudaMemcpyDeviceToHost) == cudaSuccess);
  std::string chars(static_cast<std::size_t>(offsets.back() - offsets.front()), '\0');
  if (!chars.empty()) {
    REQUIRE(cudaMemcpy(chars.data(),
                       scv.chars_begin(::cuda::stream_ref{cudaStream_t{}}) + offsets.front(),
                       chars.size(),
                       cudaMemcpyDeviceToHost) == cudaSuccess);
  }
  std::vector<std::string> out;
  out.reserve(static_cast<std::size_t>(scv.size()));
  for (std::size_t i = 0; i + 1 < offsets.size(); ++i) {
    out.push_back(chars.substr(static_cast<std::size_t>(offsets[i] - offsets.front()),
                               static_cast<std::size_t>(offsets[i + 1] - offsets[i])));
  }
  return out;
}

int device_of(void const* pointer)
{
  cudaPointerAttributes attributes{};
  REQUIRE(cudaPointerGetAttributes(&attributes, pointer) == cudaSuccess);
  REQUIRE(attributes.type == cudaMemoryTypeDevice);
  return attributes.device;
}

// Boundary rows with a nullable integer key and a string key.
std::vector<std::int32_t> const kInts{-7, 3, 11, 42, 99};
std::vector<bool> const kValid{true, false, true, true, false};
std::vector<std::string> const kStrings{"", "apple", "banana split", "cherry", "zucchini"};

void require_rows_match(cudf::table_view const& view)
{
  REQUIRE(view.num_columns() == 2);
  REQUIRE(view.num_rows() == static_cast<cudf::size_type>(kInts.size()));
  REQUIRE(read_int32(view.column(0)) == kInts);
  REQUIRE(copy_validity_to_host(view.column(0)) == kValid);
  REQUIRE(read_strings(view.column(1)) == kStrings);
}

duckdb::BoundOrderByNode ascending_int_key()
{
  return duckdb::BoundOrderByNode(
    duckdb::OrderType::ASCENDING,
    duckdb::OrderByNullType::NULLS_LAST,
    duckdb::make_uniq<duckdb::BoundReferenceExpression>(duckdb::LogicalType::INTEGER, 0));
}

duckdb::vector<sirius::logical_type> int_types()
{
  return sirius::from_duckdb_vec(duckdb::vector<duckdb::LogicalType>{duckdb::LogicalType::INTEGER});
}

}  // namespace

TEST_CASE("sort_boundaries serves the owner GPU from the copy it was built with",
          "[sort_boundaries]")
{
  auto* space = get_default_gpu_space();
  REQUIRE(space != nullptr);

  auto ints =
    make_numeric_batch_with_nulls<std::int32_t>(*space, kInts, kValid, cudf::type_id::INT32);
  auto strings = make_string_batch(*space, kStrings);
  cudf::table_view const rows({sirius::get_cudf_table_view(*ints).column(0),
                               sirius::get_cudf_table_view(*strings).column(0)});

  auto const producer = space->acquire_stream();
  sort_boundaries boundaries(rows, *space, producer);
  REQUIRE(boundaries.num_rows() == static_cast<cudf::size_type>(kInts.size()));
  REQUIRE(boundaries.num_copies() == 1);

  // A later task on the same GPU reads from its own stream.
  auto const consumer = space->acquire_stream();
  auto const first    = boundaries.view_for(*space, consumer);
  auto const second   = boundaries.view_for(*space, consumer);
  REQUIRE(boundaries.num_copies() == 1);
  REQUIRE(first.column(0).data<std::int32_t>() == second.column(0).data<std::int32_t>());
  // The copy is the object's own, not the caller's table.
  REQUIRE(first.column(0).data<std::int32_t>() != rows.column(0).data<std::int32_t>());
  require_rows_match(first);
}

TEST_CASE("sort_boundaries gives a second GPU its own copy, once", "[sort_boundaries][multi_gpu]")
{
  if (!sirius::test::has_gpus(2)) { return; }
  auto manager = initialize_memory_manager(2);
  auto* space0 = manager->get_memory_space(Tier::GPU, 0);
  auto* space1 = manager->get_memory_space(Tier::GPU, 1);
  REQUIRE(space0 != nullptr);
  REQUIRE(space1 != nullptr);

  std::unique_ptr<sort_boundaries> boundaries;
  {
    rmm::cuda_set_device_raii const on_gpu0{rmm::cuda_device_id{0}};
    auto ints =
      make_numeric_batch_with_nulls<std::int32_t>(*space0, kInts, kValid, cudf::type_id::INT32);
    auto strings = make_string_batch(*space0, kStrings);
    cudf::table_view const rows({sirius::get_cudf_table_view(*ints).column(0),
                                 sirius::get_cudf_table_view(*strings).column(0)});
    boundaries = std::make_unique<sort_boundaries>(rows, *space0, space0->acquire_stream());
    // The source table is released here; the copies must not depend on it.
  }
  REQUIRE(boundaries->num_copies() == 1);

  {
    rmm::cuda_set_device_raii const on_gpu1{rmm::cuda_device_id{1}};
    auto const stream = space1->acquire_stream();
    auto const first  = boundaries->view_for(*space1, stream);
    REQUIRE(boundaries->num_copies() == 2);
    REQUIRE(device_of(first.column(0).data<std::int32_t>()) == 1);
    require_rows_match(first);

    auto const second = boundaries->view_for(*space1, space1->acquire_stream());
    REQUIRE(boundaries->num_copies() == 2);
    REQUIRE(first.column(0).data<std::int32_t>() == second.column(0).data<std::int32_t>());
  }

  {
    rmm::cuda_set_device_raii const on_gpu0{rmm::cuda_device_id{0}};
    auto const view = boundaries->view_for(*space0, space0->acquire_stream());
    REQUIRE(boundaries->num_copies() == 2);
    REQUIRE(device_of(view.column(0).data<std::int32_t>()) == 0);
    require_rows_match(view);
  }
}

TEST_CASE("SORT_PARTITION splits a batch on a GPU other than the one that sampled",
          "[sort_boundaries][physical_sort_partition][multi_gpu]")
{
  if (!sirius::test::has_gpus(2)) { return; }
  auto manager = initialize_memory_manager(2);
  auto* space0 = manager->get_memory_space(Tier::GPU, 0);
  auto* space1 = manager->get_memory_space(Tier::GPU, 1);
  REQUIRE(space0 != nullptr);
  REQUIRE(space1 != nullptr);

  constexpr std::size_t rows_per_batch = 50;
  duckdb::vector<duckdb::BoundOrderByNode> sample_orders;
  sample_orders.push_back(ascending_int_key());
  // Two partitions: 1000 estimated rows of the sampled width exceed 2500 bytes per partition.
  sirius_physical_sort_sample sample_op(int_types(),
                                        std::move(sample_orders),
                                        1000,
                                        1'000'000,
                                        2500,
                                        sirius::config::DEFAULT_MAX_SORT_PARTITION_MEMORY_FRACTION);

  duckdb::vector<duckdb::BoundOrderByNode> partition_orders;
  partition_orders.push_back(ascending_int_key());
  sirius_physical_sort_partition partition_op(
    int_types(), std::move(partition_orders), duckdb::vector<std::size_t>{}, 1000);
  partition_op.set_sample_op(&sample_op);

  // The boundary task runs on GPU 0 and puts the single boundary at key 50.
  {
    rmm::cuda_set_device_raii const on_gpu0{rmm::cuda_device_id{0}};
    std::vector<std::int32_t> low(rows_per_batch);
    std::iota(low.begin(), low.end(), 0);
    std::vector<std::int32_t> high(rows_per_batch);
    std::iota(high.begin(), high.end(), 50);
    auto output =
      sample_op.execute(pipelineable_operator_data(
                          {make_numeric_batch<std::int32_t>(*space0, high, cudf::type_id::INT32),
                           make_numeric_batch<std::int32_t>(*space0, low, cudf::type_id::INT32)}),
                        space0->acquire_stream());
    REQUIRE(output != nullptr);
  }
  REQUIRE(sample_op.boundaries_computed());
  REQUIRE(sample_op.get_num_partitions() == 2);
  REQUIRE(sample_op.get_num_partition_boundaries() == 1);

  // A later batch lives on GPU 1, so its SORT_PARTITION task runs there.
  rmm::cuda_set_device_raii const on_gpu1{rmm::cuda_device_id{1}};
  std::vector<std::int32_t> keys(30);
  std::iota(keys.begin(), keys.end(), 40);  // 40..69: ten rows below the boundary, twenty from it
  auto batch  = make_numeric_batch<std::int32_t>(*space1, keys, cudf::type_id::INT32);
  auto output = partition_op.execute(
    pipelineable_operator_data(std::vector<std::shared_ptr<cucascade::data_batch>>{batch}),
    space1->acquire_stream());
  REQUIRE(output != nullptr);

  auto const& slices  = dynamic_cast<sort_partition_output const&>(*output);
  auto const& batches = slices.get_data_batches();
  REQUIRE(slices.partitions == std::vector<std::size_t>{0, 1});
  REQUIRE(batches.size() == 2);

  auto const below = sirius::get_cudf_table_view(*batches[0]);
  auto const above = sirius::get_cudf_table_view(*batches[1]);
  REQUIRE(device_of(below.column(0).data<std::int32_t>()) == 1);
  REQUIRE(read_int32(below.column(0)) ==
          std::vector<std::int32_t>(keys.begin(), keys.begin() + 10));
  REQUIRE(read_int32(above.column(0)) == std::vector<std::int32_t>(keys.begin() + 10, keys.end()));
}
