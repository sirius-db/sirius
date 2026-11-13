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

// The windowed merge's whole-batch fallback: a run batch that cannot be read by row range is
// converted once, in place, to an uncompressed table in the merge task's GPU space.

#include "operator/merge/merge_test_utils.hpp"
#include "operator/operator_test_utils.hpp"

#include <cudf/column/column_factories.hpp>
#include <cudf/table/table.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <cuda_runtime.h>

#include <catch.hpp>
#include <compression/compressed_representation.hpp>
#include <cucascade/cudf/gpu_data_representation.hpp>
#include <cucascade/cudf/host_data_representation.hpp>
#include <cucascade/data/data_batch.hpp>
#include <cucascade/data/data_repository.hpp>
#include <data/convertible_data_batch.hpp>
#include <data/data_batch_utils.hpp>
#include <data/sirius_converter_registry.hpp>
#include <op/merge/detail/run_fallback.hpp>

#include <chrono>
#include <cstdint>
#include <future>
#include <memory>
#include <numeric>
#include <tuple>
#include <vector>

namespace {

namespace utils = sirius::test::operator_utils;
using sirius::op::merge::detail::convert_run_batch_to_gpu;
using sirius::test::merge_utils::has_gpu;
using sirius::test::merge_utils::holds;
using sirius::test::merge_utils::scoped_spill_compression;

constexpr std::size_t num_rows = 200'000;

rmm::cuda_stream_view stream_view() { return rmm::cuda_stream_view{utils::default_stream().get()}; }

/// The host copy of a run batch's two columns.
struct run_values {
  std::vector<std::int64_t> keys;
  std::vector<std::int32_t> payload;
  bool operator==(run_values const&) const = default;
};

template <typename T>
std::unique_ptr<cudf::column> numeric_column(cudf::type_id type,
                                             std::vector<T> const& values,
                                             cucascade::memory::memory_space& space)
{
  auto stream = utils::default_stream();
  auto column = cudf::make_fixed_width_column(cudf::data_type{type},
                                              static_cast<cudf::size_type>(values.size()),
                                              cudf::mask_state::UNALLOCATED,
                                              stream,
                                              utils::get_resource_ref(space));
  REQUIRE(cudaMemcpy(column->mutable_view().head(),
                     values.data(),
                     values.size() * sizeof(T),
                     cudaMemcpyHostToDevice) == cudaSuccess);
  return column;
}

/// A sorted INT64 key column and an INT32 payload, as a GPU batch in @p space.
std::shared_ptr<cucascade::data_batch> make_run_batch(cucascade::memory::memory_space& space,
                                                      run_values& expected)
{
  expected.keys.resize(num_rows);
  std::iota(expected.keys.begin(), expected.keys.end(), std::int64_t{1'000'000});
  expected.payload.resize(num_rows);
  for (std::size_t i = 0; i < num_rows; ++i) {
    expected.payload[i] = static_cast<std::int32_t>(i % 1000);
  }
  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(numeric_column(cudf::type_id::INT64, expected.keys, space));
  columns.push_back(numeric_column(cudf::type_id::INT32, expected.payload, space));
  return sirius::make_data_batch(std::make_unique<cudf::table>(std::move(columns)),
                                 space,
                                 utils::default_stream(),
                                 sirius::telemetry::batch_telemetry_info{});
}

/// True if @p batch is an uncompressed table in @p space.
bool is_gpu_table_in(cucascade::data_batch& batch, cucascade::memory::memory_space const& space)
{
  auto ro = batch.to_read_only();
  return ro.get_current_tier() == cucascade::memory::Tier::GPU &&
         ro.get_memory_space() != nullptr && ro.get_memory_space()->get_id() == space.get_id() &&
         dynamic_cast<cucascade::gpu_table_representation const*>(ro.get_data()) != nullptr;
}

run_values read_values(cucascade::data_batch& batch)
{
  auto stream = utils::default_stream();
  auto ro     = batch.to_read_only();
  auto view   = ro.get_data()->cast<cucascade::gpu_table_representation>().get_table_view();
  return {.keys    = utils::copy_column_to_host<std::int64_t>(view.column(0), stream),
          .payload = utils::copy_column_to_host<std::int32_t>(view.column(1), stream)};
}

}  // namespace

TEST_CASE("run fallback: a packed host batch is converted in place to the GPU", "[run_fallback]")
{
  if (!has_gpu()) { SKIP("No GPU available"); }
  utils::tiered_memory_fixture fx;
  REQUIRE(fx.setup());
  run_values expected;
  auto batch = make_run_batch(*fx.gpu0, expected);
  batch->to_mutable().convert_to<cucascade::host_data_packed_representation>(
    sirius::converter_registry::get(), fx.host0, utils::default_stream());
  REQUIRE(holds<cucascade::host_data_packed_representation>(*batch));
  auto const id = batch->get_batch_id();

  // The run stays in its repository while it is converted.
  cucascade::shared_data_repository repository;
  repository.add_data_batch(batch);

  CHECK(convert_run_batch_to_gpu(*batch, *fx.gpu0, stream_view()));

  CHECK(batch->get_batch_id() == id);
  CHECK(is_gpu_table_in(*batch, *fx.gpu0));
  CHECK(read_values(*batch) == expected);
  CHECK(repository.get_data_batch_by_id(id, 0) == batch);
  CHECK(repository.get_batch_ids(0) == std::vector<std::uint64_t>{id});
  CHECK(batch->get_state() == cucascade::batch_state::idle);
  CHECK(batch->get_read_only_count() == 0);

  // Converted once: a second call finds it on the GPU.
  CHECK_FALSE(convert_run_batch_to_gpu(*batch, *fx.gpu0, stream_view()));
}

TEST_CASE("run fallback: a spill-compressed host batch is decoded to the GPU", "[run_fallback]")
{
  if (!has_gpu()) { SKIP("No GPU available"); }
  utils::tiered_memory_fixture fx;
  REQUIRE(fx.setup());
  scoped_spill_compression compression;
  run_values expected;
  auto batch = make_run_batch(*fx.gpu0, expected);
  {
    sirius::convertible_data_batch convertible(batch);
    REQUIRE(
      convertible.convert({fx.host0}, utils::default_stream(), *fx.manager, true).has_value());
  }
  REQUIRE(holds<sirius::compressed_host_representation>(*batch));
  auto const id = batch->get_batch_id();

  cucascade::shared_data_repository repository;
  repository.add_data_batch(batch);

  CHECK(convert_run_batch_to_gpu(*batch, *fx.gpu0, stream_view()));

  CHECK(batch->get_batch_id() == id);
  CHECK(is_gpu_table_in(*batch, *fx.gpu0));
  CHECK(read_values(*batch) == expected);
  CHECK(repository.get_data_batch_by_id(id, 0) == batch);
  CHECK(batch->get_state() == cucascade::batch_state::idle);
}

TEST_CASE("run fallback: a batch already on the GPU is left alone", "[run_fallback]")
{
  if (!has_gpu()) { SKIP("No GPU available"); }
  utils::tiered_memory_fixture fx;
  REQUIRE(fx.setup());
  run_values expected;
  auto batch          = make_run_batch(*fx.gpu0, expected);
  auto const id       = batch->get_batch_id();
  auto const data_ptr = [&] {
    auto ro = batch->to_read_only();
    return std::make_pair(
      ro.get_data(),
      ro.get_data()->cast<cucascade::gpu_table_representation>().get_table_view().column(0).head());
  };
  auto const before = data_ptr();

  cucascade::shared_data_repository repository;
  repository.add_data_batch(batch);

  CHECK_FALSE(convert_run_batch_to_gpu(*batch, *fx.gpu0, stream_view()));

  // Same representation object and same device buffer: nothing was converted or copied.
  CHECK(data_ptr() == before);
  CHECK(batch->get_batch_id() == id);
  CHECK(read_values(*batch) == expected);
  CHECK(repository.get_data_batch_by_id(id, 0) == batch);
  CHECK(batch->get_state() == cucascade::batch_state::idle);
}

TEST_CASE("run fallback: the conversion waits for a reader of the batch", "[run_fallback]")
{
  if (!has_gpu()) { SKIP("No GPU available"); }
  utils::tiered_memory_fixture fx;
  REQUIRE(fx.setup());
  run_values expected;
  auto batch = make_run_batch(*fx.gpu0, expected);
  batch->to_mutable().convert_to<cucascade::host_data_packed_representation>(
    sirius::converter_registry::get(), fx.host0, utils::default_stream());

  auto reader    = batch->to_read_only();
  auto converted = std::async(
    std::launch::async, [&] { return convert_run_batch_to_gpu(*batch, *fx.gpu0, stream_view()); });
  // Blocked on the exclusive lock while the read lock is held, so the reader still sees the
  // packed host data.
  CHECK(converted.wait_for(std::chrono::milliseconds(200)) == std::future_status::timeout);
  CHECK(dynamic_cast<cucascade::host_data_packed_representation const*>(reader.get_data()) !=
        nullptr);

  std::ignore = cucascade::data_batch::to_idle(std::move(reader));
  CHECK(converted.get());
  CHECK(is_gpu_table_in(*batch, *fx.gpu0));
  CHECK(read_values(*batch) == expected);
  CHECK(batch->get_state() == cucascade::batch_state::idle);
}

TEST_CASE("run fallback: the target must be a GPU space", "[run_fallback]")
{
  if (!has_gpu()) { SKIP("No GPU available"); }
  utils::tiered_memory_fixture fx;
  REQUIRE(fx.setup());
  run_values expected;
  auto batch = make_run_batch(*fx.gpu0, expected);
  CHECK_THROWS_AS(convert_run_batch_to_gpu(*batch, *fx.host0, stream_view()),
                  std::invalid_argument);
  CHECK(is_gpu_table_in(*batch, *fx.gpu0));
  CHECK(batch->get_state() == cucascade::batch_state::idle);
}
