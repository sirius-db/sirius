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

#include "catch.hpp"
#include "operator/operator_test_utils.hpp"

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/copying.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>

#include <rmm/cuda_stream.hpp>
#include <rmm/device_buffer.hpp>

#include <cuda_runtime_api.h>

#include <cucascade/cudf/gpu_data_representation.hpp>
#include <cucascade/cudf/host_data_representation.hpp>
#include <cucascade/data/data_batch.hpp>
#include <cucascade/data/disk_data_representation.hpp>
#include <cucascade/data/row_range_read.hpp>
#include <cucascade/memory/common.hpp>
#include <cucascade/memory/memory_space.hpp>
#include <data/data_batch_utils.hpp>
#include <data/sirius_row_range_reader_registry.hpp>

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <stop_token>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace {

/// Starts and ends a test with no registry, then restores the registry for the rest of the process
/// if one was installed before (a live SiriusContext initialized it, and MERGE_SORT reads it).
struct registry_reset_guard {
  bool was_initialized = sirius::row_range_reader_registry::is_initialized();

  registry_reset_guard() { sirius::row_range_reader_registry::reset_for_testing(); }
  ~registry_reset_guard()
  {
    sirius::row_range_reader_registry::reset_for_testing();
    if (was_initialized) { sirius::row_range_reader_registry::initialize(); }
  }

  registry_reset_guard(const registry_reset_guard&)            = delete;
  registry_reset_guard& operator=(const registry_reset_guard&) = delete;
};

template <typename T>
std::vector<T> device_to_host(const void* device_ptr, std::size_t count)
{
  std::vector<T> host(count);
  if (count > 0) {
    REQUIRE(cudaMemcpy(host.data(), device_ptr, count * sizeof(T), cudaMemcpyDeviceToHost) ==
            cudaSuccess);
  }
  return host;
}

/// Rows of an all-valid STRING column (INT32 offsets) copied to the host; honors a sliced view.
std::vector<std::string> strings_to_host(const cudf::column_view& col)
{
  cudf::strings_column_view strings(col);
  if (strings.size() == 0) { return {}; }
  auto const offsets_view = strings.offsets();
  REQUIRE(offsets_view.type().id() == cudf::type_id::INT32);
  auto const offsets = device_to_host<int32_t>(offsets_view.data<int32_t>() + strings.offset(),
                                               static_cast<std::size_t>(strings.size()) + 1);
  auto const chars =
    device_to_host<char>(strings.parent().head<char>() + offsets.front(),
                         static_cast<std::size_t>(offsets.back() - offsets.front()));
  std::vector<std::string> rows;
  rows.reserve(static_cast<std::size_t>(strings.size()));
  for (std::size_t i = 0; i + 1 < offsets.size(); ++i) {
    rows.emplace_back(chars.data() + (offsets[i] - offsets.front()),
                      static_cast<std::size_t>(offsets[i + 1] - offsets[i]));
  }
  return rows;
}

rmm::device_buffer make_zeroed_buffer(std::size_t bytes)
{
  rmm::device_buffer buffer(bytes, cudf::get_default_stream());
  REQUIRE(cudaMemset(buffer.data(), 0, bytes) == cudaSuccess);
  return buffer;
}

cucascade::device_buffer_slot slot_of(rmm::device_buffer& buffer)
{
  return {buffer.data(), buffer.size()};
}

}  // anonymous namespace

TEST_CASE("row_range_reader_registry registers the builtin readers", "[row_range_reader_registry]")
{
  registry_reset_guard guard;
  REQUIRE_FALSE(sirius::row_range_reader_registry::is_initialized());
  REQUIRE_THROWS_AS(sirius::row_range_reader_registry::get(), std::runtime_error);

  REQUIRE(sirius::row_range_reader_registry::initialize());
  REQUIRE(sirius::row_range_reader_registry::is_initialized());

  auto& registry = sirius::row_range_reader_registry::get();
  REQUIRE(registry.has_reader<cucascade::gpu_table_representation>());
  REQUIRE(registry.has_reader<cucascade::host_data_representation>());
  REQUIRE(registry.has_reader<cucascade::disk_data_representation>());
  REQUIRE_FALSE(registry.has_reader<cucascade::host_data_packed_representation>());

  sirius::row_range_reader_registry::shutdown();
  REQUIRE_FALSE(sirius::row_range_reader_registry::is_initialized());
}

TEST_CASE("row_range_reader_registry second initialize keeps the instance",
          "[row_range_reader_registry]")
{
  registry_reset_guard guard;

  REQUIRE(sirius::row_range_reader_registry::initialize());
  auto* first = &sirius::row_range_reader_registry::get();

  REQUIRE_FALSE(sirius::row_range_reader_registry::initialize());
  REQUIRE(&sirius::row_range_reader_registry::get() == first);
  REQUIRE(first->has_reader<cucascade::gpu_table_representation>());
}

TEST_CASE("row_range_reader_registry accessors race with initialize and shutdown",
          "[row_range_reader_registry]")
{
  registry_reset_guard guard;

  std::atomic<std::size_t> hits{0};
  // jthread: a failing REQUIRE below unwinds through the vector, which stops and joins the readers.
  std::vector<std::jthread> readers;
  for (int t = 0; t < 4; ++t) {
    readers.emplace_back([&hits](std::stop_token st) {
      while (!st.stop_requested()) {
        // The address only: the registry may be shut down right after get() returns.
        try {
          if (&sirius::row_range_reader_registry::get() != nullptr) { ++hits; }
        } catch (const std::runtime_error&) {
          // Not initialized at this instant.
        }
        (void)sirius::row_range_reader_registry::is_initialized();
      }
    });
  }

  for (int i = 0; i < 200; ++i) {
    REQUIRE(sirius::row_range_reader_registry::initialize());
    REQUIRE(sirius::row_range_reader_registry::is_initialized());
    sirius::row_range_reader_registry::shutdown();
    REQUIRE_FALSE(sirius::row_range_reader_registry::is_initialized());
  }
  REQUIRE(sirius::row_range_reader_registry::initialize());
  while (hits.load() == 0) {
    std::this_thread::yield();
  }
  for (auto& reader : readers) {
    reader.request_stop();
    reader.join();
  }
  REQUIRE(sirius::row_range_reader_registry::is_initialized());
}

TEST_CASE("row_range_reader_registry reads rows of a GPU batch",
          "[row_range_reader_registry][row_range_reader_registry_e2e]")
{
  registry_reset_guard guard;
  REQUIRE(sirius::row_range_reader_registry::initialize());

  constexpr cudf::size_type num_rows  = 32;
  constexpr cudf::size_type row_begin = 5;
  constexpr std::size_t max_rows      = 10;

  auto mgr        = sirius::test::operator_utils::initialize_memory_manager();
  auto* gpu_space = mgr->get_memory_space(cucascade::memory::Tier::GPU, 0);
  auto mr         = sirius::test::operator_utils::get_resource_ref(*gpu_space);
  auto table_strm = sirius::test::operator_utils::default_stream();

  std::vector<int64_t> ints(num_rows);
  std::vector<std::string> strs(num_rows);
  for (cudf::size_type i = 0; i < num_rows; ++i) {
    ints[i] = int64_t{1000} * i - 7;
    strs[i] = "row" + std::to_string(i) + std::string(static_cast<std::size_t>(i % 5), 'x');
  }

  auto int_col = cudf::make_numeric_column(
    cudf::data_type{cudf::type_id::INT64}, num_rows, cudf::mask_state::UNALLOCATED, table_strm, mr);
  REQUIRE(cudaMemcpy(int_col->mutable_view().data<int64_t>(),
                     ints.data(),
                     ints.size() * sizeof(int64_t),
                     cudaMemcpyHostToDevice) == cudaSuccess);
  std::vector<std::unique_ptr<cudf::column>> cols;
  cols.push_back(std::move(int_col));
  cols.push_back(sirius::test::operator_utils::make_string_column(strs, table_strm, mr));
  cudf::table table(std::move(cols));

  auto batch = sirius::make_data_batch(
    std::move(table), *gpu_space, table_strm, sirius::telemetry::batch_telemetry_info{});

  // Destination slots, zeroed so that the bits and bytes the reader must write are observable.
  auto int_data    = make_zeroed_buffer(max_rows * sizeof(int64_t));
  auto int_mask    = make_zeroed_buffer(sizeof(cudf::bitmask_type));
  auto str_chars   = make_zeroed_buffer(256);
  auto str_offsets = make_zeroed_buffer((max_rows + 1) * sizeof(int32_t));
  auto str_mask    = make_zeroed_buffer(sizeof(cudf::bitmask_type));

  std::array<cucascade::column_destination, 2> dst{};
  dst[0].type_id      = static_cast<int32_t>(cudf::type_id::INT64);
  dst[0].data         = slot_of(int_data);
  dst[0].null_mask    = slot_of(int_mask);
  dst[1].type_id      = static_cast<int32_t>(cudf::type_id::STRING);
  dst[1].data         = slot_of(str_chars);
  dst[1].null_mask    = slot_of(str_mask);
  dst[1].offsets      = slot_of(str_offsets);
  dst[1].offset_width = sizeof(int32_t);

  const std::array<std::size_t, 2> columns{0, 1};
  cucascade::row_range_request request;
  request.columns   = columns;
  request.row_begin = static_cast<std::size_t>(row_begin);
  request.max_rows  = max_rows;

  rmm::cuda_stream read_stream;
  {
    auto ro           = batch->to_read_only();
    auto const result = cucascade::read_rows_into(
      sirius::row_range_reader_registry::get(), ro, request, dst, read_stream);
    read_stream.synchronize();

    auto const end_row = row_begin + static_cast<cudf::size_type>(max_rows);
    auto const expected =
      cudf::slice(ro.get_data()->cast<cucascade::gpu_table_representation>().get_table_view(),
                  {row_begin, end_row});
    REQUIRE(expected.size() == 1);
    auto const expected_ints = device_to_host<int64_t>(
      expected[0].column(0).data<int64_t>(), static_cast<std::size_t>(expected[0].num_rows()));
    auto const expected_strs = strings_to_host(expected[0].column(1));
    std::size_t expected_chars{0};
    for (auto const& s : expected_strs) {
      expected_chars += s.size();
    }

    REQUIRE(result.rows_written == max_rows);
    REQUIRE_FALSE(result.source_exhausted);
    REQUIRE(result.data_bytes_written.size() == 2);
    REQUIRE(result.data_bytes_written[0] == max_rows * sizeof(int64_t));
    REQUIRE(result.data_bytes_written[1] == expected_chars);

    REQUIRE(device_to_host<int64_t>(int_data.data(), max_rows) == expected_ints);

    // View the destination buffers as a STRING column to compare it with the slice.
    cudf::column_view offsets_view(cudf::data_type{cudf::type_id::INT32},
                                   static_cast<cudf::size_type>(max_rows + 1),
                                   str_offsets.data(),
                                   nullptr,
                                   0);
    cudf::column_view strings_view(cudf::data_type{cudf::type_id::STRING},
                                   static_cast<cudf::size_type>(max_rows),
                                   str_chars.data(),
                                   nullptr,
                                   0,
                                   0,
                                   {offsets_view});
    REQUIRE(strings_to_host(strings_view) == expected_strs);

    // The source has no nulls: exactly the first max_rows validity bits are set.
    auto const all_valid = (cudf::bitmask_type{1} << max_rows) - 1;
    REQUIRE(device_to_host<cudf::bitmask_type>(int_mask.data(), 1).front() == all_valid);
    REQUIRE(device_to_host<cudf::bitmask_type>(str_mask.data(), 1).front() == all_valid);
  }

  batch.reset();
}
