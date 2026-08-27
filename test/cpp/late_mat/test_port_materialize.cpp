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

// [late_mat][port] — the far end of a deferral, against a real pin. GPU required.
//
// The batch that arrives here is what the ride produced: a UINT64 rowid where
// the first deferred column was, INT8 placeholders where the rest were, and
// every other column carrying its own values. What must come back is the table
// the reader would have seen had nothing been deferred — same rows, same order,
// values restored in place and everything else untouched.
//
// The pin holds row i at value i, so a restored value IS its own global row id:
// a splice that puts a column in the wrong position, or reads the wrong rows,
// produces a column that names what it actually read.

#include <cudf/column/column_factories.hpp>
#include <cudf/table/table.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/mr/per_device_resource.hpp>

#include <cuda_runtime.h>

#include <catch.hpp>
#include <late_mat/port_materialize.hpp>
#include <scan_manager/sirius_scan_manager.hpp>

#include <cstdint>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

using sirius::late_mat::column_origin;
using sirius::late_mat::make_defer_pair;
using sirius::late_mat::materialize_at_port;
using sirius::late_mat::pin_entry_handle;
using sirius::late_mat::port_directive_matches;
using sirius::scan_manager::pinned_entry;

namespace {

/// A two-column device pin whose row i holds value i in both columns.
struct fake_entry {
  pinned_entry entry;
  std::shared_ptr<pin_entry_handle> handle;

  fake_entry(std::vector<std::int64_t> const& batch_rows, rmm::cuda_stream_view stream)
  {
    entry.tier = cucascade::memory::Tier::GPU;
    for (auto const& name : {std::string{"c_name"}, std::string{"c_address"}}) {
      entry.cache_info.names.push_back(name);
      std::vector<std::shared_ptr<cudf::column>> chunks;
      std::int64_t next = 0;
      for (auto const rows : batch_rows) {
        std::vector<std::int32_t> host(static_cast<std::size_t>(rows));
        std::iota(host.begin(), host.end(), static_cast<std::int32_t>(next));
        next += rows;
        auto col = cudf::make_numeric_column(cudf::data_type{cudf::type_id::INT32},
                                             static_cast<cudf::size_type>(rows),
                                             cudf::mask_state::UNALLOCATED,
                                             stream);
        cudaMemcpyAsync(col->mutable_view().data<std::int32_t>(),
                        host.data(),
                        host.size() * sizeof(std::int32_t),
                        cudaMemcpyHostToDevice,
                        stream.value());
        chunks.push_back(std::shared_ptr<cudf::column>(std::move(col)));
        if (entry.cache_info.names.size() == 1) {
          entry.num_rows += static_cast<std::size_t>(rows);
        }
      }
      cudaStreamSynchronize(stream.value());
      entry.data_batches_by_column.emplace(name, std::move(chunks));
    }
    handle = std::make_shared<pin_entry_handle>("customer", 5);
    handle->set_entry(&entry);
  }

  [[nodiscard]] column_origin origin(std::uint32_t pos) const
  {
    column_origin o;
    o.handle     = handle;
    o.column_pos = pos;
    o.generation = handle->generation();
    return o;
  }
};

template <typename T>
std::unique_ptr<cudf::column> upload(std::vector<T> const& host,
                                     cudf::type_id id,
                                     rmm::cuda_stream_view stream)
{
  auto col = cudf::make_numeric_column(cudf::data_type{id},
                                       static_cast<cudf::size_type>(host.size()),
                                       cudf::mask_state::UNALLOCATED,
                                       stream);
  cudaMemcpyAsync(col->mutable_view().template data<T>(),
                  host.data(),
                  host.size() * sizeof(T),
                  cudaMemcpyHostToDevice,
                  stream.value());
  cudaStreamSynchronize(stream.value());
  return col;
}

template <typename T>
std::vector<T> read_back(cudf::column_view const& col)
{
  std::vector<T> host(static_cast<std::size_t>(col.size()));
  if (!host.empty()) {
    cudaMemcpy(host.data(), col.data<T>(), host.size() * sizeof(T), cudaMemcpyDeviceToHost);
  }
  return host;
}

/// The batch as it rides: [payload INT32, rowid UINT64, placeholder INT8].
///
/// The deferred pair sits at positions 1 and 2, so position 0 exercises the
/// splice's other half — a column that must come through untouched.
std::unique_ptr<cudf::table> riding_batch(std::vector<std::uint64_t> const& rowids,
                                          std::vector<std::int32_t> const& payload,
                                          rmm::cuda_stream_view stream)
{
  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(upload(payload, cudf::type_id::INT32, stream));
  columns.push_back(upload(rowids, cudf::type_id::UINT64, stream));
  columns.push_back(
    upload(std::vector<std::int8_t>(rowids.size(), 0), cudf::type_id::INT8, stream));
  return std::make_unique<cudf::table>(std::move(columns));
}

std::vector<cudf::data_type> riding_schema()
{
  return {cudf::data_type{cudf::type_id::INT32},
          cudf::data_type{cudf::type_id::INT32},
          cudf::data_type{cudf::type_id::INT32}};
}

}  // namespace

TEST_CASE("a rowid becomes its columns again, in the batch's own order", "[late_mat][port]")
{
  auto const stream = rmm::cuda_stream_view{};
  auto const mr     = rmm::mr::get_current_device_resource_ref();
  fake_entry pin({300, 150, 200}, stream);

  // A join's output: unordered, repeated, spanning all three chunks.
  std::vector<std::uint64_t> const rowids{420, 7, 420, 300, 649, 7, 0, 299, 450};
  std::vector<std::int32_t> payload(rowids.size());
  std::iota(payload.begin(), payload.end(), 1000);

  auto const pair = make_defer_pair(
    riding_schema(), {1, 2}, riding_schema(), {1, 2}, {pin.origin(0), pin.origin(1)});
  REQUIRE(pair.valid());

  auto const batch = riding_batch(rowids, payload, stream);
  REQUIRE(port_directive_matches(pair.port, batch->view()));

  auto const restored = materialize_at_port(pair.port, batch->view(), stream, mr);
  cudaStreamSynchronize(stream.value());
  REQUIRE(restored->num_columns() == 3);
  REQUIRE(restored->num_rows() == static_cast<cudf::size_type>(rowids.size()));

  // The pin holds row i at value i, so each restored value names the row it read.
  std::vector<std::int32_t> expect;
  for (auto const id : rowids) {
    expect.push_back(static_cast<std::int32_t>(id));
  }
  REQUIRE(read_back<std::int32_t>(restored->get_column(1).view()) == expect);
  REQUIRE(read_back<std::int32_t>(restored->get_column(2).view()) == expect);
  // The column that was never deferred is the one it always was.
  REQUIRE(read_back<std::int32_t>(restored->get_column(0).view()) == payload);
}

TEST_CASE("a batch of another shape is declined, not materialized", "[late_mat][port]")
{
  auto const stream = rmm::cuda_stream_view{};
  fake_entry pin({64}, stream);

  auto const pair = make_defer_pair(
    riding_schema(), {1, 2}, riding_schema(), {1, 2}, {pin.origin(0), pin.origin(1)});
  REQUIRE(pair.valid());

  // A UINT64 in the right place is not enough: an operator can receive batches
  // from more than one producer, and materializing against a stranger's batch
  // reads arbitrary rows of the pinned table.
  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(upload(std::vector<std::int32_t>(4, 0), cudf::type_id::INT32, stream));
  columns.push_back(upload(std::vector<std::uint64_t>(4, 0), cudf::type_id::UINT64, stream));
  columns.push_back(upload(std::vector<std::int32_t>(4, 0), cudf::type_id::INT32, stream));
  cudf::table const stranger(std::move(columns));

  REQUIRE_FALSE(port_directive_matches(pair.port, stranger.view()));
  REQUIRE_THROWS(materialize_at_port(
    pair.port, stranger.view(), stream, rmm::mr::get_current_device_resource_ref()));
}
