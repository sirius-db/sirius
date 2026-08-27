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

// [late_mat][materialize] — producing a deferred column. GPU required.
//
// The reference is deliberately trivial: the pinned column holds row i at value
// i, so the value a row materializes to IS its global id. A gather that reads
// the wrong batch, drops a batch's contribution, or restores the caller's order
// wrongly then shows up as a value that names the row it actually read — which
// is far easier to read in a failure than "the columns differ".
//
// The order cases carry the weight. Deferring is only invisible to the caller
// if the rows come back in the caller's own order with the caller's own
// repeats, and that path — dedup, materialize in table order, gather back — has
// three places to permute the result and no way to notice from row counts.

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/types.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/mr/per_device_resource.hpp>

#include <cuda_runtime.h>

#include <catch.hpp>
#include <late_mat/materialize.hpp>

#include <cstdint>
#include <memory>
#include <numeric>
#include <set>
#include <string>
#include <vector>

using sirius::late_mat::batch_source;
using sirius::late_mat::materialize;
using sirius::late_mat::pinned_column_view;
using sirius::late_mat::pinned_table_layout;
using sirius::late_mat::prepared_selection;
using sirius::late_mat::row_id_list;

namespace {

constexpr std::int64_t kChunk = 1024;

/// A pinned column whose row i holds the value i, split into batches.
struct fake_pin {
  std::vector<std::unique_ptr<cudf::column>> batches;
  pinned_column_view view;

  fake_pin(std::vector<std::int64_t> const& batch_rows, rmm::cuda_stream_view stream)
  {
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
      batches.push_back(std::move(col));
    }
    cudaStreamSynchronize(stream.value());

    view.dtype = cudf::data_type{cudf::type_id::INT32};
    for (std::size_t b = 0; b < batches.size(); ++b) {
      batch_source src;
      src.uncompressed = batches[b]->view();
      src.num_rows     = batch_rows[b];
      view.batches.push_back(src);
    }
  }
};

/// The same table, but as strings: row i holds the decimal text of i. Multi-batch
/// variable-width columns are what still take the canonical path, since the
/// one-pass raw gather copies a fixed element width.
struct fake_string_pin {
  std::vector<std::unique_ptr<cudf::column>> batches;
  pinned_column_view view;

  fake_string_pin(std::vector<std::int64_t> const& batch_rows, rmm::cuda_stream_view stream)
  {
    auto const mr     = rmm::mr::get_current_device_resource_ref();
    std::int64_t next = 0;
    for (auto const rows : batch_rows) {
      std::vector<std::int32_t> offsets{0};
      std::string chars;
      for (std::int64_t r = 0; r < rows; ++r) {
        chars += std::to_string(next + r);
        offsets.push_back(static_cast<std::int32_t>(chars.size()));
      }
      next += rows;

      auto offsets_col = cudf::make_numeric_column(cudf::data_type{cudf::type_id::INT32},
                                                   static_cast<cudf::size_type>(offsets.size()),
                                                   cudf::mask_state::UNALLOCATED,
                                                   stream,
                                                   mr);
      cudaMemcpyAsync(offsets_col->mutable_view().data<std::int32_t>(),
                      offsets.data(),
                      offsets.size() * sizeof(std::int32_t),
                      cudaMemcpyHostToDevice,
                      stream.value());
      rmm::device_buffer chars_buf(chars.size(), stream, mr);
      cudaMemcpyAsync(
        chars_buf.data(), chars.data(), chars.size(), cudaMemcpyHostToDevice, stream.value());
      cudaStreamSynchronize(stream.value());

      batches.push_back(cudf::make_strings_column(static_cast<cudf::size_type>(rows),
                                                  std::move(offsets_col),
                                                  std::move(chars_buf),
                                                  0,
                                                  rmm::device_buffer{0, stream, mr}));
    }

    view.dtype = cudf::data_type{cudf::type_id::STRING};
    for (std::size_t b = 0; b < batches.size(); ++b) {
      batch_source src;
      src.uncompressed = batches[b]->view();
      src.num_rows     = batch_rows[b];
      view.batches.push_back(src);
    }
  }
};

std::vector<std::string> read_back_strings(cudf::column_view const& col)
{
  cudf::strings_column_view const scv(col);
  std::vector<std::int32_t> offsets(static_cast<std::size_t>(scv.size()) + 1);
  cudaMemcpy(offsets.data(),
             scv.offsets().data<std::int32_t>() + scv.offset(),
             offsets.size() * sizeof(std::int32_t),
             cudaMemcpyDeviceToHost);
  std::string chars(static_cast<std::size_t>(offsets.back() - offsets.front()), '\0');
  if (!chars.empty()) {
    cudaMemcpy(chars.data(),
               scv.chars_begin(rmm::cuda_stream_default) + offsets.front(),
               chars.size(),
               cudaMemcpyDeviceToHost);
  }
  std::vector<std::string> out;
  out.reserve(static_cast<std::size_t>(scv.size()));
  for (std::size_t i = 0; i + 1 < offsets.size(); ++i) {
    out.push_back(chars.substr(static_cast<std::size_t>(offsets[i] - offsets.front()),
                               static_cast<std::size_t>(offsets[i + 1] - offsets[i])));
  }
  return out;
}

std::vector<std::int32_t> read_back(cudf::column_view const& col)
{
  std::vector<std::int32_t> host(static_cast<std::size_t>(col.size()));
  if (!host.empty()) {
    cudaMemcpy(host.data(),
               col.data<std::int32_t>(),
               host.size() * sizeof(std::int32_t),
               cudaMemcpyDeviceToHost);
  }
  return host;
}

rmm::device_buffer upload_ids(std::vector<std::uint64_t> const& host, rmm::cuda_stream_view stream)
{
  rmm::device_buffer buf(
    host.size() * sizeof(std::uint64_t), stream, rmm::mr::get_current_device_resource_ref());
  if (!host.empty()) {
    cudaMemcpyAsync(buf.data(),
                    host.data(),
                    host.size() * sizeof(std::uint64_t),
                    cudaMemcpyHostToDevice,
                    stream.value());
    cudaStreamSynchronize(stream.value());
  }
  return buf;
}

}  // namespace

TEST_CASE("a sorted selection materializes its rows in table order", "[late_mat][materialize]")
{
  auto const stream = rmm::cuda_stream_view{};
  auto const mr     = rmm::mr::get_current_device_resource_ref();
  std::vector<std::int64_t> const batch_rows{3 * kChunk, 2 * kChunk, 4 * kChunk};
  auto const layout = pinned_table_layout::from_batch_rows(batch_rows);
  fake_pin pin(batch_rows, stream);

  // Rows from all three batches, including the last row of the table.
  std::vector<std::uint64_t> const ids{
    0, 7, 1023, 1024, 3 * kChunk, 3 * kChunk + 1, 5 * kChunk, 9 * kChunk - 1};
  auto d_ids = upload_ids(ids, stream);

  prepared_selection const prepared(layout,
                                    row_id_list{static_cast<std::uint64_t const*>(d_ids.data()),
                                                static_cast<std::int64_t>(ids.size()),
                                                true});
  auto const column = materialize(pin.view, prepared, stream, mr);
  cudaStreamSynchronize(stream.value());

  std::vector<std::int32_t> expect;
  for (auto id : ids) {
    expect.push_back(static_cast<std::int32_t>(id));
  }
  REQUIRE(column->size() == static_cast<cudf::size_type>(ids.size()));
  REQUIRE(read_back(column->view()) == expect);
  REQUIRE_FALSE(prepared.has_canonical());
}

TEST_CASE("an unordered selection with repeats comes back in the caller's order",
          "[late_mat][materialize]")
{
  auto const stream = rmm::cuda_stream_view{};
  auto const mr     = rmm::mr::get_current_device_resource_ref();
  std::vector<std::int64_t> const batch_rows{4 * kChunk, 5 * kChunk};
  auto const layout = pinned_table_layout::from_batch_rows(batch_rows);
  fake_pin pin(batch_rows, stream);

  // Out of order, with repeats, spanning both batches — what a join hands back.
  std::vector<std::uint64_t> const ids{8000, 3, 8000, 4096, 3, 12, 8999, 4096, 0, 8999, 8999, 5000};
  auto d_ids = upload_ids(ids, stream);

  prepared_selection const prepared(layout,
                                    row_id_list{static_cast<std::uint64_t const*>(d_ids.data()),
                                                static_cast<std::int64_t>(ids.size()),
                                                false});

  auto const column = materialize(pin.view, prepared, stream, mr);
  cudaStreamSynchronize(stream.value());

  // One row per id the CALLER passed, each holding that id — repeats and all.
  std::vector<std::int32_t> expect;
  for (auto id : ids) {
    expect.push_back(static_cast<std::int32_t>(id));
  }
  REQUIRE(column->size() == static_cast<cudf::size_type>(ids.size()));
  REQUIRE(read_back(column->view()) == expect);

  // THE regression this path exists for. A gather needs neither sorted nor
  // unique ids, so canonicalizing an uncompressed column is a sort plus a
  // second gather bought for nothing.
  REQUIRE_FALSE(prepared.has_canonical());
}

TEST_CASE("a multi-batch string column takes the canonical path and still answers in order",
          "[late_mat][materialize]")
{
  auto const stream = rmm::cuda_stream_view{};
  auto const mr     = rmm::mr::get_current_device_resource_ref();
  std::vector<std::int64_t> const batch_rows{1024, 512, 700};
  auto const layout = pinned_table_layout::from_batch_rows(batch_rows);
  fake_string_pin pin(batch_rows, stream);

  // Unordered, repeated, spanning all three batches — and one batch taken whole
  // so the dense copy is exercised too.
  std::vector<std::uint64_t> ids{2000, 5, 2000, 1030, 5, 1536, 2235, 0};
  auto d_ids = upload_ids(ids, stream);

  prepared_selection const prepared(layout,
                                    row_id_list{static_cast<std::uint64_t const*>(d_ids.data()),
                                                static_cast<std::int64_t>(ids.size()),
                                                false});
  auto const column = materialize(pin.view, prepared, stream, mr);
  cudaStreamSynchronize(stream.value());

  REQUIRE(prepared.has_canonical());  // variable width, so no raw path
  std::vector<std::string> expect;
  for (auto id : ids) {
    expect.push_back(std::to_string(id));
  }
  REQUIRE(read_back_strings(column->view()) == expect);
}

TEST_CASE("a dense batch is copied whole and still lands in order", "[late_mat][materialize]")
{
  auto const stream = rmm::cuda_stream_view{};
  auto const mr     = rmm::mr::get_current_device_resource_ref();
  std::vector<std::int64_t> const batch_rows{2 * kChunk, 2 * kChunk};
  auto const layout = pinned_table_layout::from_batch_rows(batch_rows);
  fake_pin pin(batch_rows, stream);

  // All of batch 0 (dense, copied) plus two rows of batch 1 (gathered).
  std::vector<std::uint64_t> ids;
  for (std::int64_t r = 0; r < 2 * kChunk; ++r) {
    ids.push_back(static_cast<std::uint64_t>(r));
  }
  ids.push_back(static_cast<std::uint64_t>(2 * kChunk + 9));
  ids.push_back(static_cast<std::uint64_t>(4 * kChunk - 1));
  auto d_ids = upload_ids(ids, stream);

  prepared_selection const prepared(layout,
                                    row_id_list{static_cast<std::uint64_t const*>(d_ids.data()),
                                                static_cast<std::int64_t>(ids.size()),
                                                true});

  auto const column = materialize(pin.view, prepared, stream, mr);
  cudaStreamSynchronize(stream.value());

  std::vector<std::int32_t> expect;
  for (auto id : ids) {
    expect.push_back(static_cast<std::int32_t>(id));
  }
  REQUIRE(read_back(column->view()) == expect);
}

TEST_CASE("an empty selection materializes an empty column of the right type",
          "[late_mat][materialize]")
{
  auto const stream = rmm::cuda_stream_view{};
  auto const mr     = rmm::mr::get_current_device_resource_ref();
  std::vector<std::int64_t> const batch_rows{kChunk, kChunk};
  auto const layout = pinned_table_layout::from_batch_rows(batch_rows);
  fake_pin pin(batch_rows, stream);
  prepared_selection const prepared(layout, row_id_list{});

  auto const column = materialize(pin.view, prepared, stream, mr);
  REQUIRE(column->size() == 0);
  REQUIRE(column->type().id() == cudf::type_id::INT32);
}

TEST_CASE("a column whose batches disagree with the prepared layout is refused",
          "[late_mat][materialize]")
{
  auto const stream = rmm::cuda_stream_view{};
  auto const mr     = rmm::mr::get_current_device_resource_ref();
  std::vector<std::int64_t> const batch_rows{kChunk, kChunk};
  auto const layout = pinned_table_layout::from_batch_rows(batch_rows);

  // Same total rows, different split: a positional mismatch would read the
  // right NUMBER of rows out of the wrong batches, so row counts cannot catch
  // it downstream.
  fake_pin pin({2 * kChunk / 3, 2 * kChunk - 2 * kChunk / 3}, stream);

  std::vector<std::uint64_t> const ids{1, 5};
  auto d_ids = upload_ids(ids, stream);
  prepared_selection const prepared(
    layout, row_id_list{static_cast<std::uint64_t const*>(d_ids.data()), 2, true});

  REQUIRE_THROWS(materialize(pin.view, prepared, stream, mr));
}
