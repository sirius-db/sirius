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

// [late_mat][resolver] — reading a real pinned entry, end to end. GPU required.
//
// Everything else in this suite hand-builds its layouts and column views, which
// tests the materializer but not the one thing that has to agree with the scan
// manager: that a pinned entry's chunks, read through the resolver, describe
// the same rows the entry actually holds. So this case starts from a
// pinned_entry and a column_origin and goes all the way to values — origin,
// resolve, prepare, materialize.
//
// The pin holds row i at value i, so a materialized value IS its own global
// row id. A resolver that mis-orders the chunks, or reports a batch's rows
// wrongly, produces a column that names the rows it actually read.

#include "operator/operator_test_utils.hpp"

#include <cudf/column/column_factories.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/mr/per_device_resource.hpp>

#include <cuda_runtime.h>

#include <catch.hpp>
#include <compression/compressed_representation.hpp>
#include <late_mat/materialize.hpp>
#include <scan_manager/late_mat_resolver.hpp>
#include <scan_manager/sirius_scan_manager.hpp>

#include <cstdint>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

using sirius::late_mat::column_origin;
using sirius::late_mat::materialize;
using sirius::late_mat::pin_entry_handle;
using sirius::late_mat::prepared_selection;
using sirius::late_mat::row_id_list;
using sirius::scan_manager::pinned_entry;
using sirius::scan_manager::resolve_pinned_column;
using sirius::scan_manager::resolve_pinned_layout;

namespace {

/// A plain GPU pin of one column, chunked as asked, whose row i holds i.
struct fake_entry {
  pinned_entry entry;
  std::shared_ptr<pin_entry_handle> handle;

  fake_entry(std::vector<std::int64_t> const& batch_rows,
             std::string const& name,
             rmm::cuda_stream_view stream)
  {
    entry.tier = cucascade::memory::Tier::GPU;
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
      entry.num_rows += static_cast<std::size_t>(rows);
    }
    cudaStreamSynchronize(stream.value());
    entry.data_batches_by_column.emplace(name, std::move(chunks));

    handle = std::make_shared<pin_entry_handle>(name, 5);
    handle->set_entry(&entry);
  }

  [[nodiscard]] column_origin origin(std::uint32_t pos = 0) const
  {
    column_origin o;
    o.handle     = handle;
    o.column_pos = pos;
    o.generation = handle->generation();
    return o;
  }
};

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
  cudaMemcpyAsync(buf.data(),
                  host.data(),
                  host.size() * sizeof(std::uint64_t),
                  cudaMemcpyHostToDevice,
                  stream.value());
  cudaStreamSynchronize(stream.value());
  return buf;
}

}  // namespace

TEST_CASE("a pinned entry resolves to the layout it actually holds", "[late_mat][resolver]")
{
  auto const stream = rmm::cuda_stream_view{};
  fake_entry pin({1000, 250, 700}, "l_extendedprice", stream);

  auto const layout = resolve_pinned_layout(pin.origin());
  REQUIRE(layout.has_value());
  REQUIRE(layout->batch_rows == std::vector<std::int64_t>{1000, 250, 700});
  REQUIRE(layout->batch_row_start == std::vector<std::int64_t>{0, 1000, 1250, 1950});
  REQUIRE(layout->total_rows() == static_cast<std::int64_t>(pin.entry.num_rows));
  REQUIRE(layout->pin_generation == 5);
}

TEST_CASE("a column resolves to one source per batch, in pin order", "[late_mat][resolver]")
{
  auto const stream = rmm::cuda_stream_view{};
  fake_entry pin({64, 32}, "l_quantity", stream);

  auto const column = resolve_pinned_column(pin.origin());
  REQUIRE(column.has_value());
  REQUIRE(column->batches.size() == 2);
  REQUIRE(column->batches[0].num_rows == 64);
  REQUIRE(column->batches[1].num_rows == 32);
  REQUIRE_FALSE(column->batches[0].is_compressed());
  REQUIRE(column->dtype.id() == cudf::type_id::INT32);
}

TEST_CASE("a stale origin resolves to nothing at all", "[late_mat][resolver]")
{
  auto const stream = rmm::cuda_stream_view{};
  fake_entry pin({16}, "c_name", stream);
  auto const origin = pin.origin();
  REQUIRE(resolve_pinned_layout(origin).has_value());

  // Unpinned since the origin was captured: both halves must refuse, or the
  // materializer would read an entry that no longer exists.
  pin.handle->invalidate();
  REQUIRE_FALSE(resolve_pinned_layout(origin).has_value());
  REQUIRE_FALSE(resolve_pinned_column(origin).has_value());
}

TEST_CASE("a column position the entry does not have resolves to nothing", "[late_mat][resolver]")
{
  auto const stream = rmm::cuda_stream_view{};
  fake_entry pin({16}, "c_name", stream);
  REQUIRE_FALSE(resolve_pinned_column(pin.origin(7)).has_value());
}

TEST_CASE("a host-tier entry is not served", "[late_mat][resolver]")
{
  auto const stream = rmm::cuda_stream_view{};
  fake_entry pin({16}, "c_name", stream);
  pin.entry.tier = cucascade::memory::Tier::HOST;
  // Not an error — a reason not to defer. A host chunk would have to be staged
  // before any of this applies.
  REQUIRE_FALSE(resolve_pinned_layout(pin.origin()).has_value());
  REQUIRE_FALSE(resolve_pinned_column(pin.origin()).has_value());
}

TEST_CASE("a compressed chunk with no readable table is not deferred", "[late_mat][resolver]")
{
  auto const stream = rmm::cuda_stream_view{};
  auto mgr          = sirius::test::operator_utils::initialize_memory_manager();
  auto* gpu_space   = mgr->get_memory_space(cucascade::memory::Tier::GPU, 0);

  // A compressed chunk may carry no blob: serving paths that need only its row
  // count or a column projection never read one. Materializing does read it,
  // and this is the first code that does, so nothing before it would have
  // noticed the assumption.
  fake_entry pin({16}, "l_shipdate", stream);
  pin.entry.data_batches_by_column.clear();
  sirius::device_pin_chunk chunk;
  chunk.memory_space = gpu_space;
  chunk.compressed   = std::make_shared<sirius::compressed_device_representation>(
    *gpu_space,
    /*blob=*/nullptr,
    std::vector<std::string>{"l_shipdate"},
    /*compressed_bytes=*/64,
    /*uncompressed_bytes=*/256,
    /*num_rows=*/16);
  pin.entry.device_chunks.push_back(std::move(chunk));

  // The layout half still resolves — a row count needs no blob — which is why
  // the column half has to make the check itself.
  REQUIRE(resolve_pinned_layout(pin.origin()).has_value());
  REQUIRE_FALSE(resolve_pinned_column(pin.origin()).has_value());
}

TEST_CASE("an origin materializes the rows a join asked for, end to end", "[late_mat][resolver]")
{
  auto const stream = rmm::cuda_stream_view{};
  auto const mr     = rmm::mr::get_current_device_resource_ref();
  fake_entry pin({3000, 1500, 2000}, "l_extendedprice", stream);

  auto const layout = resolve_pinned_layout(pin.origin());
  auto const column = resolve_pinned_column(pin.origin());
  REQUIRE(layout.has_value());
  REQUIRE(column.has_value());

  // Unordered and repeated, spanning all three chunks — a join's output.
  std::vector<std::uint64_t> const ids{4200, 7, 4200, 3000, 6499, 7, 0, 2999, 4500};
  auto d_ids = upload_ids(ids, stream);

  prepared_selection const prepared(*layout,
                                    row_id_list{static_cast<std::uint64_t const*>(d_ids.data()),
                                                static_cast<std::int64_t>(ids.size()),
                                                false});
  auto const out = materialize(*column, prepared, stream, mr);
  cudaStreamSynchronize(stream.value());

  std::vector<std::int32_t> expect;
  for (auto const id : ids) {
    expect.push_back(static_cast<std::int32_t>(id));
  }
  REQUIRE(read_back(out->view()) == expect);
}
