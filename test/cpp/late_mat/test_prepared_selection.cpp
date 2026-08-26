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

// [late_mat][prepare] — resolving one selection against a table's batch layout.
// GPU required.
//
// The invariant under test is that the ids the caller passed and the rows the
// batches will produce are the same set, in the order the output promises.
// Every step between them can lose that quietly: the split can drop an id off
// the end of a batch, the localization can attribute it to the wrong batch, and
// the CSR can bucket it into the wrong chunk. None of those throws — they all
// produce a well-formed selection of the wrong rows. So each case reconstructs
// the global ids back out of the prepared per-batch state and requires the
// caller's own set in return.

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/mr/per_device_resource.hpp>

#include <cuda_runtime.h>

#include <catch.hpp>
#include <late_mat/prepared_selection.hpp>

#include <algorithm>
#include <cstdint>
#include <set>
#include <vector>

using sirius::late_mat::canonical_selection;
using sirius::late_mat::pinned_table_layout;
using sirius::late_mat::prepared_selection;
using sirius::late_mat::row_id_list;

namespace {

constexpr std::int64_t kChunk = 1024;

rmm::device_buffer upload(std::vector<std::uint64_t> const& host, rmm::cuda_stream_view stream)
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

template <typename T>
std::vector<T> download(void const* device, std::size_t count)
{
  std::vector<T> host(count);
  if (count != 0) { cudaMemcpy(host.data(), device, count * sizeof(T), cudaMemcpyDeviceToHost); }
  return host;
}

/// Every global id the prepared selection would materialize, in output order.
std::vector<std::int64_t> rebuild_global_ids(canonical_selection const& canonical,
                                             pinned_table_layout const& layout)
{
  std::vector<std::int64_t> ids;
  for (std::size_t b = 0; b < canonical.batches.size(); ++b) {
    auto const& batch = canonical.batches[b];
    auto const start  = layout.batch_row_start[b];
    if (batch.survivors == 0) { continue; }
    if (batch.dense) {
      for (std::int64_t r = 0; r < layout.batch_rows[b]; ++r) {
        ids.push_back(start + r);
      }
      continue;
    }
    auto const view = batch.rows.view();
    auto const chunks =
      download<std::uint32_t>(view.chunk_ids, static_cast<std::size_t>(view.num_touched));
    auto const offs =
      download<std::uint32_t>(view.block_offsets, static_cast<std::size_t>(view.num_touched) + 1);
    auto const rows =
      download<std::uint16_t>(view.in_chunk_rows, static_cast<std::size_t>(view.num_survivors));
    for (std::int64_t blk = 0; blk < view.num_touched; ++blk) {
      for (std::uint32_t k = offs[blk]; k < offs[blk + 1]; ++k) {
        ids.push_back(start + static_cast<std::int64_t>(chunks[blk]) * kChunk + rows[k]);
      }
    }
  }
  return ids;
}

}  // namespace

TEST_CASE("a layout derives its row starts from its batches", "[late_mat][prepare]")
{
  auto const layout = pinned_table_layout::from_batch_rows({100, 250, 7});
  REQUIRE(layout.num_batches() == 3);
  REQUIRE(layout.batch_row_start == std::vector<std::int64_t>{0, 100, 350, 357});
  REQUIRE(layout.total_rows() == 357);
}

TEST_CASE("an unordered selection with repeats prepares to its own id set", "[late_mat][prepare]")
{
  auto const stream = rmm::cuda_stream_view{};
  auto const mr     = rmm::mr::get_current_device_resource_ref();
  auto const layout = pinned_table_layout::from_batch_rows({4 * kChunk, 7 * kChunk, 2 * kChunk});

  // Deliberately out of order, with repeats, and touching all three batches.
  std::vector<std::uint64_t> host_ids;
  std::uint64_t x = 99;
  for (int i = 0; i < 2000; ++i) {
    x = x * 6364136223846793005ULL + 1442695040888963407ULL;
    host_ids.push_back((x >> 33) % static_cast<std::uint64_t>(layout.total_rows()));
  }
  for (std::size_t i = 0; i + 1 < host_ids.size(); i += 5) {
    host_ids[i + 1] = host_ids[i];
  }
  auto d_ids = upload(host_ids, stream);

  prepared_selection const prepared(layout,
                                    row_id_list{static_cast<std::uint64_t const*>(d_ids.data()),
                                                static_cast<std::int64_t>(host_ids.size()),
                                                false});
  auto const& canonical = prepared.canonical(stream, mr);
  cudaStreamSynchronize(stream.value());

  std::set<std::uint64_t> const distinct(host_ids.begin(), host_ids.end());
  REQUIRE(prepared.original_count() == static_cast<std::int64_t>(host_ids.size()));
  REQUIRE(canonical.total_survivors == static_cast<std::int64_t>(distinct.size()));
  REQUIRE(canonical.out_base.back() == canonical.total_survivors);
  REQUIRE(canonical.needs_restore());

  std::vector<std::int64_t> const expect(distinct.begin(), distinct.end());
  REQUIRE(rebuild_global_ids(canonical, layout) == expect);

  // Every caller id must point at its own row of the output — that is what
  // makes deduplicating invisible to the caller.
  auto const ranks = download<std::int32_t>(canonical.restore_rank.data(), host_ids.size());
  for (std::size_t i = 0; i < host_ids.size(); ++i) {
    REQUIRE(ranks[i] >= 0);
    REQUIRE(ranks[i] < canonical.total_survivors);
    REQUIRE(expect[static_cast<std::size_t>(ranks[i])] == static_cast<std::int64_t>(host_ids[i]));
  }
}

TEST_CASE("a batch whose rows all survive is marked dense and prepares nothing",
          "[late_mat][prepare]")
{
  auto const stream = rmm::cuda_stream_view{};
  auto const mr     = rmm::mr::get_current_device_resource_ref();
  auto const layout = pinned_table_layout::from_batch_rows({2 * kChunk, 3 * kChunk});

  // All of batch 0, and one row of batch 1.
  std::vector<std::uint64_t> host_ids;
  for (std::int64_t r = 0; r < 2 * kChunk; ++r) {
    host_ids.push_back(static_cast<std::uint64_t>(r));
  }
  host_ids.push_back(static_cast<std::uint64_t>(2 * kChunk + 5));
  auto d_ids = upload(host_ids, stream);

  prepared_selection const prepared(layout,
                                    row_id_list{static_cast<std::uint64_t const*>(d_ids.data()),
                                                static_cast<std::int64_t>(host_ids.size()),
                                                true});
  auto const& canonical = prepared.canonical(stream, mr);
  cudaStreamSynchronize(stream.value());

  REQUIRE(canonical.batches[0].dense);
  REQUIRE(canonical.batches[0].survivors == 2 * kChunk);
  REQUIRE(canonical.batches[0].rows.num_survivors == 0);  // no CSR built
  REQUIRE(canonical.batches[0].local_indices.size() == 0);
  REQUIRE(canonical.batches[0].density == Approx(1.0));

  REQUIRE_FALSE(canonical.batches[1].dense);
  REQUIRE(canonical.batches[1].survivors == 1);
  REQUIRE(canonical.batches[1].rows.num_touched == 1);

  REQUIRE_FALSE(canonical.needs_restore());  // the caller promised sorted + unique
  REQUIRE(canonical.out_base == std::vector<std::int64_t>{0, 2 * kChunk, 2 * kChunk + 1});

  std::vector<std::int64_t> expect(host_ids.begin(), host_ids.end());
  REQUIRE(rebuild_global_ids(canonical, layout) == expect);
}

TEST_CASE("an empty selection prepares to nothing, per batch", "[late_mat][prepare]")
{
  auto const stream = rmm::cuda_stream_view{};
  auto const mr     = rmm::mr::get_current_device_resource_ref();
  auto const layout = pinned_table_layout::from_batch_rows({kChunk, kChunk});
  prepared_selection const prepared(layout, row_id_list{});
  auto const& canonical = prepared.canonical(stream, mr);

  REQUIRE(canonical.total_survivors == 0);
  REQUIRE(canonical.batches.size() == 2);
  REQUIRE(canonical.out_base == std::vector<std::int64_t>{0, 0, 0});
  REQUIRE_FALSE(canonical.needs_restore());
}

TEST_CASE("a batch nothing selected still holds its place", "[late_mat][prepare]")
{
  auto const stream = rmm::cuda_stream_view{};
  auto const mr     = rmm::mr::get_current_device_resource_ref();
  auto const layout = pinned_table_layout::from_batch_rows({kChunk, kChunk, kChunk});

  // Batches 0 and 2 only: the middle one must stay in place, or every later
  // batch's output base is off by its survivors.
  std::vector<std::uint64_t> const host_ids{3, 7, 2 * kChunk + 1};
  auto d_ids = upload(host_ids, stream);

  prepared_selection const prepared(
    layout, row_id_list{static_cast<std::uint64_t const*>(d_ids.data()), 3, true});
  auto const& canonical = prepared.canonical(stream, mr);
  cudaStreamSynchronize(stream.value());

  REQUIRE(canonical.batches[1].survivors == 0);
  REQUIRE_FALSE(canonical.batches[1].dense);
  REQUIRE(canonical.out_base == std::vector<std::int64_t>{0, 2, 2, 3});
  REQUIRE(rebuild_global_ids(canonical, layout) == std::vector<std::int64_t>{3, 7, 2 * kChunk + 1});
}

TEST_CASE("constructing a prepared selection does no device work", "[late_mat][prepare]")
{
  auto const stream = rmm::cuda_stream_view{};
  auto const mr     = rmm::mr::get_current_device_resource_ref();
  auto const layout = pinned_table_layout::from_batch_rows({kChunk, kChunk});
  std::vector<std::uint64_t> const host_ids{9, 3, 9, 1500};
  auto d_ids = upload(host_ids, stream);

  // The canonical form is what costs a sort and a host sync, so a consumer that
  // does not need one must not have paid for it just by preparing.
  prepared_selection const prepared(
    layout, row_id_list{static_cast<std::uint64_t const*>(d_ids.data()), 4, false});
  REQUIRE_FALSE(prepared.has_canonical());
  REQUIRE(prepared.original_count() == 4);

  auto const& first = prepared.canonical(stream, mr);
  REQUIRE(prepared.has_canonical());
  // Built once and shared: a second ask is the same object, not a second sort.
  REQUIRE(&first == &prepared.canonical(stream, mr));
}

TEST_CASE("an id past the end of the pinned table is refused", "[late_mat][prepare]")
{
  auto const stream = rmm::cuda_stream_view{};
  auto const mr     = rmm::mr::get_current_device_resource_ref();
  auto const layout = pinned_table_layout::from_batch_rows({kChunk, kChunk});

  // Such an id belongs to no batch, so the split would simply not place it and
  // the row would vanish from the output rather than fail.
  std::vector<std::uint64_t> const host_ids{5, static_cast<std::uint64_t>(2 * kChunk)};
  auto d_ids = upload(host_ids, stream);

  prepared_selection const prepared(
    layout, row_id_list{static_cast<std::uint64_t const*>(d_ids.data()), 2, true});
  REQUIRE_THROWS(prepared.canonical(stream, mr));
}
