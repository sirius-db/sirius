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

#include "op/scan/detail/read_retirement.hpp"
#include "op/scan/owning_table_view.hpp"
#include "sirius/exception.hpp"
#include "utils/host_allocation_fault.hpp"

#include <cudf/column/column_factories.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_stream.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/error.hpp>

#include <cuda_runtime_api.h>

#include <catch.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <memory>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

using namespace sirius::op::scan;
using sirius::op::scan::detail::read_checkpoint;
using sirius::op::scan::detail::read_retirement_test_hooks;

namespace {

::cuda::stream_ref test_stream() { return cudf::get_default_stream(); }
rmm::device_async_resource_ref test_mr() { return cudf::get_current_device_resource_ref(); }

// Build an INT32 column whose single value tags the column for identification.
std::unique_ptr<cudf::column> tagged_column(std::int32_t tag)
{
  auto col = cudf::make_numeric_column(
    cudf::data_type{cudf::type_id::INT32}, 1, cudf::mask_state::UNALLOCATED, test_stream());
  cudaMemcpy(col->mutable_view().data<std::int32_t>(), &tag, sizeof(tag), cudaMemcpyHostToDevice);
  return col;
}

// Build a table with one INT32 column per tag.
std::unique_ptr<cudf::table> tagged_table(std::vector<std::int32_t> const& tags)
{
  std::vector<std::unique_ptr<cudf::column>> cols;
  cols.reserve(tags.size());
  for (auto t : tags) {
    cols.push_back(tagged_column(t));
  }
  return std::make_unique<cudf::table>(std::move(cols));
}

// Read back the single tag value of each column of a view.
std::vector<std::int32_t> read_tags(cudf::table_view view)
{
  std::vector<std::int32_t> out(static_cast<std::size_t>(view.num_columns()));
  for (cudf::size_type i = 0; i < view.num_columns(); ++i) {
    cudaMemcpy(&out[static_cast<std::size_t>(i)],
               view.column(i).data<std::int32_t>(),
               sizeof(std::int32_t),
               cudaMemcpyDeviceToHost);
  }
  return out;
}

// The device data pointers of each column of a view (column identity).
std::vector<void const*> data_ptrs(cudf::table_view view)
{
  std::vector<void const*> out(static_cast<std::size_t>(view.num_columns()));
  for (cudf::size_type i = 0; i < view.num_columns(); ++i) {
    out[static_cast<std::size_t>(i)] = view.column(i).head();
  }
  return out;
}

// A keep-alive owner that does NOT satisfy no_alloc_materializable (no
// dereferenceable release()), forcing the copying materialization path.
struct table_keepalive {
  std::unique_ptr<cudf::table> table;
};

}  // namespace

TEST_CASE("owning_table_view empty state", "[owning_table_view]")
{
  owning_table_view handle;
  REQUIRE_FALSE(static_cast<bool>(handle));
  REQUIRE(handle.n_columns() == 0);
  REQUIRE(handle.view().num_columns() == 0);

  // Mutating an empty handle is a no-op and leaves it empty.
  std::array<std::size_t, 1> drop{0};
  handle.drop_columns(drop);
  REQUIRE_FALSE(static_cast<bool>(handle));

  // release() on an empty handle yields nullptr.
  REQUIRE(handle.release(test_stream(), test_mr()) == nullptr);
}

TEST_CASE("owning_table_view from owned table exposes the table", "[owning_table_view]")
{
  owning_table_view handle{tagged_table({10, 20, 30})};
  REQUIRE(static_cast<bool>(handle));
  REQUIRE(handle.n_columns() == 3);
  REQUIRE(read_tags(handle.view()) == std::vector<std::int32_t>{10, 20, 30});
}

TEST_CASE("owning_table_view num_rows / column / column_types accessors", "[owning_table_view]")
{
  // 4 columns, 1 row each.
  owning_table_view handle{tagged_table({10, 20, 30, 40})};

  REQUIRE(handle.num_rows() == 1);
  REQUIRE(handle.column_types().size() == 4);
  for (auto const& t : handle.column_types()) {
    REQUIRE(t.id() == cudf::type_id::INT32);
  }

  // column() returns the column at the current position; row count is
  // unaffected by dropping columns.
  std::array<std::size_t, 1> drop{0};
  handle.drop_columns(drop);  // -> 20, 30, 40
  REQUIRE(handle.num_rows() == 1);
  REQUIRE(handle.column_types().size() == 3);

  std::int32_t tag = 0;
  cudaMemcpy(&tag, handle.column(0).data<std::int32_t>(), sizeof(tag), cudaMemcpyDeviceToHost);
  REQUIRE(tag == 20);

  REQUIRE_THROWS_AS(handle.column(3), std::out_of_range);

  // Empty handle: zero rows, no columns.
  owning_table_view empty;
  REQUIRE(empty.num_rows() == 0);
  REQUIRE(empty.column_types().empty());
  REQUIRE_THROWS_AS(empty.column(0), std::out_of_range);
}

TEST_CASE("owning_table_view drop + materialize moves buffers (no copy)", "[owning_table_view]")
{
  auto table = tagged_table({10, 20, 30});
  // Capture the original device pointers before handing the table over.
  auto original_ptrs = data_ptrs(table->view());

  owning_table_view handle{std::move(table)};

  // Drop the middle column — pure view manipulation, no allocation.
  std::array<std::size_t, 1> drop{1};
  handle.drop_columns(drop);
  REQUIRE(handle.n_columns() == 2);
  REQUIRE(read_tags(handle.view()) == std::vector<std::int32_t>{10, 30});

  // Materialize back to an owned table: surviving columns keep their buffers.
  auto result = handle.release(test_stream(), test_mr());
  REQUIRE(result != nullptr);
  REQUIRE(read_tags(result->view()) == std::vector<std::int32_t>{10, 30});

  auto result_ptrs = data_ptrs(result->view());
  REQUIRE(result_ptrs[0] == original_ptrs[0]);  // column 10 buffer preserved
  REQUIRE(result_ptrs[1] == original_ptrs[2]);  // column 30 buffer preserved

  // Handle is now empty.
  REQUIRE_FALSE(static_cast<bool>(handle));
  REQUIRE(handle.release(test_stream(), test_mr()) == nullptr);
}

TEST_CASE("owning_table_view reorder swaps columns", "[owning_table_view]")
{
  owning_table_view handle{tagged_table({10, 20, 30})};

  // Swap positions 0 and 2.
  std::array<std::pair<std::size_t, std::size_t>, 1> swaps{{{0, 2}}};
  handle.reorder_columns(swaps);
  REQUIRE(read_tags(handle.view()) == std::vector<std::int32_t>{30, 20, 10});

  auto result = handle.release(test_stream(), test_mr());
  REQUIRE(read_tags(result->view()) == std::vector<std::int32_t>{30, 20, 10});
}

TEST_CASE("owning_table_view reorder then drop compose on current positions", "[owning_table_view]")
{
  owning_table_view handle{tagged_table({10, 20, 30, 40})};

  std::array<std::pair<std::size_t, std::size_t>, 1> swaps{{{0, 3}}};
  handle.reorder_columns(swaps);  // -> 40, 20, 30, 10
  std::array<std::size_t, 2> drop{1, 2};
  handle.drop_columns(drop);  // drop current positions 1 and 2 -> 40, 10
  REQUIRE(handle.n_columns() == 2);
  REQUIRE(read_tags(handle.view()) == std::vector<std::int32_t>{40, 10});
}

TEST_CASE("owning_table_view select_columns projects and reorders (no copy)", "[owning_table_view]")
{
  auto table         = tagged_table({10, 20, 30, 40});
  auto original_ptrs = data_ptrs(table->view());

  owning_table_view handle{std::move(table)};

  // Project down to columns 3 and 0, in that order.
  std::array<std::size_t, 2> keep{3, 0};
  handle.select_columns(keep);
  REQUIRE(handle.n_columns() == 2);
  REQUIRE(read_tags(handle.view()) == std::vector<std::int32_t>{40, 10});

  // select_columns composes on current positions: now keep only position 1 (10).
  std::array<std::size_t, 1> keep2{1};
  handle.select_columns(keep2);
  REQUIRE(read_tags(handle.view()) == std::vector<std::int32_t>{10});

  // Zero-alloc: the surviving buffer is the original column 10 buffer.
  auto result = handle.release(test_stream(), test_mr());
  REQUIRE(data_ptrs(result->view())[0] == original_ptrs[0]);
}

TEST_CASE("owning_table_view select_columns rejects bad positions", "[owning_table_view]")
{
  owning_table_view handle{tagged_table({10, 20, 30})};

  std::array<std::size_t, 2> out_of_range{0, 3};
  REQUIRE_THROWS_AS(handle.select_columns(out_of_range), std::out_of_range);

  std::array<std::size_t, 2> duplicate{1, 1};
  REQUIRE_THROWS_AS(handle.select_columns(duplicate), std::invalid_argument);

  // Unchanged after rejected operations.
  REQUIRE(handle.n_columns() == 3);
  REQUIRE(read_tags(handle.view()) == std::vector<std::int32_t>{10, 20, 30});
}

TEST_CASE("owning_table_view rejects out-of-range positions", "[owning_table_view]")
{
  owning_table_view handle{tagged_table({10, 20, 30})};

  std::array<std::size_t, 1> bad_drop{3};
  REQUIRE_THROWS_AS(handle.drop_columns(bad_drop), std::out_of_range);

  std::array<std::pair<std::size_t, std::size_t>, 1> bad_swap{{{0, 3}}};
  REQUIRE_THROWS_AS(handle.reorder_columns(bad_swap), std::out_of_range);

  // The handle is unchanged after the rejected operations.
  REQUIRE(handle.n_columns() == 3);
  REQUIRE(read_tags(handle.view()) == std::vector<std::int32_t>{10, 20, 30});
}

TEST_CASE("owning_table_view generic owner materializes by copy", "[owning_table_view]")
{
  auto table         = tagged_table({10, 20, 30});
  auto original_ptrs = data_ptrs(table->view());
  auto base_view     = table->view();

  // Type-erase a non-no_alloc owner; materialization must copy.
  owning_table_view handle{table_keepalive{std::move(table)}, base_view};
  REQUIRE(handle.n_columns() == 3);

  std::array<std::size_t, 1> drop{1};
  handle.drop_columns(drop);
  REQUIRE(read_tags(handle.view()) == std::vector<std::int32_t>{10, 30});

  auto result = handle.release(test_stream(), test_mr());
  REQUIRE(read_tags(result->view()) == std::vector<std::int32_t>{10, 30});

  // Copying path: the materialized buffers differ from the originals.
  auto result_ptrs = data_ptrs(result->view());
  REQUIRE(result_ptrs[0] != original_ptrs[0]);
  REQUIRE(result_ptrs[1] != original_ptrs[2]);
}

// Stream-ordering regression: the copying materialization path must not let the
// caller reclaim the source memory while the copy is still in flight.
namespace {

constexpr std::int32_t kSourceByte  = 0x01;  // source fill  -> 0x01010101
constexpr std::int32_t kPoisonByte  = 0xFF;  // reclaim fill -> 0xFFFFFFFF
constexpr std::int32_t kSourceValue = 0x01010101;
constexpr std::int32_t kPoisonValue = -1;

// Observations recorded at the moment the owner is destroyed.
struct reclaim_probe {
  cudaStream_t work_stream{};
  cudaStream_t reclaim_stream{};
  void* source_data{};
  std::size_t source_bytes{};
  cudaError_t stream_status_at_reclaim{cudaSuccess};
  bool reclaimed{false};
};

// Stands in for the pinned-cache lock: while this owner lives the source memory
// is valid, and destroying it lets the cache overwrite that memory. Move-only so
// only the final owner performs the reclaim.
struct reclaiming_owner {
  std::shared_ptr<reclaim_probe> probe;
  std::unique_ptr<cudf::table> source;

  reclaiming_owner(std::shared_ptr<reclaim_probe> p, std::unique_ptr<cudf::table> s)
    : probe(std::move(p)), source(std::move(s))
  {
  }
  reclaiming_owner(reclaiming_owner&&)                 = default;
  reclaiming_owner& operator=(reclaiming_owner&&)      = default;
  reclaiming_owner(reclaiming_owner const&)            = delete;
  reclaiming_owner& operator=(reclaiming_owner const&) = delete;

  ~reclaiming_owner()
  {
    if (probe == nullptr || source == nullptr) { return; }  // moved-from
    probe->reclaimed                = true;
    probe->stream_status_at_reclaim = cudaStreamQuery(probe->work_stream);
    // Simulate the cache reusing the chunk the instant the lock drops.
    cudaMemsetAsync(probe->source_data, kPoisonByte, probe->source_bytes, probe->reclaim_stream);
    cudaStreamSynchronize(probe->reclaim_stream);
  }
};

static_assert(!std::copy_constructible<reclaiming_owner>);
static_assert(!detail::no_alloc_materializable<reclaiming_owner>);

owning_table_view reclaimable_table(std::shared_ptr<reclaim_probe> const& probe,
                                    ::cuda::stream_ref work_stream,
                                    ::cuda::stream_ref reclaim_stream)
{
  auto table            = tagged_table({kSourceValue, kSourceValue});
  auto view             = table->view();
  probe->work_stream    = work_stream.get();
  probe->reclaim_stream = reclaim_stream.get();
  probe->source_data    = table->get_column(0).mutable_view().data<std::int32_t>();
  probe->source_bytes   = sizeof(std::int32_t);
  return owning_table_view{reclaiming_owner{probe, std::move(table)}, view};
}

struct copy_resource_state {
  rmm::device_async_resource_ref upstream{test_mr()};
  std::size_t allocation_calls{};
  std::size_t live_allocations{};
  std::size_t fail_at{};
  std::atomic_bool failure_work_completed{false};
  cudaError_t submission_status{cudaSuccess};
  std::exception_ptr failure = std::make_exception_ptr(rmm::out_of_memory{"table copy allocation"});
  std::optional<sirius::test::scoped_host_allocation_fault>* cleanup_host_fault{};
};

struct copy_resource {
  std::shared_ptr<copy_resource_state> state = std::make_shared<copy_resource_state>();

  void* allocate_sync(std::size_t bytes, std::size_t alignment)
  {
    return state->upstream.allocate_sync(bytes, alignment);
  }

  void deallocate_sync(void* pointer, std::size_t bytes, std::size_t alignment) noexcept
  {
    state->upstream.deallocate_sync(pointer, bytes, alignment);
  }

  void* allocate(::cuda::stream_ref stream, std::size_t bytes, std::size_t alignment)
  {
    if (++state->allocation_calls == state->fail_at) {
      // This marker follows the first column's copy in the table constructor.
      state->submission_status = cudaLaunchHostFunc(
        stream.get(),
        [](void* done) { static_cast<std::atomic_bool*>(done)->store(true); },
        &state->failure_work_completed);
      if (state->cleanup_host_fault) {
        using fault = sirius::test::scoped_host_allocation_fault;
        state->cleanup_host_fault->emplace(fault::every_allocation_scope, 1);
      }
      std::rethrow_exception(state->failure);
    }
    auto* pointer = state->upstream.allocate(stream, bytes, alignment);
    ++state->live_allocations;
    return pointer;
  }

  void deallocate(::cuda::stream_ref stream,
                  void* pointer,
                  std::size_t bytes,
                  std::size_t alignment) noexcept
  {
    --state->live_allocations;
    state->upstream.deallocate(stream, pointer, bytes, alignment);
  }

  friend void get_property(copy_resource const&, ::cuda::mr::device_accessible) noexcept {}
  bool operator==(copy_resource const& other) const noexcept { return state == other.state; }
};

static_assert(std::copyable<copy_resource>);
static_assert(std::constructible_from<rmm::device_async_resource_ref, copy_resource&>);

}  // namespace

TEST_CASE("owning_table_view release awaits the copy before the owner reclaims memory",
          "[owning_table_view][stream_order]")
{
  // Non-blocking streams: the reclaim must not be implicitly ordered against the
  // copy by legacy-default-stream semantics.
  rmm::cuda_stream work_stream;
  rmm::cuda_stream reclaim_stream;

  constexpr cudf::size_type kRows = 4 * 1024 * 1024;  // 16 MiB of INT32
  auto const kSourceBytes         = static_cast<std::size_t>(kRows) * sizeof(std::int32_t);

  auto source = cudf::make_numeric_column(
    cudf::data_type{cudf::type_id::INT32}, kRows, cudf::mask_state::UNALLOCATED, work_stream);
  auto* source_data = source->mutable_view().data<std::int32_t>();
  cudaMemsetAsync(source_data, kSourceByte, kSourceBytes, work_stream.value());
  work_stream.synchronize();

  std::vector<std::unique_ptr<cudf::column>> cols;
  cols.push_back(std::move(source));
  auto source_table      = std::make_unique<cudf::table>(std::move(cols));
  auto const source_view = source_table->view();

  auto probe            = std::make_shared<reclaim_probe>();
  probe->work_stream    = work_stream.value();
  probe->reclaim_stream = reclaim_stream.value();
  probe->source_data    = source_data;
  probe->source_bytes   = kSourceBytes;

  // Queue a long blocker ahead of the copy so the materializing copy is
  // guaranteed to still be pending when release() hands back control. Without
  // the await this makes the race deterministic rather than probabilistic.
  constexpr std::size_t kBlockerBytes = 256UL * 1024 * 1024;
  rmm::device_buffer blocker(kBlockerBytes, work_stream, test_mr());
  for (int i = 0; i < 8; ++i) {
    cudaMemsetAsync(blocker.data(), 0, kBlockerBytes, work_stream.value());
  }

  // View-backed handle over a NON-no_alloc_materializable owner -> copying path.
  owning_table_view handle{reclaiming_owner{probe, std::move(source_table)}, source_view};
  REQUIRE(handle.n_columns() == 1);

  auto result = handle.release(work_stream, test_mr());
  REQUIRE(result != nullptr);

  // The owner was destroyed by release(); the copy must already have completed.
  REQUIRE(probe->reclaimed);
  CHECK(probe->stream_status_at_reclaim == cudaSuccess);

  // The materialized copy must hold the source values, not the reclaim poison.
  std::vector<std::int32_t> host(static_cast<std::size_t>(kRows));
  cudaMemcpy(host.data(),
             result->view().column(0).data<std::int32_t>(),
             kSourceBytes,
             cudaMemcpyDeviceToHost);

  auto const poisoned = std::count(host.begin(), host.end(), kPoisonValue);
  INFO("poisoned elements: " << poisoned << " / " << kRows);
  REQUIRE(poisoned == 0);
  REQUIRE(host.front() == kSourceValue);
  REQUIRE(host.back() == kSourceValue);
}

TEST_CASE("owning_table_view constructor OOM retires submitted copies and permits retry",
          "[owning_table_view][retirement]")
{
  rmm::cuda_stream work_stream;
  rmm::cuda_stream reclaim_stream;
  copy_resource resource;
  resource.state->fail_at  = 2;
  auto probe               = std::make_shared<reclaim_probe>();
  auto handle              = reclaimable_table(probe, work_stream, reclaim_stream);
  auto const original_ptrs = data_ptrs(handle.view());

  std::exception_ptr failure;
  try {
    (void)handle.release(work_stream, resource);
  } catch (...) {
    failure = std::current_exception();
  }
  REQUIRE(failure == resource.state->failure);
  REQUIRE(resource.state->allocation_calls == 2);
  REQUIRE(resource.state->submission_status == cudaSuccess);
  REQUIRE(resource.state->failure_work_completed.load());
  REQUIRE(resource.state->live_allocations == 0);
  REQUIRE_FALSE(probe->reclaimed);
  REQUIRE(data_ptrs(handle.view()) == original_ptrs);

  resource.state->fail_at = 0;
  auto output             = handle.release(work_stream, resource);
  REQUIRE_FALSE(handle);
  REQUIRE(probe->reclaimed);
  REQUIRE(probe->stream_status_at_reclaim == cudaSuccess);
  REQUIRE(read_tags(output->view()) == std::vector<std::int32_t>{kSourceValue, kSourceValue});
}

TEST_CASE("owning_table_view post-copy OOM preserves the source and releases scratch",
          "[owning_table_view][retirement]")
{
  rmm::cuda_stream work_stream;
  rmm::cuda_stream reclaim_stream;
  copy_resource resource;
  auto probe               = std::make_shared<reclaim_probe>();
  auto handle              = reclaimable_table(probe, work_stream, reclaim_stream);
  auto const original_ptrs = data_ptrs(handle.view());

  {
    read_retirement_test_hooks fault(read_checkpoint::COPY);
    REQUIRE_THROWS_AS(handle.materialize(work_stream, resource), rmm::out_of_memory);
    REQUIRE(fault.checkpoint_reached);
    REQUIRE_FALSE(fault.retained.has_value());
  }
  REQUIRE(resource.state->allocation_calls == 2);
  REQUIRE(resource.state->live_allocations == 0);
  REQUIRE_FALSE(probe->reclaimed);
  REQUIRE(data_ptrs(handle.view()) == original_ptrs);
  REQUIRE(cudaStreamQuery(work_stream.value()) == cudaSuccess);

  auto output = handle.release(work_stream, resource);
  REQUIRE(probe->reclaimed);
  REQUIRE(probe->stream_status_at_reclaim == cudaSuccess);
  REQUIRE(read_tags(output->view()) == std::vector<std::int32_t>{kSourceValue, kSourceValue});
}

TEST_CASE("owning_table_view failed completion retains noncopyable source and copied output",
          "[owning_table_view][retirement]")
{
  // Without a failure point the completion join fails; with one, the cleanup join does.
  auto const failure =
    GENERATE(as<std::optional<read_checkpoint>>{}, std::nullopt, read_checkpoint::COPY);
  rmm::cuda_stream work_stream;
  rmm::cuda_stream reclaim_stream;
  copy_resource resource;
  auto probe = std::make_shared<reclaim_probe>();
  {
    read_retirement_test_hooks fault(failure, true);
    {
      auto handle = reclaimable_table(probe, work_stream, reclaim_stream);
      REQUIRE_THROWS_AS(handle.materialize(work_stream, resource), sirius::unjoined_gpu_work);
      REQUIRE_FALSE(handle);
      REQUIRE(handle.release(work_stream, resource) == nullptr);
    }
    auto const* retained = fault.retained.get<detail::table_copy_state>();
    REQUIRE(retained != nullptr);
    REQUIRE(retained->source != nullptr);
    REQUIRE(retained->output != nullptr);
    REQUIRE_FALSE(probe->reclaimed);
    REQUIRE(resource.state->live_allocations == 2);
    // The hooks retain storage only after a physical join, so inspection and release are safe.
    REQUIRE(cudaStreamQuery(work_stream.value()) == cudaSuccess);
    REQUIRE(read_tags(retained->source->view()) ==
            std::vector<std::int32_t>{kSourceValue, kSourceValue});
    REQUIRE(read_tags(retained->output->view()) ==
            std::vector<std::int32_t>{kSourceValue, kSourceValue});
    REQUIRE(data_ptrs(retained->source->view()) != data_ptrs(retained->output->view()));
  }
  REQUIRE(resource.state->live_allocations == 0);
  REQUIRE(probe->reclaimed);
  REQUIRE(probe->stream_status_at_reclaim == cudaSuccess);
}

TEST_CASE("owning_table_view constructor OOM becomes nonretryable when cleanup fails",
          "[owning_table_view][retirement]")
{
  rmm::cuda_stream work_stream;
  rmm::cuda_stream reclaim_stream;
  copy_resource resource;
  resource.state->fail_at = 2;
  auto probe              = std::make_shared<reclaim_probe>();
  {
    read_retirement_test_hooks fault(std::nullopt, true);
    {
      auto handle = reclaimable_table(probe, work_stream, reclaim_stream);
      REQUIRE_THROWS_AS(handle.materialize(work_stream, resource), sirius::unjoined_gpu_work);
      REQUIRE_FALSE(handle);
    }
    REQUIRE(resource.state->allocation_calls == 2);
    REQUIRE(resource.state->submission_status == cudaSuccess);
    REQUIRE(resource.state->failure_work_completed.load());
    auto const* retained = fault.retained.get<detail::table_copy_state>();
    REQUIRE(retained != nullptr);
    REQUIRE(retained->source != nullptr);
    REQUIRE(retained->output == nullptr);
    REQUIRE_FALSE(probe->reclaimed);
    REQUIRE(resource.state->live_allocations == 0);
  }
  REQUIRE(probe->reclaimed);
  REQUIRE(probe->stream_status_at_reclaim == cudaSuccess);
}

TEST_CASE("owning_table_view transfer fast paths bypass copy retirement",
          "[owning_table_view][retirement]")
{
  auto const project = GENERATE(false, true);
  auto table         = tagged_table({10, 20});
  auto original_ptrs = data_ptrs(table->view());
  owning_table_view handle{std::move(table)};
  if (project) {
    handle.select_columns(std::array<std::size_t, 1>{1});
    original_ptrs.erase(original_ptrs.begin());
  }
  copy_resource resource;
  resource.state->fail_at = 1;
  read_retirement_test_hooks fault(std::nullopt, true);
  auto output = handle.release(test_stream(), resource);
  REQUIRE_FALSE(fault.retained.has_value());
  REQUIRE(data_ptrs(output->view()) == original_ptrs);
  REQUIRE(resource.state->allocation_calls == 0);
}

TEST_CASE("owning_table_view failed cleanup retains owners without host allocation",
          "[.][owning_table_view][retirement][host_allocation_fault]")
{
  using fault = sirius::test::scoped_host_allocation_fault;
  REQUIRE(fault::available());
  rmm::cuda_stream work_stream;
  rmm::cuda_stream reclaim_stream;
  copy_resource resource;
  resource.state->fail_at = 2;
  auto probe              = std::make_shared<reclaim_probe>();
  auto handle             = reclaimable_table(probe, work_stream, reclaim_stream);
  {
    read_retirement_test_hooks hooks(std::nullopt, true);
    std::optional<fault> injection;
    resource.state->cleanup_host_fault = &injection;
    bool unjoined                      = false;
    try {
      handle.materialize(work_stream, resource);
    } catch (sirius::unjoined_gpu_work const&) {
      unjoined = true;
    }
    auto const allocations = injection ? injection->stop() : 0;
    REQUIRE(injection.has_value());
    REQUIRE_FALSE(injection->fired());
    REQUIRE(allocations == 0);
    REQUIRE(unjoined);
    auto const* retained = hooks.retained.get<detail::table_copy_state>();
    REQUIRE(retained != nullptr);
    REQUIRE(retained->source != nullptr);
    REQUIRE_FALSE(probe->reclaimed);
  }
  REQUIRE(probe->reclaimed);
  REQUIRE(probe->stream_status_at_reclaim == cudaSuccess);
}

TEST_CASE("owning_table_view prelaunch host OOM preserves the source without device allocation",
          "[.][owning_table_view][retirement][host_allocation_fault]")
{
  using fault = sirius::test::scoped_host_allocation_fault;
  REQUIRE(fault::available());
  rmm::cuda_stream work_stream;
  rmm::cuda_stream reclaim_stream;
  copy_resource resource;
  auto probe               = std::make_shared<reclaim_probe>();
  auto handle              = reclaimable_table(probe, work_stream, reclaim_stream);
  auto const original_ptrs = data_ptrs(handle.view());
  bool failed              = false;
  fault injection(fault::every_allocation_scope, 1);
  try {
    (void)handle.release(work_stream, resource);
  } catch (std::bad_alloc const&) {
    failed = true;
  }
  (void)injection.stop();
  REQUIRE(injection.fired());
  REQUIRE(failed);
  REQUIRE(resource.state->allocation_calls == 0);
  REQUIRE_FALSE(probe->reclaimed);
  REQUIRE(data_ptrs(handle.view()) == original_ptrs);
}
