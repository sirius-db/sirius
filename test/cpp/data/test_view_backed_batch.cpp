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

// ---------------------------------------------------------------------------
// Reproducer for issue #1063:
//   "Ensure all data batches stored in repositories are idle (no read-only
//    pins via owning_table_view owners)"
//   https://github.com/sirius-db/sirius/issues/1063
//
// The bug: view-backed data batches created via sirius::make_data_batch_from_view
// (src/include/data/data_batch_utils.hpp) carry a type-erased owner that is
// typically a read_only_data_batch lock on a *source* batch. This is exactly what
// PROJECTION's zero-copy passthrough does (src/op/sirius_physical_projection.cpp:138)
// and what the pinned-table scan does (src/op/sirius_physical_table_scan.cpp:244).
//
// While such a view-backed batch is alive -- INCLUDING while it merely sits parked
// in a shared_data_repository -- the source batch stays pinned in
// batch_state::read_only and can never return to idle. The downgrade executor only
// spills batches that are idle (convertible_data_batch_provider::try_get_batch skips
// non-idle batches, src/include/data/convertible_data_batch.hpp:324), so the source
// batch's GPU memory becomes permanently unspillable. Under memory pressure a
// blocking downgrade of the source (to_mutable() -> convert_to<host>) waits forever
// on the read-only pin -> deadlock.
//
// These tests reproduce that scenario directly with the low-level data_batch /
// repository / convertible machinery (no full SQL query and no real memory pressure
// required), so the invariant violation and the resulting deadlock are deterministic.
//
// NOTE ON SCOPE: these are *characterisation* reproducers. They construct the
// view-backed pin directly (mirroring what the projection/scan operators emit) and
// assert the defective state that #1063 describes. They PASS on upstream/dev because
// they assert that the bug is present; each test then also demonstrates that
// releasing the pin (the direction the fix must take -- materialise view batches
// before parking them, so repo batches stay idle) restores correct behaviour. The
// flip-on-fix regression guard the issue asks for -- an assertion on add_data_batch()
// that every parked batch is idle -- belongs at the operator layer and is described
// in the accompanying README.
// ---------------------------------------------------------------------------

#include "catch.hpp"
#include "operator/operator_test_utils.hpp"

#include <rmm/cuda_stream.hpp>

#include <cucascade/cudf/host_data_representation.hpp>
#include <cucascade/data/data_batch.hpp>
#include <cucascade/data/data_repository.hpp>
#include <cucascade/memory/common.hpp>
#include <cucascade/memory/memory_space.hpp>
#include <data/convertible_data_batch.hpp>
#include <data/data_batch_utils.hpp>

#include <atomic>
#include <chrono>
#include <memory>
#include <optional>
#include <thread>
#include <vector>

namespace {

// Shared test environment: initialize the memory manager once for all tests in this
// file. Mirrors test/cpp/data/test_convertible_data_batch.cpp. Uses a real
// (non-default) CUDA stream because the cuCascade converter uses
// cudaMemcpyBatchAsync, which requires a non-default stream.
struct test_env {
  std::unique_ptr<sirius::memory::sirius_memory_reservation_manager> mgr;
  cucascade::memory::memory_space* gpu_space;
  cucascade::memory::memory_space* host_space;
  rmm::cuda_stream conv_stream;

  test_env()
    : mgr(sirius::test::operator_utils::initialize_memory_manager()),
      gpu_space(mgr->get_memory_space(cucascade::memory::Tier::GPU, 0)),
      host_space(mgr->get_memory_space(cucascade::memory::Tier::HOST, 0)),
      conv_stream()
  {
  }

  rmm::cuda_stream_view stream() { return conv_stream.view(); }
};

test_env& env()
{
  static test_env e;
  return e;
}

// Build a view-backed data_batch whose type-erased owner is a read_only_data_batch
// lock on `source` -- exactly what PROJECTION passthrough emits
// (sirius_physical_projection.cpp:138 via sirius::make_data_batch_from_view). The
// returned batch keeps `source` pinned read_only for as long as it is alive.
std::shared_ptr<cucascade::data_batch> make_passthrough_view_batch(
  cucascade::data_batch& source,
  cucascade::memory::memory_space& space,
  rmm::cuda_stream_view stream)
{
  auto source_ro          = source.to_read_only();  // shared lock on the source batch
  cudf::table_view view   = sirius::get_cudf_table_view(source_ro);
  const std::size_t bytes = source_ro.get_data()->get_size_in_bytes();

  // Owner = the source's read-only lock. std::move()-ing it into the owner keeps the
  // shared lock alive inside the new batch's type-erased owner (net read-only count
  // stays 1). Telemetry defaults to the no-op probe in tests.
  return sirius::make_data_batch_from_view(
    view, std::move(source_ro), bytes, space, stream, sirius::telemetry::batch_telemetry_info{});
}

}  // namespace

TEST_CASE(
  "issue #1063: view-backed batch parked in a repository pins its source read_only "
  "and makes it unspillable",
  "[convertible_data_batch][issue-1063]")
{
  auto& e = env();

  // --- Upstream operator's output: an owned GPU batch, parked in its repository. ---
  cucascade::shared_data_repository upstream_repo;
  auto source = sirius::test::operator_utils::make_numeric_batch(
    *e.gpu_space, std::vector<int32_t>{1, 2, 3, 4, 5, 6, 7, 8}, cudf::type_id::INT32);
  const auto source_id = source->get_batch_id();
  upstream_repo.add_data_batch(source);

  // Sanity: while idle, the source IS a valid downgrade candidate (spillable).
  {
    sirius::convertible_data_batch_provider provider(&upstream_repo);
    REQUIRE(source->get_state() == cucascade::batch_state::idle);
    REQUIRE(provider.get_all_convertible(e.gpu_space, /*front_to_back=*/false).size() == 1);
  }

  // --- Downstream PROJECTION passthrough: emit a view-backed batch that owns a
  //     read-only lock on `source`, and park it in the downstream repository. ---
  cucascade::shared_data_repository downstream_repo;
  {
    auto view_batch = make_passthrough_view_batch(*source, *e.gpu_space, e.stream());
    downstream_repo.add_data_batch(view_batch);
    // Drop the local reference: the ONLY thing keeping the view-backed batch (and thus
    // the read-only pin on `source`) alive is now the downstream repository. This is
    // the crux of #1063 -- the pin's lifetime is the repository's, not a task's.
  }

  // === The bug: `source` is parked in a repository but is NOT idle. ===
  REQUIRE(source->get_state() == cucascade::batch_state::read_only);  // invariant violated
  REQUIRE(source->get_read_only_count() == 1);
  REQUIRE(downstream_repo.size() == 1);  // the pin is held by a repo-parked batch

  // Consequence 1: the downgrade candidate scanner skips the non-idle source, so its
  // GPU memory can never be selected for spilling -> unspillable.
  {
    sirius::convertible_data_batch_provider provider(&upstream_repo);
    REQUIRE(provider.get_all_convertible(e.gpu_space, /*front_to_back=*/false).empty());
  }

  // Consequence 2: a non-blocking downgrade of the source fails outright, because
  // try_to_mutable() cannot acquire the exclusive lock while the read-only pin is held.
  {
    sirius::convertible_data_batch wrapper(source);
    auto result = wrapper.convert({e.host_space}, e.stream(), *e.mgr, /*blocking=*/false);
    REQUIRE_FALSE(result.has_value());
    REQUIRE(source->get_state() == cucascade::batch_state::read_only);
  }

  // --- Fix direction: materialise/evict the view-backed batch before it is parked
  //     (release_table() would make it owned; here we simply evict it). Dropping the
  //     view-backed batch releases the read-only owner, so `source` returns to idle. ---
  auto view_batch_ids = downstream_repo.get_batch_ids();
  REQUIRE(view_batch_ids.size() == 1);
  {
    auto evicted = downstream_repo.pop_data_batch_by_id(view_batch_ids[0]);
    REQUIRE(evicted != nullptr);
    // evicted (and its type-erased read-only owner) is destroyed at end of scope.
  }

  // Recovery: with the pin gone, the source is idle again and spillable.
  REQUIRE(source->get_state() == cucascade::batch_state::idle);
  REQUIRE(source->get_read_only_count() == 0);
  {
    sirius::convertible_data_batch_provider provider(&upstream_repo);
    REQUIRE(provider.get_all_convertible(e.gpu_space, /*front_to_back=*/false).size() == 1);

    sirius::convertible_data_batch wrapper(source);
    auto result = wrapper.convert({e.host_space}, e.stream(), *e.mgr, /*blocking=*/true);
    REQUIRE(result.has_value());
    // source is now on the HOST tier.
    auto ro = source->to_read_only();
    REQUIRE(ro.get_memory_space()->get_tier() == cucascade::memory::Tier::HOST);
  }

  REQUIRE(source->get_batch_id() == source_id);
}

// Makes the actual DEADLOCK tangible: a blocking downgrade of a read-only-pinned
// source cannot complete while the pinning view-backed batch sits in a repository.
// The test proves the block (the worker does not finish within a generous window),
// then releases the pin -- the direction the #1063 fix must take -- and shows the
// worker immediately unblocks and completes. The worker is always joined, so no thread
// is left blocked at teardown.
TEST_CASE("issue #1063: blocking downgrade deadlocks on a repo-held read-only pin",
          "[convertible_data_batch][issue-1063-deadlock]")
{
  auto& e = env();

  cucascade::shared_data_repository upstream_repo;
  auto source = sirius::test::operator_utils::make_numeric_batch(
    *e.gpu_space, std::vector<int32_t>{10, 20, 30, 40}, cudf::type_id::INT32);
  upstream_repo.add_data_batch(source);

  cucascade::shared_data_repository downstream_repo;
  {
    auto view_batch = make_passthrough_view_batch(*source, *e.gpu_space, e.stream());
    downstream_repo.add_data_batch(view_batch);
  }
  REQUIRE(source->get_state() == cucascade::batch_state::read_only);

  // Kick off a blocking downgrade of the (pinned) source on a worker thread. It blocks
  // inside convert() -> to_mutable(), waiting for the read-only count to reach zero.
  std::atomic<bool> finished{false};
  std::optional<std::vector<std::size_t>> conv_result;
  std::thread worker([&] {
    sirius::convertible_data_batch wrapper(source);
    conv_result = wrapper.convert({e.host_space}, e.stream(), *e.mgr, /*blocking=*/true);
    finished.store(true);
  });

  // While the pin is held the blocking downgrade cannot make progress: this is the
  // deadlock. A generous window keeps the assertion deterministic on slow CI.
  std::this_thread::sleep_for(std::chrono::milliseconds(500));
  REQUIRE_FALSE(finished.load());  // deadlocked on the read-only pin

  // Release the pin (what the #1063 fix guarantees by keeping repo batches idle). The
  // worker can now acquire the exclusive lock and complete.
  auto ids     = downstream_repo.get_batch_ids();
  auto evicted = downstream_repo.pop_data_batch_by_id(ids.at(0));
  evicted.reset();

  worker.join();

  REQUIRE(finished.load());
  REQUIRE(conv_result.has_value());
}
