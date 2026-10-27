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
#include "utils/sirius_test_env.hpp"

#include <rmm/cuda_device.hpp>
#include <rmm/cuda_stream.hpp>

#include <cucascade/cudf/gpu_data_representation.hpp>
#include <cucascade/cudf/host_data_representation.hpp>
#include <cucascade/data/data_batch.hpp>
#include <cucascade/data/data_repository.hpp>
#include <cucascade/memory/common.hpp>
#include <cucascade/memory/memory_space.hpp>
#include <data/convertible_data_batch.hpp>
#include <data/data_batch_utils.hpp>
#include <data/sirius_converter_registry.hpp>
#include <event/downgraded_batch.hpp>

#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

namespace {

// Shared test environment: initialize memory manager once for all tests in this file.
// Uses rmm::cuda_stream (a real non-default CUDA stream) because the cuCascade
// converter uses cudaMemcpyBatchAsync which requires a non-default stream.
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

  ::cuda::stream_ref stream() { return conv_stream; }
};

test_env& env()
{
  static test_env e;
  return e;
}

/// Helper: get the tier of a data_batch using a temporary read-only lock.
inline cucascade::memory::Tier get_batch_tier(cucascade::data_batch& batch)
{
  auto ro = batch.to_read_only();
  return ro.get_memory_space()->get_tier();
}

/// Helper: get the size in bytes of a data_batch's data using a temporary read-only lock.
inline size_t get_batch_size(cucascade::data_batch& batch)
{
  auto ro = batch.to_read_only();
  return ro.get_data()->get_size_in_bytes();
}

/// Helper: compare batch's memory_space pointer (returns true if same as given space).
inline bool batch_in_space(cucascade::data_batch& batch,
                           const cucascade::memory::memory_space* space)
{
  auto ro = batch.to_read_only();
  return ro.get_memory_space() == space;
}

}  // anonymous namespace

TEST_CASE("convertible_data_batch converts GPU batch to HOST", "[convertible_data_batch]")
{
  auto& e = env();

  auto batch = sirius::test::operator_utils::make_numeric_batch(
    *e.gpu_space, std::vector<int32_t>{1, 2, 3, 4, 5}, cudf::type_id::INT32);

  REQUIRE(batch_in_space(*batch, e.gpu_space));
  REQUIRE(batch->get_state() == cucascade::batch_state::idle);

  sirius::convertible_data_batch wrapper(batch);
  auto result = wrapper.convert({e.host_space}, e.stream(), *e.mgr, true);

  REQUIRE(result.has_value());
  REQUIRE((*result).size() == 1);
  REQUIRE((*result)[0] > 0);
  REQUIRE(get_batch_tier(*batch) == cucascade::memory::Tier::HOST);
  REQUIRE(batch->get_state() == cucascade::batch_state::idle);
}

TEST_CASE("convertible_data_batch returns nullopt with empty target_spaces",
          "[convertible_data_batch]")
{
  auto& e = env();

  auto batch = sirius::test::operator_utils::make_numeric_batch(
    *e.gpu_space, std::vector<int32_t>{10, 20, 30}, cudf::type_id::INT32);

  REQUIRE(batch_in_space(*batch, e.gpu_space));
  REQUIRE(batch->get_state() == cucascade::batch_state::idle);

  sirius::convertible_data_batch wrapper(batch);
  auto result = wrapper.convert({}, e.stream(), *e.mgr, false);

  REQUIRE_FALSE(result.has_value());
  REQUIRE(batch_in_space(*batch, e.gpu_space));
  REQUIRE(batch->get_state() == cucascade::batch_state::idle);
}

TEST_CASE("convertible_data_batch_provider get_next_convertible returns last idle batch",
          "[convertible_data_batch]")
{
  auto& e = env();

  cucascade::shared_data_repository repo;

  // Create 3 GPU batches with different sizes to distinguish them
  auto batch1 = sirius::test::operator_utils::make_numeric_batch(
    *e.gpu_space, std::vector<int32_t>{1, 2}, cudf::type_id::INT32);
  auto batch2 = sirius::test::operator_utils::make_numeric_batch(
    *e.gpu_space, std::vector<int32_t>{3, 4, 5}, cudf::type_id::INT32);
  auto batch3 = sirius::test::operator_utils::make_numeric_batch(
    *e.gpu_space, std::vector<int32_t>{6, 7, 8, 9}, cudf::type_id::INT32);

  auto batch2_size = get_batch_size(*batch2);

  // Make batch3 non-idle by holding a read-only lock on it
  auto batch3_lock = batch3->to_read_only();

  repo.add_data_batch(batch1);
  repo.add_data_batch(batch2);
  repo.add_data_batch(batch3);

  sirius::convertible_data_batch_provider provider(&repo);
  auto cd = provider.get_next_convertible(e.gpu_space, false);

  REQUIRE(cd != nullptr);
  // Last-to-first: batch3 is locked (non-idle) so skipped, batch2 is the last idle batch
  REQUIRE(cd->bytes_in_space(e.gpu_space) == batch2_size);
}

TEST_CASE("convertible_data_batch_provider get_all_convertible returns all idle batches",
          "[convertible_data_batch]")
{
  auto& e = env();

  cucascade::shared_data_repository repo;

  auto batch1 = sirius::test::operator_utils::make_numeric_batch(
    *e.gpu_space, std::vector<int32_t>{1, 2}, cudf::type_id::INT32);
  auto batch2 = sirius::test::operator_utils::make_numeric_batch(
    *e.gpu_space, std::vector<int32_t>{3, 4, 5}, cudf::type_id::INT32);
  auto batch3 = sirius::test::operator_utils::make_numeric_batch(
    *e.gpu_space, std::vector<int32_t>{6, 7, 8, 9}, cudf::type_id::INT32);

  // Make batch3 non-idle by holding a read-only lock on it
  auto batch3_lock = batch3->to_read_only();

  repo.add_data_batch(batch1);
  repo.add_data_batch(batch2);
  repo.add_data_batch(batch3);

  sirius::convertible_data_batch_provider provider(&repo);
  auto all = provider.get_all_convertible(e.gpu_space, false);

  // batch1 and batch2 are idle, batch3 is locked -> only 2 returned
  REQUIRE(all.size() == 2);
}

TEST_CASE("convertible_data_batch_provider get_all_convertible skips subscribed batches",
          "[convertible_data_batch]")
{
  auto& e = env();

  cucascade::shared_data_repository repo;

  auto batch1 = sirius::test::operator_utils::make_numeric_batch(
    *e.gpu_space, std::vector<int32_t>{1, 2}, cudf::type_id::INT32);
  auto batch2 = sirius::test::operator_utils::make_numeric_batch(
    *e.gpu_space, std::vector<int32_t>{3, 4, 5}, cudf::type_id::INT32);
  auto batch3 = sirius::test::operator_utils::make_numeric_batch(
    *e.gpu_space, std::vector<int32_t>{6, 7, 8, 9}, cudf::type_id::INT32);

  // batch2 is subscribed: a task has declared interest in it and it must not be downgraded.
  batch2->subscribe();

  repo.add_data_batch(batch1);
  repo.add_data_batch(batch2);
  repo.add_data_batch(batch3);

  sirius::convertible_data_batch_provider provider(&repo);

  SECTION("ignore_subscribed=true (default) skips the subscribed batch")
  {
    auto all = provider.get_all_convertible(e.gpu_space, false, /*ignore_subscribed=*/true);
    // batch2 is subscribed -> excluded; batch1 and batch3 remain.
    REQUIRE(all.size() == 2);
  }

  SECTION("ignore_subscribed=false includes the subscribed batch")
  {
    auto all = provider.get_all_convertible(e.gpu_space, false, /*ignore_subscribed=*/false);
    // Subscription is ignored -> all three idle batches are returned.
    REQUIRE(all.size() == 3);
  }

  batch2->unsubscribe();
}

TEST_CASE("convertible_data_batch_provider iterates multi-partition last-to-first",
          "[convertible_data_batch]")
{
  auto& e = env();

  cucascade::shared_data_repository repo;

  // Create batches with different sizes to distinguish them
  auto batch_p0 = sirius::test::operator_utils::make_numeric_batch(
    *e.gpu_space, std::vector<int32_t>{1, 2}, cudf::type_id::INT32);
  auto batch_p1 = sirius::test::operator_utils::make_numeric_batch(
    *e.gpu_space, std::vector<int32_t>{3, 4, 5, 6, 7}, cudf::type_id::INT32);

  auto batch_p1_size = get_batch_size(*batch_p1);

  repo.add_data_batch(batch_p0, 0);
  repo.add_data_batch(batch_p1, 1);

  sirius::convertible_data_batch_provider provider(&repo);
  // front_to_back=false means last partition first -> partition 1 before partition 0
  auto cd = provider.get_next_convertible(e.gpu_space, false);

  REQUIRE(cd != nullptr);
  REQUIRE(cd->bytes_in_space(e.gpu_space) == batch_p1_size);
}

TEST_CASE("convertible_data_batch skips a batch that left the space it was picked from",
          "[convertible_data_batch][multi_gpu]")
{
  if (!sirius::test::has_gpus(2)) { return; }

  auto mgr   = sirius::test::operator_utils::initialize_memory_manager(2);
  auto* gpu0 = mgr->get_memory_space(cucascade::memory::Tier::GPU, 0);
  auto* gpu1 = mgr->get_memory_space(cucascade::memory::Tier::GPU, 1);
  auto* host = mgr->get_memory_space(cucascade::memory::Tier::HOST, 0);
  REQUIRE(gpu0 != nullptr);
  REQUIRE(gpu1 != nullptr);
  REQUIRE(host != nullptr);

  cucascade::shared_data_repository repo;
  auto batch = sirius::test::operator_utils::make_numeric_batch(
    *gpu0, std::vector<int32_t>{1, 2, 3}, cudf::type_id::INT32);
  repo.add_data_batch(batch);

  // GPU 0's downgrade executor picks the batch...
  sirius::convertible_data_batch_provider provider(&repo);
  auto cd = provider.get_next_convertible(gpu0, false);
  REQUIRE(cd != nullptr);

  // ...and before its worker runs, the batch moves to GPU 1 (spilled, then upgraded there).
  rmm::cuda_stream stream0;
  auto& registry = sirius::converter_registry::get();
  {
    auto mut = batch->to_mutable();
    mut.convert_to<cucascade::host_data_representation>(registry, host, stream0);
  }
  stream0.synchronize();
  {
    rmm::cuda_set_device_raii const device{rmm::cuda_device_id{1}};
    rmm::cuda_stream stream1;
    {
      auto mut = batch->to_mutable();
      mut.convert_to<cucascade::gpu_table_representation>(registry, gpu1, stream1);
    }
    stream1.synchronize();
  }
  REQUIRE(batch_in_space(*batch, gpu1));

  // The worker's GPU 0 stream cannot serve GPU 1's memory: the batch is skipped, not converted.
  std::optional<std::vector<std::size_t>> result;
  REQUIRE_NOTHROW(result = cd->convert({host}, stream0, *mgr, true));
  CHECK_FALSE(result.has_value());
  CHECK(batch_in_space(*batch, gpu1));
  CHECK(batch->get_state() == cucascade::batch_state::idle);
}

TEST_CASE("convertible_data_batch convert fails when batch exclusively locked",
          "[convertible_data_batch]")
{
  auto& e = env();

  auto batch = sirius::test::operator_utils::make_numeric_batch(
    *e.gpu_space, std::vector<int32_t>{1, 2, 3}, cudf::type_id::INT32);

  // Hold an exclusive (mutable) lock so that try_to_mutable() inside convert() fails
  auto exclusive_lock = batch->to_mutable();

  sirius::convertible_data_batch wrapper(batch);
  // Non-blocking convert: uses try_to_mutable() internally, must return nullopt
  auto result = wrapper.convert({e.host_space}, e.stream(), *e.mgr, /*blocking=*/false);

  REQUIRE_FALSE(result.has_value());

  // Release exclusive lock so batch can be destroyed cleanly
  {
    auto discard = std::move(exclusive_lock);
  }
}

TEST_CASE("convertible_data_batch bytes_in_space returns correct size", "[convertible_data_batch]")
{
  auto& e = env();

  auto batch = sirius::test::operator_utils::make_numeric_batch(
    *e.gpu_space, std::vector<int32_t>{1, 2, 3, 4, 5}, cudf::type_id::INT32);

  sirius::convertible_data_batch wrapper(batch);

  auto size = wrapper.bytes_in_space(e.gpu_space);
  REQUIRE(size > 0);
  REQUIRE(size == get_batch_size(*batch));
  REQUIRE(wrapper.bytes_in_space(e.host_space) == 0);
}

TEST_CASE("convertible_data_batch_provider get_bytes_in_space sums batch sizes",
          "[convertible_data_batch]")
{
  auto& e = env();

  cucascade::shared_data_repository repo;

  auto batch1 = sirius::test::operator_utils::make_numeric_batch(
    *e.gpu_space, std::vector<int32_t>{1, 2, 3}, cudf::type_id::INT32);
  auto batch2 = sirius::test::operator_utils::make_numeric_batch(
    *e.gpu_space, std::vector<int32_t>{4, 5, 6, 7, 8}, cudf::type_id::INT32);

  auto batch1_size = get_batch_size(*batch1);
  auto batch2_size = get_batch_size(*batch2);

  repo.add_data_batch(batch1);
  repo.add_data_batch(batch2);

  sirius::convertible_data_batch_provider provider(&repo);

  auto total = provider.get_bytes_in_space(e.gpu_space);
  REQUIRE(total == batch1_size + batch2_size);
  REQUIRE(provider.get_bytes_in_space(e.host_space) == 0);
}

TEST_CASE("convertible_data_batch records nothing unless recording is on",
          "[convertible_data_batch]")
{
  auto& e = env();

  auto batch = sirius::test::operator_utils::make_numeric_batch(
    *e.gpu_space, std::vector<int32_t>{1, 2, 3}, cudf::type_id::INT32);

  sirius::convertible_data_batch wrapper(batch);
  auto result = wrapper.convert({e.host_space}, e.stream(), *e.mgr, true);

  REQUIRE(result.has_value());
  REQUIRE(batch_in_space(*batch, e.host_space));
  REQUIRE(wrapper.converted_batches().empty());
}

TEST_CASE("convertible_data_batch records one entry per successful conversion",
          "[convertible_data_batch]")
{
  auto& e = env();

  auto batch = sirius::test::operator_utils::make_numeric_batch(
    *e.gpu_space, std::vector<int32_t>{1, 2, 3, 4}, cudf::type_id::INT32);
  auto const gpu_size = get_batch_size(*batch);

  sirius::convertible_data_batch down(batch);
  down.record_conversions(true);
  auto const seq_before = sirius::g_conversion_seq.load();
  REQUIRE(down.convert({e.host_space}, e.stream(), *e.mgr, true).has_value());

  REQUIRE(down.converted_batches().size() == 1);
  auto const& rec = down.converted_batches().front();
  REQUIRE(rec.batch.lock() == batch);
  REQUIRE(rec.batch_id == batch->get_batch_id());
  // The record takes exactly one value from the process-wide counter.
  REQUIRE(rec.conversion_seq == seq_before);
  REQUIRE(sirius::g_conversion_seq.load() == seq_before + 1);
  REQUIRE(rec.from == e.gpu_space->get_id());
  REQUIRE(rec.to == e.host_space->get_id());
  REQUIRE(rec.bytes == gpu_size);
  REQUIRE(rec.source == sirius::event::downgrade_source::repository);
  REQUIRE(rec.repository == 0);
  REQUIRE(rec.partition == 0);
  REQUIRE_FALSE(rec.task_priority.has_value());
  REQUIRE_FALSE(rec.preferred_device.has_value());
  REQUIRE_FALSE(rec.task_query_id.has_value());

  SECTION("a later move of the same batch gets a larger sequence number")
  {
    auto const host_size = get_batch_size(*batch);
    sirius::convertible_data_batch up(batch);
    up.record_conversions(true);
    REQUIRE(up.convert({e.gpu_space}, e.stream(), *e.mgr, true).has_value());

    REQUIRE(up.converted_batches().size() == 1);
    auto const& later = up.converted_batches().front();
    REQUIRE(later.batch_id == rec.batch_id);
    REQUIRE(later.conversion_seq > rec.conversion_seq);
    REQUIRE(later.from == e.host_space->get_id());
    REQUIRE(later.to == e.gpu_space->get_id());
    REQUIRE(later.bytes == host_size);
  }

  SECTION("a conversion that does not happen records nothing")
  {
    // Already in the only target space.
    REQUIRE_FALSE(down.convert({e.host_space}, e.stream(), *e.mgr, true).has_value());
    REQUIRE(down.converted_batches().size() == 1);

    // Exclusive lock unavailable to a non-blocking attempt.
    auto exclusive_lock = batch->to_mutable();
    sirius::convertible_data_batch blocked(batch);
    blocked.record_conversions(true);
    REQUIRE_FALSE(blocked.convert({e.gpu_space}, e.stream(), *e.mgr, false).has_value());
    REQUIRE(blocked.converted_batches().empty());
    {
      auto discard = std::move(exclusive_lock);
    }

    // No target space to convert into.
    sirius::convertible_data_batch no_target(batch);
    no_target.record_conversions(true);
    auto const seq_now = sirius::g_conversion_seq.load();
    REQUIRE_FALSE(no_target.convert({}, e.stream(), *e.mgr, true).has_value());
    REQUIRE(no_target.converted_batches().empty());
    REQUIRE(sirius::g_conversion_seq.load() == seq_now);
  }
}

TEST_CASE("convertible_data_batch_provider records the batch's repository and partition",
          "[convertible_data_batch]")
{
  auto& e = env();

  cucascade::shared_data_repository repo;

  auto batch_p0 = sirius::test::operator_utils::make_numeric_batch(
    *e.gpu_space, std::vector<int32_t>{1, 2}, cudf::type_id::INT32);
  auto batch_p1 = sirius::test::operator_utils::make_numeric_batch(
    *e.gpu_space, std::vector<int32_t>{3, 4, 5}, cudf::type_id::INT32);

  repo.add_data_batch(batch_p0, 0);
  repo.add_data_batch(batch_p1, 1);

  sirius::convertible_data_batch_provider provider(&repo);
  // Back-to-front: partition 1 is found first.
  auto cd = provider.get_next_convertible(e.gpu_space, false);
  REQUIRE(cd != nullptr);
  REQUIRE(cd->converted_batches().empty());

  cd->record_conversions(true);
  REQUIRE(cd->convert({e.host_space}, e.stream(), *e.mgr, true).has_value());

  REQUIRE(cd->converted_batches().size() == 1);
  auto const& rec = cd->converted_batches().front();
  REQUIRE(rec.batch_id == batch_p1->get_batch_id());
  REQUIRE(rec.repository == reinterpret_cast<std::uintptr_t>(&repo));
  REQUIRE(rec.partition == 1);
  REQUIRE(rec.source == sirius::event::downgrade_source::repository);
}
