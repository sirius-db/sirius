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

#include <rmm/cuda_stream.hpp>

#include <cucascade/data/cpu_data_representation.hpp>
#include <cucascade/data/data_batch.hpp>
#include <cucascade/data/data_repository.hpp>
#include <cucascade/memory/common.hpp>
#include <cucascade/memory/memory_space.hpp>
#include <data/convertible_data_batch.hpp>
#include <data/data_batch_utils.hpp>

#include <memory>
#include <optional>
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

  rmm::cuda_stream_view stream() { return conv_stream.view(); }
};

test_env& env()
{
  static test_env e;
  return e;
}

}  // anonymous namespace

TEST_CASE("convertible_data_batch converts GPU batch to HOST", "[convertible_data_batch]")
{
  auto& e = env();

  auto batch = sirius::test::operator_utils::make_numeric_batch(
    *e.gpu_space, std::vector<int32_t>{1, 2, 3, 4, 5}, cudf::type_id::INT32);

  // Phase 18 / DB-03: get_memory_space is private under #117; access via accessor.
  {
    auto ro = batch->to_read_only();
    REQUIRE(ro.get_memory_space() == e.gpu_space);
  }
  REQUIRE(batch->get_state() == cucascade::batch_state::idle);

  sirius::convertible_data_batch wrapper(batch);
  auto result = wrapper.convert({e.host_space}, e.stream(), *e.mgr);

  REQUIRE(result.has_value());
  REQUIRE((*result).size() == 1);
  REQUIRE((*result)[0] > 0);
  {
    auto ro = batch->to_read_only();
    REQUIRE(ro.get_memory_space()->get_tier() == cucascade::memory::Tier::HOST);
  }
  REQUIRE(batch->get_state() == cucascade::batch_state::idle);
}

TEST_CASE("convertible_data_batch returns nullopt with empty target_spaces",
          "[convertible_data_batch]")
{
  auto& e = env();

  auto batch = sirius::test::operator_utils::make_numeric_batch(
    *e.gpu_space, std::vector<int32_t>{10, 20, 30}, cudf::type_id::INT32);

  // Phase 18 / DB-03: get_memory_space is private under #117; access via accessor.
  {
    auto ro = batch->to_read_only();
    REQUIRE(ro.get_memory_space() == e.gpu_space);
  }
  REQUIRE(batch->get_state() == cucascade::batch_state::idle);

  sirius::convertible_data_batch wrapper(batch);
  auto result = wrapper.convert({}, e.stream(), *e.mgr);

  REQUIRE_FALSE(result.has_value());
  {
    auto ro = batch->to_read_only();
    REQUIRE(ro.get_memory_space() == e.gpu_space);
  }
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

  std::size_t batch2_size = 0;
  {
    auto ro2    = batch2->to_read_only();
    batch2_size = ro2.get_data()->get_size_in_bytes();
  }

  // Make batch3 non-idle by holding a mutable accessor — under cucascade #117
  // try_to_create_task() is gone; the FSM transition was replaced with a
  // mutable-lock acquire (state = mutable_locked). Hold the accessor via
  // optional<> so we can release it later in the test.
  std::optional<cucascade::mutable_data_batch> batch3_mut;
  {
    auto opt = batch3->try_to_mutable();
    REQUIRE(opt.has_value());
    batch3_mut = std::move(*opt);
  }
  REQUIRE(batch3->get_state() == cucascade::batch_state::mutable_locked);

  repo.add_data_batch(batch1);
  repo.add_data_batch(batch2);
  repo.add_data_batch(batch3);

  sirius::convertible_data_batch_provider provider(&repo);
  auto cd = provider.get_next_convertible(e.gpu_space, false);

  REQUIRE(cd != nullptr);
  // Last-to-first: batch3 is mutable_locked so skipped, batch2 is the last idle batch
  REQUIRE(cd->bytes_in_space(e.gpu_space) == batch2_size);

  // Release batch3's mutable lock so the batch can be destroyed cleanly.
  batch3_mut.reset();
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

  // Make batch3 non-idle by holding a mutable accessor (Pitfall 5).
  std::optional<cucascade::mutable_data_batch> batch3_mut;
  {
    auto opt = batch3->try_to_mutable();
    REQUIRE(opt.has_value());
    batch3_mut = std::move(*opt);
  }

  repo.add_data_batch(batch1);
  repo.add_data_batch(batch2);
  repo.add_data_batch(batch3);

  sirius::convertible_data_batch_provider provider(&repo);
  auto all = provider.get_all_convertible(e.gpu_space, false);

  // batch1 and batch2 are idle, batch3 is mutable_locked -> only 2 returned
  REQUIRE(all.size() == 2);

  batch3_mut.reset();
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

  std::size_t batch_p1_size = 0;
  {
    auto ro_p1    = batch_p1->to_read_only();
    batch_p1_size = ro_p1.get_data()->get_size_in_bytes();
  }

  repo.add_data_batch(batch_p0, 0);
  repo.add_data_batch(batch_p1, 1);

  sirius::convertible_data_batch_provider provider(&repo);
  // front_to_back=false means last partition first -> partition 1 before partition 0
  auto cd = provider.get_next_convertible(e.gpu_space, false);

  REQUIRE(cd != nullptr);
  REQUIRE(cd->bytes_in_space(e.gpu_space) == batch_p1_size);
}

TEST_CASE("convertible_data_batch convert fails when batch already in_transit",
          "[convertible_data_batch]")
{
  auto& e = env();

  auto batch = sirius::test::operator_utils::make_numeric_batch(
    *e.gpu_space, std::vector<int32_t>{1, 2, 3}, cudf::type_id::INT32);

  // Phase 18 / DB-03 Recipe R8: pre-#117 try_to_lock_for_in_transit() is
  // gone; under #117 the equivalent is taking a mutable accessor. State
  // becomes mutable_locked (the only non-idle non-read_only state).
  std::optional<cucascade::mutable_data_batch> mut;
  {
    auto opt = batch->try_to_mutable();
    REQUIRE(opt.has_value());
    mut = std::move(*opt);
  }
  REQUIRE(batch->get_state() == cucascade::batch_state::mutable_locked);

  sirius::convertible_data_batch wrapper(batch);
  auto result = wrapper.convert({e.host_space}, e.stream(), *e.mgr);

  REQUIRE_FALSE(result.has_value());
  // State is still mutable_locked (the wrapper failed to acquire the lock).
  REQUIRE(batch->get_state() == cucascade::batch_state::mutable_locked);

  // Release the mutable accessor so batch can be destroyed cleanly.
  mut.reset();
}

TEST_CASE("convertible_data_batch bytes_in_space returns correct size", "[convertible_data_batch]")
{
  auto& e = env();

  auto batch = sirius::test::operator_utils::make_numeric_batch(
    *e.gpu_space, std::vector<int32_t>{1, 2, 3, 4, 5}, cudf::type_id::INT32);

  sirius::convertible_data_batch wrapper(batch);

  auto size = wrapper.bytes_in_space(e.gpu_space);
  REQUIRE(size > 0);
  std::size_t expected_size = 0;
  {
    auto ro       = batch->to_read_only();
    expected_size = ro.get_data()->get_size_in_bytes();
  }
  REQUIRE(size == expected_size);
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

  std::size_t batch1_size = 0;
  std::size_t batch2_size = 0;
  {
    auto ro1    = batch1->to_read_only();
    batch1_size = ro1.get_data()->get_size_in_bytes();
  }
  {
    auto ro2    = batch2->to_read_only();
    batch2_size = ro2.get_data()->get_size_in_bytes();
  }

  repo.add_data_batch(batch1);
  repo.add_data_batch(batch2);

  sirius::convertible_data_batch_provider provider(&repo);

  auto total = provider.get_bytes_in_space(e.gpu_space);
  REQUIRE(total == batch1_size + batch2_size);
  REQUIRE(provider.get_bytes_in_space(e.host_space) == 0);
}
