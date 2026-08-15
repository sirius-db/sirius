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
#include <cudf/table/table_view.hpp>

#include <rmm/cuda_stream.hpp>

#include <cuda_runtime_api.h>

#include <cucascade/cudf/host_data_representation.hpp>
#include <cucascade/data/data_batch.hpp>
#include <cucascade/data/data_repository.hpp>
#include <cucascade/memory/common.hpp>
#include <cucascade/memory/memory_space.hpp>
#include <data/convertible_data_batch.hpp>
#include <data/data_batch_utils.hpp>
#include <telemetry/data_batch_probe.hpp>

#include <memory>
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

TEST_CASE("convertible_data_batch downgrades a view-backed GPU batch by copy-out",
          "[convertible_data_batch]")
{
  auto& e = env();

  std::vector<int32_t> const values{1, 2, 3, 4, 5};
  auto owned =
    cudf::make_numeric_column(cudf::data_type{cudf::type_id::INT32},
                              static_cast<cudf::size_type>(values.size()),
                              cudf::mask_state::UNALLOCATED,
                              sirius::test::operator_utils::default_stream(),
                              sirius::test::operator_utils::get_resource_ref(*e.gpu_space));
  cudaMemcpy(owned->mutable_view().head<int32_t>(),
             values.data(),
             sizeof(int32_t) * values.size(),
             cudaMemcpyHostToDevice);
  std::shared_ptr<cudf::column> column(std::move(owned));

  // The scan's view-forward output shape: a batch viewing shared column storage it does not own.
  std::vector<cudf::column_view> views{column->view()};
  auto batch = sirius::make_data_batch_from_view(cudf::table_view{views},
                                                 std::vector<std::shared_ptr<cudf::column>>{column},
                                                 column->alloc_size(),
                                                 *e.gpu_space,
                                                 e.stream(),
                                                 sirius::telemetry::batch_telemetry_info{});

  sirius::convertible_data_batch wrapper(batch);
  auto result = wrapper.convert({e.host_space}, e.stream(), *e.mgr, true);
  REQUIRE(result.has_value());
  REQUIRE(get_batch_tier(*batch) == cucascade::memory::Tier::HOST);

  // The downgrade copied out of the view: the source column is intact and now solely ours.
  REQUIRE(column.use_count() == 1);
  REQUIRE(sirius::test::operator_utils::copy_column_to_host<int32_t>(column->view()) == values);

  // Round-trip the host bytes back to the GPU to check them.
  sirius::convertible_data_batch upgrade(batch);
  auto restored = upgrade.convert({e.gpu_space}, e.stream(), *e.mgr, true);
  REQUIRE(restored.has_value());
  e.stream().synchronize();
  auto ro   = batch->to_read_only();
  auto view = sirius::get_cudf_table_view(ro);
  REQUIRE(sirius::test::operator_utils::copy_column_to_host<int32_t>(view.column(0)) == values);
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
