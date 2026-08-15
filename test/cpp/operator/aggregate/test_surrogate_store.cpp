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

/**
 * @file test_surrogate_store.cpp
 * @brief Contract tests for `sirius::op::surrogate_deferral_store` (the surrogate-key group-by
 *        retention store): contiguous per-batch address ranges with idempotent reservation, the
 *        check-before-mutate int32 overflow guard, and the reserve/commit/snapshot/release
 *        retention protocol with its type-pinned contract violations.
 */

#include "../operator_test_utils.hpp"
#include "../operator_type_traits.hpp"
#include "op/groupby_surrogate_store.hpp"
#include "sirius/exception.hpp"
#include "utils/data_utils.hpp"

#include <cudf/table/table.hpp>

#include <catch.hpp>

#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

using namespace sirius::op;
using namespace sirius::test::operator_utils;
using sirius::test::vector_to_cudf_column;

TEST_CASE("surrogate store assigns contiguous ranges and dedupes by batch id",
          "[surrogate_groupby]")
{
  surrogate_deferral_store store;
  REQUIRE(store.reserve(join_side::left, /*batch_id=*/11, 10).base() == 0);
  REQUIRE(store.reserve(join_side::left, /*batch_id=*/22, 5).base() == 10);
  // Same batch id (a task retry, or a BUILD_PROBE build table shared by many probe tasks)
  // returns the existing range instead of burning new address space. (Dropping the earlier
  // tokens without commit is legal -- the range stays reserved.)
  REQUIRE(store.reserve(join_side::left, /*batch_id=*/11, 10).base() == 0);
  REQUIRE(store.reserve(join_side::left, /*batch_id=*/33, 1).base() == 15);
  // Sides are independent address spaces.
  REQUIRE(store.reserve(join_side::right, /*batch_id=*/11, 7).base() == 0);
  // Re-reserving with a different row count is a contract violation.
  REQUIRE_THROWS_AS(store.reserve(join_side::left, /*batch_id=*/11, 9), sirius::internal_exception);
}

TEST_CASE("surrogate store overflow guard rejects before mutating", "[surrogate_groupby]")
{
  constexpr auto max_rows = std::numeric_limits<cudf::size_type>::max();
  surrogate_deferral_store store;
  REQUIRE(store.reserve(join_side::left, 1, max_rows - 5).base() == 0);
  // Would exceed int32 addressing: throws the user-actionable overflow error...
  REQUIRE_THROWS_AS(store.reserve(join_side::left, 2, 10), std::runtime_error);
  // ...without having consumed any address space (check-before-mutate).
  REQUIRE(store.reserve(join_side::left, 3, 5).base() == max_rows - 5);
  // The deduped range is still resolvable.
  REQUIRE(store.reserve(join_side::left, 1, max_rows - 5).base() == 0);
}

TEST_CASE("surrogate store snapshot requires committed sources and release drops them",
          "[surrogate_groupby]")
{
  auto* space = get_default_gpu_space();
  REQUIRE(space != nullptr);
  auto stream = default_stream();

  surrogate_deferral_store store;
  std::vector<int64_t> values{1, 2, 3};
  auto col =
    vector_to_cudf_column<gpu_type_traits<int64_t>>(values, stream, get_resource_ref(*space));
  std::vector<std::unique_ptr<cudf::column>> cols;
  cols.push_back(std::move(col));
  auto batch = sirius::make_data_batch(std::make_unique<cudf::table>(std::move(cols)),
                                       *space,
                                       stream,
                                       sirius::telemetry::batch_telemetry_info{});

  auto first = store.reserve(join_side::left, batch->get_batch_id(), 3);
  REQUIRE(first.base() == 0);
  // Reserved but not committed: the producing task has not succeeded yet.
  REQUIRE_THROWS_AS(store.snapshot(join_side::left), sirius::internal_exception);
  std::move(first).commit(batch->to_read_only());
  // Idempotent commit (first wins): a retried task re-reserves the same id and commits again.
  auto retry = store.reserve(join_side::left, batch->get_batch_id(), 3);
  std::move(retry).commit(batch->to_read_only());
  auto sources = store.snapshot(join_side::left);
  REQUIRE(sources.size() == 1);
  REQUIRE(sources[0].base == 0);
  REQUIRE(sources[0].rows == 3);

  auto const stats = store.release();
  REQUIRE(stats.sources == 1);
  REQUIRE(stats.bytes > 0);
  auto const again = store.release();
  REQUIRE(again.sources == 0);
  // Committing a batch other than the one the token reserved is a contract violation.
  auto mismatched = store.reserve(join_side::left, /*batch_id=*/999, 3);
  REQUIRE_THROWS_AS(std::move(mismatched).commit(batch->to_read_only()),
                    sirius::internal_exception);
}
