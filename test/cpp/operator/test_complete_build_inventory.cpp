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

#include "dynamic_filter_accumulation_test_utils.hpp"
#include "op/dynamic_filter/complete_build_inventory.hpp"

#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>

#include <rmm/aligned.hpp>

#include <catch.hpp>
#include <cucascade/cudf/host_data_representation.hpp>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <future>
#include <limits>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

using inventory = sirius::op::complete_build_inventory;
using ledger    = sirius::op::build_arrival_ledger;
using geometry  = sirius::op::detail::accumulated_bloom_geometry;
namespace acc   = sirius::test::accumulation;

cudf::column_view empty_column(cudf::data_type type) { return {type, 0, nullptr, nullptr, 0}; }

/// One key array's bytes rounded up to the allocation alignment.
std::size_t aligned(std::size_t raw_bytes)
{
  return rmm::align_up(raw_bytes, rmm::CUDA_ALLOCATION_ALIGNMENT);
}

static_assert(!std::is_copy_constructible_v<inventory>);
static_assert(!std::is_copy_assignable_v<inventory>);
static_assert(std::is_nothrow_move_constructible_v<inventory>);
static_assert(std::is_nothrow_move_assignable_v<inventory>);

}  // namespace

TEST_CASE("arrival ledger certifies exactly the pushed batches as a sorted inventory",
          "[dynamic_filter][multi_partition][inventory]")
{
  acc::fixture fixture;
  auto const first  = fixture.make_batch(0, 0, 7);
  auto const second = fixture.make_batch(0, 100, 0);
  auto const third  = fixture.make_batch(0, 200, 13);
  ledger arrivals;
  arrivals.record(*third);
  arrivals.record(*first);
  arrivals.record(*second);

  auto certified = arrivals.certify({.repository_batches = 3, .partition_count = 4});
  REQUIRE(certified);
  REQUIRE(certified->total_rows() == 20);
  REQUIRE(certified->batches().size() == 3);
  REQUIRE(certified->batches()[0].batch_id < certified->batches()[1].batch_id);
  REQUIRE(certified->batches()[1].batch_id < certified->batches()[2].batch_id);
  REQUIRE(certified->find(first->get_batch_id())->rows == 7);
  REQUIRE(certified->find(second->get_batch_id())->rows == 0);
  REQUIRE(certified->find(third->get_batch_id())->rows == 13);
  REQUIRE(certified->find(third->get_batch_id() + 1000) == nullptr);
  REQUIRE(certified->consistent_type_at(0) == cudf::data_type{cudf::type_id::INT32});
  REQUIRE(certified->consistent_type_at(1) == cudf::data_type{cudf::type_id::INT64});
  REQUIRE_FALSE(certified->consistent_type_at(2));

  SECTION("certification happens at most once")
  {
    REQUIRE_FALSE(arrivals.certify({.repository_batches = 3, .partition_count = 4}));
  }
  SECTION("a push after certification is an invariant violation")
  {
    auto const late = fixture.make_batch(0, 300, 5);
    REQUIRE_THROWS_AS(arrivals.record(*late), std::logic_error);
    SECTION("abandoning the certified ledger makes later pushes harmless")
    {
      arrivals.abandon();
      REQUIRE_NOTHROW(arrivals.record(*late));
    }
  }
  SECTION("a late batch without rows cannot add keys and is ignored")
  {
    auto const empty = fixture.make_batch(0, 300, 0);
    REQUIRE_NOTHROW(arrivals.record(*empty));
    // The ledger stays certified: a late batch with rows still fails.
    REQUIRE_THROWS_AS(arrivals.record(*fixture.make_batch(0, 400, 5)), std::logic_error);
  }
  SECTION("a late batch without rows that cannot be read without blocking still fails")
  {
    auto const empty = fixture.make_batch(0, 300, 0);
    auto writer      = empty->to_mutable();
    REQUIRE_THROWS_AS(arrivals.record(*empty), std::logic_error);
  }
}

TEST_CASE("arrival ledger declines when the repository holds a batch it never saw",
          "[dynamic_filter][multi_partition][inventory]")
{
  acc::fixture fixture;
  auto const recorded = fixture.make_batch(0, 0, 4);
  ledger arrivals;
  arrivals.record(*recorded);
  REQUIRE_FALSE(arrivals.certify({.repository_batches = 2, .partition_count = 2}));
  // A declined ledger ignores later arrivals rather than failing the pushing task.
  REQUIRE_NOTHROW(arrivals.record(*fixture.make_batch(0, 10, 4)));
  REQUIRE_FALSE(arrivals.certify({.repository_batches = 2, .partition_count = 2}));
}

TEST_CASE("arrival ledger is poisoned by unprovable arrivals without blocking",
          "[dynamic_filter][multi_partition][inventory]")
{
  acc::fixture fixture;
  auto const valid = fixture.make_batch(0, 0, 4);
  ledger arrivals;
  arrivals.record(*valid);

  SECTION("a batch that is not GPU resident")
  {
    auto const spilled = fixture.make_batch(0, 10, 4);
    auto* host = fixture.manager->get_memory_spaces_for_tier(cucascade::memory::Tier::HOST).front();
    auto const stream = fixture.gpu(0).acquire_stream();
    {
      auto writable = spilled->to_mutable();
      writable.convert_to<cucascade::host_data_representation>(
        sirius::converter_registry::get(), host, stream);
    }
    stream.sync();
    arrivals.record(*spilled);
  }
  SECTION("a batch that is mutably locked at push")
  {
    auto const locked = fixture.make_batch(0, 10, 4);
    auto writer       = locked->to_mutable();
    auto recorded     = std::async(std::launch::async, [&] { arrivals.record(*locked); });
    // record() must not wait for the writer to release the batch.
    REQUIRE(recorded.wait_for(std::chrono::seconds{10}) == std::future_status::ready);
    recorded.get();
  }
  REQUIRE_FALSE(arrivals.certify({.repository_batches = 2, .partition_count = 2}));
}

TEST_CASE("certified inventories reject a single partition and transfer on move",
          "[dynamic_filter][multi_partition][inventory]")
{
  acc::fixture fixture;
  auto const batch = fixture.make_batch(0, 0, 9);
  {
    ledger single;
    single.record(*batch);
    REQUIRE_FALSE(single.certify({.repository_batches = 1, .partition_count = 1}));
  }
  ledger arrivals;
  arrivals.record(*batch);
  auto source = arrivals.certify({.repository_batches = 1, .partition_count = 3});
  REQUIRE(source);
  inventory moved{std::move(*source)};
  REQUIRE(moved.total_rows() == 9);
  REQUIRE(moved.batches().size() == 1);
}

TEST_CASE("inventory records top-level key types without constraining payload children",
          "[dynamic_filter][multi_partition][inventory]")
{
  acc::fixture fixture;
  auto const integer = empty_column(cudf::data_type{cudf::type_id::INT64});
  auto const decimal = empty_column(cudf::data_type{cudf::type_id::DECIMAL64, -2});
  cudf::column_view const nested{
    cudf::data_type{cudf::type_id::STRUCT}, 0, nullptr, nullptr, 0, 0, {integer, decimal}};
  cudf::column_view const changed{
    cudf::data_type{cudf::type_id::STRUCT}, 0, nullptr, nullptr, 0, 0, {decimal}};
  auto const make = [&](std::vector<cudf::column_view> columns) {
    return sirius::make_data_batch_from_view(cudf::table_view{columns},
                                             std::make_shared<int>(0),
                                             0,
                                             fixture.gpu(0),
                                             fixture.task_stream(0),
                                             sirius::telemetry::batch_telemetry_info{});
  };
  auto const first  = make({integer, nested});
  auto const second = make({integer, changed, decimal});
  ledger arrivals;
  arrivals.record(*first);
  arrivals.record(*second);
  auto certified = arrivals.certify({.repository_batches = 2, .partition_count = 2});
  REQUIRE(certified);
  REQUIRE(certified->consistent_type_at(0) == integer.type());
  REQUIRE(certified->consistent_type_at(1) == nested.type());
  REQUIRE_FALSE(certified->consistent_type_at(2));
}

TEST_CASE("a missing or changed key type stays unavailable while siblings remain usable",
          "[dynamic_filter][multi_partition][inventory]")
{
  acc::fixture fixture;
  auto const integer     = empty_column(cudf::data_type{cudf::type_id::INT32});
  auto const big_integer = empty_column(cudf::data_type{cudf::type_id::INT64});
  auto const make        = [&](std::vector<cudf::column_view> columns) {
    return sirius::make_data_batch_from_view(cudf::table_view{columns},
                                             std::make_shared<int>(0),
                                             0,
                                             fixture.gpu(0),
                                             fixture.task_stream(0),
                                             sirius::telemetry::batch_telemetry_info{});
  };
  auto const first  = make({integer, big_integer});
  auto const second = GENERATE(true, false) ? make({integer}) : make({integer, integer});
  auto const third  = make({integer, big_integer});
  ledger arrivals;
  for (auto const& batch : {first, second, third}) {
    arrivals.record(*batch);
  }
  auto certified = arrivals.certify({.repository_batches = 3, .partition_count = 2});
  REQUIRE(certified);
  REQUIRE(certified->consistent_type_at(0) == integer.type());
  REQUIRE_FALSE(certified->consistent_type_at(1));
}

TEST_CASE("accumulated Bloom geometry uses global rows and sums aligned key arrays",
          "[dynamic_filter][multi_partition][bloom_budget]")
{
  auto small = geometry::try_create(16, 2, 512);
  REQUIRE(small);
  REQUIRE(small->blocks == 1);
  REQUIRE(small->raw_bytes == 32);
  REQUIRE(aligned(small->raw_bytes) == 256);
  REQUIRE(small->arrays_bytes == 512);
  REQUIRE(small->chunk_bytes == 256);
  REQUIRE(small->chunks_per_key() == 1);
  REQUIRE_FALSE(geometry::try_create(16, 2, 511));

  auto next_block = geometry::try_create(129, 2, 1024);
  REQUIRE(next_block);
  REQUIRE(next_block->blocks == 9);
  REQUIRE(next_block->raw_bytes == 288);
  REQUIRE(aligned(next_block->raw_bytes) == 512);
  REQUIRE(next_block->arrays_bytes == 1024);

  auto const maximum = std::numeric_limits<std::size_t>::max();
  REQUIRE_FALSE(geometry::try_create(16, 0, maximum));
  REQUIRE_FALSE(geometry::try_create(16, 1, 0));
  REQUIRE_FALSE(geometry::try_create(maximum, 1, maximum));
  REQUIRE_FALSE(geometry::try_create(16, maximum, maximum));
  REQUIRE_FALSE(geometry::try_create(16, maximum / 256 + 1, maximum));
}

TEST_CASE("accumulated Bloom transfer chunks are aligned and cover non-aligned arrays",
          "[dynamic_filter][multi_partition][bloom_budget]")
{
  constexpr std::size_t mib = std::size_t{1} << 20;
  SECTION("one chunk when the array is below the minimum chunk")
  {
    auto const shape = geometry::try_create(1'000'001, 1, 256 * mib);
    REQUIRE(shape);
    REQUIRE(shape->raw_bytes == 2'000'032);
    REQUIRE(aligned(shape->raw_bytes) == 2'000'128);
    REQUIRE(shape->chunk_bytes == 2'000'128);
    REQUIRE(shape->chunks_per_key() == 1);
  }
  SECTION("an eighth of the array, rounded up to the alignment, with a partial last chunk")
  {
    auto const shape = geometry::try_create(20'971'536, 1, 256 * mib);
    REQUIRE(shape);
    REQUIRE(shape->raw_bytes == 41'943'072);
    REQUIRE(shape->chunk_bytes == 5'243'136);
    REQUIRE(shape->chunk_bytes % 256 == 0);
    REQUIRE(shape->chunks_per_key() == 8);
    auto const last = shape->raw_bytes - 7 * shape->chunk_bytes;
    REQUIRE(last == 5'241'120);
    REQUIRE(last % 32 == 0);
  }
  SECTION("the chunk is clamped to the maximum")
  {
    auto const shape = geometry::try_create(std::size_t{64} << 20, 1, std::size_t{1} << 40);
    REQUIRE(shape);
    REQUIRE(shape->chunk_bytes == geometry::k_max_transfer_chunk_bytes);
  }
}
