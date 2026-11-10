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

#include "compression/simpatico_compressed_representation.hpp"
#include "data/sirius_converter_registry.hpp"
#include "dynamic_filter_accumulation_test_utils.hpp"
#include "op/dynamic_filter/complete_build_inventory.hpp"

#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>

#include <rmm/aligned.hpp>

#include <catch.hpp>
#include <cucascade/cudf/host_data_representation.hpp>
#include <cucascade/data/disk_data_representation.hpp>
#include <cucascade/memory/reservation_manager_configurator.hpp>

#include <chrono>
#include <cstddef>
#include <filesystem>
#include <future>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string_view>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

namespace {

using inventory = sirius::op::complete_build_inventory;
using ledger    = sirius::op::build_arrival_ledger;
using sirius::op::construction_decline;
using sirius::op::detail::accumulated_bloom_chunk_bytes;
namespace acc = sirius::test::accumulation;

cudf::column_view empty_column(cudf::data_type type) { return {type, 0, nullptr, nullptr, 0}; }

/// The reason @p result declined, or none when it certified an inventory.
std::optional<construction_decline> decline_of(sirius::op::certified_build_input const& result)
{
  auto const* decline = std::get_if<construction_decline>(&result);
  return decline ? std::optional{*decline} : std::nullopt;
}

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
          "[dynamic_filter][construction][inventory]")
{
  acc::fixture fixture;
  auto const first  = fixture.make_batch(0, 0, 7);
  auto const second = fixture.make_batch(0, 100, 0);
  auto const third  = fixture.make_batch(0, 200, 13);
  ledger arrivals;
  arrivals.record(*third);
  arrivals.record(*first);
  arrivals.record(*second);

  auto const result     = arrivals.certify({.repository_batches = 3});
  auto const* certified = std::get_if<inventory>(&result);
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
    REQUIRE(decline_of(arrivals.certify({.repository_batches = 3})) ==
            construction_decline::inventory_mismatch);
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
          "[dynamic_filter][construction][inventory]")
{
  acc::fixture fixture;
  auto const recorded = fixture.make_batch(0, 0, 4);
  ledger arrivals;
  arrivals.record(*recorded);
  REQUIRE(decline_of(arrivals.certify({.repository_batches = 2})) ==
          construction_decline::inventory_mismatch);
  // A declined ledger ignores later arrivals rather than failing the pushing task.
  REQUIRE_NOTHROW(arrivals.record(*fixture.make_batch(0, 10, 4)));
  REQUIRE(decline_of(arrivals.certify({.repository_batches = 2})) ==
          construction_decline::inventory_mismatch);
}

TEST_CASE("arrival ledger is poisoned by unprovable arrivals without blocking",
          "[dynamic_filter][construction][inventory]")
{
  acc::fixture fixture;
  auto const valid = fixture.make_batch(0, 0, 4);
  ledger arrivals;
  arrivals.record(*valid);

  SECTION("a batch that is mutably locked at push")
  {
    auto const locked = fixture.make_batch(0, 10, 4);
    auto writer       = locked->to_mutable();
    auto recorded     = std::async(std::launch::async, [&] { arrivals.record(*locked); });
    // record() must not wait for the writer to release the batch.
    REQUIRE(recorded.wait_for(std::chrono::seconds{10}) == std::future_status::ready);
    recorded.get();
  }
  REQUIRE(decline_of(arrivals.certify({.repository_batches = 2})) ==
          construction_decline::unreadable_arrival);
}

namespace {

/// Spills @p batch to host memory, as the downgrade executor would.
void spill_to_host(acc::fixture const& fixture, acc::batch_ptr const& batch)
{
  auto* host = fixture.manager->get_memory_spaces_for_tier(cucascade::memory::Tier::HOST).front();
  auto const stream = fixture.gpu(0).acquire_stream();
  {
    auto writable = batch->to_mutable();
    writable.convert_to<cucascade::host_data_representation>(
      sirius::converter_registry::get(), host, stream);
  }
  stream.sync();
}

}  // namespace

TEST_CASE("a spilled host arrival certifies with the facts a GPU arrival records",
          "[dynamic_filter][construction][inventory]")
{
  acc::fixture fixture;
  auto const resident = fixture.make_batch(0, 0, 4);
  auto const spilled  = fixture.make_batch(0, 10, 6);
  spill_to_host(fixture, spilled);
  ledger arrivals;
  arrivals.record(*resident);
  arrivals.record(*spilled);
  auto certified = arrivals.certify({.repository_batches = 2});
  REQUIRE_FALSE(decline_of(certified));
  auto const& complete = std::get<inventory>(certified);
  REQUIRE(complete.total_rows() == 10);
  auto const index =
    static_cast<std::size_t>(complete.find(spilled->get_batch_id()) - complete.batches().data());
  auto const host_columns = complete.batch_columns(index);
  auto const gpu_columns  = complete.batch_columns(1 - index);
  REQUIRE(host_columns.size() == gpu_columns.size());
  for (std::size_t column = 0; column < host_columns.size(); ++column) {
    CHECK(host_columns[column].carrier == gpu_columns[column].carrier);
    CHECK(host_columns[column].null_count == gpu_columns[column].null_count);
  }
}

TEST_CASE("after certification a late empty host arrival is ignored, one with rows throws",
          "[dynamic_filter][construction][inventory]")
{
  acc::fixture fixture;
  auto const batch = fixture.make_batch(0, 0, 4);
  ledger arrivals;
  arrivals.record(*batch);
  REQUIRE_FALSE(decline_of(arrivals.certify({.repository_batches = 1})));
  auto const empty = fixture.make_batch(0, 100, 0);
  spill_to_host(fixture, empty);
  REQUIRE_NOTHROW(arrivals.record(*empty));
  auto const late = fixture.make_batch(0, 200, 3);
  spill_to_host(fixture, late);
  REQUIRE_THROWS_AS(arrivals.record(*late), std::logic_error);
}

TEST_CASE("certified inventories need no downstream partition count and transfer on move",
          "[dynamic_filter][construction][inventory]")
{
  acc::fixture fixture;
  auto const batch = fixture.make_batch(0, 0, 9);
  {
    ledger single;
    single.record(*batch);
    REQUIRE_FALSE(decline_of(single.certify({.repository_batches = 1})));
  }
  ledger arrivals;
  arrivals.record(*batch);
  auto source = arrivals.certify({.repository_batches = 1});
  REQUIRE(std::holds_alternative<inventory>(source));
  inventory moved{std::move(std::get<inventory>(source))};
  REQUIRE(moved.total_rows() == 9);
  REQUIRE(moved.batches().size() == 1);
}

TEST_CASE("inventory records top-level key types without constraining payload children",
          "[dynamic_filter][construction][inventory]")
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
  auto const result     = arrivals.certify({.repository_batches = 2});
  auto const* certified = std::get_if<inventory>(&result);
  REQUIRE(certified);
  REQUIRE(certified->consistent_type_at(0) == integer.type());
  REQUIRE(certified->consistent_type_at(1) == nested.type());
  REQUIRE_FALSE(certified->consistent_type_at(2));
}

TEST_CASE("a missing or changed key type stays unavailable while siblings remain usable",
          "[dynamic_filter][construction][inventory]")
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
  auto const result     = arrivals.certify({.repository_batches = 3});
  auto const* certified = std::get_if<inventory>(&result);
  REQUIRE(certified);
  REQUIRE(certified->consistent_type_at(0) == integer.type());
  REQUIRE_FALSE(certified->consistent_type_at(1));
}

namespace {

/// A compressed representation that exposes only its row count, like a compressed spill.
class fake_compressed_representation final : public sirius::simpatico_compressed_representation {
 public:
  explicit fake_compressed_representation(cucascade::memory::memory_space& space)
    : simpatico_compressed_representation(space)
  {
  }
  [[nodiscard]] std::int64_t num_rows() const noexcept override { return 4; }
  [[nodiscard]] std::size_t get_size_in_bytes() const override { return 0; }
  [[nodiscard]] std::size_t get_uncompressed_data_size_in_bytes() const override { return 0; }
  [[nodiscard]] std::unique_ptr<cucascade::idata_representation> clone(
    ::cuda::stream_ref) const override
  {
    return std::make_unique<fake_compressed_representation>(get_memory_space());
  }
};

/// A batch of @p rows rows in GPU @p space: an all-null INT32 column and a valid DECIMAL64(-2)
/// column.
acc::batch_ptr make_nullable_batch(cucascade::memory::memory_space& space, cudf::size_type rows)
{
  auto const stream = space.acquire_stream();
  auto const mr     = space.get_default_allocator();
  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(cudf::make_numeric_column(
    cudf::data_type{cudf::type_id::INT32}, rows, cudf::mask_state::ALL_NULL, stream, mr));
  columns.push_back(cudf::make_fixed_point_column(cudf::data_type{cudf::type_id::DECIMAL64, -2},
                                                  rows,
                                                  cudf::mask_state::UNALLOCATED,
                                                  stream,
                                                  mr));
  auto table = std::make_unique<cudf::table>(std::move(columns));
  stream.sync();
  return sirius::make_data_batch(
    std::move(*table), space, stream, sirius::telemetry::batch_telemetry_info{});
}

/// `make_nullable_batch` on the first GPU of @p fixture.
acc::batch_ptr make_nullable_batch(acc::fixture const& fixture, cudf::size_type rows)
{
  return make_nullable_batch(fixture.gpu(0), rows);
}

auto const k_int32     = cudf::data_type{cudf::type_id::INT32};
auto const k_decimal64 = cudf::data_type{cudf::type_id::DECIMAL64, -2};

/// The facts @p batch's arrival reports; fails the test on a decline.
sirius::op::arrival_facts facts_of(cucascade::data_batch& batch)
{
  auto arrival = sirius::op::read_arrival(batch);
  REQUIRE(std::holds_alternative<sirius::op::arrival_facts>(arrival));
  return std::get<sirius::op::arrival_facts>(std::move(arrival));
}

}  // namespace

TEST_CASE("an arrival reports each column's carrier and null count from GPU and host tables",
          "[dynamic_filter][construction][inventory]")
{
  acc::fixture fixture;
  auto const batch = make_nullable_batch(fixture, 5);
  std::vector<sirius::op::column_facts> const expected{{k_int32, 5}, {k_decimal64, 0}};

  auto const gpu = facts_of(*batch);
  CHECK(gpu.tier == cucascade::memory::Tier::GPU);
  CHECK(gpu.rows == 5);
  CHECK(gpu.columns == expected);

  auto* host = fixture.manager->get_memory_spaces_for_tier(cucascade::memory::Tier::HOST).front();
  auto const stream = fixture.gpu(0).acquire_stream();
  {
    auto writable = batch->to_mutable();
    writable.convert_to<cucascade::host_data_representation>(
      sirius::converter_registry::get(), host, stream);
  }
  stream.sync();
  auto const spilled = facts_of(*batch);
  CHECK(spilled.tier == cucascade::memory::Tier::HOST);
  CHECK(spilled.rows == 5);
  CHECK(spilled.columns == expected);
}

namespace {

/// One GPU whose memory manager also spills to a disk tier in a temporary directory.
struct disk_memory {
  std::filesystem::path directory;
  std::unique_ptr<sirius::memory::sirius_memory_reservation_manager> manager;
  cucascade::memory::memory_space* gpu  = nullptr;
  cucascade::memory::memory_space* disk = nullptr;

  explicit disk_memory(std::string_view name)
    : directory{std::filesystem::temp_directory_path() / name}
  {
    std::filesystem::create_directories(directory);
    sirius::converter_registry::reset_for_testing();
    cucascade::memory::reservation_manager_configurator builder;
    builder.set_number_of_gpus(1)
      .set_gpu_usage_limit(std::size_t{1} << 30)
      .set_reservation_fraction_per_gpu(0.75)
      .set_per_numa_region_capacity(std::size_t{1} << 30)
      .use_gpu_id_as_host_id()
      .set_reservation_fraction_per_numa_region(0.75)
      .set_disk_mounting_point(0, std::size_t{4} << 30, directory.string());
    manager = std::make_unique<sirius::memory::sirius_memory_reservation_manager>(builder.build());
    sirius::converter_registry::initialize();
    gpu  = manager->get_memory_space(cucascade::memory::Tier::GPU, 0);
    disk = const_cast<cucascade::memory::memory_space*>(
      manager->get_memory_spaces_for_tier(cucascade::memory::Tier::DISK).front());
    REQUIRE(gpu != nullptr);
  }
  disk_memory(disk_memory const&)            = delete;
  disk_memory& operator=(disk_memory const&) = delete;
  ~disk_memory()
  {
    manager.reset();
    std::filesystem::remove_all(directory);
  }

  /// Moves @p batch to the disk tier, as the downgrade executor would.
  void spill(cucascade::data_batch& batch) const
  {
    auto const stream = gpu->acquire_stream();
    {
      auto writable = batch.to_mutable();
      writable.convert_to<cucascade::disk_data_representation>(
        sirius::converter_registry::get(), disk, stream);
    }
    stream.sync();
  }
};

}  // namespace

TEST_CASE("after certification a late empty disk arrival is ignored, one with rows throws",
          "[dynamic_filter][construction][inventory]")
{
  disk_memory const memory{"sirius_test_late_disk_arrival"};
  auto const batch = make_nullable_batch(*memory.gpu, 4);
  ledger arrivals;
  arrivals.record(*batch);
  REQUIRE_FALSE(decline_of(arrivals.certify({.repository_batches = 1})));
  auto const empty = make_nullable_batch(*memory.gpu, 0);
  memory.spill(*empty);
  REQUIRE(facts_of(*empty).tier == cucascade::memory::Tier::DISK);
  REQUIRE_NOTHROW(arrivals.record(*empty));
  auto const late = make_nullable_batch(*memory.gpu, 3);
  memory.spill(*late);
  REQUIRE_THROWS_AS(arrivals.record(*late), std::logic_error);
}

TEST_CASE("an arrival reports each column's carrier and null count from a disk table",
          "[dynamic_filter][construction][inventory]")
{
  disk_memory const memory{"sirius_test_arrival_facts"};
  auto const batch = make_nullable_batch(*memory.gpu, 3);
  memory.spill(*batch);
  auto const facts = facts_of(*batch);
  CHECK(facts.tier == cucascade::memory::Tier::DISK);
  CHECK(facts.rows == 3);
  CHECK(facts.columns == std::vector<sirius::op::column_facts>{{k_int32, 3}, {k_decimal64, 0}});
}

TEST_CASE("a compressed arrival closes the ledger with its own reason",
          "[dynamic_filter][construction][inventory]")
{
  acc::fixture fixture;
  auto* host = fixture.manager->get_memory_spaces_for_tier(cucascade::memory::Tier::HOST).front();
  auto const compressed =
    cucascade::data_batch::make(sirius::get_next_batch_id(),
                                std::make_unique<fake_compressed_representation>(
                                  const_cast<cucascade::memory::memory_space&>(*host)));
  REQUIRE(std::get<construction_decline>(sirius::op::read_arrival(*compressed)) ==
          construction_decline::compressed_arrival);

  ledger arrivals;
  arrivals.record(*fixture.make_batch(0, 0, 4));
  arrivals.record(*compressed);
  REQUIRE(decline_of(arrivals.certify({.repository_batches = 2})) ==
          construction_decline::compressed_arrival);
}

TEST_CASE("an inventory keeps every batch's columns and summarizes them for policy",
          "[dynamic_filter][construction][inventory]")
{
  acc::fixture fixture;
  auto const nullable = make_nullable_batch(fixture, 6);
  auto const plain    = fixture.make_batch(0, 0, 9);  // INT32, INT64 without nulls
  ledger arrivals;
  arrivals.record(*plain);
  arrivals.record(*nullable);
  auto const result     = arrivals.certify({.repository_batches = 2});
  auto const* certified = std::get_if<inventory>(&result);
  REQUIRE(certified);

  // `batch_columns` follows `batches()`, which is sorted by batch ID.
  for (std::size_t index = 0; index < certified->batches().size(); ++index) {
    auto const id      = certified->batches()[index].batch_id;
    auto const columns = certified->batch_columns(index);
    if (id == nullable->get_batch_id()) {
      CHECK(std::vector(columns.begin(), columns.end()) ==
            std::vector<sirius::op::column_facts>{{k_int32, 6}, {k_decimal64, 0}});
    } else {
      CHECK(std::vector(columns.begin(), columns.end()) ==
            std::vector<sirius::op::column_facts>{{k_int32, 0},
                                                  {cudf::data_type{cudf::type_id::INT64}, 0}});
    }
  }
  CHECK(certified->batch_columns(2).empty());

  auto const facts = certified->facts();
  CHECK(facts.total_rows == 15);
  CHECK(facts.max_batch_rows == 9);
  REQUIRE(facts.columns.size() == 2);
  CHECK(facts.columns[0].carriers == std::vector{k_int32});
  CHECK(facts.columns[0].null_rows == 6);
  // Carriers are listed in batch-ID order; the nullable batch was created first.
  REQUIRE(nullable->get_batch_id() < plain->get_batch_id());
  CHECK(facts.columns[1].carriers ==
        std::vector{k_decimal64, cudf::data_type{cudf::type_id::INT64}});
  CHECK(facts.columns[1].null_rows == 0);
  CHECK(certified->consistent_type_at(0) == k_int32);
  CHECK_FALSE(certified->consistent_type_at(1));
}

TEST_CASE("a column absent from some batch has no carriers and no null count",
          "[dynamic_filter][construction][inventory]")
{
  acc::fixture fixture;
  auto const make = [&](std::vector<cudf::column_view> columns) {
    return sirius::make_data_batch_from_view(cudf::table_view{columns},
                                             std::make_shared<int>(0),
                                             0,
                                             fixture.gpu(0),
                                             fixture.task_stream(0),
                                             sirius::telemetry::batch_telemetry_info{});
  };
  auto const integer = empty_column(k_int32);
  ledger arrivals;
  arrivals.record(*make({integer, integer}));
  arrivals.record(*make({integer}));
  auto const result     = arrivals.certify({.repository_batches = 2});
  auto const* certified = std::get_if<inventory>(&result);
  REQUIRE(certified);
  auto const facts = certified->facts();
  REQUIRE(facts.columns.size() == 2);
  CHECK(facts.columns[0].carriers == std::vector{k_int32});
  CHECK(facts.columns[1].carriers.empty());
  CHECK_FALSE(facts.columns[1].null_rows);
}

TEST_CASE("accumulated Bloom transfer chunks are aligned and cover non-aligned arrays",
          "[dynamic_filter][construction][bloom_budget]")
{
  // The bytes of one Bloom array sized on @p rows keys.
  auto const raw_bytes_for = [](std::size_t rows) {
    return sirius::op::bloom_blocks_for(rows) * sirius::op::bloom_bytes_per_block;
  };
  SECTION("one chunk when the array is below the minimum chunk")
  {
    auto const raw_bytes = raw_bytes_for(1'000'001);
    REQUIRE(raw_bytes == 2'000'032);
    REQUIRE(aligned(raw_bytes) == 2'000'128);
    REQUIRE(accumulated_bloom_chunk_bytes(raw_bytes) == 2'000'128);
  }
  SECTION("an eighth of the array, rounded up to the alignment, with a partial last chunk")
  {
    auto const raw_bytes = raw_bytes_for(20'971'536);
    auto const chunk     = accumulated_bloom_chunk_bytes(raw_bytes);
    REQUIRE(raw_bytes == 41'943'072);
    REQUIRE(chunk == 5'243'136);
    REQUIRE(chunk % 256 == 0);
    REQUIRE(::cuda::ceil_div(raw_bytes, chunk) == 8);
    auto const last = raw_bytes - 7 * chunk;
    REQUIRE(last == 5'241'120);
    REQUIRE(last % 32 == 0);
  }
  SECTION("the chunk is clamped to the maximum")
  {
    REQUIRE(accumulated_bloom_chunk_bytes(raw_bytes_for(std::size_t{64} << 20)) ==
            sirius::op::detail::k_max_bloom_transfer_chunk_bytes);
  }
}
