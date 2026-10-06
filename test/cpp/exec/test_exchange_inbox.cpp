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

#include "catch.hpp"
#include "data/data_batch_utils.hpp"
#include "data/data_repository_manager_registry.hpp"
#include "exec/exchange_inbox.hpp"
#include "exec/exchange_staging.hpp"
#include "exec/query_lifecycle_registry.hpp"
#include "sirius/exception.hpp"

#include <cudf/column/column_factories.hpp>
#include <cudf/table/table.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <cucascade/data/data_batch.hpp>
#include <cucascade/data/data_repository.hpp>
#include <cucascade/memory/memory_space.hpp>

#include <cstdint>
#include <memory>
#include <vector>

using Catch::Matchers::ContainsSubstring;
using Catch::Matchers::MessageMatches;
using cucascade::memory::memory_space;
using namespace sirius::exec;

namespace {

constexpr cudf::size_type rows = 100;

cucascade::memory::gpu_memory_space_config gpu_config()
{
  cucascade::memory::gpu_memory_space_config config;
  config.device_id              = 0;
  config.memory_capacity        = std::size_t{64} << 20;
  config.per_stream_reservation = false;
  return config;
}

std::shared_ptr<cucascade::data_batch> int64_batch(memory_space& gpu)
{
  auto const stream = cudf::get_default_stream();
  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(cudf::make_fixed_width_column(cudf::data_type{cudf::type_id::INT64},
                                                  rows,
                                                  cudf::mask_state::UNALLOCATED,
                                                  stream,
                                                  gpu.get_default_allocator()));
  return sirius::make_data_batch(cudf::table{std::move(columns)}, gpu, stream, {});
}

void refused(exchange_inbox& exchange, std::uint64_t token)
{
  CHECK_THROWS_MATCHES(exchange.take_batch(token),
                       sirius::invalid_input_exception,
                       MessageMatches(ContainsSubstring("holds no received batch")));
}

}  // namespace

TEST_CASE("exchange_inbox hands an adopted batch to take_batch once", "[exchange_inbox]")
{
  memory_space gpu(gpu_config());
  exchange_inbox exchange(gpu);
  auto const batch = int64_batch(gpu);
  auto const first = exchange.adopt(batch);
  auto const other = exchange.adopt(int64_batch(gpu));
  CHECK(first >= 1);
  CHECK(other != first);
  CHECK(exchange.outstanding() == 2);

  refused(exchange, 0);
  CHECK_THROWS_AS(exchange.adopt(nullptr), sirius::invalid_input_exception);
  CHECK(exchange.take_batch(first) == batch);
  refused(exchange, first);
  CHECK(exchange.outstanding() == 1);
}

TEST_CASE("exchange_inbox release is idempotent and close frees every token", "[exchange_inbox]")
{
  memory_space gpu(gpu_config());
  auto const available = gpu.get_available_memory();
  {
    exchange_inbox exchange(gpu);
    auto const released = exchange.adopt(int64_batch(gpu));
    (void)exchange.adopt(int64_batch(gpu));
    exchange.release(released);
    exchange.release(released);
    exchange.release(0);
    refused(exchange, released);
    CHECK(exchange.outstanding() == 1);

    exchange.close();
    exchange.close();
    CHECK_THROWS_AS(exchange.outstanding(), sirius::invalid_input_exception);
    CHECK_THROWS_AS(exchange.adopt(int64_batch(gpu)), sirius::invalid_input_exception);
    CHECK_THROWS_AS(exchange.release(1), sirius::invalid_input_exception);
  }
  CHECK(gpu.get_available_memory() == available);
}

TEST_CASE("exchange_inbox runs an adopted token's on_gone once, however it goes",
          "[exchange_inbox]")
{
  memory_space gpu(gpu_config());
  exchange_inbox exchange(gpu);
  int taken = 0, released = 0, closed = 0;
  auto const take    = exchange.adopt(int64_batch(gpu), [&] { ++taken; });
  auto const release = exchange.adopt(int64_batch(gpu), [&] { ++released; });
  (void)exchange.adopt(int64_batch(gpu), [&] { ++closed; });

  (void)exchange.take_batch(take);
  exchange.release(release);
  exchange.release(release);
  refused(exchange, take);
  exchange.close();
  CHECK(taken == 1);
  CHECK(released == 1);
  CHECK(closed == 1);
}

TEST_CASE("exchange_inbox shows held batches to the downgrade executor once spill is on",
          "[exchange_inbox]")
{
  memory_space gpu(gpu_config());
  exchange_inbox exchange(gpu);
  sirius::data::data_repository_manager_registry registry;
  exchange_staging staging(registry);
  exchange.enable_spill(staging);

  auto const taken    = exchange.adopt(int64_batch(gpu));
  auto const released = exchange.adopt(int64_batch(gpu));
  (void)exchange.adopt(int64_batch(gpu));
  // What the downgrade executor's sweep sees under the reserved query id.
  auto const manager = registry.get(exchange_staging::query_id);
  REQUIRE(manager);
  REQUIRE(manager->get_repositories().size() == 1);
  auto const& held = *manager->get_repositories().front();
  CHECK(held.get_batch_ids().size() == 3);

  CHECK(exchange.take_batch(taken)->to_read_only().get_current_tier() ==
        cucascade::memory::Tier::GPU);
  exchange.release(released);
  CHECK(held.get_batch_ids().size() == 1);
  exchange.close();
  CHECK(held.get_batch_ids().empty());
}

TEST_CASE("exchange_staging opens its query so the downgrade executor can borrow it",
          "[exchange_inbox]")
{
  sirius::data::data_repository_manager_registry registry;
  sirius::exec::query_lifecycle_registry lifecycle;
  // Without the lifecycle, the downgrade sweep's try_acquire_work fails and skips every exchange
  // batch: held received batches, parked sender output and stream inputs.
  CHECK_FALSE(lifecycle.try_acquire_work(exchange_staging::query_id));
  exchange_staging staging(registry, &lifecycle);
  CHECK(lifecycle.try_acquire_work(exchange_staging::query_id));
  CHECK(registry.get(exchange_staging::query_id));
}

TEST_CASE("exchange_staging lists tracked repositories until they are gone", "[exchange_inbox]")
{
  memory_space gpu(gpu_config());
  sirius::data::data_repository_manager_registry registry;
  exchange_staging staging(registry);
  auto const manager = registry.get(exchange_staging::query_id);
  REQUIRE(manager);
  auto* const view = manager->get_repositories().front();

  auto repository = std::make_shared<cucascade::shared_data_repository>();
  auto batch      = int64_batch(gpu);
  auto const id   = batch->get_batch_id();
  repository->add_data_batch(batch);
  staging.track(repository);
  CHECK(staging.tracked() == 1);
  CHECK(view->get_batch_ids() == std::vector<std::uint64_t>{id});
  CHECK(view->get_data_batch_by_id(id) == batch);
  CHECK(view->pop_next_data_batch() == nullptr);  // the view owns nothing

  repository.reset();
  CHECK(staging.tracked() == 0);
  CHECK(view->get_batch_ids().empty());
}
