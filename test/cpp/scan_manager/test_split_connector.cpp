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
#include "scan_manager/load_balancing_scan_batch_coalescer.hpp"
#include "scan_manager/split_connector.hpp"

#include <cudf/types.hpp>

#include <atomic>
#include <chrono>
#include <functional>
#include <future>
#include <memory>
#include <stop_token>
#include <thread>

using sirius::scan_manager::databatch_provider;
using sirius::scan_manager::load_balancing_scan_batch_coalescer;
using sirius::scan_manager::split_connector;

// The UNION task driver nominates each arm's scan exactly once, on the strength of this contract:
// a creation loop that enters get_next_split() on an open, empty connector stays there until a
// split is pushed or the connector is closed. Make that pop non-blocking and every arm strands
// after its first round.

namespace {

// Serves one batch, then holds the stream open until released, then ends it. push_split is
// private, so the connector is fed through drain_cached_provider, as the sequencer feeds it.
struct gated_provider final : databatch_provider {
  std::shared_ptr<cucascade::data_batch> first;
  std::promise<void> release;
  std::shared_future<void> released = release.get_future().share();
  int calls                         = 0;

  databatch_provider::batch get_next_batch() override
  {
    if (calls++ == 0) { return {first, {}, false}; }
    released.wait();
    return {};
  }
};

bool wait_for(const std::function<bool()>& done,
              std::chrono::seconds timeout = std::chrono::seconds{5})
{
  auto deadline = std::chrono::steady_clock::now() + timeout;
  while (!done()) {
    if (std::chrono::steady_clock::now() > deadline) { return false; }
    std::this_thread::sleep_for(std::chrono::milliseconds{1});
  }
  return true;
}

}  // namespace

TEST_CASE("split_connector::get_next_split blocks on an open, empty connector",
          "[split_connector][scan_manager]")
{
  split_connector connector;
  std::atomic<bool> returned{false};
  std::atomic<bool> got_split{true};
  std::thread consumer([&] {
    auto split = connector.get_next_split();
    got_split.store(split.has_value());
    returned.store(true);
  });
  std::this_thread::sleep_for(std::chrono::milliseconds{50});
  REQUIRE_FALSE(returned.load());  // open and empty: the consumer is still parked
  connector.close();
  consumer.join();
  REQUIRE(returned.load());
  REQUIRE_FALSE(got_split.load());  // woken by close(), not by a split
  REQUIRE(connector.is_closed());
}

TEST_CASE("split_connector::get_next_split wakes on a push, parks again, then ends on close",
          "[split_connector][scan_manager]")
{
  auto memory_manager = sirius::test::operator_utils::initialize_memory_manager();
  auto* gpu_space = memory_manager->get_memory_space(cucascade::memory::Tier::GPU, /*device_id=*/0);
  REQUIRE(gpu_space != nullptr);

  gated_provider provider;
  provider.first = sirius::test::operator_utils::make_numeric_batch<int32_t>(
    *gpu_space, {1}, cudf::type_id::INT32);

  split_connector connector;
  std::stop_source stop;
  std::thread producer([&] {
    load_balancing_scan_batch_coalescer::drain_cached_provider(
      provider, connector, stop.get_token(), /*row_filter_pending=*/false);
  });

  std::atomic<int> pops{0};
  std::atomic<bool> first_ok{false};
  std::atomic<bool> second_empty{false};
  std::thread consumer([&] {
    auto first = connector.get_next_split();
    first_ok.store(first.has_value() && *first != nullptr);
    pops.store(1);
    auto second = connector.get_next_split();
    second_empty.store(!second.has_value());
    pops.store(2);
  });

  REQUIRE(wait_for([&] { return pops.load() == 1; }));  // got the pushed split
  std::this_thread::sleep_for(std::chrono::milliseconds{50});
  REQUIRE(pops.load() == 1);             // parked again
  REQUIRE_FALSE(connector.is_closed());  // open and empty: is_closed() is closed AND drained
  provider.release.set_value();          // provider ends the stream; the drain closes
  producer.join();
  consumer.join();
  REQUIRE(first_ok.load());
  REQUIRE(second_empty.load());  // closed and drained
  REQUIRE(pops.load() == 2);
  REQUIRE(connector.is_closed());
}
