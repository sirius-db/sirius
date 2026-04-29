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

// test
#include <catch.hpp>

// sirius
#include <op/sirius_physical_operator.hpp>
#include <scan_manager/split_connector.hpp>

// standard library
#include <atomic>
#include <chrono>
#include <future>
#include <memory>
#include <thread>
#include <unordered_set>
#include <vector>

using namespace sirius;
using namespace sirius::scan_manager;
using namespace std::chrono_literals;

namespace {

// Minimal operator_data subclass that carries an integer tag so tests can
// verify FIFO ordering and per-item identity.
struct tagged_split : public op::operator_data {
  explicit tagged_split(int t) : tag(t) {}
  int tag;
};

}  // namespace

TEST_CASE("split_connector - push then get returns FIFO", "[scan_manager][split_connector]")
{
  split_connector connector;

  connector.push_split(std::make_unique<tagged_split>(1));
  connector.push_split(std::make_unique<tagged_split>(2));
  connector.push_split(std::make_unique<tagged_split>(3));

  REQUIRE(connector.has_more_splits());
  REQUIRE_FALSE(connector.is_closed());

  for (int expected : {1, 2, 3}) {
    auto got = connector.get_next_split();
    REQUIRE(got.has_value());
    REQUIRE(*got != nullptr);
    auto* tagged = dynamic_cast<tagged_split*>(got->get());
    REQUIRE(tagged != nullptr);
    REQUIRE(tagged->tag == expected);
  }
}

TEST_CASE("split_connector - close on empty unblocks waiting consumer with nullopt",
          "[scan_manager][split_connector]")
{
  split_connector connector;

  // Park a consumer; it must be blocked until close().
  auto fut = std::async(std::launch::async, [&] { return connector.get_next_split(); });
  REQUIRE(fut.wait_for(50ms) == std::future_status::timeout);

  connector.close();

  REQUIRE(fut.wait_for(1s) == std::future_status::ready);
  auto result = fut.get();
  REQUIRE_FALSE(result.has_value());
  REQUIRE(connector.is_closed());
}

TEST_CASE("split_connector - close after pushes drains then returns nullopt",
          "[scan_manager][split_connector]")
{
  split_connector connector;
  connector.push_split(std::make_unique<tagged_split>(7));
  connector.push_split(std::make_unique<tagged_split>(8));
  connector.close();

  // Closed but not drained — is_closed() should report false until items are pulled.
  REQUIRE_FALSE(connector.is_closed());
  REQUIRE(connector.has_more_splits());

  auto first = connector.get_next_split();
  REQUIRE(first.has_value());
  REQUIRE(dynamic_cast<tagged_split*>(first->get())->tag == 7);

  auto second = connector.get_next_split();
  REQUIRE(second.has_value());
  REQUIRE(dynamic_cast<tagged_split*>(second->get())->tag == 8);

  // Drained — now is_closed() flips, and further pulls return nullopt without blocking.
  REQUIRE(connector.is_closed());
  REQUIRE_FALSE(connector.has_more_splits());

  auto eos = connector.get_next_split();
  REQUIRE_FALSE(eos.has_value());
}

TEST_CASE("split_connector - close is idempotent", "[scan_manager][split_connector]")
{
  split_connector connector;
  connector.close();
  connector.close();  // second call must not deadlock or throw
  REQUIRE(connector.is_closed());
  REQUIRE_FALSE(connector.get_next_split().has_value());
}

TEST_CASE("split_connector - multi-producer / multi-consumer preserves all items",
          "[scan_manager][split_connector]")
{
  // The connector is the choke point between N producers (e.g. file batches in
  // parquet_split_provider) and M consumers (scan operator workers). Verify that
  // every pushed split is delivered exactly once and the consumers terminate
  // after close().
  constexpr int k_producers           = 4;
  constexpr int k_consumers           = 4;
  constexpr int k_pushes_per_producer = 250;
  constexpr int k_total               = k_producers * k_pushes_per_producer;

  split_connector connector;

  std::vector<std::thread> producers;
  producers.reserve(k_producers);
  for (int p = 0; p < k_producers; ++p) {
    producers.emplace_back([&, p] {
      for (int i = 0; i < k_pushes_per_producer; ++i) {
        connector.push_split(std::make_unique<tagged_split>(p * k_pushes_per_producer + i));
      }
    });
  }

  std::mutex collected_mutex;
  std::unordered_set<int> collected;
  std::vector<std::thread> consumers;
  consumers.reserve(k_consumers);
  for (int c = 0; c < k_consumers; ++c) {
    consumers.emplace_back([&] {
      while (true) {
        auto next = connector.get_next_split();
        if (!next.has_value()) { return; }
        auto* tagged = dynamic_cast<tagged_split*>(next->get());
        REQUIRE(tagged != nullptr);
        std::lock_guard<std::mutex> lock(collected_mutex);
        auto [_, inserted] = collected.insert(tagged->tag);
        REQUIRE(inserted);  // every tag delivered exactly once
      }
    });
  }

  for (auto& t : producers) {
    t.join();
  }
  connector.close();
  for (auto& t : consumers) {
    t.join();
  }

  REQUIRE(connector.is_closed());
  REQUIRE(static_cast<int>(collected.size()) == k_total);
}
