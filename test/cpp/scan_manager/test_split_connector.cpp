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

#include "scan_manager/split_connector.hpp"

#include <catch.hpp>

#include <atomic>
#include <chrono>
#include <exception>
#include <stdexcept>
#include <thread>

using sirius::scan_manager::split_connector;
using namespace std::chrono_literals;

TEST_CASE("split_connector interrupt wakes a blocked consumer with nullopt", "[split_connector]")
{
  split_connector connector;
  std::atomic<bool> got_nullopt{false};
  std::thread consumer([&] {
    if (connector.get_next_split() == std::nullopt) { got_nullopt.store(true); }
  });

  std::this_thread::sleep_for(20ms);
  connector.interrupt();
  consumer.join();
  REQUIRE(got_nullopt.load());
}

TEST_CASE("split_connector propagates a recorded producer error past an interrupt",
          "[split_connector]")
{
  // Completion interrupts every scan source. If the producer already failed via
  // close(exception), the pull must rethrow that error, not exit quietly — swallowing it here
  // would let a failed query report success.
  split_connector connector;
  connector.close(std::make_exception_ptr(std::runtime_error("producer failed")));
  connector.interrupt();

  bool threw = false;
  try {
    (void)connector.get_next_split();
  } catch (const std::runtime_error& e) {
    threw = std::string(e.what()) == "producer failed";
  }
  REQUIRE(threw);
}
