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
#include "pipeline/pipeline_memory_history.hpp"

#include <cstddef>
#include <thread>
#include <vector>

using sirius::pipeline::pipeline_memory_history;

TEST_CASE("pipeline_memory_history returns nullopt with no records",
          "[pipeline_memory_history][estimation]")
{
  pipeline_memory_history history;
  REQUIRE_FALSE(history.estimate_peak_memory(100).has_value());
}

TEST_CASE("pipeline_memory_history returns nullopt for zero estimated_bytes",
          "[pipeline_memory_history][estimation]")
{
  pipeline_memory_history history;
  history.record({100, 200, 150});
  REQUIRE_FALSE(history.estimate_peak_memory(0).has_value());
}

TEST_CASE("pipeline_memory_history single record produces correct estimate",
          "[pipeline_memory_history][estimation]")
{
  pipeline_memory_history history;

  // Record: estimated 100 bytes, peaked at 300 bytes → ratio = 3.0
  history.record({100, 300, 200});

  // For a new task with 100 bytes estimated, expect 100 * 3.0 = 300
  auto estimate = history.estimate_peak_memory(100);
  REQUIRE(estimate.has_value());
  REQUIRE(*estimate == 300);
}

TEST_CASE("pipeline_memory_history multiple records are weighted by proximity",
          "[pipeline_memory_history][estimation]")
{
  pipeline_memory_history history;

  // A record with input close to 1000 (ratio = 2.0)
  history.record({1000, 2000, 1500});

  // A record with input far from 1000 (ratio = 5.0)
  history.record({10, 50, 40});

  // For estimated_bytes=1000, the close record (ratio=2.0) should dominate
  // over the far record (ratio=5.0), so the result should be closer to
  // 1000 * 2.0 = 2000 than to 1000 * 5.0 = 5000
  auto estimate = history.estimate_peak_memory(1000);
  REQUIRE(estimate.has_value());
  REQUIRE(*estimate >= 2000);
  REQUIRE(*estimate < 4000);
}

TEST_CASE("pipeline_memory_history skips records with zero estimated_bytes",
          "[pipeline_memory_history][estimation]")
{
  pipeline_memory_history history;

  history.record({0, 500, 300});    // Skipped by the ring: peak estimation divides by the basis
  history.record({100, 300, 200});  // ratio = 3.0

  auto estimate = history.estimate_peak_memory(100);
  REQUIRE(estimate.has_value());
  // Only the second record contributes: 100 * 3.0 = 300
  REQUIRE(*estimate == 300);

  REQUIRE(history.totals().output_records == 2);
  REQUIRE(history.totals().output_bytes == 500);
  REQUIRE(history.totals().ratio_records == 1);
  REQUIRE(history.totals().ratio_input_bytes == 100);
  auto ratio = history.totals().ratio();
  REQUIRE(ratio.has_value());
  REQUIRE(*ratio == Approx(2.0));
}

TEST_CASE("pipeline_memory_history ring buffer evicts oldest records",
          "[pipeline_memory_history][ring_buffer]")
{
  pipeline_memory_history history;

  // Fill with 64 records of ratio=2.0
  for (std::size_t i = 0; i < 64; ++i) {
    history.record({100, 200, 150});
  }

  REQUIRE(history.size() == 64);

  // Add 64 more with ratio=4.0 — should evict all the old ones
  for (std::size_t i = 0; i < 64; ++i) {
    history.record({100, 400, 300});
  }

  REQUIRE(history.size() == 64);

  // Estimate should now reflect ratio=4.0: 100 * 4.0 = 400
  auto estimate = history.estimate_peak_memory(100);
  REQUIRE(estimate.has_value());
  REQUIRE(*estimate == 400);
}

TEST_CASE("pipeline_memory_history is thread-safe for concurrent recording",
          "[pipeline_memory_history][concurrency]")
{
  pipeline_memory_history history;
  constexpr int num_threads        = 4;
  constexpr int records_per_thread = 100;

  std::vector<std::thread> threads;
  threads.reserve(num_threads);
  for (int t = 0; t < num_threads; ++t) {
    threads.emplace_back([&history, t]() {
      for (int i = 0; i < records_per_thread; ++i) {
        auto base = static_cast<std::size_t>((t + 1) * 100);
        history.record({base, base * 2, base + 50});
      }
    });
  }

  for (auto& t : threads) {
    t.join();
  }

  // All records should have been accepted (ring buffer caps at 64)
  REQUIRE(history.size() == 64);

  // Should produce a valid estimate
  auto estimate = history.estimate_peak_memory(200);
  REQUIRE(estimate.has_value());
  REQUIRE(*estimate > 0);

  // Running totals are not capped by the ring buffer.
  REQUIRE(history.totals().ratio_records ==
          static_cast<std::size_t>(num_threads) * static_cast<std::size_t>(records_per_thread));
}

TEST_CASE("pipeline_memory_history has no output/input ratio without records",
          "[pipeline_memory_history][ratio]")
{
  pipeline_memory_history history;
  REQUIRE_FALSE(history.totals().ratio().has_value());
  REQUIRE(history.totals().output_bytes == 0);
}

TEST_CASE("pipeline_memory_history aggregates the output/input ratio over all records",
          "[pipeline_memory_history][ratio]")
{
  pipeline_memory_history history;

  history.record({100, 200, 60});
  history.record({300, 600, 140});

  auto ratio = history.totals().ratio();
  REQUIRE(ratio.has_value());
  REQUIRE(*ratio == Approx(0.5));

  REQUIRE(history.totals().output_bytes == 200);
  REQUIRE(history.totals().ratio_input_bytes == 400);
  REQUIRE(history.totals().ratio_records == 2);
}

TEST_CASE("pipeline_memory_history totals survive ring-buffer eviction",
          "[pipeline_memory_history][ratio][ring_buffer]")
{
  pipeline_memory_history history;

  for (std::size_t i = 0; i < 200; ++i) {
    history.record({100, 200, 25});
  }

  REQUIRE(history.size() == 64);
  REQUIRE(history.totals().ratio_records == 200);
  REQUIRE(history.totals().output_bytes == std::size_t{200} * 25);

  auto ratio = history.totals().ratio();
  REQUIRE(ratio.has_value());
  REQUIRE(*ratio == Approx(0.25));
}

TEST_CASE("pipeline_memory_history excludes failed tasks from the output/input ratio",
          "[pipeline_memory_history][ratio]")
{
  pipeline_memory_history history;

  history.record({100, 200, 50});
  history.record_on_failure(400, 900);

  auto ratio = history.totals().ratio();
  REQUIRE(ratio.has_value());
  REQUIRE(*ratio == Approx(0.5));
  REQUIRE(history.totals().ratio_records == 1);
  REQUIRE(history.totals().ratio_input_bytes == 100);
}

TEST_CASE("pipeline_memory_history excludes resumed tasks from the output/input ratio",
          "[pipeline_memory_history][ratio]")
{
  pipeline_memory_history history;

  history.record({100, 200, 50});
  // A resumed task's intermediate input is not a valid pipeline ratio basis.
  history.record({10, 200, 50, /*ratio_eligible=*/false});

  REQUIRE(history.size() == 2);

  auto ratio = history.totals().ratio();
  REQUIRE(ratio.has_value());
  REQUIRE(*ratio == Approx(0.5));
  REQUIRE(history.totals().ratio_records == 1);
  REQUIRE(history.totals().ratio_input_bytes == 100);

  // Resumed-task output still contributes to the pipeline total.
  REQUIRE(history.totals().output_records == 2);
  REQUIRE(history.totals().output_bytes == 100);
}
