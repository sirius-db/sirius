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

// Host-only tests for the `[uring_gauges]` line formatter, fed synthetic
// uring_reactor::gauges snapshots (no reactor, no sampler thread).

#include "catch.hpp"
#include "scan_manager/uring_gauges_sampler.hpp"

#include <cucascade/io/uring/uring_reactor.hpp>

#include <cstddef>
#include <optional>
#include <string>

using gauges = cucascade::io::uring::uring_reactor::gauges;
using sirius::scan_manager::format_gauges_line;

TEST_CASE("uring gauges: an idle reactor logs nothing", "[scan_manager][uring_gauges]")
{
  gauges const zero{};
  CHECK_FALSE(format_gauges_line(0, zero, nullptr, 0.25).has_value());

  // Cumulative counters that did not move over the window are idle too.
  gauges steady{};
  steady.requests_started = 42;
  steady.bytes_submitted  = std::size_t{7} << 20;
  CHECK_FALSE(format_gauges_line(1, steady, &steady, 0.25).has_value());
}

TEST_CASE("uring gauges: the first window reports depth but no rate",
          "[scan_manager][uring_gauges]")
{
  gauges g{};
  g.inflight_ops            = 3;
  g.max_inflight_ops        = 5;
  g.pending_ops             = 2;
  g.active_remaining_slices = 1;
  g.queued_requests         = 4;
  g.queued_bytes            = std::size_t{3} << 20;
  g.requests_started        = 10;
  g.bytes_submitted         = std::size_t{64} << 20;

  // Without a previous window the cumulative counters have no delta to report.
  auto const line = format_gauges_line(2, g, nullptr, 0.5);
  REQUIRE(line.has_value());
  CHECK(*line ==
        "[uring_gauges] reactor=2 inflight=3 max_inflight=5 pending_ops=2 active_slices=1 "
        "queued_requests=4 queued_MiB=3 started=0 MiB_s=0"
        " dq_n=0 dq_sum_ms=0.0 dq_max_ms=0.0 dq_hist=-"
        " pq_n=0 pq_sum_ms=0.0 pq_max_ms=0.0 pq_hist=-");
}

TEST_CASE("uring gauges: deltas, rate and per-class queue delay", "[scan_manager][uring_gauges]")
{
  gauges previous{};
  previous.requests_started = 4;
  previous.bytes_submitted  = std::size_t{10} << 20;

  gauges g{};
  g.requests_started = 10;
  g.bytes_submitted  = std::size_t{110} << 20;

  auto& demand         = g.queue_delay[0];
  demand.count         = 3;
  demand.sum_ns        = 4'500'000;
  demand.max_ns        = 2'000'000;
  demand.histogram[3]  = 1;
  demand.histogram[11] = 2;

  auto& prefetch        = g.queue_delay[1];
  prefetch.count        = 1;
  prefetch.sum_ns       = 300'000;
  prefetch.max_ns       = 300'000;
  prefetch.histogram[9] = 1;

  // 100 MiB over 2 s; histogram buckets render as bucket:count.
  auto const line = format_gauges_line(0, g, &previous, 2.0);
  REQUIRE(line.has_value());
  CHECK(*line ==
        "[uring_gauges] reactor=0 inflight=0 max_inflight=0 pending_ops=0 active_slices=0 "
        "queued_requests=0 queued_MiB=0 started=6 MiB_s=50"
        " dq_n=3 dq_sum_ms=4.5 dq_max_ms=2.0 dq_hist=3:1,11:2"
        " pq_n=1 pq_sum_ms=0.3 pq_max_ms=0.3 pq_hist=9:1");

  // A queue-delay record alone (no movement otherwise) is not idle.
  gauges delayed_only{};
  delayed_only.queue_delay[1].count = 1;
  CHECK(format_gauges_line(0, delayed_only, &delayed_only, 0.25).has_value());
}
