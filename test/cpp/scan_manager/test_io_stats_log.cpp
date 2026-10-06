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

// Host-only tests for the per-query `[io_stats]` line: the first-I/O percentile
// helper and the formatter, fed synthetic cucascade::io::queue_stats samples.

#include "catch.hpp"
#include "scan_manager/io_stats_log.hpp"

#include <cucascade/io/types.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>

using namespace std::chrono_literals;
using cucascade::io::first_io_delay_bucket;
using cucascade::io::first_io_delay_buckets;
using cucascade::io::request_class;
using cucascade::io::request_class_index;
using sirius::scan_manager::first_io_percentile;
using sirius::scan_manager::format_io_stats_delta;
using sirius::scan_manager::io_stats_snapshot;

namespace {

using histogram_t = std::array<std::uint64_t, first_io_delay_buckets>;

/// Record @p count first-I/O delays of @p delay into @p cls, as a runner retiring them would.
void record(cucascade::io::class_stats& cls, std::uint64_t count, std::chrono::nanoseconds delay)
{
  cls.first_io_count += count;
  cls.first_io_total += delay * static_cast<std::int64_t>(count);
  cls.first_io_histogram[first_io_delay_bucket(delay)] += count;
  cls.first_io_max = std::max(cls.first_io_max, delay);
}

cucascade::io::class_stats& of(io_stats_snapshot& snapshot, request_class cls)
{
  return snapshot.stats.per_class[request_class_index(cls)];
}

}  // namespace

TEST_CASE("first_io_percentile reports the upper edge of the nearest-rank bucket", "[io_stats_log]")
{
  histogram_t histogram{};
  CHECK(first_io_percentile(histogram, 99) == 0ns);

  // Bucket 0 holds delays below 1 us.
  histogram[first_io_delay_bucket(500ns)] = 1;
  CHECK(first_io_percentile(histogram, 99) == 1us);

  // 99 delays in [4, 8) us and one in [2048, 4096) us.
  histogram                             = {};
  histogram[first_io_delay_bucket(5us)] = 99;
  histogram[first_io_delay_bucket(3ms)] = 1;
  CHECK(first_io_percentile(histogram, 50) == 8us);
  CHECK(first_io_percentile(histogram, 99) == 8us);
  CHECK(first_io_percentile(histogram, 100) == 4096us);

  // With 12 samples the 99th percentile is the largest (rank ceil(11.88) = 12).
  histogram[first_io_delay_bucket(5us)] = 11;
  CHECK(first_io_percentile(histogram, 99) == 4096us);
}

TEST_CASE("first_io_percentile reports the open-ended last bucket by its lower edge",
          "[io_stats_log]")
{
  histogram_t histogram{};
  histogram[first_io_delay_bucket(20s)] = 1;
  REQUIRE(first_io_delay_bucket(20s) == first_io_delay_buckets - 1);
  CHECK(first_io_percentile(histogram, 99) == std::chrono::microseconds{1 << 24});
}

TEST_CASE("format_io_stats_delta diffs two samples into one line", "[io_stats_log]")
{
  io_stats_snapshot before;
  record(of(before, request_class::read), 100, 1ms);
  record(of(before, request_class::write), 5, 1ms);
  before.stats.runners = {{.id = 1, .bytes_submitted = 1000}, {.id = 2, .bytes_submitted = 500}};

  io_stats_snapshot after = before;
  // reset_stats_peaks ran when `before` was taken.
  for (auto& cls : after.stats.per_class) {
    cls.first_io_max = 0ns;
  }
  // 199 latency delays in [32, 64) us and one of 1 ms: p99 is the bucket edge.
  record(of(after, request_class::latency), 199, 40us);
  record(of(after, request_class::latency), 1, 1ms);
  // Ten reads in [1024, 2048) us: the bucket edge overshoots max, so p99 is capped at it.
  record(of(after, request_class::read), 10, 1900us);
  record(of(after, request_class::background), 4, 300us);
  // Runner 1 left, runner 2 submitted 1 GiB more, runner 3 is new and counts from zero.
  after.stats.active_runners = 2;
  after.stats.runners        = {{.id = 2, .bytes_submitted = 500 + (1ULL << 30)},
                                {.id = 3, .bytes_submitted = 214748365}};

  CHECK(format_io_stats_delta(cucascade::io::io_context_type::uring, before, after) ==
        "[io_stats] uring runners=2 bytes_submitted=1.2GiB "
        "latency{n=200 mean=45us p99=64us max=1.0ms} "
        "read{n=10 mean=1.9ms p99=1.9ms max=1.9ms} "
        "background{n=4 mean=300us p99=300us max=300us}");
}

TEST_CASE("format_io_stats_delta omits classes without new first-I/O records", "[io_stats_log]")
{
  io_stats_snapshot before;
  record(of(before, request_class::read), 7, 2ms);
  auto after = before;
  CHECK(format_io_stats_delta(cucascade::io::io_context_type::restful, before, after) ==
        "[io_stats] restful runners=0 bytes_submitted=0B");

  // `write` is logged only when it has records, like every other class.
  record(of(after, request_class::write), 1, 3ms);
  after.stats.runners = {{.id = 9, .bytes_submitted = 3 * 1024}};
  CHECK(format_io_stats_delta(cucascade::io::io_context_type::restful, before, after) ==
        "[io_stats] restful runners=0 bytes_submitted=3.0KiB "
        "write{n=1 mean=3.0ms p99=3.0ms max=3.0ms}");
}
