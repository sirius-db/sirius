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

#include "scan_manager/io_stats_log.hpp"

#include <algorithm>
#include <cstddef>
#include <format>
#include <iterator>
#include <numeric>
#include <string_view>

namespace sirius::scan_manager {

namespace {

/// @p after - @p before of a monotonic counter, clamped at zero.
template <typename T>
T counter_delta(T after, T before) noexcept
{
  return after > before ? after - before : T{};
}

/// "250us", "4.2ms", "1.25s".
std::string format_duration(std::chrono::nanoseconds duration)
{
  auto const us = std::chrono::duration<double, std::micro>(duration).count();
  if (us < 1e3) { return std::format("{:.0f}us", us); }
  if (us < 1e6) { return std::format("{:.1f}ms", us / 1e3); }
  return std::format("{:.2f}s", us / 1e6);
}

/// "512B", "3.0KiB", "1.2GiB".
std::string format_bytes(std::uint64_t bytes)
{
  constexpr std::array<std::string_view, 5> units{"B", "KiB", "MiB", "GiB", "TiB"};
  auto value       = static_cast<double>(bytes);
  std::size_t unit = 0;
  while (value >= 1024.0 && unit + 1 < units.size()) {
    value /= 1024.0;
    ++unit;
  }
  return unit == 0 ? std::format("{}B", bytes) : std::format("{:.1f}{}", value, units[unit]);
}

struct named_class {
  cucascade::io::request_class cls;
  std::string_view name;
};

/// Log order of the request classes.
constexpr std::array<named_class, cucascade::io::request_class_count> logged_classes{{
  {cucascade::io::request_class::latency, "latency"},
  {cucascade::io::request_class::read, "read"},
  {cucascade::io::request_class::write, "write"},
  {cucascade::io::request_class::background, "background"},
}};

}  // namespace

char const* to_string(cucascade::io::io_context_type type) noexcept
{
  switch (type) {
    case cucascade::io::io_context_type::uring: return "uring";
    case cucascade::io::io_context_type::restful: return "restful";
    case cucascade::io::io_context_type::kvikio: return "kvikio";
    case cucascade::io::io_context_type::s3rdma: return "s3rdma";
  }
  return "unknown";
}

std::chrono::nanoseconds first_io_percentile(
  std::array<std::uint64_t, cucascade::io::first_io_delay_buckets> const& histogram,
  unsigned percent) noexcept
{
  auto const total = std::accumulate(histogram.begin(), histogram.end(), std::uint64_t{0});
  if (total == 0) { return {}; }
  percent = std::clamp(percent, 1U, 100U);
  // Nearest rank, ceil(total * percent / 100), split so the product cannot overflow.
  auto const rank    = total / 100 * percent + ((total % 100) * percent + 99) / 100;
  std::uint64_t seen = 0;
  for (std::size_t bucket = 0; bucket < histogram.size(); ++bucket) {
    seen += histogram[bucket];
    if (seen < rank) { continue; }
    // Bucket b ends at 2^b us; the open-ended last bucket reports where it starts.
    auto const exponent = bucket + 1 < histogram.size() ? bucket : bucket - 1;
    return std::chrono::microseconds{std::int64_t{1} << exponent};
  }
  return {};
}

std::string format_io_stats_delta(cucascade::io::io_context_type type,
                                  io_stats_snapshot const& before,
                                  io_stats_snapshot const& after)
{
  std::uint64_t bytes = 0;
  for (auto const& runner : after.stats.runners) {
    auto const& prior = before.stats.runners;
    auto const it =
      std::find_if(prior.begin(), prior.end(), [&](auto const& r) { return r.id == runner.id; });
    bytes += counter_delta(runner.bytes_submitted, it == prior.end() ? 0 : it->bytes_submitted);
  }
  auto line = std::format("[io_stats] {} runners={} bytes_submitted={}",
                          to_string(type),
                          after.stats.active_runners,
                          format_bytes(bytes));

  for (auto const& [cls, name] : logged_classes) {
    auto const index = cucascade::io::request_class_index(cls);
    auto const& from = before.stats.per_class[index];
    auto const& to   = after.stats.per_class[index];
    auto const count = counter_delta(to.first_io_count, from.first_io_count);
    if (count == 0) { continue; }
    std::array<std::uint64_t, cucascade::io::first_io_delay_buckets> histogram{};
    for (std::size_t b = 0; b < histogram.size(); ++b) {
      histogram[b] = counter_delta(to.first_io_histogram[b], from.first_io_histogram[b]);
    }
    auto const total = counter_delta(to.first_io_total, from.first_io_total);
    auto const mean  = std::chrono::nanoseconds{total.count() / static_cast<std::int64_t>(count)};
    // A bucket edge can overshoot the largest delay actually seen by up to 2x.
    auto p99 = first_io_percentile(histogram, 99);
    if (to.first_io_max > std::chrono::nanoseconds::zero()) {
      p99 = std::min(p99, to.first_io_max);
    }
    std::format_to(std::back_inserter(line),
                   " {}{{n={} mean={} p99={} max={}}}",
                   name,
                   count,
                   format_duration(mean),
                   format_duration(p99),
                   format_duration(to.first_io_max));
  }
  return line;
}

}  // namespace sirius::scan_manager
