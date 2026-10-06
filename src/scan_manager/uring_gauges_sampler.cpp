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

#include "scan_manager/uring_gauges_sampler.hpp"

#include "exec/thread_util.hpp"
#include "log/logging.hpp"

#include <cstdint>
#include <format>
#include <tuple>
#include <utility>
#include <vector>

namespace sirius::scan_manager {

namespace {

using gauges            = cucascade::io::uring::uring_reactor::gauges;
using queue_delay_stats = cucascade::io::uring::uring_reactor::queue_delay_stats;

[[nodiscard]] bool gauges_enabled() noexcept
{
  try {
    return sirius::log::get_sink()->should_log(sirius::log::level::debug);
  } catch (...) {
    return false;
  }
}

// Queue delay of one io_class (d = demand, p = prefetch): count, sum/max in ms and the
// non-empty log2-us histogram buckets as bucket:count (see queue_delay_stats).
[[nodiscard]] std::string format_queue_delay(queue_delay_stats const& s, char const* tag)
{
  std::string hist;
  for (std::size_t b = 0; b < s.histogram.size(); ++b) {
    if (s.histogram[b] == 0) continue;
    if (!hist.empty()) hist += ',';
    hist += std::format("{}:{}", b, s.histogram[b]);
  }
  return std::format(" {0}q_n={1} {0}q_sum_ms={2:.1f} {0}q_max_ms={3:.1f} {0}q_hist={4}",
                     tag,
                     s.count,
                     static_cast<double>(s.sum_ns) / 1e6,
                     static_cast<double>(s.max_ns) / 1e6,
                     hist.empty() ? "-" : hist);
}

}  // namespace

std::optional<std::string> format_gauges_line(std::size_t index,
                                              gauges const& current,
                                              gauges const* previous,
                                              double seconds)
{
  auto const& g = current;
  auto const started =
    previous != nullptr ? g.requests_started - previous->requests_started : std::uint64_t{0};
  auto const bytes =
    previous != nullptr ? g.bytes_submitted - previous->bytes_submitted : std::uint64_t{0};
  bool const idle = g.max_inflight_ops == 0 && g.pending_ops == 0 &&
                    g.active_remaining_slices == 0 && g.queued_requests == 0 && started == 0 &&
                    bytes == 0 && g.queue_delay[0].count == 0 && g.queue_delay[1].count == 0;
  if (idle) { return std::nullopt; }
  return std::format(
    "[uring_gauges] reactor={} inflight={} max_inflight={} pending_ops={} "
    "active_slices={} queued_requests={} queued_MiB={} started={} MiB_s={:.0f}{}{}",
    index,
    g.inflight_ops,
    g.max_inflight_ops,
    g.pending_ops,
    g.active_remaining_slices,
    g.queued_requests,
    g.queued_bytes >> 20,
    started,
    seconds > 0 ? static_cast<double>(bytes) / static_cast<double>(1 << 20) / seconds : 0.0,
    format_queue_delay(g.queue_delay[0], "d"),
    format_queue_delay(g.queue_delay[1], "p"));
}

uring_gauges_sampler::uring_gauges_sampler(
  std::shared_ptr<cucascade::io::uring::uring_ioctx> io_ctx)
  : _io_ctx(std::move(io_ctx))
{
}

uring_gauges_sampler::~uring_gauges_sampler() { stop(); }

void uring_gauges_sampler::start()
{
  if (_thread.joinable() || !_io_ctx) { return; }
  _thread     = std::jthread([this](std::stop_token const& st) { run(st); });
  std::ignore = sirius::exec::thread_util::set_thread_name(_thread, "uring_gauges");
}

void uring_gauges_sampler::stop() noexcept
{
  if (!_thread.joinable()) { return; }
  _thread.request_stop();
  try {
    _thread.join();
  } catch (...) {  // NOLINT(bugprone-empty-catch)
    // Joining a live sampler cannot fail in practice; never let teardown throw.
  }
}

void uring_gauges_sampler::run(std::stop_token const& stop_token)
{
  using clock = std::chrono::steady_clock;
  std::vector<gauges> previous;
  auto previous_at = clock::now();

  while (!stop_token.stop_requested()) {
    {
      std::unique_lock lock(_mutex);
      // Returns early on stop; the predicate never fires otherwise.
      std::ignore = _cv.wait_for(lock, stop_token, sample_period, [] { return false; });
    }
    if (stop_token.stop_requested()) { break; }
    if (!gauges_enabled()) {
      previous.clear();
      continue;
    }

    auto const now     = clock::now();
    auto const current = _io_ctx->reactor_gauges();
    auto const seconds = std::chrono::duration<double>(now - previous_at).count();
    try {
      for (std::size_t i = 0; i < current.size(); ++i) {
        auto const line =
          format_gauges_line(i, current[i], i < previous.size() ? &previous[i] : nullptr, seconds);
        if (line) { SIRIUS_LOG_DEBUG("{}", *line); }
      }
      previous    = current;
      previous_at = now;
    } catch (...) {  // NOLINT(bugprone-empty-catch)
      // Diagnostics only: a failed format or copy must not take the sampler down.
    }
  }
}

}  // namespace sirius::scan_manager
