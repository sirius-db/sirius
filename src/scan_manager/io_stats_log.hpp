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

#pragma once

#include <cucascade/io/io_context.hpp>
#include <cucascade/io/types.hpp>

#include <array>
#include <chrono>
#include <cstdint>
#include <string>

namespace sirius::scan_manager {

/// The backend's name, for log lines and exception text.
[[nodiscard]] char const* to_string(cucascade::io::io_context_type type) noexcept;

/// One sample of an ioctx's queue / runner statistics.  The scan manager keeps
/// one per started ioctx and diffs consecutive samples at each query boundary.
struct io_stats_snapshot {
  cucascade::io::queue_stats stats;

  /// Sample @p io_ctx; @c ioctx::stats() is @c noexcept.
  [[nodiscard]] static io_stats_snapshot take(cucascade::io::ioctx const& io_ctx)
  {
    return {.stats = io_ctx.stats()};
  }
};

/**
 * @brief Upper edge of the first-I/O histogram bucket holding the @p percent -th
 * percentile (nearest rank) of @p histogram.
 *
 * Buckets are log2 µs (@ref cucascade::io::first_io_delay_bucket): bucket 0 is
 * below 1 µs, bucket @c b holds [2^(b-1), 2^b) µs.  The last bucket is
 * open-ended, so its lower edge is returned instead: a floor, not a bound.
 *
 * @param histogram First-I/O delay counts per bucket (typically a difference of
 *        two samples).
 * @param percent Percentile, clamped to [1, 100].
 * @return The edge, or zero when @p histogram is empty.
 */
[[nodiscard]] std::chrono::nanoseconds first_io_percentile(
  std::array<std::uint64_t, cucascade::io::first_io_delay_buckets> const& histogram,
  unsigned percent) noexcept;

/**
 * @brief One `[io_stats]` log line describing what an ioctx did between two samples.
 *
 * Per request class with new first-I/O records (classes without any are
 * omitted): count, mean first-I/O delay (Δtotal / Δcount), p99 from the Δ
 * histogram (@ref first_io_percentile, capped at max), and max (@c first_io_max:
 * the peak since the last @c ioctx::reset_stats_peaks, so the caller resets
 * peaks whenever it takes @p before).  Also the active runners of @p after and the
 * bytes its runners submitted since @p before, matched by runner id; a runner
 * absent from @p before counts from zero, and one gone by @p after takes its
 * bytes with it.
 */
[[nodiscard]] std::string format_io_stats_delta(cucascade::io::io_context_type type,
                                                io_stats_snapshot const& before,
                                                io_stats_snapshot const& after);

}  // namespace sirius::scan_manager
