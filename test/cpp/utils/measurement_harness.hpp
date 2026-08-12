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

/**
 * @file measurement_harness.hpp
 * @brief Feature-off/feature-on wall-clock comparison that survives a shared host
 *
 * This header owns the *schedule* and the *statistics* of an A/B wall-clock comparison. It knows
 * nothing about queries, settings, or counters: the caller supplies a callable that performs one
 * execution of one arm and reports both the interval it chose to time and whether that execution
 * may be kept. That split keeps the arming and correctness evidence -- which is specific to the
 * feature under test -- with the test, and keeps the sampling discipline here where every
 * measurement can share it.
 *
 * Two properties of the schedule are not options, because measurements taken without them have
 * already been shown to be wrong on this codebase:
 *
 * - **Arms interleave, they never run in blocks.** Running every baseline sample and then every
 *   treatment sample turns any drift in host or device state -- clock ramping, page cache, pool
 *   growth, an arriving co-tenant -- into an apparent effect of the treatment. A blocked run of
 *   the Top-N dynamic filter reported a 24% regression that an interleaved rerun of the same
 *   queries reduced to 1.9%, inside the noise spread.
 * - **Which arm leads alternates between pairs.** Inside one pair the second execution runs
 *   against a warmer machine than the first, so a fixed order would credit that warmth to
 *   whichever arm always ran second.
 *
 * A pair is the unit of acceptance: if either of its executions reports itself invalid, both
 * samples are dropped, so the surviving samples always pair up and the paired-ratio statistic
 * stays meaningful.
 *
 * `run_interleaved_ab` reports several statistics over the same samples, because on a shared host
 * their disagreement is itself information and collapsing them to one number would hide it. The
 * ratio of medians answers "how much slower was the treatment overall"; the median of per-pair
 * ratios answers the same question with slow drift shared by both arms divided out; the ratio of
 * minima answers it using each arm's least-disturbed sample, interference being one-sided.
 *
 * The statistic that actually decides a threshold, though, is the confidence interval on the paired
 * geometric mean. A point estimate cannot distinguish a small effect from the sampling noise of the
 * pairs that produced it, and this feature's criteria are stated in single-digit percentages -- so
 * a cell whose interval straddles its threshold has not measured the thing, however tidy its point
 * estimate looks, and needs more pairs or a query long enough to be worth timing.
 *
 * The GPU occupancy bracket names the condition under which a number was taken. A wall-clock
 * result gathered while another tenant held device memory is not evidence, so
 * `observe_gpu_occupancy` is meant to be called immediately before and after a measurement and
 * both observations reported alongside the numbers.
 */

#include <unistd.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <utility>
#include <vector>

namespace sirius::test {

/// Wall-clock samples for one arm, in the order they were taken.
class sample_series {
 public:
  void add(std::chrono::nanoseconds elapsed)
  {
    samples_ms_.push_back(std::chrono::duration<double, std::milli>(elapsed).count());
  }

  [[nodiscard]] std::size_t size() const noexcept { return samples_ms_.size(); }
  [[nodiscard]] bool empty() const noexcept { return samples_ms_.empty(); }
  [[nodiscard]] const std::vector<double>& samples_ms() const noexcept { return samples_ms_; }

  [[nodiscard]] double median_ms() const { return median_of(samples_ms_); }

  [[nodiscard]] double min_ms() const
  {
    return samples_ms_.empty() ? 0.0 : *std::min_element(samples_ms_.begin(), samples_ms_.end());
  }

  [[nodiscard]] double max_ms() const
  {
    return samples_ms_.empty() ? 0.0 : *std::max_element(samples_ms_.begin(), samples_ms_.end());
  }

  /// Peak-to-peak range as a fraction of the median -- the width of the noise band the arm's own
  /// samples occupy, which any claimed effect has to clear to mean anything.
  [[nodiscard]] double spread_fraction() const
  {
    auto const mid = median_ms();
    return mid > 0.0 ? (max_ms() - min_ms()) / mid : 0.0;
  }

  /// Median of an arbitrary sample set, averaging the middle pair for an even count.
  [[nodiscard]] static double median_of(std::vector<double> values)
  {
    if (values.empty()) { return 0.0; }
    auto const middle = values.size() / 2;
    std::nth_element(
      values.begin(), values.begin() + static_cast<std::ptrdiff_t>(middle), values.end());
    auto const upper = values[middle];
    if (values.size() % 2 == 1) { return upper; }
    auto const lower =
      *std::max_element(values.begin(), values.begin() + static_cast<std::ptrdiff_t>(middle));
    return 0.5 * (lower + upper);
  }

 private:
  std::vector<double> samples_ms_;
};

/// Whether one execution may be kept, and why not when it may not.
struct run_validity {
  bool valid = true;
  std::string reason;  ///< Empty when valid; shown in the report when not
};

/// One execution's outcome: the interval the caller chose to time, and its validity.
struct timed_run {
  std::chrono::nanoseconds elapsed{};
  run_validity validity;
};

/**
 * @brief Device memory attributed to this process and to any other, as `nvidia-smi` reports it
 *
 * `self_attributed` is the honesty flag. When this process is known to hold device memory and its
 * PID is still absent from the compute-process list -- a PID namespace, or a driver that does not
 * expose process accounting -- foreign memory cannot be told apart from our own, and a report that
 * quoted `foreign_bytes` would be claiming a quiet host it never observed.
 */
struct gpu_occupancy {
  bool available              = false;  ///< `nvidia-smi` ran and its output parsed
  bool self_attributed        = false;  ///< This process's PID appeared among the compute processes
  std::uint64_t self_bytes    = 0;
  std::uint64_t foreign_bytes = 0;
  std::size_t foreign_process_count = 0;

  [[nodiscard]] std::string describe() const
  {
    if (!available) { return "unavailable (nvidia-smi did not run)"; }
    if (foreign_process_count == 0) {
      return self_attributed ? "quiet (self " + std::to_string(self_bytes >> 20U) + " MiB)"
                             : "quiet, self not attributed";
    }
    return std::to_string(foreign_process_count) + " foreign process(es) holding " +
           std::to_string(foreign_bytes >> 20U) + " MiB" +
           (self_attributed ? "" : " (self not attributed; attribution unreliable)");
  }
};

/// Reads the compute processes on the default device and splits their memory into this process's
/// and everyone else's. Returns an unavailable observation rather than failing when `nvidia-smi`
/// cannot be reached, so a host without it degrades to "condition unknown" and never to a silent
/// claim of exclusivity.
[[nodiscard]] inline gpu_occupancy observe_gpu_occupancy()
{
  gpu_occupancy occupancy;
  auto* pipe = ::popen(
    "nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2>/dev/null",
    "r");
  if (pipe == nullptr) { return occupancy; }

  auto const self_pid = static_cast<long>(::getpid());
  char line[256];
  while (std::fgets(line, sizeof(line), pipe) != nullptr) {
    long pid      = 0;
    long long mib = 0;
    if (std::sscanf(line, "%ld , %lld", &pid, &mib) != 2) { continue; }
    auto const bytes = static_cast<std::uint64_t>(mib) << 20U;
    if (pid == self_pid) {
      occupancy.self_attributed = true;
      occupancy.self_bytes += bytes;
    } else {
      occupancy.foreign_bytes += bytes;
      ++occupancy.foreign_process_count;
    }
  }
  occupancy.available = ::pclose(pipe) == 0;
  return occupancy;
}

/**
 * @brief Sampling schedule for one A/B cell
 *
 * The pair count is an outcome, not a setting. How many pairs it takes to resolve an effect
 * depends on how noisy the query is, which is not known before running it -- so the caller states
 * the effect size the cell has to be able to see and the harness samples until it can see it, or
 * until @ref max_pairs says the query is too noisy to answer the question at any affordable count.
 * A fixed count instead produces the failure this replaced: a confidence interval wide enough to
 * contain both "no effect" and the threshold being tested, quoted as though it were a measurement.
 */
struct interleaved_ab_plan {
  int warmup_pairs = 3;  ///< Discarded pairs run before sampling, to absorb JIT, pool growth,
                         ///< and page-cache effects
  //! Smallest effect the cell must be able to distinguish from zero, as a fraction (0.02 = 2%).
  //! Sampling continues until the 95% interval on the paired geometric mean is at least this
  //! tight. Set it from the criterion the cell has to decide, not from taste.
  double target_resolution = 0.02;
  int min_pairs            = 21;   ///< Sampled before the interval is first consulted
  int max_pairs            = 401;  ///< Cap; reaching it means the cell did not resolve
};

/// Everything one A/B cell produced, including the conditions it was produced under.
struct interleaved_ab_result {
  std::array<std::string, 2> arm_names{"baseline", "treatment"};
  std::array<sample_series, 2> arms;
  std::vector<double> paired_ratios;       ///< Treatment / baseline, one per surviving pair
  std::vector<std::string> discard_notes;  ///< One entry per dropped pair, with its reason
  int attempted_pairs = 0;
  gpu_occupancy before;
  gpu_occupancy after;
  /// The heaviest foreign occupancy seen anywhere in the cell, sampled once per pair rather than
  /// only at the cell's edges. A tenant that arrives and leaves between two pairs is invisible to
  /// an edge-only bracket, and its whole residency lands inside the samples it corrupted.
  gpu_occupancy worst;
  int pairs_sampled_with_foreign = 0;
  double target_resolution       = 0.0;  ///< What the cell was asked to resolve

  /// Half-width of the 95% interval on the paired geometric mean, as a fraction. This is a
  /// property of *this cell's* samples -- its query's noise and its pair count -- and never a
  /// constant of the harness: the same schedule on a noisier query resolves less.
  [[nodiscard]] double achieved_resolution() const
  {
    auto const interval = paired_ratio_interval();
    return 0.5 * (interval[2] - interval[0]);
  }

  /// Whether the samples can distinguish an effect of @ref target_resolution from zero. A cell
  /// that answers false has measured its query's noise, not its treatment.
  [[nodiscard]] bool resolved() const
  {
    return paired_ratios.size() >= 2 && target_resolution > 0.0 &&
           achieved_resolution() <= target_resolution;
  }

  /// Pairs the observed spread implies are needed to reach @ref target_resolution, for a caller
  /// deciding whether a cell is worth re-running longer or is simply too noisy to answer.
  [[nodiscard]] std::size_t pairs_needed_for_target() const
  {
    auto const count = paired_ratios.size();
    if (count < 2 || target_resolution <= 0.0) { return 0; }
    auto const half_width = 1.96 * log_ratio_stddev() / std::sqrt(static_cast<double>(count));
    if (half_width <= 0.0) { return count; }
    auto const scale = half_width / std::log1p(target_resolution);
    return static_cast<std::size_t>(std::ceil(static_cast<double>(count) * scale * scale));
  }

  /// True when every pair survived and no sample saw another tenant. A false result is not
  /// necessarily wrong, but it must not be quoted without the reason alongside it.
  [[nodiscard]] bool conditions_clean() const
  {
    return discard_notes.empty() && !arms[0].empty() &&
           arms[0].size() == static_cast<std::size_t>(attempted_pairs) && before.available &&
           after.available && worst.foreign_process_count == 0;
  }

  /// Treatment median over baseline median: the overall effect, drift included.
  [[nodiscard]] double median_ratio() const
  {
    auto const baseline = arms[0].median_ms();
    return baseline > 0.0 ? arms[1].median_ms() / baseline : 0.0;
  }

  /// Median of the per-pair ratios: the same effect with drift shared by both arms divided out.
  [[nodiscard]] double paired_median_ratio() const
  {
    return sample_series::median_of(paired_ratios);
  }

  /// Treatment best over baseline best. Interference on a shared host only ever adds time, so the
  /// fastest sample of each arm is the one least contaminated by it; when the medians and the
  /// minima disagree, the difference between them is interference rather than treatment.
  [[nodiscard]] double min_ratio() const
  {
    auto const baseline = arms[0].min_ms();
    return baseline > 0.0 ? arms[1].min_ms() / baseline : 0.0;
  }

  /**
   * @brief Geometric mean of the per-pair ratios with its 95% confidence interval
   *
   * This is the statistic a threshold can actually be tested against. A point estimate cannot say
   * whether a 2% difference is an effect or the sampling noise of this many pairs, so the interval
   * is what decides: a criterion of the form "within +2%" is met when the whole interval sits
   * below 1.02, is violated when the whole interval sits above it, and is simply unresolved --
   * needing more pairs or a longer query -- when the interval straddles it. Reporting the point
   * estimate alone is how an unresolved measurement gets quoted later as a result.
   *
   * Ratios combine multiplicatively, so the mean and interval are computed over log ratios and
   * exponentiated back; the arithmetic mean of ratios is biased upward and would flatter the
   * baseline arm.
   *
   * @return `{low, point, high}`, all 1.0 when fewer than two pairs survived
   */
  [[nodiscard]] std::array<double, 3> paired_ratio_interval() const
  {
    auto const count = paired_ratios.size();
    if (count < 2) { return {1.0, 1.0, 1.0}; }
    double sum = 0.0;
    for (auto const ratio : paired_ratios) {
      sum += std::log(ratio);
    }
    auto const mean       = sum / static_cast<double>(count);
    auto const half_width = 1.96 * log_ratio_stddev() / std::sqrt(static_cast<double>(count));
    return {std::exp(mean - half_width), std::exp(mean), std::exp(mean + half_width)};
  }

  /// Sample standard deviation of the per-pair log ratios -- the query's own noise, independent of
  /// how many pairs have been taken.
  [[nodiscard]] double log_ratio_stddev() const
  {
    auto const count = paired_ratios.size();
    if (count < 2) { return 0.0; }
    double sum = 0.0;
    for (auto const ratio : paired_ratios) {
      sum += std::log(ratio);
    }
    auto const mean = sum / static_cast<double>(count);
    double variance = 0.0;
    for (auto const ratio : paired_ratios) {
      auto const deviation = std::log(ratio) - mean;
      variance += deviation * deviation;
    }
    return std::sqrt(variance / static_cast<double>(count - 1));
  }

  [[nodiscard]] std::string report() const;
};

/**
 * @brief Run one A/B cell: warm up, then sample @p plan.pairs interleaved pairs
 *
 * @param arm_names Names for arm 0 (baseline) and arm 1 (treatment), used only in the report
 * @param arm Callable invoked as `arm(index)` with index 0 or 1; performs one execution of that
 *            arm and returns the interval it timed together with that execution's validity. Any
 *            work that must not be timed -- changing a setting, snapshotting counters -- belongs
 *            inside the callable but outside the interval it reports.
 * @param plan Warm-up and sample counts
 * @return The samples, the paired ratios, and the GPU occupancy bracket
 */
template <typename ArmFn>
[[nodiscard]] interleaved_ab_result run_interleaved_ab(const std::array<std::string, 2>& arm_names,
                                                       ArmFn&& arm,
                                                       const interleaved_ab_plan& plan = {})
{
  interleaved_ab_result result;
  result.arm_names         = arm_names;
  result.target_resolution = plan.target_resolution;

  for (int i = 0; i < plan.warmup_pairs; ++i) {
    (void)arm(0);
    (void)arm(1);
  }

  auto note_occupancy = [&result](const gpu_occupancy& observed) {
    if (observed.available && observed.foreign_bytes >= result.worst.foreign_bytes) {
      result.worst = observed;
    }
  };

  result.before = observe_gpu_occupancy();
  result.worst  = result.before;
  for (int pair = 0; pair < plan.max_pairs; ++pair) {
    // Stop as soon as the interval is tight enough to decide the effect the cell was given, but
    // never before `min_pairs`: an early interval estimated from a handful of pairs is itself too
    // noisy to be trusted as a stopping signal.
    if (pair >= plan.min_pairs && result.resolved()) { break; }
    ++result.attempted_pairs;
    // The arm that runs first alternates, so the warmer second slot is shared evenly.
    int const lead        = pair % 2;
    auto const first      = arm(lead);
    auto const second     = arm(1 - lead);
    auto const& baseline  = lead == 0 ? first : second;
    auto const& treatment = lead == 0 ? second : first;

    // Sampled between pairs, never inside a timed interval, so watching for a tenant cannot
    // lengthen the thing being timed.
    auto const observed = observe_gpu_occupancy();
    note_occupancy(observed);
    if (observed.available && observed.foreign_process_count > 0) {
      ++result.pairs_sampled_with_foreign;
    }

    if (!baseline.validity.valid || !treatment.validity.valid) {
      auto const& reason =
        baseline.validity.valid ? treatment.validity.reason : baseline.validity.reason;
      result.discard_notes.push_back("pair " + std::to_string(pair) + ": " + reason);
      continue;
    }
    result.arms[0].add(baseline.elapsed);
    result.arms[1].add(treatment.elapsed);
    auto const baseline_ms = std::chrono::duration<double, std::milli>(baseline.elapsed).count();
    if (baseline_ms > 0.0) {
      result.paired_ratios.push_back(
        std::chrono::duration<double, std::milli>(treatment.elapsed).count() / baseline_ms);
    }
  }
  result.after = observe_gpu_occupancy();
  note_occupancy(result.after);
  return result;
}

namespace detail {

inline std::string fixed(double value, int decimals)
{
  char buffer[64];
  std::snprintf(buffer, sizeof(buffer), "%.*f", decimals, value);
  return buffer;
}

inline std::string percent(double ratio) { return fixed((ratio - 1.0) * 100.0, 1) + "%"; }

}  // namespace detail

inline std::string interleaved_ab_result::report() const
{
  auto const arm_line = [](const std::string& name, const sample_series& series) {
    return "  " + name + ": median " + detail::fixed(series.median_ms(), 2) + " ms, min " +
           detail::fixed(series.min_ms(), 2) + ", max " + detail::fixed(series.max_ms(), 2) +
           ", spread " + detail::fixed(series.spread_fraction() * 100.0, 1) +
           "% of median, n=" + std::to_string(series.size()) + "\n";
  };

  std::string text;
  text += arm_line(arm_names[0], arms[0]);
  text += arm_line(arm_names[1], arms[1]);
  text += "  ratio of medians: " + detail::fixed(median_ratio(), 4) + " (" +
          detail::percent(median_ratio()) + ")\n";
  text += "  median of paired ratios: " + detail::fixed(paired_median_ratio(), 4) + " (" +
          detail::percent(paired_median_ratio()) + ")";
  if (!paired_ratios.empty()) {
    auto const bounds = std::minmax_element(paired_ratios.begin(), paired_ratios.end());
    text += ", per-pair range " + detail::fixed(*bounds.first, 4) + " to " +
            detail::fixed(*bounds.second, 4);
  }
  text += "\n";
  text += "  ratio of minima: " + detail::fixed(min_ratio(), 4) + " (" +
          detail::percent(min_ratio()) + ")\n";
  auto const interval = paired_ratio_interval();
  text += "  paired geometric mean: " + detail::fixed(interval[1], 4) + " (" +
          detail::percent(interval[1]) + "), 95% CI " + detail::fixed(interval[0], 4) + " to " +
          detail::fixed(interval[2], 4) + " (" + detail::percent(interval[0]) + " to " +
          detail::percent(interval[2]) + ")\n";
  text += "  resolution: achieved +/-" + detail::fixed(achieved_resolution() * 100.0, 2) +
          "% against a target of +/-" + detail::fixed(target_resolution * 100.0, 2) + "% -- " +
          (resolved() ? "RESOLVED" : "NOT RESOLVED") + " after " +
          std::to_string(paired_ratios.size()) + " pairs";
  if (!resolved()) {
    auto const needed = pairs_needed_for_target();
    text += "; this query's spread implies about " + std::to_string(needed) +
            " pairs would be required, so any effect narrower than the interval above is "
            "unmeasured, not absent";
  }
  text += "\n";
  text += "  gpu before: " + before.describe() + "\n";
  text += "  gpu after:  " + after.describe() + "\n";
  text += "  gpu worst:  " + worst.describe() + " (" + std::to_string(pairs_sampled_with_foreign) +
          " of " + std::to_string(attempted_pairs) + " pairs sampled with a tenant present)\n";
  if (discard_notes.empty()) {
    text += "  pairs kept: " + std::to_string(arms[0].size()) + "/" +
            std::to_string(attempted_pairs) + "\n";
  } else {
    text += "  pairs kept: " + std::to_string(arms[0].size()) + "/" +
            std::to_string(attempted_pairs) + " -- DISCARDS:\n";
    for (auto const& note : discard_notes) {
      text += "    " + note + "\n";
    }
  }
  return text;
}

}  // namespace sirius::test
