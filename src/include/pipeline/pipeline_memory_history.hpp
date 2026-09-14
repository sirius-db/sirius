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

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <mutex>
#include <optional>
#include <vector>

namespace sirius::pipeline {

/**
 * @brief A single observation from one completed task.
 *
 * For GPU pipeline tasks, estimated_bytes is the input data size.
 * For scan tasks, estimated_bytes is the decoded projected-column estimate
 * before row filtering.
 */
struct task_memory_record {
  std::size_t estimated_bytes;              ///< Pre-history estimation basis
  std::size_t peak_memory_bytes;            ///< Actual peak allocated bytes during execution
  std::optional<std::size_t> output_bytes;  ///< Total output data size (nullopt if task OOM'd)
  /// Whether @ref estimated_bytes is in pipeline-input units. False for mid-pipeline resumes;
  /// they remain valid for per-task @ref estimate_peak_memory.
  bool ratio_eligible = true;
};

/**
 * @brief Non-evicting totals over successful pipeline tasks.
 *
 * Ratio terms require matching usable input and output; output terms include every successful
 * task. Keeping them separate prevents unsized inputs from reducing the emitted total.
 */
struct history_totals {
  /// Tasks with nonzero, ratio-eligible input and recorded output.
  std::size_t ratio_input_bytes  = 0;
  std::size_t ratio_output_bytes = 0;
  std::size_t ratio_records      = 0;
  /// Every task with recorded output, including zero-basis and resumed tasks.
  std::size_t output_bytes   = 0;
  std::size_t output_records = 0;

  /// Aggregate ratio over eligible tasks, or nullopt without usable input.
  [[nodiscard]] std::optional<double> ratio() const
  {
    if (ratio_records == 0 || ratio_input_bytes == 0) { return std::nullopt; }
    return static_cast<double>(ratio_output_bytes) / static_cast<double>(ratio_input_bytes);
  }
};

/**
 * @brief Thread-safe collection of memory records for a single pipeline.
 *
 * Stores a bounded ring buffer of recent task_memory_records and provides
 * an estimation function that uses historical ratios to predict peak memory
 * consumption for a new task given its estimation basis.
 * Also maintains @ref history_totals beyond ring-buffer eviction.
 */
class pipeline_memory_history {
 public:
  pipeline_memory_history() = default;

  /**
   * @brief Record a completed task's memory metrics.
   *
   * Thread-safe. May be called concurrently from multiple executor threads.
   *
   * @param rec The memory record to store.
   */
  void record(task_memory_record rec)
  {
    std::lock_guard<std::mutex> lock(_mutex);
    // OOM'd tasks have no output and contribute to neither total.
    if (rec.output_bytes.has_value()) {
      // Output totals include every successful task.
      _totals.output_bytes += *rec.output_bytes;
      _totals.output_records++;
      // Ratio totals additionally require a usable pipeline-input basis.
      if (rec.estimated_bytes > 0 && rec.ratio_eligible) {
        _totals.ratio_input_bytes += rec.estimated_bytes;
        _totals.ratio_output_bytes += *rec.output_bytes;
        _totals.ratio_records++;
      }
    }
    // estimate_peak_memory divides by the basis, so zero-basis records skip the ring.
    if (rec.estimated_bytes == 0) { return; }
    if (_records.size() < kMaxRecords) {
      _records.push_back(rec);
    } else {
      _records[_write_pos] = rec;
      _write_pos           = (_write_pos + 1) % kMaxRecords;
    }
  }

  /**
   * @brief Record a failed task's memory metrics.
   *
   * This checks if a previous failed task was recorded with the same input size, if so, we want to
   * keep the one with the largest peak consumption, so that the next attempt has a higher memory
   * estimation.
   *
   * Thread-safe. May be called concurrently from multiple executor threads.
   *
   * @param estimated_bytes The estimation basis for the failed task.
   * @param peak_memory_bytes The actual peak allocated bytes during the failed execution.
   */
  void record_on_failure(std::size_t estimated_bytes, std::size_t peak_memory_bytes)
  {
    if (estimated_bytes == 0) { return; }

    std::lock_guard<std::mutex> lock(_mutex);

    // Look for existing records with the same estimated_bytes
    bool found = false;
    for (auto& rec : _records) {
      if (rec.estimated_bytes == estimated_bytes) {
        rec.peak_memory_bytes = std::max(rec.peak_memory_bytes, peak_memory_bytes);
        found                 = true;
      }
    }
    if (found) { return; }

    // No existing failure record found; insert a new one into the ring buffer
    task_memory_record rec{estimated_bytes, peak_memory_bytes, std::nullopt};
    if (_records.size() < kMaxRecords) {
      _records.push_back(rec);
    } else {
      _records[_write_pos] = rec;
      _write_pos           = (_write_pos + 1) % kMaxRecords;
    }
  }

  /**
   * @brief Estimate peak memory for a new task given its estimation basis.
   *
   * Uses a weighted average of historical peak/estimated ratios, where records
   * with similar estimation bases receive higher weight.
   *
   * @param estimated_bytes The estimation basis for the new task.
   * @return The estimated peak memory in bytes, or nullopt if no history exists.
   */
  [[nodiscard]] std::optional<std::size_t> estimate_peak_memory(std::size_t estimated_bytes) const
  {
    std::lock_guard<std::mutex> lock(_mutex);

    if (_records.empty() || estimated_bytes == 0) { return std::nullopt; }

    double weighted_ratio_sum = 0.0;
    double weight_sum         = 0.0;

    for (const auto& rec : _records) {
      double ratio =
        static_cast<double>(rec.peak_memory_bytes) / static_cast<double>(rec.estimated_bytes);

      // Weight by proximity of estimation basis: records with similar sizes
      // are more predictive. Uses log-ratio distance so that 50MB vs 100MB
      // has the same weight decay as 500MB vs 1000MB.
      double log_ratio = std::abs(
        std::log(static_cast<double>(rec.estimated_bytes) / static_cast<double>(estimated_bytes)));
      double weight = 1.0 / (1.0 + log_ratio);

      weighted_ratio_sum += ratio * weight;
      weight_sum += weight;
    }

    double avg_ratio = weighted_ratio_sum / weight_sum;

    return static_cast<std::size_t>(static_cast<double>(estimated_bytes) * avg_ratio);
  }

  /**
   * @brief Get the number of records currently stored.
   */
  [[nodiscard]] std::size_t size() const
  {
    std::lock_guard<std::mutex> lock(_mutex);
    return _records.size();
  }

  /// Non-evicting totals over successful tasks.
  [[nodiscard]] history_totals totals() const
  {
    std::lock_guard<std::mutex> lock(_mutex);
    return _totals;
  }

 private:
  static constexpr std::size_t kMaxRecords = 64;

  mutable std::mutex _mutex;
  std::vector<task_memory_record> _records;
  std::size_t _write_pos = 0;
  history_totals _totals;
};

}  // namespace sirius::pipeline
