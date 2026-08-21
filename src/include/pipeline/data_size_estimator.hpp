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

#include <cstddef>
#include <optional>
#include <string_view>

namespace sirius {
namespace op {
class sirius_physical_operator;
}  // namespace op

namespace pipeline {

class sirius_pipeline;

/**
 * @brief Projected whole-query byte total, not bytes seen so far.
 * @see docs/super-sirius/data-size-estimation.md
 */
struct data_size_estimate {
  std::size_t bytes = 0;
  /// True when anchored on an exact total with no learned ratio applied.
  bool exact = false;
  /// Pipelines traversed; ratio error compounds per hop.
  std::size_t hops = 0;
  /// Completed tasks behind the weakest measured ratio; zero means no measured ratio applies.
  std::size_t ratio_samples = 0;
  /// True when anchored on a planner estimate. Sticky and independent of @ref ratio_samples.
  bool planner_derived = false;
};

/// Tuning for a single estimation call.
struct size_estimate_options {
  /// Substitute 1:1 for unavailable pipeline ratios; never applies to fan-in or marks exact.
  bool assume_unit_ratio = false;
  /// Recursion and cycle guard.
  std::size_t max_hops = 16;
  /// Minimum samples for a single-input ratio; @ref assume_unit_ratio may replace weaker ratios.
  std::size_t min_ratio_samples = 4;
  /// Hard fan-in sample floor; returns nullopt below it even with @ref assume_unit_ratio.
  /// See docs/super-sirius/data-size-estimation.md#fan-in.
  std::size_t min_fan_in_ratio_samples = 16;
};

/**
 * @brief Project the total bytes arriving at @p op's @p port_id input port.
 *
 * @return nullopt for a missing port, a dependency-only port (null repo), a port with no
 *         producer, or when the upstream walk cannot produce an estimate.
 */
[[nodiscard]] std::optional<data_size_estimate> estimate_port_total_input_bytes(
  op::sirius_physical_operator& op, std::string_view port_id, size_estimate_options options = {});

/**
 * @brief Project the total bytes @p pipeline will emit over the whole query.
 *
 * Walks upstream to a known total, then applies intervening output/input ratios:
 *
 *  1. finished pipeline: exact recorded output;
 *  2. fan-in: nominated primary port;
 *  3. leaf: source total;
 *  4. single producer: recurse and scale.
 *
 * @return nullopt for an unknown link or an unfinished output-capped pipeline.
 */
[[nodiscard]] std::optional<data_size_estimate> estimate_pipeline_total_output_bytes(
  sirius_pipeline& pipeline, size_estimate_options options = {});

/**
 * @brief `bytes * ratio`, or nullopt if the product would not survive narrowing to std::size_t.
 *
 * Shared with the source hooks that feed the estimator.
 */
[[nodiscard]] std::optional<std::size_t> scale_bytes_checked(std::size_t bytes, double ratio);

/**
 * @brief A leaf source's whole-query output total, projected from a planner row estimate and
 *        floored at what it has already emitted.
 *
 * `estimated_cardinality x (emitted_bytes / emitted_rows)`, then `max(..., emitted_bytes)`.
 *
 * @return nullopt when either measured count is zero or the product cannot narrow to size_t.
 */
[[nodiscard]] std::optional<std::size_t> project_source_output_bytes(
  std::size_t estimated_cardinality, std::size_t emitted_rows, std::size_t emitted_bytes);

}  // namespace pipeline
}  // namespace sirius
