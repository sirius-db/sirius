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
 * @brief Projected total bytes that will flow through a point in the plan.
 *
 * A total for the whole query, not a so-far figure: "how many bytes will this port have
 * received once its producer is done?". See docs/super-sirius/data-size-estimation.md.
 */
struct data_size_estimate {
  std::size_t bytes = 0;
  /// Measured rather than projected: the walk anchored on a finished pipeline or an exactly
  /// known source total, with no learned ratio applied anywhere along the chain.
  bool exact = false;
  /// Pipelines traversed. Diagnostic — ratio error compounds per hop.
  std::size_t hops = 0;
  /// Completed tasks behind the *weakest measured* ratio in the chain. Separates "we sampled too
  /// early" from "the model is wrong" when an estimate misses. Zero means no measured ratio backs
  /// the result: an exact estimate, or — under @ref size_estimate_options::assume_unit_ratio — a
  /// chain that substituted 1:1 at every link. @ref exact separates the two.
  std::size_t ratio_samples = 0;
  /// The walk anchored on a planner guess rather than a measurement — in practice
  /// `GPU_SCAN::total_source_output_bytes()`, built from DuckDB's `estimated_cardinality`.
  /// Sticky: every estimate chained on top of such an anchor inherits it. Not inferable from
  /// `ratio_samples == 0`, which holds only for the anchor's own pipeline — one downstream hop
  /// with a trusted ratio overwrites that zero (see weaker_sample_count in the .cpp).
  bool planner_derived = false;
};

/// Tuning for a single estimation call.
struct size_estimate_options {
  /// Use a 1:1 ratio where the measured one is unusable, instead of returning nullopt. Never
  /// marks the result exact. Applies to *pipeline* ratios only: 1:1 reads as "assume
  /// pass-through", which a join — free to multiply or divide its input volume — cannot claim.
  bool assume_unit_ratio = false;
  /// Recursion guard against pathological plan depth, and defensively against graph cycles.
  std::size_t max_hops = 16;
  /// Sample floor for a single-input pipeline ratio. Below it the ratio is treated as absent, so
  /// @ref assume_unit_ratio still applies.
  std::size_t min_ratio_samples = 4;
  /// Sample floor for a fan-in ratio. Far higher, because that ratio is systematically biased low
  /// while tasks are in flight where a single-input one is merely noisy — and a **hard gate**:
  /// nullopt below it even under @ref assume_unit_ratio. See data-size-estimation.md#fan-in.
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
 * Walks upstream to the first known total, then chains each intervening pipeline's measured
 * output/input ratio back down. Four terminating cases, in the order the implementation tries
 * them (see data_size_estimator.cpp for why each is shaped as it is):
 *
 *  1. finished pipeline — its recorded output total, exactly;
 *  2. fan-in — follow only the source's nominated primary port;
 *  3. leaf — anchor on the source's own total;
 *  4. single producer — recurse, then apply this pipeline's ratio.
 *
 * An unfinished pipeline holding an operator that caps its output by row count (a LIMIT) short
 * circuits to nullopt ahead of 2-4: no ratio extrapolates through a cap.
 *
 * @return nullopt whenever any link in the chain is unknown.
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
 * Backs `total_source_output_bytes()` on scan-like sources.
 *
 * The floor is not defensive rounding. A planner cardinality is a pre-execution guess at a
 * post-filter row count, unbounded below and routinely under the rows already produced; DuckDB
 * pins it to exactly zero whenever the base table cardinality reads zero, which a stale or absent
 * stat on a non-empty table will do. Unfloored, a leaf anchor could report zero and every
 * downstream hop would multiply it out to a zero whole-query total.
 *
 * @return nullopt when no batch has been measured yet (either count zero), or when the product
 *         would not survive narrowing.
 */
[[nodiscard]] std::optional<std::size_t> project_source_output_bytes(
  std::size_t estimated_cardinality, std::size_t emitted_rows, std::size_t emitted_bytes);

}  // namespace pipeline
}  // namespace sirius
