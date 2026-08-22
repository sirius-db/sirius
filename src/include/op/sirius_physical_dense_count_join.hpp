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

#include "op/sirius_physical_operator.hpp"

#include <cstdint>
#include <optional>
#include <string_view>

namespace sirius::op {

/** @brief Input container that retains the logical side of every dense-count batch. */
class dense_count_join_input : public pipelineable_operator_data {
 public:
  enum class input_side : uint8_t { PRESERVED, COUNTED };

  dense_count_join_input(std::vector<std::shared_ptr<::cucascade::data_batch>> preserved_batches,
                         std::vector<std::shared_ptr<::cucascade::data_batch>> counted_batches);

  [[nodiscard]] std::vector<input_side> const& input_sides() const noexcept { return _input_sides; }

 private:
  struct tagged_batches {
    std::vector<std::shared_ptr<::cucascade::data_batch>> batches;
    std::vector<input_side> sides;
  };

  explicit dense_count_join_input(tagged_batches input);
  [[nodiscard]] static tagged_batches tag_batches(
    std::vector<std::shared_ptr<::cucascade::data_batch>> preserved_batches,
    std::vector<std::shared_ptr<::cucascade::data_batch>> counted_batches);

  std::vector<input_side> _input_sides;
};

/**
 * @brief Fused count-join: computes `SELECT key, COUNT(col | *) ... GROUP BY key` over a
 * preserved-side outer equi-join without materializing the join.
 *
 * Replaces the plan fragment
 *   HASH_GROUP_BY(groups = [preserved-side join key], aggregates = [COUNT(counted-side col)
 *   or COUNT(*)]) over LEFT/RIGHT LogicalComparisonJoin(single integer equality condition)
 * detected by `sirius_physical_plan_generator::try_plan_dense_count_join`. Children are
 * normalized: children[0] = preserved side (the outer-preserved, grouped input), children[1] =
 * counted side (the other input, which may contain a pushed-down filter).
 *
 * Both input ports are FULL barriers; a single task drains both sides once both child pipelines
 * finish and picks one of two exact execution strategies from the ACTUAL preserved-side key
 * min/max and input size (never from estimates):
 *  - **dense**: two direct-address histograms over [min, max] (see `dense_count_state`) when
 *    the range is sufficiently dense and the combined histogram bytes fit both the configured
 *    histogram budget and a multiple of the input size — the dense direct-address fast path;
 *  - **sparse**: eager aggregation via cuDF (per-batch groupby-count partials, merge, left join
 *    of the distinct preserved keys against the counted aggregate) when the key range is too
 *    wide for a direct-address array.
 * Both strategies produce identical results: one row per distinct non-NULL preserved key
 * (duplicate preserved keys multiply the per-key match count, matching join-then-group-by
 * semantics exactly), plus the SQL NULL group when the preserved side has NULL keys.
 *
 * Output schema: [group key (input key type), COUNT (BIGINT)].
 */
class sirius_physical_dense_count_join : public sirius_physical_operator {
 public:
  static constexpr SiriusPhysicalOperatorType TYPE = SiriusPhysicalOperatorType::DENSE_COUNT_JOIN;

  /// Input port fed by the preserved-side child pipeline (children[0]).
  static constexpr std::string_view PRESERVED_PORT = "preserved";
  /// Input port fed by the counted-side child pipeline (children[1]).
  static constexpr std::string_view COUNTED_PORT = "counted";

  /**
   * @param types Output schema: [key logical type, BIGINT].
   * @param estimated_cardinality Estimated number of output groups.
   * @param preserved_key_idx Join-key column index within the preserved child's output.
   * @param counted_key_idx Join-key column index within the counted child's output.
   * @param counted_value_idx COUNT(col) argument column index within the counted child's
   *        output (its validity mask supplies the exact COUNT NULL semantics), or std::nullopt
   *        for COUNT(*).
   * @param max_bins_bytes Cap on the combined direct-address histogram size. The runtime gate
   *        also rejects ranges that are sparse relative to the actual rows or input bytes.
   */
  sirius_physical_dense_count_join(duckdb::vector<sirius::logical_type> types,
                                   std::size_t estimated_cardinality,
                                   std::size_t preserved_key_idx,
                                   std::size_t counted_key_idx,
                                   std::optional<std::size_t> counted_value_idx,
                                   uint64_t max_bins_bytes);

  std::string params_to_string() const override;
  [[nodiscard]] std::string_view input_port_for(
    sirius_physical_operator const& producer) const override;
  [[nodiscard]] MemoryBarrierType input_barrier_for(
    sirius_physical_operator const& producer) const override;

  std::unique_ptr<operator_data> execute(const operator_data& input_data,
                                         rmm::cuda_stream_view stream) override;

  bool is_source() const override { return true; }
  bool is_sink() const override { return true; }

  void build_pipelines(pipeline::sirius_pipeline& current,
                       pipeline::sirius_meta_pipeline& meta_pipeline) override;

  /// FULL-barrier semantics on both ports, but READY once both producers finished and ANY
  /// batches are queued (the base hint requires every port non-empty, which would strand the
  /// one-empty-side cases: an empty counted side must still emit the all-zero-count groups).
  std::optional<task_creation_hint> get_next_task_hint() override;

  /// Drains ALL batches from both ports into a single `dense_count_join_input` (one task
  /// computes the whole aggregate).
  std::unique_ptr<operator_data> get_next_task_input_data() override;

  /// Conservative first-run peak across the dense, sparse, and extrema-reduction strategies.
  [[nodiscard]] std::size_t no_history_peak_memory_estimate(
    const input_stats& stats) const override;

  [[nodiscard]] std::size_t preserved_key_idx() const noexcept { return _preserved_key_idx; }
  [[nodiscard]] std::size_t counted_key_idx() const noexcept { return _counted_key_idx; }
  [[nodiscard]] std::optional<std::size_t> counted_value_idx() const noexcept
  {
    return _counted_value_idx;
  }
  [[nodiscard]] uint64_t max_bins_bytes() const noexcept { return _max_bins_bytes; }

  /// Strategy taken by the most recent execute() — for logging and tests.
  enum class strategy : uint8_t { NOT_RUN, DENSE, SPARSE };
  [[nodiscard]] strategy last_strategy() const noexcept { return _last_strategy; }

 private:
  std::size_t _preserved_key_idx;
  std::size_t _counted_key_idx;
  std::optional<std::size_t> _counted_value_idx;
  uint64_t _max_bins_bytes;
  strategy _last_strategy = strategy::NOT_RUN;
};

}  // namespace sirius::op
