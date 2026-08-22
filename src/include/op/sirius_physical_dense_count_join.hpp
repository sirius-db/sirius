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
 * @brief Fuse an eligible preserved-side outer equi-join and grouped COUNT
 *
 * Children are [preserved, counted] and output is [key, BIGINT]. Runtime selects direct-address or
 * exact sparse aggregation while preserving outer-join NULL semantics.
 */
class sirius_physical_dense_count_join : public sirius_physical_operator {
 public:
  static constexpr SiriusPhysicalOperatorType TYPE = SiriusPhysicalOperatorType::DENSE_COUNT_JOIN;

  static constexpr std::string_view PRESERVED_PORT = "preserved";
  static constexpr std::string_view COUNTED_PORT   = "counted";

  /// @p counted_value_idx selects COUNT(col) validity; std::nullopt means COUNT(*).
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

  std::optional<task_creation_hint> get_next_task_hint() override;

  std::unique_ptr<operator_data> get_next_task_input_data() override;

  [[nodiscard]] std::size_t no_history_peak_memory_estimate(
    const input_stats& stats) const override;

  [[nodiscard]] std::size_t preserved_key_idx() const noexcept { return _preserved_key_idx; }
  [[nodiscard]] std::size_t counted_key_idx() const noexcept { return _counted_key_idx; }
  [[nodiscard]] std::optional<std::size_t> counted_value_idx() const noexcept
  {
    return _counted_value_idx;
  }
  [[nodiscard]] uint64_t max_bins_bytes() const noexcept { return _max_bins_bytes; }

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
