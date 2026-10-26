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

#include "duckdb/execution/physical_operator.hpp"
#include "op/sirius_physical_operator.hpp"
#include "op/sirius_physical_partition_consumer_operator.hpp"
#include "sirius_config.hpp"

namespace sirius {
namespace op {

/**
 * @brief Groups batches within an existing input partition layout
 *
 * Traverses batches in repository order for each partition and releases groups based on
 * `_concat_batch_bytes`. `get_next_task_hint` and `get_next_task_input_data` use the same grouping
 * policy so a ready hint always corresponds to data that can be pulled.
 */
class sirius_physical_concat : public sirius_physical_partition_consumer_operator {
 public:
  static constexpr const SiriusPhysicalOperatorType TYPE = SiriusPhysicalOperatorType::CONCAT;

  explicit sirius_physical_concat(
    duckdb::vector<sirius::logical_type> types,
    std::size_t estimated_cardinality,
    bool concat_all,
    uint64_t concat_batch_bytes = sirius::config::DEFAULT_CONCAT_BATCH_BYTES);

  std::string get_name() const override;

  bool is_source() const override;

  bool is_sink() const override;

  [[nodiscard]] MemoryBarrierType input_barrier_for(
    sirius_physical_operator const& producer) const override;

  std::optional<task_creation_hint> get_next_task_hint() override;

  std::unique_ptr<operator_data> get_next_task_input_data() override;

  std::unique_ptr<operator_data> execute(const operator_data& input_data,
                                         ::cuda::stream_ref stream) override;

  void sink(const operator_data& output_data, ::cuda::stream_ref stream) override;

  //! Configure complete-partition grouping before the first claim. Repeating the current
  //! requirement is safe; changing it after any claim throws std::logic_error.
  void set_concat_all(bool concat_all);

  //! Whether each partition is released as one group once the source pipeline finishes.
  [[nodiscard]] bool concat_all() const { return _concat_all; }

  [[nodiscard]] std::size_t no_history_peak_memory_estimate(
    const op::input_stats& stats) const override;

 private:
  /**
   * @brief Plans the next batch group for one partition without removing data
   *
   * A group contains the batches before the first batch that would make its cumulative size exceed
   * `_concat_batch_bytes`; the overflowing batch remains for the next group. A batch that exceeds
   * the threshold by itself becomes a single-batch group after another batch arrives or the source
   * pipeline finishes. Finishing the pipeline also releases an under-threshold tail. When
   * `_concat_all` is set, the complete partition becomes one group after the pipeline finishes.
   * Known row counts also bound each group by the cuDF row-count limit; a complete partition
   * above that limit is rejected before any batches are claimed or transferred.
   *
   * The caller must hold the operator mutex `lock`. The plan remains valid while the lock is held
   * because consumers remove batches under the same lock and producers only append batches.
   *
   * @param[in] repo Repository for the operator's input port
   * @param[in] partition_idx Partition to inspect
   * @param[in] pipeline_finished Whether the source pipeline has finished
   * @return Batch identifiers in pull order, or `std::nullopt` when no group is ready
   */
  [[nodiscard]] std::optional<std::vector<uint64_t>> plan_pull_for_partition(
    ::cucascade::shared_data_repository& repo,
    std::size_t partition_idx,
    bool pipeline_finished) const;

  bool _input_claimed = false;
  bool _concat_all;
  uint64_t _concat_batch_bytes;
};

}  // namespace op
}  // namespace sirius
