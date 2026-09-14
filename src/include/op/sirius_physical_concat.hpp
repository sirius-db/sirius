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
#include "op/sirius_physical_grouped_aggregate.hpp"
#include "op/sirius_physical_hash_join.hpp"
#include "op/sirius_physical_operator.hpp"
#include "op/sirius_physical_order.hpp"
#include "op/sirius_physical_partition_consumer_operator.hpp"
#include "op/sirius_physical_top_n.hpp"
#include "sirius_config.hpp"

namespace sirius {
namespace op {

/**
 * @brief Groups partitioned input batches before a downstream join
 *
 * Buffers batches in arrival order for each partition and releases groups based on
 * `_concat_batch_bytes`. `get_next_task_hint` and `get_next_task_input_data` use the same grouping
 * policy so a ready hint always corresponds to data that can be pulled.
 */
class sirius_physical_concat : public sirius_physical_partition_consumer_operator {
 public:
  static constexpr const SiriusPhysicalOperatorType TYPE = SiriusPhysicalOperatorType::CONCAT;

  //! `downstream_join` is the HJ/NLJ this CONCAT feeds — not the tree parent (that is
  //! `_parent_op`, stamped by `set_parent_ops`). Its join type picks `_concat_all`; the
  //! pointer is retained for the legacy converter's destination lookup.
  explicit sirius_physical_concat(
    duckdb::vector<sirius::logical_type> types,
    std::size_t estimated_cardinality,
    sirius_physical_operator* downstream_join,
    bool is_build,
    uint64_t concat_batch_bytes = sirius::config::DEFAULT_CONCAT_BATCH_BYTES);

  std::string get_name() const override;

  bool is_source() const override;

  bool is_sink() const override;

  bool is_build_concat() const;

  //! The downstream HJ/NLJ this CONCAT feeds; distinct from `get_parent_op()` (tree parent).
  [[nodiscard]] sirius_physical_operator* get_downstream_join() const noexcept
  {
    return _downstream_join;
  }

  std::optional<task_creation_hint> get_next_task_hint() override;

  std::unique_ptr<operator_data> get_next_task_input_data() override;

  std::unique_ptr<operator_data> execute(const operator_data& input_data,
                                         rmm::cuda_stream_view stream) override;

  void sink(const operator_data& output_data, rmm::cuda_stream_view stream) override;

  //! Used when PARTITION + `get_partition_strategy` selects BUILD_PROBE: merge all build batches
  //! before the join so the hash join sees a single build batch.
  void set_concat_all(bool concat_all);

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

  bool _is_build;
  bool _concat_all;
  uint64_t _concat_batch_bytes;
  //! Non-owning. Captured at construction from the `downstream_join` ctor argument.
  sirius_physical_operator* _downstream_join = nullptr;
};

}  // namespace op
}  // namespace sirius
