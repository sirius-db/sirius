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

#include "duckdb/planner/bound_query_node.hpp"
#include "op/sirius_physical_operator.hpp"
#include "op/sirius_physical_order.hpp"
#include "op/sort_comparison_config.hpp"

#include <cucascade/data/data_repository_manager.hpp>

#include <limits>
#include <mutex>

namespace sirius {
namespace op {

class sirius_physical_merge_sort : public sirius_physical_operator {
 public:
  [[nodiscard]] sirius::OrderPreservationType output_order_preservation() const override
  {
    return sirius::OrderPreservationType::FIXED_ORDER;
  }

  static constexpr const SiriusPhysicalOperatorType TYPE = SiriusPhysicalOperatorType::MERGE_SORT;

 public:
  sirius_physical_merge_sort(sirius_physical_order* order_by);

  sirius_physical_merge_sort(duckdb::vector<sirius::logical_type> types,
                             duckdb::vector<duckdb::BoundOrderByNode> orders,
                             duckdb::vector<std::size_t> projections_p,
                             std::size_t estimated_cardinality,
                             bool is_index_sort_p                                       = false,
                             std::shared_ptr<const sort_comparison_config> comparison_p = nullptr);

  //! Input data
  duckdb::vector<duckdb::BoundOrderByNode> orders;
  std::shared_ptr<const sort_comparison_config> comparison;
  duckdb::vector<std::size_t> projections;
  bool is_index_sort;

  // Positive sizing target for the common merge implementation.
  std::size_t chunk_bytes{config::DEFAULT_SORT_CHUNK_BYTES};
  std::size_t reservation_limit{std::numeric_limits<std::size_t>::max()};
  std::size_t no_history_peak_memory_estimate(const input_stats& stats) const override;
  // Runtime wiring registers idle intermediates before any creator or downgrade readers run.
  void register_scratch(cucascade::shared_data_repository_manager& manager);
  bool all_ports_empty() override;
  std::optional<task_creation_hint> get_next_task_hint() override;
  void sink(const operator_data& output, ::cuda::stream_ref stream) override;
  // gpu_pipeline_task calls this only after successful sink publication and stream completion.
  void on_task_output_published(const operator_data& output) override;
  // Before publication, preparation can request the same finite subdivision
  // used by compute OOM. The returned empty task input commits through the
  // ordinary sink/completion path. nullptr permits reservation-growth retry.
  std::unique_ptr<operator_data> recover_preparation_failure(const operator_data& input,
                                                             task_attempt_context attempt) override;
  void notify_task_attempt(operator_data& input, task_attempt_context attempt) override;
  bool declared_output_schema_is_runtime_schema() const noexcept override
  {
    // Intermediate bounded steps retain hidden keys until final publication.
    return false;
  }

  sirius_physical_operator* child_op{nullptr};
  sirius_physical_operator* get_child_op() const { return child_op; }

 public:
  // Source interface
  bool is_source() const override { return true; }

 public:
  // Sink interface
  bool is_sink() const override { return true; }

 public:
  std::unique_ptr<operator_data> get_next_task_input_data() override;

  std::unique_ptr<operator_data> execute(const operator_data& input_data,
                                         ::cuda::stream_ref stream) override;

  //! Set the final output projection (applied after merge, to remove sort-key-only columns)
  void set_final_projections(duckdb::vector<std::size_t> proj,
                             duckdb::vector<sirius::logical_type> output_types)
  {
    _final_projections = std::move(proj);
    types              = std::move(output_types);
  }

 private:
  //! Final projection to apply after merge (empty = no extra projection)
  duckdb::vector<std::size_t> _final_projections;
  //! Guards concurrent calls to get_next_task_input_data().
  std::mutex _drain_mutex;

  struct tiled_state;
  std::shared_ptr<tiled_state> _tiled_state;
  std::vector<cucascade::shared_data_repository*> _tile_storage;
};

}  // namespace op
}  // namespace sirius
