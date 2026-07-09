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

#include "op/sirius_physical_operator.hpp"
#include "vss/vss_pattern.hpp"

#include <atomic>
#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace duckdb {
class SiriusContext;
}  // namespace duckdb

namespace sirius {
namespace op {

/**
 * @brief Approximate vector top-k via a pinned cuVS IVF-Flat index.
 *
 * A pure source (no operator children). At execute, it:
 *   1. looks up the pinned IVF-Flat index (by table + vector column + metric),
 *   2. searches it for the global k = offset+limit nearest rows of the query,
 *   3. gathers the requested output columns from the GPU-resident pinned table
 *      (whose rows are in the same order the index was built in) by the returned
 *      row indices, and
 *   4. assembles + slices the projection's output (passthroughs + distance).
 *
 * Both the index (via @c sirius_create_ann_index) and the table (via
 * @c pin_table tier='gpu') must be resident. Produces a single
 * task, then signals exhaustion.
 */
class sirius_physical_vss_ann_ivf_flat : public sirius_physical_operator {
 public:
  static constexpr const SiriusPhysicalOperatorType TYPE = SiriusPhysicalOperatorType::ANN_IVF_FLAT;

  sirius_physical_vss_ann_ivf_flat(duckdb::vector<sirius::logical_type> types_p,
                                   sirius::vss::vss_top_k_pattern pattern_p,
                                   std::size_t limit,
                                   std::size_t offset,
                                   std::size_t estimated_cardinality,
                                   duckdb::SiriusContext* sirius_context,
                                   std::string table_name,
                                   std::string vector_column_name,
                                   std::vector<std::string> output_column_names);
  ~sirius_physical_vss_ann_ivf_flat() override;

  sirius::vss::vss_top_k_pattern pattern;
  std::size_t limit;
  std::size_t offset;

  // Source interface
  bool is_source() const override { return true; }
  sirius::OrderPreservationType source_order() const override
  {
    return sirius::OrderPreservationType::FIXED_ORDER;
  }

  std::optional<task_creation_hint> get_next_task_hint() override;
  std::unique_ptr<operator_data> get_next_task_input_data() override;
  bool all_ports_empty() override;

  std::unique_ptr<operator_data> execute(const operator_data& input_data,
                                         rmm::cuda_stream_view stream) override;

 private:
  /// Session context (outlives the query) for the scan manager, memory manager,
  /// and cuVS index cache used at execute. Non-owning.
  duckdb::SiriusContext* sirius_context_;
  /// Catalog-resolved table name, keys both the pinned index (with
  /// @c vector_column_name_ + @c pattern.metric) and the pinned table.
  std::string table_name_;
  /// Vector column the index was built on (index lookup key).
  std::string vector_column_name_;
  /// Base-table column name for each @c pattern.output_columns entry of kind
  /// gather_input (aligned by index; distance entries hold an empty string).
  std::vector<std::string> output_column_names_;

  /// One-shot dispatch state: flipped true once the single task is handed out,
  /// so the pipeline schedules exactly one task and then finishes.
  std::atomic<bool> dispatched_{false};
};

}  // namespace op
}  // namespace sirius
