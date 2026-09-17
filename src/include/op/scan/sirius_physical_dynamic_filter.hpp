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

#include <op/dynamic_filter/sirius_dynamic_filter.hpp>
#include <op/scan/dynamic_filter_gate.hpp>
#include <op/scan/dynamic_filter_merge.hpp>
#include <op/sirius_physical_operator.hpp>

#include <cstddef>
#include <memory>

namespace sirius::op::scan {

/// @brief Applies visible dynamic filters at scan or direct-route endpoints.
///
/// Mode controls whether AST masks supplement membership masks; finalization closes the channel.
class sirius_physical_dynamic_filter : public sirius_physical_operator {
 public:
  static constexpr SiriusPhysicalOperatorType TYPE = SiriusPhysicalOperatorType::DYNAMIC_FILTER;

  sirius_physical_dynamic_filter(
    duckdb::vector<sirius::logical_type> types,
    std::size_t estimated_cardinality,
    std::shared_ptr<sirius::op::sirius_dynamic_filter_set> filters,
    double gate_keep_threshold     = dynamic_filter_gate::k_default_keep_threshold,
    dynamic_filter_apply_mode mode = dynamic_filter_apply_mode::membership_masks_only);

  std::unique_ptr<operator_data> execute(const operator_data& input_data,
                                         rmm::cuda_stream_view stream) override;

  void on_finalize_operator() override;

  /// Filtering never expands its input, so the peak estimate is the input footprint.
  [[nodiscard]] std::size_t no_history_peak_memory_estimate(const input_stats& stats) const override
  {
    return stats.bytes;
  }

 private:
  std::shared_ptr<sirius::op::sirius_dynamic_filter_set> _filters;
  dynamic_filter_gate _gate;
  dynamic_filter_apply_mode _mode;
};

}  // namespace sirius::op::scan
