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
#include "op/sirius_physical_order.hpp"
#include "op/sort_key_sample.hpp"

#include <cstddef>
#include <memory>

namespace sirius {
namespace op {

/**
 * @brief Keeps evenly spaced sort-key rows of every sorted batch, for SORT_SAMPLE to pick the
 * range boundaries from.
 *
 * Runs after ORDER_BY and ends its pipeline. The batches pass through unchanged.
 */
class sirius_physical_sort_key_sample : public sirius_physical_operator {
 public:
  [[nodiscard]] sirius::OrderPreservationType output_order_preservation() const override
  {
    return sirius::OrderPreservationType::NO_ORDER;
  }

  static constexpr const SiriusPhysicalOperatorType TYPE =
    SiriusPhysicalOperatorType::SORT_KEY_SAMPLE;

  //! @param key_sample Where the key rows are kept: the store the plan's SORT_SAMPLE reads.
  sirius_physical_sort_key_sample(sirius_physical_order* order_by,
                                  std::shared_ptr<sort_key_sample> key_sample);

  sirius_physical_sort_key_sample(duckdb::vector<sirius::logical_type> types,
                                  duckdb::vector<duckdb::BoundOrderByNode> orders,
                                  std::size_t estimated_cardinality,
                                  std::shared_ptr<sort_key_sample> key_sample);

  //! Order specification (copied from ORDER_BY): the columns whose rows are kept.
  duckdb::vector<duckdb::BoundOrderByNode> orders;

  bool is_source() const override { return true; }
  bool is_sink() const override { return true; }

  //! Keeps the key rows of each batch and returns the batches unchanged. A batch that a retried
  //! task hands over again replaces its earlier rows.
  std::unique_ptr<operator_data> execute(const operator_data& input_data,
                                         ::cuda::stream_ref stream) override;

  //! The key rows are a small part of a batch that ORDER_BY has already sized the task for.
  [[nodiscard]] std::size_t no_history_peak_memory_estimate(
    const input_stats& /*stats*/) const override
  {
    return 0;
  }

  [[nodiscard]] std::shared_ptr<sort_key_sample> const& key_sample() const noexcept
  {
    return _key_sample;
  }

 private:
  std::shared_ptr<sort_key_sample> _key_sample;
};

}  // namespace op
}  // namespace sirius
