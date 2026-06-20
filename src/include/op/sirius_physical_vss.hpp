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

#include <cstddef>
#include <memory>

namespace sirius {
namespace op {

/**
 * @brief Per-batch brute-force vector top-k.
 *
 * Each batch is reduced to its local k nearest rows and the global merge is
 * done via sirius_physical_vss_merge.
 */
class sirius_physical_vss : public sirius_physical_operator {
 public:
  static constexpr const SiriusPhysicalOperatorType TYPE = SiriusPhysicalOperatorType::VSS;

 public:
  sirius_physical_vss(duckdb::vector<sirius::logical_type> types_p,
                      sirius::vss::vss_top_k_pattern pattern,
                      std::size_t limit,
                      std::size_t offset,
                      std::size_t estimated_cardinality);
  ~sirius_physical_vss() override;

  sirius::vss::vss_top_k_pattern pattern;
  std::size_t limit;
  std::size_t offset;

 public:
  bool is_source() const override { return true; }
  sirius::OrderPreservationType source_order() const override
  {
    return sirius::OrderPreservationType::FIXED_ORDER;
  }

 public:
  bool is_sink() const override { return true; }
  std::unique_ptr<operator_data> execute(const operator_data& input_data,
                                         rmm::cuda_stream_view stream) override;
};

}  // namespace op
}  // namespace sirius
