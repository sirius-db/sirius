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

namespace sirius {
namespace op {

class sirius_physical_dummy_scan : public sirius_physical_operator {
 public:
  static constexpr const SiriusPhysicalOperatorType TYPE = SiriusPhysicalOperatorType::DUMMY_SCAN;

 public:
  explicit sirius_physical_dummy_scan(duckdb::vector<sirius::logical_type> types,
                                      std::size_t estimated_cardinality)
    : sirius_physical_operator(
        SiriusPhysicalOperatorType::DUMMY_SCAN, std::move(types), estimated_cardinality)
  {
  }

 public:
  bool is_source() const override { return true; }

  //! True when this DUMMY_SCAN is the synthetic build placeholder under a
  //! RIGHT_DELIM_JOIN's internal join. Makes plan-gen skip `wrap_cpu_source`: the
  //! placeholder carries no runtime data (sink() runs partition_join inline). Real
  //! DUMMY_SCAN usages (constant-row subqueries) keep the wrap.
  [[nodiscard]] bool is_delim_join_placeholder() const noexcept
  {
    return _is_delim_join_placeholder;
  }
  void set_delim_join_placeholder(bool value) noexcept { _is_delim_join_placeholder = value; }

 protected:
  bool _is_delim_join_placeholder = false;
};

}  // namespace op
}  // namespace sirius
