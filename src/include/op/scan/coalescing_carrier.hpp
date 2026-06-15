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

// sirius
#include <op/scan/coalescing_unit.hpp>
#include <op/sirius_physical_operator.hpp>

// standard library
#include <vector>

namespace sirius::op::scan {

/**
 * @brief Format-blind transport from a parallel producer (split provider) to a single coalescing
 * consumer.
 *
 * One carrier holds the coalescing_units a single producer task (scan manager thread) parsed.
 */
class coalescing_carrier : public op::operator_data {
 public:
  std::vector<coalescing_unit> units;

  [[nodiscard]] op::operator_data_type get_type() const override
  {
    return op::operator_data_type::GPU_SCAN;
  }
};

}  // namespace sirius::op::scan
