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

#include "expression/ast/node.hpp"
#include "op/sirius_physical_operator.hpp"

#include <memory>

namespace sirius {
namespace op {

class sirius_physical_projection : public sirius_physical_operator {
 public:
  static constexpr const SiriusPhysicalOperatorType TYPE = SiriusPhysicalOperatorType::PROJECTION;

 public:
  sirius_physical_projection(duckdb::vector<sirius::logical_type> types,
                             duckdb::vector<std::unique_ptr<sirius::ast::node>> select_list,
                             std::size_t estimated_cardinality);

  std::unique_ptr<operator_data> execute(const operator_data& input_data,
                                         rmm::cuda_stream_view stream) override;

  //! One-line list of the projected expressions for telemetry / debug display.
  std::string params_to_string() const override;

  duckdb::vector<std::unique_ptr<sirius::ast::node>> select_list;
};

}  // namespace op
}  // namespace sirius
