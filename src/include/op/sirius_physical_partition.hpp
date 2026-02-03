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
#include "duckdb/planner/expression/bound_reference_expression.hpp"
#include "op/sirius_physical_grouped_aggregate.hpp"
#include "op/sirius_physical_hash_join.hpp"
#include "op/sirius_physical_operator.hpp"
#include "op/sirius_physical_order.hpp"
#include "op/sirius_physical_top_n.hpp"

#define PARTITION_SIZE 10000000

namespace sirius {
namespace op {

class sirius_physical_partition : public sirius_physical_operator {
 public:
  static constexpr const SiriusPhysicalOperatorType TYPE = SiriusPhysicalOperatorType::PARTITION;

  explicit sirius_physical_partition(duckdb::vector<duckdb::LogicalType> types,
                                     duckdb::idx_t estimated_cardinality,
                                     sirius_physical_operator* parent_op,
                                     bool is_build = false);

  std::string get_name() const override;

  bool is_source() const override;

  bool is_sink() const override;

  bool is_build_partition();

  //! Get the parent operator (e.g., HASH_JOIN for build partition)
  sirius_physical_operator* get_parent_op() const { return _parent_op; }

 private:
  void get_partition_keys(sirius_physical_operator* op, bool is_build = false);
  sirius_physical_operator* _parent_op;
  duckdb::vector<duckdb::idx_t> _partition_keys;
  duckdb::idx_t _num_partitions;
  bool _is_build;
};

}  // namespace op
}  // namespace sirius
