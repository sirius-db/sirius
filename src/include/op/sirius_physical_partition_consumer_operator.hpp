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

namespace sirius {
namespace op {

//! sirius_physical_partition_consumer_operator is an interface for operators
//! that can consume partitioned data batches
class sirius_physical_partition_consumer_operator : public sirius_physical_operator {
 public:
  sirius_physical_partition_consumer_operator(SiriusPhysicalOperatorType type,
                                              duckdb::vector<duckdb::LogicalType> types,
                                              duckdb::idx_t estimated_cardinality)
    : sirius_physical_operator(type, std::move(types), estimated_cardinality)
  {
  }

  virtual ~sirius_physical_partition_consumer_operator();

  //! Push a data batch to a specific port with partition information
  //! @param port_id The port identifier
  //! @param batch The data batch to push
  //! @param partition_idx The partition index
  virtual void push_data_batch_partitioned(std::string_view port_id,
                                           std::shared_ptr<::cucascade::data_batch> batch,
                                           duckdb::idx_t partition_idx);
};

}  // namespace op
}  // namespace sirius
