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

#include "op/sirius_physical_partition_consumer_operator.hpp"

namespace sirius {
namespace op {

sirius_physical_partition_consumer_operator::~sirius_physical_partition_consumer_operator() {}

void sirius_physical_partition_consumer_operator::push_data_batch_partitioned(
  std::string_view port_id,
  std::shared_ptr<::cucascade::data_batch> batch,
  duckdb::idx_t partition_idx)
{
  auto* p = get_port(port_id);
  if (p && p->repo) { p->repo->add_data_batch(batch, partition_idx); }
}

}  // namespace op
}  // namespace sirius
