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

#include "op/partition/consumer_partition_policy.hpp"
#include "op/sirius_physical_hash_join.hpp"
#include "op/sirius_physical_partition.hpp"

namespace sirius::test::operator_utils {
inline op::partition_spec hash_partition_spec(op::sirius_physical_hash_join const& join, bool build)
{
  op::partition_spec spec;
  for (auto const& key : join.prepared_keys()) {
    spec.key_columns.push_back(build ? key.right_index : key.left_index);
    spec.key_cast_types.push_back(key.type);
  }
  return spec;
}
inline op::partition_spec single_partition_spec()
{
  op::partition_spec spec;
  spec.mode = op::partition_mode::single;
  return spec;
}
inline op::partition_layout fixed_hash_layout(std::size_t count)
{
  return {op::partition_mode::hash, op::partition_placement::unpinned(count)};
}
}  // namespace sirius::test::operator_utils
