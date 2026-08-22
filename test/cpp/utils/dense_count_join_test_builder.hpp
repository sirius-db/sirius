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

#include "helper/logical_type.hpp"
#include "op/sirius_physical_dense_count_join.hpp"
#include "op/sirius_physical_operator.hpp"

#include <duckdb/common/helper.hpp>

#include <cstddef>
#include <cstdint>
#include <optional>
#include <utility>

namespace sirius::test {

/** @brief Build the canonical two-child dense-count operator used by planner-pass tests. */
inline duckdb::unique_ptr<op::sirius_physical_dense_count_join> make_dense_count_join(
  std::size_t preserved_key_idx,
  std::size_t counted_key_idx,
  std::optional<std::size_t> counted_value_idx,
  duckdb::unique_ptr<op::sirius_physical_operator> preserved,
  duckdb::unique_ptr<op::sirius_physical_operator> counted)
{
  duckdb::vector<sirius::logical_type> output_types;
  output_types.push_back(sirius::logical_type::make(sirius::type_id::INTEGER));
  output_types.push_back(sirius::logical_type::make(sirius::type_id::BIGINT));
  auto join = duckdb::make_uniq<op::sirius_physical_dense_count_join>(std::move(output_types),
                                                                      /*estimated_cardinality=*/1,
                                                                      preserved_key_idx,
                                                                      counted_key_idx,
                                                                      counted_value_idx,
                                                                      std::uint64_t{1} << 20);
  join->children.push_back(std::move(preserved));
  join->children.push_back(std::move(counted));
  return join;
}

}  // namespace sirius::test
