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

#include "op/aggregate/group_key_labels.hpp"

#include <cudf/column/column_factories.hpp>
#include <cudf/join/distinct_hash_join.hpp>
#include <cudf/sorting.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/utilities/error.hpp>

#include <rmm/device_buffer.hpp>

#include <numeric>
#include <utility>
#include <vector>

namespace sirius::op::detail {

group_key_labels make_group_key_labels(cudf::table_view const& keys,
                                       rmm::cuda_stream_view stream,
                                       rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(keys.num_columns() > 0, "Group key table must have at least one column");

  std::vector<cudf::size_type> key_indices(keys.num_columns());
  std::iota(key_indices.begin(), key_indices.end(), 0);
  auto unique_keys = cudf::distinct(keys,
                                    key_indices,
                                    cudf::duplicate_keep_option::KEEP_ANY,
                                    cudf::null_equality::EQUAL,
                                    cudf::nan_equality::ALL_EQUAL,
                                    stream,
                                    mr);

  if (keys.num_rows() == 0) {
    return {.sorted_unique_keys = std::move(unique_keys),
            .labels             = cudf::make_empty_column(cudf::data_type{cudf::type_id::INT32})};
  }

  auto const column_order = std::vector<cudf::order>(keys.num_columns(), cudf::order::ASCENDING);
  auto const null_precedence =
    std::vector<cudf::null_order>(keys.num_columns(), cudf::null_order::AFTER);
  auto sorted_unique_keys =
    cudf::sort(unique_keys->view(), column_order, null_precedence, stream, mr);

  // The build table is unique and sorted, so each matched build row index is the dense label.
  // distinct_hash_join has no build-resource parameter; Sirius installs the active memory-space
  // allocator as cuDF's current device resource before operators execute.
  cudf::distinct_hash_join label_lookup(
    sorted_unique_keys->view(), cudf::null_equality::EQUAL, 0.5, stream);
  auto label_indices = label_lookup.left_join(keys, stream, mr);
  auto labels        = std::make_unique<cudf::column>(cudf::data_type{cudf::type_id::INT32},
                                               keys.num_rows(),
                                               label_indices->release(),
                                               rmm::device_buffer{},
                                               0);

  return {.sorted_unique_keys = std::move(sorted_unique_keys), .labels = std::move(labels)};
}

}  // namespace sirius::op::detail
