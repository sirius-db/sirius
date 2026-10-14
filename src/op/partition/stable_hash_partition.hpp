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

#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/resource_ref.hpp>

#include <memory>
#include <utility>
#include <vector>

namespace sirius::op {

/**
 * @brief Hash-partitions the rows of @p input like `cudf::hash_partition`, but keeps the input's
 * row order within each partition.
 *
 * A row's partition is its MurmurHash3 (x86, 32-bit) row hash over @p key_indices modulo
 * @p num_partitions. Equal keys therefore land in the same partition, as long as every producer
 * feeding the same consumer partitions with this function. Keeping the order lets a consumer
 * merge partitions of sorted inputs as sorted data.
 *
 * @return The reordered table and the start offset of each partition, in the same form as
 * `cudf::hash_partition`.
 */
std::pair<std::unique_ptr<cudf::table>, std::vector<cudf::size_type>> stable_hash_partition(
  cudf::table_view const& input,
  std::vector<cudf::size_type> const& key_indices,
  int num_partitions,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr);

}  // namespace sirius::op
