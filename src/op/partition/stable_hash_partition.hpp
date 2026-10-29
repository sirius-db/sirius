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

/// The reordered table and the start offset of each partition, in the form `cudf::hash_partition`
/// returns.
using hash_partition_result = std::pair<std::unique_ptr<cudf::table>, std::vector<cudf::size_type>>;

/**
 * @brief Hash-partitions the rows of @p input and keeps their input order within each
 * partition.
 *
 * A row's partition is its MurmurHash3 (x86, 32-bit) row hash over @p key_indices modulo
 * @p num_partitions, the same as in `unordered_hash_partition`, so equal keys land in the same
 * partition whichever of the two partitions a batch. Keeping the order lets a consumer merge
 * partitions of sorted inputs as sorted data.
 */
hash_partition_result stable_hash_partition(cudf::table_view const& input,
                                            std::vector<cudf::size_type> const& key_indices,
                                            int num_partitions,
                                            rmm::cuda_stream_view stream,
                                            rmm::device_async_resource_ref mr);

/**
 * @brief Hash-partitions the rows of @p input without any ordering guarantee within a partition.
 *
 * Assigns partitions exactly as `stable_hash_partition` does, but skips its sort, so it is cheaper
 * for input whose order is of no use to the consumer.
 */
hash_partition_result unordered_hash_partition(cudf::table_view const& input,
                                               std::vector<cudf::size_type> const& key_indices,
                                               int num_partitions,
                                               rmm::cuda_stream_view stream,
                                               rmm::device_async_resource_ref mr);

}  // namespace sirius::op
