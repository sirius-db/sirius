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

#include <cudf/binaryop.hpp>
#include <cudf/column/column.hpp>
#include <cudf/copying.hpp>
#include <cudf/filling.hpp>
#include <cudf/hashing.hpp>
#include <cudf/partitioning.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/search.hpp>
#include <cudf/sorting.hpp>
#include <cudf/unary.hpp>
#include <cudf/utilities/error.hpp>

#include <op/partition/stable_hash_partition.hpp>

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

namespace sirius::op {

namespace {

/// Partition number of each row: the MurmurHash3 row hash modulo @p num_partitions.
std::unique_ptr<cudf::column> partition_ids(cudf::table_view const& input,
                                            std::vector<cudf::size_type> const& key_indices,
                                            int num_partitions,
                                            rmm::cuda_stream_view stream)
{
  CUDF_EXPECTS(num_partitions > 0, "hash partition: num_partitions must be positive");
  auto const temp_mr = cudf::get_current_device_resource_ref();
  auto const hashes  = cudf::hashing::murmurhash3_x86_32(
    input.select(key_indices), cudf::DEFAULT_HASH_SEED, stream, temp_mr);
  cudf::numeric_scalar<uint32_t> const partition_count(
    static_cast<uint32_t>(num_partitions), true, stream, temp_mr);
  auto const ids = cudf::binary_operation(hashes->view(),
                                          partition_count,
                                          cudf::binary_operator::MOD,
                                          cudf::data_type{cudf::type_id::UINT32},
                                          stream,
                                          temp_mr);
  return cudf::cast(ids->view(), cudf::data_type{cudf::type_id::INT32}, stream, temp_mr);
}

}  // namespace

hash_partition_result stable_hash_partition(cudf::table_view const& input,
                                            std::vector<cudf::size_type> const& key_indices,
                                            int num_partitions,
                                            rmm::cuda_stream_view stream,
                                            rmm::device_async_resource_ref mr)
{
  auto const temp_mr = cudf::get_current_device_resource_ref();
  auto const ids     = partition_ids(input, key_indices, num_partitions, stream);

  // The stable sort of the partition numbers is the gather map, so rows of one partition keep
  // their input order.
  auto const row_order = cudf::stable_sorted_order(
    cudf::table_view({ids->view()}), {cudf::order::ASCENDING}, {}, stream, temp_mr);
  auto reordered =
    cudf::gather(input, row_order->view(), cudf::out_of_bounds_policy::DONT_CHECK, stream, mr);

  // Partition start offsets: the first row of each partition number in the sorted ids.
  auto const sorted_ids = cudf::gather(cudf::table_view({ids->view()}),
                                       row_order->view(),
                                       cudf::out_of_bounds_policy::DONT_CHECK,
                                       stream,
                                       temp_mr);
  cudf::numeric_scalar<int32_t> const first_partition(0, true, stream, temp_mr);
  auto const needles        = cudf::sequence(num_partitions, first_partition, stream, temp_mr);
  auto const device_offsets = cudf::lower_bound(sorted_ids->view(),
                                                cudf::table_view({needles->view()}),
                                                {cudf::order::ASCENDING},
                                                {cudf::null_order::AFTER},
                                                stream,
                                                temp_mr);
  std::vector<cudf::size_type> offsets(num_partitions);
  CUDF_CUDA_TRY(cudaMemcpyAsync(offsets.data(),
                                device_offsets->view().data<cudf::size_type>(),
                                num_partitions * sizeof(cudf::size_type),
                                cudaMemcpyDeviceToHost,
                                stream.value()));
  stream.synchronize();
  return {std::move(reordered), std::move(offsets)};
}

hash_partition_result unordered_hash_partition(cudf::table_view const& input,
                                               std::vector<cudf::size_type> const& key_indices,
                                               int num_partitions,
                                               rmm::cuda_stream_view stream,
                                               rmm::device_async_resource_ref mr)
{
  auto const ids = partition_ids(input, key_indices, num_partitions, stream);
  auto result    = cudf::partition(input, ids->view(), num_partitions, stream, mr);
  // cuDF may append the end offset, which the callers do not use.
  result.second.resize(num_partitions);
  return result;
}

}  // namespace sirius::op
