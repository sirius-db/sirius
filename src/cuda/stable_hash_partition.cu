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

// clang-format off
#include <op/partition/stable_hash_partition.hpp>

#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/copying.hpp>
#include <cudf/hashing.hpp>
#include <cudf/utilities/error.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/binary_search.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/transform.h>

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>
// clang-format on

namespace sirius::op {

std::pair<std::unique_ptr<cudf::table>, std::vector<cudf::size_type>> stable_hash_partition(
  cudf::table_view const& input,
  std::vector<cudf::size_type> const& key_indices,
  int num_partitions,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(num_partitions > 0, "stable_hash_partition: num_partitions must be positive");
  cudf::size_type const num_rows = input.num_rows();

  auto const hashes     = cudf::hashing::murmurhash3_x86_32(input.select(key_indices),
                                                        cudf::DEFAULT_HASH_SEED,
                                                        stream,
                                                        cudf::get_current_device_resource_ref());
  auto const* hash_data = hashes->view().data<uint32_t>();

  // Partition number per row, then a stable radix sort of (partition, row index) pairs: rows of
  // one partition keep their input order.
  rmm::device_uvector<uint32_t> partition_ids(num_rows, stream);
  rmm::device_uvector<cudf::size_type> row_order(num_rows, stream);
  auto const policy              = rmm::exec_policy_nosync(stream);
  uint32_t const partition_count = static_cast<uint32_t>(num_partitions);
  thrust::transform(policy,
                    hash_data,
                    hash_data + num_rows,
                    partition_ids.begin(),
                    [partition_count] __device__(uint32_t hash) { return hash % partition_count; });
  thrust::sequence(policy, row_order.begin(), row_order.end(), 0);
  thrust::stable_sort_by_key(policy, partition_ids.begin(), partition_ids.end(), row_order.begin());

  // Partition start offsets: the first row of each partition number in the sorted ids.
  rmm::device_uvector<cudf::size_type> device_offsets(num_partitions, stream);
  thrust::lower_bound(policy,
                      partition_ids.begin(),
                      partition_ids.end(),
                      thrust::counting_iterator<uint32_t>(0),
                      thrust::counting_iterator<uint32_t>(partition_count),
                      device_offsets.begin());
  std::vector<cudf::size_type> offsets(num_partitions);
  CUDF_CUDA_TRY(cudaMemcpyAsync(offsets.data(),
                                device_offsets.data(),
                                num_partitions * sizeof(cudf::size_type),
                                cudaMemcpyDeviceToHost,
                                stream.value()));

  cudf::column_view const gather_map(
    cudf::data_type{cudf::type_id::INT32}, num_rows, row_order.data(), nullptr, 0);
  auto reordered =
    cudf::gather(input, gather_map, cudf::out_of_bounds_policy::DONT_CHECK, stream, mr);
  stream.synchronize();
  return {std::move(reordered), std::move(offsets)};
}

}  // namespace sirius::op
