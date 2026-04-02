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

#include "op/build_hash_table.hpp"

#include <numeric>
#include <stdexcept>

namespace sirius::op {

/// Create a device_uvector containing the sequence [0, 1, 2, ..., num_rows-1].
static std::unique_ptr<rmm::device_uvector<cudf::size_type>> make_sequential_indices(
  cudf::size_type num_rows, rmm::cuda_stream_view stream)
{
  auto indices  = std::make_unique<rmm::device_uvector<cudf::size_type>>(num_rows, stream);
  auto host_seq = std::vector<cudf::size_type>(num_rows);
  std::iota(host_seq.begin(), host_seq.end(), 0);
  cudaMemcpyAsync(indices->data(),
                  host_seq.data(),
                  num_rows * sizeof(cudf::size_type),
                  cudaMemcpyHostToDevice,
                  stream.value());
  return indices;
}

void build_hash_table::build(cudf::table_view build_keys,
                             bool unique_keys,
                             cudf::null_equality null_eq,
                             rmm::cuda_stream_view stream)
{
  reset();
  if (unique_keys) {
    _distinct = std::make_unique<cudf::distinct_hash_join>(build_keys, null_eq, 0.5, stream);
  } else {
    _generic = std::make_unique<cudf::hash_join>(build_keys, null_eq, stream);
  }
}

join_index_pair build_hash_table::inner_join(cudf::table_view probe_keys,
                                             rmm::cuda_stream_view stream) const
{
  if (_distinct) { return _distinct->inner_join(probe_keys, stream); }
  return _generic->inner_join(probe_keys, {}, stream);
}

join_index_pair build_hash_table::left_join(cudf::table_view probe_keys,
                                            rmm::cuda_stream_view stream) const
{
  if (_distinct) {
    auto build_indices = _distinct->left_join(probe_keys, stream);
    auto probe_indices =
      make_sequential_indices(static_cast<cudf::size_type>(build_indices->size()), stream);
    return {std::move(probe_indices), std::move(build_indices)};
  }
  return _generic->left_join(probe_keys, {}, stream);
}

join_index_pair build_hash_table::full_join(cudf::table_view probe_keys,
                                            rmm::cuda_stream_view stream) const
{
  if (!_generic) { throw std::runtime_error("full_join is not supported with distinct_hash_join"); }
  return _generic->full_join(probe_keys, {}, stream);
}

void build_hash_table::reset()
{
  _generic.reset();
  _distinct.reset();
}

bool build_hash_table::is_built() const { return _generic != nullptr || _distinct != nullptr; }

bool build_hash_table::is_distinct() const { return _distinct != nullptr; }

}  // namespace sirius::op
