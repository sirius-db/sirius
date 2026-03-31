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

#include <cudf/column/column_factories.hpp>
#include <cudf/join/join.hpp>
#include <cudf/types.hpp>

#include <rmm/exec_policy.hpp>

#include <thrust/transform.h>

#include <op/scan/iceberg_equality_delete_mask.hpp>

namespace sirius::op::scan {

namespace {

/// Device functor: converts a hash-join build index to a keep/discard flag.
/// JoinNoMatch means "not in delete set" → keep (1).
/// Any other value means "matched a delete row" → discard (0).
struct anti_join_to_bool {
  __host__ __device__ uint8_t operator()(cudf::size_type idx) const
  {
    return idx == cudf::JoinNoMatch ? uint8_t{1} : uint8_t{0};
  }
};

}  // anonymous namespace

std::unique_ptr<cudf::column> make_anti_join_mask(
  rmm::device_uvector<cudf::size_type> const& build_indices,
  cudf::size_type n_rows,
  rmm::cuda_stream_view stream)
{
  auto bool_col = cudf::make_fixed_width_column(
    cudf::data_type{cudf::type_id::BOOL8}, n_rows, cudf::mask_state::UNALLOCATED, stream);

  thrust::transform(rmm::exec_policy_nosync(stream),
                    build_indices.begin(),
                    build_indices.begin() + n_rows,
                    bool_col->mutable_view().data<uint8_t>(),
                    anti_join_to_bool{});

  return bool_col;
}

}  // namespace sirius::op::scan
