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

#include "vss/cudf_raft_interop.hpp"

#include <cudf/lists/lists_column_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/error.hpp>

#include <raft/core/device_mdspan.hpp>

namespace sirius::vss {

dataset_matrix_view list_column_as_dataset_view(cudf::column_view const& list_col, int64_t dim)
{
  CUDF_EXPECTS(list_col.type().id() == cudf::type_id::LIST,
               "VSS dataset column must be a LIST (fixed-size array) column");
  CUDF_EXPECTS(dim > 0, "VSS dataset dimensionality must be positive");

  // No slicing: a non-zero parent offset means row 0 does not start at the
  // beginning of the values buffer, so a flat reinterpretation would be wrong.
  CUDF_EXPECTS(list_col.offset() == 0,
               "VSS dataset column must not be sliced (non-zero offset unsupported)");
  // RAFT reads the buffer as a raw blob and would treat null slots as real
  // values, so we reject nulls outright for now.
  CUDF_EXPECTS(list_col.null_count() == 0, "VSS dataset column must not contain null rows");

  cudf::lists_column_view const lists{list_col};
  cudf::column_view const values = lists.child();

  CUDF_EXPECTS(values.type().id() == cudf::type_id::FLOAT32, "VSS dataset values must be FLOAT32");
  CUDF_EXPECTS(values.null_count() == 0, "VSS dataset values must not contain nulls");

  auto const n_rows = static_cast<int64_t>(list_col.size());
  CUDF_EXPECTS(static_cast<int64_t>(values.size()) == n_rows * dim,
               "VSS dataset values size does not match n_rows * dim (non-uniform lists?)");

  // Zero-copy: borrow the values buffer as a [n_rows, dim] row-major matrix.
  // column_view::data<float>() is already offset-adjusted for the child.
  return raft::make_device_matrix_view<const float, int64_t, raft::row_major>(
    values.data<float>(), n_rows, dim);
}

}  // namespace sirius::vss
