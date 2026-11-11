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

#include "op/window/float_extremum.hpp"

#include <cudf/binaryop.hpp>
#include <cudf/copying.hpp>
#include <cudf/reduction.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/transform.hpp>
#include <cudf/unary.hpp>

#include <limits>

namespace sirius::op {

nan_free_argument make_nan_free_argument(cudf::column_view values,
                                         ::cuda::stream_ref stream,
                                         rmm::device_async_resource_ref mr)
{
  // is_nan gives false for NULL rows and has no NULLs itself.
  auto flags     = cudf::is_nan(values, stream, mr);
  auto const any = cudf::reduce(flags->view(),
                                *cudf::make_any_aggregation<cudf::reduce_aggregation>(),
                                cudf::data_type{cudf::type_id::BOOL8},
                                stream,
                                mr);
  if (!any->is_valid(stream) ||
      !static_cast<cudf::numeric_scalar<bool> const&>(*any).value(stream)) {
    return {};
  }
  auto int8_flags = cudf::cast(flags->view(), cudf::data_type{cudf::type_id::INT8}, stream, mr);
  flags.reset();
  return {std::move(int8_flags), cudf::column_nans_to_nulls(values, stream, mr)};
}

std::unique_ptr<cudf::column> restore_nan(std::unique_ptr<cudf::column> result,
                                          cudf::column_view any_nan,
                                          aggregate_id function,
                                          ::cuda::stream_ref stream,
                                          rmm::device_async_resource_ref mr)
{
  auto const boolean = cudf::data_type{cudf::type_id::BOOL8};
  auto replace       = cudf::cast(any_nan, boolean, stream, mr);
  if (function == aggregate_id::min) {
    auto empty = cudf::is_null(result->view(), stream, mr);
    replace    = cudf::binary_operation(
      replace->view(), empty->view(), cudf::binary_operator::LOGICAL_AND, boolean, stream, mr);
  }
  auto nan = cudf::make_numeric_scalar(result->type(), stream, mr);
  if (result->type().id() == cudf::type_id::FLOAT32) {
    static_cast<cudf::numeric_scalar<float>&>(*nan).set_value(
      std::numeric_limits<float>::quiet_NaN(), stream);
  } else {
    static_cast<cudf::numeric_scalar<double>&>(*nan).set_value(
      std::numeric_limits<double>::quiet_NaN(), stream);
  }
  nan->set_valid_async(true, stream);
  return cudf::copy_if_else(*nan, result->view(), replace->view(), stream, mr);
}

}  // namespace sirius::op
