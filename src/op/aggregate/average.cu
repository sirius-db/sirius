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

#include "op/aggregate/average.hpp"
#include "sirius/exception.hpp"

#include <cudf/binaryop.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/unary.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/exec_policy.hpp>

#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform.h>

#include <cmath>
#include <cstdint>

namespace sirius::op {

namespace {

/// Row i's unscaled sum over its count times @p divisor_scale (10^scale), as DuckDB's AVG does.
template <typename Rep>
struct unscaled_average {
  cudf::column_device_view sum;
  cudf::column_device_view count;
  double divisor_scale;

  __device__ double operator()(cudf::size_type i) const
  {
    return static_cast<double>(sum.data<Rep>()[i]) /
           (static_cast<double>(count.element<int64_t>(i)) * divisor_scale);
  }
};

template <typename Rep>
void divide_unscaled(cudf::column_view const& sum,
                     cudf::column_view const& count,
                     cudf::mutable_column_view out,
                     ::cuda::stream_ref stream)
{
  auto const sum_device   = cudf::column_device_view::create(sum, stream);
  auto const count_device = cudf::column_device_view::create(count, stream);
  // cuDF scales are negated: DECIMAL(p, s) has scale -s.
  auto const divisor_scale = std::pow(10.0, -sum.type().scale());
  thrust::transform(rmm::exec_policy_nosync(stream),
                    thrust::make_counting_iterator<cudf::size_type>(0),
                    thrust::make_counting_iterator<cudf::size_type>(sum.size()),
                    out.data<double>(),
                    unscaled_average<Rep>{*sum_device, *count_device, divisor_scale});
}

}  // namespace

std::unique_ptr<cudf::column> average_as_double(cudf::column_view const& sum,
                                                cudf::column_view const& count,
                                                ::cuda::stream_ref stream,
                                                rmm::device_async_resource_ref mr)
{
  auto const f64 = cudf::data_type{cudf::type_id::FLOAT64};
  if (!cudf::is_fixed_point(sum.type())) {
    auto const sum_f64   = cudf::cast(sum, f64, stream, mr);
    auto const count_f64 = cudf::cast(count, f64, stream, mr);
    return cudf::binary_operation(
      sum_f64->view(), count_f64->view(), cudf::binary_operator::DIV, f64, stream, mr);
  }
  if (sum.type().scale() > 0) {
    throw sirius::internal_exception("average_as_double: unexpected positive DECIMAL scale {}",
                                     sum.type().scale());
  }

  std::unique_ptr<cudf::column> count_i64;
  auto count_view = count;
  if (count.type().id() != cudf::type_id::INT64) {
    count_i64  = cudf::cast(count, cudf::data_type{cudf::type_id::INT64}, stream, mr);
    count_view = count_i64->view();
  }

  auto [null_mask, null_count] = cudf::bitmask_and(cudf::table_view{{sum, count_view}}, stream, mr);
  auto result =
    cudf::make_numeric_column(f64, sum.size(), std::move(null_mask), null_count, stream, mr);
  if (sum.is_empty()) { return result; }

  switch (sum.type().id()) {
    case cudf::type_id::DECIMAL32:
      divide_unscaled<int32_t>(sum, count_view, result->mutable_view(), stream);
      break;
    case cudf::type_id::DECIMAL64:
      divide_unscaled<int64_t>(sum, count_view, result->mutable_view(), stream);
      break;
    case cudf::type_id::DECIMAL128:
      divide_unscaled<__int128_t>(sum, count_view, result->mutable_view(), stream);
      break;
    default:
      throw sirius::internal_exception("average_as_double: unexpected sum type {}",
                                       cudf::type_to_name(sum.type()));
  }
  return result;
}

}  // namespace sirius::op
