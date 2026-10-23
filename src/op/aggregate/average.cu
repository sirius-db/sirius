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

/// 10^@p scale rounded to DOUBLE the way DuckDB's AVG bind data rounds it: the HUGEINT's
/// low and high 64-bit words converted separately and added.
double duckdb_power_of_ten(int scale)
{
  unsigned __int128 power = 1;
  for (int i = 0; i < scale; ++i) {
    power *= 10;
  }
  auto const lower = static_cast<uint64_t>(power);
  auto const upper = static_cast<uint64_t>(power >> 64);
  return static_cast<double>(lower) + static_cast<double>(upper) * 0x1p64;
}

/// @p value minus @p high, where @p high is DOUBLE(@p value), as a DOUBLE. @p high can round to
/// +-2^(bits - 1); +2^(bits - 1) has no Int value, so either limit is subtracted in two halves
/// instead of being converted back.
template <typename Int>
__device__ double low_part(Int value, double high)
{
  constexpr int bits = static_cast<int>(sizeof(Int) * 8);
  constexpr Int half = Int{1} << (bits - 2);
  double const limit = static_cast<double>(half) * 2.0;
  if (high >= limit) { return static_cast<double>((value - half) - half); }
  if (high <= -limit) { return static_cast<double>((value + half) + half); }
  return static_cast<double>(value - static_cast<Int>(high));
}

/// @p sum / (@p divisor_hi + @p divisor_lo) rounded once to DOUBLE. The sum is split into two
/// DOUBLEs and the quotient corrected by its remainder, so it carries about 106 bits until the
/// final rounding.
__device__ double divide_double_double(__int128_t sum, double divisor_hi, double divisor_lo)
{
  double const sum_hi        = static_cast<double>(sum);
  double const sum_lo        = low_part(sum, sum_hi);
  double const quotient      = sum_hi / divisor_hi;
  double const product       = quotient * divisor_hi;
  double const product_error = fma(quotient, divisor_hi, -product);
  // sum_hi - product is exact: the two are within a factor of 2.
  double const remainder = ((sum_hi - product) - product_error) + sum_lo - quotient * divisor_lo;
  return quotient + remainder / divisor_hi;
}

/// Row i's unscaled sum over its count times @p power_of_ten, as DuckDB's AVG computes it.
template <typename Rep>
struct unscaled_average {
  cudf::column_device_view sum;
  cudf::column_device_view count;
  double power_of_ten;

  __device__ double operator()(cudf::size_type i) const
  {
    // count * power_of_ten as two DOUBLEs. The count is split too, since an INT64 count above
    // 2^53 does not fit one DOUBLE.
    auto const count_value  = count.element<int64_t>(i);
    double const count_hi   = static_cast<double>(count_value);
    double const count_lo   = low_part(count_value, count_hi);
    double const product    = count_hi * power_of_ten;
    double const error      = fma(count_hi, power_of_ten, -product) + count_lo * power_of_ten;
    double const divisor_hi = product + error;
    double const divisor_lo = error - (divisor_hi - product);
    return divide_double_double(
      static_cast<__int128_t>(sum.data<Rep>()[i]), divisor_hi, divisor_lo);
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
  auto const power_of_ten = duckdb_power_of_ten(-sum.type().scale());
  thrust::transform(rmm::exec_policy_nosync(stream),
                    thrust::make_counting_iterator<cudf::size_type>(0),
                    thrust::make_counting_iterator<cudf::size_type>(sum.size()),
                    out.data<double>(),
                    unscaled_average<Rep>{*sum_device, *count_device, power_of_ten});
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
