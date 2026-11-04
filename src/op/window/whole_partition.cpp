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

#include "op/window/whole_partition.hpp"

#include "op/aggregate/aggregate_op_util.hpp"
#include "op/aggregate/wide_integer_sum.hpp"

#include <cudf/aggregation.hpp>
#include <cudf/binaryop.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/groupby.hpp>
#include <cudf/join/distinct_hash_join.hpp>
#include <cudf/reduction.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/transform.hpp>
#include <cudf/unary.hpp>
#include <cudf/utilities/traits.hpp>

#include <algorithm>
#include <limits>
#include <optional>
#include <stdexcept>
#include <unordered_set>

namespace sirius::op {
namespace {

// The type a `sum` over `input` accumulates in.
cudf::data_type accumulator_type(cudf::data_type input)
{
  using cudf::type_id;
  switch (input.id()) {
    case type_id::INT8:
    case type_id::INT16:
    case type_id::INT32:
    case type_id::INT64: return wide_integer_sum_type();
    case type_id::DECIMAL32:
    case type_id::DECIMAL64:
    case type_id::DECIMAL128: return cudf::data_type{type_id::DECIMAL128, input.scale()};
    case type_id::FLOAT32:
    case type_id::FLOAT64: return cudf::data_type{type_id::FLOAT64};
    default: throw std::invalid_argument("Unsupported whole-partition window argument type");
  }
}

// An 8 to 32 bit integer has a magnitude of at most 2^31 and a column holds fewer than 2^31 rows,
// so the sum of a partition fits INT64.
bool is_narrow_integer(cudf::data_type type)
{
  return type.id() == cudf::type_id::INT8 || type.id() == cudf::type_id::INT16 ||
         type.id() == cudf::type_id::INT32;
}

// Throws unless `type` is supported as a partition column.
void validate_key(cudf::data_type type)
{
  using cudf::type_id;
  switch (type.id()) {
    case type_id::BOOL8:
    case type_id::INT8:
    case type_id::INT16:
    case type_id::INT32:
    case type_id::INT64:
    case type_id::TIMESTAMP_DAYS:
    case type_id::TIMESTAMP_SECONDS:
    case type_id::TIMESTAMP_MILLISECONDS:
    case type_id::TIMESTAMP_MICROSECONDS:
    case type_id::TIMESTAMP_NANOSECONDS:
    case type_id::DECIMAL32:
    case type_id::DECIMAL64:
    case type_id::DECIMAL128:
    case type_id::STRING: return;
    default: throw std::invalid_argument("Unsupported whole-partition window key type");
  }
}

// Unwraps the SUM_OVERFLOW struct into its sum, using the struct's validity, and throws if the
// overflow flag is set.
std::unique_ptr<cudf::column> checked_sum(std::unique_ptr<cudf::column> result,
                                          ::cuda::stream_ref stream,
                                          rmm::device_async_resource_ref mr)
{
  if (result->type().id() != cudf::type_id::STRUCT) { return result; }
  auto child = result->child(1).view();
  cudf::column_view flags(
    child.type(), child.size(), child.head(), result->view().null_mask(), result->null_count());
  auto overflow = cudf::reduce(flags,
                               *cudf::make_any_aggregation<cudf::reduce_aggregation>(),
                               cudf::data_type{cudf::type_id::BOOL8},
                               stream,
                               mr);
  if (overflow->is_valid(stream) &&
      static_cast<cudf::numeric_scalar<bool> const&>(*overflow).value(stream)) {
    throw std::overflow_error("Whole-partition window sum exceeds its 128-bit accumulator");
  }
  auto const nulls = result->null_count();
  auto contents    = result->release();
  auto sum         = std::move(contents.children[0]);
  if (nulls) { sum->set_null_mask(std::move(*contents.null_mask), nulls); }
  return sum;
}

// Whether `function` is computed from a per group sum.
bool uses_sum(aggregate_id function)
{
  return function == aggregate_id::sum || function == aggregate_id::avg;
}

// The cuDF aggregation for `count`, `count_star`, `min` or `max`.
template <typename Base>
std::unique_ptr<Base> simple_aggregation(aggregate_id function)
{
  switch (function) {
    case aggregate_id::count: return cudf::make_count_aggregation<Base>();
    case aggregate_id::count_star:
      return cudf::make_count_aggregation<Base>(cudf::null_policy::INCLUDE);
    case aggregate_id::min: return cudf::make_min_aggregation<Base>();
    case aggregate_id::max: return cudf::make_max_aggregation<Base>();
    default: throw std::invalid_argument("Unsupported whole-partition window function");
  }
}

// The type of the result column of `agg` over an argument of type `input`.
cudf::data_type result_type(whole_partition_aggregate const& agg, cudf::data_type input)
{
  switch (agg.function) {
    case aggregate_id::sum: return accumulator_type(input);
    case aggregate_id::avg: return cudf::data_type{cudf::type_id::FLOAT64};
    case aggregate_id::count:
    case aggregate_id::count_star: return cudf::data_type{cudf::type_id::INT64};
    case aggregate_id::min:
    case aggregate_id::max: return input;
    default: throw std::invalid_argument("Unsupported whole-partition window function");
  }
}

// A SUM_OVERFLOW aggregation for DECIMAL128 values and a plain SUM for other types.
template <typename Base>
std::unique_ptr<Base> sum_aggregation(cudf::data_type type)
{
  if (type.id() == cudf::type_id::DECIMAL128) {
    return cudf::make_sum_overflow_aggregation<Base>();
  }
  return cudf::make_sum_aggregation<Base>();
}

// Reduces `values` to a column of one row.
std::unique_ptr<cudf::column> reduce_to_column(cudf::column_view values,
                                               cudf::reduce_aggregation const& agg,
                                               cudf::data_type output_type,
                                               ::cuda::stream_ref stream,
                                               rmm::device_async_resource_ref mr)
{
  auto scalar = cudf::reduce(values, agg, output_type, stream, mr);
  return cudf::make_column_from_scalar(*scalar, 1, stream, mr);
}

// The sum of `values` as a column of one row. A DECIMAL128 sum is a SUM_OVERFLOW struct.
std::unique_ptr<cudf::column> reduce_sum(cudf::column_view values,
                                         ::cuda::stream_ref stream,
                                         rmm::device_async_resource_ref mr)
{
  auto const type        = values.type();
  auto const output_type = type.id() == cudf::type_id::DECIMAL128
                             ? cudf::data_type{cudf::type_id::STRUCT}
                           : is_narrow_integer(type) ? cudf::data_type{cudf::type_id::INT64}
                                                     : type;
  return reduce_to_column(
    values, *sum_aggregation<cudf::reduce_aggregation>(type), output_type, stream, mr);
}

// The number of non-NULL values of `values` as a column of one row.
std::unique_ptr<cudf::column> reduce_count(cudf::column_view values,
                                           ::cuda::stream_ref stream,
                                           rmm::device_async_resource_ref mr)
{
  return reduce_to_column(values,
                          *cudf::make_count_aggregation<cudf::reduce_aggregation>(),
                          cudf::data_type{cudf::type_id::INT64},
                          stream,
                          mr);
}

// The `sum` and `avg` aggregates over one argument column, which share one per group sum.
struct argument_state {
  cudf::size_type column;                 // Index of the argument column in the input.
  bool needs_avg;                         // Whether any aggregate over the column is an `avg`.
  std::unique_ptr<cudf::column> widened;  // The argument cast to its accumulator type, if needed.
  std::unique_ptr<cudf::column> sum;      // The per group sum, once computed.
  std::unique_ptr<cudf::column> avg;      // The per group average, when `needs_avg`.
};

// The argument columns whose sum cannot overflow the column's own type in `input`, and so is
// computed in that type.
std::unordered_set<int> sums_in_place(cudf::table_view input,
                                      std::vector<argument_state> const& arguments,
                                      ::cuda::stream_ref stream,
                                      rmm::device_async_resource_ref mr)
{
  std::vector<int> candidates;
  for (auto const& arg : arguments) {
    if (widened_sum_type(input.column(arg.column).type())) { candidates.push_back(arg.column); }
  }
  auto const needs_widening = sums_needing_widening(input, candidates, stream, mr);
  std::unordered_set<int> ret;
  for (auto const column : candidates) {
    if (!needs_widening.contains(column)) { ret.insert(column); }
  }
  return ret;
}

// sum_no_overflow is a SUM, and any id the evaluator does not compute is rejected.
std::vector<whole_partition_aggregate> normalize(std::vector<whole_partition_aggregate> aggregates)
{
  for (auto& aggregate : aggregates) {
    switch (aggregate.function) {
      case aggregate_id::sum_no_overflow: aggregate.function = aggregate_id::sum; break;
      case aggregate_id::sum:
      case aggregate_id::avg:
      case aggregate_id::count:
      case aggregate_id::count_star:
      case aggregate_id::min:
      case aggregate_id::max: break;
      default: throw std::invalid_argument("Unsupported whole-partition window function");
    }
  }
  return aggregates;
}

}  // namespace

std::unique_ptr<cudf::table> evaluate_whole_partition_window(
  cudf::table_view input,
  std::vector<cudf::size_type> const& partition_columns,
  std::vector<whole_partition_aggregate> const& requested,
  ::cuda::stream_ref stream,
  rmm::device_async_resource_ref mr)
{
  // The evaluation has these steps:
  //   1. Validate the request and group the `sum` and `avg` aggregates by argument column, so
  //      aggregates over one column share one sum.
  //   2. Build the group by requests: one per shared sum, then one per `count`, `count_star`,
  //      `min` and `max`. A floating point `min` or `max` also gets counts that restore NaN.
  //   3. Run the requests as a group by, or as reductions without partition columns.
  //   4. Turn the group results of each argument into its sum and average, checking overflow.
  //   5. Collect one column per aggregate in request order, restoring NaN in `min` and `max`.
  //   6. Give every input row the value of its group.

  // 1. Validate the request and group the `sum` and `avg` aggregates by argument column.
  auto const aggregates = normalize(requested);
  if (aggregates.empty()) { throw std::invalid_argument("Window requires at least one aggregate"); }
  auto check_index = [&](cudf::size_type index) {
    if (index < 0 || index >= input.num_columns()) {
      throw std::invalid_argument("Whole-partition window column is outside the input schema");
    }
  };
  for (auto index : partition_columns) {
    check_index(index);
    validate_key(input.column(index).type());
  }
  std::vector<argument_state> arguments;
  std::vector<std::size_t> argument_indices;
  for (auto const& agg : aggregates) {
    check_index(agg.argument_column);
    (void)result_type(agg, input.column(agg.argument_column).type());
    if (!uses_sum(agg.function)) {
      argument_indices.push_back(0);  // Assigned after the shared SUM/AVG requests below.
      continue;
    }
    auto found = std::find_if(arguments.begin(), arguments.end(), [&](auto const& arg) {
      return arg.column == agg.argument_column;
    });
    if (found == arguments.end()) {
      argument_indices.push_back(arguments.size());
      arguments.push_back({agg.argument_column, agg.function == aggregate_id::avg, {}, {}, {}});
    } else {
      argument_indices.push_back(static_cast<std::size_t>(found - arguments.begin()));
      found->needs_avg |= agg.function == aggregate_id::avg;
    }
  }
  if (input.num_rows() == 0) {
    std::vector<std::unique_ptr<cudf::column>> empty;
    for (auto const& agg : aggregates) {
      auto type = result_type(agg, input.column(agg.argument_column).type());
      empty.push_back(cudf::make_empty_column(type));
    }
    return std::make_unique<cudf::table>(std::move(empty));
  }

  // 2. Build the group by requests. The shared sums come first, each with a count when an `avg`
  // needs one.
  std::vector<cudf::groupby::aggregation_request> requests;
  auto const in_place = sums_in_place(input, arguments, stream, mr);
  for (auto& arg : arguments) {
    auto values = input.column(arg.column);
    auto type   = accumulator_type(values.type());
    if (is_narrow_integer(values.type())) {
      type = cudf::data_type{cudf::type_id::INT64};
    } else if (in_place.contains(arg.column)) {
      type = values.type();
    }
    if (values.type() != type && !is_narrow_integer(values.type())) {
      arg.widened = cudf::cast(values, type, stream, mr);
      values      = arg.widened->view();
    }
    cudf::groupby::aggregation_request request;
    request.values = values;
    request.aggregations.push_back(sum_aggregation<cudf::groupby_aggregation>(type));
    if (arg.needs_avg) {
      request.aggregations.push_back(cudf::make_count_aggregation<cudf::groupby_aggregation>());
    }
    requests.push_back(std::move(request));
  }

  std::vector<whole_partition_aggregate> simple_aggregates;
  std::vector<std::unique_ptr<cudf::column>> finite_arguments;
  std::vector<cudf::size_type> finite_argument_columns;
  std::vector<std::optional<std::size_t>> nan_count_indices(aggregates.size());
  for (std::size_t i = 0; i < aggregates.size(); ++i) {
    auto const& agg = aggregates[i];
    if (uses_sum(agg.function)) { continue; }
    argument_indices[i] = requests.size();
    simple_aggregates.push_back(agg);
    cudf::groupby::aggregation_request request;
    request.values = input.column(agg.argument_column);
    auto const float_extrema =
      (agg.function == aggregate_id::min || agg.function == aggregate_id::max) &&
      cudf::is_floating_point(request.values.type());
    if (float_extrema) {
      // NaN orders above every other value. Floating point extrema ignore NaN, and the NaN
      // results are restored afterwards from the counts. Aggregates over one column share
      // one NaN free copy.
      auto const existing = std::find(
        finite_argument_columns.begin(), finite_argument_columns.end(), agg.argument_column);
      if (existing == finite_argument_columns.end()) {
        finite_arguments.push_back(cudf::column_nans_to_nulls(request.values, stream, mr));
        finite_argument_columns.push_back(agg.argument_column);
        request.values = finite_arguments.back()->view();
      } else {
        request.values = finite_arguments[existing - finite_argument_columns.begin()]->view();
      }
    }
    request.aggregations.push_back(simple_aggregation<cudf::groupby_aggregation>(agg.function));
    if (float_extrema) {
      request.aggregations.push_back(cudf::make_count_aggregation<cudf::groupby_aggregation>());
    }
    requests.push_back(std::move(request));
    if (float_extrema) {
      nan_count_indices[i] = requests.size();
      cudf::groupby::aggregation_request original_count;
      original_count.values = input.column(agg.argument_column);
      original_count.aggregations.push_back(
        cudf::make_count_aggregation<cudf::groupby_aggregation>());
      requests.push_back(std::move(original_count));
      simple_aggregates.push_back({aggregate_id::count, agg.argument_column});
    }
  }

  // 3. Run the requests. The group by also gives the group keys.
  std::unique_ptr<cudf::table> grouped_keys;
  std::vector<cudf::groupby::aggregation_result> results;
  if (!partition_columns.empty()) {
    cudf::groupby::groupby groupby(input.select(partition_columns), cudf::null_policy::INCLUDE);
    auto grouped = groupby.aggregate(requests, stream, mr);
    grouped_keys = std::move(grouped.first);
    results      = std::move(grouped.second);
  } else {
    for (std::size_t i = 0; i < requests.size(); ++i) {
      auto const& request = requests[i];
      cudf::groupby::aggregation_result result;
      if (i >= arguments.size()) {
        auto const& agg = simple_aggregates[i - arguments.size()];
        result.results.push_back(
          reduce_to_column(request.values,
                           *simple_aggregation<cudf::reduce_aggregation>(agg.function),
                           result_type(agg, request.values.type()),
                           stream,
                           mr));
        if (request.aggregations.size() > 1) {
          result.results.push_back(reduce_count(request.values, stream, mr));
        }
      } else {
        result.results.push_back(reduce_sum(request.values, stream, mr));
        if (arguments[i].needs_avg) {
          result.results.push_back(reduce_count(request.values, stream, mr));
        }
      }
      results.push_back(std::move(result));
    }
  }
  requests.clear();
  finite_arguments.clear();

  // 4. Turn the group results of each argument into its sum and average.
  auto const f64 = cudf::data_type{cudf::type_id::FLOAT64};
  for (std::size_t i = 0; i < arguments.size(); ++i) {
    auto& arg = arguments[i];
    arg.widened.reset();
    arg.sum = checked_sum(std::move(results[i].results[0]), stream, mr);
    if (auto const target = accumulator_type(input.column(arg.column).type());
        arg.sum->type() != target) {
      arg.sum = cudf::cast(arg.sum->view(), target, stream, mr);
    }
    // A requested decimal SUM must fit DECIMAL(38, scale). The internal sum of AVG has no such
    // bound.
    auto const decimal_sum =
      cudf::is_fixed_point(input.column(arg.column).type()) &&
      std::any_of(aggregates.begin(), aggregates.end(), [&](auto const& aggregate) {
        return aggregate.argument_column == arg.column && aggregate.function == aggregate_id::sum;
      });
    if (decimal_sum) {
      constexpr auto limit = int128_pow10(38);
      auto bounds          = cudf::minmax(arg.sum->view(), stream, mr);
      for (auto const* bound : {bounds.first.get(), bounds.second.get()}) {
        if (bound->is_valid(stream)) {
          auto const value =
            static_cast<cudf::fixed_point_scalar<numeric::decimal128> const&>(*bound).value(stream);
          if (value <= -limit || value >= limit) {
            throw std::overflow_error("Whole-partition window sum exceeds DECIMAL(38)");
          }
        }
      }
    }
    if (arg.needs_avg) {
      auto sum   = cudf::cast(arg.sum->view(), f64, stream, mr);
      auto count = cudf::cast(results[i].results[1]->view(), f64, stream, mr);
      arg.avg    = cudf::binary_operation(
        sum->view(), count->view(), cudf::binary_operator::DIV, f64, stream, mr);
    }
  }
  // 5. Collect one column per aggregate in request order, with one value per group.
  std::vector<cudf::column_view> output;
  for (std::size_t i = 0; i < aggregates.size(); ++i) {
    if (!uses_sum(aggregates[i].function)) {
      auto& value = results[argument_indices[i]].results[0];
      // NaN is the result of a `max` of a group with any NaN, and of a `min` of a group whose
      // non-NULL values are all NaN.
      if (nan_count_indices[i]) {
        auto const& original_count = results[*nan_count_indices[i]].results[0];
        auto const& finite_count   = results[argument_indices[i]].results[1];
        auto replace               = cudf::binary_operation(original_count->view(),
                                              finite_count->view(),
                                              cudf::binary_operator::GREATER,
                                              cudf::data_type{cudf::type_id::BOOL8},
                                              stream,
                                              mr);
        if (aggregates[i].function == aggregate_id::min) {
          auto all_nan = cudf::is_null(value->view(), stream, mr);
          replace      = cudf::binary_operation(replace->view(),
                                           all_nan->view(),
                                           cudf::binary_operator::LOGICAL_AND,
                                           cudf::data_type{cudf::type_id::BOOL8},
                                           stream,
                                           mr);
        }
        auto nan = cudf::make_numeric_scalar(value->type(), stream, mr);
        if (value->type().id() == cudf::type_id::FLOAT32) {
          static_cast<cudf::numeric_scalar<float>&>(*nan).set_value(
            std::numeric_limits<float>::quiet_NaN(), stream);
        } else {
          static_cast<cudf::numeric_scalar<double>&>(*nan).set_value(
            std::numeric_limits<double>::quiet_NaN(), stream);
        }
        nan->set_valid_async(true, stream);
        value = cudf::copy_if_else(*nan, value->view(), replace->view(), stream, mr);
      }
      auto const type =
        result_type(aggregates[i], input.column(aggregates[i].argument_column).type());
      if (value->type() != type) { value = cudf::cast(value->view(), type, stream, mr); }
      output.push_back(value->view());
      continue;
    }
    auto const& arg = arguments[argument_indices[i]];
    output.push_back(aggregates[i].function == aggregate_id::sum ? arg.sum->view()
                                                                 : arg.avg->view());
  }
  // 6. Give every input row the value of its group.
  if (partition_columns.empty()) {
    std::vector<std::unique_ptr<cudf::column>> expanded;
    for (auto const& col : output) {
      auto scalar = cudf::get_element(col, 0, stream, mr);
      expanded.push_back(cudf::make_column_from_scalar(*scalar, input.num_rows(), stream, mr));
    }
    return std::make_unique<cudf::table>(std::move(expanded));
  }

  // Probe the group keys, including NULL keys, with nulls equal. distinct_hash_join returns one
  // build index per input row, in input order.
  cudf::distinct_hash_join lookup(grouped_keys->view(), cudf::null_equality::EQUAL, 0.5, stream);
  auto indices = lookup.left_join(input.select(partition_columns), stream, mr);
  cudf::column_view map(
    cudf::data_type{cudf::type_id::INT32}, input.num_rows(), indices->data(), nullptr, 0);
  return cudf::gather(
    cudf::table_view(output), map, cudf::out_of_bounds_policy::DONT_CHECK, stream, mr);
}

}  // namespace sirius::op
