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
#include "op/window/aggregate_types.hpp"

#include <cudf/aggregation.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/groupby.hpp>
#include <cudf/join/distinct_hash_join.hpp>
#include <cudf/reduction.hpp>
#include <cudf/scalar/scalar.hpp>

#include <algorithm>
#include <optional>
#include <stdexcept>
#include <utility>

namespace sirius::op {
namespace {

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

using plan_type = whole_partition_plan;

// The values a `sum` over an argument of `type` reads. An 8 to 32 bit integer is summed in INT64
// as it is, a DECIMAL128 with an overflow check, and floating point in FLOAT64.
std::size_t sum_values(detail::window_aggregate_plan_builder& builder,
                       cudf::size_type input,
                       cudf::data_type type)
{
  using preparation = plan_type::preparation;
  auto const sum    = window_sum_type(type);
  if (is_narrow_integer(type)) { return builder.values(input, preparation::none, type); }
  if (widened_sum_type(type)) { return builder.values(input, preparation::widen_if_needed, sum); }
  return builder.values(input, preparation::cast, sum);
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

// The cuDF aggregation of @p kind over values of @p type. A DECIMAL128 sum checks for overflow.
template <typename Base>
std::unique_ptr<Base> make_aggregation(plan_type::aggregation kind, cudf::data_type type)
{
  using aggregation = plan_type::aggregation;
  switch (kind) {
    case aggregation::sum:
      return type.id() == cudf::type_id::DECIMAL128 ? cudf::make_sum_overflow_aggregation<Base>()
                                                    : cudf::make_sum_aggregation<Base>();
    case aggregation::count_valid: return cudf::make_count_aggregation<Base>();
    case aggregation::count_all:
      return cudf::make_count_aggregation<Base>(cudf::null_policy::INCLUDE);
    case aggregation::min: return cudf::make_min_aggregation<Base>();
    case aggregation::max: return cudf::make_max_aggregation<Base>();
  }
  throw std::logic_error("Unknown whole-partition window aggregation");
}

// The result of @p kind over all of @p values as a column of one row.
std::unique_ptr<cudf::column> reduce_to_column(cudf::column_view values,
                                               plan_type::aggregation kind,
                                               ::cuda::stream_ref stream,
                                               rmm::device_async_resource_ref mr)
{
  using aggregation = plan_type::aggregation;
  auto const type   = values.type();
  auto output_type  = type;
  if (kind == aggregation::count_valid || kind == aggregation::count_all) {
    output_type = cudf::data_type{cudf::type_id::INT64};
  } else if (kind == aggregation::sum && type.id() == cudf::type_id::DECIMAL128) {
    output_type = cudf::data_type{cudf::type_id::STRUCT};
  } else if (kind == aggregation::sum && is_narrow_integer(type)) {
    output_type = cudf::data_type{cudf::type_id::INT64};
  }
  auto scalar = cudf::reduce(
    values, *make_aggregation<cudf::reduce_aggregation>(kind, type), output_type, stream, mr);
  return cudf::make_column_from_scalar(*scalar, 1, stream, mr);
}

}  // namespace

whole_partition_plan plan_whole_partition_window(
  std::vector<cudf::data_type> const& input_types,
  std::vector<cudf::size_type> const& partition_columns,
  std::vector<window_aggregate> const& aggregates)
{
  if (aggregates.empty()) { throw std::invalid_argument("Window requires at least one aggregate"); }
  for (auto index : partition_columns) {
    detail::check_window_column(input_types, index);
    validate_key(input_types[index]);
  }
  plan_type plan;
  plan.group_columns = partition_columns;
  detail::window_aggregate_plan_builder builder(input_types, plan);
  // All requests run in one group by.
  std::vector<std::size_t> requests;
  for (auto const& requested : aggregates) {
    auto const function = normalize_window_aggregate(requested.function);
    auto const input    = requested.argument_column;
    detail::check_window_column(input_types, input);
    auto const uses_sum = function == aggregate_id::sum || function == aggregate_id::avg;
    builder.add_output(
      requests, function, input, uses_sum ? sum_values(builder, input, input_types[input]) : 0);
  }
  return plan;
}

std::vector<cudf::size_type> logical_type_columns(whole_partition_plan const& plan)
{
  auto columns = plan.group_columns;
  for (auto column : plan.argument_columns) {
    if (std::find(columns.begin(), columns.end(), column) == columns.end()) {
      columns.push_back(column);
    }
  }
  return columns;
}

std::unique_ptr<cudf::table> evaluate_whole_partition_window(whole_partition_plan const& plan,
                                                             cudf::table_view input,
                                                             ::cuda::stream_ref stream,
                                                             rmm::device_async_resource_ref mr)
{
  detail::check_window_input(plan, plan.group_columns, input);
  if (input.num_rows() == 0) {
    std::vector<std::unique_ptr<cudf::column>> empty;
    for (auto const& output : plan.outputs) {
      empty.push_back(cudf::make_empty_column(output.type));
    }
    return std::make_unique<cudf::table>(std::move(empty));
  }

  // 1. Prepare the value columns.
  auto values = detail::prepare_window_values(plan, input, stream, mr);

  // 2. Run the requests, giving one row per group: a group by, or reductions without partition
  // columns. The group by also gives the group keys.
  std::vector<std::unique_ptr<cudf::column>> results(plan.requests.size());
  std::unique_ptr<cudf::table> grouped_keys;
  if (plan.group_columns.empty()) {
    for (std::size_t r = 0; r < plan.requests.size(); ++r) {
      if (!detail::runs(plan, values, r)) { continue; }
      auto const& request = plan.requests[r];
      results[r] = reduce_to_column(*values.views[request.values], request.kind, stream, mr);
    }
  } else {
    // One cuDF request per value column, holding the aggregations over it.
    std::vector<cudf::groupby::aggregation_request> requests;
    std::vector<std::optional<std::size_t>> request_of_values(plan.values.size());
    std::vector<std::optional<std::pair<std::size_t, std::size_t>>> positions(plan.requests.size());
    for (std::size_t r = 0; r < plan.requests.size(); ++r) {
      if (!detail::runs(plan, values, r)) { continue; }
      auto const& request = plan.requests[r];
      auto const view     = *values.views[request.values];
      auto& index         = request_of_values[request.values];
      if (!index) {
        index                          = requests.size();
        requests.emplace_back().values = view;
      }
      auto& aggregations = requests[*index].aggregations;
      positions[r]       = {*index, aggregations.size()};
      aggregations.push_back(
        make_aggregation<cudf::groupby_aggregation>(request.kind, view.type()));
    }
    cudf::groupby::groupby groupby(input.select(plan.group_columns), cudf::null_policy::INCLUDE);
    auto grouped = groupby.aggregate(requests, stream, mr);
    grouped_keys = std::move(grouped.first);
    for (std::size_t r = 0; r < plan.requests.size(); ++r) {
      if (!positions[r]) { continue; }
      auto const [index, position] = *positions[r];
      results[r]                   = std::move(grouped.second[index].results[position]);
    }
  }
  values = {};
  for (std::size_t r = 0; r < plan.requests.size(); ++r) {
    if (results[r] && plan.requests[r].kind == plan_type::aggregation::sum) {
      results[r] = checked_sum(std::move(results[r]), stream, mr);
    }
  }

  // 3. Make each output from the request results, with one value per group.
  auto outputs = detail::make_window_outputs(plan, std::move(results), stream, mr);

  // 4. Give every input row the value of its group.
  if (plan.group_columns.empty()) {
    std::vector<std::unique_ptr<cudf::column>> expanded;
    for (auto const& column : outputs) {
      auto scalar = cudf::get_element(column->view(), 0, stream, mr);
      expanded.push_back(cudf::make_column_from_scalar(*scalar, input.num_rows(), stream, mr));
    }
    return std::make_unique<cudf::table>(std::move(expanded));
  }
  // Probe the group keys, including NULL keys, with nulls equal. distinct_hash_join returns one
  // build index per input row, in input order.
  cudf::distinct_hash_join lookup(grouped_keys->view(), cudf::null_equality::EQUAL, 0.5, stream);
  auto indices = lookup.left_join(input.select(plan.group_columns), stream, mr);
  cudf::column_view map(
    cudf::data_type{cudf::type_id::INT32}, input.num_rows(), indices->data(), nullptr, 0);
  return cudf::gather(cudf::table(std::move(outputs)).view(),
                      map,
                      cudf::out_of_bounds_policy::DONT_CHECK,
                      stream,
                      mr);
}

std::unique_ptr<cudf::table> evaluate_whole_partition_window(
  cudf::table_view input,
  std::vector<cudf::size_type> const& partition_columns,
  std::vector<window_aggregate> const& aggregates,
  ::cuda::stream_ref stream,
  rmm::device_async_resource_ref mr)
{
  std::vector<cudf::data_type> types;
  for (auto const& column : input) {
    types.push_back(column.type());
  }
  return evaluate_whole_partition_window(
    plan_whole_partition_window(types, partition_columns, aggregates), input, stream, mr);
}

}  // namespace sirius::op
