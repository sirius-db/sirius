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

#include "op/window/aggregate_plan.hpp"

#include "op/aggregate/aggregate_op_util.hpp"
#include "op/window/aggregate_types.hpp"
#include "op/window/float_extremum.hpp"

#include <cudf/binaryop.hpp>
#include <cudf/copying.hpp>
#include <cudf/replace.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/unary.hpp>
#include <cudf/utilities/traits.hpp>

#include <algorithm>
#include <stdexcept>
#include <type_traits>
#include <unordered_set>
#include <utility>

namespace sirius::op::detail {
namespace {

using plan_type = window_aggregate_plan;

bool is_count(plan_type::aggregation kind)
{
  return kind == plan_type::aggregation::count_valid || kind == plan_type::aggregation::count_all;
}

// Calls @p f with each request @p recipe reads.
template <typename F>
void for_each_request(plan_type::output_recipe const& recipe, F&& f)
{
  std::visit(
    [&](auto const& r) {
      using T = std::decay_t<decltype(r)>;
      if constexpr (std::is_same_v<T, plan_type::output_request>) {
        f(r.request);
      } else if constexpr (std::is_same_v<T, plan_type::output_average>) {
        f(r.sum);
        f(r.count);
      } else {
        f(r.extremum);
        f(r.any_nan);
      }
    },
    recipe);
}

}  // namespace

window_aggregate_plan_builder::window_aggregate_plan_builder(
  std::vector<cudf::data_type> const& input_types, plan_type& plan)
  : _input_types(input_types), _plan(plan)
{
}

std::size_t window_aggregate_plan_builder::values(cudf::size_type input,
                                                  plan_type::preparation how,
                                                  cudf::data_type type)
{
  auto const input_type = _input_types[input];
  if (how == plan_type::preparation::cast && type == input_type) {
    how = plan_type::preparation::none;
  }
  auto found = std::find_if(_plan.values.begin(), _plan.values.end(), [&](auto const& value) {
    return value.input_column == input && value.how == how && value.type == type;
  });
  if (found != _plan.values.end()) {
    return static_cast<std::size_t>(found - _plan.values.begin());
  }
  _plan.values.push_back({input, input_type, how, type});
  return _plan.values.size() - 1;
}

std::size_t window_aggregate_plan_builder::request(std::vector<std::size_t>& members,
                                                   std::size_t values,
                                                   plan_type::aggregation kind)
{
  auto found = std::find_if(members.begin(), members.end(), [&](std::size_t r) {
    return _plan.requests[r].values == values && _plan.requests[r].kind == kind;
  });
  if (found != members.end()) { return *found; }
  auto const& value = _plan.values[values];
  auto const type   = is_count(kind)                        ? cudf::data_type{cudf::type_id::INT64}
                      : kind == plan_type::aggregation::sum ? window_sum_type(value.input_type)
                                                            : value.type;
  _plan.requests.push_back({values, kind, type});
  members.push_back(_plan.requests.size() - 1);
  return _plan.requests.size() - 1;
}

void window_aggregate_plan_builder::add_output(std::vector<std::size_t>& members,
                                               sirius::aggregate_id function,
                                               cudf::size_type input,
                                               std::size_t sum_values)
{
  using preparation = plan_type::preparation;
  using aggregation = plan_type::aggregation;
  auto const type   = _input_types[input];
  plan_type::output output{plan_type::output_request{0}, window_result_type(function, type), false};
  // The argument as it is. Only added to the plan by the aggregates that read it.
  auto const raw = [&] { return values(input, preparation::none, type); };
  switch (function) {
    case aggregate_id::sum:
      output.recipe = plan_type::output_request{request(members, sum_values, aggregation::sum)};
      // Narrower arguments cannot reach DECIMAL(38) in fewer than 2^31 rows.
      output.check_decimal38 = type.id() == cudf::type_id::DECIMAL128;
      break;
    case aggregate_id::avg:
      output.recipe = plan_type::output_average{request(members, sum_values, aggregation::sum),
                                                request(members, raw(), aggregation::count_valid)};
      break;
    case aggregate_id::count:
      output.recipe = plan_type::output_request{request(members, raw(), aggregation::count_valid)};
      break;
    case aggregate_id::count_star:
      output.recipe = plan_type::output_request{request(members, raw(), aggregation::count_all)};
      break;
    case aggregate_id::min:
    case aggregate_id::max: {
      auto const kind = function == aggregate_id::min ? aggregation::min : aggregation::max;
      if (!cudf::is_floating_point(type)) {
        output.recipe = plan_type::output_request{request(members, raw(), kind)};
        break;
      }
      // NaN orders above every other value. The extremum ignores NaN, and NaN is restored from
      // the maximum of the NaN flags.
      auto const finite = values(input, preparation::nans_to_nulls, type);
      auto const flags =
        values(input, preparation::nan_flags, cudf::data_type{cudf::type_id::BOOL8});
      output.recipe = plan_type::output_float_extremum{request(members, finite, kind),
                                                       request(members, flags, aggregation::max),
                                                       function == aggregate_id::min};
      break;
    }
    default: throw std::invalid_argument("Unsupported window function");
  }
  _plan.outputs.push_back(output);
  auto& arguments = _plan.argument_columns;
  if (std::find(arguments.begin(), arguments.end(), input) == arguments.end()) {
    arguments.push_back(input);
  }
}

void check_window_column(std::vector<cudf::data_type> const& input_types, cudf::size_type index)
{
  if (index < 0 || static_cast<std::size_t>(index) >= input_types.size()) {
    throw std::invalid_argument("Window column is outside the input schema");
  }
}

void check_window_input(window_aggregate_plan const& plan,
                        std::vector<cudf::size_type> const& indices,
                        cudf::table_view input)
{
  auto const check = [&](cudf::size_type index) {
    if (index < 0 || index >= input.num_columns()) {
      throw std::invalid_argument("Window input lacks a column of its plan");
    }
  };
  for (auto index : indices) {
    check(index);
  }
  for (auto const& value : plan.values) {
    check(value.input_column);
    if (input.column(value.input_column).type() != value.input_type) {
      throw std::invalid_argument("Window input type differs from its plan");
    }
  }
}

prepared_window_values prepare_window_values(window_aggregate_plan const& plan,
                                             cudf::table_view input,
                                             ::cuda::stream_ref stream,
                                             rmm::device_async_resource_ref mr)
{
  using preparation = plan_type::preparation;
  std::vector<int> candidates;
  for (auto const& value : plan.values) {
    if (value.how == preparation::widen_if_needed) { candidates.push_back(value.input_column); }
  }
  // One measurement, and one synchronization, for all the columns that may need widening.
  auto const widen = candidates.empty() ? std::unordered_set<int>{}
                                        : sums_needing_widening(input, candidates, stream, mr);
  // One NaN check for each floating point input whose NaN must be handled, shared by its value
  // columns.
  std::vector<std::pair<cudf::size_type, nan_free_argument>> nan_free;
  auto const nan_free_of = [&](cudf::size_type column) -> nan_free_argument& {
    auto found = std::find_if(
      nan_free.begin(), nan_free.end(), [&](auto const& entry) { return entry.first == column; });
    if (found != nan_free.end()) { return found->second; }
    return nan_free.emplace_back(column, make_nan_free_argument(input.column(column), stream, mr))
      .second;
  };
  prepared_window_values prepared;
  for (auto const& value : plan.values) {
    auto const column = input.column(value.input_column);
    switch (value.how) {
      case preparation::none: prepared.views.push_back(column); continue;
      case preparation::nans_to_nulls: {
        auto const& argument = nan_free_of(value.input_column);
        prepared.views.push_back(argument.finite ? argument.finite->view() : column);
        continue;
      }
      case preparation::nan_flags: {
        auto const& argument = nan_free_of(value.input_column);
        prepared.views.push_back(argument.flags ? std::optional{argument.flags->view()}
                                                : std::nullopt);
        continue;
      }
      case preparation::widen_if_needed:
        if (!widen.contains(value.input_column)) {
          prepared.views.push_back(column);
          continue;
        }
        [[fallthrough]];
      case preparation::cast:
        prepared.owned.push_back(cudf::cast(column, value.type, stream, mr));
        break;
    }
    prepared.views.push_back(prepared.owned.back()->view());
  }
  // The NaN free copies and flags stay alive with the views.
  for (auto& [column, argument] : nan_free) {
    if (argument.flags) {
      prepared.owned.push_back(std::move(argument.flags));
      prepared.owned.push_back(std::move(argument.finite));
    }
  }
  return prepared;
}

bool runs(window_aggregate_plan const& plan, prepared_window_values const& values, std::size_t r)
{
  return values.views[plan.requests[r].values].has_value();
}

std::vector<std::unique_ptr<cudf::column>> make_window_outputs(
  window_aggregate_plan const& plan,
  std::vector<std::unique_ptr<cudf::column>> results,
  ::cuda::stream_ref stream,
  rmm::device_async_resource_ref mr)
{
  // Bring every result to its request's type. Counts are also 0 instead of NULL.
  for (std::size_t r = 0; r < plan.requests.size(); ++r) {
    auto const& request = plan.requests[r];
    auto& result        = results[r];
    if (!result) { continue; }
    if (result->type() != request.type) {
      result = cudf::cast(result->view(), request.type, stream, mr);
    }
    if (is_count(request.kind) && result->has_nulls()) {
      auto const zero = cudf::numeric_scalar<std::int64_t>(0, true, stream, mr);
      result          = cudf::replace_nulls(result->view(), zero, stream, mr);
    }
    // The maximum of the NaN flags of an empty frame is NULL, and an empty frame has no NaN.
    if (plan.values[request.values].how == plan_type::preparation::nan_flags &&
        result->has_nulls()) {
      auto const no = cudf::numeric_scalar<bool>(false, true, stream, mr);
      result        = cudf::replace_nulls(result->view(), no, stream, mr);
    }
  }

  // A result is moved to its last reader and copied for the others.
  std::vector<int> readers(plan.requests.size(), 0);
  for (auto const& output : plan.outputs) {
    for_each_request(output.recipe, [&](std::size_t r) { ++readers[r]; });
  }
  auto const take = [&](std::size_t r) {
    if (readers[r] == 1 || !results[r]) { return std::move(results[r]); }
    return std::make_unique<cudf::column>(results[r]->view(), stream, mr);
  };
  auto const f64 = cudf::data_type{cudf::type_id::FLOAT64};
  std::vector<std::unique_ptr<cudf::column>> columns;
  for (auto const& output : plan.outputs) {
    std::unique_ptr<cudf::column> column;
    if (auto const* take_request = std::get_if<plan_type::output_request>(&output.recipe)) {
      column = take(take_request->request);
    } else if (auto const* average = std::get_if<plan_type::output_average>(&output.recipe)) {
      auto sum   = cudf::cast(results[average->sum]->view(), f64, stream, mr);
      auto count = cudf::cast(results[average->count]->view(), f64, stream, mr);
      column     = cudf::binary_operation(
        sum->view(), count->view(), cudf::binary_operator::DIV, f64, stream, mr);
    } else {
      auto const& extremum = std::get<plan_type::output_float_extremum>(output.recipe);
      column               = take(extremum.extremum);
      if (results[extremum.any_nan]) {
        column = restore_nan(std::move(column),
                             results[extremum.any_nan]->view(),
                             extremum.is_min ? aggregate_id::min : aggregate_id::max,
                             stream,
                             mr);
      }
    }
    for_each_request(output.recipe, [&](std::size_t r) {
      if (--readers[r] == 0) { results[r].reset(); }
    });
    if (output.check_decimal38) { check_decimal_sum_fits(column->view(), stream, mr); }
    columns.push_back(std::move(column));
  }
  return columns;
}

}  // namespace sirius::op::detail
