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

#include "op/window/framed.hpp"

#include <cudf/aggregation.hpp>
#include <cudf/binaryop.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/fixed_point/fixed_point.hpp>
#include <cudf/rolling.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/wrappers/durations.hpp>

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace sirius::op {

void validate(window_frame const& frame, std::size_t order_key_count)
{
  auto const check = [&](frame_bound const& bound) {
    if (auto const* rows = std::get_if<frame_row_offset>(&bound)) {
      if (frame.unit != frame_unit::rows) {
        throw std::invalid_argument("A row offset requires a ROWS frame");
      }
      if (rows->rows < 0) { throw std::invalid_argument("A window frame offset is negative"); }
    } else if (auto const* range = std::get_if<frame_range_offset>(&bound)) {
      if (frame.unit != frame_unit::range) {
        throw std::invalid_argument("A value offset requires a RANGE frame");
      }
      if (range->delta < 0) { throw std::invalid_argument("A window frame offset is negative"); }
      if (order_key_count != 1) {
        throw std::invalid_argument("A RANGE frame with an offset requires exactly one order key");
      }
    }
  };
  check(frame.start);
  check(frame.end);
}

namespace {

using plan_type = framed_window_plan;

// ---------------------------------------------------------------------------------------------
// Planning: types only, no device memory.
// ---------------------------------------------------------------------------------------------

// The position of a bound relative to the current row, in rows or in key units. Unbounded is
// outside every offset.
__int128_t relative_position(frame_bound const& bound, bool is_start)
{
  constexpr auto infinity = std::numeric_limits<__int128_t>::max();
  return std::visit(
    [&](auto const& b) -> __int128_t {
      using T = std::decay_t<decltype(b)>;
      if constexpr (std::is_same_v<T, frame_unbounded>) {
        return is_start ? -infinity : infinity;
      } else if constexpr (std::is_same_v<T, frame_current_row>) {
        return 0;
      } else if constexpr (std::is_same_v<T, frame_row_offset>) {
        return b.preceding ? -static_cast<__int128_t>(b.rows) : b.rows;
      } else {
        return b.preceding ? -b.delta : b.delta;
      }
    },
    bound);
}

// Whether the start of a frame lies after its end. A ROWS frame like that is empty for every row.
// A RANGE frame like that is empty for every non-NULL key, but the windows cuDF finds for a row
// can then end before they start.
bool start_after_end(window_frame const& frame)
{
  return relative_position(frame.start, true) > relative_position(frame.end, false);
}

// No two rows of a table are further apart than this, so a larger ROWS offset acts like this one.
// Clamping to it keeps cuDF's `-(preceding - 1) <= following` check free of overflow.
constexpr std::int64_t max_row_offset = std::numeric_limits<cudf::size_type>::max() - 1;

// The cuDF row bound of one side of a ROWS frame. cuDF's preceding size counts the current row,
// its following size does not.
std::optional<cudf::size_type> rows_bound(frame_bound const& bound, bool is_start)
{
  if (std::holds_alternative<frame_unbounded>(bound)) { return std::nullopt; }
  if (std::holds_alternative<frame_current_row>(bound)) { return is_start ? 1 : 0; }
  auto const& offset = std::get<frame_row_offset>(bound);
  auto const rows    = static_cast<cudf::size_type>(std::min(offset.rows, max_row_offset));
  if (is_start) { return offset.preceding ? rows + 1 : 1 - rows; }
  return offset.preceding ? -rows : rows;
}

// The largest magnitude a RANGE offset can have over an order key of type @p key.
__int128_t max_range_delta(cudf::data_type key)
{
  using cudf::type_id;
  switch (key.id()) {
    case type_id::INT8: return std::numeric_limits<std::int8_t>::max();
    case type_id::INT16: return std::numeric_limits<std::int16_t>::max();
    case type_id::INT32:
    case type_id::DECIMAL32:
    case type_id::TIMESTAMP_DAYS: return std::numeric_limits<std::int32_t>::max();
    case type_id::INT64:
    case type_id::DECIMAL64:
    case type_id::TIMESTAMP_SECONDS:
    case type_id::TIMESTAMP_MILLISECONDS:
    case type_id::TIMESTAMP_MICROSECONDS:
    case type_id::TIMESTAMP_NANOSECONDS: return std::numeric_limits<std::int64_t>::max();
    case type_id::DECIMAL128: return std::numeric_limits<__int128_t>::max();
    default: throw std::invalid_argument("Unsupported order key type for a RANGE frame offset");
  }
}

// One end of a RANGE frame. An offset delta is positive in the endpoint's own direction, before
// the row for a preceding start, after it for a following end, and negative across the row.
plan_type::range_endpoint range_endpoint(frame_bound const& bound,
                                         bool is_start,
                                         std::optional<cudf::data_type> key)
{
  using kind = plan_type::range_endpoint::kind;
  if (std::holds_alternative<frame_unbounded>(bound)) { return {kind::unbounded, 0}; }
  if (std::holds_alternative<frame_current_row>(bound)) { return {kind::current_row, 0}; }
  auto const& offset = std::get<frame_range_offset>(bound);
  // The magnitude must fit, so that the delta fits with either sign.
  if (offset.delta > max_range_delta(*key)) {
    throw std::invalid_argument("A RANGE frame offset does not fit its order key type");
  }
  return {kind::offset, offset.preceding == is_start ? offset.delta : -offset.delta};
}

bool has_range_offset(window_frame const& frame)
{
  return std::holds_alternative<frame_range_offset>(frame.start) ||
         std::holds_alternative<frame_range_offset>(frame.end);
}

plan_type::frame_strategy frame_strategy(window_frame const& frame,
                                         window_order const& order,
                                         std::vector<cudf::data_type> const& input_types)
{
  if (frame.unit == frame_unit::rows) {
    // cuDF rejects row bounds whose start lies after their end. Such a frame is empty.
    if (start_after_end(frame)) { return plan_type::empty_frame{}; }
    return plan_type::rows_frame{rows_bound(frame.start, true), rows_bound(frame.end, false)};
  }
  if (!has_range_offset(frame)) {
    return plan_type::peer_range_frame{range_endpoint(frame.start, true, std::nullopt),
                                       range_endpoint(frame.end, false, std::nullopt)};
  }
  auto const key = input_types[order.columns.front()];
  return plan_type::offset_range_frame{range_endpoint(frame.start, true, key),
                                       range_endpoint(frame.end, false, key),
                                       key,
                                       start_after_end(frame)};
}

// Builds a plan, sharing value columns, frames and requests between aggregates.
class plan_builder {
 public:
  plan_builder(std::vector<cudf::data_type> const& input_types, window_order const& order)
    : aggregates(input_types, plan), _input_types(input_types), _order(order)
  {
  }

  // The requests of the frame for @p frame.
  std::vector<std::size_t>& frame(window_frame const& frame)
  {
    auto found = std::find(_frames.begin(), _frames.end(), frame);
    if (found != _frames.end()) { return plan.frames[found - _frames.begin()].requests; }
    _frames.push_back(frame);
    plan.frames.push_back({frame_strategy(frame, _order, _input_types), {}});
    return plan.frames.back().requests;
  }

  plan_type plan;
  detail::window_aggregate_plan_builder aggregates;

 private:
  std::vector<cudf::data_type> const& _input_types;
  window_order const& _order;
  std::vector<window_frame> _frames;
};

// ---------------------------------------------------------------------------------------------
// Execution.
// ---------------------------------------------------------------------------------------------

std::unique_ptr<cudf::rolling_aggregation> make_rolling_aggregation(plan_type::aggregation kind)
{
  using aggregation = plan_type::aggregation;
  switch (kind) {
    case aggregation::sum: return cudf::make_sum_aggregation<cudf::rolling_aggregation>();
    case aggregation::count_valid: return cudf::make_count_aggregation<cudf::rolling_aggregation>();
    case aggregation::count_all:
      return cudf::make_count_aggregation<cudf::rolling_aggregation>(cudf::null_policy::INCLUDE);
    case aggregation::min: return cudf::make_min_aggregation<cudf::rolling_aggregation>();
    case aggregation::max: return cudf::make_max_aggregation<cudf::rolling_aggregation>();
  }
  throw std::logic_error("Unknown rolling window aggregation");
}

// A valid scalar holding @p delta in the unit of the order key type @p key. The plan checked that
// it fits.
std::unique_ptr<cudf::scalar> range_delta_scalar(__int128_t delta,
                                                 cudf::data_type key,
                                                 ::cuda::stream_ref stream,
                                                 rmm::device_async_resource_ref mr)
{
  auto const integer = [&]<typename T>() -> std::unique_ptr<cudf::scalar> {
    return std::make_unique<cudf::numeric_scalar<T>>(static_cast<T>(delta), true, stream, mr);
  };
  auto const decimal = [&]<typename T>() -> std::unique_ptr<cudf::scalar> {
    return std::make_unique<cudf::fixed_point_scalar<T>>(
      static_cast<typename T::rep>(delta), numeric::scale_type{key.scale()}, true, stream, mr);
  };
  auto const duration = [&]<typename D>() -> std::unique_ptr<cudf::scalar> {
    return std::make_unique<cudf::duration_scalar<D>>(
      D{static_cast<typename D::rep>(delta)}, true, stream, mr);
  };
  using cudf::type_id;
  switch (key.id()) {
    case type_id::INT8: return integer.template operator()<std::int8_t>();
    case type_id::INT16: return integer.template operator()<std::int16_t>();
    case type_id::INT32: return integer.template operator()<std::int32_t>();
    case type_id::INT64: return integer.template operator()<std::int64_t>();
    case type_id::DECIMAL32: return decimal.template operator()<numeric::decimal32>();
    case type_id::DECIMAL64: return decimal.template operator()<numeric::decimal64>();
    case type_id::DECIMAL128: return decimal.template operator()<numeric::decimal128>();
    case type_id::TIMESTAMP_DAYS: return duration.template operator()<cudf::duration_D>();
    case type_id::TIMESTAMP_SECONDS: return duration.template operator()<cudf::duration_s>();
    case type_id::TIMESTAMP_MILLISECONDS: return duration.template operator()<cudf::duration_ms>();
    case type_id::TIMESTAMP_MICROSECONDS: return duration.template operator()<cudf::duration_us>();
    case type_id::TIMESTAMP_NANOSECONDS: return duration.template operator()<cudf::duration_ns>();
    default: throw std::logic_error("Unsupported order key type for a RANGE frame offset");
  }
}

// The cuDF endpoint of @p endpoint. An offset endpoint refers to @p delta, which must outlive it.
cudf::range_window_type cudf_endpoint(plan_type::range_endpoint const& endpoint,
                                      std::unique_ptr<cudf::scalar> const& delta)
{
  using kind = plan_type::range_endpoint::kind;
  switch (endpoint.type) {
    case kind::unbounded: return cudf::unbounded{};
    case kind::current_row: return cudf::current_row{};
    case kind::offset: return cudf::bounded_closed{*delta};
  }
  throw std::logic_error("Unknown RANGE frame endpoint");
}

// An all-NULL column of @p type with @p size rows.
std::unique_ptr<cudf::column> all_null(cudf::data_type type,
                                       cudf::size_type size,
                                       ::cuda::stream_ref stream,
                                       rmm::device_async_resource_ref mr)
{
  auto scalar = cudf::make_default_constructed_scalar(type, stream, mr);
  scalar->set_valid_async(false, stream);
  return cudf::make_column_from_scalar(*scalar, size, stream, mr);
}

// Runs the requests of one frame and stores their raw cuDF results in @p results.
class frame_runner {
 public:
  frame_runner(plan_type const& plan,
               cudf::table_view input,
               detail::prepared_window_values const& values,
               std::vector<std::unique_ptr<cudf::column>>& results,
               ::cuda::stream_ref stream,
               rmm::device_async_resource_ref mr)
    : _plan(plan),
      _input(input),
      _group_keys(input.select(plan.group_columns)),
      _values(values),
      _results(results),
      _stream(stream),
      _mr(mr)
  {
  }

  void run(plan_type::frame const& frame)
  {
    // Requests over absent NaN flags do not run.
    _running.clear();
    for (auto r : frame.requests) {
      if (detail::runs(_plan, _values, r)) { _running.push_back(r); }
    }
    if (_running.empty()) { return; }
    _members = &_running;
    std::visit([&](auto const& strategy) { (*this)(strategy); }, frame.strategy);
  }

  void operator()(plan_type::empty_frame const&)
  {
    for (auto r : *_members) {
      _results[r] = all_null(_plan.requests[r].type, _input.num_rows(), _stream, _mr);
    }
  }

  void operator()(plan_type::rows_frame const& rows)
  {
    auto const bound = [](std::optional<cudf::size_type> value) {
      return value ? cudf::window_bounds::get(*value) : cudf::window_bounds::unbounded();
    };
    for (auto r : *_members) {
      _results[r] = cudf::grouped_rolling_window(_group_keys,
                                                 values(r),
                                                 bound(rows.preceding),
                                                 bound(rows.following),
                                                 1,
                                                 *make_rolling_aggregation(_plan.requests[r].kind),
                                                 _stream,
                                                 _mr);
    }
  }

  void operator()(plan_type::peer_range_frame const& range)
  {
    std::vector<cudf::rolling_request> requests;
    for (auto r : *_members) {
      requests.push_back({values(r), 1, make_rolling_aggregation(_plan.requests[r].kind)});
    }
    auto table   = cudf::grouped_range_rolling_window(_group_keys,
                                                    _input.select(_plan.order.columns),
                                                    _plan.order.directions,
                                                    _plan.order.nulls,
                                                    cudf_endpoint(range.start, nullptr),
                                                    cudf_endpoint(range.end, nullptr),
                                                    requests,
                                                    _stream,
                                                    _mr);
    auto columns = table->release();
    for (std::size_t m = 0; m < _members->size(); ++m) {
      _results[(*_members)[m]] = std::move(columns[m]);
    }
  }

  void operator()(plan_type::offset_range_frame const& range)
  {
    auto const delta = [&](plan_type::range_endpoint const& endpoint) {
      return endpoint.type == plan_type::range_endpoint::kind::offset
               ? range_delta_scalar(endpoint.delta, range.key_type, _stream, _mr)
               : nullptr;
    };
    auto const start_delta      = delta(range.start);
    auto const end_delta        = delta(range.end);
    auto [preceding, following] = cudf::make_range_windows(_group_keys,
                                                           _input.column(_plan.order.columns[0]),
                                                           _plan.order.directions[0],
                                                           _plan.order.nulls[0],
                                                           cudf_endpoint(range.start, start_delta),
                                                           cudf_endpoint(range.end, end_delta),
                                                           _stream,
                                                           _mr);
    if (range.clamp_reversed) {
      // Move the end of a window that ends before it starts up to its start. A window covers
      // rows [i - preceding + 1, i + following], so it is empty when following = -preceding.
      auto const i32  = cudf::data_type{cudf::type_id::INT32};
      auto const zero = cudf::numeric_scalar<cudf::size_type>(0, true, _stream, _mr);
      auto negated    = cudf::binary_operation(
        zero, preceding->view(), cudf::binary_operator::SUB, i32, _stream, _mr);
      following = cudf::binary_operation(
        following->view(), negated->view(), cudf::binary_operator::NULL_MAX, i32, _stream, _mr);
    }
    for (auto r : *_members) {
      _results[r] = cudf::rolling_window(values(r),
                                         preceding->view(),
                                         following->view(),
                                         1,
                                         *make_rolling_aggregation(_plan.requests[r].kind),
                                         _stream,
                                         _mr);
    }
  }

 private:
  cudf::column_view values(std::size_t request) const
  {
    return *_values.views[_plan.requests[request].values];
  }

  plan_type const& _plan;
  cudf::table_view _input;
  cudf::table_view _group_keys;
  detail::prepared_window_values const& _values;
  std::vector<std::unique_ptr<cudf::column>>& _results;
  ::cuda::stream_ref _stream;
  rmm::device_async_resource_ref _mr;
  std::vector<std::size_t> _running;
  std::vector<std::size_t> const* _members = nullptr;
};

// Throws unless @p input has the columns the plan reads, with the types it was built for.
void check_input(plan_type const& plan, cudf::table_view input)
{
  auto indices = plan.group_columns;
  indices.insert(indices.end(), plan.order.columns.begin(), plan.order.columns.end());
  detail::check_window_input(plan, indices, input);
  for (auto const& frame : plan.frames) {
    if (auto const* range = std::get_if<plan_type::offset_range_frame>(&frame.strategy);
        range && input.column(plan.order.columns[0]).type() != range->key_type) {
      throw std::invalid_argument("Window input type differs from its plan");
    }
  }
}

}  // namespace

framed_window_plan plan_framed_window(std::vector<cudf::data_type> const& input_types,
                                      std::vector<cudf::size_type> const& group_columns,
                                      window_order const& order,
                                      std::vector<framed_aggregate> const& aggregates)
{
  if (aggregates.empty()) { throw std::invalid_argument("Window requires at least one aggregate"); }
  auto const check_index = [&](cudf::size_type index) {
    detail::check_window_column(input_types, index);
  };
  for (auto index : group_columns) {
    check_index(index);
  }
  if (order.directions.size() != order.columns.size() ||
      order.nulls.size() != order.columns.size()) {
    throw std::invalid_argument("Window order has mismatched lengths");
  }
  for (auto index : order.columns) {
    check_index(index);
  }

  plan_builder builder(input_types, order);
  builder.plan.group_columns = group_columns;
  builder.plan.order         = order;
  for (auto const& requested : aggregates) {
    auto const function = normalize_window_aggregate(requested.function);
    auto const input    = requested.argument_column;
    check_index(input);
    validate(requested.frame, order.columns.size());
    auto const type = input_types[input];
    if ((function == aggregate_id::min || function == aggregate_id::max) &&
        type.id() != cudf::type_id::STRING &&
        !cudf::is_valid_rolling_aggregation(
          type, function == aggregate_id::min ? cudf::aggregation::MIN : cudf::aggregation::MAX)) {
      // cuDF's support check does not know that rolling MIN and MAX run on strings through ARGMIN
      // and ARGMAX.
      throw std::invalid_argument("Unsupported window argument type");
    }
    // Without order keys every row is a peer of every other, so a RANGE frame (which then has no
    // offset) is the whole partition.
    auto& frame =
      builder.frame(requested.frame.unit == frame_unit::range && order.columns.empty()
                      ? window_frame{frame_unit::rows, frame_unbounded{}, frame_unbounded{}}
                      : requested.frame);
    // A rolling sum accumulates in the type of its result.
    auto const sum_values =
      function == aggregate_id::sum || function == aggregate_id::avg
        ? builder.aggregates.values(input, plan_type::preparation::cast, window_sum_type(type))
        : 0;
    builder.aggregates.add_output(frame, function, input, sum_values);
  }
  return std::move(builder.plan);
}

std::vector<cudf::size_type> logical_type_columns(framed_window_plan const& plan)
{
  auto columns          = plan.argument_columns;
  auto const has_offset = std::any_of(plan.frames.begin(), plan.frames.end(), [](auto const& f) {
    return std::holds_alternative<plan_type::offset_range_frame>(f.strategy);
  });
  if (has_offset &&
      std::find(columns.begin(), columns.end(), plan.order.columns[0]) == columns.end()) {
    columns.push_back(plan.order.columns[0]);
  }
  return columns;
}

std::unique_ptr<cudf::table> evaluate_framed_window(framed_window_plan const& plan,
                                                    cudf::table_view sorted_input,
                                                    ::cuda::stream_ref stream,
                                                    rmm::device_async_resource_ref mr)
{
  check_input(plan, sorted_input);
  auto const num_rows = sorted_input.num_rows();
  if (num_rows == 0) {
    std::vector<std::unique_ptr<cudf::column>> empty;
    for (auto const& output : plan.outputs) {
      empty.push_back(cudf::make_empty_column(output.type));
    }
    return std::make_unique<cudf::table>(std::move(empty));
  }

  // 1. Prepare the value columns.
  auto values = detail::prepare_window_values(plan, sorted_input, stream, mr);

  // 2. Run the requests, frame by frame.
  std::vector<std::unique_ptr<cudf::column>> results(plan.requests.size());
  frame_runner runner(plan, sorted_input, values, results, stream, mr);
  for (auto const& frame : plan.frames) {
    runner.run(frame);
  }
  values = {};

  // 3. Make each output from the request results.
  return std::make_unique<cudf::table>(
    detail::make_window_outputs(plan, std::move(results), stream, mr));
}

std::unique_ptr<cudf::table> evaluate_framed_window(
  cudf::table_view sorted_input,
  std::vector<cudf::size_type> const& group_columns,
  window_order const& order,
  std::vector<framed_aggregate> const& aggregates,
  ::cuda::stream_ref stream,
  rmm::device_async_resource_ref mr)
{
  std::vector<cudf::data_type> types;
  for (auto const& column : sorted_input) {
    types.push_back(column.type());
  }
  return evaluate_framed_window(
    plan_framed_window(types, group_columns, order, aggregates), sorted_input, stream, mr);
}

}  // namespace sirius::op
