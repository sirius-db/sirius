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

#pragma once

#include "expression/aggregate_id.hpp"
#include "op/window/aggregate_types.hpp"
#include "op/window/frame.hpp"

#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>

#include <rmm/resource_ref.hpp>

#include <cuda/stream_ref>

#include <cstdint>
#include <memory>
#include <optional>
#include <variant>
#include <vector>

namespace sirius::op {

/**
 * @brief One aggregate of a window, over its own frame.
 */
struct framed_aggregate {
  /// One of `sum` (`sum_no_overflow` is treated as `sum`), `avg`, `count`, `count_star`, `min` and
  /// `max`. Any other id is rejected.
  sirius::aggregate_id function;
  /// Index of the argument column in the input table. `count_star` counts rows including NULLs, so
  /// any column serves as its argument.
  cudf::size_type argument_column;
  /// The frame the aggregate is computed over.
  window_frame frame;
};

/**
 * @brief How the evaluation of framed window aggregates runs, decided from types alone.
 *
 * plan_framed_window() builds it from the input schema and the requested aggregates, without
 * touching device memory, so a planner can build it once, decline the window when building
 * throws, and hand the plan to the operator. evaluate_framed_window() then executes it on each
 * table. The evaluation has three stages, each a list in this struct:
 *
 * 1. `values`: the columns the rolling aggregations read, derived from input columns.
 * 2. `requests`: the cuDF rolling aggregations, each over one value column, grouped by frame in
 *    `frames`. SQL aggregates that need the same aggregation of the same values over the same
 *    frame share one request.
 * 3. `outputs`: how each SQL aggregate's result is made from the request results.
 */
struct framed_window_plan {
  /// How a value column is derived from its input column.
  enum class preparation : std::uint8_t {
    none,          ///< The input column itself.
    cast,          ///< The input cast to `type`.
    nans_to_nulls  ///< The floating point input with NaN replaced by NULL.
  };

  struct value_column {
    cudf::size_type input_column;
    preparation how;
    /// The type of the column after preparation.
    cudf::data_type type;
  };

  /// A cuDF rolling aggregation.
  enum class aggregation : std::uint8_t { sum, count_valid, count_all, min, max };

  struct request {
    /// Index into `values`.
    std::size_t values;
    aggregation kind;
    /// The type of the request's result: the value type, except INT64 for a count, which is also
    /// never NULL (an empty frame counts 0).
    cudf::data_type type;
  };

  /// A frame that is empty for every row. Its requests are NULL, or 0 for a count, without
  /// calling cuDF.
  struct empty_frame {};

  /// A ROWS frame as cuDF row bounds. `preceding` counts the current row and `following` does not;
  /// nullopt is unbounded. The start never lies after the end.
  struct rows_frame {
    std::optional<cudf::size_type> preceding;
    std::optional<cudf::size_type> following;
  };

  /// One end of a RANGE frame.
  struct range_endpoint {
    enum class kind : std::uint8_t { unbounded, current_row, offset };
    kind type;
    /// For an offset, the signed distance from the current row's key, in the unit of the order
    /// key type and known to fit it. Positive moves the endpoint outward on its own side, before
    /// the row for the start and after it for the end. `δ FOLLOWING` as a start and `δ PRECEDING`
    /// as an end are negative.
    __int128_t delta;
  };

  /// A RANGE frame with `unbounded` and `current_row` endpoints only, over every order key.
  struct peer_range_frame {
    range_endpoint start;
    range_endpoint end;
  };

  /// A RANGE frame with at least one offset endpoint, over the single order key.
  struct offset_range_frame {
    range_endpoint start;
    range_endpoint end;
    /// The type of the order key, which the offsets are expressed in.
    cudf::data_type key_type;
    /// The frame's start lies after its end, so the windows found for a row can end before they
    /// start. cuDF requires start <= end, so such windows are made empty.
    bool clamp_reversed;
  };

  using frame_strategy =
    std::variant<empty_frame, rows_frame, peer_range_frame, offset_range_frame>;

  struct frame {
    frame_strategy strategy;
    /// Indices into `requests` of the requests over this frame.
    std::vector<std::size_t> requests;
  };

  /// The output is a request result as is.
  struct output_request {
    std::size_t request;
  };
  /// The output is the sum request divided by the count request, in FLOAT64.
  struct output_average {
    std::size_t sum;
    std::size_t count;
  };
  /// The output is a floating point `min` or `max` over NaN free values, made NaN where NaN
  /// orders above every other value: for `max` when the frame holds a NaN, and for `min` when
  /// every non-NULL value of the frame is NaN. A frame holds a NaN where `count` exceeds
  /// `finite_count`.
  struct output_float_extremum {
    std::size_t extremum;
    std::size_t finite_count;
    std::size_t count;
    bool is_min;
  };
  using output_recipe = std::variant<output_request, output_average, output_float_extremum>;

  struct output {
    output_recipe recipe;
    cudf::data_type type;
    /// Throw std::overflow_error if a value reaches 10^38 (a SUM over DECIMAL128).
    bool check_decimal38;
  };

  std::vector<cudf::size_type> group_columns;
  window_order order;
  std::vector<value_column> values;
  std::vector<request> requests;
  std::vector<frame> frames;
  std::vector<output> outputs;
};

/**
 * @brief Plan the evaluation of framed window aggregates over tables of the schema @p input_types.
 *
 * Rows with equal group column values form one partition, and NULL group values are equal to each
 * other. Without group columns the whole table is one partition. `ROWS` frames ignore the order
 * keys. A `RANGE` frame without order keys spans the whole partition, since every row is a peer
 * of every other. A `RANGE` frame with an offset and a NULL order key spans the row's NULL peers on
 * the offset side. A frame whose start lies after its end is empty.
 *
 * Group and order columns can be in any representation that preserves equality and order, so an
 * exact narrower carrier is fine, except the order key of a frame with a frame_range_offset: it
 * must have its logical type, since the offset is in that type's unit. Such a key is an 8 to 64 bit
 * signed integer, a decimal or a timestamp. Argument columns must have their logical types,
 * including decimal scales.
 *
 * Result types:
 * - `sum` over signed integers is DECIMAL128 with scale 0, the exact carrier of DuckDB's HUGEINT.
 *   Over decimals it is DECIMAL128 at the input scale and over floating point it is FLOAT64. The
 *   argument is cast to the result type before the rolling sum.
 * - `avg` is FLOAT64.
 * - `count` and `count_star` are INT64 and 0 for an empty frame.
 * - `min` and `max` keep the argument type. NaN orders above every other value. Strings compare
 *   by their bytes.
 *
 * `sum`, `avg`, `min` and `max` are NULL for an empty frame and for a frame whose arguments are
 * all NULL. Floating point sums may add in any order.
 *
 * @param input_types The column types of the tables the plan is evaluated on.
 * @param group_columns Indices of the group columns. Empty for a window without `PARTITION BY`.
 * @param order The window ORDER BY. May be empty.
 * @param aggregates The aggregates to evaluate, each with its frame. Must not be empty.
 *
 * @throws std::invalid_argument if @p aggregates is empty, a column index is outside
 * @p input_types, `order` has mismatched lengths, a frame is invalid (see validate()), an
 * aggregate id or argument type is unsupported, or a `RANGE` offset does not fit, or is not
 * supported by, the type of its order key.
 */
framed_window_plan plan_framed_window(std::vector<cudf::data_type> const& input_types,
                                      std::vector<cudf::size_type> const& group_columns,
                                      window_order const& order,
                                      std::vector<framed_aggregate> const& aggregates);

/**
 * @brief Evaluate a framed window plan over a table that is already sorted.
 *
 * Preconditions, not checked:
 * - @p sorted_input is sorted on the plan's group columns followed by its order keys, with the
 *   order's directions and NULL placement. The direction of the group columns does not matter:
 *   rows with equal group values only have to be adjacent.
 * - Every partition is complete: no row of a partition is missing from the table.
 *
 * The input is evaluated as one cuDF table, so it must fit in GPU memory and hold fewer than 2^31
 * rows.
 *
 * @param plan A plan from plan_framed_window() for the schema of @p sorted_input.
 * @param sorted_input The table to evaluate.
 * @param stream CUDA stream used for all device work.
 * @param mr Device memory resource used to allocate the result columns and the intermediate
 * columns created by the evaluator.
 *
 * @return One column per aggregate the plan was built for, in request order, each with
 * `sorted_input.num_rows()` rows in the row order of @p sorted_input. Nothing is sorted, gathered
 * or scattered. An empty input gives empty typed columns.
 *
 * @throws std::invalid_argument if a column the plan reads is missing from @p sorted_input or has
 * another type than the plan was built for.
 * @throws std::overflow_error if a `sum` over a DECIMAL128 argument reaches 10^38. A rolling sum
 * that wraps its 128 bit accumulator is not detected.
 *
 * @note Temporary allocations made inside cuDF use cuDF's current device resource.
 */
std::unique_ptr<cudf::table> evaluate_framed_window(framed_window_plan const& plan,
                                                    cudf::table_view sorted_input,
                                                    ::cuda::stream_ref stream,
                                                    rmm::device_async_resource_ref mr);

/**
 * @brief Plan and evaluate framed window aggregates over @p sorted_input in one call.
 *
 * Equivalent to evaluate_framed_window(plan_framed_window(types of @p sorted_input, ...), ...).
 * See plan_framed_window() for the arguments and the semantics.
 */
std::unique_ptr<cudf::table> evaluate_framed_window(
  cudf::table_view sorted_input,
  std::vector<cudf::size_type> const& group_columns,
  window_order const& order,
  std::vector<framed_aggregate> const& aggregates,
  ::cuda::stream_ref stream,
  rmm::device_async_resource_ref mr);

}  // namespace sirius::op
