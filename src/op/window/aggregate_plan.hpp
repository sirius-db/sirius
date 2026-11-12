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

#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>

#include <rmm/resource_ref.hpp>

#include <cuda/stream_ref>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <variant>
#include <vector>

namespace sirius::op {

/**
 * @brief The steps shared by the plans of the window aggregate evaluators.
 *
 * A plan is decided from types alone. Its evaluation has three stages, each a list in this struct:
 *
 * 1. `values`: the columns the cuDF aggregations read, derived from input columns.
 * 2. `requests`: the cuDF aggregations, each over one value column. SQL aggregates that need the
 *    same aggregation of the same values share one request. How the requests run, over frames or
 *    over whole groups, belongs to the evaluator.
 * 3. `outputs`: how each SQL aggregate's result is made from the request results.
 */
struct window_aggregate_plan {
  /// How a value column is derived from its input column.
  enum class preparation : std::uint8_t {
    none,  ///< The input column itself.
    cast,  ///< The input cast to `type`.
    /// The floating point input with NaN replaced by NULL. The input itself when it holds no NaN,
    /// which is decided for each table.
    nans_to_nulls,
    /// BOOL8 flags of the floating point input, true for NaN and false otherwise, including for
    /// NULL. Absent when the input holds no NaN, and then requests over it are not run.
    nan_flags,
    /// The input itself when a sum over all of its rows provably fits the input type, which is
    /// decided for each table, and otherwise the input cast to `type`. For INT64, DECIMAL32 and
    /// DECIMAL64 inputs only.
    widen_if_needed
  };

  struct value_column {
    cudf::size_type input_column;
    /// The type of the input column.
    cudf::data_type input_type;
    preparation how;
    /// The type of the column after preparation, or after widening for `widen_if_needed`.
    cudf::data_type type;
  };

  /// A cuDF aggregation.
  enum class aggregation : std::uint8_t { sum, count_valid, count_all, min, max };

  struct request {
    /// Index into `values`.
    std::size_t values;
    aggregation kind;
    /// The type of the request's result once the evaluator has run it: the type of a sum after
    /// any widening, INT64 for a count, which is also never NULL, and the value type otherwise.
    cudf::data_type type;
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
  /// The output is a floating point `min` or `max` over NaN free values, made NaN by
  /// `restore_nan` (see float_extremum.hpp) where NaN orders above every other value. `any_nan` is
  /// the `max` of the NaN flags over the same rows. It is not run, and the extremum is the output
  /// as is, when the argument holds no NaN.
  struct output_float_extremum {
    std::size_t extremum;
    std::size_t any_nan;
    bool is_min;
  };
  using output_recipe = std::variant<output_request, output_average, output_float_extremum>;

  struct output {
    output_recipe recipe;
    cudf::data_type type;
    /// Throw std::overflow_error if a value reaches 10^38 (a SUM over DECIMAL128).
    bool check_decimal38;
  };

  /// Every value column is read by at least one request.
  std::vector<value_column> values;
  std::vector<request> requests;
  std::vector<output> outputs;
  /// The distinct input columns the SQL aggregates take as arguments, in first use order. They
  /// must have their logical types, so a caller holding them in another carrier restores these.
  std::vector<cudf::size_type> argument_columns;
};

namespace detail {

/**
 * @brief Collects the value columns and requests of a plan, sharing equal ones.
 */
class window_aggregate_plan_builder {
 public:
  using plan_type = window_aggregate_plan;

  /**
   * @param input_types The column types of the input.
   * @param plan The plan to add value columns and requests to.
   */
  window_aggregate_plan_builder(std::vector<cudf::data_type> const& input_types, plan_type& plan);

  /// The index of the value column derived from @p input by @p how. A cast to the input's own
  /// type is the input itself.
  std::size_t values(cudf::size_type input, plan_type::preparation how, cudf::data_type type);

  /// The index of the request for @p kind over @p values among @p members, the requests that run
  /// together. A new request is added to @p members.
  std::size_t request(std::vector<std::size_t>& members,
                      std::size_t values,
                      plan_type::aggregation kind);

  /**
   * @brief Add the requests and the output of one SQL aggregate.
   *
   * @param members The requests that run together with the aggregate's requests.
   * @param function A normalized aggregate id.
   * @param input The argument column.
   * @param sum_values The value column the aggregate's `sum`, for `sum` and `avg`, reads.
   * @throws std::invalid_argument if @p function is unsupported.
   */
  void add_output(std::vector<std::size_t>& members,
                  sirius::aggregate_id function,
                  cudf::size_type input,
                  std::size_t sum_values);

 private:
  std::vector<cudf::data_type> const& _input_types;
  plan_type& _plan;
};

/// Throws std::invalid_argument unless @p index is a column of @p input_types.
void check_window_column(std::vector<cudf::data_type> const& input_types, cudf::size_type index);

/// Throws std::invalid_argument unless @p input has the value columns @p plan reads, with their
/// types, and the columns of @p indices.
void check_window_input(window_aggregate_plan const& plan,
                        std::vector<cudf::size_type> const& indices,
                        cudf::table_view input);

/// The value columns of @p plan over @p input. `owned` holds the columns made by preparation.
struct prepared_window_values {
  std::vector<std::unique_ptr<cudf::column>> owned;
  /// One view per value column of the plan. Empty for an absent `nan_flags` column.
  std::vector<std::optional<cudf::column_view>> views;
};

/// Whether @p plan's request @p r runs over @p values: false for a request over an absent
/// `nan_flags` column, whose result stays nullptr.
bool runs(window_aggregate_plan const& plan, prepared_window_values const& values, std::size_t r);

/**
 * @brief Prepares the value columns of @p plan from @p input.
 *
 * A floating point input with `nans_to_nulls` or `nan_flags` value columns is checked for NaN
 * once, which synchronizes @p stream.
 */
prepared_window_values prepare_window_values(window_aggregate_plan const& plan,
                                             cudf::table_view input,
                                             ::cuda::stream_ref stream,
                                             rmm::device_async_resource_ref mr);

/**
 * @brief Make the outputs of @p plan from the request results.
 *
 * Each result must have its request's type, except that a count may have any integer type and
 * NULLs, which become 0, and a sum may have a narrower type than its request's, to which it is
 * cast. NULLs of a `max` over NaN flags become false. The result of a request that did not run
 * (see `runs`) is nullptr. All results have the same number of rows. The results are released as
 * their last output is made.
 *
 * @throws std::overflow_error if an output with `check_decimal38` reaches 10^38.
 */
std::vector<std::unique_ptr<cudf::column>> make_window_outputs(
  window_aggregate_plan const& plan,
  std::vector<std::unique_ptr<cudf::column>> results,
  ::cuda::stream_ref stream,
  rmm::device_async_resource_ref mr);

}  // namespace detail
}  // namespace sirius::op
