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

#include "op/aggregate/stddev.hpp"

#include "duckdb/common/exception.hpp"

#include <cudf/aggregation.hpp>
#include <cudf/ast/expressions.hpp>
#include <cudf/binaryop.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/groupby.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/reduction.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/structs/structs_column_view.hpp>
#include <cudf/transform.hpp>
#include <cudf/unary.hpp>

#include <limits>

namespace sirius::op {
std::unique_ptr<cudf::column> make_stddev_state(std::unique_ptr<cudf::column> count,
                                                std::unique_ptr<cudf::column> mean,
                                                std::unique_ptr<cudf::column> m2,
                                                ::cuda::stream_ref stream,
                                                rmm::device_async_resource_ref mr)
{
  auto const rows = count->size();
  std::vector<std::unique_ptr<cudf::column>> children;
  if (count->type().id() != cudf::type_id::INT64) {
    count = cudf::cast(count->view(), cudf::data_type{cudf::type_id::INT64}, stream, mr);
  }
  children.push_back(std::move(count));
  children.push_back(std::move(mean));
  children.push_back(std::move(m2));
  return cudf::make_structs_column(rows, std::move(children), 0, rmm::device_buffer{}, stream, mr);
}

std::unique_ptr<cudf::column> local_stddev_state(cudf::column_view input,
                                                 ::cuda::stream_ref stream,
                                                 rmm::device_async_resource_ref mr)
{
  auto const f64 = cudf::data_type{cudf::type_id::FLOAT64};
  // M2 has no reduce API. Population variance uses the same centered accumulation; multiply
  // by the valid count to recover M2, retaining a zero M2 for singleton partials.
  auto count = cudf::reduce(input,
                            *cudf::make_count_aggregation<cudf::reduce_aggregation>(),
                            cudf::data_type{cudf::type_id::INT64},
                            stream,
                            mr);
  auto mean =
    cudf::reduce(input, *cudf::make_mean_aggregation<cudf::reduce_aggregation>(), f64, stream, mr);
  auto variance = cudf::reduce(
    input, *cudf::make_variance_aggregation<cudf::reduce_aggregation>(0), f64, stream, mr);
  auto count_col    = cudf::make_column_from_scalar(*count, 1, stream, mr);
  auto variance_col = cudf::make_column_from_scalar(*variance, 1, stream, mr);
  auto m2           = cudf::binary_operation(
    variance_col->view(), *count, cudf::binary_operator::MUL, f64, stream, mr);
  return make_stddev_state(std::move(count_col),
                           cudf::make_column_from_scalar(*mean, 1, stream, mr),
                           std::move(m2),
                           stream,
                           mr);
}

std::unique_ptr<cudf::column> merge_stddev_states(cudf::column_view states,
                                                  ::cuda::stream_ref stream,
                                                  rmm::device_async_resource_ref mr)
{
  // cuDF exposes MERGE_M2 only through groupby. The input contains one row per partial,
  // so a constant key here groups the small state column, rather than the original rows.
  cudf::numeric_scalar<int8_t> zero(0, true, stream, mr);
  auto keys = cudf::make_column_from_scalar(zero, states.size(), stream, mr);
  cudf::groupby::groupby groupby(cudf::table_view{{keys->view()}});
  std::vector<cudf::groupby::aggregation_request> requests(1);
  requests[0].values = states;
  requests[0].aggregations.push_back(cudf::make_merge_m2_aggregation<cudf::groupby_aggregation>());
  auto result = groupby.aggregate(requests, stream, mr);
  return std::move(result.second[0].results[0]);
}

std::unique_ptr<cudf::column> finalize_stddev(cudf::column_view states,
                                              ::cuda::stream_ref stream,
                                              rmm::device_async_resource_ref mr)
{
  cudf::structs_column_view state(states);
  auto count = state.get_sliced_child(0, stream);
  auto m2    = state.get_sliced_child(2, stream);
  cudf::numeric_scalar<int64_t> one(1, true, stream, mr);
  cudf::numeric_scalar<double> null(0.0, false, stream, mr);
  // Fuse sqrt(M2 / (count - 1)) and the count <= 1 NULL result into one JIT expression.
  // Ordinary floating arithmetic preserves NaN/Inf for the DuckDB-specific check below.
  cudf::ast::tree tree;
  auto const& count_ref       = tree.emplace<cudf::ast::column_reference>(0);
  auto const& m2_ref          = tree.emplace<cudf::ast::column_reference>(1);
  auto const& one_ref         = tree.emplace<cudf::ast::literal>(one);
  auto const& null_ref        = tree.emplace<cudf::ast::literal>(null);
  using op                    = cudf::ast::jit::op;
  auto const& enough_values   = cudf::ast::jit::operation(tree, op::GREATER, {count_ref, one_ref});
  auto const& denominator     = cudf::ast::jit::operation(tree, op::SUB, {count_ref, one_ref});
  auto const& denominator_f64 = cudf::ast::jit::operation(tree, op::CAST_TO_FLOAT64, {denominator});
  auto const& variance        = cudf::ast::jit::operation(tree, op::DIV, {m2_ref, denominator_f64});
  auto const& deviation       = cudf::ast::jit::operation(tree, op::SQRT, {variance});
  auto const& result =
    cudf::ast::jit::operation(tree, op::IF_ELSE, {deviation, null_ref, enough_values});
  auto output = cudf::compute_column_jit(cudf::table_view{{count, m2}}, result, stream, mr);

  // DuckDB rejects non-finite finalized results (but returns NULL for singleton NaN/Inf).
  auto nan = cudf::is_nan(output->view(), stream, mr);
  cudf::numeric_scalar<double> maximum(std::numeric_limits<double>::max(), true, stream, mr);
  auto infinite    = cudf::binary_operation(output->view(),
                                         maximum,
                                         cudf::binary_operator::GREATER,
                                         cudf::data_type{cudf::type_id::BOOL8},
                                         stream,
                                         mr);
  auto invalid     = cudf::binary_operation(nan->view(),
                                        infinite->view(),
                                        cudf::binary_operator::LOGICAL_OR,
                                        cudf::data_type{cudf::type_id::BOOL8},
                                        stream,
                                        mr);
  auto any_invalid = cudf::reduce(invalid->view(),
                                  *cudf::make_any_aggregation<cudf::reduce_aggregation>(),
                                  cudf::data_type{cudf::type_id::BOOL8},
                                  stream,
                                  mr);
  if (any_invalid->is_valid(stream) &&
      static_cast<cudf::numeric_scalar<bool> const&>(*any_invalid).value(stream)) {
    throw duckdb::OutOfRangeException("STDDEV_SAMP is out of range!");
  }
  return output;
}
}  // namespace sirius::op
