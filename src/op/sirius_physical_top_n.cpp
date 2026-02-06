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

#include "op/sirius_physical_top_n.hpp"

#include "data/data_batch_utils.hpp"
#include "duckdb/common/assert.hpp"
#include "duckdb/execution/expression_executor.hpp"
#include "duckdb/function/create_sort_key.hpp"
#include "duckdb/planner/expression/bound_reference_expression.hpp"
#include "duckdb/planner/filter/dynamic_filter.hpp"
#include "duckdb/storage/data_table.hpp"
#include "log/logging.hpp"
#include "op/sirius_physical_order.hpp"
#include "op/sirius_physical_top_n_merge.hpp"
#include "utils.hpp"

#include <cudf/column/column_factories.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/copying.hpp>
#include <cudf/cudf_utils.hpp>
#include <cudf/sorting.hpp>

#include <rmm/resource_ref.hpp>

#include <cucascade/data/gpu_data_representation.hpp>

#include <algorithm>
#include <memory>

namespace sirius {
namespace op {

namespace {

std::unique_ptr<cudf::table> compute_top_n_table(
  cudf::table_view input,
  duckdb::vector<duckdb::BoundOrderByNode> const& orders,
  duckdb::idx_t limit,
  duckdb::idx_t offset,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref memory_resource)
{
  if (limit == 0 || input.num_rows() == 0) { return duckdb::make_empty_like(input); }
  if (orders.empty()) {
    throw duckdb::InternalException("TopN requires at least one ordering key");
  }

  auto const keep_rows =
    std::min<cudf::size_type>(input.num_rows(), static_cast<cudf::size_type>(offset + limit));
  if (keep_rows == 0) { return duckdb::make_empty_like(input); }

  std::unique_ptr<cudf::table> kept;
  if (orders.size() == 1) {
    auto const& ord = orders[0];
    if (ord.expression->expression_class != duckdb::ExpressionClass::BOUND_REF) {
      throw duckdb::NotImplementedException("TopN only supports bound reference expressions");
    }
    auto const idx =
      static_cast<cudf::size_type>(ord.expression->Cast<duckdb::BoundReferenceExpression>().index);
    if (idx < 0 || idx >= input.num_columns()) {
      throw duckdb::InternalException("TopN order index out of range");
    }

    auto order =
      ord.type == duckdb::OrderType::ASCENDING ? cudf::order::ASCENDING : cudf::order::DESCENDING;
    auto indices = cudf::top_k_order(input.column(idx), keep_rows, order, stream, memory_resource);
    kept         = cudf::gather(
      input, indices->view(), cudf::out_of_bounds_policy::DONT_CHECK, stream, memory_resource);
  } else {
    // Multi-key: fall back to full sort_by_key
    std::vector<cudf::column_view> key_views;
    key_views.reserve(orders.size());
    std::vector<cudf::order> key_orders;
    key_orders.reserve(orders.size());
    std::vector<cudf::null_order> null_orders;
    null_orders.reserve(orders.size());

    for (auto const& ord : orders) {
      if (ord.expression->expression_class != duckdb::ExpressionClass::BOUND_REF) {
        throw duckdb::NotImplementedException("TopN only supports bound reference expressions");
      }
      auto const idx = static_cast<cudf::size_type>(
        ord.expression->Cast<duckdb::BoundReferenceExpression>().index);
      if (idx < 0 || idx >= input.num_columns()) {
        throw duckdb::InternalException("TopN order index out of range");
      }
      key_views.push_back(input.column(idx));
      key_orders.push_back(ord.type == duckdb::OrderType::ASCENDING ? cudf::order::ASCENDING
                                                                    : cudf::order::DESCENDING);
      null_orders.push_back(ord.null_order == duckdb::OrderByNullType::NULLS_FIRST
                              ? cudf::null_order::BEFORE
                              : cudf::null_order::AFTER);
    }

    auto keys_table = cudf::table_view(key_views);
    auto sorted =
      cudf::sort_by_key(input, keys_table, key_orders, null_orders, stream, memory_resource);

    if (keep_rows == sorted->num_rows()) {
      kept = std::move(sorted);
    } else {
      auto slices = cudf::slice(sorted->view(), {0, keep_rows}, stream);
      kept        = std::make_unique<cudf::table>(slices.front(), stream, memory_resource);
    }
  }

  return kept;
}

}  // namespace

sirius_physical_top_n::sirius_physical_top_n(
  duckdb::vector<duckdb::LogicalType> types_p,
  duckdb::vector<duckdb::BoundOrderByNode> orders,
  duckdb::idx_t limit,
  duckdb::idx_t offset,
  duckdb::shared_ptr<duckdb::DynamicFilterData> dynamic_filter_p,
  duckdb::idx_t estimated_cardinality)
  : sirius_physical_operator(
      SiriusPhysicalOperatorType::TOP_N, std::move(types_p), estimated_cardinality),
    orders(std::move(orders)),
    limit(limit),
    offset(offset),
    dynamic_filter(std::move(dynamic_filter_p))
{
}

sirius_physical_top_n::~sirius_physical_top_n() {}

std::vector<std::shared_ptr<cucascade::data_batch>> sirius_physical_top_n::execute(
  const std::vector<std::shared_ptr<cucascade::data_batch>>& input_batches,
  rmm::cuda_stream_view stream)
{
  if (limit == 0) { return {}; }

  std::shared_ptr<cucascade::data_batch> input_batch;
  for (auto const& batch : input_batches) {
    if (batch) {
      if (input_batch) {
        throw duckdb::InternalException("TopN expects a single input batch per execution");
      }
      input_batch = batch;
    }
  }
  if (!input_batch) { return {}; }

  auto* space = input_batch->get_memory_space();
  if (space == nullptr) { return {}; }

  auto input_table =
    input_batch->get_data()->cast<cucascade::gpu_table_representation>().get_table();
  auto output_table =
    compute_top_n_table(input_table, orders, limit, offset, stream, space->get_default_allocator());

  std::vector<std::shared_ptr<cucascade::data_batch>> outputs;
  std::unique_ptr<cucascade::idata_representation> output_data =
    std::make_unique<cucascade::gpu_table_representation>(std::move(output_table), *space);
  outputs.push_back(
    std::make_shared<cucascade::data_batch>(::sirius::get_next_batch_id(), std::move(output_data)));
  return outputs;
}

static duckdb::vector<duckdb::BoundOrderByNode> copy_orders(
  const duckdb::vector<duckdb::BoundOrderByNode>& src)
{
  duckdb::vector<duckdb::BoundOrderByNode> result;
  result.reserve(src.size());
  for (const auto& order : src) {
    result.push_back(order.Copy());
  }
  return result;
}

sirius_physical_top_n_merge::sirius_physical_top_n_merge(sirius_physical_top_n* top_n)
  : sirius_physical_top_n_merge(
      top_n->types,                // copied by value
      copy_orders(top_n->orders),  // deep copy
      top_n->limit,                // primitive
      top_n->offset,               // primitive
      top_n->dynamic_filter,       // shared_ptr - shares ownership (reference count increases)
      top_n->estimated_cardinality)
{
  child_op = top_n;
}

sirius_physical_top_n_merge::sirius_physical_top_n_merge(
  duckdb::vector<duckdb::LogicalType> types_p,
  duckdb::vector<duckdb::BoundOrderByNode> orders,
  duckdb::idx_t limit,
  duckdb::idx_t offset,
  duckdb::shared_ptr<duckdb::DynamicFilterData> dynamic_filter_p,
  duckdb::idx_t estimated_cardinality)
  : sirius_physical_operator(
      SiriusPhysicalOperatorType::MERGE_TOP_N, std::move(types_p), estimated_cardinality),
    orders(std::move(orders)),
    limit(limit),
    offset(offset),
    dynamic_filter(std::move(dynamic_filter_p))
{
}

std::vector<std::shared_ptr<cucascade::data_batch>> sirius_physical_top_n_merge::execute(
  const std::vector<std::shared_ptr<cucascade::data_batch>>& input_batches,
  rmm::cuda_stream_view stream)
{
  if (limit == 0) { return {}; }

  // Use the memory space from the first valid batch (all batches are expected to share the same
  // space in practice).
  cucascade::memory::memory_space* space = nullptr;
  for (auto const& batch : input_batches) {
    if (batch) {
      space = batch->get_memory_space();
      break;
    }
  }
  if (space == nullptr) { return {}; }

  std::vector<std::unique_ptr<cudf::table>> owned_tables;
  std::vector<cudf::table_view> concat_views;
  for (auto const& batch : input_batches) {
    if (!batch) { continue; }
    auto table = std::make_unique<cudf::table>(
      batch->get_data()->cast<cucascade::gpu_table_representation>().get_table(),
      stream,
      space->get_default_allocator());
    concat_views.push_back(table->view());
    owned_tables.push_back(std::move(table));
  }

  if (concat_views.empty()) { return {}; }

  std::unique_ptr<cudf::table> combined;
  if (concat_views.size() == 1) {
    combined =
      std::make_unique<cudf::table>(concat_views.front(), stream, space->get_default_allocator());
  } else {
    combined = cudf::concatenate(concat_views, stream, space->get_default_allocator());
  }

  auto output_table = compute_top_n_table(
    combined->view(), orders, limit, offset, stream, space->get_default_allocator());
  if (output_table->num_rows() <= static_cast<cudf::size_type>(offset)) {
    output_table = duckdb::make_empty_like(output_table->view());
  } else if (offset > 0) {
    auto out_start = static_cast<cudf::size_type>(offset);
    auto out_slices =
      cudf::slice(output_table->view(), {out_start, output_table->num_rows()}, stream);
    output_table =
      std::make_unique<cudf::table>(out_slices.front(), stream, space->get_default_allocator());
  }

  std::vector<std::shared_ptr<cucascade::data_batch>> outputs;
  std::unique_ptr<cucascade::idata_representation> output_data =
    std::make_unique<cucascade::gpu_table_representation>(std::move(output_table), *space);
  outputs.push_back(
    std::make_shared<cucascade::data_batch>(::sirius::get_next_batch_id(), std::move(output_data)));
  return outputs;
}

}  // namespace op
}  // namespace sirius
