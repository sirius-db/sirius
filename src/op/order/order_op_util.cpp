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

#include "op/order/order_op_util.hpp"

#include "duckdb/common/exception.hpp"
#include "duckdb/planner/expression/bound_reference_expression.hpp"

namespace sirius {
namespace op {

void build_order_vectors(const duckdb::vector<duckdb::BoundOrderByNode>& orders,
                         const char* operator_name,
                         std::vector<int>& order_key_idx,
                         std::vector<cudf::order>& column_order,
                         std::vector<cudf::null_order>& null_precedence)
{
  order_key_idx.reserve(orders.size());
  column_order.reserve(orders.size());
  null_precedence.reserve(orders.size());

  for (auto const& ord : orders) {
    if (ord.expression->expression_class != duckdb::ExpressionClass::BOUND_REF) {
      throw duckdb::NotImplementedException("{} only supports bound reference expressions",
                                            operator_name);
    }
    auto idx = static_cast<int>(ord.expression->Cast<duckdb::BoundReferenceExpression>().index);
    order_key_idx.push_back(idx);
    column_order.push_back(ord.type == duckdb::OrderType::ASCENDING ? cudf::order::ASCENDING
                                                                    : cudf::order::DESCENDING);
    null_precedence.push_back(ord.null_order == duckdb::OrderByNullType::NULLS_FIRST
                                ? cudf::null_order::BEFORE
                                : cudf::null_order::AFTER);
  }
}

}  // namespace op
}  // namespace sirius
