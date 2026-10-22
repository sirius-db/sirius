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

#include "planner/distributed_plan.hpp"

#include "data/data_batch_utils.hpp"
#include "exchange/all_gather.hpp"
#include "exchange/exchange_service.hpp"
#include "op/cudf_sort_order.hpp"
#include "op/sirius_physical_exchange_sink.hpp"
#include "op/sirius_physical_exchange_source.hpp"
#include "op/sirius_physical_grouped_aggregate.hpp"
#include "op/sirius_physical_partition.hpp"
#include "op/sirius_physical_sort_sample.hpp"
#include "sirius/exception.hpp"

#include <cudf/column/column_factories.hpp>
#include <cudf/table/table.hpp>

#include <cuda_runtime_api.h>

#include <cucascade/data/data_batch.hpp>
#include <duckdb/planner/expression/bound_reference_expression.hpp>

#include <functional>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace sirius::planner {
namespace {

using op::exchange_spec;
using op::SiriusPhysicalOperatorType;
using T = SiriusPhysicalOperatorType;

/// Where a subtree's rows are.
enum class placement {
  spread,  ///< Across ranks, each rank holding some.
  ranged,  ///< Across ranks, rank i holding the i-th range of a sort order.
  rank0,   ///< On rank 0; what other ranks hold is not part of the result.
};

struct rewriter {
  std::shared_ptr<exchange::exchange_service> service;
  cucascade::memory::memory_space* gpu;
  std::uint64_t query;
  std::uint32_t next_stream{1};
  /// The plan's shape, compared across ranks.
  std::string shape;

  exchange::stream_id stream() { return (query << 20) | next_stream++; }

  op::sirius_physical_operator& expect(op::sirius_physical_operator& node, T type)
  {
    if (node.type != type) {
      throw sirius::internal_exception("distributed plan: expected {} under {}, found {}",
                                       op::SiriusPhysicalOperatorToString(type),
                                       node.get_name(),
                                       op::SiriusPhysicalOperatorToString(node.type));
    }
    return node;
  }

  /// Puts EXCHANGE_SOURCE → EXCHANGE_SINK(@p spec) in @p slot, above what was there.
  op::sirius_physical_exchange_sink& insert(duckdb::unique_ptr<op::sirius_physical_operator>& slot,
                                            exchange_spec spec,
                                            bool ordered)
  {
    auto child       = std::move(slot);
    auto const id    = stream();
    auto const types = child->types;
    auto const rows  = child->estimated_cardinality;
    std::set<exchange::rank_t> senders;
    if (spec.routing != exchange_spec::mode::gather || service->rank() == spec.target) {
      for (exchange::rank_t r = 0; r < service->world_size(); ++r) {
        senders.insert(r);
      }
    }
    shape += "X" + std::to_string(static_cast<int>(spec.routing)) + "/";
    for (auto const key : spec.key_columns) {
      shape += std::to_string(key) + ",";
    }
    auto sink = duckdb::make_uniq<op::sirius_physical_exchange_sink>(
      types, rows, service, id, std::move(spec));
    auto source = duckdb::make_uniq<op::sirius_physical_exchange_source>(
      types, rows, service, id, std::move(senders), ordered);
    if (child->has_physical_overrides()) {
      sink->set_physical_types(child->get_physical_types());
      source->set_physical_types(child->get_physical_types());
    }
    auto& sink_ref = *sink;
    sink->children.push_back(std::move(child));
    source->children.push_back(std::move(sink));
    slot = std::move(source);
    return sink_ref;
  }

  /// A hash exchange on @p partition's keys, between it and its input.
  placement hash_below(op::sirius_physical_operator& partition_node)
  {
    auto& partition  = expect(partition_node, T::PARTITION).Cast<op::sirius_physical_partition>();
    auto const below = visit(partition.children[0]);
    if (below == placement::rank0) { return placement::rank0; }
    exchange_spec spec;
    spec.key_columns    = partition.partition_keys();
    spec.key_cast_types = partition.specification().key_cast_types;
    insert(partition.children[0], std::move(spec), false);
    return placement::spread;
  }

  placement visit(duckdb::unique_ptr<op::sirius_physical_operator>& slot)
  {
    auto& node = *slot;
    shape += op::SiriusPhysicalOperatorToString(node.type) + "(";
    auto const done = [&](placement p) {
      shape += ")";
      return p;
    };
    switch (node.type) {
      case T::GPU_SCAN: return done(placement::spread);
      case T::FILTER:
      case T::PROJECTION:
      // A partial aggregate keeps its input's placement; its merge above decides the exchange.
      case T::HASH_GROUP_BY: return done(visit(node.children[0]));

      case T::HASH_JOIN: {
        // Both sides are CONCAT → PARTITION → input; each side's rows meet their matches once
        // both are exchanged on the join keys.
        for (auto& side : node.children) {
          auto& concat = expect(*side, T::CONCAT);
          if (hash_below(*concat.children[0]) == placement::rank0) {
            throw sirius::not_implemented_exception(
              "distributed execution: a join over a gathered input");
          }
        }
        return done(placement::spread);
      }

      case T::MERGE_GROUP_BY: {
        // MERGE_GROUP_BY → PARTITION → HASH_GROUP_BY: partial results meet by group key.
        auto& partition = *node.children[0];
        auto& grouped   = expect(*partition.children[0], T::HASH_GROUP_BY)
                          .Cast<op::sirius_physical_grouped_aggregate>();
        if (grouped.has_count_distinct) {
          throw sirius::not_implemented_exception(
            "distributed execution: COUNT(DISTINCT) partial results");
        }
        return done(hash_below(partition));
      }

      case T::MERGE_AGGREGATE: {
        // MERGE_AGGREGATE → UNGROUPED_AGGREGATE: partial results gathered to rank 0.
        auto& partial = expect(*node.children[0], T::UNGROUPED_AGGREGATE);
        if (visit(partial.children[0]) == placement::rank0) { return done(placement::rank0); }
        insert(node.children[0], exchange_spec{.routing = exchange_spec::mode::gather}, false);
        return done(placement::rank0);
      }

      case T::MERGE_SORT: {
        // MERGE_SORT → SORT_PARTITION → SORT_SAMPLE → ORDER_BY → input: rows range-exchanged
        // after the sample, on boundaries every rank's sample agrees on.
        auto& partition = expect(*node.children[0], T::SORT_PARTITION);
        auto& sample =
          expect(*partition.children[0], T::SORT_SAMPLE).Cast<op::sirius_physical_sort_sample>();
        auto& order = expect(*sample.children[0], T::ORDER_BY);
        if (visit(order.children[0]) == placement::rank0) { return done(placement::rank0); }
        exchange_spec spec;
        spec.routing = exchange_spec::mode::range;
        for (auto const& o : sample.orders) {
          if (o.expression->GetExpressionClass() != duckdb::ExpressionClass::BOUND_REF) {
            throw sirius::not_implemented_exception(
              "distributed execution: ORDER BY on an expression");
          }
          spec.key_columns.push_back(
            static_cast<int>(o.expression->Cast<duckdb::BoundReferenceExpression>().Index()));
          spec.column_order.push_back(op::to_cudf_order(o.type));
          spec.null_precedence.push_back(op::to_cudf_null_order(o.type, o.null_order));
        }
        auto* sink = &insert(partition.children[0], std::move(spec), false);
        sample.set_distributed({service, stream(), gpu, [sink](auto boundaries) {
                                  sink->set_range_boundaries(std::move(boundaries));
                                }});
        return done(placement::ranged);
      }

      default:
        throw sirius::not_implemented_exception("distributed execution does not support {}",
                                                op::SiriusPhysicalOperatorToString(node.type));
    }
  }
};

/// Fails the query on every rank unless every rank planned the same shape.
void require_same_shape(rewriter& r)
{
  auto const hash = static_cast<std::int64_t>(std::hash<std::string>{}(r.shape));
  auto column     = cudf::make_numeric_column(cudf::data_type{cudf::type_id::INT64}, 1);
  RMM_CUDA_TRY(cudaMemcpy(
    column->mutable_view().data<std::int64_t>(), &hash, sizeof(hash), cudaMemcpyHostToDevice));
  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(std::move(column));
  auto mine      = sirius::make_data_batch(std::make_unique<cudf::table>(std::move(columns)),
                                      *r.gpu,
                                      cudf::get_default_stream(),
                                      telemetry::batch_telemetry_info{});
  auto const all = exchange::all_gather(
    *r.service, (r.query << 20) | 0xfffff, mine, *r.gpu, cudf::get_default_stream());
  for (exchange::rank_t rank = 0; rank < all.size(); ++rank) {
    std::int64_t theirs = 0;
    auto ro             = all[rank]->to_read_only();
    RMM_CUDA_TRY(cudaMemcpy(&theirs,
                            get_cudf_table_view(ro).column(0).data<std::int64_t>(),
                            sizeof(theirs),
                            cudaMemcpyDeviceToHost));
    if (theirs != hash) {
      throw sirius::invalid_input_exception(
        "distributed execution: rank {} planned this query differently from rank {}",
        rank,
        r.service->rank());
    }
  }
}

}  // namespace

void distribute_plan(duckdb::unique_ptr<op::sirius_physical_operator>& plan,
                     std::shared_ptr<exchange::exchange_service> service,
                     cucascade::memory::memory_space& gpu,
                     std::uint64_t query)
{
  rewriter r{std::move(service), &gpu, query};
  auto const where = r.visit(plan);
  if (where != placement::rank0) {
    r.insert(plan,
             exchange_spec{.routing = exchange_spec::mode::gather},
             /*ordered=*/where == placement::ranged);
  }
  require_same_shape(r);
}

}  // namespace sirius::planner
