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
#include "op/sirius_physical_cte.hpp"
#include "op/sirius_physical_delim_join.hpp"
#include "op/sirius_physical_exchange_sink.hpp"
#include "op/sirius_physical_exchange_source.hpp"
#include "op/sirius_physical_grouped_aggregate.hpp"
#include "op/sirius_physical_hash_join.hpp"
#include "op/sirius_physical_nested_loop_join.hpp"
#include "op/sirius_physical_partition.hpp"
#include "op/sirius_physical_sort_sample.hpp"
#include "planner/gpu_admission.hpp"
#include "sirius/exception.hpp"

#include <cudf/column/column_factories.hpp>
#include <cudf/table/table.hpp>

#include <cuda_runtime_api.h>

#include <cucascade/data/data_batch.hpp>
#include <duckdb/planner/expression/bound_reference_expression.hpp>

#include <algorithm>
#include <functional>
#include <limits>
#include <map>
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

/// A join's Bloom filters are sized for its estimated build rows, within these bounds: the same
/// size on every rank, since every rank plans the same estimate.
constexpr std::size_t min_bloom_keys = std::size_t{1} << 16;
constexpr std::size_t max_bloom_keys = std::size_t{1} << 30;
/// Stream ids a join reserves for merging its filters, one per admitted key.
constexpr std::uint32_t max_filter_keys = 64;
/// A join whose estimated build is smaller publishes no filters: its build can be empty on some
/// rank, and an empty build only takes part in the merge once its probe has finished, while the
/// ranks with build rows hold their probe until the merge completes.
constexpr std::size_t min_merged_build_rows = std::size_t{1} << 14;
/// A join's build side is broadcast, and its probe side stays where it is, when the whole build
/// fits this budget on every GPU and sending it to every rank moves fewer bytes than hashing both
/// sides would.
constexpr std::uint64_t max_broadcast_build_bytes = std::uint64_t{1} << 30;
/// Width assumed for a variable-width column when estimating bytes.
constexpr std::uint64_t estimated_var_bytes = 32;

std::uint64_t estimated_bytes(op::sirius_physical_operator const& node)
{
  auto const width = estimate_bytes_per_row(node.types, estimated_var_bytes);
  auto const rows  = static_cast<std::uint64_t>(node.estimated_cardinality);
  return rows != 0 && width > std::numeric_limits<std::uint64_t>::max() / rows
           ? std::numeric_limits<std::uint64_t>::max()
           : rows * width;
}

/// Join types that never emit a build row without a probe match, so a build seen by every rank
/// yields each output row on one rank only.
bool emits_build_rows_only_with_probe(duckdb::JoinType type)
{
  return type != duckdb::JoinType::RIGHT && type != duckdb::JoinType::OUTER &&
         type != duckdb::JoinType::RIGHT_SEMI && type != duckdb::JoinType::RIGHT_ANTI;
}

struct rewriter {
  std::shared_ptr<exchange::exchange_service> service;
  cucascade::memory::memory_space* gpu;
  std::uint64_t query;
  exec::query_lifecycle_registry* lifecycle;
  std::uint32_t next_stream{1};
  /// The plan's shape, compared across ranks.
  std::string shape;
  /// Each CTE_SCAN's rows are where its CTE's materialization put them.
  std::map<op::sirius_physical_operator const*, placement> cte_scans;

  exchange::stream_id stream() { return (query << exchange::query_stream_bits) | next_stream++; }

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
    source->set_query_lifecycle(*lifecycle);
    source->set_memory_space(*gpu);
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

  /// A hash exchange on @p partition's keys, between it and its input. An input on rank 0 alone
  /// (a gathered aggregate) is spread from there; the other ranks' copies of it hold no rows.
  placement hash_below(op::sirius_physical_operator& partition_node,
                       bool null_keys_to_every_rank = false)
  {
    auto& partition = expect(partition_node, T::PARTITION).Cast<op::sirius_physical_partition>();
    visit(partition.children[0]);
    exchange_spec spec;
    spec.key_columns             = partition.partition_keys();
    spec.key_cast_types          = partition.specification().key_cast_types;
    spec.null_keys_to_every_rank = null_keys_to_every_rank;
    insert(partition.children[0], std::move(spec), false);
    return placement::spread;
  }

  /// After hash exchanges a rank's build holds only its share of the keys: a filter from it
  /// alone would drop rows other ranks' builds match, so every rank's Bloom is merged.
  void merge_filters_across_ranks(op::sirius_physical_hash_join& join)
  {
    if (!join.publishes_dynamic_filters()) { return; }
    auto const build_rows = join.children[1]->estimated_cardinality;
    if (build_rows < min_merged_build_rows) {
      join.cancel_dynamic_filter_publication();
      shape += "f";
      return;
    }
    auto const first = stream();
    next_stream += max_filter_keys;
    join.set_distributed_dynamic_filters(
      {.service      = service,
       .first_stream = first,
       .gpu          = gpu,
       .bloom_keys   = std::clamp(build_rows, min_bloom_keys, max_bloom_keys)});
    shape += "F";
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
      // A join-edge dynamic filter: its filter holds every rank's keys (see HASH_JOIN).
      case T::DYNAMIC_FILTER:
      // A partial aggregate keeps its input's placement; its merge above decides the exchange.
      case T::HASH_GROUP_BY: return done(visit(node.children[0]));

      case T::HASH_JOIN: {
        // Both sides are CONCAT → PARTITION → input.
        auto& join        = node.Cast<op::sirius_physical_hash_join>();
        auto& probe       = expect(*expect(*node.children[0], T::CONCAT).children[0], T::PARTITION);
        auto& build       = expect(*expect(*node.children[1], T::CONCAT).children[0], T::PARTITION);
        auto const world  = static_cast<std::uint64_t>(service->world_size());
        auto const bbytes = estimated_bytes(*build.children[0]);
        if (emits_build_rows_only_with_probe(join.join_type) &&
            bbytes <= max_broadcast_build_bytes &&
            bbytes <= estimated_bytes(*probe.children[0]) / std::max<std::uint64_t>(world - 1, 1)) {
          // Every rank gets the whole build and keeps its probe rows. Its build being whole, its
          // Bloom filters hold every key, as on one GPU: nothing to merge.
          shape += "B";
          auto const probe_rows = visit(probe.children[0]);
          visit(build.children[0]);
          insert(
            build.children[0], exchange_spec{.routing = exchange_spec::mode::broadcast}, false);
          return done(probe_rows == placement::rank0 ? placement::rank0 : placement::spread);
        }
        merge_filters_across_ranks(join);
        // Each side's rows meet their matches once both are exchanged on the join keys. A MARK
        // join's NULL build keys match nothing yet turn a missed probe row's mark NULL, so every
        // rank needs them.
        hash_below(probe);
        hash_below(build, join.join_type == duckdb::JoinType::MARK);
        return done(placement::spread);
      }

      case T::LEFT_DELIM_JOIN:
      case T::RIGHT_DELIM_JOIN: {
        // A delim join fans one input out to a side of its inner join (LEFT: the probe, RIGHT:
        // the build) and to the DISTINCT that the DELIM_SCANs replay. That input is hashed on the
        // delim columns before the fan-out, exactly as its join side would be: each rank's
        // DISTINCT then holds keys no other rank holds, and that join side needs no exchange.
        auto& delim = node.Cast<op::sirius_physical_delim_join>();
        if (!delim.join || delim.join->type != T::HASH_JOIN || delim.distinct == nullptr) {
          throw sirius::not_implemented_exception(
            "distributed execution: a delim join without a hash join");
        }
        auto& join  = delim.join->Cast<op::sirius_physical_hash_join>();
        auto& probe = expect(*expect(*join.children[0], T::CONCAT).children[0], T::PARTITION)
                        .Cast<op::sirius_physical_partition>();
        auto& build = expect(*expect(*join.children[1], T::CONCAT).children[0], T::PARTITION)
                        .Cast<op::sirius_physical_partition>();
        bool const left  = node.type == T::LEFT_DELIM_JOIN;
        auto& fanned     = left ? probe : build;
        auto& other      = left ? build : probe;
        auto const& keys = fanned.partition_keys();
        if (std::set<int>(keys.begin(), keys.end()) !=
            std::set<int>(delim.distinct->group_idx.begin(), delim.distinct->group_idx.end())) {
          throw sirius::not_implemented_exception(
            "distributed execution: a delim join whose join keys are not its delim columns");
        }
        visit(node.children[0]);
        exchange_spec spec;
        spec.key_columns    = keys;
        spec.key_cast_types = fanned.specification().key_cast_types;
        insert(node.children[0], std::move(spec), false);
        merge_filters_across_ranks(join);
        hash_below(other);
        return done(placement::spread);
      }

      // A rank's share of its delim join's DISTINCT (see LEFT_DELIM_JOIN).
      case T::DELIM_SCAN: return done(placement::spread);

      case T::NESTED_LOOP_JOIN: {
        // NESTED_LOOP_JOIN → CONCAT → PARTITION → input, probe then build. Every rank keeps its
        // probe rows and gets the whole build side, so each pair meets once; an unmatched build
        // row would be emitted by every rank, so no join type that emits those.
        auto const join_type = node.Cast<op::sirius_physical_nested_loop_join>().join_type;
        if (join_type == duckdb::JoinType::RIGHT || join_type == duckdb::JoinType::OUTER ||
            join_type == duckdb::JoinType::RIGHT_SEMI ||
            join_type == duckdb::JoinType::RIGHT_ANTI) {
          throw sirius::not_implemented_exception("distributed execution: a {} nested-loop join",
                                                  duckdb::JoinTypeToString(join_type));
        }
        auto& probe = expect(*expect(*node.children[0], T::CONCAT).children[0], T::PARTITION);
        auto& build = expect(*expect(*node.children[1], T::CONCAT).children[0], T::PARTITION);
        auto const probe_rows = visit(probe.children[0]);
        visit(build.children[0]);
        insert(build.children[0], exchange_spec{.routing = exchange_spec::mode::broadcast}, false);
        return done(probe_rows == placement::rank0 ? placement::rank0 : placement::spread);
      }

      case T::DENSE_COUNT_JOIN: {
        // DENSE_COUNT_JOIN → PARTITION → input, both sides: exchanged on the join key, a rank
        // counts the preserved keys it holds, every match of which is on it too.
        for (auto& side : node.children) {
          hash_below(*side);
        }
        return done(placement::spread);
      }

      case T::CTE: {
        // CTE → (materialization, consumer): each rank materializes its share, and the CTE_SCANs
        // under the consumer read it where it is.
        auto const materialized = visit(node.children[0]);
        for (auto const& scan : node.Cast<op::sirius_physical_cte>().cte_scans) {
          cte_scans[&scan.get()] = materialized;
        }
        return done(visit(node.children[1]));
      }

      case T::CTE_SCAN: return done(cte_scans.at(&node));

      case T::MERGE_GROUP_BY: {
        // MERGE_GROUP_BY → PARTITION → HASH_GROUP_BY: partial results meet by group key.
        auto& partition = *node.children[0];
        auto& grouped   = expect(*partition.children[0], T::HASH_GROUP_BY)
                          .Cast<op::sirius_physical_grouped_aggregate>();
        if (grouped.has_count_distinct) {
          // Partial distinct sets are LIST columns, which the exchange does not carry: the input
          // rows meet by group key instead, and each rank's partials hold whole groups.
          if (grouped.has_grouping_sets()) {
            throw sirius::not_implemented_exception(
              "distributed execution: COUNT(DISTINCT) with grouping sets");
          }
          visit(grouped.children[0]);
          exchange_spec spec;
          spec.key_columns = grouped.group_idx;
          insert(grouped.children[0], std::move(spec), false);
          return done(placement::spread);
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

      case T::MERGE_TOP_N: {
        // MERGE_TOP_N → TOP_N: each rank's top rows gathered to rank 0, whose merge keeps the
        // overall top. A row outside its rank's top N is outside the overall top N too.
        auto& partial = expect(*node.children[0], T::TOP_N);
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
                     std::uint64_t query,
                     exec::query_lifecycle_registry& lifecycle)
{
  rewriter r{std::move(service), &gpu, query, &lifecycle};
  auto const where = r.visit(plan);
  if (where != placement::rank0) {
    r.insert(plan,
             exchange_spec{.routing = exchange_spec::mode::gather},
             /*ordered=*/where == placement::ranged);
  }
  require_same_shape(r);
}

std::vector<file_byte_range> byte_ranges_for_rank(std::span<std::uint64_t const> file_bytes,
                                                  std::size_t rank,
                                                  std::size_t world)
{
  std::uint64_t total = 0;
  for (auto const bytes : file_bytes) {
    total += bytes;
  }
  // floor(total * r / world), without the product overflowing.
  auto const cut = [total, world](std::size_t r) {
    return total / world * r + total % world * r / world;
  };
  auto const begin = cut(rank);
  auto const end   = cut(rank + 1);
  std::vector<file_byte_range> mine;
  std::uint64_t offset = 0;
  for (std::size_t f = 0; f < file_bytes.size(); offset += file_bytes[f], ++f) {
    auto const from = std::max(begin, offset);
    auto const to   = std::min(end, offset + file_bytes[f]);
    if (from < to) { mine.push_back({f, from - offset, to - from}); }
  }
  return mine;
}

}  // namespace sirius::planner
