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

#include "cudf/cudf_utils.hpp"
#include "data/data_batch_utils.hpp"
#include "exchange/all_gather.hpp"
#include "exchange/exchange_service.hpp"
#include "exchange/wire.hpp"
#include "expression/ast/utils.hpp"
#include "log/logging.hpp"
#include "op/cudf_sort_order.hpp"
#include "op/distributed/bloom_all_gather_merge.hpp"
#include "op/distributed/sample_all_gather_boundaries.hpp"
#include "op/scan/parquet_scan_types.hpp"
#include "op/scan/parquet_schema_mapping.hpp"
#include "op/scan/scan_plan.hpp"
#include "op/scan/sirius_gpu_scan_operator.hpp"
#include "op/sirius_physical_cte.hpp"
#include "op/sirius_physical_delim_join.hpp"
#include "op/sirius_physical_exchange_sink.hpp"
#include "op/sirius_physical_exchange_source.hpp"
#include "op/sirius_physical_filter.hpp"
#include "op/sirius_physical_gpu_values.hpp"
#include "op/sirius_physical_grouped_aggregate.hpp"
#include "op/sirius_physical_hash_join.hpp"
#include "op/sirius_physical_nested_loop_join.hpp"
#include "op/sirius_physical_partition.hpp"
#include "op/sirius_physical_projection.hpp"
#include "op/sirius_physical_sort_sample.hpp"
#include "op/sirius_physical_table_scan.hpp"
#include "planner/connector_registry.hpp"
#include "planner/gpu_admission.hpp"
#include "planner/substrait_scan_ranges.hpp"
#include "sirius/exception.hpp"
#include "sirius_config.hpp"

#include <cudf/aggregation.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/table/table.hpp>

#include <cuda_runtime_api.h>

#include <cucascade/data/data_batch.hpp>
#include <duckdb/planner/expression/bound_reference_expression.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <functional>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <variant>
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

/// A hash join whose estimated build has fewer rows publishes no merged Bloom filters.
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

/// The cuDF types of @p types; a type cuDF cannot hold is left out.
std::vector<cudf::data_type> physical_types_of(duckdb::vector<sirius::logical_type> const& types)
{
  std::vector<cudf::data_type> out;
  for (auto const& type : types) {
    if (auto const cudf_type = get_cudf_type(type)) { out.push_back(*cudf_type); }
  }
  return out;
}

/// The scan output column that column @p column of @p node is, through projections and filters.
std::optional<std::pair<op::scan::sirius_gpu_scan_operator const*, std::size_t>> scan_column(
  op::sirius_physical_operator const& node, std::size_t column)
{
  switch (node.type) {
    case T::GPU_SCAN: return std::pair{&node.Cast<op::scan::sirius_gpu_scan_operator>(), column};
    case T::PROJECTION: {
      auto const& list = node.Cast<op::sirius_physical_projection>().select_list;
      if (column >= list.size() || !list[column]->is_reference()) { return std::nullopt; }
      return scan_column(*node.children[0], list[column]->as_reference().column_index);
    }
    case T::FILTER: {
      auto const& filter = node.Cast<op::sirius_physical_filter>();
      auto const* kept   = std::get_if<std::vector<cudf::size_type>>(&filter.output_columns);
      if (kept == nullptr) { return scan_column(*node.children[0], column); }
      if (column >= kept->size()) { return std::nullopt; }
      return scan_column(*node.children[0], static_cast<std::size_t>((*kept)[column]));
    }
    default: return std::nullopt;
  }
}

/// A row group's [min, max] of one integer column; min > max when the row group has no
/// statistics for it.
using key_range = std::pair<std::int64_t, std::int64_t>;

std::optional<std::int64_t> decode_integer(std::optional<std::vector<std::uint8_t>> const& bytes,
                                           cudf::io::parquet::Type type)
{
  if (!bytes) { return std::nullopt; }
  if (type == cudf::io::parquet::Type::INT64 && bytes->size() == sizeof(std::int64_t)) {
    std::int64_t value = 0;
    std::memcpy(&value, bytes->data(), sizeof(value));
    return value;
  }
  if (type == cudf::io::parquet::Type::INT32 && bytes->size() == sizeof(std::int32_t)) {
    std::int32_t value = 0;
    std::memcpy(&value, bytes->data(), sizeof(value));
    return value;
  }
  return std::nullopt;
}

/// The key ranges of @p column over the row groups of @p file that hold a non-NULL value of it,
/// in file order; nullopt when one of them has no exact min and max.
std::optional<std::vector<key_range>> row_group_key_ranges(
  cudf::io::parquet::FileMetaData const& file, std::string const& column)
{
  if (file.row_groups.empty()) { return std::vector<key_range>{}; }
  auto const leaves = op::scan::detail::leaf_indices_for_column(file, column);
  if (leaves.size() != 1) { return std::nullopt; }
  std::vector<key_range> ranges;
  ranges.reserve(file.row_groups.size());
  for (auto const& row_group : file.row_groups) {
    if (leaves[0] >= row_group.columns.size()) { return std::nullopt; }
    auto const& meta  = row_group.columns[leaves[0]].meta_data;
    auto const& stats = meta.statistics;
    if (row_group.num_rows == 0 || (stats.null_count && *stats.null_count == row_group.num_rows)) {
      continue;
    }
    if (stats.is_min_value_exact == false || stats.is_max_value_exact == false) {
      return std::nullopt;
    }
    auto const lo = decode_integer(stats.min_value ? stats.min_value : stats.min, meta.type);
    auto const hi = decode_integer(stats.max_value ? stats.max_value : stats.max, meta.type);
    if (!lo || !hi || *lo > *hi) { return std::nullopt; }
    ranges.emplace_back(*lo, *hi);
  }
  return ranges;
}

/// Whether @p ranges, sorted, each end at or before the next one starts. A value strictly inside
/// one of them is then in no other.
bool meet_at_most_at_ends(std::vector<key_range>& ranges)
{
  std::sort(ranges.begin(), ranges.end());
  for (std::size_t i = 1; i < ranges.size(); ++i) {
    if (ranges[i - 1].second > ranges[i].first) { return false; }
  }
  return true;
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
  parquet_footer_reader footers;

  exchange::stream_id stream() { return exchange::make_stream_id(query, next_stream++); }

  /// Gives the partial aggregate under @p merge the HAVING @p having above it, when the
  /// condition of complete_group_having holds on every rank for one of its group keys.
  void push_having_to_partials(op::sirius_physical_filter const& having,
                               op::sirius_physical_operator const& merge)
  {
    if (!footers) { return; }  // turned off (operator_params.enable_complete_group_having)
    auto const skip = [](std::string_view why) {
      SIRIUS_LOG_INFO("[complete-group HAVING] not applied: {}", why);
    };
    auto* below = merge.children[0]->children[0].get();
    while (below->type == T::EXCHANGE_SOURCE || below->type == T::EXCHANGE_SINK) {
      below = below->children[0].get();
    }
    if (below->type != T::HASH_GROUP_BY) { return skip("no partial aggregate"); }
    auto& grouped = below->Cast<op::sirius_physical_grouped_aggregate>();
    // A whole group's partial must be its merged value, in the merge's types.
    using K               = cudf::aggregation::Kind;
    bool const final_kind = std::ranges::all_of(grouped.cudf_aggregates, [](K kind) {
      return kind == K::SUM || kind == K::MIN || kind == K::MAX || kind == K::COUNT_VALID ||
             kind == K::COUNT_ALL;
    });
    bool refs_in_output   = true;
    sirius::ast::visit_references(*having.expression, [&](sirius::ast::reference const& ref) {
      refs_in_output = refs_in_output && ref.column_index < grouped.types.size();
    });
    if (!final_kind || !refs_in_output || grouped.has_grouping_sets() || grouped.has_avg ||
        grouped.has_count_distinct || grouped.has_stddev_samp() ||
        std::ranges::any_of(grouped.cudf_aggregate_is_wide_integer_sum, std::identity{})) {
      return skip("an aggregate whose partial is not its final value");
    }
    if (grouped.has_physical_overrides() || merge.has_physical_overrides() ||
        grouped.types != merge.types) {
      return skip("partial and merged types differ");
    }
    for (std::size_t k = 0; k < grouped.group_idx.size(); ++k) {
      auto const type = grouped.types[k].id();
      if (type != sirius::type_id::BIGINT && type != sirius::type_id::INTEGER) { continue; }
      auto const source =
        scan_column(*grouped.children[0], static_cast<std::size_t>(grouped.group_idx[k]));
      if (!source || source->first->types[source->second] != grouped.types[k]) {
        skip("group key not read unchanged from a scan");
        continue;
      }
      auto const* info = dynamic_cast<op::scan::parquet_ingestible_table_info const*>(
        &source->first->get_ingestible().table_info());
      if (info == nullptr || !info->partition_indices.empty() || info->field_mapping_requirements ||
          info->physical_schema) {
        skip("group key not from a plain Parquet scan");
        continue;
      }
      auto const index = info->projection_ids.empty()
                           ? source->second
                           : (source->second < info->projection_ids.size()
                                ? static_cast<std::size_t>(info->projection_ids[source->second])
                                : info->column_ids.size());
      if (index >= info->column_ids.size()) {
        skip("group key not a scan column");
        continue;
      }
      auto const& column = info->column_ids[index];
      if (column.IsRowIdColumn() || column.IsVirtualColumn() || column.HasChildren() ||
          column.GetPrimaryIndex() >= info->names.size()) {
        skip("group key not a stored column");
        continue;
      }
      auto bounds =
        row_group_bounds(info->resolved_file_paths, info->names[column.GetPrimaryIndex()]);
      if (!bounds) { return; }
      auto filter              = std::make_shared<op::complete_group_having>();
      filter->predicate        = sirius::ast::clone(*having.expression);
      filter->key              = static_cast<cudf::size_type>(k);
      filter->row_group_bounds = device_column(
        *bounds, type == sirius::type_id::BIGINT ? cudf::type_id::INT64 : cudf::type_id::INT32);
      grouped.complete_groups = std::move(filter);
      shape += "H";
      return;
    }
    skip("no integer group key");
  }

  /// The sorted mins and maxes of @p column over the row groups of @p files, when on every rank
  /// the row groups of every file of the scan meet at most at the ends of their key ranges.
  std::optional<std::vector<std::int64_t>> row_group_bounds(std::vector<std::string> const& files,
                                                            std::string const& column)
  {
    auto const began = std::chrono::steady_clock::now();
    std::vector<std::optional<std::vector<key_range>>> ranges(files.size());
    std::atomic<std::size_t> next{0};
    auto const read = [&] {
      for (auto i = next++; i < files.size(); i = next++) {
        try {
          if (auto const file = footers(files[i])) {
            ranges[i] = row_group_key_ranges(*file, column);
          }
        } catch (...) {
          ranges[i].reset();
        }
      }
    };
    {
      std::vector<std::jthread> readers(std::min<std::size_t>(16, files.size()));
      for (auto& reader : readers) {
        reader = std::jthread(read);
      }
    }
    // One row per file: its name's hash and its key range. A rank that cannot prove its files
    // sends the empty range (1, 0).
    std::vector<std::array<std::int64_t, 3>> mine;
    std::vector<std::int64_t> bounds;
    for (std::size_t i = 0; i < files.size(); ++i) {
      if (!ranges[i] || !meet_at_most_at_ends(*ranges[i])) {
        mine   = {{0, 1, 0}};
        bounds = {};
        break;
      }
      if (ranges[i]->empty()) { continue; }
      mine.push_back({static_cast<std::int64_t>(std::hash<std::string>{}(files[i])),
                      ranges[i]->front().first,
                      ranges[i]->back().second});
      for (auto const& [lo, hi] : *ranges[i]) {
        bounds.push_back(lo);
        bounds.push_back(hi);
      }
    }
    auto files_ranges = gather_rows(mine);
    bool proven =
      std::ranges::none_of(files_ranges, [](auto const& row) { return row[1] > row[2]; });
    std::sort(files_ranges.begin(), files_ranges.end());
    files_ranges.erase(std::unique(files_ranges.begin(), files_ranges.end()), files_ranges.end());
    std::vector<key_range> hulls;
    for (auto const& row : files_ranges) {
      hulls.emplace_back(row[1], row[2]);
    }
    proven = proven && meet_at_most_at_ends(hulls);
    std::sort(bounds.begin(), bounds.end());
    bounds.erase(std::unique(bounds.begin(), bounds.end()), bounds.end());
    SIRIUS_LOG_INFO(
      "[complete-group HAVING] {}: {} over {} files of {} ranks, {} row-group bounds here, {} ms",
      column,
      proven ? "proven" : "not proven",
      hulls.size(),
      service->world_size(),
      bounds.size(),
      std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() -
                                                            began)
        .count());
    if (!proven) { return std::nullopt; }
    return bounds;
  }

  /// Every rank's @p rows, in rank order.
  std::vector<std::array<std::int64_t, 3>> gather_rows(
    std::vector<std::array<std::int64_t, 3>> const& rows)
  {
    std::vector<std::unique_ptr<cudf::column>> columns;
    for (std::size_t c = 0; c < 3; ++c) {
      std::vector<std::int64_t> values;
      values.reserve(rows.size());
      for (auto const& row : rows) {
        values.push_back(row[c]);
      }
      columns.push_back(device_column(values, cudf::type_id::INT64));
    }
    auto mine = sirius::make_data_batch(std::make_unique<cudf::table>(std::move(columns)),
                                        *gpu,
                                        cudf::get_default_stream(),
                                        telemetry::batch_telemetry_info{});
    auto const all =
      exchange::all_gather(*service, stream(), mine, *gpu, cudf::get_default_stream());
    std::vector<std::array<std::int64_t, 3>> out;
    for (auto const& theirs : all) {
      if (!theirs) { continue; }
      auto ro         = theirs->to_read_only();
      auto const view = get_cudf_table_view(ro);
      auto const n    = static_cast<std::size_t>(view.num_rows());
      std::vector<std::int64_t> values(3 * n);
      for (cudf::size_type c = 0; c < 3 && n != 0; ++c) {
        RMM_CUDA_TRY(cudaMemcpy(values.data() + c * n,
                                view.column(c).data<std::int64_t>(),
                                n * sizeof(std::int64_t),
                                cudaMemcpyDeviceToHost));
      }
      for (std::size_t i = 0; i < n; ++i) {
        out.push_back({values[i], values[n + i], values[2 * n + i]});
      }
    }
    return out;
  }

  static std::unique_ptr<cudf::column> device_column(std::vector<std::int64_t> const& values,
                                                     cudf::type_id type)
  {
    auto const n = static_cast<cudf::size_type>(values.size());
    auto column  = cudf::make_numeric_column(cudf::data_type{type}, n);
    if (n == 0) { return column; }
    if (type == cudf::type_id::INT64) {
      RMM_CUDA_TRY(cudaMemcpy(column->mutable_view().data<std::int64_t>(),
                              values.data(),
                              values.size() * sizeof(std::int64_t),
                              cudaMemcpyHostToDevice));
    } else {
      std::vector<std::int32_t> narrow(values.begin(), values.end());
      RMM_CUDA_TRY(cudaMemcpy(column->mutable_view().data<std::int32_t>(),
                              narrow.data(),
                              narrow.size() * sizeof(std::int32_t),
                              cudaMemcpyHostToDevice));
    }
    return column;
  }

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
    // Refused before any rank runs: the wire would refuse the first batch. Grouped STDDEV_SAMP
    // partials are STRUCT columns, an ungrouped COUNT(DISTINCT) partial a LIST.
    for (auto const& type : slot->has_physical_overrides() ? slot->get_physical_types()
                                                           : physical_types_of(slot->types)) {
      if (!exchange::sendable(type)) {
        throw sirius::not_implemented_exception(
          "distributed execution: an exchange of a column of cuDF type id {}",
          static_cast<std::int32_t>(type.id()));
      }
    }
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
    // A right semi or anti join publishes before its probe side drains; a merge would make that
    // wait on every rank's build.
    if (build_rows < min_merged_build_rows || join.join_type == duckdb::JoinType::RIGHT_SEMI ||
        join.join_type == duckdb::JoinType::RIGHT_ANTI) {
      join.cancel_dynamic_filter_publication();
      shape += "f";
      return;
    }
    // One stream per admitted key: key k merges on first + k. The keys are fixed while planning,
    // so every rank reserves the same streams.
    auto const first = exchange::make_stream_id(query, next_stream);
    next_stream += static_cast<std::uint32_t>(join.dynamic_filter_plan().admitted_keys().size());
    install_filter_merge(join,
                         std::make_shared<op::bloom_all_gather_merge>(
                           service, first, join.dynamic_filter_plan().admitted_keys().size(), *gpu),
                         service->world_size());
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
      case T::FILTER: {
        auto const where = visit(node.children[0]);
        if (node.children[0]->type == T::MERGE_GROUP_BY) {
          push_having_to_partials(node.Cast<op::sirius_physical_filter>(), *node.children[0]);
        }
        return done(where);
      }
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
        // A BUILD_FILTER (right semi and anti joins only, which publish nothing distributed)
        // would publish this rank's keys alone, before the exchange: it goes.
        if (build.children[0]->type == T::BUILD_FILTER) {
          auto input        = std::move(build.children[0]->children[0]);
          build.children[0] = std::move(input);
        }
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
        // rank needs them; its sink also sends every rank one build row, so that a NULL probe
        // key's mark is NULL on any rank once the build has rows.
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

      // A literal table (VALUES, or a count DuckDB took from Parquet metadata): every rank planned
      // the same rows, so rank 0 keeps them and the other ranks emit none.
      case T::GPU_VALUES: {
        if (service->rank() != 0) {
          auto none = duckdb::make_uniq<op::sirius_physical_gpu_values>(
            node.types, node.estimated_cardinality, /*produce_single_row=*/false);
          if (node.has_physical_overrides()) {
            none->set_physical_types(node.get_physical_types());
          }
          slot = std::move(none);
        }
        return done(placement::rank0);
      }

      case T::NESTED_LOOP_JOIN: {
        // NESTED_LOOP_JOIN → CONCAT → PARTITION → input, probe then build. Every rank keeps its
        // probe rows and gets the whole build side, so each pair meets once; an unmatched build
        // row would be emitted by every rank, so no join type that emits those.
        auto const join_type = node.Cast<op::sirius_physical_nested_loop_join>().join_type;
        if (!emits_build_rows_only_with_probe(join_type)) {
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
        // Its own statement: the exchange takes its stream id before the all-gather takes one.
        auto& sink = insert(partition.children[0], std::move(spec), false);
        // The sample, now below the sink, takes its boundaries from every rank's sample.
        install_sort_boundaries(sample, sink, service, stream(), *gpu);
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
  auto const hash   = static_cast<std::int64_t>(std::hash<std::string>{}(r.shape));
  auto const stream = r.gpu->acquire_stream();
  auto column       = cudf::make_numeric_column(cudf::data_type{cudf::type_id::INT64},
                                          1,
                                          cudf::mask_state::UNALLOCATED,
                                          stream,
                                          r.gpu->get_default_allocator());
  RMM_CUDA_TRY(cudaMemcpyAsync(column->mutable_view().data<std::int64_t>(),
                               &hash,
                               sizeof(hash),
                               cudaMemcpyHostToDevice,
                               stream.get()));
  // The exchange sends from its own stream.
  stream.sync();
  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(std::move(column));
  auto mine = sirius::make_data_batch(std::make_unique<cudf::table>(std::move(columns)),
                                      *r.gpu,
                                      stream,
                                      telemetry::batch_telemetry_info{});
  // The query's last stream id: the rewrite numbers its exchanges up from 1.
  auto const all = exchange::all_gather(
    *r.service,
    exchange::make_stream_id(r.query, (std::uint64_t{1} << exchange::query_stream_bits) - 1),
    mine,
    *r.gpu,
    stream.get());
  std::vector<std::int64_t> theirs(all.size());
  // Held until the copies finish, so nothing spills the hashes meanwhile.
  std::vector<decltype(all.front()->to_read_only())> held;
  for (exchange::rank_t rank = 0; rank < all.size(); ++rank) {
    auto const& ro = held.emplace_back(all[rank]->to_read_only());
    RMM_CUDA_TRY(cudaMemcpyAsync(&theirs[rank],
                                 get_cudf_table_view(ro).column(0).data<std::int64_t>(),
                                 sizeof(std::int64_t),
                                 cudaMemcpyDeviceToHost,
                                 stream.get()));
  }
  stream.sync();
  for (exchange::rank_t rank = 0; rank < all.size(); ++rank) {
    if (theirs[rank] != hash) {
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
                     exec::query_lifecycle_registry& lifecycle,
                     parquet_footer_reader footers)
{
  rewriter r{service, &gpu, query, &lifecycle};
  r.footers = std::move(footers);
  try {
    auto const where = r.visit(plan);
    if (where != placement::rank0) {
      r.insert(plan,
               exchange_spec{.routing = exchange_spec::mode::gather},
               /*ordered=*/where == placement::ranged);
    }
    require_same_shape(r);
  } catch (...) {
    // The query is numbered, so a peer that planned it may already wait for this rank in the
    // shape all-gather, and nothing else tells it: no window runs a plan that failed here. This
    // rank never runs the number, so it retires it too. Best effort: the failure itself goes on.
    std::string what = "unknown error";
    try {
      throw;
    } catch (std::exception const& e) {
      what = e.what();
    } catch (...) {
    }
    try {
      service->abort(query, what);
      service->forget(query);
    } catch (...) {
    }
    throw;
  }
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

void assign_files_to_rank(op::scan::parquet_ingestible_table_info& info,
                          std::vector<std::string> files,
                          operator_params const& op_params)
{
  if (!op_params.distributed()) {
    info.resolved_file_paths = std::move(files);
    return;
  }
  // A rank numbers the files of its own share from 0, so their index and a filter on their name
  // would mean different files on different ranks.
  using kind              = op::scan::scan_plan::parquet_virtual_column_kind;
  auto const names_a_file = [&](auto const& column) {
    return column.HasPrimaryIndex() &&
           std::any_of(
             info.virtual_columns.begin(), info.virtual_columns.end(), [&](auto const& v) {
               return v.column_id == column.GetPrimaryIndex() &&
                      (v.kind == kind::FILENAME || v.kind == kind::FILE_INDEX);
             });
  };
  if (std::any_of(info.column_ids.begin(), info.column_ids.end(), names_a_file)) {
    throw sirius::not_implemented_exception(
      "distributed execution: filename or file_index of a table split across ranks");
  }
  auto const world  = op_params.exchange_peers.size();
  auto const rank   = op_params.exchange_rank;
  auto const remote = std::any_of(
    files.begin(), files.end(), [](auto const& f) { return f.find("://") != std::string::npos; });
  if (remote) {
    // TODO(#2377): split remote files by byte range too, as local ones are.
    if (files.size() < world) {
      throw sirius::not_implemented_exception(
        "distributed execution: a remote table of {} files across {} ranks; each rank needs one. "
        "Split it into at least {} files, or read it from local files, which split by byte range",
        files.size(),
        world,
        world);
    }
    for (std::size_t i = rank; i < files.size(); i += world) {
      info.resolved_file_paths.push_back(std::move(files[i]));
    }
    return;
  }
  std::vector<std::uint64_t> bytes;
  bytes.reserve(files.size());
  for (auto const& f : files) {
    bytes.push_back(std::filesystem::file_size(f));
  }
  for (auto const& r : byte_ranges_for_rank(bytes, rank, world)) {
    info.resolved_file_paths.push_back(std::move(files[r.file]));
    info.resolved_file_ranges.push_back({{r.start, r.length}});
  }
}

void check_distributed_scan(op::sirius_physical_table_scan const& scan,
                            connector const& source,
                            duckdb::ClientContext& context)
{
  // Nothing splits a DuckDB table by rank: every rank would scan all of it.
  if (source.kind == op::scan::source_kind::duckdb_native) {
    throw sirius::not_implemented_exception(
      "distributed execution: a DuckDB table; read the table from Parquet files instead");
  }
  // One URI, expanded only when read: nothing splits it, so every rank would read all of it.
  if (scan.function.GetName() == "sirius_read_parquet") {
    throw sirius::not_implemented_exception(
      "distributed execution: read_parquet of one remote URI; every rank would read all of it");
  }
  // The plan's own ranges would replace this rank's share (lower_parquet_scan attaches them).
  if (source.lower == &lower_parquet_scan && context.registered_state &&
      context.registered_state->Get<scan_byte_ranges_state>(scan_byte_ranges_state::kStateKey)) {
    throw sirius::not_implemented_exception(
      "distributed execution: a plan that carries its own byte ranges");
  }
}

void install_filter_merge(op::sirius_physical_hash_join& join,
                          std::shared_ptr<op::cross_rank_filter_merge> merge,
                          std::size_t ranks)
{
  join.dynamic_filter_session().set_cross_rank_merge(
    std::move(merge), op::merged_bloom_policy(join.dynamic_filter_plan().policy(), ranks));
  // finish_input() never waits: a publication not yet started leaves each key's merge, and one
  // already running completes on its own.
  join.children[1]->add_finalize_listener(
    [&session = join.dynamic_filter_session()] { session.finish_input(); });
}

void install_sort_boundaries(op::sirius_physical_sort_sample& sample,
                             op::sirius_physical_exchange_sink& sink,
                             std::shared_ptr<exchange::exchange_service> service,
                             exchange::stream_id stream,
                             cucascade::memory::memory_space& gpu)
{
  sample.set_cross_rank_boundaries(std::make_shared<op::sample_all_gather_boundaries>(
    std::move(service),
    stream,
    gpu,
    sink.spec().column_order,
    sink.spec().null_precedence,
    [&sink](auto boundaries) { sink.set_range_boundaries(std::move(boundaries)); }));
}
}  // namespace sirius::planner
