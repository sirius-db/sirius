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

#include "op/aggregate/clustered_merge_bypass.hpp"

#include "data/data_batch_utils.hpp"
#include "expression_evaluator/expression_evaluator.hpp"
#include "log/logging.hpp"
#include "op/merge/gpu_merge_impl.hpp"
#include "op/partition/gpu_partition_impl.hpp"
#include "op/sirius_physical_partition_consumer_operator.hpp"

#include <cudf/aggregation.hpp>
#include <cudf/ast/expressions.hpp>
#include <cudf/binaryop.hpp>
#include <cudf/column/column.hpp>
#include <cudf/copying.hpp>
#include <cudf/reduction.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/transform.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <cucascade/cudf/gpu_data_representation.hpp>

#include <algorithm>
#include <stdexcept>

namespace sirius {
namespace op {
namespace clustered_bypass {

namespace {

/// Absolute adjacent-overlap width (in keys) always admitted regardless of the fractional gate.
/// Safe by construction: a partial batch is a group-by output (one row per distinct key), so the
/// rows routed to the fix-up re-group by an admitted window are bounded by the window width.
constexpr __int128 kOverlapAbsoluteFloor = 1024;

template <typename T>
__int128 numeric_scalar_value(cudf::scalar& s, rmm::cuda_stream_view stream)
{
  return static_cast<__int128>(static_cast<cudf::numeric_scalar<T>&>(s).value(stream));
}

template <typename T>
__int128 timestamp_scalar_value(cudf::scalar& s, rmm::cuda_stream_view stream)
{
  return static_cast<__int128>(
    static_cast<cudf::timestamp_scalar<T>&>(s).value(stream).time_since_epoch().count());
}

/// Host-widen a key scalar to __int128 (analysis side). Must stay in lockstep with
/// supported_key_type and make_key_scalar.
__int128 host_key_value(cudf::scalar& s, rmm::cuda_stream_view stream)
{
  switch (s.type().id()) {
    case cudf::type_id::INT8: return numeric_scalar_value<int8_t>(s, stream);
    case cudf::type_id::INT16: return numeric_scalar_value<int16_t>(s, stream);
    case cudf::type_id::INT32: return numeric_scalar_value<int32_t>(s, stream);
    case cudf::type_id::INT64: return numeric_scalar_value<int64_t>(s, stream);
    case cudf::type_id::UINT8: return numeric_scalar_value<uint8_t>(s, stream);
    case cudf::type_id::UINT16: return numeric_scalar_value<uint16_t>(s, stream);
    case cudf::type_id::UINT32: return numeric_scalar_value<uint32_t>(s, stream);
    case cudf::type_id::UINT64: return numeric_scalar_value<uint64_t>(s, stream);
    case cudf::type_id::TIMESTAMP_DAYS: return timestamp_scalar_value<cudf::timestamp_D>(s, stream);
    case cudf::type_id::TIMESTAMP_SECONDS:
      return timestamp_scalar_value<cudf::timestamp_s>(s, stream);
    case cudf::type_id::TIMESTAMP_MILLISECONDS:
      return timestamp_scalar_value<cudf::timestamp_ms>(s, stream);
    case cudf::type_id::TIMESTAMP_MICROSECONDS:
      return timestamp_scalar_value<cudf::timestamp_us>(s, stream);
    case cudf::type_id::TIMESTAMP_NANOSECONDS:
      return timestamp_scalar_value<cudf::timestamp_ns>(s, stream);
    default: throw std::logic_error("clustered_bypass: host_key_value on unsupported key type");
  }
}

template <typename T>
std::unique_ptr<cudf::scalar> make_numeric_key_scalar(__int128 value,
                                                      rmm::cuda_stream_view stream,
                                                      rmm::device_async_resource_ref mr)
{
  return std::make_unique<cudf::numeric_scalar<T>>(static_cast<T>(value), true, stream, mr);
}

template <typename T>
std::unique_ptr<cudf::scalar> make_timestamp_key_scalar(__int128 value,
                                                        rmm::cuda_stream_view stream,
                                                        rmm::device_async_resource_ref mr)
{
  using rep = typename T::rep;
  return std::make_unique<cudf::timestamp_scalar<T>>(
    typename T::duration{static_cast<rep>(value)}, true, stream, mr);
}

/// Rebuild a device scalar of @p key_type from a host-widened bound (execute side).
std::unique_ptr<cudf::scalar> make_key_scalar(cudf::data_type key_type,
                                              __int128 value,
                                              rmm::cuda_stream_view stream,
                                              rmm::device_async_resource_ref mr)
{
  switch (key_type.id()) {
    case cudf::type_id::INT8: return make_numeric_key_scalar<int8_t>(value, stream, mr);
    case cudf::type_id::INT16: return make_numeric_key_scalar<int16_t>(value, stream, mr);
    case cudf::type_id::INT32: return make_numeric_key_scalar<int32_t>(value, stream, mr);
    case cudf::type_id::INT64: return make_numeric_key_scalar<int64_t>(value, stream, mr);
    case cudf::type_id::UINT8: return make_numeric_key_scalar<uint8_t>(value, stream, mr);
    case cudf::type_id::UINT16: return make_numeric_key_scalar<uint16_t>(value, stream, mr);
    case cudf::type_id::UINT32: return make_numeric_key_scalar<uint32_t>(value, stream, mr);
    case cudf::type_id::UINT64: return make_numeric_key_scalar<uint64_t>(value, stream, mr);
    case cudf::type_id::TIMESTAMP_DAYS:
      return make_timestamp_key_scalar<cudf::timestamp_D>(value, stream, mr);
    case cudf::type_id::TIMESTAMP_SECONDS:
      return make_timestamp_key_scalar<cudf::timestamp_s>(value, stream, mr);
    case cudf::type_id::TIMESTAMP_MILLISECONDS:
      return make_timestamp_key_scalar<cudf::timestamp_ms>(value, stream, mr);
    case cudf::type_id::TIMESTAMP_MICROSECONDS:
      return make_timestamp_key_scalar<cudf::timestamp_us>(value, stream, mr);
    case cudf::type_id::TIMESTAMP_NANOSECONDS:
      return make_timestamp_key_scalar<cudf::timestamp_ns>(value, stream, mr);
    default: throw std::logic_error("clustered_bypass: make_key_scalar on unsupported key type");
  }
}

/// Emplace a cudf AST literal over @p s (which must outlive the AST evaluation).
const cudf::ast::expression& add_literal(cudf::ast::tree& tree, cudf::scalar& s)
{
  switch (s.type().id()) {
    case cudf::type_id::INT8:
      return tree.emplace<cudf::ast::literal>(static_cast<cudf::numeric_scalar<int8_t>&>(s));
    case cudf::type_id::INT16:
      return tree.emplace<cudf::ast::literal>(static_cast<cudf::numeric_scalar<int16_t>&>(s));
    case cudf::type_id::INT32:
      return tree.emplace<cudf::ast::literal>(static_cast<cudf::numeric_scalar<int32_t>&>(s));
    case cudf::type_id::INT64:
      return tree.emplace<cudf::ast::literal>(static_cast<cudf::numeric_scalar<int64_t>&>(s));
    case cudf::type_id::UINT8:
      return tree.emplace<cudf::ast::literal>(static_cast<cudf::numeric_scalar<uint8_t>&>(s));
    case cudf::type_id::UINT16:
      return tree.emplace<cudf::ast::literal>(static_cast<cudf::numeric_scalar<uint16_t>&>(s));
    case cudf::type_id::UINT32:
      return tree.emplace<cudf::ast::literal>(static_cast<cudf::numeric_scalar<uint32_t>&>(s));
    case cudf::type_id::UINT64:
      return tree.emplace<cudf::ast::literal>(static_cast<cudf::numeric_scalar<uint64_t>&>(s));
    case cudf::type_id::TIMESTAMP_DAYS:
      return tree.emplace<cudf::ast::literal>(
        static_cast<cudf::timestamp_scalar<cudf::timestamp_D>&>(s));
    case cudf::type_id::TIMESTAMP_SECONDS:
      return tree.emplace<cudf::ast::literal>(
        static_cast<cudf::timestamp_scalar<cudf::timestamp_s>&>(s));
    case cudf::type_id::TIMESTAMP_MILLISECONDS:
      return tree.emplace<cudf::ast::literal>(
        static_cast<cudf::timestamp_scalar<cudf::timestamp_ms>&>(s));
    case cudf::type_id::TIMESTAMP_MICROSECONDS:
      return tree.emplace<cudf::ast::literal>(
        static_cast<cudf::timestamp_scalar<cudf::timestamp_us>&>(s));
    case cudf::type_id::TIMESTAMP_NANOSECONDS:
      return tree.emplace<cudf::ast::literal>(
        static_cast<cudf::timestamp_scalar<cudf::timestamp_ns>&>(s));
    default: throw std::logic_error("clustered_bypass: add_literal on unsupported key type");
  }
}

/// One partial batch's observed key range (host-widened, inclusive).
struct batch_range {
  uint64_t batch_id;
  __int128 lo;
  __int128 hi;
};

/// Re-group @p parts (partial-aggregate shaped batches whose keys are unique WITHIN each part)
/// with the merge combine, hash-partitioning first when the total exceeds
/// @p hash_partition_bytes so device memory stays bounded exactly like the normal merge chain.
std::vector<std::shared_ptr<cucascade::data_batch>> regroup_parts(
  const std::vector<std::shared_ptr<cucascade::data_batch>>& parts,
  const std::vector<int>& group_indices,
  const std::vector<cudf::aggregation::Kind>& merge_aggregates,
  uint64_t hash_partition_bytes,
  int num_gpus,
  rmm::cuda_stream_view stream,
  const telemetry::batch_telemetry_info& telemetry_info)
{
  if (parts.empty()) { return {}; }
  // A single part is a group-by output: its keys are already unique, so the re-group would be
  // an identity — emit it as-is.
  if (parts.size() == 1) { return {parts[0]}; }

  std::vector<cucascade::read_only_data_batch> part_ros;
  part_ros.reserve(parts.size());
  uint64_t total_bytes = 0;
  for (const auto& part : parts) {
    auto ro = part->to_read_only();
    if (ro.get_data()) { total_bytes += ro.get_data()->get_size_in_bytes(); }
    part_ros.push_back(std::move(ro));
  }
  auto* space = part_ros[0].get_memory_space();
  if (space == nullptr) {
    throw std::runtime_error("clustered_bypass: re-group part has no memory space");
  }

  int const num_partitions = natural_num_partitions(total_bytes, hash_partition_bytes, num_gpus);
  if (num_partitions <= 1) {
    return {gpu_merge_impl::merge_grouped_aggregate(part_ros,
                                                    static_cast<int>(group_indices.size()),
                                                    merge_aggregates,
                                                    stream,
                                                    *space,
                                                    telemetry_info)};
  }

  // Same key -> same bucket, so each bucket can be combined independently and exactly.
  std::vector<std::vector<std::shared_ptr<cucascade::data_batch>>> buckets(
    static_cast<std::size_t>(num_partitions));
  for (const auto& ro : part_ros) {
    auto pieces = gpu_partition_impl::hash_partition(
      ro, group_indices, num_partitions, stream, *space, telemetry_info);
    if (pieces.size() != static_cast<std::size_t>(num_partitions)) {
      throw std::runtime_error("clustered_bypass: hash_partition returned unexpected piece count");
    }
    for (std::size_t bucket = 0; bucket < pieces.size(); ++bucket) {
      buckets[bucket].push_back(std::move(pieces[bucket]));
    }
  }

  std::vector<std::shared_ptr<cucascade::data_batch>> outputs;
  outputs.reserve(buckets.size());
  for (auto& bucket : buckets) {
    if (bucket.empty()) { continue; }
    if (bucket.size() == 1) {
      outputs.push_back(std::move(bucket[0]));
      continue;
    }
    std::vector<cucascade::read_only_data_batch> bucket_ros;
    bucket_ros.reserve(bucket.size());
    for (const auto& piece : bucket) {
      bucket_ros.push_back(piece->to_read_only());
    }
    outputs.push_back(
      gpu_merge_impl::merge_grouped_aggregate(bucket_ros,
                                              static_cast<int>(group_indices.size()),
                                              merge_aggregates,
                                              stream,
                                              *space,
                                              telemetry_info));
  }
  return outputs;
}

}  // namespace

bool supported_key_type(cudf::data_type t)
{
  switch (t.id()) {
    case cudf::type_id::INT8:
    case cudf::type_id::INT16:
    case cudf::type_id::INT32:
    case cudf::type_id::INT64:
    case cudf::type_id::UINT8:
    case cudf::type_id::UINT16:
    case cudf::type_id::UINT32:
    case cudf::type_id::UINT64:
    case cudf::type_id::TIMESTAMP_DAYS:
    case cudf::type_id::TIMESTAMP_SECONDS:
    case cudf::type_id::TIMESTAMP_MILLISECONDS:
    case cudf::type_id::TIMESTAMP_MICROSECONDS:
    case cudf::type_id::TIMESTAMP_NANOSECONDS: return true;
    default: return false;
  }
}

std::optional<plan> analyze_partial_ranges(
  const std::vector<std::shared_ptr<cucascade::data_batch>>& batches,
  int key_column_index,
  double max_overlap_fraction)
{
  plan out;
  std::vector<batch_range> ranges;
  ranges.reserve(batches.size());
  auto const stream = cudf::get_default_stream();

  for (const auto& batch : batches) {
    if (!batch) { return std::nullopt; }
    auto ro    = batch->to_read_only();
    auto* data = ro.get_data();
    // Only GPU-resident, uncompressed partials are analyzable; a spilled batch means the query
    // is already under memory pressure — take the normal partitioned merge.
    const auto* gpu = dynamic_cast<const cucascade::gpu_table_representation*>(data);
    if (gpu == nullptr) { return std::nullopt; }
    auto view = gpu->get_table_view();
    if (key_column_index < 0 || key_column_index >= view.num_columns()) { return std::nullopt; }
    if (view.num_rows() == 0) {
      // Empty partials carry no keys; remember them so execute-side lookups succeed.
      out.batch_regions.emplace(ro.get_batch_id(), std::vector<int>{});
      continue;
    }
    auto key = view.column(key_column_index);
    if (!supported_key_type(key.type())) { return std::nullopt; }
    if (out.key_type.id() == cudf::type_id::EMPTY) {
      out.key_type = key.type();
    } else if (key.type() != out.key_type) {
      return std::nullopt;  // mixed carriers across batches — no single comparison domain
    }
    // A NULL group key would occur in any batch regardless of ranges (min/max skip nulls),
    // breaking the containment proof — bail.
    if (key.has_nulls()) { return std::nullopt; }
    auto* space = ro.get_memory_space();
    if (space == nullptr) { return std::nullopt; }
    auto mr       = space->get_default_allocator();
    auto min_agg  = cudf::make_min_aggregation<cudf::reduce_aggregation>();
    auto max_agg  = cudf::make_max_aggregation<cudf::reduce_aggregation>();
    auto min_scal = cudf::reduce(key, *min_agg, key.type(), stream, mr);
    auto max_scal = cudf::reduce(key, *max_agg, key.type(), stream, mr);
    if (!min_scal->is_valid(stream) || !max_scal->is_valid(stream)) { return std::nullopt; }
    ranges.push_back(batch_range{
      ro.get_batch_id(), host_key_value(*min_scal, stream), host_key_value(*max_scal, stream)});
    SIRIUS_LOG_DEBUG("clustered_bypass: partial batch id={} rows={} key_range=[{}, {}]",
                     ro.get_batch_id(),
                     view.num_rows(),
                     static_cast<int64_t>(ranges.back().lo),
                     static_cast<int64_t>(ranges.back().hi));
  }

  // With fewer than two non-empty partials the existing single-batch merge fast path is already
  // optimal — nothing to bypass.
  if (ranges.size() < 2) { return std::nullopt; }

  std::sort(ranges.begin(), ranges.end(), [](const batch_range& a, const batch_range& b) {
    return a.lo != b.lo ? a.lo < b.lo : a.hi < b.hi;
  });

  // THE PROOF (see header): with ranges sorted by (lo, hi),
  //  - every non-adjacent pair must be strictly disjoint (mins are sorted, so max_i < lo_{i+2}
  //    implies max_i < lo_j for all j > i+1);
  //  - adjacent pairs may share a small window (a clustered scan splits at most a handful of
  //    keys across a batch boundary). The window sizes gate profitability, not correctness.
  for (std::size_t i = 0; i + 2 < ranges.size(); ++i) {
    if (ranges[i].hi >= ranges[i + 2].lo) {
      SIRIUS_LOG_DEBUG(
        "clustered_bypass: rejected — non-adjacent range overlap (sorted idx {} hi={} >= idx {} "
        "lo={})",
        i,
        static_cast<int64_t>(ranges[i].hi),
        i + 2,
        static_cast<int64_t>(ranges[i + 2].lo));
      return std::nullopt;
    }
  }
  for (std::size_t i = 0; i + 1 < ranges.size(); ++i) {
    if (ranges[i].hi < ranges[i + 1].lo) { continue; }  // strictly disjoint neighbors
    __int128 const width = ranges[i].hi - ranges[i + 1].lo + 1;
    __int128 const span =
      std::min(ranges[i].hi - ranges[i].lo, ranges[i + 1].hi - ranges[i + 1].lo) + 1;
    bool const admitted =
      width <= kOverlapAbsoluteFloor ||
      static_cast<double>(width) <= max_overlap_fraction * static_cast<double>(span);
    if (!admitted) {
      SIRIUS_LOG_DEBUG("clustered_bypass: rejected — adjacent overlap too wide (width={}, span={})",
                       static_cast<int64_t>(width),
                       static_cast<int64_t>(span));
      return std::nullopt;
    }
    out.regions.push_back({ranges[i + 1].lo, ranges[i].hi});
  }

  // Map every batch to the regions intersecting its range (at most its two boundary windows,
  // by the adjacency proof above; computed generically anyway).
  for (const auto& range : ranges) {
    auto& relevant = out.batch_regions[range.batch_id];
    for (std::size_t region_idx = 0; region_idx < out.regions.size(); ++region_idx) {
      const auto& r = out.regions[region_idx];
      if (r.hi >= range.lo && r.lo <= range.hi) {
        relevant.push_back(static_cast<int>(region_idx));
      }
    }
  }

  SIRIUS_LOG_INFO(
    "clustered_bypass: armed — {} partial batches proven range-disjoint ({} boundary regions)",
    ranges.size(),
    out.regions.size());
  return out;
}

std::vector<std::shared_ptr<cucascade::data_batch>> execute_bypass(
  const std::vector<cucascade::read_only_data_batch>& batches,
  const plan& bypass_plan,
  const sirius::ast::node* filter_expression,
  const std::vector<int>& group_indices,
  const std::vector<cudf::aggregation::Kind>& merge_aggregates,
  uint64_t hash_partition_bytes,
  int num_gpus,
  rmm::cuda_stream_view stream,
  const telemetry::batch_telemetry_info& telemetry_info)
{
  if (batches.empty()) {
    throw std::runtime_error("clustered_bypass: execute_bypass with no input batches");
  }

  // A batch the range proof never saw (it was not on the partition port at sizing time) voids
  // the per-batch filtering: fall back to re-grouping EVERY input row through the partitioned
  // merge combine — exact and memory-bounded, just without the HAVING pre-drop.
  bool fallback = filter_expression == nullptr;
  for (const auto& ro : batches) {
    if (!bypass_plan.batch_regions.contains(ro.get_batch_id())) {
      fallback = true;
      break;
    }
  }

  std::vector<std::shared_ptr<cucascade::data_batch>> survivor_parts;
  if (!fallback) {
    for (const auto& ro : batches) {
      auto view = get_cudf_table_view(ro);
      if (view.num_rows() == 0) { continue; }
      auto* space = ro.get_memory_space();
      if (space == nullptr) {
        throw std::runtime_error("clustered_bypass: input batch has no memory space");
      }
      auto mr = space->get_default_allocator();

      // The downstream FILTER's predicate, evaluated on the partial rows. Exact because a
      // non-boundary partial row IS the final merged row (see the header proof).
      sirius::expression_evaluator evaluator(filter_expression, mr, stream);
      auto mask_table = evaluator.evaluate(view);
      if (mask_table->num_columns() != 1 ||
          mask_table->get_column(0).type().id() != cudf::type_id::BOOL8) {
        throw std::runtime_error("clustered_bypass: filter predicate did not yield a BOOL8 mask");
      }
      auto keep = std::move(mask_table->release()[0]);

      const auto& relevant = bypass_plan.batch_regions.at(ro.get_batch_id());
      if (!relevant.empty()) {
        // Region membership: keys inside a boundary window are kept unconditionally so the
        // fix-up re-group sees every fragment of a split key. One fused AST pass.
        std::vector<std::unique_ptr<cudf::scalar>> bound_scalars;
        bound_scalars.reserve(relevant.size() * 2);
        cudf::ast::tree tree;
        const auto& key_ref = tree.emplace<cudf::ast::column_reference>(
          static_cast<cudf::size_type>(group_indices.empty() ? 0 : group_indices[0]));
        const cudf::ast::expression* membership = nullptr;
        for (int region_idx : relevant) {
          const auto& region = bypass_plan.regions[static_cast<std::size_t>(region_idx)];
          bound_scalars.push_back(make_key_scalar(bypass_plan.key_type, region.lo, stream, mr));
          const auto& lo_lit = add_literal(tree, *bound_scalars.back());
          bound_scalars.push_back(make_key_scalar(bypass_plan.key_type, region.hi, stream, mr));
          const auto& hi_lit = add_literal(tree, *bound_scalars.back());
          const auto& ge     = tree.emplace<cudf::ast::operation>(
            cudf::ast::ast_operator::GREATER_EQUAL, key_ref, lo_lit);
          const auto& le = tree.emplace<cudf::ast::operation>(
            cudf::ast::ast_operator::LESS_EQUAL, key_ref, hi_lit);
          const auto& in_window =
            tree.emplace<cudf::ast::operation>(cudf::ast::ast_operator::LOGICAL_AND, ge, le);
          membership = membership == nullptr
                         ? &in_window
                         : &tree.emplace<cudf::ast::operation>(
                             cudf::ast::ast_operator::LOGICAL_OR, *membership, in_window);
        }
        auto region_mask = cudf::compute_column(view, *membership, stream, mr);
        // SQL null-or: a NULL predicate result must still keep a boundary row (its final value
        // is decided after the re-group), and NULL-or-FALSE stays NULL, which the gather drops —
        // exactly what the downstream filter would have done.
        keep = cudf::binary_operation(keep->view(),
                                      region_mask->view(),
                                      cudf::binary_operator::NULL_LOGICAL_OR,
                                      cudf::data_type{cudf::type_id::BOOL8},
                                      stream,
                                      mr);
      }

      auto survivors = cudf::apply_boolean_mask(view, keep->view(), stream, mr);
      if (survivors->num_rows() == 0) { continue; }
      survivor_parts.push_back(
        sirius::make_data_batch(std::move(survivors), *space, stream, telemetry_info));
    }
  } else {
    SIRIUS_LOG_DEBUG(
      "clustered_bypass: input batch unknown to the range proof — re-grouping all rows");
    for (const auto& ro : batches) {
      auto view = get_cudf_table_view(ro);
      if (view.num_rows() == 0) { continue; }
      auto* space = ro.get_memory_space();
      if (space == nullptr) {
        throw std::runtime_error("clustered_bypass: input batch has no memory space");
      }
      // Copy through a fresh batch so regroup_parts owns uniform inputs. This path is
      // defensive — it is not reachable when the plan was armed over the same port content the
      // merge drains, which the pipeline ordering guarantees.
      auto copy = std::make_unique<cudf::table>(view, stream, space->get_default_allocator());
      survivor_parts.push_back(
        sirius::make_data_batch(std::move(copy), *space, stream, telemetry_info));
    }
    // Force the re-group below even when the proof found no overlap regions.
    return [&]() {
      auto outputs = regroup_parts(survivor_parts,
                                   group_indices,
                                   merge_aggregates,
                                   hash_partition_bytes,
                                   num_gpus,
                                   stream,
                                   telemetry_info);
      if (outputs.empty()) {
        auto* space = batches[0].get_memory_space();
        auto empty  = cudf::empty_like(get_cudf_table_view(batches[0]));
        outputs.push_back(
          sirius::make_data_batch(std::move(empty), *space, stream, telemetry_info));
      }
      return outputs;
    }();
  }

  std::vector<std::shared_ptr<cucascade::data_batch>> outputs;
  if (bypass_plan.regions.empty()) {
    // Fully disjoint ranges: every surviving partial row is final; no re-group needed.
    outputs = std::move(survivor_parts);
  } else {
    outputs = regroup_parts(survivor_parts,
                            group_indices,
                            merge_aggregates,
                            hash_partition_bytes,
                            num_gpus,
                            stream,
                            telemetry_info);
  }
  if (outputs.empty()) {
    // Nothing survived the pushed-down filter; emit one empty batch of the input schema so the
    // pipeline has a well-formed (0-row) result to run the downstream operators over.
    auto* space = batches[0].get_memory_space();
    auto empty  = cudf::empty_like(get_cudf_table_view(batches[0]));
    outputs.push_back(sirius::make_data_batch(std::move(empty), *space, stream, telemetry_info));
  }
  SIRIUS_LOG_INFO("clustered_bypass: bypass executed over {} partials -> {} output batch(es)",
                  batches.size(),
                  outputs.size());
  return outputs;
}

}  // namespace clustered_bypass
}  // namespace op
}  // namespace sirius
