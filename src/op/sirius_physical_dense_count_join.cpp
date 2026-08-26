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

#include "op/sirius_physical_dense_count_join.hpp"

#include "cudf/cudf_utils.hpp"
#include "data/data_batch_utils.hpp"
#include "log/logging.hpp"
#include "memory/size_arithmetic.hpp"
#include "op/aggregate/dense_count_join_impl.hpp"
#include "pipeline/sirius_meta_pipeline.hpp"
#include "pipeline/sirius_pipeline.hpp"
#include "sirius/exception.hpp"

#include <cudf/aggregation.hpp>
#include <cudf/binaryop.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/copying.hpp>
#include <cudf/groupby.hpp>
#include <cudf/join/join.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/replace.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/unary.hpp>

#include <rmm/aligned.hpp>

#include <nvtx3/nvtx3.hpp>

#include <cucascade/memory/memory_space.hpp>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <limits>
#include <optional>
#include <string_view>
#include <utility>
#include <vector>

namespace sirius::op {

namespace {
/** @brief Byte ceiling under which a dense histogram is admitted.
 *
 * The one fact shared by the dense-versus-sparse gate and no_history_peak_memory_estimate. The
 * estimator has no key domain, so this bound -- not the layout -- is what the two have in common.
 *
 * Both terms are measured over the one partition a task holds, and the budget is not divided by the
 * partition count: hash partitioning gives every task the full key domain, so tasks replicate a
 * full-width histogram rather than splitting one, and a smaller budget would drop the fused path
 * without reclaiming the replication.
 */
[[nodiscard]] std::size_t max_admitted_histogram_bytes(uint64_t max_bins_bytes,
                                                       std::size_t input_logical_bytes) noexcept
{
  // Lossless on the 64-bit targets this builds for.
  return std::min(static_cast<std::size_t>(max_bins_bytes),
                  sirius::memory::saturating_mul(4, input_logical_bytes));
}

/** @brief Whether a representable layout is also desirable: within budget and not sparse.
 *
 * Hash partitioning divides the rows but not the key domain, so a task plans a full-width histogram
 * over its share of the rows. Comparing that width against the rows of this task alone is
 * deliberate: it holds the operator's total slot work to the same multiple of the row count
 * whatever the partition count, which relaxing the row terms by the partition count would forfeit.
 */
[[nodiscard]] bool dense_admits(dense_count_layout const& layout,
                                std::size_t non_null_preserved_rows,
                                std::size_t total_input_rows,
                                std::size_t input_logical_bytes,
                                uint64_t max_bins_bytes) noexcept
{
  using sirius::memory::saturating_mul;
  return layout.total_bytes() <=
           max_admitted_histogram_bytes(max_bins_bytes, input_logical_bytes) &&
         layout.slots() <= saturating_mul(8, non_null_preserved_rows) &&
         layout.slots() <= saturating_mul(2, total_input_rows);
}

// EXCLUDE implements COUNT(col); INCLUDE implements COUNT(*) and preserved-key presence.
std::unique_ptr<cudf::table> sparse_partial_count(cudf::column_view const& keys,
                                                  cudf::column_view const& values,
                                                  cudf::null_policy value_policy,
                                                  rmm::cuda_stream_view stream,
                                                  rmm::device_async_resource_ref mr)
{
  cudf::groupby::groupby gb(cudf::table_view({keys}), cudf::null_policy::EXCLUDE, cudf::sorted::NO);
  std::vector<cudf::groupby::aggregation_request> requests(1);
  requests[0].values = values;
  requests[0].aggregations.push_back(
    cudf::make_count_aggregation<cudf::groupby_aggregation>(value_policy));
  auto [group_keys, results] = gb.aggregate(requests, stream, mr);
  // cuDF groupby COUNT emits size_type (INT32); widen so the partial merge sums in INT64.
  auto count64 =
    cudf::cast(results[0].results[0]->view(), cudf::data_type{cudf::type_id::INT64}, stream, mr);
  auto key_cols = group_keys->release();
  return sirius::make_table(std::move(key_cols[0]), std::move(count64));
}

std::unique_ptr<cudf::table> sparse_merge_pair(std::unique_ptr<cudf::table> lhs,
                                               std::unique_ptr<cudf::table> rhs,
                                               rmm::cuda_stream_view stream,
                                               rmm::device_async_resource_ref mr)
{
  std::vector<cudf::table_view> views{lhs->view(), rhs->view()};
  auto combined = cudf::concatenate(views, stream, mr);
  lhs.reset();
  rhs.reset();

  auto merged = [&] {
    cudf::groupby::groupby gb(
      cudf::table_view({combined->view().column(0)}), cudf::null_policy::EXCLUDE, cudf::sorted::NO);
    std::vector<cudf::groupby::aggregation_request> requests(1);
    requests[0].values = combined->view().column(1);
    requests[0].aggregations.push_back(cudf::make_sum_aggregation<cudf::groupby_aggregation>());
    auto [group_keys, results] = gb.aggregate(requests, stream, mr);
    auto key_cols              = group_keys->release();
    return sirius::make_table(std::move(key_cols[0]), std::move(results[0].results[0]));
  }();
  combined.reset();
  return merged;
}

// Merge in balanced pairs to avoid one all-partials concatenation.
std::unique_ptr<cudf::table> sparse_merge_partials(
  std::vector<std::unique_ptr<cudf::table>> partials,
  cudf::data_type key_type,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  if (partials.empty()) {
    return sirius::make_empty_table({key_type, cudf::data_type{cudf::type_id::INT64}});
  }
  while (partials.size() > 1) {
    std::vector<std::unique_ptr<cudf::table>> next;
    next.reserve(partials.size() / 2 + partials.size() % 2);
    for (std::size_t i = 0; i < partials.size(); i += 2) {
      if (i + 1 == partials.size()) {
        next.push_back(std::move(partials[i]));
      } else {
        auto lhs = std::move(partials[i]);
        auto rhs = std::move(partials[i + 1]);
        next.push_back(sparse_merge_pair(std::move(lhs), std::move(rhs), stream, mr));
      }
    }
    partials = std::move(next);
  }
  return std::move(partials.front());
}

cudf::column_view gather_map_view(rmm::device_uvector<cudf::size_type> const& indices)
{
  return cudf::column_view(cudf::device_span<cudf::size_type const>(indices));
}

// Moving the elements leaves both source vectors at their original size, so the constructor can
// still read the split index after the base class has taken the combined sequence.
std::vector<std::shared_ptr<::cucascade::data_batch>> combine_sides(
  std::vector<std::shared_ptr<::cucascade::data_batch>>& preserved_batches,
  std::vector<std::shared_ptr<::cucascade::data_batch>>& counted_batches)
{
  std::vector<std::shared_ptr<::cucascade::data_batch>> batches;
  batches.reserve(preserved_batches.size() + counted_batches.size());
  auto append = [&batches](std::vector<std::shared_ptr<::cucascade::data_batch>>& source) {
    for (auto& batch : source) {
      batches.push_back(std::move(batch));
    }
  };
  append(preserved_batches);
  append(counted_batches);
  return batches;
}

}  // namespace

dense_count_join_input::dense_count_join_input(
  std::vector<std::shared_ptr<::cucascade::data_batch>> preserved_batches,
  std::vector<std::shared_ptr<::cucascade::data_batch>> counted_batches)
  : partitioned_operator_data(combine_sides(preserved_batches, counted_batches)),
    _preserved_count(preserved_batches.size()),
    _counted_count(counted_batches.size())
{
}

dense_count_join_input::dense_count_join_input(
  std::vector<std::shared_ptr<::cucascade::data_batch>> preserved_batches,
  std::vector<std::shared_ptr<::cucascade::data_batch>> counted_batches,
  std::size_t partition_idx)
  : partitioned_operator_data(combine_sides(preserved_batches, counted_batches), partition_idx),
    _preserved_count(preserved_batches.size()),
    _counted_count(counted_batches.size())
{
}

sirius_physical_dense_count_join::sirius_physical_dense_count_join(
  duckdb::vector<sirius::logical_type> types,
  std::size_t estimated_cardinality,
  std::size_t preserved_key_idx,
  std::size_t counted_key_idx,
  std::optional<std::size_t> counted_value_idx,
  uint64_t max_bins_bytes,
  uint64_t hash_partition_bytes)
  : sirius_physical_partition_consumer_operator(
      SiriusPhysicalOperatorType::DENSE_COUNT_JOIN, std::move(types), estimated_cardinality),
    _preserved_key_idx(preserved_key_idx),
    _counted_key_idx(counted_key_idx),
    _counted_value_idx(counted_value_idx),
    _max_bins_bytes(max_bins_bytes)
{
  _hash_partition_bytes = hash_partition_bytes;
  D_ASSERT(this->types.size() == 2);  // [group key, BIGINT count]
}

std::string sirius_physical_dense_count_join::params_to_string() const
{
  return " (preserved_key=" + std::to_string(_preserved_key_idx) +
         ", counted_key=" + std::to_string(_counted_key_idx) +
         (_counted_value_idx ? ", count_col=" + std::to_string(*_counted_value_idx)
                             : std::string(", count_star")) +
         ", max_bins_bytes=" + std::to_string(_max_bins_bytes) + ")";
}

std::string_view sirius_physical_dense_count_join::input_port_for(
  sirius_physical_operator const& producer) const
{
  if (children[0].get() == &producer) { return PRESERVED_PORT; }
  if (children[1].get() == &producer) { return COUNTED_PORT; }
  throw sirius::internal_exception(
    "DENSE_COUNT_JOIN repository wiring source is not a direct child");
}

MemoryBarrierType sirius_physical_dense_count_join::input_barrier_for(
  sirius_physical_operator const& /*producer*/) const
{
  return MemoryBarrierType::FULL;
}

void sirius_physical_dense_count_join::build_pipelines(
  pipeline::sirius_pipeline& current, pipeline::sirius_meta_pipeline& meta_pipeline)
{
  // Each child reports is_sink() because this operator is its tree parent, so its own
  // build_pipelines terminates a producer pipeline that feeds one input port. Counted first,
  // mirroring the hash join's build-before-probe order.
  D_ASSERT(children.size() == 2);
  auto& sink_meta    = meta_pipeline.create_child_meta_pipeline(current, *this);
  auto& host_current = *sink_meta.get_base_pipeline();
  for (auto* child : {children[1].get(), children[0].get()}) {
    D_ASSERT(child->is_sink());
    child->build_pipelines(host_current, sink_meta);
  }
  // Inputs arrive only through ports: this pipeline is exactly [DENSE_COUNT_JOIN].
  D_ASSERT(host_current.get_operators().size() == 1);
}

std::size_t sirius_physical_dense_count_join::partition_count() const
{
  std::size_t count = 1;
  for (auto const& [_, input_port] : ports) {
    if (input_port->repo != nullptr) {
      count = std::max(count, input_port->repo->num_partitions());
    }
  }
  return count;
}

partition_strategy sirius_physical_dense_count_join::get_partition_strategy(
  const partition_sizing_input& in)
{
  // A task of this operator holds one partition of BOTH sides at once.
  // Additionally, the usage of a sirius_physical_dense_count_join operator is limited to narrow
  // tables and the output is expected to be smaller due to the aggregating nature of the operator,
  // so we want a larger target partition size.
  uint64_t const bytes_per_partition = sirius::memory::saturating_mul(6, _hash_partition_bytes);
  if (bytes_per_partition == 0) {
    throw sirius::internal_exception("dense_count_join: hash_partition_bytes must be non-zero");
  }
  uint64_t const total_bytes = in.combined_total_bytes;

  auto num_partitions =
    static_cast<int>(std::max(uint64_t{1},
                              total_bytes / bytes_per_partition +
                                static_cast<uint64_t>(total_bytes % bytes_per_partition != 0)));

  int const min_parts = partition_min_num_partitions(_num_gpus);
  // For multi-gpu execution, there is a balance between doing any partitioning (added compute) vs
  // distributing the load across more GPUs. This is a rough heuristic threshold to determine if we
  // would rather share the load or avoid partitioning which adds compute.
  if (min_parts > 1 && total_bytes >= _hash_partition_bytes) {
    num_partitions = std::max(num_partitions, min_parts);
  }

  // Pre-size both input repositories so every partition slot exists before batches arrive.
  // Guarded on strictly-greater to respect the repository's set_num_partitions contract.
  if (num_partitions > 1) {
    std::lock_guard<std::mutex> guard(lock);
    for (auto& [_, input_port] : ports) {
      if (input_port->repo != nullptr &&
          static_cast<std::size_t>(num_partitions) > input_port->repo->num_partitions()) {
        input_port->repo->set_num_partitions(static_cast<std::size_t>(num_partitions));
      }
    }
  }
  return {num_partitions, /*broadcast=*/false, /*build_probe=*/false};
}

std::optional<task_creation_hint> sirius_physical_dense_count_join::get_next_task_hint()
{
  std::lock_guard<std::mutex> guard(lock);
  if (ports.empty()) { return std::nullopt; }

  // Either input may be empty, but both producers must finish before any task runs: the partition
  // count is negotiated from the two sides combined, and a task owns a partition of each.
  for (auto const& p : _ports_list) {
    if (p->src_pipeline && !p->src_pipeline->is_pipeline_finished()) {
      auto* producer = &(p->src_pipeline->get_operators()[0].get());
      return task_creation_hint{TaskCreationHint::WAITING_FOR_INPUT_DATA, producer};
    }
  }
  if (_current_partition_index < partition_count()) {
    return task_creation_hint{TaskCreationHint::READY, this};
  }
  return std::nullopt;
}

std::unique_ptr<operator_data> sirius_physical_dense_count_join::get_next_task_input_data()
{
  // Concurrent task creation can reach this for the same operator; the partition cursor and the
  // two repositories must advance together or two tasks would share a partition.
  std::lock_guard<std::mutex> guard(lock);

  auto* preserved_port = get_port(PRESERVED_PORT);
  auto* counted_port   = get_port(COUNTED_PORT);
  if (preserved_port == nullptr || counted_port == nullptr) { return nullptr; }

  auto drain = [](port* input_port,
                  std::size_t partition_idx,
                  std::vector<std::shared_ptr<::cucascade::data_batch>>& destination) {
    if (input_port->repo == nullptr) { return; }
    while (auto batch = input_port->repo->pop_next_data_batch(partition_idx)) {
      destination.push_back(std::move(batch));
    }
  };

  auto const num_partitions = partition_count();
  // Skip forward over empty partitions rather than reporting exhaustion: returning nullptr here
  // ends task creation, which would drop every partition after the first empty one.
  while (_current_partition_index < num_partitions) {
    auto const this_partition = _current_partition_index++;
    std::vector<std::shared_ptr<::cucascade::data_batch>> preserved_batches;
    std::vector<std::shared_ptr<::cucascade::data_batch>> counted_batches;
    drain(preserved_port, this_partition, preserved_batches);
    drain(counted_port, this_partition, counted_batches);
    if (preserved_batches.empty() && counted_batches.empty()) { continue; }

    // A single partition imposes no cross-task device agreement, so leave it untagged and let the
    // scheduler place it on whichever GPU already holds the data. With more than one, every task
    // of a partition must share a device, which the partition index pins.
    if (num_partitions == 1) {
      return std::make_unique<dense_count_join_input>(std::move(preserved_batches),
                                                      std::move(counted_batches));
    }
    return std::make_unique<dense_count_join_input>(
      std::move(preserved_batches), std::move(counted_batches), this_partition);
  }
  return nullptr;
}

std::size_t sirius_physical_dense_count_join::no_history_peak_memory_estimate(
  input_stats const& stats) const
{
  // 3 possible peaks:
  // - dense execution
  // - sparse execution
  // - initial min/max calculation
  // non-overlapping so max over peaks
  using sirius::memory::saturating_add;
  using sirius::memory::saturating_mul;

  constexpr std::size_t allocation_floor = 1024 * 1024;
  constexpr auto allocation_alignment    = rmm::CUDA_ALLOCATION_ALIGNMENT;

  auto const aligned_charge = [](std::size_t bytes) {
    if (bytes == 0) { return bytes; }
    auto const padded = saturating_add(bytes, static_cast<std::size_t>(allocation_alignment - 1));
    if (padded == std::numeric_limits<std::size_t>::max()) { return padded; }
    return (padded / allocation_alignment) * allocation_alignment;
  };

  auto const histogram_bytes = max_admitted_histogram_bytes(_max_bins_bytes, stats.bytes);

  auto const key_width      = sirius::get_cudf_type(types[0]).id() == cudf::type_id::INT32
                                ? sizeof(int32_t)
                                : sizeof(int64_t);
  auto const cudf_row_limit = static_cast<std::size_t>(std::numeric_limits<cudf::size_type>::max());
  auto const output_rows    = std::min(stats.bytes / key_width, cudf_row_limit);
  auto const selected_bytes =
    saturating_mul(sizeof(int64_t), output_rows);  // histogram bin index per key
  auto const output_bytes =
    saturating_mul(key_width + sizeof(int64_t), output_rows);  // key + COUNT
  auto const mask_bytes = static_cast<std::size_t>(
    cudf::bitmask_allocation_size_bytes(static_cast<cudf::size_type>(output_rows)));

  auto dense_peak = saturating_add(allocation_floor, histogram_bytes);
  dense_peak      = saturating_add(dense_peak, selected_bytes);
  dense_peak      = saturating_add(dense_peak, output_bytes);
  dense_peak      = saturating_add(dense_peak, mask_bytes);
  dense_peak      = saturating_add(dense_peak, histogram_bytes);  // selection/CUB workspace

  // Min/max calculation
  auto const scalar_bytes = saturating_add(aligned_charge(key_width), aligned_charge(sizeof(bool)));
  auto const extrema_per_batch = saturating_mul(2, scalar_bytes);
  auto const minmax_peak =
    saturating_add(allocation_floor, saturating_mul(stats.num_batches, extrema_per_batch));

  // Sparse execution: 16 is a heuristic expansion factor
  auto const sparse_peak = saturating_add(allocation_floor, saturating_mul(16, stats.bytes));
  return std::max({dense_peak, sparse_peak, minmax_peak});
}

std::unique_ptr<operator_data> sirius_physical_dense_count_join::execute(
  operator_data const& input_data, rmm::cuda_stream_view stream)
{
  nvtx3::scoped_range nvtx_range{"sirius_physical_dense_count_join::execute"};
  auto const& input          = dynamic_cast<dense_count_join_input const&>(input_data);
  auto const ro_batches      = input.get_read_only_batches();
  auto const preserved_count = input.preserved_count();
  auto const input_batches   = preserved_count + input.counted_count();
  // Materialization drops unreadable batches, which would shift the split and silently undercount.
  if (ro_batches.size() != input_batches) {
    throw sirius::internal_exception(
      "dense_count_join: {} materialized batches do not match the {} supplied inputs",
      ro_batches.size(),
      input_batches);
  }

  // Task preparation colocates every input in the reservation space, so any batch supplies the
  // allocator.
  D_ASSERT(!ro_batches.empty());
  auto* space = ro_batches.front().get_memory_space();
  auto mr     = space->get_default_allocator();

  auto const key_type   = sirius::get_cudf_type(types[0]);
  auto require_key_type = [&](cudf::column_view const& col, char const* side) {
    if (col.type().id() != key_type.id()) {
      throw sirius::internal_exception(
        "dense_count_join: {} key column carrier {} does not match declared key type {}",
        side,
        static_cast<int32_t>(col.type().id()),
        static_cast<int32_t>(key_type.id()));
    }
  };

  std::vector<cudf::column_view> preserved_keys;
  std::vector<cudf::column_view> counted_keys;
  std::vector<std::optional<cudf::column_view>> counted_values;
  preserved_keys.reserve(ro_batches.size());
  counted_keys.reserve(ro_batches.size());
  counted_values.reserve(ro_batches.size());
  int64_t preserved_rows      = 0;
  int64_t preserved_null_keys = 0;
  int64_t counted_rows        = 0;

  auto const input_logical_bytes = input.get_estimated_size_in_bytes();
  for (std::size_t i = 0; i < ro_batches.size(); ++i) {
    // Rejects a batch that carries no data representation.
    auto const batch_view = sirius::get_cudf_table_view(ro_batches[i]);
    if (i < preserved_count) {
      auto const& col = batch_view.column(static_cast<cudf::size_type>(_preserved_key_idx));
      require_key_type(col, "preserved");
      preserved_rows += col.size();
      preserved_null_keys += col.null_count();
      preserved_keys.push_back(col);
    } else {
      auto const& col = batch_view.column(static_cast<cudf::size_type>(_counted_key_idx));
      require_key_type(col, "counted");
      counted_rows += col.size();
      counted_keys.push_back(col);
      if (_counted_value_idx) {
        counted_values.emplace_back(
          batch_view.column(static_cast<cudf::size_type>(*_counted_value_idx)));
      } else {
        counted_values.emplace_back(std::nullopt);
      }
    }
  }

  auto const semantics = dense_count_semantics::for_count_star(!_counted_value_idx.has_value());
  dense_count_bounds const bounds{preserved_rows, semantics.max_matched(counted_rows)};
  auto const non_null_keys = preserved_rows - preserved_null_keys;
  auto const non_null_rows = static_cast<std::size_t>(non_null_keys);
  auto const total_rows = sirius::memory::saturating_add(static_cast<std::size_t>(preserved_rows),
                                                         static_cast<std::size_t>(counted_rows));

  std::optional<std::pair<int64_t, int64_t>> min_max;
  if (non_null_keys != 0) { min_max = dense_count_global_minmax(preserved_keys, stream, mr); }
  std::optional<dense_count_layout> layout;
  if (min_max) {
    layout =
      dense_count_layout::plan(min_max->first, min_max->second, preserved_rows, counted_rows);
  }

  // NULL is one SQL group, so only one partition may carry null keys. Hash partitioning sends
  // every null key to the same partition; if that stopped holding, the output would carry one NULL
  // group per task carrying them and the row would silently duplicate.
  //
  // Keyed by partition rather than by a bare flag because an OOM'd task is rescheduled with this
  // same input and re-enters execute() from the start: a retry must not read as a second partition.
  if (preserved_null_keys > 0) {
    auto const this_partition = input.get_partition_idx();
    std::lock_guard<std::mutex> guard(lock);
    if (_null_group_partition.has_value() && *_null_group_partition != this_partition) {
      throw sirius::internal_exception(
        "dense_count_join: a second partition received {} NULL preserved keys; null keys must all "
        "land in one partition or the NULL group is emitted more than once",
        preserved_null_keys);
    }
    _null_group_partition = this_partition;
  }

  std::unique_ptr<cudf::table> output;
  auto chosen = strategy::NOT_RUN;
  if (!min_max) {
    chosen = strategy::DENSE;
    output = make_null_group_table(key_type, semantics, preserved_null_keys, stream, mr);
  } else if (layout &&
             dense_admits(
               *layout, non_null_rows, total_rows, input_logical_bytes, _max_bins_bytes)) {
    chosen = strategy::DENSE;
    SIRIUS_LOG_INFO(
      "[dense_count_join] dense path: keys in [{}, {}] (range {}, {}-bit slots), preserved "
      "rows {} (null keys {}), counted rows {}",
      min_max->first,
      min_max->second,
      layout->slots(),
      8 * layout->slot_bytes(),
      preserved_rows,
      preserved_null_keys,
      counted_rows);
    dense_count_state state(*layout, stream, mr);
    for (auto const& col : preserved_keys) {
      state.accumulate_preserved(col, stream);
    }
    for (std::size_t i = 0; i < counted_keys.size(); ++i) {
      state.accumulate_counted(counted_keys[i], counted_values[i], stream);
    }
    output = state.emit(key_type, semantics, preserved_null_keys, bounds, stream, mr);
  } else {
    chosen = strategy::SPARSE;
    if (layout) {
      SIRIUS_LOG_INFO(
        "[dense_count_join] sparse path: keys in [{}, {}], range {}, histogram bytes {}, "
        "input bytes {}, budget {}",
        min_max->first,
        min_max->second,
        layout->slots(),
        layout->total_bytes(),
        input_logical_bytes,
        _max_bins_bytes);
    } else {
      SIRIUS_LOG_INFO(
        "[dense_count_join] sparse path: keys in [{}, {}] span a domain no histogram can "
        "represent, input bytes {}, budget {}",
        min_max->first,
        min_max->second,
        input_logical_bytes,
        _max_bins_bytes);
    }

    std::vector<std::unique_ptr<cudf::table>> counted_partials;
    for (std::size_t i = 0; i < counted_keys.size(); ++i) {
      if (counted_keys[i].size() == 0) { continue; }
      auto const& values = counted_values[i] ? *counted_values[i] : counted_keys[i];
      counted_partials.push_back(
        sparse_partial_count(counted_keys[i], values, semantics.counted_null_policy, stream, mr));
    }
    auto counted_agg = sparse_merge_partials(std::move(counted_partials), key_type, stream, mr);

    // Preserved side: distinct keys with their multiplicity (duplicate preserved keys multiply
    // the per-key match count, matching join-then-group-by semantics).
    std::vector<std::unique_ptr<cudf::table>> preserved_partials;
    for (auto const& col : preserved_keys) {
      if (col.size() == 0) { continue; }
      preserved_partials.push_back(
        sparse_partial_count(col, col, cudf::null_policy::INCLUDE, stream, mr));
    }
    auto preserved_agg = sparse_merge_partials(std::move(preserved_partials), key_type, stream, mr);

    auto const preserved_view          = preserved_agg->view();
    auto const preserved_key_view      = cudf::table_view({preserved_view.column(0)});
    auto const counted_key_view        = cudf::table_view({counted_agg->view().column(0)});
    auto [left_indices, right_indices] = cudf::left_join(
      preserved_key_view, counted_key_view, cudf::null_equality::UNEQUAL, stream, mr);

    // One gather over [group key, preserved multiplicity] keeps both columns row-aligned with the
    // join result.
    auto preserved_joined = cudf::gather(preserved_view,
                                         gather_map_view(*left_indices),
                                         cudf::out_of_bounds_policy::DONT_CHECK,
                                         stream,
                                         mr);
    auto matched          = cudf::gather(cudf::table_view({counted_agg->view().column(1)}),
                                gather_map_view(*right_indices),
                                cudf::out_of_bounds_policy::NULLIFY,
                                stream,
                                mr);
    left_indices.reset();
    right_indices.reset();
    preserved_agg.reset();
    counted_agg.reset();

    auto preserved_columns = preserved_joined->release();
    D_ASSERT(preserved_columns.size() == 2);  // [key, presence]
    auto keys_out = std::move(preserved_columns[0]);
    auto presence = std::move(preserved_columns[1]);

    // A preserved key with no counted match survives the outer join as one row per preserved row,
    // so COUNT(*) fills 1 and COUNT(col) fills 0. Matched groups need no floor: a counted group
    // exists only for a key with at least one counted row, and the COUNT(*) partials are taken
    // with null_policy::INCLUDE, so a matched COUNT(*) count is never below 1.
    cudf::numeric_scalar<int64_t> const unmatched_fill(semantics.unmatched_fill, true, stream, mr);
    auto matched_filled =
      cudf::replace_nulls(matched->view().column(0), unmatched_fill, stream, mr);
    matched.reset();
    if (bounds.may_exceed_bigint()) {
      throw_if_count_product_overflows(presence->view(), matched_filled->view(), stream, mr);
    }
    auto values = cudf::binary_operation(presence->view(),
                                         matched_filled->view(),
                                         cudf::binary_operator::MUL,
                                         cudf::data_type{cudf::type_id::INT64},
                                         stream,
                                         mr);
    presence.reset();
    matched_filled.reset();

    output = sirius::make_table(std::move(keys_out), std::move(values));

    if (preserved_null_keys > 0) {
      // cudf::concatenate rejects a total row count beyond cudf::size_type.
      auto null_group = make_null_group_table(key_type, semantics, preserved_null_keys, stream, mr);
      std::vector<cudf::table_view> parts{output->view(), null_group->view()};
      output = cudf::concatenate(parts, stream, mr);
    }
  }

  _last_strategy.store(chosen, std::memory_order_relaxed);
  SIRIUS_LOG_INFO("[dense_count_join] emitted {} group rows ({} strategy)",
                  output->num_rows(),
                  chosen == strategy::DENSE ? "dense" : "sparse");

  std::vector<std::shared_ptr<::cucascade::data_batch>> results;
  results.push_back(sirius::make_data_batch(std::move(output), *space, stream, batch_telemetry()));
  return std::make_unique<pipelineable_operator_data>(std::move(results));
}

}  // namespace sirius::op
