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

#include "op/sirius_physical_partition.hpp"

#include "config.hpp"
#include "cudf/cudf_utils.hpp"
#include "data/data_batch_utils.hpp"
#include "duckdb/planner/expression/bound_cast_expression.hpp"
#include "duckdb/planner/expression/bound_reference_expression.hpp"
#include "expression/ast/to_duckdb.hpp"
#include "log/logging.hpp"
#include "memory/size_arithmetic.hpp"
#include "op/partition/gpu_partition_impl.hpp"
#include "op/sirius_physical_concat.hpp"
#include "op/sirius_physical_dense_count_join.hpp"
#include "op/sirius_physical_grouped_aggregate_merge.hpp"
#include "op/sirius_physical_hash_join.hpp"
#include "pipeline/sirius_meta_pipeline.hpp"
#include "pipeline/sirius_pipeline.hpp"
#include "sirius_context.hpp"

#include <nvtx3/nvtx3.hpp>

#include <algorithm>
#include <mutex>

namespace sirius {
namespace op {

namespace {

std::optional<std::size_t> extract_bound_ref_index(const duckdb::Expression& expr)
{
  if (expr.GetExpressionClass() == duckdb::ExpressionClass::BOUND_REF) {
    return expr.Cast<duckdb::BoundReferenceExpression>().index;
  }
  if (expr.GetExpressionClass() == duckdb::ExpressionClass::BOUND_CAST) {
    auto& cast_expr = expr.Cast<duckdb::BoundCastExpression>();
    if (cast_expr.child->GetExpressionClass() == duckdb::ExpressionClass::BOUND_REF) {
      return cast_expr.child->Cast<duckdb::BoundReferenceExpression>().index;
    }
  }
  return std::nullopt;
}

constexpr bool producer_requires_full_partition_input(SiriusPhysicalOperatorType type) noexcept
{
  switch (type) {
    case SiriusPhysicalOperatorType::PARTITION:
    case SiriusPhysicalOperatorType::UNGROUPED_AGGREGATE:
    case SiriusPhysicalOperatorType::TOP_N:
    case SiriusPhysicalOperatorType::SORT_PARTITION: return true;
    default: return false;
  }
}

}  // namespace

/// Display name for the basis a partition count was chosen on. Shared by the DEBUG line emitted
/// at the decision and the INFO line emitted at finalize, so the two can never disagree.
const char* sirius_physical_partition::sizing_basis_name(sizing_basis basis)
{
  switch (basis) {
    case sizing_basis::measured: return "measured";
    case sizing_basis::upstream_complete: return "upstream-complete";
    case sizing_basis::projected: return "projected";
  }
  return "unknown";
}

void sirius_physical_partition::record_sizing_basis() const
{
  if (_context_observer == nullptr) { return; }
  switch (_sizing_basis) {
    case sizing_basis::measured: _context_observer->record_partition_sized_from_measured(); return;
    case sizing_basis::upstream_complete:
      _context_observer->record_partition_sized_from_upstream_complete();
      return;
    case sizing_basis::projected:
      _context_observer->record_partition_sized_from_projection();
      return;
  }
}

sirius_physical_partition::sirius_physical_partition(
  duckdb::vector<sirius::logical_type> types,
  std::size_t estimated_cardinality,
  sirius_physical_operator* key_source,
  bool is_build,
  duckdb::SiriusContext* compressed_materialization_observer,
  bool enable_size_estimation,
  double size_estimate_safety_factor)
  : sirius_physical_operator(
      SiriusPhysicalOperatorType::PARTITION, std::move(types), estimated_cardinality),
    _context_observer(compressed_materialization_observer),
    _enable_size_estimation(enable_size_estimation),
    _size_estimate_safety_factor(size_estimate_safety_factor)
{
  _is_build = is_build;
  // Capture partition keys/types from `key_source` and, for joins, the downstream sizing consumer.
  // The tree parent is `_parent_op`, stamped later by `set_parent_ops`.
  get_partition_keys_and_type(key_source, is_build);
  _drives_partition_count = _is_build;
}

std::string sirius_physical_partition::get_name() const { return "PARTITION"; }

bool sirius_physical_partition::is_source() const { return true; }

bool sirius_physical_partition::is_sink() const { return true; }

void sirius_physical_partition::build_pipelines(pipeline::sirius_pipeline& current,
                                                pipeline::sirius_meta_pipeline& meta_pipeline)
{
  // PARTITION is always its own single-operator pipeline. The child is guaranteed to be
  // a sink (its `_parent_op` is this PARTITION, so the base `is_sink()` returns true), so
  // create our own meta and let the child's protocol build its own boundary.
  D_ASSERT(children.size() == 1);
  D_ASSERT(children[0]->is_sink());
  auto& partition_meta = meta_pipeline.create_child_meta_pipeline(current, *this);
  children[0]->build_pipelines(*partition_meta.get_base_pipeline(), partition_meta);
}

void sirius_physical_partition::get_partition_keys_and_type(sirius_physical_operator* op,
                                                            bool is_build)
{
  if (op->type == SiriusPhysicalOperatorType::HASH_JOIN) {
    // For a join, key_source is the join itself, which is also the downstream sizing consumer.
    _downstream_consumer_op = op;
    _partition_type         = PartitionType::HASH;
    auto& hash_join_op      = op->Cast<sirius_physical_hash_join>();
    for (std::size_t cond_idx = 0; cond_idx < hash_join_op.conditions.size(); cond_idx++) {
      auto& condition = hash_join_op.conditions[cond_idx];
      if (condition.comparison != sirius::comparison_type::equal &&
          condition.comparison != sirius::comparison_type::not_distinct_from) {
        continue;
      }
      auto left_owned  = sirius::ast::to_duckdb(*hash_join_op.conditions[cond_idx].left);
      auto right_owned = sirius::ast::to_duckdb(*hash_join_op.conditions[cond_idx].right);
      std::optional<std::size_t> left_index  = extract_bound_ref_index(*left_owned);
      std::optional<std::size_t> right_index = extract_bound_ref_index(*right_owned);
      if (left_index.has_value() && right_index.has_value()) {
        // Determine if a type cast is needed for hash alignment.
        // When the join condition has a BOUND_CAST on one side, the two sides have different
        // physical column types (e.g. INT32 vs INT64). cuDF's murmur3 produces different hash
        // values for the same integer in different representations, so without a cast, matching
        // keys would land in different partitions. We apply the same cast used by the join
        // condition so both sides hash identically.
        const auto& key_expr = is_build ? *right_owned : *left_owned;
        if (is_build) {
          _partition_keys.push_back(right_index.value());
        } else {
          _partition_keys.push_back(left_index.value());
        }
        if (key_expr.GetExpressionClass() == duckdb::ExpressionClass::BOUND_CAST) {
          _partition_key_cast_types.push_back(duckdb::GetCudfType(key_expr.return_type));
        } else {
          _partition_key_cast_types.push_back(cudf::data_type{cudf::type_id::EMPTY});
        }
      }
    }
  } else if (op->type == SiriusPhysicalOperatorType::NESTED_LOOP_JOIN) {
    // NLJ is the downstream sizing consumer too; it always reports a single partition.
    _downstream_consumer_op = op;
    _partition_type         = PartitionType::NONE;
  } else if (op->type == SiriusPhysicalOperatorType::HASH_GROUP_BY) {
    _partition_type            = PartitionType::HASH;
    auto& grouped_aggregate_op = op->Cast<sirius_physical_grouped_aggregate>();
    _partition_keys            = grouped_aggregate_op.get_output_grouping_indices();

    // WSM TODO: this is the original code for getting the partition keys from the grouped aggregate
    // operator which may be what we want to use when we care about grouping sets for (std::size_t
    // i = 0; i < grouped_aggregate_op.groupings.size(); i++) {
    //   auto& grouping = grouped_aggregate_op.groupings[i];
    //   for (auto& group_idx : grouped_aggregate_op.grouping_sets[i]) {
    //     auto& group = grouped_aggregate_op.grouped_aggregate_data.groups[group_idx];
    //     if (group->GetExpressionClass() == duckdb::ExpressionClass::BOUND_REF) {
    //       _partition_keys.push_back(group->Cast<duckdb::BoundReferenceExpression>().index);
    //     }
    //   }
    // }
  } else if (op->type == SiriusPhysicalOperatorType::DENSE_COUNT_JOIN) {
    // The fused count-join is both the key source and the downstream sizing consumer. Its two
    // inputs partition on their respective join keys so equal keys co-locate.
    _downstream_consumer_op = op;
    _partition_type         = PartitionType::HASH;
    auto& dense_count_op    = op->Cast<sirius_physical_dense_count_join>();
    _partition_keys.push_back(static_cast<int>(is_build ? dense_count_op.preserved_key_idx()
                                                        : dense_count_op.counted_key_idx()));
    // No hash-alignment cast: detect_dense_count_join admits the fusion only when both join keys
    // share a LogicalTypeId, so the two sides already hash identically.
    _partition_key_cast_types.push_back(cudf::data_type{cudf::type_id::EMPTY});
  } else if (op->type == SiriusPhysicalOperatorType::MERGE_GROUP_BY) {
    // key_source is the merge itself, which is also the downstream sizing consumer.
    _downstream_consumer_op          = op;
    _partition_type                  = PartitionType::HASH;
    auto& grouped_aggregate_merge_op = op->Cast<sirius_physical_grouped_aggregate_merge>();
    _partition_keys                  = grouped_aggregate_merge_op.get_output_grouping_indices();

  } else {
    // `key_source` must be a key-bearing consumer (HJ/NLJ or HGB/MERGE_GROUP_BY);
    // callers pass the join/group-by directly, never a CONCAT wrapper.
    throw std::runtime_error("Unsupported key_source for partition: " + op->get_name());
  }
}

bool sirius_physical_partition::is_build_partition() const { return _is_build; }
MemoryBarrierType sirius_physical_partition::input_barrier_for(
  sirius_physical_operator const& producer) const
{
  if (producer.type == SiriusPhysicalOperatorType::ORDER_BY) {
    return sirius_physical_operator::input_barrier_for(producer);
  }
  if (producer_requires_full_partition_input(producer.type)) { return MemoryBarrierType::FULL; }

  auto* partition_parent = get_parent_op();
  // An aggregate-fanout PARTITION (parent MERGE_GROUP_BY) is PARTIAL only when estimation is
  // enabled on it, so feature-off keeps the original FULL barrier rather than an emulation of
  // it — and so does the delim-join DISTINCT partition, which sits under a
  // grouped_aggregate_merge (likewise typed MERGE_GROUP_BY) but is never constructed with
  // estimation enabled, and which the type test alone cannot tell apart. Only the
  // producer→PARTITION edge is relaxed; the PARTITION→MERGE_GROUP_BY edge stays FULL, since
  // the merge still needs every bucket.
  bool const aggregate_fanout =
    partition_parent != nullptr &&
    partition_parent->type == SiriusPhysicalOperatorType::MERGE_GROUP_BY;
  if (aggregate_fanout && is_size_estimation_enabled()) { return MemoryBarrierType::PARTIAL; }
  bool const join_feeder =
    partition_parent != nullptr && partition_parent->type == SiriusPhysicalOperatorType::CONCAT;
  auto* join = join_feeder ? partition_parent->get_parent_op() : nullptr;
  bool const right_family_full =
    join != nullptr && join->type == SiriusPhysicalOperatorType::HASH_JOIN &&
    join->Cast<sirius_physical_hash_join>().is_right_family() &&
    !(join->get_parent_op() &&
      join->get_parent_op()->type == SiriusPhysicalOperatorType::RIGHT_DELIM_JOIN);
  if (!is_build_partition() && join_feeder && !right_family_full) {
    return MemoryBarrierType::PARTIAL;
  }
  return sirius_physical_operator::input_barrier_for(producer);
}

std::unique_ptr<operator_data> sirius_physical_partition::execute(const operator_data& input_data,
                                                                  rmm::cuda_stream_view stream)
{
  nvtx3::scoped_range nvtx_range{"sirius_physical_partition::execute"};
  auto& input               = dynamic_cast<const pipelineable_operator_data&>(input_data);
  const auto& input_batches = input.get_read_only_batches();
  if (input_batches.size() != 1) {
    throw std::runtime_error("We expect only one input batch for partition operator " +
                             std::to_string(this->get_operator_id()));
  }
  if (!_num_partitions.has_value()) {
    throw std::runtime_error("Num partitions was not set in sirius_physical_partition operator " +
                             std::to_string(this->get_operator_id()));
  }

  auto const& input_batch_ro = input_batches[0];
  auto* space                = input_batch_ro.get_memory_space();

  // Running tally of everything this partition actually processed; on_finalize_operator
  // compares it against the projection that sized the partition count.
  if (input_batch_ro.get_data()) {
    _actual_bytes.fetch_add(input_batch_ro.get_data()->get_size_in_bytes(),
                            std::memory_order_relaxed);
  }

  // Broadcast mode never hash-partitions: the build side replicates its (small) batch to every
  // slot and the probe side streams through unpartitioned. In both cases execute() just forwards
  // the input batches; the fan-out to slots happens in sink().
  if (_broadcast || _num_partitions.value() < 2 || _partition_keys.empty()) {
    return std::make_unique<pipelineable_operator_data>(input.get_read_only_batches());
  }

  std::vector<std::shared_ptr<cucascade::data_batch>> partitioned_results;
  switch (_partition_type) {
    case PartitionType::HASH:
      // Narrow-passthrough observability: count input columns whose actual carrier is narrower
      // than the native mapping of this operator's logical schema. The counter reads actual batch
      // types, so a regression anywhere in the narrow-carrier chain drops it to zero.
      if (has_physical_overrides() && _context_observer != nullptr) {
        auto const view = get_cudf_table_view(input_batch_ro);
        auto const width =
          std::min<std::size_t>(static_cast<std::size_t>(view.num_columns()), types.size());
        uint64_t narrow_columns = 0;
        for (std::size_t column_idx = 0; column_idx < width; ++column_idx) {
          if (view.column(static_cast<cudf::size_type>(column_idx)).type() !=
              sirius::get_cudf_type(types[column_idx])) {
            ++narrow_columns;
          }
        }
        if (narrow_columns > 0) {
          _context_observer->record_compressed_materialization_partition_narrow_columns(
            narrow_columns);
        }
      }
      partitioned_results = gpu_partition_impl::hash_partition(input_batch_ro,
                                                               _partition_keys,
                                                               _partition_key_cast_types,
                                                               _num_partitions.value(),
                                                               stream,
                                                               *space,
                                                               batch_telemetry());
      break;
    case PartitionType::RANGE:
      throw std::runtime_error("Range partitioning is not implemented yet");
    case PartitionType::EVENLY:
      partitioned_results = gpu_partition_impl::evenly_partition(
        input_batch_ro, _num_partitions.value(), stream, *space, batch_telemetry());
      break;
    case PartitionType::NONE: {
      const auto clone_batch_id = sirius::get_next_batch_id();
      partitioned_results       = {input_batch_ro.clone(
        clone_batch_id,
        stream,
        telemetry::quent_data_batch_probe::create(batch_telemetry(), clone_batch_id))};
      break;
    }
    case PartitionType::CUSTOM:
      throw std::runtime_error("Custom partitioning is not implemented yet");
    default:
      throw std::runtime_error("Unsupported partition type: " +
                               partition_type_to_string(_partition_type));
  }
  return std::make_unique<pipelineable_operator_data>(partitioned_results);
}

void sirius_physical_partition::sink(const operator_data& input_data, rmm::cuda_stream_view stream)
{
  nvtx3::scoped_range nvtx_range{"sirius_physical_partition::sink"};
  auto& pipelineable_input  = dynamic_cast<const pipelineable_operator_data&>(input_data);
  const auto& input_batches = pipelineable_input.get_data_batches();
  (void)stream;  // sink does not use stream for push_data_batch_partitioned

  auto deposit = [&](const std::shared_ptr<cucascade::data_batch>& batch, std::size_t slot) {
    for (auto& next_port_info : next_port_after_sink) {
      // the next operator is a partition consumer operator, so we push the batch into the slot
      auto partition_consumer_op =
        dynamic_cast<sirius_physical_partition_consumer_operator*>(next_port_info.next_operator);
      if (!partition_consumer_op) {
        throw std::runtime_error("Next operator is not a partition consumer operator");
      }
      partition_consumer_op->push_data_batch_partitioned(
        next_port_info.next_operator_port_name, batch, slot);
    }
  };

  if (_broadcast) {
    // Broadcast mode (small build table replicated across GPUs):
    //  - Build side: deposit each (zero-copy shared_ptr) batch into EVERY slot. The executor
    //    peer-clones it onto each slot's GPU on demand at build time, non-destructively.
    //  - Probe side: stream each batch into the slot matching its CURRENT GPU, so the per-GPU
    //    join task finds its probe data already local (no cross-device copy).
    auto const slots = static_cast<std::size_t>(_num_partitions.value());
    for (auto& batch : input_batches) {
      if (_is_build) {
        for (std::size_t slot = 0; slot < slots; ++slot) {
          deposit(batch, slot);
        }
      } else {
        std::size_t slot = 0;
        auto ro          = batch->to_read_only();
        if (auto* ms = ro.get_memory_space(); ms != nullptr) {
          slot = slot_for_device(ms->get_device_id());
        }
        deposit(batch, slot);
      }
    }
    return;
  }

  std::size_t partition_id = 0;
  for (auto& batch : input_batches) {
    deposit(batch, partition_id);
    partition_id++;
  }
}

uint64_t sirius_physical_partition::compute_total_bytes()
{
  if (ports.find("default") == ports.end()) {
    throw std::runtime_error(
      "sirius_physical_partition::compute_total_bytes() did not find default repo for id " +
      std::to_string(this->get_operator_id()));
  }
  auto& repo           = ports.at("default")->repo;
  auto batch_ids       = repo->get_batch_ids(0);
  uint64_t total_bytes = 0;
  for (auto batch_id : batch_ids) {
    auto batch = repo->get_data_batch_by_id(batch_id, 0);
    if (batch) {
      auto ro = batch->to_read_only();
      if (ro.get_data()) { total_bytes += ro.get_data()->get_size_in_bytes(); }
    }
  }
  return total_bytes;
}

std::optional<uint64_t> sirius_physical_partition::estimated_total_input_bytes()
{
  if (!_enable_size_estimation) { return std::nullopt; }
  if (ports.size() != 1) { return std::nullopt; }

  if (!_size_estimate.has_value()) {
    _size_estimate = pipeline::estimate_port_total_input_bytes(*this, ports.begin()->first);
    if (!_size_estimate.has_value()) { return std::nullopt; }
    SIRIUS_LOG_DEBUG(
      "sirius_physical_partition id {} projected {} bytes on its input port (exact={}, hops={}, "
      "samples={})",
      this->get_operator_id(),
      _size_estimate->bytes,
      _size_estimate->exact,
      _size_estimate->hops,
      _size_estimate->ratio_samples);
  }

  // A measured total needs no padding; only a projection does.
  auto const scaled = _size_estimate->exact
                        ? static_cast<double>(_size_estimate->bytes)
                        : static_cast<double>(_size_estimate->bytes) * _size_estimate_safety_factor;
  // Never size for less than what has already landed: an undershooting projection would
  // otherwise produce fewer partitions than the bytes on hand already justify.
  return std::max(static_cast<uint64_t>(scaled), compute_total_bytes());
}

void sirius_physical_partition::set_num_partitions(int num_partitions)
{
  std::lock_guard<std::mutex> guard(lock);
  _num_partitions = num_partitions;
}

void sirius_physical_partition::on_finalize_operator()
{
  // Post-query accuracy report, keyed off _sizing_basis / _sizing_bytes as recorded at the
  // decision — not off _size_estimate, whose raw value differs once the safety factor and the
  // already-arrived floor apply.
  if (!_enable_size_estimation) { return; }
  std::lock_guard<std::mutex> guard(lock);
  auto const actual = _actual_bytes.load(std::memory_order_relaxed);

  if (_sizing_basis == sizing_basis::measured) {
    // No projection when the count was fixed, so an estimated-vs-actual error is meaningless.
    // Nor can one have arrived since: get_next_task_hint() tests
    // `!_num_partitions.has_value()` before calling the estimator, so once the count is set
    // nothing consults it again and _size_estimate stays empty.
    SIRIUS_LOG_INFO(
      "sirius_physical_partition id {} size estimate: sized from measured input ({} bytes); "
      "no projection was available; partitions {}",
      this->get_operator_id(),
      _sizing_bytes,
      _num_partitions.value_or(0));
    return;
  }

  // A projection (or an already-complete upstream total) chose the count, so comparing it
  // against the bytes that actually arrived is meaningful.
  double const error_pct = actual == 0
                             ? 0.0
                             : (static_cast<double>(_sizing_bytes) - static_cast<double>(actual)) /
                                 static_cast<double>(actual) * 100.0;
  SIRIUS_LOG_INFO(
    "sirius_physical_partition id {} size estimate: sized from {} {} bytes (exact={}, hops={}, "
    "samples={}), actual {} bytes, error {:+.1f}%, partitions {}",
    this->get_operator_id(),
    sizing_basis_name(_sizing_basis),
    _sizing_bytes,
    _size_estimate.has_value() && _size_estimate->exact,
    _size_estimate.has_value() ? _size_estimate->hops : 0,
    _size_estimate.has_value() ? _size_estimate->ratio_samples : 0,
    actual,
    error_pct,
    _num_partitions.value_or(0));
}

std::size_t sirius_physical_partition::slot_for_device(int device_id) const
{
  for (std::size_t i = 0; i < _active_gpu_ids.size(); ++i) {
    if (_active_gpu_ids[i] == device_id) { return i; }
  }
  SIRIUS_LOG_WARN(
    "slot_for_device: device_id {} not found in active GPU list, falling back to slot 0",
    device_id);
  return 0;
}

std::optional<task_creation_hint> sirius_physical_partition::get_next_task_hint()
{
  std::lock_guard<std::mutex> guard(lock);
  // Consumers that size from both inputs cannot be measured until both sides have stopped
  // producing, and the driving partition's own readiness says nothing about the sibling's.
  if (_sizing_requires_sibling_input && !_num_partitions.has_value() &&
      _sibling_partition_op != nullptr) {
    auto& sibling = _sibling_partition_op->Cast<sirius_physical_partition>();
    if (sibling.ports.size() == 1) {
      auto sibling_port = sibling.ports.begin()->second;
      if (sibling_port->src_pipeline && !sibling_port->src_pipeline->is_pipeline_finished()) {
        auto* producer = &(sibling_port->src_pipeline->get_operators()[0].get());
        return task_creation_hint{TaskCreationHint::WAITING_FOR_INPUT_DATA, producer};
      }
    }
  }
  if (!_num_partitions.has_value() && !_drives_partition_count &&
      _sibling_partition_op != nullptr) {
    // The non-driver normally waits for the sizing side. If that side finished
    // without input, no task can negotiate a count; elect one partition so the
    // downstream zero-side path can proceed.
    auto& sizing_partition     = _sibling_partition_op->Cast<sirius_physical_partition>();
    bool sizing_finished_empty = false;
    if (sizing_partition.ports.size() == 1) {
      auto sizing_port      = sizing_partition.ports.begin()->second;
      sizing_finished_empty = sizing_port->src_pipeline &&
                              sizing_port->src_pipeline->is_pipeline_finished() &&
                              sizing_port->repo && sizing_port->repo->total_size() == 0;
    }
    if (!sizing_finished_empty) { return _sibling_partition_op->get_next_task_hint(); }
    _num_partitions = 1;
    sizing_partition.set_num_partitions(1);
  }
  if (_num_partitions.has_value() && !_is_build && _sibling_partition_op != nullptr) {
    // If this is part of a join and its on the probe side, and we have determined the number of
    // partitions, we have this behave as a pipeline operator and just schedule tasks

    if (ports.size() != 1) {  // Ensure that it only has one port
      throw std::runtime_error("sirius_physical_concat: there should be only one port");
    }
    auto port_ptr = ports.begin()->second;
    if (port_ptr->repo->total_size() > 0) {
      return task_creation_hint{TaskCreationHint::READY, this};
    } else if (port_ptr->src_pipeline && !port_ptr->src_pipeline->is_pipeline_finished()) {
      auto* producer = &(port_ptr->src_pipeline->get_operators()[0].get());
      return task_creation_hint{TaskCreationHint::WAITING_FOR_INPUT_DATA, producer};
    } else {
      return std::nullopt;
    }
  }

  // Aggregate-fanout partition with estimation enabled — the one case resolve_barrier wires
  // PARTIAL. Tasks may start before the producer drains, but only once the count is pinned:
  // it is fixed for the operator's life, since re-sizing would route later batches to slots
  // computed under a different modulus. Until a projection exists, reproduce the FULL wait.
  //
  // Guarded on _enable_size_estimation because that is when this operator is the authority;
  // with the feature off the port is still FULL and the base hint below enforces the wait.
  if (_enable_size_estimation && _sibling_partition_op == nullptr && !_num_partitions.has_value() &&
      !ports.empty() && !estimated_total_input_bytes().has_value()) {
    auto* port_ptr = ports.begin()->second;
    if (port_ptr->src_pipeline && !port_ptr->src_pipeline->is_pipeline_finished()) {
      auto* producer = &(port_ptr->src_pipeline->get_operators()[0].get());
      return task_creation_hint{TaskCreationHint::WAITING_FOR_INPUT_DATA, producer};
    }
  }

  return sirius_physical_operator::get_next_task_hint();
}

std::unique_ptr<operator_data> sirius_physical_partition::get_next_task_input_data()
{
  // Detect whether either partition has a build-side CONCAT downstream that can fold the build side
  // into a single batch. BUILD_PROBE mode requires exactly one build batch at runtime; the only
  // mechanism that guarantees it is the build-side CONCAT with concat_all enabled. The consumer
  // needs this fact to decide BUILD_PROBE eligibility, and it is partition-side wiring the consumer
  // cannot see, so we compute it here and pass it in.
  auto has_build_concat = [](sirius_physical_operator& part_op) {
    for (auto& next_port : part_op.get_next_ports_after_sink()) {
      if (next_port.next_operator->type != SiriusPhysicalOperatorType::CONCAT) { continue; }
      auto& concat = next_port.next_operator->Cast<sirius_physical_concat>();
      if (concat.is_build_concat()) { return true; }
    }
    return false;
  };
  // Once BUILD_PROBE is chosen, the build-side CONCAT must be told to fold all build batches into
  // one. Either sibling may run the decision first, so both configure the build-side CONCAT.
  // Returns whether a build-side CONCAT was found, so the caller can enforce the BUILD_PROBE
  // invariant: BUILD_PROBE needs exactly one folded build batch at runtime, which only a
  // concat_all'd build-side CONCAT guarantees.
  auto enable_build_concat_all = [](sirius_physical_operator& part_op) {
    bool found = false;
    for (auto& next_port : part_op.get_next_ports_after_sink()) {
      if (next_port.next_operator->type != SiriusPhysicalOperatorType::CONCAT) { continue; }
      auto& concat = next_port.next_operator->Cast<sirius_physical_concat>();
      if (concat.is_build_concat()) {
        concat.set_concat_all(true);
        found = true;
      }
    }
    return found;
  };

  auto* consumer =
    dynamic_cast<sirius_physical_partition_consumer_operator*>(_downstream_consumer_op);
  if (consumer == nullptr) {
    throw std::runtime_error("sirius_physical_partition id " +
                             std::to_string(this->get_operator_id()) +
                             " has no downstream partition-sizing consumer set");
  }

  // Lock both this and the sibling partition atomically to prevent ABBA deadlock: without this, two
  // threads entering get_next_task_input_data on sibling partitions simultaneously would each hold
  // their own lock while trying to acquire the other's.
  if (_sibling_partition_op) {
    auto& sibling = _sibling_partition_op->Cast<sirius_physical_partition>();
    std::scoped_lock guard(lock, sibling.lock);
    if (!_num_partitions.has_value()) {
      auto& sizing_partition = _drives_partition_count ? *this : sibling;
      // Both locks are held here, so the two sides can be measured together for consumers that
      // size from the combined input.
      auto const this_bytes    = compute_total_bytes();
      auto const sibling_bytes = sibling.compute_total_bytes();
      partition_sizing_input const in{sizing_partition.compute_total_bytes(),
                                      sizing_partition._is_build,
                                      has_build_concat(*this) || has_build_concat(sibling),
                                      sirius::memory::saturating_add(this_bytes, sibling_bytes)};
      // The consumer owns the decision: it computes the count / broadcast flag, updates its own
      // execution state (e.g. hash-join BUILD_PROBE mode), and pre-sizes its own input repos.
      auto const strategy      = consumer->get_partition_strategy(in);
      auto* hash_join          = dynamic_cast<sirius_physical_hash_join*>(consumer);
      bool build_arrives_whole = false;
      if (strategy.build_probe) {
        // Configure both siblings' build-side CONCAT (do not short-circuit) and require that at
        // least one build-side CONCAT exists — BUILD_PROBE cannot run without a concat_all'd build.
        bool const found_this    = enable_build_concat_all(*this);
        bool const found_sibling = enable_build_concat_all(sibling);
        if (!found_this && !found_sibling) {
          throw std::runtime_error("sirius_physical_partition id " +
                                   std::to_string(this->get_operator_id()) +
                                   ": BUILD_PROBE was selected but no build-side CONCAT was found "
                                   "to fold the build into a "
                                   "single batch (concat_all)");
        }
        build_arrives_whole = strategy.num_partitions == 1 || strategy.broadcast;
      } else if (in.is_build_side && strategy.num_partitions == 1 && hash_join != nullptr &&
                 hash_join->publishes_dynamic_filters() &&
                 in.total_bytes < hash_join->max_build_hash_table_bytes()) {
        // Not BUILD_PROBE, but the build lands in one partition and this join publishes a filter
        // from a single build batch, so folding it only moves a batch boundary. Best-effort: no
        // build-side CONCAT means no publication. Build-side sizing is required — right-family
        // joins size from the probe, where one partition says nothing about the build's size.
        // The byte bound matters when hash_partition_bytes exceeds the build budget: a build
        // refused BUILD_PROBE for being too large must not be folded whole for a filter either.
        bool const found_this    = enable_build_concat_all(*this);
        bool const found_sibling = enable_build_concat_all(sibling);
        build_arrives_whole      = found_this || found_sibling;
      }
      if (hash_join != nullptr) { hash_join->set_build_arrives_whole(build_arrives_whole); }
      _broadcast              = strategy.broadcast;
      sibling._broadcast      = strategy.broadcast;
      _num_partitions         = strategy.num_partitions;
      sibling._num_partitions = strategy.num_partitions;
      SIRIUS_LOG_DEBUG(
        "sirius_physical_partition id {} sized {} partitions on sizing id {} ({} side){}, sibling "
        "id {}",
        this->get_operator_id(),
        strategy.num_partitions,
        sizing_partition.get_operator_id(),
        (sizing_partition._is_build ? "build" : "probe"),
        (strategy.broadcast ? " [broadcast]" : ""),
        _sibling_partition_op->get_operator_id());
    }
  } else {
    std::lock_guard<std::mutex> guard(lock);
    if (!_num_partitions.has_value()) {
      // Size from the projected total when one is available (the port may hold only the first
      // few batches), otherwise from what has arrived — which, without a projection, is the
      // complete input because get_next_task_hint held the task back until then.
      auto const estimated   = estimated_total_input_bytes();
      auto const total_bytes = estimated.value_or(compute_total_bytes());
      partition_sizing_input const in{total_bytes,
                                      _is_build,
                                      /*build_foldable=*/false,
                                      /*combined_total_bytes=*/total_bytes};
      auto const strategy = consumer->get_partition_strategy(in);
      _num_partitions     = strategy.num_partitions;
      // Record what actually drove the decision. This is the authoritative answer to "did the
      // feature do anything" — _size_estimate is not, because it can latch afterwards.
      _sizing_bytes = in.total_bytes;
      if (estimated.has_value()) {
        _sizing_basis = (_size_estimate && _size_estimate->exact) ? sizing_basis::upstream_complete
                                                                  : sizing_basis::projected;
      } else {
        _sizing_basis = sizing_basis::measured;
      }
      record_sizing_basis();
      SIRIUS_LOG_DEBUG("sirius_physical_partition id {} sized {} partitions from {} bytes ({})",
                       this->get_operator_id(),
                       strategy.num_partitions,
                       in.total_bytes,
                       sizing_basis_name(_sizing_basis));
    }
  }
  return sirius_physical_operator::get_next_task_input_data();
}

std::size_t sirius_physical_partition::no_history_peak_memory_estimate(
  const op::input_stats& stats) const
{
  if (_num_partitions.has_value() && *_num_partitions == 1) { return 0; }
  return stats.bytes * 2;
}

}  // namespace op
}  // namespace sirius
