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

#pragma once

#include "duckdb/execution/physical_operator.hpp"
#include "op/sirius_physical_grouped_aggregate.hpp"
#include "op/sirius_physical_hash_join.hpp"
#include "op/sirius_physical_operator.hpp"
#include "op/sirius_physical_order.hpp"
#include "op/sirius_physical_partition_consumer_operator.hpp"
#include "op/sirius_physical_top_n.hpp"
#include "pipeline/data_size_estimator.hpp"
#include "sirius_config.hpp"

#include <atomic>
#include <optional>

namespace duckdb {
class SiriusContext;
}  // namespace duckdb

namespace sirius {
namespace op {

enum class PartitionType { HASH, RANGE, EVENLY, CUSTOM, NONE };

// PartitionType to string
inline std::string partition_type_to_string(PartitionType type)
{
  switch (type) {
    case PartitionType::HASH: return "HASH";
    case PartitionType::RANGE: return "RANGE";
    case PartitionType::EVENLY: return "EVENLY";
    case PartitionType::CUSTOM: return "CUSTOM";
    case PartitionType::NONE: return "NONE";
  }
  return "UNKNOWN";
}

class sirius_physical_partition : public sirius_physical_operator {
 public:
  static constexpr const SiriusPhysicalOperatorType TYPE = SiriusPhysicalOperatorType::PARTITION;

  //! `key_source` is the downstream consumer whose keys determine partitioning (HJ join
  //! conditions, HGB/MERGE_GROUP_BY grouping columns). Captured at construction, never
  //! stored — the tree parent is `_parent_op`, stamped later by `set_parent_ops`.
  explicit sirius_physical_partition(
    duckdb::vector<sirius::logical_type> types,
    std::size_t estimated_cardinality,
    sirius_physical_operator* key_source,
    bool is_build                                              = false,
    duckdb::SiriusContext* compressed_materialization_observer = nullptr,
    bool enable_size_estimation                                = false,
    double size_estimate_safety_factor                         = 1.0);

  std::string get_name() const override;

  bool is_source() const override;

  bool is_sink() const override;
  [[nodiscard]] MemoryBarrierType input_barrier_for(
    sirius_physical_operator const& producer) const override;

  void build_pipelines(pipeline::sirius_pipeline& current,
                       pipeline::sirius_meta_pipeline& meta_pipeline) override;

  bool is_build_partition() const;

  /// operator_params::enable_runtime_size_estimation, threaded in at plan generation. Read by
  /// input_barrier_for, which relaxes this partition's ingress only when it is set.
  [[nodiscard]] bool is_size_estimation_enabled() const { return _enable_size_estimation; }

  void set_drives_partition_count(bool drives) { _drives_partition_count = drives; }

  //! Get the parent operator (e.g., HASH_JOIN for build partition)
  [[nodiscard]] sirius_physical_operator* get_parent_op() const { return _parent_op; }

  [[nodiscard]] sirius_physical_operator* get_sibling_partition_op() const
  {
    return _sibling_partition_op;
  }

  bool has_sibling() const { return _has_sibling_partition_op; }

  void set_sibling_partition_op(sirius_physical_operator* sibling_partition_op)
  {
    _sibling_partition_op = sibling_partition_op;
  }

  /// Hold off sizing until the sibling partition's input pipeline has also finished. Set for
  /// consumers whose `get_partition_strategy` reads `combined_total_bytes`: the driving partition
  /// otherwise measures the pair as soon as its own side completes, while the sibling is still
  /// filling. Costs nothing once the count is decided.
  void set_sizing_requires_sibling_input(bool requires_sibling)
  {
    _sizing_requires_sibling_input = requires_sibling;
  }

  std::unique_ptr<operator_data> execute(const operator_data& input_data,
                                         rmm::cuda_stream_view stream) override;

  void sink(const operator_data& input_data, rmm::cuda_stream_view stream) override;

  std::optional<task_creation_hint> get_next_task_hint() override;

  std::unique_ptr<operator_data> get_next_task_input_data() override;

  void set_num_partitions(int num_partitions);

  /// The downstream consumer that decides this partition's count / broadcast strategy (the
  /// HASH_JOIN / NESTED_LOOP_JOIN this partition feeds, or the MERGE_GROUP_BY above a group-by
  /// partition). Distinct from `key_source` (which only supplies partition keys) and from the batch
  /// receiver in `next_port_after_sink` (a CONCAT, for joins). Set at plan time.
  void set_downstream_consumer_op(sirius_physical_operator* consumer)
  {
    _downstream_consumer_op = consumer;
  }

  [[nodiscard]] sirius_physical_operator* get_downstream_consumer_op() const
  {
    return _downstream_consumer_op;
  }

  /// Input positions this partition hashes to place a row — its `key_source`'s keys,
  /// resolved at construction. Exposed for late materialization, which must never let one
  /// of these ride as a rowid: a rowid hashes differently from the value it stands for, so
  /// equal keys would land in different partitions and the consuming join would miss matches.
  [[nodiscard]] std::vector<int> const& partition_keys() const noexcept { return _partition_keys; }

  /// The sorted, deduped device ids of the GPUs the query runs on — identical to the list
  /// task_creator routes partitions across (`_active_gpu_ids[partition_idx % size]`). Used by
  /// broadcast mode to map a probe batch's residence GPU back to its partition slot.
  void set_active_gpu_ids(std::vector<int> active_gpu_ids)
  {
    _active_gpu_ids = std::move(active_gpu_ids);
  }

  [[nodiscard]] std::size_t no_history_peak_memory_estimate(
    const op::input_stats& stats) const override;

 protected:
  /// Emit the estimated-vs-actual size comparison for this partition (see @ref _actual_bytes).
  void on_finalize_operator() override;

 private:
  void get_partition_keys_and_type(sirius_physical_operator* op, bool is_build = false);

  /// Sum the bytes of all batches waiting on this partition's input port. Fed to the downstream
  /// consumer's get_partition_strategy, which turns it into a partition count.
  uint64_t compute_total_bytes();

  /// Bytes this partition should size itself for: the projected total for the input port, scaled
  /// by the safety factor and floored at what has already arrived, so an undershooting
  /// projection cannot ask for fewer partitions than the observed bytes justify. Latched in
  /// @ref _size_estimate so the hint and the sizing decision cannot disagree. nullopt when
  /// estimation is off or no projection is available yet; callers then wait for the whole input.
  ///
  /// @pre `lock` is held.
  std::optional<uint64_t> estimated_total_input_bytes();

  /// The partition slot for a batch residing on `device_id`: its index in `_active_gpu_ids`
  /// (so task_creator routes that slot back to the same GPU). Returns 0 if not found (a
  /// safe fallback that keeps the batch on some valid slot).
  [[nodiscard]] std::size_t slot_for_device(int device_id) const;
  sirius_physical_operator* _sibling_partition_op = nullptr;
  //! The downstream consumer that decides this partition's count / broadcast (see
  //! set_downstream_consumer_op). Always a partition-sizing consumer (HASH_JOIN / NESTED_LOOP_JOIN
  //! / MERGE_GROUP_BY).
  sirius_physical_operator* _downstream_consumer_op = nullptr;
  std::vector<int> _partition_keys;
  /// One entry per partition key. type_id::EMPTY means "hash as-is"; any other id means
  /// cast the key column to this type before hashing.  Used to align hash values when the
  /// two join sides have different physical column types for the same logical key.
  std::vector<cudf::data_type> _partition_key_cast_types;
  std::optional<int> _num_partitions;
  bool _is_build;
  bool _drives_partition_count{false};
  /// See set_sizing_requires_sibling_input.
  bool _sizing_requires_sibling_input{false};
  bool _has_sibling_partition_op;
  PartitionType _partition_type;
  /// Sorted, deduped active GPU device ids (see set_active_gpu_ids). Empty when unset / single-GPU.
  std::vector<int> _active_gpu_ids;
  /// Broadcast mode: the build table is small enough to replicate to every GPU instead of
  /// hash-partitioning. Set on BOTH sibling partition ops when the join accepts BUILD_PROBE at
  /// num_gpus partitions. Build side deposits its batch into every slot; probe side deposits each
  /// batch into the slot matching its current GPU. See get_next_task_input_data / sink.
  bool _broadcast{false};
  /// Non-owning sink for this operator's observability counters — the narrow-passthrough count
  /// and the sizing basis below. The registered-state shared_ptr owns the context for at least
  /// as long as the query plan; unit-test operators may leave it null.
  duckdb::SiriusContext* _context_observer = nullptr;
  /// operator_params::enable_runtime_size_estimation. Only the grouped-aggregation partition
  /// sets it; join partitions keep sizing from the measured build side.
  bool _enable_size_estimation{false};
  /// operator_params::size_estimate_safety_factor, applied to a projected total.
  double _size_estimate_safety_factor{1.0};
  /// Projection from estimated_total_input_bytes(), latched on first success so the task hint
  /// and the sizing decision cannot disagree. Guarded by `lock`. NOT the number that sized the
  /// partition — the safety factor and the already-arrived floor are applied after it; report
  /// @ref _sizing_bytes / @ref _sizing_basis, recorded at the decision itself.
  std::optional<pipeline::data_size_estimate> _size_estimate;

  /// How the partition count was *actually* chosen.
  enum class sizing_basis : uint8_t {
    measured,           ///< no projection available; sized from the drained port
    upstream_complete,  ///< an already-finished producer's exact total — too late to relax
    projected,          ///< a real prediction, made while the producer was still running
  };
  static const char* sizing_basis_name(sizing_basis basis);
  /// Publish @ref _sizing_basis to the context observer, once, at the decision. Only the
  /// grouped-aggregation path calls this; counting join-side sizing would dilute the signal.
  void record_sizing_basis() const;
  sizing_basis _sizing_basis{sizing_basis::measured};
  /// The byte total the count was derived from, whatever the basis.
  uint64_t _sizing_bytes{0};
  /// Bytes actually partitioned, accumulated in execute() and compared against @ref _sizing_bytes
  /// at finalize — how the projection's accuracy is checked from a query log.
  std::atomic<uint64_t> _actual_bytes{0};
};

}  // namespace op
}  // namespace sirius
