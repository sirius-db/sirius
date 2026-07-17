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
#include "sirius_config.hpp"

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

/// The broadcast-partitioning decision for a build/probe sibling pair, split into the two phases
/// the runtime needs. Pure/unit-testable counterpart of the decision block in
/// sirius_physical_partition::get_next_task_input_data.
///
/// A broadcast join replicates a *small* build table to every GPU (one hash table per GPU) instead
/// of routing the whole build to a single GPU. The decision is two-phase because the partition
/// count fed to update_join_exec_mode() must be chosen *before* the join reports whether it accepts
/// BUILD_PROBE, and the final broadcast flag depends on that acceptance:
///   1. `candidate` / `proposed_parts` are known up front (small build, multi-GPU, build side).
///   2. `broadcast()` / `num_partitions()` finalize once `is_build_probe` is known.
struct broadcast_partition_decision {
  bool candidate;      ///< small build on multi-GPU, replicate-worthy before join eligibility
  int proposed_parts;  ///< partition count to propose to update_join_exec_mode() (num_gpus if
                       ///< candidate, else the natural count)
  int natural_parts;   ///< the non-broadcast partition count (used when broadcast is not taken)

  /// Broadcast is taken only when the candidate condition holds AND the join accepted BUILD_PROBE
  /// for `proposed_parts` (right-family / mixed joins reject it, falling back to the natural
  /// count).
  [[nodiscard]] bool broadcast(bool is_build_probe) const { return candidate && is_build_probe; }

  /// Final partition count: num_gpus when broadcasting, otherwise the natural count.
  [[nodiscard]] int num_partitions(bool is_build_probe, std::size_t num_gpus) const
  {
    return broadcast(is_build_probe) ? static_cast<int>(num_gpus) : natural_parts;
  }
};

/// Compute the up-front (phase 1) broadcast decision. `is_build_side` is whether the sizing
/// partition drives the build side; only the build side can drive broadcast. A build smaller than
/// `small_table_bytes` on more than one GPU is a broadcast candidate, in which case we propose one
/// partition per GPU; otherwise we keep `natural_num_partitions`.
[[nodiscard]] broadcast_partition_decision make_broadcast_partition_decision(
  bool is_build_side,
  std::size_t num_gpus,
  uint64_t total_bytes,
  uint64_t small_table_bytes,
  int natural_num_partitions);

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
    bool is_build                 = false,
    uint64_t hash_partition_bytes = sirius::config::DEFAULT_HASH_PARTITION_BYTES);

  std::string get_name() const override;

  bool is_source() const override;

  bool is_sink() const override;

  void build_pipelines(pipeline::sirius_pipeline& current,
                       pipeline::sirius_meta_pipeline& meta_pipeline) override;

  bool is_build_partition() const;

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

  std::unique_ptr<operator_data> execute(const operator_data& input_data,
                                         rmm::cuda_stream_view stream) override;

  void sink(const operator_data& input_data, rmm::cuda_stream_view stream) override;

  std::optional<task_creation_hint> get_next_task_hint() override;

  std::unique_ptr<operator_data> get_next_task_input_data() override;

  void set_num_partitions(int num_partitions);

  /// Set a floor for num_partitions. The partition-consumer downstream (hash
  /// join, merge_group_by) pins each partition to partition_idx % num_gpus,
  /// so we need at least num_gpus partitions for all GPUs to see work on big
  /// inputs. Small inputs fall below `small_table_bytes` and stay at one
  /// partition (runs on a single GPU).
  void set_min_num_partitions(int min_num_partitions, uint64_t small_table_bytes)
  {
    _min_num_partitions = min_num_partitions;
    _small_table_bytes  = small_table_bytes;
  }

  /// The sorted, deduped device ids of the GPUs the query runs on — identical to the list
  /// task_creator routes partitions across (`_active_gpu_ids[partition_idx % size]`). Used by
  /// broadcast mode to map a probe batch's residence GPU back to its partition slot.
  void set_active_gpu_ids(std::vector<int> active_gpu_ids)
  {
    _active_gpu_ids = std::move(active_gpu_ids);
  }

  [[nodiscard]] std::size_t no_history_peak_memory_estimate(
    const op::input_stats& stats) const override;

 private:
  void get_partition_keys_and_type(sirius_physical_operator* op, bool is_build = false);

  /// Looks at the amount of data waiting on the input port and determines the number of partitions
  /// to create. Returns a pair of (num_partitions, total_bytes).
  std::pair<int, uint64_t> determine_num_partitions();

  /// Grow the downstream hash-join input repository for this partition's side to
  /// `num_partitions`. No-op when this partition does not feed a hash join.
  void resize_join_input_repo(int num_partitions);

  /// The partition slot for a batch residing on `device_id`: its index in `_active_gpu_ids`
  /// (so task_creator routes that slot back to the same GPU). Returns 0 if not found (a
  /// safe fallback that keeps the batch on some valid slot).
  [[nodiscard]] std::size_t slot_for_device(int device_id) const;
  sirius_physical_operator* _sibling_partition_op = nullptr;
  sirius_physical_operator* _hash_join_op =
    nullptr;  // hash join operator that this partition operator feeds into (optional: for
              // hash_joins only)
  std::vector<int> _partition_keys;
  /// One entry per partition key. type_id::EMPTY means "hash as-is"; any other id means
  /// cast the key column to this type before hashing.  Used to align hash values when the
  /// two join sides have different physical column types for the same logical key.
  std::vector<cudf::data_type> _partition_key_cast_types;
  std::optional<int> _num_partitions;
  bool _is_build;
  bool _drives_partition_count{false};
  bool _has_sibling_partition_op;
  PartitionType _partition_type;
  uint64_t s_partition_size;
  int _min_num_partitions{1};
  uint64_t _small_table_bytes{0};
  /// Sorted, deduped active GPU device ids (see set_active_gpu_ids). Empty when unset / single-GPU.
  std::vector<int> _active_gpu_ids;
  /// Broadcast mode: the build table is small enough to replicate to every GPU instead of
  /// hash-partitioning. Set on BOTH sibling partition ops when the join accepts BUILD_PROBE at
  /// num_gpus partitions. Build side deposits its batch into every slot; probe side deposits each
  /// batch into the slot matching its current GPU. See get_next_task_input_data / sink.
  bool _broadcast{false};
};

}  // namespace op
}  // namespace sirius
