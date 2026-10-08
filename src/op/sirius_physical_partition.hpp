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
#include "op/dynamic_filter/complete_build_inventory.hpp"
#include "op/sirius_physical_grouped_aggregate.hpp"
#include "op/sirius_physical_hash_join.hpp"
#include "op/sirius_physical_operator.hpp"
#include "op/sirius_physical_order.hpp"
#include "op/sirius_physical_partition_consumer_operator.hpp"
#include "op/sirius_physical_top_n.hpp"
#include "pipeline/data_size_estimator.hpp"
#include "sirius_config.hpp"

#include <atomic>
#include <cstdint>
#include <mutex>
#include <optional>
#include <string_view>

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
    bool enable_size_estimation                                = false);

  std::string get_name() const override;

  bool is_source() const override;

  bool is_sink() const override;
  [[nodiscard]] MemoryBarrierType input_barrier_for(
    sirius_physical_operator const& producer) const override;

  void build_pipelines(pipeline::sirius_pipeline& current,
                       pipeline::sirius_meta_pipeline& meta_pipeline) override;

  bool is_build_partition() const;

  /// Whether this partition may use a projected input size.
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
                                         ::cuda::stream_ref stream) override;

  void sink(const operator_data& input_data, ::cuda::stream_ref stream) override;

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

  [[nodiscard]] std::size_t no_history_peak_memory_estimate(
    const op::input_stats& stats) const override;

 protected:
  /**
   * @brief Closes accumulated input and releases retired storage after build tasks drain, then
   * reports the size estimate.
   */
  void on_finalize_operator() override;

  /**
   * @brief Records each batch pushed into the build side's FULL `default` port in the arrival
   * ledger, when the downstream join accumulates dynamic filters.
   *
   * @throw std::logic_error if a batch arrives after the ledger was certified
   */
  void on_input_batch_pushed(std::string_view port_id, cucascade::data_batch& batch) override;

 private:
  /**
   * @brief The downstream join if this is the build side of a non-broadcast HASH partition with
   * more than one partition and the join accumulates dynamic filters; otherwise null.
   */
  [[nodiscard]] sirius_physical_hash_join* accumulated_filter_join() const noexcept;

  /**
   * @brief Decides once, before the first batch leaves the input repository, whether the join
   * accumulates a filter from this input.
   *
   * Certifies the arrival ledger only for a closed input: exactly one data-bearing port, named
   * `default`, with one repository partition, a FULL barrier, and a finished source pipeline. Every
   * other input declines.
   *
   * @return false iff the source pipeline has not finished yet, so no batch may be popped
   */
  [[nodiscard]] bool decide_accumulation();

  /**
   * @brief Contributes a HASH build task's input batch to @p join's accumulated Bloom filter, from
   * `execute` before the batch is scattered.
   *
   * An input that is not exactly one batch with one original ID, or that carries a
   * late-materialization directive, ends the accumulation instead.
   *
   * @param join The join whose accumulation started
   * @param input The task's input
   * @param batch The read accessor of the input's one batch
   * @param stream The task's stream
   * @return The original ID that `execute` publishes with after scatter, or no value if the input
   * declined
   */
  [[nodiscard]] std::optional<std::uint64_t> contribute_to_accumulated_filter(
    sirius_physical_hash_join& join,
    pipelineable_operator_data const& input,
    cucascade::read_only_data_batch const& batch,
    ::cuda::stream_ref stream);

  void get_partition_keys_and_type(sirius_physical_operator* op, bool is_build = false);

  /// Sum the bytes of all batches waiting on this partition's input port. Fed to the downstream
  /// consumer's get_partition_strategy, which turns it into a partition count.
  uint64_t compute_total_bytes();

  /// Return a latched projected total, scaled and floored at bytes already received.
  /// Returns nullopt when estimation is disabled or unavailable.
  /// @pre `lock` is held.
  std::optional<uint64_t> estimated_total_input_bytes();

  /// Latch `strategy`'s count, broadcast flag, and placement on this partition and its sibling,
  /// and install the placement on `consumer` and on every operator either partition sinks into,
  /// so every emitter of this exchange's partitioned data stamps the same devices.
  /// @throws sirius::internal_exception if the placement names a GPU outside
  ///         `consumer.active_gpu_ids()`.
  /// @pre `lock` held; if there is a sibling, its `lock` held too.
  void apply_partition_strategy(partition_strategy const& strategy,
                                sirius_physical_partition_consumer_operator& consumer);
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
  /// Broadcast mode: the build table is small enough to replicate to every GPU instead of
  /// hash-partitioning. Set on BOTH sibling partition ops when the join accepts BUILD_PROBE with
  /// one partition per GPU. Build side deposits its batch into every slot; probe side deposits
  /// each batch into the slot placed on its current GPU. See get_next_task_input_data / sink.
  bool _broadcast{false};
  /// Which GPU each partition runs on, latched with `_num_partitions` by
  /// apply_partition_strategy (null until then). Broadcast probe batches are routed by it.
  std::shared_ptr<const partition_placement> _placement;
  /// Non-owning context providing narrow-passthrough events. The registered-state shared_ptr owns
  /// the context for at least as long as the query plan; unit-test operators may leave it null.
  duckdb::SiriusContext* _compressed_materialization_observer = nullptr;
  /// Ledger of build-side arrivals, for the join to certify its dynamic filter accumulation. Only
  /// used when the join accumulates a filter from this input, which is only true for
  /// non-broadcast HASH joins with more than one partition. See decide_accumulation().
  build_arrival_ledger _arrival_ledger;
  /// Serializes decide_accumulation().
  std::mutex _accumulation_mutex;
  /// Whether decide_accumulation() has already been called. Guarded by `_accumulation_mutex`.
  bool _accumulation_decided = false;
  /// The join whose accumulation `decide_accumulation` started, or null: tasks skip the publication
  /// session entirely without one.
  std::atomic<sirius_physical_hash_join*> _accumulating_join{nullptr};

  /// Enabled only for grouped-aggregation partitions.
  bool _enable_size_estimation{false};
  /// Raw projection, latched so the task hint and sizing decision agree. Guarded by `lock`.
  std::optional<pipeline::data_size_estimate> _size_estimate;

  enum class sizing_basis : uint8_t {
    measured,
    upstream_complete,
    projected,
  };
  static const char* sizing_basis_name(sizing_basis basis);
  sizing_basis _sizing_basis{sizing_basis::measured};
  uint64_t _sizing_bytes{0};
  /// Bytes processed, used to report projection error.
  std::atomic<uint64_t> _actual_bytes{0};
};

}  // namespace op
}  // namespace sirius
