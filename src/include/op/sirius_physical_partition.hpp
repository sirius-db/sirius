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

class sirius_physical_partition : public sirius_physical_operator {
 public:
  static constexpr const SiriusPhysicalOperatorType TYPE = SiriusPhysicalOperatorType::PARTITION;

  explicit sirius_physical_partition(
    duckdb::vector<sirius::logical_type> types,
    std::size_t estimated_cardinality,
    sirius_physical_operator* parent_op,
    bool is_build                 = false,
    uint64_t hash_partition_bytes = sirius::config::DEFAULT_HASH_PARTITION_BYTES);

  std::string get_name() const override;

  bool is_source() const override;

  bool is_sink() const override;

  bool is_build_partition();

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

  std::optional<task_creation_hint> get_next_task_hint(
    std::optional<std::size_t> downstream_request = std::nullopt) override;

  //! Partition turns one input batch into N output partitions.
  [[nodiscard]] TaskCountRelation upstream_to_downstream_relation() const override
  {
    return TaskCountRelation::FAN_OUT;
  }

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

  [[nodiscard]] std::size_t no_history_peak_memory_estimate(
    const op::input_stats& stats) const override;

 private:
  void get_partition_keys_and_type(sirius_physical_operator* op, bool is_build = false);

  /// Looks at the amount of data waiting on the input port and determines the number of partitions
  /// to create. Returns a pair of (num_partitions, total_bytes).
  std::pair<int, uint64_t> determine_num_partitions();
  sirius_physical_operator* _parent_op            = nullptr;
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
  bool _has_sibling_partition_op;
  PartitionType _partition_type;
  uint64_t s_partition_size;
  int _min_num_partitions{1};
  uint64_t _small_table_bytes{0};
};

}  // namespace op
}  // namespace sirius
