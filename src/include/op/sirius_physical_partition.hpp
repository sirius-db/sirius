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
#include "duckdb/planner/expression/bound_reference_expression.hpp"
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
    duckdb::vector<duckdb::LogicalType> types,
    duckdb::idx_t estimated_cardinality,
    sirius_physical_operator* parent_op,
    bool is_build                 = false,
    uint64_t hash_partition_bytes = sirius::config::DEFAULT_HASH_PARTITION_BYTES);

  std::string get_name() const override;

  bool is_source() const override;

  bool is_sink() const override;

  bool is_build_partition();

  //! Get the parent operator (e.g., HASH_JOIN for build partition)
  [[nodiscard]] sirius_physical_operator* get_parent_op() const { return _parent_op; }

  std::unique_ptr<operator_data> execute(const operator_data& input_data,
                                         rmm::cuda_stream_view stream) override;

  void sink(const operator_data& input_data, rmm::cuda_stream_view stream) override;

 private:
  void get_partition_keys_and_type(sirius_physical_operator* op, bool is_build = false);
  sirius_physical_operator* _parent_op;
  std::vector<int> _partition_keys;
  /// One entry per partition key. type_id::EMPTY means "hash as-is"; any other id means
  /// cast the key column to this type before hashing.  Used to align hash values when the
  /// two join sides have different physical column types for the same logical key.
  std::vector<cudf::data_type> _partition_key_cast_types;
  int _num_partitions;
  bool _is_build;
  PartitionType _partition_type;
  uint64_t s_partition_size;
};

}  // namespace op
}  // namespace sirius
