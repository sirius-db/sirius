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

#include "op/sirius_physical_operator.hpp"
#include "sirius_config.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <stdexcept>
#include <vector>

namespace sirius {
namespace op {

/// One physical column of a partition's input, as the PARTITION actually observed it on the
/// device. Recorded per column rather than summarized because only the consumer knows what each
/// column means (for a grouped aggregation: which are grouping keys and which partial states).
struct observed_column_meta {
  /// `cudf::type_id` value, kept as an int so this header stays free of cuDF includes.
  int type_id = 0;
  /// Fixed element width in bytes; 0 means the type is not fixed-width (STRING/LIST/STRUCT/
  /// DICTIONARY32).
  uint32_t fixed_width_bytes = 0;
  /// The column carries a validity mask in at least one input batch, so concatenating the batches
  /// allocates one for it.
  bool nullable = false;
};

/// A snapshot of what is waiting on a PARTITION's single input port, for consumers whose sizing
/// decision depends on more than the byte total.
///
/// Built by the PARTITION operator, which owns the input repository and its lock, and offered to
/// the consumer through @ref partition_sizing_input::input_metadata_source. Nothing is collected
/// unless the consumer asks, so consumers that ignore it cost nothing.
///
/// Every quantity that can be genuinely unknown is an optional. A missing value and a real zero
/// are different answers — "this input has no nullable columns" must not be confused with "the
/// column metadata could not be read".
struct observed_input_metadata {
  /// The partition's input pipeline has finished: every batch has actually arrived. This is what
  /// makes the row/type metadata below a fact rather than a forecast, and it is a stronger claim
  /// than `data_size_estimate::exact`, which only says the byte total is known.
  bool upstream_complete = false;

  /// Every batch is a plain cuDF table resident in exactly one GPU memory space.
  bool single_gpu_resident = false;

  /// Per-column physical metadata, in table order. Absent when the schema could not be read, or
  /// was inconsistent between batches.
  std::optional<std::vector<observed_column_meta>> columns;

  /// Total rows over all batches; absent when the metadata could not be read.
  std::optional<uint64_t> total_rows;

  /// Bytes a new reservation could still obtain on the space the input lives in, from
  /// memory::gpu_reservable_bytes() (not `memory_space::get_available_memory()`, which over-states
  /// it). Absent when it could not be determined.
  std::optional<uint64_t> admissible_additional_budget;

  /// Device the input actually lives on. -1 when unknown; never assumed to be 0.
  int target_device_id = -1;
};

/// What the upstream PARTITION operator measures/knows and forwards to its downstream consumer so
/// the consumer can decide how many partitions to produce (and whether to broadcast).
struct partition_sizing_input {
  uint64_t total_bytes;  ///< Bytes waiting on the sizing partition's input port.
  bool is_build_side;    ///< The sizing partition drives the build side (only the build side can
                         ///< drive broadcast / build-probe).
  bool build_foldable;   ///< A downstream build-side CONCAT can concat_all to a single batch.
  /// Bytes waiting on both sibling partitions' input ports (equal to `total_bytes` when the
  /// partition has no sibling). A consumer whose task holds both join inputs at once sizes from
  /// this rather than `total_bytes`; one whose task holds only the sizing side uses `total_bytes`.
  uint64_t combined_total_bytes;

  /// Collects @ref observed_input_metadata on demand; empty when the partition has no single input
  /// port to describe (sibling-partition joins). Lazy so a consumer that does not need it, or
  /// whose cheap gates already decide, never walks the batches. Callable only during the
  /// get_partition_strategy call, which runs under the partition's lock. Individual batches are
  /// read-locked only while being inspected; residency and the budget can change after that
  /// snapshot.
  std::function<std::optional<observed_input_metadata>()> input_metadata_source;
};

/// The partitioning decision returned by a consumer's get_partition_strategy. `num_partitions` is
/// applied to the partition operator(s); `broadcast`/`build_probe` are reported back so the
/// partition can configure its own wiring (e.g. enabling build-side concat_all).
struct partition_strategy {
  int num_partitions;
  bool broadcast;
  bool build_probe;
};

/// Multi-GPU partition floor derived purely from the GPU count: below the small-table threshold a
/// consumer stays on one partition (single GPU); at/above it we force one partition per GPU so
/// every GPU sees work.
[[nodiscard]] constexpr int partition_min_num_partitions(int num_gpus)
{
  return num_gpus > 1 ? num_gpus : 1;
}

/// Small-table byte threshold derived from the GPU count (see PARTITION_SMALL_TABLE_BYTES_PER_GPU).
[[nodiscard]] constexpr uint64_t partition_small_table_bytes(int num_gpus)
{
  return static_cast<uint64_t>(num_gpus) * config::PARTITION_SMALL_TABLE_BYTES_PER_GPU;
}

/// The "natural" partition count for `total_bytes`: ceil(total_bytes / hash_partition_bytes),
/// floored to `partition_min_num_partitions(num_gpus)` once the input clears the small-table
/// threshold. Pure counterpart of the old sirius_physical_partition::determine_num_partitions
/// arithmetic.
[[nodiscard]] inline int natural_num_partitions(uint64_t total_bytes,
                                                uint64_t hash_partition_bytes,
                                                int num_gpus)
{
  if (hash_partition_bytes == 0) {
    throw std::invalid_argument("hash_partition_bytes must be greater than zero");
  }
  int num_partitions =
    static_cast<int>(std::max(uint64_t{1},
                              total_bytes / hash_partition_bytes +
                                static_cast<uint64_t>(total_bytes % hash_partition_bytes != 0)));
  int const min_parts = partition_min_num_partitions(num_gpus);
  if (min_parts > 1 && total_bytes >= partition_small_table_bytes(num_gpus)) {
    num_partitions = std::max(num_partitions, min_parts);
  }
  return num_partitions;
}

//! sirius_physical_partition_consumer_operator is an interface for operators
//! that can consume partitioned data batches
class sirius_physical_partition_consumer_operator : public sirius_physical_operator {
 public:
  sirius_physical_partition_consumer_operator(SiriusPhysicalOperatorType type,
                                              duckdb::vector<sirius::logical_type> types,
                                              std::size_t estimated_cardinality)
    : sirius_physical_operator(type, std::move(types), estimated_cardinality)
  {
  }

  virtual ~sirius_physical_partition_consumer_operator();

  //! Push a data batch to a specific port with partition information
  //! @param port_id The port identifier
  //! @param batch The data batch to push
  //! @param partition_idx The partition index
  virtual void push_data_batch_partitioned(std::string_view port_id,
                                           std::shared_ptr<::cucascade::data_batch> batch,
                                           std::size_t partition_idx);

  /// @brief Decide how the upstream PARTITION operator should partition its input for this
  /// consumer, given what the partition measured (`in`). Implementations own the count/broadcast
  /// decision, update any of their own execution state (e.g. hash-join BUILD_PROBE mode), and
  /// pre-size their own input repositories under their own lock. The base default throws — only
  /// consumers that actually drive a partition count (HASH_JOIN, NESTED_LOOP_JOIN, MERGE_GROUP_BY)
  /// override it; batch-only consumers such as CONCAT never appear as a partition's downstream
  /// sizing consumer.
  virtual partition_strategy get_partition_strategy(const partition_sizing_input& in);

  /// @brief Inform the consumer how many GPUs the query runs on. Set at plan/convert time; drives
  /// the multi-GPU partition floor and (for joins) BUILD_PROBE / broadcast admission. Defaults
  /// to 1.
  void set_num_gpus(int num_gpus) { _num_gpus = num_gpus; }

 protected:
  //! Target size (bytes) per hash partition — the natural-count divisor. Set from operator_params
  //! at construction (a config option, not a runtime setter).
  uint64_t _hash_partition_bytes = config::DEFAULT_HASH_PARTITION_BYTES;

  //! Number of GPUs the query runs on (set at plan time via set_num_gpus).
  int _num_gpus = 1;
};

}  // namespace op
}  // namespace sirius
