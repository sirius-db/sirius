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

#include <cudf/types.hpp>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace sirius {
namespace op {

//! What the discard sink checks while it drops its input.
struct discard_sink_options {
  //! Verify that the rows arrive sorted on `key_columns`.
  bool check_order{false};
  //! Input column indices of the sort keys; empty means no order check.
  std::vector<cudf::size_type> key_columns;
  //! Sort direction of each key column (parallel to `key_columns`).
  std::vector<cudf::order> column_order;
  //! Null placement of each key column (parallel to `key_columns`).
  std::vector<cudf::null_order> null_precedence;
};

//! Streaming operator placed right after the query's root: for each input batch it records
//! rows/bytes, optionally checks key order (within the batch, and across batches by their
//! positions), drops the batch, and emits a one-row INT64 table holding the batch's row count.
//! The planner sums those rows (an ungrouped SUM) to produce COPY's single count row.
class sirius_physical_discard_sink : public sirius_physical_operator {
 public:
  static constexpr const SiriusPhysicalOperatorType TYPE = SiriusPhysicalOperatorType::DISCARD_SINK;

  //! @param input_types  Schema of the batches to discard (the query root's output).
  //! @param options      Order check to run on the input, if any.
  sirius_physical_discard_sink(duckdb::vector<sirius::logical_type> input_types,
                               discard_sink_options options,
                               std::size_t estimated_cardinality);
  ~sirius_physical_discard_sink() override;

  std::string get_name() const override;

  //! Drops the input batch and returns a `pipelineable_operator_data` holding one batch of one
  //! INT64 column (the input's row count), carrying the input's position.
  std::unique_ptr<operator_data> execute(const operator_data& input_data,
                                         ::cuda::stream_ref stream) override;

  [[nodiscard]] std::size_t no_history_peak_memory_estimate(
    const op::input_stats& stats) const override;

  [[nodiscard]] sirius::OrderPreservationType output_order_preservation() const override;

  //! The input streams: each batch is dropped as it arrives. Under a FULL barrier the whole
  //! result would wait in the sink's repository until the producer finishes.
  [[nodiscard]] MemoryBarrierType input_barrier_for(
    sirius_physical_operator const& producer) const override;

  //! Input batches are counted where they are, so a batch that spilled while it waited is not read
  //! back; only the order check needs them on the GPU. The repository is also the last one the
  //! downgrade sweep spills: spilling a batch that is about to be dropped is wasted work.
  [[nodiscard]] port_residency_policy input_residency_policy(
    std::string_view port_id) const override;

  //! Wraps the task input so that it remembers the GPU memory space the task runs in, which is
  //! where the count row goes when no input batch is on the GPU.
  std::unique_ptr<operator_data> get_next_task_input_data() override;

  //! The order check this sink was built with.
  [[nodiscard]] const discard_sink_options& options() const noexcept { return _options; }
  //! Schema of the batches the sink drops. `get_types()` is what it emits: one BIGINT count.
  [[nodiscard]] const duckdb::vector<sirius::logical_type>& input_types() const noexcept
  {
    return _input_types;
  }

  //! Rows dropped so far.
  [[nodiscard]] std::uint64_t rows_discarded() const noexcept;
  //! Bytes of the batches dropped so far.
  [[nodiscard]] std::uint64_t bytes_discarded() const noexcept;
  //! Key-order violations found so far, within a batch and across batches.
  [[nodiscard]] std::uint64_t order_violations() const noexcept;

  //! Runs the cross-batch order check and logs
  //! "[discard_sink] rows=… bytes=… order_violations=…" at INFO. The engine calls it when the
  //! operator's pipeline finishes; a second call does nothing.
  void finish();

 protected:
  void on_finalize_operator() override { finish(); }

 private:
  //! What the order check remembers between tasks, and whether finish() already ran.
  struct order_check_state;

  discard_sink_options _options;
  duckdb::vector<sirius::logical_type> _input_types;
  std::unique_ptr<order_check_state> _state;
  std::atomic<std::uint64_t> _rows_discarded{0};
  std::atomic<std::uint64_t> _bytes_discarded{0};
  std::atomic<std::uint64_t> _order_violations{0};
};

}  // namespace op
}  // namespace sirius
