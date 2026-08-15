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

#pragma once

#include "expression/ast/node.hpp"
#include "op/sirius_physical_filter.hpp"  // output_mask / passthrough
#include "op/sirius_physical_operator.hpp"

#include <cudf/types.hpp>

#include <atomic>
#include <memory>
#include <vector>

namespace sirius {
namespace op {

/**
 * @brief Routing-only anchor for the second output of a fused twin scan.
 *
 * Occupies the tree slot of the rewritten-away second scan (+ residual FILTER) so the
 * downstream feeder chain (PARTITION/CONCAT wraps, column bindings) is planned unchanged.
 * Like DELIM_SCAN it never lands in any pipeline's operators[]: the converter wires the
 * TWIN_SCAN_SPLIT's second output edge to this node's tree parent's pipeline, and
 * `build_pipelines` appends nothing.
 */
class sirius_physical_twin_scan_ref : public sirius_physical_operator {
 public:
  static constexpr const SiriusPhysicalOperatorType TYPE =
    SiriusPhysicalOperatorType::TWIN_SCAN_REF;

  sirius_physical_twin_scan_ref(duckdb::vector<sirius::logical_type> types,
                                std::size_t estimated_cardinality)
    : sirius_physical_operator(
        SiriusPhysicalOperatorType::TWIN_SCAN_REF, std::move(types), estimated_cardinality)
  {
  }

  //! Routing-only: the consumer pipeline's input arrives through the wiring emitted for the
  //! owning TWIN_SCAN_SPLIT, so this node must not open a pipeline of its own.
  void build_pipelines(pipeline::sirius_pipeline& current,
                       pipeline::sirius_meta_pipeline& meta_pipeline) override
  {
  }
};

/**
 * @brief Fan-out sink of a fused "twin scan" pipeline: one decoded + dynamically-filtered
 *        stream, two outputs.
 *
 * Pipeline shape: [GPU_SCAN(union columns), DYNAMIC_FILTER(shared channel), TWIN_SCAN_SPLIT].
 * Per input batch, execute() materializes
 *   - out-A: the first-scan projection (`output_indices_a` into the fused layout), a FRESH
 *     owned cudf::table gathered on the task stream from the current device resource and
 *     wrapped via `make_data_batch` into the input batch's memory space;
 *   - out-B: the second scan's residual filter (`residual` evaluated through an
 *     `expression_evaluator`, mirroring `sirius_physical_filter`) with the original FILTER's
 *     output projection (`output_columns_b`).
 * After execute() returns, neither half aliases the input batch, so the input may be freed or
 * spilled independently -- the generic streaming-operator memory discipline that lets the
 * downgrade executor treat this operator like any other with no type-specific handling.
 * sink() then routes out-A to the tree parent's pipeline and out-B to the twin ref's
 * consumer pipeline (both edges are emitted by the pipeline converter).
 *
 * `types` (and the physical sidecar) describe out-A — the schema the tree parent's feeder
 * chain was planned against. Out-B's schema is `types_b`, mirrored on the twin ref node.
 */
class sirius_physical_twin_scan_split : public sirius_physical_operator {
 public:
  static constexpr const SiriusPhysicalOperatorType TYPE =
    SiriusPhysicalOperatorType::TWIN_SCAN_SPLIT;

  sirius_physical_twin_scan_split(duckdb::vector<sirius::logical_type> types_a,
                                  std::vector<cudf::size_type> output_indices_a,
                                  std::unique_ptr<sirius::ast::node> residual,
                                  output_mask output_columns_b,
                                  duckdb::vector<sirius::logical_type> types_b,
                                  std::size_t estimated_cardinality,
                                  sirius_physical_twin_scan_ref* twin_ref);

  std::string params_to_string() const override;

  //! execute() emits a twin_split_output_data whose two halves carry different schemas
  //! (out-A == `types`, out-B == `types_b`), so the task-level schema diagnostic must not
  //! compare them against the declared output.
  [[nodiscard]] bool declared_output_schema_is_runtime_schema() const noexcept override
  {
    return false;
  }

  std::unique_ptr<operator_data> execute(const operator_data& input_data,
                                         rmm::cuda_stream_view stream) override;

  //! Route the two halves of the twin output to their consumer pipelines.
  void sink(const operator_data& input_data, rmm::cuda_stream_view stream) override;

  [[nodiscard]] sirius_physical_twin_scan_ref* twin_ref() const noexcept { return _twin_ref; }

 protected:
  //! Log the shared-stream / out-A / out-B row counts for validation against an unfused run.
  void on_finalize_operator() override;

 private:
  //! Positions of the first scan's output columns within the fused scan's output layout.
  std::vector<cudf::size_type> _output_indices_a;
  //! The second scan's residual filter predicate, bound to the fused scan's output layout.
  std::unique_ptr<sirius::ast::node> _residual;
  //! The residual FILTER's output projection (see sirius_physical_filter::output_columns).
  output_mask _output_columns_b;
  //! Out-B's schema (the rewritten-away FILTER's output schema).
  duckdb::vector<sirius::logical_type> _types_b;
  //! Anchor of the second output's consumer pipeline; its tree parent identifies the B edge
  //! among `next_port_after_sink` at runtime.
  sirius_physical_twin_scan_ref* _twin_ref;

  //! Cumulative row counts for the fused stream and both outputs (validation telemetry).
  std::atomic<std::size_t> _rows_in{0};
  std::atomic<std::size_t> _rows_a{0};
  std::atomic<std::size_t> _rows_b{0};
};

}  // namespace op
}  // namespace sirius
