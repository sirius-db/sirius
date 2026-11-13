// Copyright 2026, Sirius Contributors. SPDX-License-Identifier: Apache-2.0
#pragma once
#include "op/sirius_physical_operator.hpp"
#include "output/parquet_session.hpp"

namespace sirius::op {
class sirius_physical_parquet_sink : public sirius_physical_operator {
 public:
  static constexpr auto TYPE = SiriusPhysicalOperatorType::COPY_TO_FILE;
  sirius_physical_parquet_sink(duckdb::vector<logical_type> input_types,
                               output::parquet_export_spec spec);
  bool declared_output_schema_is_runtime_schema() const noexcept override { return false; }
  bool is_sink() const override { return true; }
  MemoryBarrierType input_barrier_for(sirius_physical_operator const&) const override
  {
    return MemoryBarrierType::PIPELINE;
  }
  OrderPreservationType output_order_preservation() const override
  {
    return OrderPreservationType::INSERTION_ORDER;
  }
  void build_pipelines(pipeline::sirius_pipeline& current,
                       pipeline::sirius_meta_pipeline& meta) override;
  std::unique_ptr<operator_data> execute(operator_data const& data,
                                         ::cuda::stream_ref stream) override;
  std::unique_ptr<operator_data> execute_with_context(
    operator_data const& data,
    ::cuda::stream_ref stream,
    operator_execution_context const& context) override;
  void sink(operator_data const&, ::cuda::stream_ref) override {}
  std::size_t no_history_peak_memory_estimate(input_stats const& stats) const override;
  output::parquet_export_result finish(
    std::function<void(output::parquet_export_result const&)> const& prepare_completion = {});
  void abort() noexcept { session_.abort(); }

 private:
  output::parquet_session session_;
};
}  // namespace sirius::op
