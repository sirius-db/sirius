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

#include <op/scan/duckdb_native_scan_info.hpp>
#include <op/sirius_physical_operator.hpp>
#include <op/sirius_physical_operator_type.hpp>

#include <memory>
#include <optional>

namespace sirius::scan_manager {
class split_connector;
class sirius_scan_manager;
}  // namespace sirius::scan_manager

namespace sirius::op::scan {

//===----------------------------------------------------------------------===//
// GPU-native DuckDB scan operator.
//
// Sibling to sirius_gpu_parquet_scan_operator, but the source is a
// DuckTableEntry inside an attached .duckdb file. The split provider walks
// the per-row-group segment metadata once at prepare_for_query and emits
// one split per row-group batch; each split carries a slice of the walker's
// row_groups vector + a shared scan_info handle. execute() decodes the
// segments described by the split into a cudf::table.
//===----------------------------------------------------------------------===//
class sirius_gpu_duckdb_native_scan_operator : public sirius_physical_operator {
 public:
  static constexpr SiriusPhysicalOperatorType TYPE =
    SiriusPhysicalOperatorType::GPU_DUCKDB_NATIVE_SCAN;

  sirius_gpu_duckdb_native_scan_operator(duckdb::vector<sirius::logical_type> types,
                                         duckdb::idx_t estimated_cardinality,
                                         std::unique_ptr<duckdb_native_scan_info> scan_info);

  ~sirius_gpu_duckdb_native_scan_operator() override;

  bool is_source() const override { return true; }

  std::optional<task_creation_hint> get_next_task_hint() override;
  [[nodiscard]] bool all_ports_empty() override;
  std::unique_ptr<operator_data> get_next_task_input_data() override;

  std::unique_ptr<operator_data> execute(const operator_data& input_data,
                                         rmm::cuda_stream_view stream) override;

  [[nodiscard]] std::size_t no_history_peak_memory_estimate(
    const op::input_stats& stats) const override;

 private:
  friend class scan_manager::sirius_scan_manager;

  std::unique_ptr<duckdb_native_scan_info> take_scan_info();
  void set_split_connector(std::unique_ptr<scan_manager::split_connector> connector);
  scan_manager::split_connector* get_split_connector() noexcept { return _split_connector.get(); }

  std::unique_ptr<scan_manager::split_connector> _split_connector;
  std::unique_ptr<duckdb_native_scan_info> _scan_info;
};

}  // namespace sirius::op::scan
