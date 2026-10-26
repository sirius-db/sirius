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
#include "ducklake/scan_options.hpp"
#include "op/scan/gpu_ingestible.hpp"
#include "op/scan/table_scan/materializer_capabilities.hpp"
#include "scan_manager/preparation_ledger.hpp"
namespace sirius::ducklake {
class binding_handle;
// One original binding, with bounded unit discovery on the existing owner lane.
class scan_ingestible final : public op::scan::gpu_ingestible {
 public:
  scan_ingestible(std::shared_ptr<binding_handle>,
                  uint64_t contract,
                  uint64_t generation,
                  std::shared_ptr<scan_manager::preparation_admission>,
                  std::vector<size_t> required_columns,
                  bool count_only,
                  unit_scan_options);
  ~scan_ingestible();
  static scan_manager::unit_envelope envelope();
  void preflight_for_query() override;
  bool has_processed_all_metadata() const override;
  metadata_scan_task_t next_split_provider(io::ioctx_resolver) override;
  std::vector<preparation_source> make_preparation_sources(io::ioctx_resolver) override;
  std::unique_ptr<op::scan::batch_coalescer> create_batch_coalescer() const override;
  std::shared_ptr<scan_manager::preparation_admission> preparation_admission() const override;
  void attach_preparation_ledger(std::shared_ptr<scan_manager::preparation_ledger>) override;
  void stop_preparation() noexcept override;
  void finish_preparation() noexcept override;
  op::scan::ingestible_table_info const& table_info() const noexcept override;
  std::vector<size_t> materialized_column_order() const override;
  // Every child completes its residual and output projection before returning.
  bool output_assembly_is_leading_identity() const noexcept override { return true; }
  bool has_row_filter() const noexcept override;
  op::scan::materializer_contract_identity materializer() const;
  op::scan::filtered_table materialize_metadata_to_table(
    op::scan::scan_info const&,
    cucascade::memory::memory_space const&,
    ::cuda::stream_ref,
    bool,
    std::shared_ptr<sirius::like_multiliteral_cache const>,
    expression_query_policy = {}) override;
  std::unique_ptr<cudf::table> post_filter_and_project(
    op::scan::filtered_table&&,
    cucascade::memory::memory_space const&,
    ::cuda::stream_ref,
    bool                                                   = false,
    std::shared_ptr<sirius::like_multiliteral_cache const> = nullptr,
    std::unique_ptr<cudf::column>*                         = nullptr,
    std::span<size_t const>                                = {},
    expression_query_policy                                = {}) override;

 private:
  struct impl;
  std::shared_ptr<impl> impl_;
};
}  // namespace sirius::ducklake
