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
#include "ducklake/unit_descriptor.hpp"
#include "op/scan/gpu_ingestible.hpp"
#include "scan_manager/preparation_ledger.hpp"
namespace sirius::ducklake {
class binding_handle;
struct unit_scan_options;
// Inline profile: up to 32 fixed scalar/VARCHAR columns with a bounded cell budget.
// count_only keeps an internal row-count carrier, not a public SQL COUNT lowering.
class inline_unit_ingestible final : public op::scan::gpu_ingestible {
 public:
  ~inline_unit_ingestible();
  static constexpr size_t max_batch_rows = 2048, max_columns = 32;
  static constexpr size_t max_batch_cells = 8192;
  static size_t batch_rows(size_t columns);
  // Aggregate character bytes across selected VARCHAR columns. Exceeding one
  // row is a classified preparation resource failure, never a truncated value.
  static constexpr size_t max_string_bytes = 1024 * 1024;
  // Independent ceiling for normalized Arrow offsets, validity and Boolean bytes.
  static constexpr size_t max_staging_bytes = 64 * 1024;
  static uint64_t batch_capacity(size_t columns, size_t string_columns = 0);
  static scan_manager::unit_envelope envelope(size_t columns, size_t string_columns = 0);
  void preflight_for_query() override;
  bool has_processed_all_metadata() const override;
  metadata_scan_task_t next_split_provider(io::ioctx_resolver) override;
  metadata_claim try_claim_split(io::ioctx_resolver) override;
  preparation_source make_preparation_source(io::ioctx_resolver) override;
  std::unique_ptr<op::scan::batch_coalescer> create_batch_coalescer() const override;
  std::shared_ptr<scan_manager::preparation_admission> preparation_admission() const override;
  void attach_preparation_ledger(std::shared_ptr<scan_manager::preparation_ledger>) override;
  void bind_preparation_wakeup(scan_manager::preparation_wakeup) override;
  void stop_preparation() noexcept override;
  void finish_preparation() noexcept override;
  op::scan::ingestible_table_info const& table_info() const noexcept override;
  std::vector<size_t> materialized_column_order() const override;
  op::scan::materializer_contract_identity materializer() const;
  unit_descriptor const& unit() const noexcept;
  bool has_row_filter() const noexcept override;
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
  friend class binding_handle;
  inline_unit_ingestible(std::shared_ptr<binding_handle>,
                         unit_descriptor,
                         uint64_t,
                         uint64_t,
                         std::shared_ptr<scan_manager::preparation_admission>,
                         bool,
                         std::vector<size_t>,
                         unit_scan_options);
  struct impl;
  std::unique_ptr<impl> impl_;
};
}  // namespace sirius::ducklake
