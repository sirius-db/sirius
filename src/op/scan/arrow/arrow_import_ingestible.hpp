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
#include "op/scan/arrow/arrow_batch_producer.hpp"
#include "op/scan/gpu_ingestible.hpp"
#include "op/scan/table_scan/owned_arrow_scan.hpp"

namespace sirius::op::scan {
struct arrow_import_binding {
  materialization_program program;
  std::shared_ptr<physical_profile_table> profiles;
  // Retains the original bound read-view evidence; identity is compared by ownership.
  std::shared_ptr<void const> read_identity;
  std::string display_name;
  std::vector<std::string> column_names;
  std::vector<size_t> output_columns;
  std::shared_ptr<sirius::ast::node const> predicate;
  // Trusted binding hook, never carried by a produced input. Also checks rootless EOS.
  std::function<void(owned_arrow_scan_info const&)> validate_provenance;
  std::shared_ptr<scan_manager::preparation_admission> admission;
  std::shared_ptr<arrow_input_gate> gate;
  std::function<uint64_t()> next_unit_id;
  bool owns_admission = true;
  // A mixed outer scan can supply its own single empty carrier instead.
  bool emit_empty_source = true;
};
class arrow_import_ingestible final : public gpu_ingestible {
 public:
  arrow_import_ingestible(arrow_import_binding, std::shared_ptr<arrow_batch_producer>);
  ~arrow_import_ingestible() override;
  void preflight_for_query() override;
  bool has_processed_all_metadata() const override { return false; }
  metadata_scan_task_t next_split_provider(io::ioctx_resolver) override;
  metadata_claim try_claim_split(io::ioctx_resolver) override;
  preparation_source make_preparation_source(io::ioctx_resolver) override;
  std::unique_ptr<batch_coalescer> create_batch_coalescer() const override;
  std::shared_ptr<scan_manager::preparation_admission> preparation_admission() const override;
  void attach_preparation_ledger(std::shared_ptr<scan_manager::preparation_ledger>) override;
  void bind_preparation_wakeup(scan_manager::preparation_wakeup) override;
  void stop_preparation() noexcept override;
  void finish_preparation() noexcept override;
  ingestible_table_info const& table_info() const noexcept override;
  std::vector<size_t> materialized_column_order() const override;
  materializer_contract_identity materializer() const;
  bool has_row_filter() const noexcept override;
  filtered_table materialize_metadata_to_table(
    scan_info const&,
    cucascade::memory::memory_space const&,
    ::cuda::stream_ref,
    bool                                                   = false,
    std::shared_ptr<sirius::like_multiliteral_cache const> = {},
    expression_query_policy                                = {}) override;
  std::unique_ptr<cudf::table> post_filter_and_project(
    filtered_table&&,
    cucascade::memory::memory_space const&,
    ::cuda::stream_ref,
    bool                                                   = false,
    std::shared_ptr<sirius::like_multiliteral_cache const> = {},
    std::unique_ptr<cudf::column>*                         = nullptr,
    std::span<size_t const>                                = {},
    expression_query_policy                                = {}) override;

 private:
  struct impl;
  std::shared_ptr<impl> impl_;
};
}  // namespace sirius::op::scan
