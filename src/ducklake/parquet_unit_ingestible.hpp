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
#include "op/scan/parquet_gpu_ingestible.hpp"
#include "scan_manager/preparation_ledger.hpp"

namespace sirius::ducklake {
class binding_handle;
struct unit_scan_options;
struct parquet_visibility_state;
struct inline_delete_preparation;
// One original unit, not a whole-table scan. Only a verified binding can create
// it. Registration stays closed until the complete input profile is qualified.
class parquet_unit_ingestible final : public op::scan::parquet_gpu_ingestible {
 public:
  static scan_manager::unit_envelope envelope(unit_descriptor const&);
  void preflight_for_query() override;
  metadata_scan_task_t next_split_provider(io::ioctx_resolver) override;
  metadata_claim try_claim_split(io::ioctx_resolver) override;
  std::vector<preparation_source> make_preparation_sources(io::ioctx_resolver) override;
  std::shared_ptr<scan_manager::preparation_admission> preparation_admission() const override;
  void attach_preparation_ledger(std::shared_ptr<scan_manager::preparation_ledger>) override;
  void bind_preparation_wakeup(scan_manager::preparation_wakeup) override;
  void stop_preparation() noexcept override;
  void finish_preparation() noexcept override;
  op::scan::filtered_table materialize_metadata_to_table(
    op::scan::scan_info const&,
    cucascade::memory::memory_space const&,
    ::cuda::stream_ref,
    bool,
    std::shared_ptr<sirius::like_multiliteral_cache const>,
    expression_query_policy = {}) override;
  op::scan::materializer_contract_identity materializer() const;
  unit_descriptor const& unit() const noexcept { return unit_; }

 private:
  friend class binding_handle;
  parquet_unit_ingestible(std::shared_ptr<binding_handle>,
                          unit_descriptor,
                          std::vector<op::scan::field_requirement>,
                          std::vector<size_t>,
                          op::scan::scan_contract_id,
                          uint64_t,
                          std::shared_ptr<scan_manager::preparation_admission>,
                          bool count_only,
                          unit_scan_options);
  std::shared_ptr<binding_handle> binding_;
  unit_descriptor unit_;
  uint64_t generation_;
  uint64_t preparation_id_;
  std::shared_ptr<parquet_visibility_state> visibility_;
  std::shared_ptr<inline_delete_preparation> inline_deletes_;
  std::shared_ptr<op::scan::split_validator const> validator_;
  std::shared_ptr<scan_manager::preparation_admission> admission_;
  std::shared_ptr<scan_manager::preparation_ledger> ledger_;
};
}  // namespace sirius::ducklake
