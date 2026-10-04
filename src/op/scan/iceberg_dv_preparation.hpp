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
#include "op/scan/iceberg_metadata_reader.hpp"
#include "scan_manager/preparation_ledger.hpp"

namespace sirius::op::scan {
struct iceberg_delete_set;
class iceberg_dv_preparation {
 public:
  static scan_manager::scan_envelope envelope(iceberg_delete_discovery const&,
                                              std::vector<std::string> const&);
  iceberg_dv_preparation(scan_contract_id,
                         std::string const& table,
                         std::vector<std::string> const&,
                         std::shared_ptr<iceberg_delete_discovery const>,
                         scan_manager::preparation_ledger&);
  ~iceberg_dv_preparation();
  std::shared_ptr<void const> path_owner() const;
  void finish() noexcept;
  iceberg_dv_preparation(iceberg_dv_preparation const&)            = delete;
  iceberg_dv_preparation& operator=(iceberg_dv_preparation const&) = delete;
  struct file {
    std::string_view path;
    IcebergDeleteFileEntry const* dv = nullptr;
    std::weak_ptr<iceberg_delete_set const> result;
  };
  std::span<file> files;
  scan_contract_id const contract;
  std::shared_ptr<scan_manager::preparation_admission> admission;
  std::shared_ptr<scan_manager::preparation_ledger> ledger;

 private:
  std::shared_ptr<iceberg_delete_discovery const> inventory_;
  struct storage;
  std::shared_ptr<storage> storage_;
};
}  // namespace sirius::op::scan
