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
class iceberg_dv_attempt;
class iceberg_gpu_ingestible;

/// Plan-time descriptors of a scan's deletion vectors: which scanned file each manifest entry
/// refers to, in one retained arena charged to the statement admission at lowering. Read-only
/// after construction; the per-execution result slots inside the records are written only
/// through an iceberg_dv_attempt.
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
  iceberg_dv_preparation(iceberg_dv_preparation const&)            = delete;
  iceberg_dv_preparation& operator=(iceberg_dv_preparation const&) = delete;
  struct file {
    std::string_view path;
    IcebergDeleteFileEntry const* dv = nullptr;
    std::weak_ptr<iceberg_delete_set const> result;
  };
  std::span<file const> files() const noexcept { return files_; }
  scan_contract_id contract() const noexcept { return contract_; }
  std::shared_ptr<void const> path_owner() const;
  void bind_admission(std::shared_ptr<scan_manager::preparation_admission>);
  std::shared_ptr<scan_manager::preparation_admission> const& admission() const noexcept
  {
    return admission_;
  }

 private:
  friend class iceberg_dv_attempt;
  friend class iceberg_gpu_ingestible;
  std::span<file> slots() noexcept { return files_; }
  void release() noexcept;

  scan_contract_id const contract_;
  std::span<file> files_;
  std::shared_ptr<scan_manager::preparation_admission> admission_;
  std::shared_ptr<iceberg_delete_discovery const> inventory_;
  struct storage;
  std::shared_ptr<storage> storage_;
};

/// One execution of a prepared scan: owns the attempt ledger handed over when the statement
/// admission is consumed and the write side of the plan's result slots. A prepared scan must
/// have only one active attempt — two would share those slots and either finish() would retire
/// the descriptors under the other — so begin_preparation() rejects a second installation on the
/// ingestible. stop() drops the ledger once metadata work has drained; finish() additionally
/// retires the plan's descriptors.
class iceberg_dv_attempt {
 public:
  iceberg_dv_attempt(std::shared_ptr<iceberg_dv_preparation>,
                     std::shared_ptr<scan_manager::preparation_ledger>);
  iceberg_dv_attempt(iceberg_dv_attempt const&)            = delete;
  iceberg_dv_attempt& operator=(iceberg_dv_attempt const&) = delete;
  iceberg_dv_preparation const& plan() const noexcept { return *plan_; }
  std::shared_ptr<scan_manager::preparation_ledger> const& ledger() const noexcept
  {
    return ledger_;
  }
  void record(size_t index, std::shared_ptr<iceberg_delete_set const> const&);
  std::shared_ptr<iceberg_delete_set const> result(size_t index) const noexcept;
  void stop() noexcept { ledger_.reset(); }
  void finish() noexcept;

 private:
  std::shared_ptr<iceberg_dv_preparation> plan_;
  std::shared_ptr<scan_manager::preparation_ledger> ledger_;
};
}  // namespace sirius::op::scan
