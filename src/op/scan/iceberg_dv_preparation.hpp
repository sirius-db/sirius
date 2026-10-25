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
#include "scan_manager/preparation_claim.hpp"
#include "scan_manager/preparation_ledger.hpp"

namespace sirius::op::scan {
struct iceberg_delete_set;
class iceberg_dv_attempt;
class iceberg_gpu_ingestible;
struct iceberg_dv_results;

/// Plan-time descriptors of a scan's deletion vectors: which scanned file each manifest entry
/// refers to, in one retained arena charged to the statement admission at lowering. Read-only
/// after construction. Reserved result storage is transferred once to the execution attempt.
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
  void release() noexcept;

  scan_contract_id const contract_;
  std::span<file> files_;
  std::shared_ptr<scan_manager::preparation_admission> admission_;
  std::shared_ptr<iceberg_delete_discovery const> inventory_;
  struct storage;
  std::shared_ptr<storage> storage_;
  std::unique_ptr<iceberg_dv_results> results_;
};

/// One execution of a prepared scan owns its ledger, cursor and result slots. Result storage
/// transfers from the plan once. stop() drops the ledger after preparation drains; finish()
/// releases result slots and descriptors after GPU consumers retire.
class iceberg_dv_attempt : public std::enable_shared_from_this<iceberg_dv_attempt> {
 public:
  iceberg_dv_attempt(std::shared_ptr<iceberg_dv_preparation>,
                     std::shared_ptr<scan_manager::preparation_ledger>);
  ~iceberg_dv_attempt();
  iceberg_dv_attempt(iceberg_dv_attempt const&)            = delete;
  iceberg_dv_attempt& operator=(iceberg_dv_attempt const&) = delete;
  void bind_wakeup(scan_manager::preparation_wakeup);
  class file_job {
   public:
    ~file_job();
    file_job(file_job const&)            = delete;
    file_job& operator=(file_job const&) = delete;
    iceberg_dv_preparation::file const& descriptor() const noexcept { return descriptor_; }
    std::shared_ptr<void const> const& path_owner() const noexcept { return storage_; }
    scan_manager::charging_allocator allocator() const { return ledger_->allocator(permit_); }
    scan_manager::unit_key key() const noexcept;
    void commit();
    void complete(std::shared_ptr<iceberg_delete_set const> const&);

   private:
    friend class iceberg_dv_attempt;
    file_job(std::shared_ptr<iceberg_dv_attempt>, size_t);
    std::shared_ptr<iceberg_dv_attempt> attempt_;
    size_t index_;
    iceberg_dv_preparation::file descriptor_;
    std::shared_ptr<void const> storage_;
    std::shared_ptr<scan_manager::preparation_ledger> ledger_;
    scan_manager::w_permit permit_;
    bool accepted_ = false, registered_ = false, committed_ = false, completed_ = false;
  };
  scan_manager::preparation_claim<std::shared_ptr<file_job>> try_claim();
  std::shared_ptr<iceberg_delete_set const> result(size_t index,
                                                   std::string_view path) const noexcept;
  void stop() noexcept;
  void finish() noexcept;

 private:
  mutable std::mutex mutex_;
  size_t next_   = 0;
  bool claiming_ = false, claims_abandoned_ = false, stopped_ = false;
  std::shared_ptr<iceberg_dv_preparation> plan_;
  std::shared_ptr<scan_manager::preparation_ledger> ledger_;
  std::unique_ptr<iceberg_dv_results> results_;
};
}  // namespace sirius::op::scan
