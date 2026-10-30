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
#include "scan_manager/preparation_ledger.hpp"

namespace sirius::scan_manager {
// Register one retained budget for a scan whose files are discovered lazily.
// A file keeps its permit until every consumer of its visibility has finished.
// Register a fixed set of slots and reuse their permits across files. Retirement
// releases the account when unused, but does not refund the ledger's cumulative
// registered allowance; constructing a fresh slot per file exhausts that allowance.
class preparation_slot {
 public:
  preparation_slot(std::shared_ptr<scan_manager::preparation_ledger> ledger,
                   scan_manager::unit_key key,
                   uint64_t retained_limit)
    : ledger_(std::move(ledger)), key_(key)
  {
    if (!ledger_) throw std::invalid_argument("Scan preparation slot requires a ledger");
    ledger_->register_unit(key_, retained_limit);
  }
  preparation_slot(preparation_slot const&)            = delete;
  preparation_slot& operator=(preparation_slot const&) = delete;
  ~preparation_slot() { ledger_->retire_unit(key_); }
  scan_manager::unit_key key() const { return key_; }
  scan_manager::w_permit try_acquire() { return ledger_->acquire_permit(key_); }

 private:
  std::shared_ptr<scan_manager::preparation_ledger> ledger_;
  scan_manager::unit_key key_;
};
}  // namespace sirius::scan_manager
