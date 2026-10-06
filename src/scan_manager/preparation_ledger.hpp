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
#include "scan_manager/preparation.hpp"

#include <cucascade/memory/common.hpp>

#include <span>
#include <utility>
#include <vector>

namespace cucascade::memory {
class memory_reservation_manager;
}
namespace sirius::scan_manager {
using memory_space_id = cucascade::memory::memory_space_id;
enum class scan_eligibility : uint8_t { eligible, legacy_v2, legacy_mixed, legacy_admission };
enum class admission_reason : uint8_t {
  admitted,
  unqualified,
  overflow,
  statement_dv_limit,
  no_grant,
  short_grant,
  capacity
};
struct unit_envelope {
  uint64_t blob, footer_parser, roaring, positions;
};
struct scan_envelope {
  std::vector<unit_envelope> units;
  uint64_t retained_descriptors = 0;
  uint64_t resolver_index       = 0;
  bool qualified                = false;
};
struct admission_decision {
  bool deferred           = false;
  uint64_t sigma_retained = 0, w = 0, c_obtained = 0;
  uint32_t n_permits            = 0;
  scan_eligibility route_reason = scan_eligibility::legacy_admission;
  admission_reason reason       = admission_reason::unqualified;
};
// Every allocation returns its real backing owner. No reserved-credit + unrelated malloc path.
struct backing_allocation {
  std::byte* data;
  std::shared_ptr<void> owner;
};
class reservation_backing {
 public:
  virtual ~reservation_backing()                      = default;
  virtual uint64_t charge_size(uint64_t bytes) const  = 0;
  virtual backing_allocation allocate(uint64_t bytes) = 0;
};
struct reservation_grant {
  uint64_t bytes;
  memory_space_id space;
  std::shared_ptr<reservation_backing> handle;
};
struct reservation_provider {
  virtual ~reservation_provider()                                                   = default;
  virtual std::optional<reservation_grant> request(memory_space_id, uint64_t bytes) = 0;
  virtual uint64_t allocation_granularity(memory_space_id) const { return 1; }
  virtual void release(reservation_grant&& grant) noexcept { grant.handle.reset(); }
};
class host_reservation_provider final : public reservation_provider {
 public:
  explicit host_reservation_provider(cucascade::memory::memory_reservation_manager& manager,
                                     std::shared_ptr<void> lifetime = {})
    : manager_(manager), lifetime_(std::move(lifetime))
  {
  }
  std::optional<reservation_grant> request(memory_space_id, uint64_t bytes) override;
  uint64_t allocation_granularity(memory_space_id) const override;

 private:
  cucascade::memory::memory_reservation_manager& manager_;
  std::shared_ptr<void> lifetime_;
};
class preparation_resource_error : public transparent::classified_execution_error {
 public:
  preparation_resource_error(std::string message, bool exceeded, std::exception_ptr error = {})
    : classified_execution_error(transparent::late_failure_cause::resource, std::move(message)),
      bound_exceeded(exceeded),
      original(std::move(error))
  {
  }
  bool bound_exceeded;
  std::exception_ptr original;
};
struct ledger_state;
struct permit_state;
class charged_block {
 public:
  charged_block()                                = default;
  charged_block(charged_block const&)            = delete;
  charged_block& operator=(charged_block const&) = delete;
  charged_block(charged_block&& other) noexcept { *this = std::move(other); }
  charged_block& operator=(charged_block&& other) noexcept
  {
    if (this != &other) {
      release();
      data_     = std::exchange(other.data_, nullptr);
      size_     = std::exchange(other.size_, 0);
      retained_ = std::exchange(other.retained_, false);
      owner_    = std::move(other.owner_);
    }
    return *this;
  }
  void release() noexcept
  {
    owner_.reset();
    data_     = nullptr;
    size_     = 0;
    retained_ = false;
  }
  std::byte* data() const noexcept { return data_; }
  uint64_t size() const noexcept { return size_; }
  explicit operator bool() const noexcept { return bool(owner_); }
  bool retained() const noexcept { return retained_; }

 private:
  friend class charging_allocator;
  std::byte* data_ = nullptr;
  uint64_t size_   = 0;
  bool retained_   = false;
  std::shared_ptr<void> owner_;
};
class w_permit {
 public:
  w_permit()                           = default;
  w_permit(w_permit const&)            = delete;
  w_permit& operator=(w_permit const&) = delete;
  w_permit(w_permit&&) noexcept        = default;
  w_permit& operator=(w_permit&&) noexcept;
  ~w_permit();
  void release() noexcept;
  explicit operator bool() const noexcept { return bool(state_); }

 private:
  friend class preparation_ledger;
  friend class charging_allocator;
  std::shared_ptr<permit_state> state_;
};
class charging_allocator {
 public:
  charged_block allocate(uint64_t bytes);
  charged_block allocate_retained(uint64_t bytes);

 private:
  friend class preparation_ledger;
  explicit charging_allocator(std::weak_ptr<permit_state> state) : state_(std::move(state)) {}
  charged_block allocate_impl(uint64_t, bool retained);
  std::weak_ptr<permit_state> state_;
};
class preparation_ledger {
 public:
  explicit preparation_ledger(reservation_provider& provider) : provider_(provider) {}
  ~preparation_ledger() { close(); }
  preparation_ledger(preparation_ledger const&)            = delete;
  preparation_ledger& operator=(preparation_ledger const&) = delete;
  admission_decision admit(std::span<scan_envelope const>, memory_space_id);
  // Bind file-local retained limits during registration, before any execution work.
  void register_unit(unit_key, uint64_t retained_limit);
  void retire_unit(unit_key key) noexcept;
  size_t tracked_units() const;
  w_permit acquire_permit(
    unit_key);  // Nonblocking; empty means coordinator must wait for an event.
  charging_allocator allocator(w_permit const&);
  uint64_t charged_bytes(unit_key) const;
  uint64_t granted_capacity() const;
  uint32_t permits_in_flight() const;
  uint64_t bound_exceeded() const;
  uint64_t allocation_failures() const;
  void close() noexcept;  // Stops admission; existing backing remains alive through its last user.
 private:
  reservation_provider& provider_;  // Used only by the one lowering-time admit call.
  bool admitted_ = false;
  std::shared_ptr<ledger_state> state_;
};
// Single owning statement admission, shared by scan references. Consumption happens once.
class preparation_admission {
 public:
  preparation_admission(std::unique_ptr<preparation_ledger>, admission_decision);
  std::unique_ptr<preparation_ledger> consume();
  void finish() noexcept;
  admission_decision const decision;

 private:
  std::mutex mutex_;
  bool consumed_ = false;
  std::unique_ptr<preparation_ledger> ledger_;
};
struct scan_dv_count {
  op::scan::scan_contract_id contract;
  uint64_t live_positions;
};
bool statement_dv_route_allowed(std::span<scan_dv_count const>);
bool statement_dv_route_allowed(std::span<scan_dv_count const>, uint64_t limit);
}  // namespace sirius::scan_manager
