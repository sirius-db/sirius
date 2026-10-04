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

#include "scan_manager/preparation_ledger.hpp"

#include "op/scan/puffin_reader.hpp"

#include <cucascade/memory/fixed_size_host_memory_resource.hpp>
#include <cucascade/memory/memory_reservation_manager.hpp>
#include <cucascade/memory/memory_space.hpp>

#include <algorithm>
#include <limits>
#include <map>

namespace sirius::scan_manager {
namespace {
uint64_t add(uint64_t a, uint64_t b)
{
  if (b > std::numeric_limits<uint64_t>::max() - a)
    throw std::overflow_error("preparation envelope overflow");
  return a + b;
}
uint64_t round(uint64_t n, uint64_t quantum)
{
  if (!quantum) throw std::invalid_argument("zero allocation granularity");
  return n == 0 ? 0 : add(n, quantum - 1) / quantum * quantum;
}
class host_backing final : public reservation_backing {
 public:
  explicit host_backing(std::unique_ptr<cucascade::memory::reservation> reservation,
                        std::shared_ptr<void> lifetime)
    : lifetime_(std::move(lifetime)),
      reservation_(std::move(reservation)),
      resource_(reservation_->get_memory_resource_of<cucascade::memory::Tier::HOST>())
  {
  }
  uint64_t charge_size(uint64_t n) const override { return round(n, resource_->get_block_size()); }
  backing_allocation allocate(uint64_t bytes) override
  {
    auto blocks   = resource_->allocate_multiple_blocks(bytes, reservation_.get());
    auto pointers = blocks->get_blocks();
    std::vector<std::byte*> ordered(pointers.begin(), pointers.end());
    std::sort(ordered.begin(), ordered.end(), [](auto* a, auto* b) {
      return reinterpret_cast<uintptr_t>(a) < reinterpret_cast<uintptr_t>(b);
    });
    // A contiguous buffer cannot silently fall back to an unrelated heap allocation.
    // Fragmentation is an allocation failure; this adapter requires contiguous buffers.
    for (size_t i = 1; i < ordered.size(); ++i)
      if (reinterpret_cast<uintptr_t>(ordered[i]) - reinterpret_cast<uintptr_t>(ordered[i - 1]) !=
          resource_->get_block_size())
        throw std::bad_alloc();
    auto* data = ordered.empty() ? nullptr : ordered.front();
    return {data,
            std::shared_ptr<
              cucascade::memory::fixed_size_host_memory_resource::multiple_blocks_allocation>(
              std::move(blocks))};
  }

 private:
  std::shared_ptr<void> lifetime_;
  std::unique_ptr<cucascade::memory::reservation> reservation_;
  cucascade::memory::fixed_size_host_memory_resource* resource_;
};
}  // namespace
uint64_t host_reservation_provider::allocation_granularity(memory_space_id space) const
{
  if (space.tier != cucascade::memory::Tier::HOST)
    throw std::invalid_argument("preparation requires HOST");
  auto* memory = manager_.get_memory_space(space.tier, space.device_id);
  if (!memory) throw std::invalid_argument("preparation HOST space missing");
  return memory->get_memory_resource_of<cucascade::memory::Tier::HOST>()->get_block_size();
}
std::optional<reservation_grant> host_reservation_provider::request(memory_space_id space,
                                                                    uint64_t bytes)
{
  if (space.tier != cucascade::memory::Tier::HOST)
    throw std::invalid_argument("preparation requires HOST");
  if (bytes > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) return std::nullopt;
  auto* memory = manager_.get_memory_space(space.tier, space.device_id);
  if (!memory || bytes > memory->get_max_memory()) return std::nullopt;
  auto reservation = memory->make_reservation_or_null(bytes);
  if (!reservation) return std::nullopt;
  auto capacity = reservation->size();
  return reservation_grant{
    capacity, space, std::make_shared<host_backing>(std::move(reservation), lifetime_)};
}
struct unit_account {
  uint64_t retained_limit = 0, retained = 0;
  std::weak_ptr<permit_state> permit;
  bool retired = false;
};
struct ledger_state {
  mutable std::mutex mutex;
  std::optional<reservation_grant> grant;
  uint64_t retained_limit = 0, registered = 0, retained = 0, w = 0;
  uint32_t permits = 0, active = 0;
  uint64_t exceeded = 0, allocation_failures = 0;
  bool closed = false;
  std::map<unit_key, unit_account> units;
};
namespace {
void erase_retired(ledger_state& ledger, unit_key key)
{
  auto it = ledger.units.find(key);
  if (it != ledger.units.end() && it->second.retired && !it->second.retained &&
      it->second.permit.expired())
    ledger.units.erase(it);
}
}  // namespace
struct permit_state {
  std::shared_ptr<ledger_state> ledger;
  unit_key key;
  uint64_t temporary = 0;
  bool accepting     = true;
  ~permit_state()
  {
    std::lock_guard lock(ledger->mutex);
    --ledger->active;
    erase_retired(*ledger, key);
  }
};
namespace {
struct allocation_owner {
  std::shared_ptr<ledger_state> ledger;
  std::shared_ptr<permit_state> permit;  // Only temporary buffers keep the W permit alive.
  unit_key key;
  uint64_t charge;
  bool retained;
  backing_allocation allocation;
  ~allocation_owner()
  {
    allocation.owner.reset();  // Free physical backing before returning credit.
    std::lock_guard lock(ledger->mutex);
    if (retained) {
      ledger->retained -= charge;
      ledger->units.at(key).retained -= charge;
      erase_retired(*ledger, key);
    } else
      permit->temporary -= charge;
  }
};
}  // namespace
admission_decision preparation_ledger::admit(std::span<scan_envelope const> scans,
                                             memory_space_id space)
{
  if (admitted_) throw std::logic_error("preparation admission is single use");
  admitted_ = true;
  admission_decision d;
  if (space.tier != cucascade::memory::Tier::HOST)
    throw std::invalid_argument("preparation requires HOST");
  try {
    auto quantum = provider_.allocation_granularity(space);
    for (auto const& scan : scans) {
      if (!scan.qualified) return d;
      d.sigma_retained = add(d.sigma_retained, round(scan.retained_descriptors, quantum));
      d.sigma_retained = add(d.sigma_retained, round(scan.resolver_index, quantum));
      for (auto const& unit : scan.units) {
        auto unit_w = add(add(round(unit.blob, quantum), round(unit.footer_parser, quantum)),
                          round(unit.roaring, quantum));
        if (unit.positions && !unit_w) return d;
        d.sigma_retained = add(d.sigma_retained, round(unit.positions, quantum));
        d.w              = std::max(d.w, unit_w);
      }
    }
    auto requested = add(d.sigma_retained, d.w);
    state_         = std::make_shared<ledger_state>();
    if (requested) {
      auto grant = provider_.request(space, requested);
      if (!grant) {
        d.reason = admission_reason::no_grant;
        state_.reset();
        return d;
      }
      d.c_obtained = grant->bytes;
      if (!grant->handle || grant->space != space || grant->bytes < requested) {
        provider_.release(std::move(*grant));
        state_.reset();
        d.reason = admission_reason::short_grant;
        return d;
      }
      state_->grant = std::move(grant);
    }
    d.n_permits =
      d.w ? static_cast<uint32_t>(std::min<uint64_t>((d.c_obtained - d.sigma_retained) / d.w,
                                                     std::numeric_limits<uint32_t>::max()))
          : 0;
    state_->retained_limit = d.sigma_retained;
    state_->w              = d.w;
    state_->permits        = d.n_permits;
    d.deferred             = true;
    d.route_reason         = scan_eligibility::eligible;
    d.reason               = admission_reason::admitted;
    return d;
  } catch (std::overflow_error const&) {
    state_.reset();
    d.reason = admission_reason::overflow;
    return d;
  } catch (std::bad_alloc const&) {
    state_.reset();
    d.reason = admission_reason::no_grant;
    return d;
  }
}
void preparation_ledger::register_unit(unit_key key, uint64_t retained_limit)
{
  if (!state_) throw std::logic_error("preparation is not admitted");
  std::lock_guard lock(state_->mutex);
  if (state_->closed) throw std::logic_error("preparation closed");
  auto charge = retained_limit && state_->grant ? state_->grant->handle->charge_size(retained_limit)
                                                : retained_limit;
  if (state_->units.contains(key)) throw std::logic_error("duplicate preparation unit");
  if (charge > state_->retained_limit - state_->registered)
    throw std::invalid_argument("unit retained budgets exceed admission");
  state_->units.emplace(key, unit_account{charge, 0, {}});
  state_->registered += charge;
}
void preparation_ledger::retire_unit(unit_key key) noexcept
{
  if (!state_) return;
  std::lock_guard lock(state_->mutex);
  auto it = state_->units.find(key);
  if (it == state_->units.end()) return;
  it->second.retired = true;
  erase_retired(*state_, key);
}
size_t preparation_ledger::tracked_units() const
{
  if (!state_) return 0;
  std::lock_guard lock(state_->mutex);
  return state_->units.size();
}
w_permit preparation_ledger::acquire_permit(unit_key key)
{
  w_permit result;
  if (!state_) return result;
  std::lock_guard lock(state_->mutex);
  if (state_->closed || !state_->w || state_->active == state_->permits) return result;
  auto it = state_->units.find(key);
  if (it == state_->units.end()) throw std::logic_error("unregistered preparation unit");
  if (it->second.retired || !it->second.permit.expired()) return result;
  auto permit    = std::make_shared<permit_state>();
  permit->ledger = state_;
  permit->key    = key;
  ++state_->active;
  it->second.permit = permit;
  result.state_     = std::move(permit);
  return result;
}
w_permit::~w_permit() { release(); }
w_permit& w_permit::operator=(w_permit&& other) noexcept
{
  if (this != &other) {
    release();
    state_ = std::move(other.state_);
  }
  return *this;
}
void w_permit::release() noexcept
{
  if (!state_) return;
  {
    std::lock_guard lock(state_->ledger->mutex);
    state_->accepting = false;
  }
  state_.reset();
}
charging_allocator preparation_ledger::allocator(w_permit const& permit)
{
  if (!permit.state_ || permit.state_->ledger != state_)
    throw std::invalid_argument("foreign or empty W permit");
  return charging_allocator(permit.state_);
}
charged_block charging_allocator::allocate(uint64_t bytes) { return allocate_impl(bytes, false); }
charged_block charging_allocator::allocate_retained(uint64_t bytes)
{
  return allocate_impl(bytes, true);
}
charged_block charging_allocator::allocate_impl(uint64_t bytes, bool retained)
{
  auto permit = state_.lock();
  if (!permit) throw std::logic_error("released W permit");
  auto ledger     = permit->ledger;
  uint64_t charge = 0;
  {
    std::lock_guard lock(ledger->mutex);
    if (ledger->closed || !permit->accepting)
      throw std::logic_error("preparation allocation closed");
    if (!bytes) return {};
    auto& unit    = ledger->units.at(permit->key);
    bool overflow = false;
    try {
      charge = ledger->grant->handle->charge_size(bytes);
    } catch (std::overflow_error const&) {
      overflow = true;
    }
    if (overflow || charge < bytes ||
        (retained ? charge > unit.retained_limit - unit.retained ||
                      charge > ledger->retained_limit - ledger->retained
                  : charge > ledger->w - permit->temporary)) {
      ++ledger->exceeded;
      throw preparation_resource_error("preparation allocation exceeds proved envelope", true);
    }
    if (retained) {
      ledger->retained += charge;
      unit.retained += charge;
    } else
      permit->temporary += charge;
  }
  try {
    auto allocation = ledger->grant->handle->allocate(bytes);
    if (!allocation.owner || !allocation.data) throw std::bad_alloc();
    auto owner        = std::make_shared<allocation_owner>();
    owner->ledger     = ledger;
    owner->permit     = retained ? nullptr : permit;
    owner->key        = permit->key;
    owner->charge     = charge;
    owner->retained   = retained;
    owner->allocation = std::move(allocation);
    charged_block block;
    block.data_     = owner->allocation.data;
    block.size_     = bytes;
    block.retained_ = retained;
    block.owner_    = std::move(owner);
    return block;
  } catch (...) {
    auto original = std::current_exception();
    std::lock_guard lock(ledger->mutex);
    if (retained) {
      ledger->retained -= charge;
      ledger->units.at(permit->key).retained -= charge;
    } else
      permit->temporary -= charge;
    ++ledger->allocation_failures;
    throw preparation_resource_error("preparation backing allocation failed", false, original);
  }
}
uint64_t preparation_ledger::charged_bytes(unit_key key) const
{
  if (!state_) return 0;
  std::lock_guard lock(state_->mutex);
  auto it = state_->units.find(key);
  return it == state_->units.end() ? 0 : it->second.retained;
}
uint64_t preparation_ledger::granted_capacity() const
{
  return state_ && state_->grant ? state_->grant->bytes : 0;
}
uint32_t preparation_ledger::permits_in_flight() const
{
  if (!state_) return 0;
  std::lock_guard lock(state_->mutex);
  return state_->active;
}
uint64_t preparation_ledger::bound_exceeded() const
{
  if (!state_) return 0;
  std::lock_guard lock(state_->mutex);
  return state_->exceeded;
}
uint64_t preparation_ledger::allocation_failures() const
{
  if (!state_) return 0;
  std::lock_guard lock(state_->mutex);
  return state_->allocation_failures;
}
void preparation_ledger::close() noexcept
{
  if (state_) {
    std::lock_guard lock(state_->mutex);
    state_->closed = true;
  }
}
preparation_admission::preparation_admission(std::unique_ptr<preparation_ledger> ledger,
                                             admission_decision d)
  : decision(d), ledger_(std::move(ledger))
{
  if (!decision.deferred || !ledger_)
    throw std::invalid_argument("owning admission requires an admitted ledger");
}
std::unique_ptr<preparation_ledger> preparation_admission::consume()
{
  std::lock_guard lock(mutex_);
  if (consumed_) throw std::logic_error("preparation admission already consumed");
  consumed_ = true;
  return std::move(ledger_);
}
void preparation_admission::finish() noexcept
{
  std::lock_guard lock(mutex_);
  consumed_ = true;
  ledger_.reset();
}
bool statement_dv_route_allowed(std::span<scan_dv_count const> scans)
{
  return statement_dv_route_allowed(scans, op::scan::kMaxDeletionVectorPositionsPerStatement);
}
bool statement_dv_route_allowed(std::span<scan_dv_count const> scans, uint64_t limit)
{
  std::map<op::scan::scan_contract_id, uint64_t> seen;
  uint64_t total = 0;
  for (auto const& scan : scans) {
    auto [it, inserted] = seen.emplace(scan.contract, scan.live_positions);
    if (!inserted) {
      if (it->second != scan.live_positions) return false;
      continue;
    }
    if (total > limit || scan.live_positions > limit - total) return false;
    total += scan.live_positions;
  }
  return true;
}
std::optional<uint64_t> footer_envelope(uint64_t file_size, envelope_bound const& proof)
{
  return proof ? proof(file_size, 0) : std::nullopt;
}
std::optional<uint64_t> roaring_envelope(uint64_t encoded_size,
                                         uint64_t record_count,
                                         envelope_bound const& proof)
{
  return proof ? proof(encoded_size, record_count) : std::nullopt;
}
}  // namespace sirius::scan_manager
