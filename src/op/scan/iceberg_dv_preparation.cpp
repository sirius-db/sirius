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

#include "op/scan/iceberg_dv_preparation.hpp"

#include "op/scan/puffin_reader.hpp"
#include "yyjson.hpp"

#include <duckdb/common/exception.hpp>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <new>

namespace sirius::op::scan {
namespace {
using result_slot = std::weak_ptr<iceberg_delete_set const>;
uint64_t add(uint64_t a, uint64_t b)
{
  if (b > std::numeric_limits<uint64_t>::max() - a)
    throw std::overflow_error("DV envelope overflow");
  return a + b;
}
uint64_t arena_bytes(std::vector<std::string> const& paths)
{
  if (paths.size() > std::numeric_limits<uint64_t>::max() /
                       (sizeof(iceberg_dv_preparation::file) +
                        sizeof(iceberg_dv_preparation::file*) + sizeof(result_slot)))
    throw std::overflow_error("DV resolver overflow");
  auto n = paths.size() * (sizeof(iceberg_dv_preparation::file) +
                           sizeof(iceberg_dv_preparation::file*) + sizeof(result_slot));
  for (auto const& p : paths)
    n = add(n, add(p.size(), 1));
  return n;
}
std::string_view bare(std::string_view path)
{
  return path.starts_with("file://") ? path.substr(7) : path;
}
std::string_view basename(std::string_view path)
{
  return path.substr(path.find_last_of('/') == std::string_view::npos ? 0
                                                                      : path.find_last_of('/') + 1);
}
bool same_file(std::string_view a, std::string_view b)
{
  a = bare(a);
  b = bare(b);
  if (a == b) return true;
  auto longer  = a.size() >= b.size() ? a : b;
  auto shorter = a.size() >= b.size() ? b : a;
  return !shorter.empty() && longer.ends_with(shorter) &&
         longer[longer.size() - shorter.size() - 1] == '/';
}
}  // namespace
scan_manager::scan_envelope iceberg_dv_preparation::envelope(
  iceberg_delete_discovery const& discovery, std::vector<std::string> const& paths)
{
  scan_manager::scan_envelope out;
  if (!discovery.positional_delete_files.empty() || !discovery.equality_delete_entries.empty())
    return out;
  if (discovery.deletion_vector_entries.empty()) {
    out.qualified = true;
    return out;
  }
  out.retained.push_back(arena_bytes(paths));
  for (auto const& dv : discovery.deletion_vector_entries) {
    if (dv.record_count == 0 || dv.file_size_in_bytes < 20 || !dv.has_complete_descriptor() ||
        !dv.has_decodable_record_count())
      return out;
    auto s = static_cast<uint64_t>(dv.file_size_in_bytes);
    if (add(add(static_cast<uint64_t>(dv.content_offset),
                static_cast<uint64_t>(dv.content_size_in_bytes)),
            20) > s)
      return out;
    auto json = duckdb_yyjson::yyjson_read_max_memory_usage(s, 0);
    if (!json) return out;
    // yyjson has one local pool, with text rounded up for its alignment.
    out.units.push_back({{static_cast<uint64_t>(dv.content_size_in_bytes), add(add(s, 15), json)},
                         static_cast<uint64_t>(dv.record_count) * sizeof(int64_t)});
  }
  out.qualified = true;
  return out;
}
// Result objects own their lifetime independently of descriptors while sharing reserved backing.
struct iceberg_dv_results {
  iceberg_dv_results(std::shared_ptr<void const> backing, result_slot* data, size_t count)
    : backing(std::move(backing)), slots(data, count)
  {
    std::uninitialized_value_construct_n(slots.data(), slots.size());
  }
  ~iceberg_dv_results() { std::destroy_n(slots.data(), slots.size()); }
  iceberg_dv_results(iceberg_dv_results const&)            = delete;
  iceberg_dv_results& operator=(iceberg_dv_results const&) = delete;
  std::shared_ptr<void const> backing;
  std::span<result_slot> slots;
};
// One charged block holds descriptors, the resolver index, reserved result slots and paths.
// This owner constructs only descriptors; iceberg_dv_results manages the result objects.
struct iceberg_dv_preparation::storage {
  storage(scan_manager::charged_block block, size_t capacity)
    : arena_(std::move(block)), capacity_(capacity)
  {
    if (!capacity_) return;
    auto* base = arena_.data();
    if (!base || reinterpret_cast<std::uintptr_t>(base) % alignof(file) != 0 ||
        (capacity_ * sizeof(file)) % alignof(file*) != 0 ||
        (capacity_ * (sizeof(file) + sizeof(file*))) % alignof(result_slot) != 0)
      throw std::logic_error("descriptor arena is not aligned for its records");
    records_   = reinterpret_cast<file*>(base);
    index_     = reinterpret_cast<file**>(records_ + capacity_);
    results_   = reinterpret_cast<result_slot*>(index_ + capacity_);
    chars_     = reinterpret_cast<char*>(results_ + capacity_);
    chars_end_ = reinterpret_cast<char*>(base + arena_.size());
  }
  ~storage() { std::destroy_n(records_, count_); }
  storage(storage const&)            = delete;
  storage& operator=(storage const&) = delete;
  void emplace(std::string_view path)
  {
    if (count_ == capacity_ || static_cast<size_t>(chars_end_ - chars_) < path.size() + 1)
      throw std::logic_error("descriptor arena is smaller than its paths");
    std::memcpy(chars_, path.data(), path.size());
    chars_[path.size()] = '\0';
    index_[count_] =
      std::construct_at(records_ + count_, file{std::string_view(chars_, path.size()), nullptr});
    chars_ += path.size() + 1;
    ++count_;
  }
  std::span<file> records() noexcept { return {records_, count_}; }
  std::span<file*> index() noexcept { return {index_, count_}; }
  result_slot* results() noexcept { return results_; }

 private:
  scan_manager::charged_block arena_;
  size_t capacity_;
  size_t count_         = 0;
  file* records_        = nullptr;
  file** index_         = nullptr;
  result_slot* results_ = nullptr;
  char* chars_          = nullptr;
  char* chars_end_      = nullptr;
};
iceberg_dv_preparation::iceberg_dv_preparation(
  scan_contract_id id,
  std::string const& table,
  std::vector<std::string> const& paths,
  std::shared_ptr<iceberg_delete_discovery const> discovery,
  scan_manager::preparation_ledger& owner)
  : contract_(id), inventory_(std::move(discovery))
{
  auto bytes = arena_bytes(paths);
  owner.register_unit({id, 0}, bytes);
  auto permit = owner.acquire_permit({id, 0});
  if (bytes && !permit)
    throw std::logic_error("descriptor arena requires an admitted temporary-memory permit");
  storage_ = std::make_shared<storage>(
    bytes ? owner.allocator(permit).allocate_retained(bytes) : scan_manager::charged_block{},
    paths.size());
  results_ = std::make_unique<iceberg_dv_results>(storage_, storage_->results(), paths.size());
  try {
    for (auto const& path : paths)
      storage_->emplace(path);
    files_      = storage_->records();
    auto sorted = storage_->index();
    std::sort(sorted.begin(), sorted.end(), [](auto* a, auto* b) {
      return basename(a->path) < basename(b->path);
    });
    for (auto const& dv : inventory_->deletion_vector_entries) {
      file* match = nullptr;
      auto name   = basename(dv.referenced_data_file);
      auto it =
        std::lower_bound(sorted.begin(), sorted.end(), name, [](auto* f, std::string_view key) {
          return basename(f->path) < key;
        });
      for (; it != sorted.end() && basename((*it)->path) == name; ++it)
        if (same_file(dv.referenced_data_file, (*it)->path)) {
          if (match)
            throw duckdb::NotImplementedException(
              "iceberg table '{}': delete file entry '{}' matches more than one scanned data file, "
              "so its deleted rows cannot be attributed",
              table,
              dv.referenced_data_file);
          match = *it;
        }
      // Unscanned data files do not contribute inputs or cause Puffin reads.
      if (!match) continue;
      if (match->dv)
        throw duckdb::NotImplementedException(
          "iceberg table '{}': scanned data file '{}' is named by {} different manifest entries "
          "(including '{}' and '{}'), so its deletes cannot be attributed to one of them",
          table,
          std::string(match->path),
          2,
          match->dv->referenced_data_file,
          dv.referenced_data_file);
      match->dv = &dv;
    }
    permit.release();
    owner.retire_unit({id, 0});
  } catch (...) {
    files_ = {};
    throw;
  }
}
iceberg_dv_preparation::~iceberg_dv_preparation() = default;
std::shared_ptr<void const> iceberg_dv_preparation::path_owner() const { return storage_; }
void iceberg_dv_preparation::bind_admission(
  std::shared_ptr<scan_manager::preparation_admission> admission)
{
  if (!admission) throw std::logic_error("DV preparation requires a statement admission");
  if (admission_) throw std::logic_error("DV preparation already bound to a statement admission");
  admission_ = std::move(admission);
}
void iceberg_dv_preparation::release() noexcept
{
  files_ = {};
  results_.reset();
  storage_.reset();
  // The plan no longer needs to retain the statement admission once its
  // descriptors have been retired.  Reset only this plan's reference: a
  // shared admission remains alive through any other scan that still owns it.
  admission_.reset();
}

iceberg_dv_attempt::iceberg_dv_attempt(std::shared_ptr<iceberg_dv_preparation> plan,
                                       std::shared_ptr<scan_manager::preparation_ledger> ledger)
  : plan_(std::move(plan)), ledger_(std::move(ledger))
{
  if (!plan_ || !ledger_) throw std::logic_error("DV attempt requires a plan and a ledger");
  if (!plan_->results_) throw std::logic_error("DV result storage already consumed");
  results_ = std::move(plan_->results_);
}
iceberg_dv_attempt::~iceberg_dv_attempt() = default;
std::shared_ptr<iceberg_delete_set const> iceberg_dv_attempt::result(
  size_t index, std::string_view path) const noexcept
{
  std::lock_guard lock(mutex_);
  auto files = plan_->files();
  return results_ && index < files.size() && files[index].path == path
           ? results_->slots[index].lock()
           : nullptr;
}
void iceberg_dv_attempt::bind_wakeup(scan_manager::preparation_wakeup wake)
{
  std::shared_ptr<scan_manager::preparation_ledger> ledger;
  {
    std::lock_guard lock(mutex_);
    ledger = ledger_;
  }
  if (ledger) ledger->bind_wakeup(std::move(wake));
}
void iceberg_dv_attempt::stop() noexcept
{
  std::lock_guard lock(mutex_);
  stopped_ = true;
  ledger_.reset();
}
iceberg_dv_attempt::file_job::file_job(std::shared_ptr<iceberg_dv_attempt> attempt, size_t index)
  : attempt_(std::move(attempt)),
    index_(index),
    descriptor_(attempt_->plan_->files()[index]),
    storage_(attempt_->plan_->path_owner()),
    ledger_(attempt_->ledger_)
{
}
scan_manager::unit_key iceberg_dv_attempt::file_job::key() const noexcept
{
  return {attempt_->plan_->contract(), index_ + 1};
}
iceberg_dv_attempt::file_job::~file_job()
{
  {
    std::lock_guard lock(attempt_->mutex_);
    if (!committed_) attempt_->claiming_ = false;
    // Stop enumeration after an abandoned claim, without invalidating running siblings.
    // The coordinator owns cancellation and rejects publication after normal EOS.
    if (accepted_ && !completed_) attempt_->claims_abandoned_ = true;
  }
  permit_.release();
  if (registered_) ledger_->retire_unit(key());
}
void iceberg_dv_attempt::file_job::commit()
{
  std::lock_guard lock(attempt_->mutex_);
  if (attempt_->stopped_ || committed_ || attempt_->next_ != index_)
    throw std::logic_error("invalid DV claim commit");
  ++attempt_->next_;
  attempt_->claiming_ = false;
  committed_          = true;
}
void iceberg_dv_attempt::file_job::complete(std::shared_ptr<iceberg_delete_set const> const& result)
{
  std::lock_guard lock(attempt_->mutex_);
  if (!committed_ || completed_ || attempt_->stopped_)
    throw std::logic_error("invalid DV job completion");
  attempt_->results_->slots[index_] = result;
  completed_                        = true;
}
scan_manager::preparation_claim<std::shared_ptr<iceberg_dv_attempt::file_job>>
iceberg_dv_attempt::try_claim()
{
  std::unique_lock lock(mutex_);
  if (stopped_ || claims_abandoned_) throw std::logic_error("DV preparation attempt stopped");
  if (claiming_)
    throw std::logic_error("DV claim must be committed or abandoned before next claim");
  if (next_ == plan_->files().size()) return scan_manager::preparation_exhausted{};
  // Allocate before admission. The owning job is destroyed only after unlocking this owner.
  auto allocation = std::unique_ptr<file_job>(new file_job(shared_from_this(), next_));
  claiming_       = true;
  lock.unlock();
  auto owned = std::shared_ptr<file_job>(std::move(allocation));
  if (auto* dv = owned->descriptor().dv) {
    owned->permit_ =
      owned->ledger_->try_acquire_unit(owned->key(), dv->record_count * sizeof(int64_t));
    if (!owned->permit_) return scan_manager::preparation_pending{};
    owned->registered_ = true;
  }
  lock.lock();
  if (stopped_) {
    lock.unlock();
    throw std::logic_error("DV preparation stopped during claim");
  }
  owned->accepted_ = true;
  lock.unlock();
  return owned;
}
void iceberg_dv_attempt::finish() noexcept
{
  std::lock_guard lock(mutex_);
  stopped_ = true;
  ledger_.reset();
  results_.reset();
  plan_->release();
}
}  // namespace sirius::op::scan
