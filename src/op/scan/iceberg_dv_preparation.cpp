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
uint64_t add(uint64_t a, uint64_t b)
{
  if (b > std::numeric_limits<uint64_t>::max() - a)
    throw std::overflow_error("DV envelope overflow");
  return a + b;
}
uint64_t arena_bytes(std::vector<std::string> const& paths)
{
  if (paths.size() > std::numeric_limits<uint64_t>::max() / (sizeof(iceberg_dv_preparation::file) +
                                                             sizeof(iceberg_dv_preparation::file*)))
    throw std::overflow_error("DV resolver overflow");
  auto n =
    paths.size() * (sizeof(iceberg_dv_preparation::file) + sizeof(iceberg_dv_preparation::file*));
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
  out.retained_descriptors = arena_bytes(paths);
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
    out.units.push_back({static_cast<uint64_t>(dv.content_size_in_bytes),
                         add(add(s, 15), json),
                         0,
                         static_cast<uint64_t>(dv.record_count) * sizeof(int64_t)});
  }
  out.qualified = true;
  return out;
}
// The retained block from arena_bytes(), carved into the records, the pointer index the DV
// match sorts, and the NUL-terminated path characters the records view. Only the records it
// constructed are destroyed.
struct iceberg_dv_preparation::storage {
  storage(scan_manager::charged_block block, size_t capacity)
    : arena_(std::move(block)), capacity_(capacity)
  {
    if (!capacity_) return;
    auto* base = arena_.data();
    if (!base || reinterpret_cast<std::uintptr_t>(base) % alignof(file) != 0 ||
        (capacity_ * sizeof(file)) % alignof(file*) != 0)
      throw std::logic_error("descriptor arena is not aligned for its records");
    records_   = reinterpret_cast<file*>(base);
    index_     = reinterpret_cast<file**>(records_ + capacity_);
    chars_     = reinterpret_cast<char*>(index_ + capacity_);
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
    index_[count_]      = std::construct_at(records_ + count_,
                                       file{std::string_view(chars_, path.size()), nullptr, {}});
    chars_ += path.size() + 1;
    ++count_;
  }
  std::span<file> records() noexcept { return {records_, count_}; }
  std::span<file*> index() noexcept { return {index_, count_}; }

 private:
  scan_manager::charged_block arena_;
  size_t capacity_;
  size_t count_    = 0;
  file* records_   = nullptr;
  file** index_    = nullptr;
  char* chars_     = nullptr;
  char* chars_end_ = nullptr;
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
  storage_.reset();
}

iceberg_dv_attempt::iceberg_dv_attempt(std::shared_ptr<iceberg_dv_preparation> plan,
                                       std::shared_ptr<scan_manager::preparation_ledger> ledger)
  : plan_(std::move(plan)), ledger_(std::move(ledger))
{
  if (!plan_ || !ledger_) throw std::logic_error("DV attempt requires a plan and a ledger");
}
void iceberg_dv_attempt::record(size_t index, std::shared_ptr<iceberg_delete_set const> const& set)
{
  auto slots = plan_->slots();
  if (index >= slots.size())
    throw std::logic_error("DV result recorded outside the attempt's descriptors");
  slots[index].result = set;
}
std::shared_ptr<iceberg_delete_set const> iceberg_dv_attempt::result(size_t index) const noexcept
{
  auto files = plan_->files();
  return index < files.size() ? files[index].result.lock() : nullptr;
}
void iceberg_dv_attempt::finish() noexcept
{
  stop();
  plan_->release();
}
}  // namespace sirius::op::scan
