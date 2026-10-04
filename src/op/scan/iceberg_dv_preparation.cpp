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
struct iceberg_dv_preparation::storage {
  scan_manager::charged_block arena;
  size_t count = 0;
  ~storage() { std::destroy_n(reinterpret_cast<file*>(arena.data()), count); }
};
iceberg_dv_preparation::iceberg_dv_preparation(
  scan_contract_id id,
  std::string const& table,
  std::vector<std::string> const& paths,
  std::shared_ptr<iceberg_delete_discovery const> discovery,
  scan_manager::preparation_ledger& owner)
  : contract(id), inventory_(std::move(discovery))
{
  auto bytes = arena_bytes(paths);
  owner.register_unit({id, 0}, bytes);
  auto permit = owner.acquire_permit({id, 0});
  if (bytes && !permit) throw std::logic_error("descriptor arena requires the admitted W permit");
  storage_ = std::make_shared<storage>();
  if (bytes) storage_->arena = owner.allocator(permit).allocate_retained(bytes);
  auto* records      = reinterpret_cast<file*>(storage_->arena.data());
  auto* index        = records ? reinterpret_cast<file**>(records + paths.size()) : nullptr;
  auto* chars        = index ? reinterpret_cast<char*>(index + paths.size()) : nullptr;
  size_t constructed = 0;
  try {
    for (auto const& path : paths) {
      std::memcpy(chars, path.c_str(), path.size() + 1);
      std::construct_at(records + constructed,
                        file{std::string_view(chars, path.size()), nullptr, {}});
      index[constructed] = records + constructed;
      chars += path.size() + 1;
      ++constructed;
      storage_->count = constructed;
    }
    files = {records, paths.size()};
    std::span<file*> sorted(index, paths.size());
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
    files = {};
    throw;
  }
}
iceberg_dv_preparation::~iceberg_dv_preparation() = default;
std::shared_ptr<void const> iceberg_dv_preparation::path_owner() const { return storage_; }
void iceberg_dv_preparation::finish() noexcept
{
  files = {};
  storage_.reset();
  ledger.reset();
}
}  // namespace sirius::op::scan
