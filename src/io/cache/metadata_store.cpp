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

#include "io/cache/metadata_store.hpp"

#include <algorithm>
#include <iterator>
#include <sstream>
#include <system_error>
#include <utility>

namespace sirius::io::cache {

metadata_store::~metadata_store() { clear(); }

void metadata_store::open_observed(io_object const& obj)
{
  entries retired;
  std::lock_guard lock{_mutex};
  observe_locked(obj, retired);
}

void metadata_store::observe_locked(io_object const& obj, entries& retired)
{
  auto identity = obj.identity();
  if (identity.kind == identity_kind::opened_handle) return;
  auto [it, inserted] =
    _observed.try_emplace(obj.object_path(), observation{identity, obj.open_generation()});
  if (inserted) return;
  if (it->second.identity == identity) {
    if (identity.kind == identity_kind::local_stat &&
        obj.open_generation() > it->second.generation) {
      it->second.generation = obj.open_generation();
    }
    it->second.generation = std::max(it->second.generation, obj.open_generation());
    return;
  } else {
    if (obj.open_generation() <= it->second.generation) return;
    it->second = observation{std::move(identity), obj.open_generation()};
  }
  auto const candidate = cache_key(obj.object_path(), "");
  for (;;) {
    auto candidates = _candidate_entries.find(candidate);
    if (candidates == _candidate_entries.end() || candidates->second.empty()) break;
    auto const key = *candidates->second.begin();
    auto current   = _index.find(key);
    if (current == _index.end()) {
      candidates->second.erase(key);
      if (candidates->second.empty()) _candidate_entries.erase(candidates);
      continue;
    }
    ++_stats.replacements;
    retire(current->second, retired);
  }
  auto path = _legacy_key_by_path.find(obj.object_path());
  if (path != _legacy_key_by_path.end()) {
    auto legacy = _legacy_by_key.find(path->second);
    if (legacy != _legacy_by_key.end()) {
      retired.push_back({{}, {}, legacy->second.generation, {}, std::move(legacy->second.value)});
      _legacy_by_key.erase(legacy);
    }
    _legacy_key_by_path.erase(path);
  }
}

std::shared_ptr<io_object_metadata> metadata_store::get_or_load(
  io_object const& obj,
  std::string const& profile,
  std::function<std::shared_ptr<io_object_metadata>()> const& loader,
  bool* cache_hit,
  std::stop_token stop)
{
  if (stop.stop_requested())
    throw std::system_error(std::make_error_code(std::errc::operation_canceled));
  open_observed(obj);
  auto const lookup     = cache_key(obj.object_path(), profile);
  auto const identity   = obj.identity();
  auto const flight_key = cache_key(obj.identity_cache_key(), lookup);
  std::shared_ptr<flight> work;
  {
    std::unique_lock lock{_mutex};
    auto found = _index.find(lookup);
    if (found != _index.end() && found->second->identity == identity &&
        (identity.kind != identity_kind::local_stat ||
         found->second->generation == obj.open_generation())) {
      ++_stats.hits;
      if (cache_hit) *cache_hit = true;
      return found->second->value;
    }
    ++_stats.misses;
    if (cache_hit) *cache_hit = false;
    auto active = _flights.find(flight_key);
    if (active != _flights.end()) {
      work = active->second;
      ++_stats.joined;
      if (!work->ready.wait(lock, stop, [&] { return work->done; })) {
        ++_stats.cancelled_waiters;
        throw std::system_error(std::make_error_code(std::errc::operation_canceled));
      }
      if (work->error) std::rethrow_exception(work->error);
      return work->value;
    }
    work = std::make_shared<flight>();
    _flights.emplace(flight_key, work);
    ++_stats.builds;
  }
  // No store lock during I/O, parsing, or publication. Waiters hold their own
  // reference to this flight, including when it is removed after an error.
  std::shared_ptr<io_object_metadata> value;
  std::exception_ptr error;
  try {
    value = loader();
    register_metadata(obj, value, profile);
  } catch (...) {
    error = std::current_exception();
  }
  {
    std::lock_guard lock{_mutex};
    work->value = value;
    work->error = error;
    work->done  = true;
    if (error) ++_stats.failures;
    _flights.erase(flight_key);
  }
  work->ready.notify_all();
  if (error) std::rethrow_exception(error);
  return value;
}

std::string metadata_store::cache_key(std::string const& path, std::string const& profile)
{
  return std::to_string(path.size()) + ":" + path + ":" + profile;
}

void metadata_store::retire(iterator it, entries& retired)
{
  --_stats.entries;
  _index.erase(it->key);
  auto candidate = _candidate_entries.find(it->candidate_key);
  if (candidate != _candidate_entries.end()) {
    candidate->second.erase(it->key);
    if (candidate->second.empty()) _candidate_entries.erase(candidate);
  }
  retired.splice(retired.end(), _entries, it);
}

void metadata_store::clear()
{
  entries retired;
  decltype(_legacy_by_key) legacy;
  {
    std::lock_guard lock{_mutex};
    for (auto it = _entries.begin(); it != _entries.end();) {
      auto current = it++;
      retire(current, retired);
    }
    _miss_locks.clear();
    legacy.swap(_legacy_by_key);
    _legacy_key_by_path.clear();
  }
}

bool metadata_store::put_identity(std::string const& path,
                                  std::string const& profile,
                                  object_identity identity,
                                  uint64_t generation,
                                  std::shared_ptr<io_object_metadata> value,
                                  entries& retired)
{
  if (!value) return false;
  if (identity.kind != identity_kind::opened_handle) {
    auto current = _observed.find(path);
    if (current != _observed.end() && current->second.identity != identity) {
      ++_stats.bypasses;
      ++_stats.stale;
      retired.push_back({{}, {}, generation, {}, std::move(value)});
      return false;
    }
  }
  auto lookup = cache_key(path, profile);
  // Open-local evidence cannot help the next open; keep its footer-probe fast path.
  auto candidate_key =
    identity.kind == identity_kind::opened_handle ? std::string{} : cache_key(path, "");
  auto defer_value = [&] {
    // Keep discarded payloads alive until the caller releases _mutex. Metadata
    // destructors are allowed to inspect their owning ioctx.
    retired.push_back({{}, {}, generation, {}, std::move(value)});
  };
  auto found = _index.find(lookup);
  if (found != _index.end()) {
    auto it = found->second;
    if (generation < it->generation) {
      ++_stats.bypasses;
      defer_value();
      return false;
    }
    if (it->identity == identity &&
        (identity.kind != identity_kind::local_stat || it->generation == generation)) {
      // A duplicate publication still advances the local ordering fence.
      // Otherwise an older, different-version fill could replace this record.
      it->generation = generation;
      defer_value();
      return true;
    }
    ++_stats.replacements;
    retire(it, retired);
  }
  _entries.push_front({std::move(lookup),
                       std::move(candidate_key),
                       generation,
                       std::move(identity),
                       std::move(value)});
  try {
    if (!_entries.front().candidate_key.empty()) {
      _candidate_entries[_entries.front().candidate_key].insert(_entries.front().key);
    }
    _index.emplace(_entries.front().key, _entries.begin());
  } catch (...) {
    auto candidate = _candidate_entries.find(_entries.front().candidate_key);
    if (candidate != _candidate_entries.end()) {
      candidate->second.erase(_entries.front().key);
      if (candidate->second.empty()) _candidate_entries.erase(candidate);
    }
    retired.splice(retired.end(), _entries, _entries.begin());
    throw;
  }
  ++_stats.entries;
  return true;
}

void metadata_store::register_metadata(io_object const& obj,
                                       std::shared_ptr<io_object_metadata> metadata,
                                       std::string const& profile)
{
  if (!metadata) return;
  auto legacy_metadata = metadata;
  entries retired;
  std::unique_lock lock{_mutex};
  observe_locked(obj, retired);
  if (!put_identity(obj.object_path(),
                    profile,
                    obj.identity(),
                    obj.open_generation(),
                    std::move(metadata),
                    retired)) {
    return;
  }
  if (profile != "parquet-v1") return;

  auto const& path = obj.object_path();
  auto const& key  = obj.raw_file_cache_id();
  if (auto path_it = _legacy_key_by_path.find(path);
      path_it != _legacy_key_by_path.end() && path_it->second != key) {
    auto previous = _legacy_by_key.find(path_it->second);
    if (previous != _legacy_by_key.end()) {
      retired.push_back(
        {{}, {}, previous->second.generation, {}, std::move(previous->second.value)});
      _legacy_by_key.erase(previous);
    }
    path_it->second = key;
  } else if (path_it == _legacy_key_by_path.end()) {
    _legacy_key_by_path.emplace(path, key);
  }
  auto const current = _legacy_by_key.find(key);
  if (current == _legacy_by_key.end() || obj.open_generation() >= current->second.generation) {
    if (current != _legacy_by_key.end()) {
      retired.push_back({{}, {}, current->second.generation, {}, std::move(current->second.value)});
    }
    _legacy_by_key.insert_or_assign(
      key, legacy_entry{obj.open_generation(), std::move(legacy_metadata)});
  }
}
std::shared_ptr<io_object_metadata> metadata_store::get_metadata(io_object const& obj,
                                                                 std::string const& profile) const
{
  auto lookup   = cache_key(obj.object_path(), profile);
  auto identity = obj.identity();
  std::lock_guard lock{_mutex};
  auto found = _index.find(lookup);
  if (found != _index.end()) {
    auto it = found->second;
    if (it->identity == identity &&
        (identity.kind != identity_kind::local_stat || it->generation == obj.open_generation())) {
      ++_stats.hits;
      return it->value;
    }
  }
  ++_stats.misses;
  return {};
}
std::unique_lock<std::mutex> metadata_store::lock_metadata_miss(io_object const& obj,
                                                                std::string const& profile) const
{
  auto lookup = cache_key(obj.object_path(), profile);
  std::shared_ptr<std::mutex> gate;
  {
    std::lock_guard lock{_mutex};
    auto& slot = _miss_locks[std::move(lookup)];
    if (!slot) slot = std::make_shared<std::mutex>();
    gate = slot;
  }
  return std::unique_lock<std::mutex>{*gate};
}
std::shared_ptr<io_object_metadata> metadata_store::get_metadata(std::string_view cache_key) const
{
  std::lock_guard lock{_mutex};
  auto const it = _legacy_by_key.find(std::string(cache_key));
  return it == _legacy_by_key.end() ? nullptr : it->second.value;
}
bool metadata_store::has_candidate(std::string_view path) const
{
  std::lock_guard lock{_mutex};
  return _candidate_entries.contains(cache_key(std::string(path), ""));
}

metadata_store::counters metadata_store::statistics() const
{
  std::lock_guard lock{_mutex};
  return _stats;
}

std::string metadata_store::describe_metadata_store() const
{
  std::lock_guard lock{_mutex};
  std::ostringstream out;
  out << "entries=" << _stats.entries << " hits=" << _stats.hits << " misses=" << _stats.misses
      << " joined=" << _stats.joined << " builds=" << _stats.builds
      << " failures=" << _stats.failures << " cancelled_waiters=" << _stats.cancelled_waiters
      << " stale=" << _stats.stale << " in_flight=" << _flights.size() << '\n';
  for (auto const& entry : _entries) {
    out << "key=" << entry.key << " identity=" << entry.identity.cache_key("")
        << " generation=" << entry.generation << " holders=" << entry.value.use_count() << '\n';
  }
  return out.str();
}
}  // namespace sirius::io::cache
