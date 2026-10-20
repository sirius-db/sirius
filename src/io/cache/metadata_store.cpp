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
    it->second.generation = std::max(it->second.generation, obj.open_generation());
    return;
  } else {
    if (obj.open_generation() <= it->second.generation) return;
    it->second = observation{std::move(identity), obj.open_generation()};
  }
  for (auto entry = _entries.begin(); entry != _entries.end();) {
    auto current = entry++;
    if (current->path != obj.object_path()) continue;
    ++_stats.replacements;
    retire(current, retired);
  }
}

std::shared_ptr<io_object_metadata> metadata_store::get_or_load(
  io_object const& obj,
  std::function<std::shared_ptr<io_object_metadata>()> const& loader,
  bool* cache_hit,
  std::stop_token stop,
  std::string_view profile)
{
  if (stop.stop_requested())
    throw std::system_error(std::make_error_code(std::errc::operation_canceled));
  open_observed(obj);
  auto const lookup   = cache_key(obj.object_path(), profile);
  auto const identity = obj.identity();
  // A local stat identity is obtained from the opened descriptor and may be
  // shared by separate opens while the stat tuple is unchanged. Keep the
  // generation as an ordering fence for stale publications, but do not let it
  // defeat coalescing of two opens of the same file.
  auto const identity_key = identity.kind == identity_kind::local_stat
                              ? identity.cache_key(obj.raw_file_cache_id())
                              : obj.identity_cache_key();
  auto const flight_key   = identity_key + ":" + lookup;
  std::shared_ptr<flight> work;
  {
    std::unique_lock lock{_mutex};
    auto found = _index.find(lookup);
    if (found != _index.end() && found->second->identity == identity) {
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

std::string metadata_store::cache_key(std::string const& path, std::string_view profile)
{
  if (profile.empty()) return path;
  return path + "\nprofile=" + std::string(profile);
}

void metadata_store::retire(iterator it, entries& retired)
{
  --_stats.entries;
  _index.erase(it->key);
  retired.splice(retired.end(), _entries, it);
}

void metadata_store::clear()
{
  entries retired;
  {
    std::lock_guard lock{_mutex};
    for (auto it = _entries.begin(); it != _entries.end();) {
      auto current = it++;
      retire(current, retired);
    }
  }
}

bool metadata_store::invalidate(std::string_view path, std::string_view profile)
{
  entries retired;
  std::lock_guard lock{_mutex};
  auto const lookup = cache_key(std::string(path), profile);
  auto const found  = _index.find(lookup);
  if (found == _index.end()) return false;
  retire(found->second, retired);
  return true;
}

bool metadata_store::put_identity(std::string const& path,
                                  object_identity identity,
                                  uint64_t generation,
                                  std::shared_ptr<io_object_metadata> value,
                                  std::string_view profile,
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
    if (it->identity == identity) {
      // A duplicate publication still advances the local ordering fence.
      // Otherwise an older, different-version fill could replace this record.
      it->generation = std::max(it->generation, generation);
      defer_value();
      return true;
    }
    ++_stats.replacements;
    retire(it, retired);
  }
  _entries.push_front({std::move(lookup), path, generation, std::move(identity), std::move(value)});
  try {
    _index.emplace(_entries.front().key, _entries.begin());
  } catch (...) {
    retired.splice(retired.end(), _entries, _entries.begin());
    throw;
  }
  ++_stats.entries;
  return true;
}

void metadata_store::register_metadata(io_object const& obj,
                                       std::shared_ptr<io_object_metadata> metadata,
                                       std::string_view profile)
{
  if (!metadata) return;
  entries retired;
  std::unique_lock lock{_mutex};
  observe_locked(obj, retired);
  if (!put_identity(obj.object_path(),
                    obj.identity(),
                    obj.open_generation(),
                    std::move(metadata),
                    profile,
                    retired)) {
    return;
  }
}
std::shared_ptr<io_object_metadata> metadata_store::get_metadata(io_object const& obj,
                                                                 std::string_view profile) const
{
  auto lookup   = cache_key(obj.object_path(), profile);
  auto identity = obj.identity();
  std::lock_guard lock{_mutex};
  auto found = _index.find(lookup);
  if (found != _index.end()) {
    auto it = found->second;
    if (it->identity == identity) {
      ++_stats.hits;
      return it->value;
    }
  }
  ++_stats.misses;
  return {};
}
bool metadata_store::has_path(std::string_view path) const
{
  std::lock_guard lock{_mutex};
  auto const it = _index.find(cache_key(std::string(path)));
  if (it == _index.end()) return false;
  // An opened-handle identity is scoped to one datasource and cannot justify
  // replacing a footer probe for a later open. Strong object tags and local
  // stat identities are reusable candidates; the full identity check still
  // happens after the datasource is opened.
  return it->second->identity.kind != identity_kind::opened_handle;
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
