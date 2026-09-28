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

#include <cstddef>
#include <string>
#include <string_view>
#include <unordered_map>

namespace sirius::io::cache {

/// Which cache, if any, the read path goes through.
enum class cache_mode {
  /// O_DIRECT reads, no cache anywhere.
  none,
  /// Buffered reads through the OS page cache; Sirius keeps no cache of its own.
  os,
  /// O_DIRECT reads into Sirius's own pinned prefetching cache.
  sirius,
};

/// What retires a chunk from the Sirius cache once nothing is reading it.
/// Only meaningful under @ref cache_mode::sirius.
enum class eviction_policy {
  /// Drop a chunk as soon as it goes idle: the cache is a prefetch staging area,
  /// sized for the reads in flight rather than for reuse.
  idle,
  /// Keep idle chunks for reuse and evict least-recently-used ones once the pool
  /// fills past @c eviction_threshold_fraction.
  lru,
};

/// Parse a @ref cache_mode from its lowercase YAML spelling.
inline bool string_to_enum(std::string_view sv, cache_mode& out)
{
  static const std::unordered_map<std::string_view, cache_mode> map = {
    {"none", cache_mode::none},
    {"os", cache_mode::os},
    {"sirius", cache_mode::sirius},
  };
  auto it = map.find(sv);
  if (it == map.end()) { return false; }
  out = it->second;
  return true;
}

/// Render a @ref cache_mode as its canonical lowercase name.
inline bool enum_to_string(cache_mode mode, std::string& out)
{
  switch (mode) {
    case cache_mode::none: out = "none"; return true;
    case cache_mode::os: out = "os"; return true;
    case cache_mode::sirius: out = "sirius"; return true;
  }
  return false;
}

/// Parse an @ref eviction_policy from its lowercase YAML spelling.
inline bool string_to_enum(std::string_view sv, eviction_policy& out)
{
  static const std::unordered_map<std::string_view, eviction_policy> map = {
    {"idle", eviction_policy::idle},
    {"lru", eviction_policy::lru},
  };
  auto it = map.find(sv);
  if (it == map.end()) { return false; }
  out = it->second;
  return true;
}

/// Render an @ref eviction_policy as its canonical lowercase name.
inline bool enum_to_string(eviction_policy policy, std::string& out)
{
  switch (policy) {
    case eviction_policy::idle: out = "idle"; return true;
    case eviction_policy::lru: out = "lru"; return true;
  }
  return false;
}

/**
 * @brief The read path's caching configuration — the whole of it.
 *
 * @c mode and @c eviction are the two knobs; everything the read path derives
 * from caching (O_DIRECT vs buffered reads, whether the prefetching cache is
 * armed, whether idle chunks are dropped) follows from them, so those derived
 * settings are not separately configurable. The remaining fields size the
 * prefetching cache and only matter under @ref cache_mode::sirius.
 */
struct config {
  /// Which cache the read path goes through.
  cache_mode mode{cache_mode::none};

  /// What retires an idle chunk from the Sirius cache.
  eviction_policy eviction{eviction_policy::lru};

  /// Floor of the cache pool reserved for prefetching, as a fraction of the pool.
  double min_prefetching_budget_fraction{0.05};

  /// Start evicting once the pool fills to this fraction of its capacity.
  double eviction_threshold_fraction{0.8};

  /// Derived from @ref eviction by @ref apply_mode; not settable on its own.
  bool dispose_on_idle{false};

  /// Any caching at all — false only for @ref cache_mode::none. Reads that are
  /// cached somewhere (here or in the page cache) are worth ordering ahead of
  /// demand, so this is also what gates the readahead by default.
  [[nodiscard]] bool enabled() const noexcept { return mode != cache_mode::none; }

  /// Whether reads are served through Sirius's own pinned prefetching cache.
  [[nodiscard]] bool use_prefetching_cache() const noexcept { return mode == cache_mode::sirius; }

  /// Whether the local backend reads with O_DIRECT: everything but @c os, which
  /// exists precisely to read through the kernel page cache.
  [[nodiscard]] bool use_odirect() const noexcept { return mode != cache_mode::os; }

  /// Refresh the knobs derived from @ref mode / @ref eviction.
  void apply_mode() noexcept { dispose_on_idle = eviction == eviction_policy::idle; }
};

}  // namespace sirius::io::cache
