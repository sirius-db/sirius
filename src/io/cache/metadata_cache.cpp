/*
 * Copyright 2026, Sirius Contributors.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "io/cache/metadata_cache.hpp"

namespace sirius::io::cache {
metadata_cache::metadata_cache(options) {}
metadata_cache::~metadata_cache() = default;
std::string metadata_cache::key(uint64_t scope, std::string const& path, std::string const& profile)
{
  return std::to_string(scope) + ":" + std::to_string(path.size()) + ":" + path + profile;
}
void metadata_cache::retire(iterator it, entries& retired)
{
  --stats_.entries;
  index_.erase(it->key);
  auto candidate = candidates_.find(it->candidate_key);
  if (candidate != candidates_.end() && --candidate->second == 0) candidates_.erase(candidate);
  retired.splice(retired.end(), lru_, it);
}
std::shared_ptr<io_object_metadata> metadata_cache::get(uint64_t scope,
                                                        std::string const& path,
                                                        std::string const& profile,
                                                        object_identity const& identity)
{
  auto lookup = key(scope, path, profile);
  std::lock_guard lock(mutex_);
  auto found = index_.find(lookup);
  if (found != index_.end()) {
    auto it = found->second;
    if (it->identity == identity) {
      lru_.splice(lru_.begin(), lru_, it);
      ++stats_.hits;
      return it->value;
    }
  }
  ++stats_.misses;
  return {};
}
std::unique_lock<std::mutex> metadata_cache::lock_miss(uint64_t scope,
                                                       std::string const& path,
                                                       std::string const& profile)
{
  auto lookup = key(scope, path, profile);
  std::shared_ptr<std::mutex> gate;
  {
    std::lock_guard lock(mutex_);
    auto& slot = miss_locks_[std::move(lookup)];
    if (!slot) slot = std::make_shared<std::mutex>();
    gate = slot;
  }
  return std::unique_lock<std::mutex>(*gate);
}
bool metadata_cache::contains(uint64_t scope, std::string const& path)
{
  std::lock_guard lock(mutex_);
  return candidates_.contains(key(scope, path, ""));
}
void metadata_cache::put(uint64_t scope,
                         std::string const& path,
                         std::string const& profile,
                         object_identity identity,
                         uint64_t generation,
                         std::shared_ptr<io_object_metadata> value)
{
  if (!value) return;
  auto lookup = key(scope, path, profile);
  // Open-local evidence cannot help the next open; keep its footer-probe fast path.
  auto candidate_key =
    identity.kind == identity_kind::opened_handle ? std::string{} : key(scope, path, "");
  entries retired;
  std::lock_guard lock(mutex_);
  auto found = index_.find(lookup);
  if (found != index_.end()) {
    auto it = found->second;
    if (generation < it->generation) {
      ++stats_.bypasses;
      return;
    }
    if (it->identity == identity) {
      // A duplicate publication still advances the local ordering fence.
      // Otherwise an older, different-version fill could replace this record.
      it->generation = generation;
      return;
    }
    ++stats_.replacements;
    retire(it, retired);
  }
  lru_.push_front({std::move(lookup),
                   std::move(candidate_key),
                   scope,
                   generation,
                   std::move(identity),
                   std::move(value)});
  try {
    if (!lru_.front().candidate_key.empty()) ++candidates_[lru_.front().candidate_key];
    try {
      index_.emplace(lru_.front().key, lru_.begin());
    } catch (...) {
      auto candidate = candidates_.find(lru_.front().candidate_key);
      if (candidate != candidates_.end() && --candidate->second == 0) candidates_.erase(candidate);
      throw;
    }
  } catch (...) {
    retired.splice(retired.end(), lru_, lru_.begin());
    throw;
  }
  ++stats_.entries;
}
void metadata_cache::erase_scope(uint64_t scope)
{
  entries retired;
  std::lock_guard lock(mutex_);
  for (auto it = lru_.begin(); it != lru_.end();) {
    auto current = it++;
    if (current->scope == scope) retire(current, retired);
  }
}
metadata_cache::counters metadata_cache::statistics() const
{
  std::lock_guard lock(mutex_);
  return stats_;
}
}  // namespace sirius::io::cache
