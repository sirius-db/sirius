/*
 * Copyright 2026, Sirius Contributors.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "io/object_identity.hpp"

#include <chrono>
#include <condition_variable>
#include <functional>
#include <list>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_map>

namespace sirius::io {
class io_object_metadata {
 public:
  virtual ~io_object_metadata() { live_charge_.fetch_sub(charge_, std::memory_order_relaxed); }
  size_t account_retention() const
  {
    std::call_once(accounted_, [this] {
      charge_ = retained_bytes();
      live_charge_.fetch_add(charge_, std::memory_order_relaxed);
    });
    return charge_;
  }
  static size_t live_retained_bytes() noexcept
  {
    return live_charge_.load(std::memory_order_relaxed);
  }
  /// Zero means unaccounted: usable by the scan but not admitted to the cache.
  [[nodiscard]] virtual size_t retained_bytes() const noexcept { return 0; }

 private:
  mutable std::once_flag accounted_;
  mutable size_t charge_ = 0;
  inline static std::atomic<size_t> live_charge_{0};
};

namespace cache {
/// Process-wide retention, independent of query lifetimes. Stores supply unique
/// namespaces, so sharing capacity never widens access authority.
class metadata_cache {
 public:
  using clock = std::chrono::steady_clock;
  struct options {
    size_t capacity                        = size_t{1} << 30;
    clock::duration idle                   = std::chrono::minutes(30);
    clock::duration sweep                  = std::chrono::seconds(60);
    size_t batch                           = 256;
    std::function<clock::time_point()> now = [] { return clock::now(); };
    bool background                        = true;
  };
  struct counters {
    size_t retained_bytes = 0, entries = 0;
    uint64_t hits = 0, misses = 0, capacity_evictions = 0, idle_evictions = 0;
    uint64_t replacements = 0, bypasses = 0;
  };
  explicit metadata_cache(options config);
  ~metadata_cache();
  metadata_cache(metadata_cache const&)            = delete;
  metadata_cache& operator=(metadata_cache const&) = delete;
  static std::shared_ptr<metadata_cache> global();

  std::shared_ptr<io_object_metadata> get(uint64_t scope,
                                          std::string const& path,
                                          std::string const& profile,
                                          object_identity const& identity);
  /// Candidate hint only: does not validate identity or refresh idle time.
  bool contains(uint64_t scope, std::string const& path);
  void put(uint64_t scope,
           std::string const& path,
           std::string const& profile,
           object_identity identity,
           uint64_t generation,
           std::shared_ptr<io_object_metadata> value);
  void erase_scope(uint64_t scope);
  size_t sweep();
  counters statistics() const;

 private:
  struct entry {
    std::string key, path;
    uint64_t scope, generation;
    object_identity identity;
    std::shared_ptr<io_object_metadata> value;
    size_t charge;
    clock::time_point accessed;
  };
  using entries  = std::list<entry>;
  using iterator = entries::iterator;
  static std::string key(uint64_t scope, std::string const& path, std::string const& profile);
  void retire(iterator it, entries& retired);
  bool expired(entry const& value, clock::time_point now) const;
  options config_;
  mutable std::mutex mutex_;
  std::condition_variable wake_;
  entries lru_;
  std::unordered_map<std::string, iterator> index_;
  std::unordered_map<std::string, size_t> candidates_;
  counters stats_;
  bool stopping_ = false;
  std::thread worker_;
};
}  // namespace cache
}  // namespace sirius::io
