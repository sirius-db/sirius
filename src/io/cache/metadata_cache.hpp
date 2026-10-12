/*
 * Copyright 2026, Sirius Contributors.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "io/object_identity.hpp"

#include <list>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>

namespace sirius::io {
class io_object_metadata {
 public:
  virtual ~io_object_metadata() = default;
};

namespace cache {
class metadata_cache {
 public:
  struct options {};
  struct counters {
    size_t entries = 0;
    uint64_t hits = 0, misses = 0;
    uint64_t replacements = 0, bypasses = 0;
  };
  metadata_cache() : metadata_cache(options{}) {}
  explicit metadata_cache(options config);
  ~metadata_cache();
  metadata_cache(metadata_cache const&)            = delete;
  metadata_cache& operator=(metadata_cache const&) = delete;

  std::shared_ptr<io_object_metadata> get(uint64_t scope,
                                          std::string const& path,
                                          std::string const& profile,
                                          object_identity const& identity);
  std::unique_lock<std::mutex> lock_miss(uint64_t scope,
                                         std::string const& path,
                                         std::string const& profile);
  bool contains(uint64_t scope, std::string const& path);
  void put(uint64_t scope,
           std::string const& path,
           std::string const& profile,
           object_identity identity,
           uint64_t generation,
           std::shared_ptr<io_object_metadata> value);
  void erase_scope(uint64_t scope);
  counters statistics() const;

 private:
  struct entry {
    std::string key, candidate_key;
    uint64_t scope, generation;
    object_identity identity;
    std::shared_ptr<io_object_metadata> value;
  };
  using entries  = std::list<entry>;
  using iterator = entries::iterator;
  static std::string key(uint64_t scope, std::string const& path, std::string const& profile);
  void retire(iterator it, entries& retired);
  mutable std::mutex mutex_;
  entries lru_;
  std::unordered_map<std::string, iterator> index_;
  std::unordered_map<std::string, size_t> candidates_;
  std::unordered_map<std::string, std::shared_ptr<std::mutex>> miss_locks_;
  counters stats_;
};
}  // namespace cache
}  // namespace sirius::io
