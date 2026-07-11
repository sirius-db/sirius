/*
 * Copyright 2026, Sirius Contributors.
 * Licensed under the Apache License, Version 2.0 (see LICENSE).
 */
//! @file cuCascade data_repository_manager — ROCm stub (fully inline).

#pragma once
#include "cucascade/data/data_repository.hpp"
#include <cstdint>
#include <functional>
#include <memory>
#include <string_view>
#include <utility>
#include <vector>

namespace cucascade {

class data_repository_manager {
 public:
  struct operator_port_key {
    std::size_t operator_id;
    std::string_view port_name;
    bool operator==(operator_port_key const& o) const {
      return operator_id == o.operator_id && port_name == o.port_name;
    }
  };

  struct leaked_repository_info {
    std::size_t operator_id;
    std::string_view port_name;
    std::size_t batch_count;
  };

  void add_new_repository(std::size_t operator_id, std::string_view port_name,
                          std::unique_ptr<data_repository> repo) {
    repos_[{operator_id, port_name}] = std::move(repo);
  }

  void add_data_batch(std::shared_ptr<data_batch> batch,
                      std::vector<std::pair<std::size_t, std::string_view>> const& targets) {
    for (auto const& [op_id, port] : targets) {
      auto it = repos_.find({op_id, port});
      if (it != repos_.end()) {
        it->second->add_data_batch(batch);
      }
    }
  }

  data_repository* get_repository(std::size_t operator_id, std::string_view port_name) {
    auto it = repos_.find({operator_id, port_name});
    return it != repos_.end() ? it->second.get() : nullptr;
  }

  uint64_t get_next_data_batch_id() { return next_batch_id_++; }

  std::vector<leaked_repository_info> clear_all_repositories() {
    std::vector<leaked_repository_info> leaked;
    for (auto& [key, repo] : repos_) {
      if (repo && !repo->all_empty()) {
        leaked.push_back({key.operator_id, key.port_name, repo->total_size()});
      }
    }
    repos_.clear();
    return leaked;
  }

  void for_each_repository(std::function<void(data_repository*)> const& fn) {
    for (auto& [_, repo] : repos_) {
      if (repo) fn(repo.get());
    }
  }

  std::vector<data_repository*> get_repositories() {
    std::vector<data_repository*> result;
    for (auto& [_, repo] : repos_) {
      if (repo) result.push_back(repo.get());
    }
    return result;
  }

 private:
  struct op_port_hash {
    std::size_t operator()(operator_port_key const& k) const {
      return std::hash<std::size_t>{}(k.operator_id) ^
             std::hash<std::string_view>{}(k.port_name);
    }
  };

  std::unordered_map<operator_port_key, std::unique_ptr<data_repository>, op_port_hash> repos_;
  uint64_t next_batch_id_{0};
};

using shared_data_repository_manager = data_repository_manager;

}  // namespace cucascade

namespace std {
template <>
struct hash<cucascade::data_repository_manager::operator_port_key> {
  std::size_t operator()(cucascade::data_repository_manager::operator_port_key const& k) const {
    return std::hash<std::size_t>{}(k.operator_id) ^ std::hash<std::string_view>{}(k.port_name);
  }
};
}  // namespace std
