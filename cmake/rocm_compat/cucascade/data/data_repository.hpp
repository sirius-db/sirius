/*
 * Copyright 2026, Sirius Contributors.
 * Licensed under the Apache License, Version 2.0 (see LICENSE).
 */
//! @file cuCascade data_repository — ROCm stub (fully inline, nearly verbatim).

#pragma once
#include "cucascade/data/data_batch.hpp"
#include <cstdint>
#include <memory>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace cucascade {

/// A data repository holds data batches organized by partition.
class data_repository {
 public:
  virtual ~data_repository() = default;

  virtual void add_data_batch(std::shared_ptr<data_batch> batch, std::size_t partition_idx = 0) {
    if (partition_idx >= partitions_.size()) partitions_.resize(partition_idx + 1);
    partitions_[partition_idx].push_back(std::move(batch));
  }

  virtual std::shared_ptr<data_batch> pop_next_data_batch(std::size_t partition_idx = 0) {
    if (partition_idx >= partitions_.size() || partitions_[partition_idx].empty()) return nullptr;
    auto batch = std::move(partitions_[partition_idx].front());
    partitions_[partition_idx].erase(partitions_[partition_idx].begin());
    return batch;
  }

  virtual std::shared_ptr<data_batch> pop_data_batch_by_id(uint64_t id, std::size_t partition_idx = 0) {
    if (partition_idx >= partitions_.size()) return nullptr;
    for (auto it = partitions_[partition_idx].begin(); it != partitions_[partition_idx].end(); ++it) {
      if (*it && (*it)->get_batch_id() == id) {
        auto batch = std::move(*it);
        partitions_[partition_idx].erase(it);
        return batch;
      }
    }
    return nullptr;
  }

  virtual std::shared_ptr<data_batch> get_data_batch_by_id(uint64_t id, std::size_t partition_idx = 0) const {
    if (partition_idx >= partitions_.size()) return nullptr;
    for (auto const& batch : partitions_[partition_idx]) {
      if (batch && batch->get_batch_id() == id) return batch;
    }
    return nullptr;
  }

  virtual std::vector<uint64_t> get_batch_ids(std::size_t partition_idx = 0) const {
    std::vector<uint64_t> ids;
    if (partition_idx < partitions_.size()) {
      for (auto const& batch : partitions_[partition_idx]) {
        if (batch) ids.push_back(batch->get_batch_id());
      }
    }
    return ids;
  }

  virtual std::size_t size(std::size_t partition_idx = 0) const {
    return partition_idx < partitions_.size() ? partitions_[partition_idx].size() : 0;
  }
  virtual bool empty(std::size_t partition_idx = 0) const { return size(partition_idx) == 0; }
  virtual std::size_t total_size() const {
    std::size_t total = 0;
    for (auto const& p : partitions_) total += p.size();
    return total;
  }
  virtual bool all_empty() const {
    for (auto const& p : partitions_) if (!p.empty()) return false;
    return true;
  }
  virtual std::size_t num_partitions() const { return partitions_.size(); }

 protected:
  std::vector<std::vector<std::shared_ptr<data_batch>>> partitions_;
};

using shared_data_repository = data_repository;

}  // namespace cucascade
