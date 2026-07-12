/*
 * Copyright 2026, Sirius Contributors.
 * Licensed under the Apache License, Version 2.0 (see LICENSE).
 */

//! @file
//! cuCascade memory_reservation_manager — ROCm stub.
//! This is a REAL virtual base class — Sirius's sirius_memory_reservation_manager
//! inherits from it.

#pragma once

#include "cucascade/memory/common.hpp"
#include "cucascade/memory/memory_reservation.hpp"
#include "cucascade/memory/memory_space.hpp"
#include <rmm/cuda_stream.hpp>
#include <cstddef>
#include <functional>
#include <map>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace cucascade::memory {

/// Request strategy abstract base.
struct reservation_request_strategy {
  virtual ~reservation_request_strategy() = default;
  virtual bool matches(memory_space const&) const { return true; }
};

struct any_memory_space_in_tier_with_preference : reservation_request_strategy {
  Tier tier;
  int32_t preferred_device{-1};
  explicit any_memory_space_in_tier_with_preference(Tier t, int32_t dev = -1) : tier(t), preferred_device(dev) {}
};

struct any_memory_space_in_tier : reservation_request_strategy {
  Tier tier;
  explicit any_memory_space_in_tier(Tier t) : tier(t) {}
};

struct any_memory_space_in_tiers : reservation_request_strategy {
  std::vector<Tier> tiers;
  explicit any_memory_space_in_tiers(std::vector<Tier> t) : tiers(std::move(t)) {}
};

struct specific_memory_space : reservation_request_strategy {
  memory_space_id id;
  explicit specific_memory_space(memory_space_id i) : id(i) {}
};

struct any_memory_space_to_downgrade : reservation_request_strategy {};
struct any_memory_space_to_upgrade : reservation_request_strategy {};

/// Manages memory spaces and reservations across GPU/Host/Disk tiers.
/// Sirius subclasses this (sirius_memory_reservation_manager) — must be a
/// real polymorphic base.
class memory_reservation_manager {
 public:
  explicit memory_reservation_manager(std::vector<memory_space_config> const& configs) {
    for (auto const& cfg : configs) {
      if (std::holds_alternative<gpu_memory_space_config>(cfg)) {
        auto const& gpu = std::get<gpu_memory_space_config>(cfg);
        auto space = std::make_unique<memory_space>(
          Tier::GPU, gpu.device_id,
          rmm::device_async_resource_ref{},  // will be set by subclass
          std::make_shared<exclusive_stream_pool>(rmm::cuda_device_id{gpu.device_id}, 4));
        spaces_.push_back(std::move(space));
      } else if (std::holds_alternative<host_memory_space_config>(cfg)) {
        auto const& host = std::get<host_memory_space_config>(cfg);
        auto space = std::make_unique<memory_space>(
          Tier::HOST, host.device_id,
          rmm::device_async_resource_ref{},
          std::make_shared<exclusive_stream_pool>(rmm::cuda_device_id{host.device_id}, 2));
        spaces_.push_back(std::move(space));
      }
    }
    // Build tier-indexed views
    for (auto const& s : spaces_) {
      tier_spaces_[s->get_tier()].push_back(s.get());
      all_spaces_.push_back(s.get());
    }
  }

  virtual ~memory_reservation_manager() = default;

  virtual memory_space* get_memory_space(Tier tier, int32_t device_id) {
    for (auto* s : tier_spaces_[tier]) {
      if (s->get_device_id() == device_id) return s;
    }
    return nullptr;
  }
  virtual memory_space const* get_memory_space(Tier tier, int32_t device_id) const {
    for (auto* s : tier_spaces_.at(tier)) {
      if (s->get_device_id() == device_id) return s;
    }
    return nullptr;
  }

  virtual std::span<memory_space* const> get_all_memory_spaces() {
    return std::span<memory_space* const>{all_spaces_};
  }
  virtual std::span<memory_space* const> get_memory_spaces_for_tier(Tier tier) {
    return std::span<memory_space* const>{tier_spaces_[tier]};
  }

  virtual std::unique_ptr<reservation> request_reservation(
    reservation_request_strategy const& strategy, std::size_t bytes) {
    // Find a matching space with available memory
    for (auto* s : all_spaces_) {
      if (strategy.matches(*s) && s->get_available_memory() >= bytes) {
        return s->make_reservation(bytes);
      }
    }
    // Try make_reservation_or_null on any matching space
    for (auto* s : all_spaces_) {
      if (strategy.matches(*s)) {
        auto r = s->make_reservation_or_null(bytes);
        if (r) return r;
      }
    }
    throw std::runtime_error("cuCascade: no memory space can satisfy reservation request");
  }

  virtual std::size_t get_total_available_memory() const {
    std::size_t total = 0;
    for (auto const& s : spaces_) total += s->get_available_memory();
    return total;
  }
  virtual std::size_t get_total_reserved_memory() const {
    std::size_t total = 0;
    for (auto const& s : spaces_) total += s->get_total_reserved_memory();
    return total;
  }
  virtual std::size_t get_active_reservation_count() const { return active_reservations_; }

  virtual void shutdown() { spaces_.clear(); all_spaces_.clear(); tier_spaces_.clear(); }

  virtual std::string get_name() const { return "cuCascade ROCm manager"; }

 protected:
  std::vector<std::unique_ptr<memory_space>> spaces_;
  std::vector<memory_space*> all_spaces_;
  std::map<Tier, std::vector<memory_space*>> tier_spaces_;
  std::size_t active_reservations_{0};
};

}  // namespace cucascade::memory
