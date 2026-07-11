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
  explicit memory_reservation_manager(std::vector<memory_space_config> const& /*configs*/) {}

  virtual ~memory_reservation_manager() = default;

  virtual memory_space* get_memory_space(Tier tier, int32_t device_id) {
    throw std::runtime_error("cuCascade stub: get_memory_space");
  }
  virtual memory_space const* get_memory_space(Tier tier, int32_t device_id) const {
    throw std::runtime_error("cuCascade stub: get_memory_space");
  }

  virtual std::span<memory_space* const> get_all_memory_spaces() { return {}; }
  virtual std::span<memory_space* const> get_memory_spaces_for_tier(Tier) { return {}; }

  virtual std::unique_ptr<reservation> request_reservation(
    reservation_request_strategy const&, std::size_t /*bytes*/) {
    throw std::runtime_error("cuCascade stub: request_reservation");
  }

  virtual std::size_t get_total_available_memory() const { return 0; }
  virtual std::size_t get_total_reserved_memory() const { return 0; }
  virtual std::size_t get_active_reservation_count() const { return 0; }

  virtual void shutdown() {}

  virtual std::string get_name() const { return "cuCascade stub manager"; }
};

}  // namespace cucascade::memory
