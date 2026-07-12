/*
 * Copyright 2026, Sirius Contributors.
 * Licensed under the Apache License, Version 2.0 (see LICENSE).
 */

//! @file
//! cuCascade memory reservation — ROCm stub.

#pragma once

#include "cucascade/memory/common.hpp"
#include <cstddef>
#include <memory>
#include <stdexcept>

namespace cucascade::memory {

class memory_space;

/// Abstract base for a reserved memory arena.
class reserved_arena {
 public:
  virtual ~reserved_arena() = default;
  virtual std::size_t size() const = 0;
  virtual void grow_by(std::size_t) {}
  virtual void shrink_to_fit() {}
};

/// Reservation limit policies.
class reservation_limit_policy {
 public:
  virtual ~reservation_limit_policy() = default;
  virtual bool can_reserve(std::size_t /*requested*/, std::size_t /*current*/) const { return true; }
};

class ignore_reservation_limit_policy : public reservation_limit_policy {};
class fail_reservation_limit_policy : public reservation_limit_policy {
  bool can_reserve(std::size_t, std::size_t) const override { return false; }
};
class increase_reservation_limit_policy : public reservation_limit_policy {};

/// A memory reservation — represents reserved capacity on a memory space.
/// Stub: all methods throw or return defaults.
class reservation {
 public:
  reservation(memory_space& space, std::unique_ptr<reserved_arena> arena)
    : space_(&space), arena_(std::move(arena)) {}

  std::size_t size() const {
    return arena_ ? arena_->size() : 0;
  }
  Tier tier() const { return space_ ? space_->get_tier() : Tier::GPU; }
  int32_t device_id() const { return space_ ? space_->get_device_id() : 0; }
  memory_space const& get_memory_space() const { return *space_; }

  void grow_by(std::size_t n) { if (arena_) arena_->grow_by(n); }
  void shrink_to_fit() { if (arena_) arena_->shrink_to_fit(); }

  template <typename T>
  T* get_memory_resource_as() {
    throw std::runtime_error("cuCascade stub: reservation::get_memory_resource_as<T>()");
  }

  template <Tier TIER>
  typename tier_memory_resource_trait<TIER>::type* get_memory_resource_of() {
    throw std::runtime_error("cuCascade stub: reservation::get_memory_resource_of<TIER>()");
  }

  static std::unique_ptr<reservation> create(memory_space& space,
                                             std::unique_ptr<reserved_arena> arena) {
    return std::make_unique<reservation>(space, std::move(arena));
  }

 private:
  memory_space* space_;
  std::unique_ptr<reserved_arena> arena_;
};

}  // namespace cucascade::memory
