/*
 * Copyright 2026, Sirius Contributors.
 * Licensed under the Apache License, Version 2.0 (see LICENSE).
 */

//! @file
//! Minimal type stubs for the Rust FFI (cxx) types used by telemetry_context.hpp.
//! When SIRIUS_BUILD_TELEMETRY=OFF, the real cxx-generated headers are absent.
//! These stubs provide the minimum types needed for the header to compile:
//! rust::Box<T>, rust::Fn<...>, and the quent::* observer types as empty structs.
//! All methods are no-ops; the telemetry functionality is simply absent.

#pragma once

#include <cstddef>
#include <functional>
#include <memory>
#include <string>
#include <vector>

// --- rust::Box (cxx FFI) — a unique_ptr-like wrapper ---
namespace rust {
template <typename T>
class Box {
 public:
  Box() = default;
  explicit Box(T* p) : ptr_(p) {}
  Box(Box&& o) noexcept : ptr_(o.ptr_) { o.ptr_ = nullptr; }
  Box& operator=(Box&& o) noexcept { ptr_ = o.ptr_; o.ptr_ = nullptr; return *this; }
  Box(Box const&) = delete;
  Box& operator=(Box const&) = delete;
  T& operator*() const { return *ptr_; }
  T* operator->() const { return ptr_; }
  T* get() const { return ptr_; }
  void reset() { ptr_ = nullptr; }
  ~Box() = default;
 private:
  T* ptr_{nullptr};
};

// rust::Fn is a type-erased function pointer (cxx). Stub as std::function.
template <typename Signature>
using Fn = std::function<Signature>;
}  // namespace rust

// --- uuid::UUID — a 128-bit UUID ---
namespace uuid {
struct UUID {
  uint64_t hi{0};
  uint64_t lo{0};
  bool operator==(UUID const& o) const { return hi == o.hi && lo == o.lo; }
  bool operator!=(UUID const& o) const { return !(*this == o); }
  std::string to_string() const { return "00000000-0000-0000-0000-000000000000"; }
  static UUID now_v7() { return UUID{}; }
  static UUID nil() { return UUID{}; }
};
}  // namespace uuid

// --- quent::* types — opaque observer types ---
namespace quent {

struct Context {
  void* context() { return nullptr; }
};

namespace engine {
struct EngineObserver {
  void init(const uuid::UUID&, const void*) {}
  void declaration(const uuid::UUID&, const void*) {}
};
}  // namespace engine

namespace worker {
struct Init { std::string instance_name; };
struct WorkerObserver {
  void init(const uuid::UUID&, const Init&) {}
  void declaration(const uuid::UUID&, const void*) {}
};
}  // namespace worker

namespace query_group {
struct Declaration { std::string instance_name; };
struct QueryGroupObserver {
  void declaration(const uuid::UUID&, const Declaration&) {}
};
}  // namespace query_group

namespace gpu_device {
struct Declaration { int device_id{0}; std::string name; };
struct GpuDeviceObserver {
  static rust::Box<GpuDeviceObserver> create_observer(const quent::Context&) {
    return rust::Box<GpuDeviceObserver>{};
  }
  void declaration(int, const Declaration&) {}
};
}  // namespace gpu_device

namespace thread_group {
struct Declaration { std::string instance_name; };
struct ThreadGroupObserver {
  static rust::Box<ThreadGroupObserver> create_observer(const quent::Context&) {
    return rust::Box<ThreadGroupObserver>{};
  }
  void declaration(const uuid::UUID&, const Declaration&) {}
};
}  // namespace thread_group

namespace executor_thread {
struct Declaration { std::string instance_name; };
struct ExecutorThreadObserver {
  static rust::Box<ExecutorThreadObserver> create_observer(const quent::Context&) {
    return rust::Box<ExecutorThreadObserver>{};
  }
  void declaration(const uuid::UUID&, const Declaration&) {}
};
}  // namespace executor_thread

namespace task_manager_loop_thread {
struct Declaration { std::string instance_name; };
struct TaskManagerLoopThreadObserver {
  static rust::Box<TaskManagerLoopThreadObserver> create_observer(const quent::Context&) {
    return rust::Box<TaskManagerLoopThreadObserver>{};
  }
  void declaration(const uuid::UUID&, const Declaration&) {}
};
}  // namespace task_manager_loop_thread

namespace task_queue {
struct Declaration { std::string instance_name; };
}  // namespace task_queue

}  // namespace quent
