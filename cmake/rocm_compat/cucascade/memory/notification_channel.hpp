/*
 * Copyright 2026, Sirius Contributors.
 * Licensed under the Apache License, Version 2.0 (see LICENSE).
 */
//! @file cuCascade notification_channel — ROCm stub.

#pragma once
#include <functional>

namespace cucascade::memory {

class notification_channel {
 public:
  class event_notifier {
   public:
    void notify() {}
  };
};

/// RAII guard that calls a function on destruction.
template <typename Fn>
class notify_on_exit {
 public:
  explicit notify_on_exit(Fn fn) : fn_(std::move(fn)) {}
  ~notify_on_exit() { fn_(); }
  notify_on_exit(notify_on_exit const&) = delete;
  notify_on_exit& operator=(notify_on_exit const&) = delete;
 private:
  Fn fn_;
};

template <typename Fn>
notify_on_exit<Fn> make_notify_on_exit(Fn fn) {
  return notify_on_exit<Fn>(std::move(fn));
}

}  // namespace cucascade::memory
