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
  template <typename Fn>
  notify_on_exit(Fn&&) -> notify_on_exit<std::decay_t<Fn>>;
  template <typename Fn>
  class notify_on_exit {
   public:
    explicit notify_on_exit(Fn fn) : fn_(std::move(fn)) {}
    ~notify_on_exit() { fn_(); }
   private:
    Fn fn_;
  };
};

}  // namespace cucascade::memory
