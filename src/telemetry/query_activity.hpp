/*
 * Copyright 2026, Sirius Contributors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include "telemetry/telemetry_context.hpp"

#include <exception>
#include <mutex>

namespace sirius::telemetry {

/// A scoped, explicitly parented interval. Its state outlives either the scope or
/// the query owner, so query-end cleanup cannot leave a dangling observer borrow.
class query_activity {
 public:
  using id_type                                    = quent::query_activity::QueryActivityId;
  query_activity()                                 = default;
  query_activity(const query_activity&)            = delete;
  query_activity& operator=(const query_activity&) = delete;
  query_activity(query_activity&&) noexcept        = default;
  query_activity& operator=(query_activity&& other) noexcept
  {
    if (this != &other) {
      finish();
      state_      = std::move(other.state_);
      exceptions_ = other.exceptions_;
    }
    return *this;
  }
  ~query_activity() noexcept { finish(); }
  [[nodiscard]] std::optional<id_type> id() const noexcept
  {
    return state_ ? std::optional(state_->handle.id()) : std::nullopt;
  }
  void finish(const char* outcome, const char* decision = nullptr) noexcept
  {
    if (state_) {
      state_->finish(outcome, decision);
      state_.reset();
    }
  }
  void finish() noexcept
  {
    finish(std::uncaught_exceptions() > exceptions_ ? "failed" : "success");
  }

 private:
  friend class query_observer;
  struct state {
    std::shared_ptr<const telemetry_context> context;
    quent::DynamicFsmHandle<quent::QueryActivity> handle;
    std::mutex mutex;
    bool finished = false;
    state(std::shared_ptr<const telemetry_context> context,
          quent::DynamicFsmHandle<quent::QueryActivity>&& handle)
      : context(std::move(context)), handle(std::move(handle))
    {
    }
    ~state() noexcept { finish("abandoned"); }
    void finish(const char* outcome, const char* decision = nullptr) noexcept
    {
      std::lock_guard lock(mutex);
      if (finished) return;
      finished = true;
      try {
        handle.finished(
          {.outcome  = outcome ? std::optional<std::string>(outcome) : std::nullopt,
           .decision = decision ? std::optional<std::string>(decision) : std::nullopt});
      } catch (...) {
        sirius::log::log_noexcept([] { SIRIUS_LOG_ERROR("query activity emission failed"); });
      }
    }
  };
  explicit query_activity(std::shared_ptr<state> state) : state_(std::move(state)) {}
  std::shared_ptr<state> state_;
  int exceptions_ = std::uncaught_exceptions();
};

}  // namespace sirius::telemetry
