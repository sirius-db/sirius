/*
 * Copyright 2026, Sirius Contributors.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * See the License for the specific language governing permissions and limitations.
 */
#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <optional>
#include <stop_token>
#include <utility>

namespace sirius::scan_manager {
// Lower values have stronger intent. Cleanup must never downgrade an earlier request.
enum class stop_reason : uint8_t { user_cancel, failure_induced, normal_eos };

class preparation_coordinator;

// Read-only attempt stop intent. It neither owns a provider nor decides query success/failure.
class preparation_stop_token {
 public:
  preparation_stop_token() = default;
  explicit preparation_stop_token(std::stop_token token) : token_(std::move(token)) {}
  bool stop_requested() const noexcept { return token_.stop_requested(); }
  // Standard callbacks fire once; a later reason upgrade is visible through reason()
  // but does not notify them again. No provider interrupt is implied by this token.
  std::stop_token token() const noexcept { return token_; }
  std::optional<stop_reason> reason() const noexcept
  {
    return reason_ ? reason_->get() : std::nullopt;
  }

 private:
  friend class preparation_coordinator;
  struct reason_state {
    std::atomic<int> value{-1};
    void request(stop_reason reason) noexcept
    {
      auto desired = static_cast<int>(reason);
      auto current = value.load();
      while ((current < 0 || desired < current) && !value.compare_exchange_weak(current, desired)) {
      }
    }
    std::optional<stop_reason> get() const noexcept
    {
      auto current = value.load();
      return current < 0 ? std::nullopt : std::optional{static_cast<stop_reason>(current)};
    }
  };
  preparation_stop_token(std::stop_token token, std::shared_ptr<reason_state const> reason)
    : token_(std::move(token)), reason_(std::move(reason))
  {
  }
  std::stop_token token_;
  std::shared_ptr<reason_state const> reason_;
};
}  // namespace sirius::scan_manager
