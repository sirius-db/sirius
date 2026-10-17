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

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <optional>

namespace sirius::memory {

/// A retry budget, not a wall-clock deadline. Only the requested backoff is charged;
/// scheduler delay beyond it is excluded. Observed progress starts a fresh budget.
class reservation_wait {
 public:
  using clock = std::chrono::steady_clock;
  static constexpr std::chrono::milliseconds default_timeout{30000};

  bool waiting() const noexcept { return _last_attempt.has_value(); }
  void reset() noexcept { *this = {}; }

  bool retry(clock::time_point now,
             std::size_t available,
             std::uint64_t progress,
             std::chrono::milliseconds timeout,
             int device = 0) noexcept
  {
    if (!_last_attempt || available > _available || progress != _progress || device != _device) {
      _waited  = clock::duration::zero();
      _backoff = std::chrono::milliseconds{5};
    } else {
      _waited += std::min(now - *_last_attempt, clock::duration{_backoff});
      _backoff = std::min(_backoff * 2, std::chrono::milliseconds{50});
    }
    _last_attempt = now;
    _available    = available;
    _progress     = progress;
    _device       = device;
    return _waited < timeout;
  }

  clock::time_point retry_at() const noexcept { return *_last_attempt + _backoff; }

 private:
  std::optional<clock::time_point> _last_attempt;
  clock::duration _waited{};
  std::chrono::milliseconds _backoff{5};
  std::size_t _available  = 0;
  std::uint64_t _progress = 0;
  int _device             = 0;
};

}  // namespace sirius::memory
