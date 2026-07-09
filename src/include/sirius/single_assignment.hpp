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

// sirius
#include <sirius/exception.hpp>

// standard library
#include <atomic>
#include <cstdint>
#include <type_traits>
#include <utility>

namespace sirius {

template <class T>
class single_assignment final {
  static_assert(std::is_nothrow_default_constructible_v<T> && std::is_nothrow_move_assignable_v<T>,
                "single_assignment<T> requires T to be nothrow default-constructible and nothrow "
                "move-assignable so the commit path cannot throw.");

  enum class slot_state : std::uint8_t { empty, pending, assigned };

 public:
  class assignment_token final {
   public:
    assignment_token(assignment_token&& other) noexcept
      : _slot(std::exchange(other._slot, nullptr)), _value(std::move(other._value))
    {
    }
    assignment_token& operator=(assignment_token&&)      = delete;
    assignment_token(assignment_token const&)            = delete;
    assignment_token& operator=(assignment_token const&) = delete;
    ~assignment_token() noexcept
    {
      // Uncommitted token: roll back the slot to empty so a later token can be created.
      if (_slot) { _slot->_state.store(slot_state::empty, std::memory_order_release); }
    }

   private:
    friend class single_assignment;
    assignment_token(single_assignment* slot, T value) noexcept
      : _slot(slot), _value(std::move(value))
    {
    }

    single_assignment* _slot;  ///< null once commit_assignment() or move-constructed away
    T _value;
  };

  single_assignment()                                    = default;
  single_assignment(single_assignment const&)            = delete;
  single_assignment& operator=(single_assignment const&) = delete;

  /// @brief Phase 1: claim a slot and carry @p value into the token
  [[nodiscard]] assignment_token prepare_assignment(T value)
  {
    auto expected = slot_state::empty;
    if (!_state.compare_exchange_strong(expected, slot_state::pending, std::memory_order_acq_rel)) {
      throw sirius::internal_exception(
        "[single_assignment] prepare_assignment on a non-empty slot.");
    }
    return assignment_token(this, std::move(value));
  }

  /// @brief Phase 2: publish the already-built value
  void commit_assignment(assignment_token&& token) noexcept
  {
    auto* slot = std::exchange(token._slot, nullptr);
    if (slot != this) { return; }
    _value = std::move(token._value);
    _state.store(slot_state::assigned, std::memory_order_release);
  }

  /// @brief The ordinary checked one-shot assignment (prepare + commit in one step).
  void assign(T value) { commit_assignment(prepare_assignment(std::move(value))); }

  [[nodiscard]] bool is_assigned() const noexcept
  { return _state.load(std::memory_order_acquire) == slot_state::assigned; }

  /// @brief The committed value. Reading before the freeze is an internal error.
  [[nodiscard]] T const& get() const
  {
    if (!is_assigned()) {
      throw sirius::internal_exception(
        "[single_assignment] read before the one-shot assignment was committed");
    }
    return _value;
  }

 private:
  std::atomic<slot_state> _state{slot_state::empty};
  T _value{};
};

}  // namespace sirius
