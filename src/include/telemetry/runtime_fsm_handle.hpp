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

#include "telemetry-bridge/gen/quent.hpp"

#include <concepts>
#include <cstddef>
#include <functional>
#include <type_traits>
#include <utility>
#include <variant>

namespace sirius::telemetry {

/// Owns an FSM handle whose current state is known only at runtime.
///
/// This occurs in Sirius when query, task, data-batch, and batch-placement
/// lifecycles persist between independently invoked callbacks. Each callback
/// may leave the entity in one of a fixed set of typestates, so the owning
/// object's static type cannot represent its current state. A transition
/// consumes the active handle only when its state is one of the specified
/// source states.
template <typename Entity, typename... States>
class runtime_fsm_handle {
 public:
  using entity_type  = Entity;
  using variant_type = std::variant<quent::FsmHandle<Entity, States>...>;

  template <typename State>
  using handle_type = quent::FsmHandle<Entity, State>;

 private:
  template <typename State>
  static constexpr bool owns_state = (std::same_as<State, States> || ...);

  template <typename Handle>
  static constexpr bool owns_handle = (std::same_as<Handle, handle_type<States>> || ...);

  template <typename State>
  static constexpr std::size_t state_count =
    (std::size_t{0} + ... + static_cast<std::size_t>(std::same_as<State, States>));

  template <typename Transition, typename State>
  static constexpr bool valid_transition =
    requires(Transition&& transition, handle_type<State>&& handle) {
      std::invoke(std::forward<Transition>(transition), std::move(handle));
      requires owns_handle<decltype(std::invoke(std::forward<Transition>(transition),
                                                std::move(handle)))>;
    };

 public:
  static_assert(sizeof...(States) > 0, "a runtime FSM handle needs at least one state");
  static_assert(((state_count<States> == 1) && ...),
                "a runtime FSM handle cannot contain duplicate states");
  static_assert(std::is_nothrow_move_constructible_v<variant_type>);
  static_assert(std::is_nothrow_move_assignable_v<variant_type>);

  template <typename State>
    requires owns_state<State>
  explicit runtime_fsm_handle(handle_type<State>&& handle) noexcept(
    std::is_nothrow_constructible_v<variant_type, handle_type<State>&&>)
    : handle_(std::move(handle))
  {
  }

  runtime_fsm_handle(const runtime_fsm_handle&)                = delete;
  runtime_fsm_handle& operator=(const runtime_fsm_handle&)     = delete;
  runtime_fsm_handle(runtime_fsm_handle&&) noexcept            = default;
  runtime_fsm_handle& operator=(runtime_fsm_handle&&) noexcept = default;

  [[nodiscard]] quent::Uuid uuid() const
  {
    return std::visit([](const auto& handle) { return handle.id().raw(); }, handle_);
  }

  template <typename State>
    requires owns_state<State>
  [[nodiscard]] bool holds() const noexcept
  {
    return std::holds_alternative<handle_type<State>>(handle_);
  }

  template <typename State>
    requires owns_state<State>
  [[nodiscard]] const handle_type<State>* get_if() const noexcept
  {
    return std::get_if<handle_type<State>>(&handle_);
  }

  /// Applies `transition` when the active state is one of `FromStates`.
  ///
  /// Returns whether a transition was applied. The callable is not invoked
  /// when none of the source states is active.
  template <typename... FromStates, typename Transition>
    requires(sizeof...(FromStates) > 0 && (owns_state<FromStates> && ...) &&
             (valid_transition<Transition, FromStates> && ...))
  [[nodiscard]] bool transition(Transition&& transition)
  {
    return (transition_from<FromStates>(std::forward<Transition>(transition)) || ...);
  }

 private:
  template <typename State, typename Transition>
  bool transition_from(Transition&& transition)
  {
    auto* current = std::get_if<handle_type<State>>(&handle_);
    if (current == nullptr) { return false; }
    handle_ = std::invoke(std::forward<Transition>(transition), std::move(*current));
    return true;
  }

  variant_type handle_;
};

}  // namespace sirius::telemetry
