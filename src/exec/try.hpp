/*
 * Copyright 2025, Sirius Contributors.
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
 * this code has been minimally adopted from folly's SemiFuture implementation, with some
 * modifications to fit our needs.
 */

#pragma once

#include <exception>
#include <functional>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <variant>

namespace sirius::exec {

class try_empty_error : public std::logic_error {
 public:
  try_empty_error() : std::logic_error("try_t is empty") {}
};

template <class value_t>
class try_t {
  static_assert(!std::is_reference_v<value_t>, "try_t<value_t&> is not supported");

 public:
  using value_type = value_t;

  try_t() noexcept = default;

  template <class upstream_t      = value_t,
            std::enable_if_t<std::is_constructible_v<value_t, upstream_t&&> &&
                               !std::is_same_v<std::decay_t<upstream_t>, try_t<value_t>> &&
                               !std::is_same_v<std::decay_t<upstream_t>, std::exception_ptr> &&
                               !std::is_same_v<std::decay_t<upstream_t>, std::in_place_t>,
                             int> = 0>
  /* implicit */ try_t(upstream_t&& v) : _v(std::in_place_index<1>, std::forward<upstream_t>(v))
  {
  }

  template <class... arg_ts>
  explicit try_t(std::in_place_t, arg_ts&&... args)
    : _v(std::in_place_index<1>, std::forward<arg_ts>(args)...)
  {
  }

  /* implicit */ try_t(std::exception_ptr e) noexcept : _v(std::in_place_index<2>, std::move(e)) {}

  try_t(try_t&&) noexcept(std::is_nothrow_move_constructible_v<value_t>)         = default;
  try_t& operator=(try_t&&) noexcept(std::is_nothrow_move_assignable_v<value_t>) = default;
  try_t(try_t const&)                                                            = default;
  try_t& operator=(try_t const&)                                                 = default;

  [[nodiscard]] bool has_value() const noexcept { return _v.index() == 1; }
  [[nodiscard]] bool has_exception() const noexcept { return _v.index() == 2; }
  [[nodiscard]] bool is_empty() const noexcept { return _v.index() == 0; }
  explicit operator bool() const noexcept { return !is_empty(); }

  value_t& value() & { return std::get<1>(_v); }
  value_t const& value() const& { return std::get<1>(_v); }
  value_t&& value() && { return std::move(std::get<1>(_v)); }

  [[nodiscard]] std::exception_ptr const& exception() const& { return std::get<2>(_v); }
  [[nodiscard]] std::exception_ptr exception() && { return std::move(std::get<2>(_v)); }

  void set_exception(std::exception_ptr e) noexcept { _v.template emplace<2>(std::move(e)); }

  template <class... arg_ts>
  value_t& emplace(arg_ts&&... args)
  {
    return _v.template emplace<1>(std::forward<arg_ts>(args)...);
  }

  // Returns the value or rethrows the captured exception.
  value_t get() &&
  {
    if (has_exception()) { std::rethrow_exception(std::get<2>(_v)); }
    if (is_empty()) { throw try_empty_error(); }
    return std::move(std::get<1>(_v));
  }

  value_t const& get_ref() const&
  {
    if (has_exception()) { std::rethrow_exception(std::get<2>(_v)); }
    if (is_empty()) { throw try_empty_error(); }
    return std::get<1>(_v);
  }

 private:
  std::variant<std::monostate, value_t, std::exception_ptr> _v;
};

template <>
class try_t<void> {
 public:
  using value_type = void;

  try_t() noexcept = default;
  /* implicit */ try_t(std::exception_ptr e) noexcept : _e(std::move(e)) {}

  struct value_tag {};
  explicit try_t(value_tag) noexcept : _has_value(true) {}

  static try_t make_value() noexcept { return try_t{value_tag{}}; }

  try_t(try_t&&) noexcept            = default;
  try_t& operator=(try_t&&) noexcept = default;
  try_t(try_t const&)                = default;
  try_t& operator=(try_t const&)     = default;

  [[nodiscard]] bool has_value() const noexcept { return _has_value && !_e; }
  [[nodiscard]] bool has_exception() const noexcept { return static_cast<bool>(_e); }
  [[nodiscard]] bool is_empty() const noexcept { return !_has_value && !_e; }
  explicit operator bool() const noexcept { return !is_empty(); }

  void value() const
  {
    if (has_exception()) { std::rethrow_exception(_e); }
    if (is_empty()) { throw try_empty_error(); }
  }

  [[nodiscard]] std::exception_ptr const& exception() const& { return _e; }
  [[nodiscard]] std::exception_ptr exception() && { return std::move(_e); }

  void set_exception(std::exception_ptr e) noexcept
  {
    _e         = std::move(e);
    _has_value = false;
  }

  void emplace() noexcept
  {
    _has_value = true;
    _e         = nullptr;
  }

  void get() && { value(); }

 private:
  bool _has_value = false;
  std::exception_ptr _e;
};

// Invoke f and capture the result (value or exception) in a try_t<r_t>.
// r_t is the return type of f(args...); deduced if not provided explicitly.
template <class fn_t, class... arg_ts>
auto make_try_with(fn_t&& f, arg_ts&&... args)
{
  using r_t = std::invoke_result_t<fn_t, arg_ts...>;
  if constexpr (std::is_void_v<r_t>) {
    try {
      std::invoke(std::forward<fn_t>(f), std::forward<arg_ts>(args)...);
      return try_t<void>::make_value();
    } catch (...) {
      return try_t<void>{std::current_exception()};
    }
  } else {
    try {
      return try_t<r_t>{std::invoke(std::forward<fn_t>(f), std::forward<arg_ts>(args)...)};
    } catch (...) {
      return try_t<r_t>{std::current_exception()};
    }
  }
}

template <class r_t>
auto make_empty_try()
{
  return try_t<r_t>{};
}

}  // namespace sirius::exec
