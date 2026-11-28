// Copyright 2026, Sirius Contributors. SPDX-License-Identifier: Apache-2.0
#pragma once

#include "config/enum_concepts.hpp"

#include <array>
#include <charconv>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <type_traits>
#include <utility>

namespace sirius::config {

/// Outcome of parsing one scalar: success, or the reason the text was rejected. The reason is
/// framed by the caller (`'<dotted.path>': <reason>`), so it does not repeat the key.
class [[nodiscard]] parse_status {
 public:
  parse_status() = default;
  [[nodiscard]] static parse_status failure(std::string reason)
  {
    parse_status status;
    status._error = std::move(reason);
    status._ok    = false;
    return status;
  }

  [[nodiscard]] explicit operator bool() const noexcept { return _ok; }
  [[nodiscard]] const std::string& error() const noexcept { return _error; }

 private:
  bool _ok = true;
  std::string _error;
};

[[nodiscard]] constexpr std::string_view trim(std::string_view text) noexcept
{
  constexpr std::string_view whitespace = " \t\r\n";
  auto const first                      = text.find_first_not_of(whitespace);
  if (first == std::string_view::npos) { return {}; }
  return text.substr(first, text.find_last_not_of(whitespace) - first + 1);
}

namespace detail {

[[nodiscard]] constexpr char ascii_lower(char c) noexcept
{
  return c >= 'A' && c <= 'Z' ? static_cast<char>(c - 'A' + 'a') : c;
}

[[nodiscard]] constexpr bool iequals(std::string_view lhs, std::string_view rhs) noexcept
{
  if (lhs.size() != rhs.size()) { return false; }
  for (std::size_t i = 0; i < lhs.size(); ++i) {
    if (ascii_lower(lhs[i]) != ascii_lower(rhs[i])) { return false; }
  }
  return true;
}

[[nodiscard]] inline std::string quoted(std::string_view text)
{
  return "'" + std::string(text) + "'";
}

}  // namespace detail

/// Customization point: text -> T. A specialization provides
///
///     parse_status operator()(std::string_view text, T& out) const;
///
/// and leaves @p out untouched on failure. The primary template is undefined, so an option of an
/// unsupported type does not compile. Specializations for sizes and durations live in size.hpp and
/// chrono.hpp; enums with string_to_enum / enum_to_string converters are supported here.
template <class T>
struct parser;

/// Customization point: T -> text that parser<T> reads back to an equal value.
template <class T>
struct formatter;

// ---- bool: the spellings yaml-cpp accepts --------------------------------------------------

template <>
struct parser<bool> {
  parse_status operator()(std::string_view text, bool& out) const
  {
    text = trim(text);
    for (auto spelling : {"true", "yes", "y", "on"}) {
      if (detail::iequals(text, spelling)) {
        out = true;
        return {};
      }
    }
    for (auto spelling : {"false", "no", "n", "off"}) {
      if (detail::iequals(text, spelling)) {
        out = false;
        return {};
      }
    }
    return parse_status::failure("invalid boolean value: " + detail::quoted(text));
  }
};

template <>
struct formatter<bool> {
  std::string operator()(bool value) const { return value ? "true" : "false"; }
};

// ---- integers: decimal, optional sign, no wrap-around into unsigned types ------------------
// Character types are integers here too: they read and write numbers, not characters. Hex and
// leading-zero spellings are rejected: yaml-cpp's conversion read "0x10" as hex and "010" as
// octal, and reading "010" as ten instead would change a value silently.

template <class T>
  requires std::integral<T> && (!std::same_as<T, bool>)
struct parser<T> {
  parse_status operator()(std::string_view text, T& out) const
  {
    auto const original = text;
    text                = trim(text);
    bool negative       = false;
    if (!text.empty() && (text.front() == '+' || text.front() == '-')) {
      negative = text.front() == '-';
      text.remove_prefix(1);
    }
    std::uintmax_t magnitude{};
    auto const* const last  = text.data() + text.size();
    auto const [end, error] = std::from_chars(text.data(), last, magnitude);
    if (text.empty() || error == std::errc::invalid_argument || end != last) {
      return parse_status::failure("invalid integer value: " + detail::quoted(trim(original)));
    }
    if (text.size() > 1 && text.front() == '0') {
      return parse_status::failure("invalid integer value: " + detail::quoted(trim(original)) +
                                   " (leading zeros are not allowed)");
    }
    if (error == std::errc::result_out_of_range) { return out_of_range(original); }
    if (negative && magnitude != 0) {
      if constexpr (std::is_unsigned_v<T>) {
        return parse_status::failure("must be a nonnegative integer, got " +
                                     detail::quoted(trim(original)));
      } else {
        // |min| = max + 1 for two's complement.
        auto const limit = static_cast<std::uintmax_t>(std::numeric_limits<T>::max()) + 1;
        if (magnitude > limit) { return out_of_range(original); }
        out = magnitude == limit ? std::numeric_limits<T>::min()
                                 : static_cast<T>(-static_cast<std::intmax_t>(magnitude));
        return {};
      }
    }
    if (magnitude > static_cast<std::uintmax_t>(std::numeric_limits<T>::max())) {
      return out_of_range(original);
    }
    out = static_cast<T>(magnitude);
    return {};
  }

 private:
  static parse_status out_of_range(std::string_view original)
  {
    return parse_status::failure("integer out of range: " + detail::quoted(trim(original)));
  }
};

template <class T>
  requires std::integral<T> && (!std::same_as<T, bool>)
struct formatter<T> {
  std::string operator()(T value) const { return std::to_string(value); }
};

// ---- floating point: decimal or exponent form, plus YAML's .inf / .nan ---------------------

template <std::floating_point T>
struct parser<T> {
  parse_status operator()(std::string_view text, T& out) const
  {
    auto const original = trim(text);
    text                = original;
    bool negative       = false;
    if (!text.empty() && (text.front() == '+' || text.front() == '-')) {
      negative = text.front() == '-';
      text.remove_prefix(1);
    }
    if (detail::iequals(text, ".inf") || detail::iequals(text, "inf") ||
        detail::iequals(text, "infinity")) {
      out = negative ? -std::numeric_limits<T>::infinity() : std::numeric_limits<T>::infinity();
      return {};
    }
    if (detail::iequals(text, ".nan") || detail::iequals(text, "nan")) {
      out = std::numeric_limits<T>::quiet_NaN();
      return {};
    }
    T value{};
    auto const* const last  = text.data() + text.size();
    auto const [end, error] = std::from_chars(text.data(), last, value);
    // Only the spellings above name inf and nan; from_chars' own (e.g. "nan(abc)") are rejected,
    // as is a second sign. Subnormals parse; a value too small or too large for T does not.
    if (text.empty() || error == std::errc::invalid_argument || end != last ||
        text.front() == '+' || text.front() == '-' ||
        (error == std::errc{} && !std::isfinite(value))) {
      return parse_status::failure("invalid number: " + detail::quoted(original));
    }
    if (error == std::errc::result_out_of_range) {
      return parse_status::failure("number out of range: " + detail::quoted(original));
    }
    out = negative ? -value : value;
    return {};
  }
};

template <std::floating_point T>
struct formatter<T> {
  std::string operator()(T value) const
  {
    if (std::isnan(value)) { return ".nan"; }
    if (std::isinf(value)) { return value < 0 ? "-.inf" : ".inf"; }
    std::array<char, 64> buffer{};
    // Shortest representation that reads back to the same value.
    auto const [end, error] = std::to_chars(buffer.data(), buffer.data() + buffer.size(), value);
    if (error != std::errc{}) { throw std::logic_error("formatter: cannot format a number"); }
    std::string text(buffer.data(), end);
    // Keep a float a float when it is read back by YAML-typed tools.
    if (text.find_first_of(".e") == std::string::npos) { text += ".0"; }
    return text;
  }
};

// ---- strings -------------------------------------------------------------------------------

template <>
struct parser<std::string> {
  parse_status operator()(std::string_view text, std::string& out) const
  {
    out.assign(text);
    return {};
  }
};

template <>
struct formatter<std::string> {
  std::string operator()(const std::string& value) const { return value; }
};

// ---- enums ---------------------------------------------------------------------------------

template <string_enum T>
struct parser<T> {
  parse_status operator()(std::string_view text, T& out) const
  {
    text = trim(text);
    T value{};
    if (!string_to_enum(text, value)) {
      return parse_status::failure("invalid enum value " + detail::quoted(text));
    }
    out = value;
    return {};
  }
};

/// @throws std::logic_error when enum_to_string has no spelling for the value.
template <string_enum T>
struct formatter<T> {
  std::string operator()(T value) const
  {
    std::string text;
    if (!enum_to_string(value, text)) {
      throw std::logic_error("formatter: enum value " +
                             std::to_string(static_cast<std::underlying_type_t<T>>(value)) +
                             " has no spelling");
    }
    return text;
  }
};

template <int_enum T>
struct parser<T> {
  parse_status operator()(std::string_view text, T& out) const
  {
    std::underlying_type_t<T> raw{};
    auto status = parser<std::underlying_type_t<T>>{}(text, raw);
    if (status) { out = static_cast<T>(raw); }
    return status;
  }
};

template <int_enum T>
struct formatter<T> {
  std::string operator()(T value) const
  {
    return formatter<std::underlying_type_t<T>>{}(static_cast<std::underlying_type_t<T>>(value));
  }
};

/// True when parser<T> and formatter<T> are both defined.
template <class T>
concept parsable = requires(std::string_view text, T& value) {
  { parser<T>{}(text, value) } -> std::same_as<parse_status>;
  { formatter<T>{}(std::as_const(value)) } -> std::convertible_to<std::string>;
};

}  // namespace sirius::config
