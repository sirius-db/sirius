// Copyright 2026, Sirius Contributors. SPDX-License-Identifier: Apache-2.0
#pragma once

#include <concepts>
#include <string>
#include <string_view>
#include <type_traits>

namespace sirius::config {

/// An enum with ADL-found spelling converters, declared next to the enum:
///
///     bool string_to_enum(std::string_view text, E& out);   // false: unknown spelling
///     bool enum_to_string(E value, std::string& out);       // false: value has no spelling
///
/// parser<E> and formatter<E> use them, so an option of such an enum needs no custom parser.
/// string_to_enum may accept aliases (e.g. a deprecated spelling); enum_to_string picks the one
/// the writer emits. Declare both in the enum's own header, so every translation unit that
/// parses the enum sees them.
template <class T>
concept string_enum =
  std::is_enum_v<T> && requires(std::string_view text, T& value, std::string& s) {
    { string_to_enum(text, value) } -> std::same_as<bool>;
    { enum_to_string(value, s) } -> std::same_as<bool>;
  };

/// Opt-in for reading and writing an enum without spelling converters as its underlying integer:
///
///     template <> inline constexpr bool sirius::config::enable_int_enum<my_enum> = true;
///
/// Without it such an enum is not a config value at all, so an enum whose converters are missing,
/// misdeclared or not visible fails to compile instead of silently reading integers.
template <class T>
inline constexpr bool enable_int_enum = false;

/// An enum read and written as its underlying integer: opted in, without converters, and with a
/// fixed underlying type (so every integer of that type is a valid value).
template <class T>
concept int_enum = std::is_enum_v<T> && !string_enum<T> && enable_int_enum<T> &&
                   requires { T{std::underlying_type_t<T>{}}; };

}  // namespace sirius::config
