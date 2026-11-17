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
/// the writer emits.
template <class T>
concept string_enum =
  std::is_enum_v<T> && requires(std::string_view text, T& value, std::string& s) {
    { string_to_enum(text, value) } -> std::same_as<bool>;
    { enum_to_string(value, s) } -> std::same_as<bool>;
  };

/// An enum without spelling converters: read and written as its underlying integer.
template <class T>
concept int_enum = std::is_enum_v<T> && !string_enum<T>;

}  // namespace sirius::config
