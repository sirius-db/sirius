// Copyright 2026, Sirius Contributors. SPDX-License-Identifier: Apache-2.0
#pragma once

#include "config/parser.hpp"
#include "config/quantity.hpp"

#include <chrono>
#include <concepts>
#include <string>
#include <string_view>

namespace sirius::config {

/// "250ms", "1.5 s", "2 min", "500" (bare: ticks of the target unit). Units are matched
/// case-insensitively and include today's long spellings (msec, sec, seconds, minutes, hr, ...).
/// Exact: 1.5 s into std::chrono::seconds is rejected rather than truncated.
template <class Rep, class Period>
  requires std::integral<Rep> && (!std::same_as<Rep, bool>)
struct parser<std::chrono::duration<Rep, Period>> {
  parse_status operator()(std::string_view text, std::chrono::duration<Rep, Period>& out) const
  {
    Rep count{};
    auto status = quantity::parse<Rep, Period>(text, count, quantity::time_kind);
    if (status) { out = std::chrono::duration<Rep, Period>(count); }
    return status;
  }
};

/// The largest unit that represents the value exactly: 1500ms -> "1500 ms", 120s -> "2 min".
template <class Rep, class Period>
  requires std::integral<Rep> && (!std::same_as<Rep, bool>)
struct formatter<std::chrono::duration<Rep, Period>> {
  std::string operator()(const std::chrono::duration<Rep, Period>& value) const
  {
    return quantity::format<Rep, Period>(value.count(), quantity::time_kind);
  }
};

}  // namespace sirius::config
