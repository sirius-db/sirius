// Copyright 2026, Sirius Contributors. SPDX-License-Identifier: Apache-2.0
#pragma once

#include "config/option.hpp"
#include "config/size.hpp"

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>

namespace sirius::config {

/// Accepts [0, 1] (NaN fails).
struct fraction {
  [[nodiscard]] bool operator()(double value) const noexcept
  {
    return value >= 0.0 && value <= 1.0;
  }
  [[nodiscard]] static constexpr const char* description() noexcept
  {
    return "must be between 0.0 and 1.0";
  }
};

/// Accepts (0, 1] (NaN fails).
struct positive_fraction {
  [[nodiscard]] bool operator()(double value) const noexcept { return value > 0.0 && value <= 1.0; }
  [[nodiscard]] static constexpr const char* description() noexcept
  {
    return "must be greater than 0.0 and at most 1.0";
  }
};

/// Accepts values above T{} (zero for numbers, sizes and durations).
template <class T>
struct positive {
  [[nodiscard]] bool operator()(const T& value) const { return T{} < value; }
  [[nodiscard]] static constexpr const char* description() noexcept
  {
    return "must be greater than 0";
  }
};

/// Accepts values at or above T{}.
template <class T>
struct non_negative {
  [[nodiscard]] bool operator()(const T& value) const { return T{} <= value; }  // NaN fails
  [[nodiscard]] static constexpr const char* description() noexcept
  {
    return "must not be negative";
  }
};

/// A runtime default byte count as a config value. @throws std::logic_error above INT64_MAX.
[[nodiscard]] inline bytes to_bytes(std::uint64_t count)
{
  if (count > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max())) {
    throw std::logic_error("config: byte default out of range");
  }
  return bytes(static_cast<std::int64_t>(count));
}

/// The downgrade pair rule shared by every memory tier: 0 < stop < trigger.
/// @throws config_error naming @p path.
inline void check_downgrade_fractions(std::string_view path, double trigger, double stop)
{
  if (stop <= 0.0) {
    throw config_error(detail::framed(path, "downgrade_stop_fraction must be greater than zero"));
  }
  if (stop >= trigger) {
    throw config_error(
      detail::framed(path, "downgrade_stop_fraction must be less than downgrade_trigger_fraction"));
  }
}

}  // namespace sirius::config
