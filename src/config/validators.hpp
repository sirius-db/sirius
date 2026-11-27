// Copyright 2026, Sirius Contributors. SPDX-License-Identifier: Apache-2.0
#pragma once

#include "config/option.hpp"
#include "config/size.hpp"

#include <cmath>
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

/// Accepts finite values above 0 (NaN and infinity fail).
struct positive_finite {
  [[nodiscard]] bool operator()(double value) const noexcept
  {
    return std::isfinite(value) && value > 0.0;
  }
  [[nodiscard]] static constexpr const char* description() noexcept
  {
    return "must be finite and greater than 0";
  }
};

/// Accepts finite values at or above 0 (NaN and infinity fail).
struct non_negative_finite {
  [[nodiscard]] bool operator()(double value) const noexcept
  {
    return std::isfinite(value) && value >= 0.0;
  }
  [[nodiscard]] static constexpr const char* description() noexcept
  {
    return "must be finite and not negative";
  }
};

/// Accepts values above T{} (zero for numbers, sizes and durations).
template <class T>
struct positive {
  [[nodiscard]] bool operator()(const T& value) const { return T{} < value; }
  [[nodiscard]] static constexpr const char* description() noexcept
  {
    return "must be greater than zero";
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

/// Accepts values at or below a limit.
template <class T>
struct at_most {
  T limit;
  [[nodiscard]] bool operator()(const T& value) const { return !(limit < value); }
  [[nodiscard]] std::string description() const
  {
    return "must be at most " + formatter<T>{}(limit);
  }
};

/// Accepts values at or above a limit.
template <class T>
struct at_least {
  T limit;
  [[nodiscard]] bool operator()(const T& value) const { return !(value < limit); }
  [[nodiscard]] std::string description() const
  {
    return "must be at least " + formatter<T>{}(limit);
  }
};

/// Accepts a non-empty string.
struct non_empty {
  [[nodiscard]] bool operator()(const std::string& value) const noexcept { return !value.empty(); }
  [[nodiscard]] static constexpr const char* description() noexcept { return "must not be empty"; }
};

/// The downgrade pair rule shared by every memory tier: 0 < stop < trigger.
/// @throws config_error naming @p path's downgrade_stop_fraction.
inline void check_downgrade_fractions(std::string_view path, double trigger, double stop)
{
  auto const stop_key = "'" + join_path(path, "downgrade_stop_fraction") + "': ";
  if (stop <= 0.0) { throw config_error(stop_key + "must be greater than zero"); }
  if (stop >= trigger) {
    throw config_error(stop_key + "must be less than downgrade_trigger_fraction");
  }
}

}  // namespace sirius::config
