// Copyright 2026, Sirius Contributors. SPDX-License-Identifier: Apache-2.0
#pragma once

#include "config/parser.hpp"

#include <array>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <numeric>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>

namespace sirius::config::quantity {

using uint_t = std::uintmax_t;

/// One accepted unit suffix: its value is num/den base units (bytes, seconds).
struct unit {
  std::string_view suffix;
  uint_t num;
  uint_t den;
};

/// Byte units, matched case-insensitively: K = 1000, Ki = 1024 (the Kubernetes/systemd
/// convention), with an optional trailing B. The first entry of each size is the spelling the
/// writer emits.
inline constexpr unit size_units[] = {
  {"B", 1, 1},
  {"KB", 1000, 1},
  {"K", 1000, 1},
  {"MB", 1000 * 1000, 1},
  {"M", 1000 * 1000, 1},
  {"GB", 1000ULL * 1000 * 1000, 1},
  {"G", 1000ULL * 1000 * 1000, 1},
  {"TB", 1000ULL * 1000 * 1000 * 1000, 1},
  {"T", 1000ULL * 1000 * 1000 * 1000, 1},
  {"KiB", 1ULL << 10, 1},
  {"Ki", 1ULL << 10, 1},
  {"MiB", 1ULL << 20, 1},
  {"Mi", 1ULL << 20, 1},
  {"GiB", 1ULL << 30, 1},
  {"Gi", 1ULL << 30, 1},
  {"TiB", 1ULL << 40, 1},
  {"Ti", 1ULL << 40, 1},
};

/// Time units in seconds, matched case-insensitively. The first entry of each size is the
/// spelling the writer emits.
inline constexpr unit time_units[] = {
  {"ns", 1, 1000 * 1000 * 1000},
  {"nsec", 1, 1000 * 1000 * 1000},
  {"us", 1, 1000 * 1000},
  {"usec", 1, 1000 * 1000},
  {"ms", 1, 1000},
  {"msec", 1, 1000},
  {"s", 1, 1},
  {"sec", 1, 1},
  {"seconds", 1, 1},
  {"min", 60, 1},
  {"m", 60, 1},
  {"minutes", 60, 1},
  {"h", 3600, 1},
  {"hr", 3600, 1},
  {"hours", 3600, 1},
};

/// What the quantity measures, for error messages ("byte value", "time value").
struct kind {
  std::string_view noun;
  std::string_view suffix_noun;
  std::span<const unit> units;
  /// Whether a leading '-' is accepted (durations); sizes reject it.
  bool allow_negative;
};

inline constexpr kind size_kind{"byte value", "byte suffix", size_units, false};
inline constexpr kind time_kind{"time value", "time suffix", time_units, true};

/// The canonical suffix of one tick of Period (e.g. "B", "ms"), or "ticks" when no unit of @p k
/// is exactly one tick.
template <class Period>
std::string tick_name(const kind& k)
{
  for (auto const& u : k.units) {
    if (u.num == static_cast<uint_t>(Period::num) && u.den == static_cast<uint_t>(Period::den)) {
      return std::string(u.suffix);
    }
  }
  return "ticks";
}

/**
 * @brief Parse a decimal number with an optional unit suffix into a whole count of ticks of
 * Period, exactly.
 *
 * The number is `[-]digits[.digits]` (no exponent); the suffix, separated by optional
 * whitespace, is one of @p k.units. A bare number is in the target's own unit (Period). The value
 * is scaled with integer arithmetic only, and anything that is not a whole number of ticks (e.g.
 * 0.1 B, or 1.5 s into std::chrono::seconds) is rejected rather than rounded.
 *
 * @tparam Rep integral representation of the target.
 * @tparam Period std::ratio: seconds or bytes per tick.
 */
template <class Rep, class Period>
parse_status parse(std::string_view text, Rep& out, const kind& k)
{
  static_assert(std::is_integral_v<Rep> && !std::is_same_v<Rep, bool>,
                "quantity::parse needs an integral representation");
  static_assert(Period::num > 0 && Period::den > 0);
  auto const original = trim(text);
  auto invalid        = [&] {
    return parse_status::failure("invalid " + std::string(k.noun) + ": " +
                                 detail::quoted(original));
  };
  auto out_of_range = [&] {
    return parse_status::failure(std::string(k.noun) +
                                 " out of range: " + detail::quoted(original));
  };

  text          = original;
  bool negative = false;
  if (!text.empty() && text.front() == '-') {
    negative = true;
    text.remove_prefix(1);
  }

  // Mantissa digits, with the fraction's trailing zeros dropped so "1.50" scales like "1.5".
  constexpr uint_t max = std::numeric_limits<uint_t>::max();
  uint_t mantissa = 0, scale = 1;
  std::size_t i = 0, pending_zeros = 0;
  bool dot = false, digits = false;
  for (; i < text.size(); ++i) {
    char const c = text[i];
    if (c == '.' && !dot) {
      dot = true;
      continue;
    }
    if (c < '0' || c > '9') { break; }
    digits = true;
    if (dot && c == '0') {
      ++pending_zeros;
      continue;
    }
    for (; pending_zeros > 0; --pending_zeros) {
      if (mantissa > max / 10 || scale > max / 10) { return out_of_range(); }
      mantissa *= 10;
      scale *= 10;
    }
    auto const digit = static_cast<uint_t>(c - '0');
    if (mantissa > (max - digit) / 10) { return out_of_range(); }
    mantissa = mantissa * 10 + digit;
    if (dot) {
      if (scale > max / 10) { return out_of_range(); }
      scale *= 10;
    }
  }
  if (!digits) { return invalid(); }

  auto const suffix = trim(text.substr(i));
  uint_t unit_num = 1, unit_den = 1;
  if (!suffix.empty()) {
    unit const* match = nullptr;
    for (auto const& u : k.units) {
      if (detail::iequals(suffix, u.suffix)) {
        match = &u;
        break;
      }
    }
    if (match == nullptr) {
      return parse_status::failure("unknown " + std::string(k.suffix_noun) + ": " +
                                   detail::quoted(suffix));
    }
    // Express the unit in ticks of Period: (num/den) / (P::num/P::den).
    unit_num = match->num;
    unit_den = match->den;
  } else {
    // A bare number counts ticks of Period directly.
    unit_num = Period::num;
    unit_den = Period::den;
  }

  if (negative && mantissa != 0 && (!k.allow_negative || std::is_unsigned_v<Rep>)) {
    return parse_status::failure(std::string(k.noun) + " must be non-negative");
  }

  // ticks = mantissa / scale * unit_num / unit_den * P::den / P::num, exactly.
  std::array<uint_t, 3> numerators{mantissa, unit_num, static_cast<uint_t>(Period::den)};
  std::array<uint_t, 3> denominators{scale, unit_den, static_cast<uint_t>(Period::num)};
  for (auto& denominator : denominators) {
    for (auto& numerator : numerators) {
      auto const divisor = std::gcd(numerator, denominator);
      numerator /= divisor;
      denominator /= divisor;
    }
    if (denominator != 1) {
      return parse_status::failure(detail::quoted(original) + " is not a whole number of " +
                                   tick_name<Period>(k) + " (it would need rounding)");
    }
  }
  auto const limit = static_cast<uint_t>(std::numeric_limits<Rep>::max());
  uint_t count     = 1;
  for (auto const numerator : numerators) {
    if (numerator != 0 && count > limit / numerator) { return out_of_range(); }
    count *= numerator;
  }
  out = negative ? static_cast<Rep>(-static_cast<std::intmax_t>(count)) : static_cast<Rep>(count);
  return {};
}

/**
 * @brief Format @p count ticks of Period with the largest unit of @p k that represents it
 * exactly, e.g. 1536 MiB, 1 GB, 1500 ms. Zero is written in the tick's own unit ("0 ms").
 */
template <class Rep, class Period>
std::string format(Rep count, const kind& k)
{
  static_assert(std::is_integral_v<Rep>);
  bool const negative = count < 0;
  // Magnitude in ticks; the cast through intmax avoids negating Rep's minimum in Rep.
  uint_t const ticks =
    negative ? static_cast<uint_t>(-static_cast<std::intmax_t>(count)) : static_cast<uint_t>(count);
  std::string const sign = negative ? "-" : "";
  if (ticks == 0) {
    auto const tick = tick_name<Period>(k);
    return tick == "ticks" ? std::string("0") : "0 " + tick;
  }

  // A unit u divides the value when ticks * P::num / P::den * u.den / u.num is whole.
  unit const* best  = nullptr;
  uint_t best_count = 0;
  for (auto const& u : k.units) {
    // Only the first (canonical) spelling of each value is a candidate.
    bool canonical = true;
    for (auto const& other : k.units) {
      if (&other == &u) { break; }
      if (other.num == u.num && other.den == u.den) {
        canonical = false;
        break;
      }
    }
    if (!canonical) { continue; }
    std::array<uint_t, 3> numerators{ticks, static_cast<uint_t>(Period::num), u.den};
    std::array<uint_t, 3> denominators{static_cast<uint_t>(Period::den), u.num, 1};
    bool whole = true;
    for (auto& denominator : denominators) {
      for (auto& numerator : numerators) {
        auto const divisor = std::gcd(numerator, denominator);
        numerator /= divisor;
        denominator /= divisor;
      }
      whole = whole && denominator == 1;
    }
    if (!whole) { continue; }
    uint_t value = 1;
    bool fits    = true;
    for (auto const numerator : numerators) {
      if (numerator != 0 && value > std::numeric_limits<uint_t>::max() / numerator) {
        fits = false;
        break;
      }
      value *= numerator;
    }
    if (!fits) { continue; }
    // Prefer the smallest count, i.e. the largest unit.
    if (best == nullptr || value < best_count) {
      best       = &u;
      best_count = value;
    }
  }
  if (best == nullptr) {
    // No unit of k is a whole multiple of the tick: write the tick count bare.
    return sign + std::to_string(ticks);
  }
  return sign + std::to_string(best_count) + " " + std::string(best->suffix);
}

}  // namespace sirius::config::quantity
