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

// Exact decimal division, the arithmetic behind sirius_decimal_divide(), for host and device code.
//
// The DuckDB scalar function (src/expression/decimal_divide_function.cpp) and the GPU kernel
// (src/cuda/decimal_divide.cu) both call decimal_divide_round() below, so the two cannot drift.
// Like src/helper/numeric_carrier_rule.hpp, this header only needs cudf/types.hpp (for the
// CUDF_HOST_DEVICE macro), so a CUDA translation unit can include it.

#include <cudf/types.hpp>

#include <cstdint>

namespace sirius {

/// Name of the SQL function, as DuckDB's catalog and a Substrait plan spell it.
inline constexpr char const* decimal_divide_function_name = "sirius_decimal_divide";

/// Digits of the DECIMAL(38, s) result.
inline constexpr int decimal_divide_max_digits = 38;

/// Outcome of decimal_divide_round(): a quotient, a zero divisor (the SQL result is NULL), or a
/// quotient that needs more than 38 digits.
enum class decimal_divide_status : uint8_t { ok, divide_by_zero, overflow };

namespace detail {

using decimal_divide_u128 = unsigned __int128;

/// 10^0 through 10^19, the powers of ten that fit an unsigned 64-bit limb.
CUDF_HOST_DEVICE constexpr uint64_t decimal_divide_pow10_u64(int exponent) noexcept
{
  uint64_t value = 1;
  for (int i = 0; i < exponent; ++i) {
    value *= 10;
  }
  return value;
}

/// A 256-bit unsigned integer as four little-endian 64-bit limbs.
struct decimal_divide_u256 {
  uint64_t limb[4];

  CUDF_HOST_DEVICE static constexpr decimal_divide_u256 from(decimal_divide_u128 value) noexcept
  {
    return {{static_cast<uint64_t>(value), static_cast<uint64_t>(value >> 64), 0, 0}};
  }

  CUDF_HOST_DEVICE constexpr decimal_divide_u128 low() const noexcept
  {
    return (static_cast<decimal_divide_u128>(limb[1]) << 64) | limb[0];
  }

  CUDF_HOST_DEVICE constexpr decimal_divide_u128 high() const noexcept
  {
    return (static_cast<decimal_divide_u128>(limb[3]) << 64) | limb[2];
  }

  /// Multiplies in place by @p factor; returns false if the product needs more than 256 bits.
  CUDF_HOST_DEVICE constexpr bool multiply(uint64_t factor) noexcept
  {
    uint64_t carry = 0;
    for (auto& word : limb) {
      auto const product = static_cast<decimal_divide_u128>(word) * factor + carry;
      word               = static_cast<uint64_t>(product);
      carry              = static_cast<uint64_t>(product >> 64);
    }
    return carry == 0;
  }

  /// Multiplies in place by 10^@p exponent; returns false if the product needs more than 256 bits.
  CUDF_HOST_DEVICE constexpr bool multiply_pow10(int exponent) noexcept
  {
    constexpr int max_limb_exponent = 19;
    while (exponent > 0) {
      auto const step = exponent < max_limb_exponent ? exponent : max_limb_exponent;
      if (!multiply(decimal_divide_pow10_u64(step))) { return false; }
      exponent -= step;
    }
    return true;
  }
};

/// 10^38 - 1, the largest magnitude of a 38-digit result.
CUDF_HOST_DEVICE constexpr decimal_divide_u128 decimal_divide_max_quotient() noexcept
{
  decimal_divide_u128 value = 1;
  for (int i = 0; i < decimal_divide_max_digits; ++i) {
    value *= 10;
  }
  return value - 1;
}

CUDF_HOST_DEVICE constexpr decimal_divide_u128 decimal_divide_magnitude(__int128_t value) noexcept
{
  // Unsigned negation, so the most negative value has a magnitude too.
  auto const bits = static_cast<decimal_divide_u128>(value);
  return value < 0 ? ~bits + 1 : bits;
}

}  // namespace detail

/**
 * @brief Divides two unscaled decimals exactly and rounds the quotient half away from zero.
 *
 * Computes `round(dividend * 10^exponent / divisor)` without loss, where @p exponent is
 * `result_scale + divisor_scale - dividend_scale` (negative when the result has fewer fractional
 * digits than the operands provide). This is StarRocks' DECIMALV3 `/` (`div_round` after scaling
 * the dividend by `10^adjust`), except that the scaled dividend is held in 256 bits rather than
 * 128, so it never overflows where the quotient itself fits.
 *
 * - The scaled dividend `|dividend| * 10^exponent` is formed in 256 bits. If it needs more, the
 *   quotient is at least 2^255 / 2^127 = 2^128 and cannot fit 38 digits: `overflow`.
 * - A negative @p exponent scales the divisor instead. A scaled divisor of 2^128 or more exceeds
 *   twice any 128-bit dividend, so the quotient rounds to 0.
 * - The quotient is a 128-bit division when the scaled dividend fits 128 bits, and a 256-by-128
 *   shift-subtract long division otherwise.
 *
 * @param dividend Unscaled dividend.
 * @param divisor  Unscaled divisor.
 * @param exponent Power of ten applied to the dividend (or, if negative, to the divisor); -38..76.
 * @param quotient Set to the rounded quotient when the status is `ok`.
 * @return `ok`, `divide_by_zero` when @p divisor is 0, or `overflow` when the rounded quotient
 *         needs more than 38 decimal digits.
 */
CUDF_HOST_DEVICE inline decimal_divide_status decimal_divide_round(__int128_t dividend,
                                                                   __int128_t divisor,
                                                                   int exponent,
                                                                   __int128_t* quotient) noexcept
{
  using detail::decimal_divide_u128;
  using detail::decimal_divide_u256;
  if (divisor == 0) { return decimal_divide_status::divide_by_zero; }
  bool const negative = (dividend < 0) != (divisor < 0);

  auto numerator   = decimal_divide_u256::from(detail::decimal_divide_magnitude(dividend));
  auto denominator = decimal_divide_u256::from(detail::decimal_divide_magnitude(divisor));
  if (exponent >= 0) {
    if (!numerator.multiply_pow10(exponent)) { return decimal_divide_status::overflow; }
  } else {
    // |dividend| <= 2^127, and the scaled divisor is a multiple of 10, so it never equals
    // 2 * |dividend| = 2^128: the quotient is below one half.
    if (!denominator.multiply_pow10(-exponent) || denominator.high() != 0) {
      *quotient = 0;
      return decimal_divide_status::ok;
    }
  }

  auto const d = denominator.low();
  auto const n = numerator.low();
  decimal_divide_u128 q;
  decimal_divide_u128 r;
  if (numerator.high() == 0) {
    if ((n >> 64) == 0 && (d >> 64) == 0) {
      // 64-bit division is much cheaper than 128-bit, on the GPU especially.
      q = static_cast<uint64_t>(n) / static_cast<uint64_t>(d);
    } else {
      q = n / d;
    }
    r = n - q * d;
  } else {
    r = numerator.high();
    // A high half of at least d means a quotient of at least 2^128.
    if (r >= d) { return decimal_divide_status::overflow; }
    // Only a non-negative exponent widens the dividend past 128 bits, so d is |divisor| <= 2^127.
    // r < d keeps r below 2^127, and the shift below never carries out of 128 bits.
    q = 0;
    for (int bit = 127; bit >= 0; --bit) {
      r = (r << 1) | ((n >> bit) & 1);
      q <<= 1;
      if (r >= d) {
        r -= d;
        q |= 1;
      }
    }
  }
  // Half away from zero: round the magnitude up when the remainder is at least half the divisor.
  // (Rounding 10^38 - 1 up to 10^38 can't happen for 128-bit operands: it needs a dividend of at
  // least 2 * 10^38 - 1 units. The check covers it anyway.)
  bool const round_up         = r >= d - r;
  constexpr auto max_quotient = detail::decimal_divide_max_quotient();
  if (q > max_quotient - round_up) { return decimal_divide_status::overflow; }
  q += round_up;

  auto const magnitude = static_cast<__int128_t>(q);
  *quotient            = negative ? -magnitude : magnitude;
  return decimal_divide_status::ok;
}

}  // namespace sirius
