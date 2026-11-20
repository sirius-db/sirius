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

/**
 * @file
 * @brief The fixed-width integer types, numeric limits and traits, the range check and checked
 *        integer arithmetic, the @c reg_val register carrier, and the validity-bit read.
 *
 * Fragment 2 of the NVRTC expression evaluator device runtime.
 */

//===----------INTEGER TYPES----------===//
#if defined(__CUDACC_RTC__)
using int8_t    = signed char;
using int16_t   = signed short;
using int32_t   = signed int;
using int64_t   = signed long long;
using uint8_t   = unsigned char;
using uint16_t  = unsigned short;
using uint32_t  = unsigned int;
using uint64_t  = unsigned long long;
using uintptr_t = uint64_t;
// clang-format off
static_assert(sizeof(int16_t) == 2 &&
              sizeof(int32_t) == 4 &&
              sizeof(int64_t) == 8 &&
              sizeof(uintptr_t) == sizeof(void*),
              "the fixed-width integer aliases assume NVRTC's 64-bit data model");
// clang-format on
#else
#include <stddef.h>
#include <stdint.h>
#endif

//===----------LIMITS----------===//
template <typename T>
struct limits;
template <>
struct limits<bool> {
  static constexpr bool lowest() { return false; }
  static constexpr bool max() { return true; }
};
template <>
struct limits<int8_t> {
  static constexpr int8_t lowest() { return -128; }
  static constexpr int8_t max() { return 127; }
};
template <>
struct limits<int16_t> {
  static constexpr int16_t lowest() { return -32768; }
  static constexpr int16_t max() { return 32767; }
};
template <>
struct limits<int32_t> {
  static constexpr int32_t lowest() { return -2147483647 - 1; }
  static constexpr int32_t max() { return 2147483647; }
};
template <>
struct limits<int64_t> {
  static constexpr int64_t lowest() { return -9223372036854775807LL - 1; }
  static constexpr int64_t max() { return 9223372036854775807LL; }
};
template <>
struct limits<uint8_t> {
  static constexpr uint8_t lowest() { return 0; }
  static constexpr uint8_t max() { return 255; }
};
template <>
struct limits<uint16_t> {
  static constexpr uint16_t lowest() { return 0; }
  static constexpr uint16_t max() { return 65535; }
};
template <>
struct limits<uint32_t> {
  static constexpr uint32_t lowest() { return 0; }
  static constexpr uint32_t max() { return 4294967295u; }
};
template <>
struct limits<uint64_t> {
  static constexpr uint64_t lowest() { return 0; }
  static constexpr uint64_t max() { return 18446744073709551615ULL; }
};
template <>
struct limits<__int128_t> {
  static constexpr __int128_t max()
  {
    return static_cast<__int128_t>((static_cast<unsigned __int128>(1) << 127) - 1);
  }
  static constexpr __int128_t lowest() { return -max() - 1; }
};
template <>
struct limits<unsigned __int128> {
  static constexpr unsigned __int128 lowest() { return 0; }
  static constexpr unsigned __int128 max() { return ~static_cast<unsigned __int128>(0); }
};
template <>
struct limits<float> {
  static constexpr float lowest() { return -3.402823466e+38F; }
  static constexpr float max() { return 3.402823466e+38F; }
};
template <>
struct limits<double> {
  static constexpr double lowest() { return -1.7976931348623157e+308; }
  static constexpr double max() { return 1.7976931348623157e+308; }
};

//===----------TYPE TRAITS----------===//
template <typename T>
inline constexpr bool is_floating_point_v = false;
template <>
inline constexpr bool is_floating_point_v<float> = true;
template <>
inline constexpr bool is_floating_point_v<double> = true;
template <typename T>
inline constexpr bool is_signed_v = static_cast<T>(-1) < static_cast<T>(0);
template <bool B, typename T, typename F>
struct conditional {
  using type = T;
};
template <typename T, typename F>
struct conditional<false, T, F> {
  using type = F;
};
template <bool B, typename T, typename F>
using conditional_t = typename conditional<B, T, F>::type;

//===----------RANGE CHECK----------===//
template <typename Dst, typename Src>
__device__ __forceinline__ constexpr bool in_range(Src v)
{
  if constexpr (is_signed_v<Dst> == is_signed_v<Src> && sizeof(Dst) >= sizeof(Src)) {
    return true;
  } else if constexpr (is_signed_v<Dst> == is_signed_v<Src>) {
    using W = conditional_t<is_signed_v<Dst>, __int128_t, unsigned __int128>;
    return static_cast<W>(v) >= static_cast<W>(limits<Dst>::lowest()) &&
           static_cast<W>(v) <= static_cast<W>(limits<Dst>::max());
  } else if constexpr (is_signed_v<Dst>) {
    return static_cast<unsigned __int128>(v) <= static_cast<unsigned __int128>(limits<Dst>::max());
  } else {
    return v >= Src{0} &&
           static_cast<unsigned __int128>(v) <= static_cast<unsigned __int128>(limits<Dst>::max());
  }
}

//===----------CHECKED INTEGER ARITHMETIC----------===//
template <typename T>
struct make_unsigned;
template <>
struct make_unsigned<uint8_t> {
  using type = uint8_t;
};
template <>
struct make_unsigned<uint16_t> {
  using type = uint16_t;
};
template <>
struct make_unsigned<uint32_t> {
  using type = uint32_t;
};
template <>
struct make_unsigned<uint64_t> {
  using type = uint64_t;
};
template <>
struct make_unsigned<unsigned __int128> {
  using type = unsigned __int128;
};
template <>
struct make_unsigned<int8_t> {
  using type = uint8_t;
};
template <>
struct make_unsigned<int16_t> {
  using type = uint16_t;
};
template <>
struct make_unsigned<int32_t> {
  using type = uint32_t;
};
template <>
struct make_unsigned<int64_t> {
  using type = uint64_t;
};
template <>
struct make_unsigned<__int128_t> {
  using type = unsigned __int128;
};
template <typename T>
using make_unsigned_t = typename make_unsigned<T>::type;

template <typename T>
struct ovf_result {
  T value;
  bool overflow;
};

template <typename Dst, typename Src>
__device__ __forceinline__ ovf_result<Dst> checked_narrow(Src v)
{
  return ovf_result<Dst>{static_cast<Dst>(v), !in_range<Dst>(v)};
}

template <typename T>
__device__ __forceinline__ ovf_result<T> checked_add(T a, T b)
{
  using U = make_unsigned_t<T>;
  if constexpr (sizeof(T) >= 8) {
    // Read the overflow off the sign bits (signed) or the carry (unsigned) instead of widening.
    T const r = static_cast<T>(static_cast<U>(a) + static_cast<U>(b));
    if constexpr (is_signed_v<T>) {
      return ovf_result<T>{r, ((a ^ r) & (b ^ r)) < 0};
    } else {
      return ovf_result<T>{r, r < a};
    }
  } else {
    using Wide = conditional_t<is_signed_v<T>, int64_t, uint64_t>;
    return checked_narrow<T>(static_cast<Wide>(a) + static_cast<Wide>(b));
  }
}

template <typename T>
__device__ __forceinline__ ovf_result<T> checked_sub(T a, T b)
{
  using U = make_unsigned_t<T>;
  if constexpr (!is_signed_v<T>) {
    return ovf_result<T>{static_cast<T>(a - b), a < b};
  } else if constexpr (sizeof(T) >= 8) {
    T const r = static_cast<T>(static_cast<U>(a) - static_cast<U>(b));
    return ovf_result<T>{r, ((a ^ b) & (a ^ r)) < 0};
  } else {
    return checked_narrow<T>(static_cast<int64_t>(a) - static_cast<int64_t>(b));
  }
}

/**
 * @brief The 128-bit product of @p a and @p b, with the overflow DuckDB raises for it.
 *
 * DuckDB multiplies the magnitudes of the operands and raises for a product of 2^127 or more
 * (`Hugeint::TryMultiply`). The lowest value (`-2^127`) has that magnitude, so it is a result only
 * where DuckDB does not multiply (multiplication by 1 or 0).
 */
__device__ __forceinline__ ovf_result<__int128_t> checked_mul128(__int128_t a, __int128_t b)
{
  using u128          = unsigned __int128;
  bool const negative = (a < 0) != (b < 0);
  u128 const ua       = a < 0 ? u128{0} - static_cast<u128>(a) : static_cast<u128>(a);
  u128 const ub       = b < 0 ? u128{0} - static_cast<u128>(b) : static_cast<u128>(b);
  uint64_t const a_lo = static_cast<uint64_t>(ua), a_hi = static_cast<uint64_t>(ua >> 64);
  uint64_t const b_lo = static_cast<uint64_t>(ub), b_hi = static_cast<uint64_t>(ub >> 64);
  // The magnitude fits 128 bits only if at most one high limb is set, so one cross product is
  // enough.
  u128 const cross = (a_hi != 0U) ? static_cast<u128>(a_hi) * b_lo : static_cast<u128>(b_hi) * a_lo;
  u128 const lo    = static_cast<u128>(a_lo) * b_lo;
  u128 const hi_sum = (lo >> 64) + static_cast<uint64_t>(cross);
  u128 const mag    = (hi_sum << 64) | static_cast<uint64_t>(lo);
  // Check for -2^127 * 1
  // clang-format off
  bool const lowest_by_one = negative &&
                             (a_hi | b_hi) == (uint64_t{1} << 63) && // 2^63 in high limb
                             (a_lo | b_lo) == uint64_t{1};           // 1 in low limb
  // clang-format on
  u128 const mag_limit = (u128{1} << 127) - (lowest_by_one ? u128{0} : u128{1});
  // clang-format off
  bool const ovf = (a_hi != 0U && b_hi != 0U) ||
                   static_cast<uint64_t>(cross >> 64) != 0U ||
                   static_cast<uint64_t>(hi_sum >> 64) != 0U ||
                   mag > mag_limit;
  // clang-format on
  return ovf_result<__int128_t>{static_cast<__int128_t>(negative ? u128{0} - mag : mag), ovf};
}

template <typename T>
__device__ __forceinline__ ovf_result<T> checked_mul(T a, T b)
{
  if constexpr (sizeof(T) == 16) {
    return checked_mul128(a, b);
  } else if constexpr (sizeof(T) == 8) {
    // The product fits iff the high limb of the 128-bit product is the sign extension of the low
    // limb (signed) or zero (unsigned).
    T const lo = static_cast<T>(static_cast<uint64_t>(a) * static_cast<uint64_t>(b));
    if constexpr (is_signed_v<T>) {
      return ovf_result<T>{lo, __mul64hi(a, b) != (lo >> 63)};
    } else {
      return ovf_result<T>{lo, __umul64hi(a, b) != 0ULL};
    }
  } else {
    using Wide = conditional_t<is_signed_v<T>, int64_t, uint64_t>;
    return checked_narrow<T>(static_cast<Wide>(a) * static_cast<Wide>(b));
  }
}

//===----------REGISTER CARRIER----------===//
/**
 * @brief One Register Value of a generated kernel's row: the payload, its SQL validity, and its
 *        Overflow Poison.
 *
 * ==========OVERFLOW POISON==========
 * Every function of this runtime and every statement that `translation_context` emits keeps one
 * invariant: an operation sets @c overflow only by propagating an operand's Overflow Poison, or
 * from its own computation over operands that are all valid and carry no Overflow Poison, and it
 * never clears the Overflow Poison of an operand it evaluated: only TRY turns that into NULL.
 * Overflow Poison therefore always comes from an Evaluated Operation, and the NVRTC Sinks of the
 * `ProjectSink` and `FilterSink` kernels report every Overflow Poison that reaches them, whatever
 * the value's final validity: e.g., `n + abs(x)` with `n` NULL and `x` the lowest INTEGER is a NULL
 * that still carries the Overflow Poison of `abs(x)`.
 *
 * The payload of a value that is invalid or carries Overflow Poison is unspecified: it is whatever
 * the operation computes from its operands' bits, and it never decides validity or overflow. The
 * arithmetic and the numeric casts do not return early for such an operand, because a branch per
 * operation costs a chain of operations more than computing every row does; `operation_result`
 * applies the invariant to the flags instead. Whatever reads a payload must therefore be defined
 * for every bit pattern (i.e., the following must be avoided): no signed overflow, no division by
 * zero or of the lowest value by -1, no float-to-integer conversion out of range.
 */
template <typename T>
struct reg_val {
  T val;
  bool valid;
  bool overflow;
};

/**
 * @brief The Register Value @p val of validity @p valid that an operation returns: it carries
 *        Overflow Poison when an operand does (@p operand_poison), or when it is valid and the
 *        operation's own check over the payloads reports @p overflows.
 */
template <typename T>
__device__ __forceinline__ reg_val<T> operation_result(T val,
                                                       bool valid,
                                                       bool operand_poison,
                                                       bool overflows)
{
  return reg_val<T>{val, valid, operand_poison || (valid && overflows)};
}

//===----------BIT CHECK----------===//
__device__ __forceinline__ bool is_bit_set(uint32_t const* mask, int32_t bit_idx)
{
  return ((mask[bit_idx >> 5] >> (bit_idx & 31)) & 1U) != 0U;
}
