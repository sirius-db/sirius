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
 */

#pragma once

/**
 * @file
 * @brief Numeric limits and traits, the range check and checked 128-bit arithmetic, the @c reg_val
 *        register carrier, and the validity-bit read.
 *
 * Fragment 2 of the NVRTC expression evaluator device runtime.
 */

#include "codegen/stdint_shim.hpp"

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

//===----------OVERFLOW----------===//
template <typename T>
struct ovf_result {
  T value;
  bool overflow;
};
template <typename Dst>
__device__ __forceinline__ ovf_result<Dst> overflow_cast(__int128_t v)
{
  bool const fits = in_range<Dst>(v);
  return ovf_result<Dst>{fits ? static_cast<Dst>(v) : Dst{}, !fits};
}
__device__ __forceinline__ ovf_result<__int128_t> add_overflow128(__int128_t a, __int128_t b)
{
  __int128_t const r =
    static_cast<__int128_t>(static_cast<unsigned __int128>(a) + static_cast<unsigned __int128>(b));
  return ovf_result<__int128_t>{r, ((a ^ r) & (b ^ r)) < 0};
}
__device__ __forceinline__ ovf_result<__int128_t> sub_overflow128(__int128_t a, __int128_t b)
{
  auto const r =
    static_cast<__int128_t>(static_cast<unsigned __int128>(a) - static_cast<unsigned __int128>(b));
  return ovf_result<__int128_t>{r, ((a ^ b) & (a ^ r)) < 0};
}
__device__ __forceinline__ ovf_result<__int128_t> mul_overflow128(__int128_t a, __int128_t b)
{
  using u128          = unsigned __int128;
  bool const negative = (a < 0) != (b < 0);
  u128 const ua       = a < 0 ? u128{0} - static_cast<u128>(a) : static_cast<u128>(a);
  u128 const ub       = b < 0 ? u128{0} - static_cast<u128>(b) : static_cast<u128>(b);
  uint64_t const a_lo = static_cast<uint64_t>(ua), a_hi = static_cast<uint64_t>(ua >> 64);
  uint64_t const b_lo = static_cast<uint64_t>(ub), b_hi = static_cast<uint64_t>(ub >> 64);
  if (a_hi != 0u && b_hi != 0u) { return ovf_result<__int128_t>{0, true}; }
  u128 const cross = (a_hi != 0u) ? static_cast<u128>(a_hi) * b_lo : static_cast<u128>(b_hi) * a_lo;
  if (static_cast<uint64_t>(cross >> 64) != 0u) { return ovf_result<__int128_t>{0, true}; }
  u128 const lo     = static_cast<u128>(a_lo) * b_lo;
  u128 const hi_sum = (lo >> 64) + cross;
  if (static_cast<uint64_t>(hi_sum >> 64) != 0u) { return ovf_result<__int128_t>{0, true}; }
  u128 const mag   = (hi_sum << 64) | static_cast<uint64_t>(lo);
  u128 const limit = (u128{1} << 127) - (negative ? u128{0} : u128{1});
  if (mag > limit) { return ovf_result<__int128_t>{0, true}; }
  return ovf_result<__int128_t>{static_cast<__int128_t>(negative ? u128{0} - mag : mag), false};
}
template <typename T>
__device__ __forceinline__ ovf_result<T> mul_overflow(T a, T b)
{
  if constexpr (sizeof(T) == 16) {
    return mul_overflow128(a, b);
  } else if constexpr (sizeof(T) == 8) {
    T const lo     = static_cast<T>(static_cast<uint64_t>(a) * static_cast<uint64_t>(b));
    bool const ovf = __mul64hi(a, b) != (lo >> 63);
    return ovf_result<T>{ovf ? T{} : lo, ovf};
  } else {
    int64_t const r = static_cast<int64_t>(a) * static_cast<int64_t>(b);
    bool const fits = in_range<T>(r);
    return ovf_result<T>{fits ? static_cast<T>(r) : T{}, !fits};
  }
}

//===----------REGISTER CARRIER----------===//
/**
 * @brief One Register Value of a generated kernel's row: the payload, its SQL validity, and its
 * Overflow Poison.
 *
 * Every function of this runtime and every statement that `translation_context` emits keeps one
 * invariant: an operation sets @c overflow only by propagating an operand's Overflow Poison, or
 * from its own computation over operands that are all valid and carry no Overflow Poison. The
 * payload of an invalid value is unspecified and never decides overflow. Overflow Poison therefore
 * always comes from an Evaluated Operation, and the NVRTC Sinks of the `ProjectSink` and
 * `FilterSink` kernels report every Overflow Poison that reaches them, whatever the value's final
 * validity: `n + abs(x)` with `n` NULL and `x` the lowest INTEGER is a NULL that still carries the
 * Overflow Poison of `abs(x)`.
 */
template <typename T>
struct reg_val {
  T val;
  bool valid;
  bool overflow;
};

//===----------BIT CHECK----------===//
__device__ __forceinline__ bool is_bit_set(uint32_t const* mask, int32_t bit_idx)
{
  return ((mask[bit_idx >> 5] >> (bit_idx & 31)) & 1U) != 0U;
}
