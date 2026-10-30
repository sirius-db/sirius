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
 * @brief Kleene AND/OR, checked integer and floating-point arithmetic, abs, round, and the numeric
 *        casts that do not involve decimals.
 *
 * Fragment 3 of the NVRTC expression evaluator device runtime.
 */

#include "codegen/stdint_shim.hpp"

//===----------KLEENE AND/OR----------===//
__device__ __forceinline__ reg_val<bool> eval_kleene_and(reg_val<bool> acc, reg_val<bool> rhs)
{
  bool const acc_false = acc.valid && !acc.overflow && !acc.val;
  bool const rhs_false = rhs.valid && !rhs.overflow && !rhs.val;
  if (acc_false || rhs_false) { return reg_val<bool>{false, true, false}; }
  if (acc.overflow || rhs.overflow) { return reg_val<bool>{false, true, true}; }
  if (acc.valid && rhs.valid) { return reg_val<bool>{true, true, false}; }
  return reg_val<bool>{false, false, false};
}

__device__ __forceinline__ reg_val<bool> eval_kleene_or(reg_val<bool> acc, reg_val<bool> rhs)
{
  bool const acc_true = acc.valid && !acc.overflow && acc.val;
  bool const rhs_true = rhs.valid && !rhs.overflow && rhs.val;
  if (acc_true || rhs_true) { return reg_val<bool>{true, true, false}; }
  if (acc.overflow || rhs.overflow) { return reg_val<bool>{false, true, true}; }
  if (acc.valid && rhs.valid) { return reg_val<bool>{false, true, false}; }
  return reg_val<bool>{false, false, false};
}

//===----------CHECKED ARITHMETIC----------===//
template <typename T, typename L, typename R>
__device__ __forceinline__ reg_val<T> eval_add(reg_val<L> l, reg_val<R> r)
{
  bool const ovf_in = l.overflow || r.overflow;
  bool const valid  = l.valid && r.valid;
  if (!valid || ovf_in) { return reg_val<T>{T{}, valid, ovf_in}; }
  if constexpr (is_floating_point_v<T>) {
    return reg_val<T>{static_cast<T>(l.val) + static_cast<T>(r.val), true, false};
    // 64-bit sums: wrap in the unsigned domain and read the overflow off the sign bits (signed) or
    // the carry (unsigned) instead of widening to 128 bits.
  } else if constexpr (sizeof(T) == 8) {
    T const lv  = static_cast<T>(l.val);
    T const rv  = static_cast<T>(r.val);
    T const res = static_cast<T>(static_cast<uint64_t>(lv) + static_cast<uint64_t>(rv));
    bool ovf;
    if constexpr (is_signed_v<T>) {
      ovf = static_cast<int64_t>((lv ^ res) & (rv ^ res)) < 0;
    } else {
      ovf = res < lv;
    }
    return reg_val<T>{ovf ? T{} : res, true, ovf};
  } else if constexpr (is_signed_v<T>) {
    using Wide = conditional_t<(sizeof(T) < 8), int64_t, __int128_t>;
    Wide const res =
      static_cast<Wide>(static_cast<T>(l.val)) + static_cast<Wide>(static_cast<T>(r.val));
    bool const ovf =
      (res < static_cast<Wide>(limits<T>::lowest())) || (res > static_cast<Wide>(limits<T>::max()));
    return reg_val<T>{ovf ? T{} : static_cast<T>(res), true, ovf};
  } else {
    using Wide = conditional_t<(sizeof(T) < 8), uint64_t, unsigned __int128>;
    Wide const res =
      static_cast<Wide>(static_cast<T>(l.val)) + static_cast<Wide>(static_cast<T>(r.val));
    bool const ovf = res > static_cast<Wide>(limits<T>::max());
    return reg_val<T>{ovf ? T{} : static_cast<T>(res), true, ovf};
  }
}

template <typename T, typename L, typename R>
__device__ __forceinline__ reg_val<T> eval_sub(reg_val<L> l, reg_val<R> r)
{
  bool const ovf_in = l.overflow || r.overflow;
  bool const valid  = l.valid && r.valid;
  if (!valid || ovf_in) { return reg_val<T>{T{}, valid, ovf_in}; }
  if constexpr (is_floating_point_v<T>) {
    return reg_val<T>{static_cast<T>(l.val) - static_cast<T>(r.val), true, false};
  } else if constexpr (sizeof(T) == 8 && is_signed_v<T>) {
    T const lv     = static_cast<T>(l.val);
    T const rv     = static_cast<T>(r.val);
    T const res    = static_cast<T>(static_cast<uint64_t>(lv) - static_cast<uint64_t>(rv));
    bool const ovf = static_cast<int64_t>((lv ^ rv) & (lv ^ res)) < 0;
    return reg_val<T>{ovf ? T{} : res, true, ovf};
  } else if constexpr (is_signed_v<T>) {
    using Wide = conditional_t<(sizeof(T) < 8), int64_t, __int128_t>;
    Wide const res =
      static_cast<Wide>(static_cast<T>(l.val)) - static_cast<Wide>(static_cast<T>(r.val));
    bool const ovf =
      (res < static_cast<Wide>(limits<T>::lowest())) || (res > static_cast<Wide>(limits<T>::max()));
    return reg_val<T>{ovf ? T{} : static_cast<T>(res), true, ovf};
  } else {
    T const lv     = static_cast<T>(l.val);
    T const rv     = static_cast<T>(r.val);
    bool const ovf = lv < rv;
    return reg_val<T>{ovf ? T{} : static_cast<T>(lv - rv), true, ovf};
  }
}

template <typename T, typename L, typename R>
__device__ __forceinline__ reg_val<T> eval_mul(reg_val<L> l, reg_val<R> r)
{
  bool const ovf_in = l.overflow || r.overflow;
  bool const valid  = l.valid && r.valid;
  if (!valid || ovf_in) { return reg_val<T>{T{}, valid, ovf_in}; }
  if constexpr (is_floating_point_v<T>) {
    return reg_val<T>{static_cast<T>(l.val) * static_cast<T>(r.val), true, false};
  } else if constexpr (sizeof(T) == 8) {
    // 64-bit products: the high half of the 128-bit product is one instruction, and the product
    // fits iff that half is the sign extension of the low half (signed) or zero (unsigned).
    T const lv = static_cast<T>(l.val);
    T const rv = static_cast<T>(r.val);
    T const lo = static_cast<T>(static_cast<uint64_t>(lv) * static_cast<uint64_t>(rv));
    bool ovf;
    if constexpr (is_signed_v<T>) {
      ovf = __mul64hi(lv, rv) != (lo >> 63);
    } else {
      ovf = __umul64hi(lv, rv) != 0ULL;
    }
    return reg_val<T>{ovf ? T{} : lo, true, ovf};
  } else if constexpr (is_signed_v<T>) {
    int64_t const res =
      static_cast<int64_t>(static_cast<T>(l.val)) * static_cast<int64_t>(static_cast<T>(r.val));
    bool const ovf = (res < static_cast<int64_t>(limits<T>::lowest())) ||
                     (res > static_cast<int64_t>(limits<T>::max()));
    return reg_val<T>{ovf ? T{} : static_cast<T>(res), true, ovf};
  } else {
    uint64_t const res =
      static_cast<uint64_t>(static_cast<T>(l.val)) * static_cast<uint64_t>(static_cast<T>(r.val));
    bool const ovf = res > static_cast<uint64_t>(limits<T>::max());
    return reg_val<T>{ovf ? T{} : static_cast<T>(res), true, ovf};
  }
}

template <typename T, typename L, typename R>
__device__ __forceinline__ reg_val<T> eval_div(reg_val<L> l, reg_val<R> r)
{
  bool const ovf_in = l.overflow || r.overflow;
  bool const valid  = l.valid && r.valid;
  if (!valid || ovf_in) { return reg_val<T>{T{}, valid, ovf_in}; }
  T const lv = static_cast<T>(l.val);
  T const rv = static_cast<T>(r.val);
  if constexpr (is_floating_point_v<T>) {
    return reg_val<T>{lv / rv, true, false};
  } else {
    if (rv == T{0}) { return reg_val<T>{T{}, false, false}; }
    if constexpr (is_signed_v<T>) {
      if (lv == limits<T>::lowest() && rv == T{-1}) { return reg_val<T>{T{}, true, true}; }
    }
    return reg_val<T>{static_cast<T>(lv / rv), true, false};
  }
}

template <typename T, typename L, typename R>
__device__ __forceinline__ reg_val<T> eval_mod(reg_val<L> l, reg_val<R> r)
{
  bool const ovf_in = l.overflow || r.overflow;
  bool const valid  = l.valid && r.valid;
  if (!valid || ovf_in) { return reg_val<T>{T{}, valid, ovf_in}; }
  T const lv = static_cast<T>(l.val);
  T const rv = static_cast<T>(r.val);
  if constexpr (is_floating_point_v<T>) {
    if constexpr (sizeof(T) == 4) {
      return reg_val<T>{::fmodf(lv, rv), true, false};
    } else {
      return reg_val<T>{::fmod(lv, rv), true, false};
    }
  } else {
    if (rv == T{0}) { return reg_val<T>{T{}, false, false}; }
    if constexpr (is_signed_v<T>) {
      if (lv == limits<T>::lowest() && rv == T{-1}) { return reg_val<T>{T{0}, true, false}; }
    }
    return reg_val<T>{static_cast<T>(lv % rv), true, false};
  }
}

template <typename T, typename InT>
__device__ __forceinline__ reg_val<T> eval_abs(reg_val<InT> in)
{
  if (!in.valid || in.overflow) { return reg_val<T>{T{}, in.valid, in.overflow}; }
  T const v = static_cast<T>(in.val);
  if constexpr (is_floating_point_v<T>) {
    if constexpr (sizeof(T) == 4) {
      return reg_val<T>{::fabsf(v), true, false};
    } else {
      return reg_val<T>{::fabs(v), true, false};
    }
  } else if constexpr (is_signed_v<T>) {
    if (v == limits<T>::lowest()) { return reg_val<T>{T{}, true, true}; }
    return reg_val<T>{static_cast<T>(v < T{0} ? -v : v), true, false};
  } else {
    return reg_val<T>{v, true, false};
  }
}

//===----------ROUND----------===//
__device__ __forceinline__ bool is_finite_double(double x) { return (x - x) == 0.0; }

template <typename T, bool NegativePrecision>
__device__ __forceinline__ reg_val<T> eval_round(reg_val<T> in, double modifier)
{
  if (!in.valid || in.overflow) { return reg_val<T>{T{}, in.valid, in.overflow}; }
  double const x = static_cast<double>(in.val);
  double rounded;
  if constexpr (NegativePrecision) {
    rounded = ::round(x / modifier) * modifier;
    if (!is_finite_double(rounded)) { rounded = 0.0; }
  } else {
    rounded = ::round(x * modifier) / modifier;
    if (!is_finite_double(rounded)) { rounded = x; }
  }
  return reg_val<T>{static_cast<T>(rounded), true, false};
}

//===----------CAST----------===//
template <typename DstT, typename SrcT>
__device__ __forceinline__ reg_val<DstT> eval_cast_int_to_int(reg_val<SrcT> in)
{
  if (!in.valid || in.overflow) { return reg_val<DstT>{DstT{}, in.valid, in.overflow}; }
  bool const fits = in_range<DstT>(in.val);
  return reg_val<DstT>{fits ? static_cast<DstT>(in.val) : DstT{}, true, !fits};
}

template <typename SrcT>
__device__ __forceinline__ reg_val<bool> eval_cast_to_bool(reg_val<SrcT> in)
{
  if (!in.valid || in.overflow) { return reg_val<bool>{false, in.valid, in.overflow}; }
  return reg_val<bool>{in.val != SrcT{0}, true, false};
}

template <typename DstFloat, typename SrcFloat>
__device__ __forceinline__ reg_val<DstFloat> eval_cast_float_to_float(reg_val<SrcFloat> in)
{
  if (!in.valid || in.overflow) { return reg_val<DstFloat>{DstFloat{}, in.valid, in.overflow}; }
  if constexpr (sizeof(DstFloat) < sizeof(SrcFloat)) {
    double const d = static_cast<double>(in.val);
    if (!is_finite_double(d)) { return reg_val<DstFloat>{static_cast<DstFloat>(d), true, false}; }
    float const f       = static_cast<float>(d);
    bool const f_finite = (f - f) == 0.0f;
    return reg_val<DstFloat>{f_finite ? f : DstFloat{}, true, !f_finite};
  } else {
    return reg_val<DstFloat>{static_cast<DstFloat>(in.val), true, false};
  }
}

/**
 * @brief Float/double to integer cast.
 *
 * @note Follows DuckDB's @c TryCastWithOverflowCheckFloat: the value is rounded to the nearest
 * integer with ties to even (std::nearbyint under the default rounding mode), and it is the rounded
 * value that must lie in [lowest, max + 1); a non-finite input overflows.
 */
template <typename DstInt, typename SrcFloat>
__device__ __forceinline__ reg_val<DstInt> eval_cast_float_to_int(reg_val<SrcFloat> in)
{
  if (!in.valid || in.overflow) { return reg_val<DstInt>{DstInt{}, in.valid, in.overflow}; }
  double const d = static_cast<double>(in.val);
  if (!is_finite_double(d)) { return reg_val<DstInt>{DstInt{}, true, true}; }
  double const rounded = ::rint(d);
  // The bounds are the exact doubles -2^(bits - 1) (or 0) and 2^(bits - 1) (or 2^bits): for the
  // 64-bit types max() already rounds up to that power of two and the + 1.0 is absorbed.
  constexpr double one_past_max = static_cast<double>(limits<DstInt>::max()) + 1.0;
  constexpr double lo           = static_cast<double>(limits<DstInt>::lowest());
  if (rounded < lo || rounded >= one_past_max) { return reg_val<DstInt>{DstInt{}, true, true}; }
  return reg_val<DstInt>{static_cast<DstInt>(rounded), true, false};
}
