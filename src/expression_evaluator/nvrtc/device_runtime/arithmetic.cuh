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
 * @brief Kleene AND/OR, checked integer and floating-point arithmetic, abs, the division by a
 *        literal power of ten, round, and the numeric casts that do not involve decimals.
 *
 * Fragment 3 of the NVRTC expression evaluator device runtime.
 */

//===----------KLEENE AND/OR----------===//
// REMINDER: Kleene logic: TRUE = 1, FALSE = 0, NULL = 0.5,
//                         L AND R = min(L, R), L OR R = max(L, R).
//
// `translation_context` accumulates AND/OR operands one at a time using `eval_kleene_and` or
// `eval_kleene_or`. Each helper receives two already-evaluated values and preserves Overflow Poison
// from either one, even for FALSE AND ... or TRUE OR ....
//
// The helpers check for a deciding FALSE (AND) or TRUE (OR) first, guarded by `!poisoned`.
// When the compiler can prove neither operand can overflow, this simplifies to plain Kleene logic.
//
// Where skipping a later SQL operand is allowed, the generated code uses the accumulator
// checks defined below to decide whether to evaluate that operand. A skipped operand produces
// no Overflow Poison; an evaluated operand's Overflow Poison must propagate

/**
 * @brief Whether a later AND operand may change the accumulator: false only for a FALSE without
 *        Overflow Poison.
 */
__device__ __forceinline__ bool kleene_and_may_change(reg_val<bool> acc)
{
  return !(acc.valid && !acc.overflow && !acc.val);
}

/**
 * @brief Whether a later OR operand may change the accumulator: false only for a TRUE without
 *        Overflow Poison.
 */
__device__ __forceinline__ bool kleene_or_may_change(reg_val<bool> acc)
{
  return !(acc.valid && !acc.overflow && acc.val);
}

/**
 * @brief Whether DuckDB's `Select` runs a later operand of an AND condition for the row: only while
 *        every earlier operand is TRUE.
 */
__device__ __forceinline__ bool condition_and_continues(reg_val<bool> acc)
{
  return acc.valid && !acc.overflow && acc.val;
}

__device__ __forceinline__ reg_val<bool> eval_kleene_and(reg_val<bool> acc, reg_val<bool> rhs)
{
  bool const poisoned  = acc.overflow || rhs.overflow;
  bool const acc_false = acc.valid && !acc.val;
  bool const rhs_false = rhs.valid && !rhs.val;
  if (!poisoned && (acc_false || rhs_false)) { return reg_val<bool>{false, true, false}; }
  if (poisoned) { return reg_val<bool>{false, true, true}; }
  if (acc.valid && rhs.valid) { return reg_val<bool>{true, true, false}; }
  return reg_val<bool>{false, false, false};
}

__device__ __forceinline__ reg_val<bool> eval_kleene_or(reg_val<bool> acc, reg_val<bool> rhs)
{
  bool const poisoned = acc.overflow || rhs.overflow;
  bool const acc_true = acc.valid && acc.val;
  bool const rhs_true = rhs.valid && rhs.val;
  if (!poisoned && (acc_true || rhs_true)) { return reg_val<bool>{true, true, false}; }
  if (poisoned) { return reg_val<bool>{false, true, true}; }
  if (acc.valid && rhs.valid) { return reg_val<bool>{false, true, false}; }
  return reg_val<bool>{false, false, false};
}

//===----------FLOATING-POINT COMPARISON----------===//
// DuckDB orders NaN above every other value, +inf included, and treats all NaNs as equal.
template <typename T>
__device__ __forceinline__ bool fp_is_nan(T v)
{
  return v != v;
}

template <typename T>
__device__ __forceinline__ bool eval_fp_eq(T a, T b)
{
  return (fp_is_nan(a) && fp_is_nan(b)) || a == b;
}

template <typename T>
__device__ __forceinline__ bool eval_fp_lt(T a, T b)
{
  return !fp_is_nan(a) && (fp_is_nan(b) || a < b);
}

template <typename T>
__device__ __forceinline__ bool eval_fp_le(T a, T b)
{
  return fp_is_nan(b) || (!fp_is_nan(a) && a <= b);
}

//===----------CHECKED ARITHMETIC----------===//
template <typename T, typename L, typename R>
__device__ __forceinline__ reg_val<T> eval_add(reg_val<L> l, reg_val<R> r)
{
  bool const poison = l.overflow || r.overflow;
  bool const valid  = l.valid && r.valid;
  if constexpr (is_floating_point_v<T>) {
    return reg_val<T>{static_cast<T>(l.val) + static_cast<T>(r.val), valid, poison};
  } else {
    auto const sum = checked_add<T>(static_cast<T>(l.val), static_cast<T>(r.val));
    return operation_result<T>(sum.value, valid, poison, sum.overflow);
  }
}

template <typename T, typename L, typename R>
__device__ __forceinline__ reg_val<T> eval_sub(reg_val<L> l, reg_val<R> r)
{
  bool const poison = l.overflow || r.overflow;
  bool const valid  = l.valid && r.valid;
  if constexpr (is_floating_point_v<T>) {
    return reg_val<T>{static_cast<T>(l.val) - static_cast<T>(r.val), valid, poison};
  } else {
    auto const diff = checked_sub<T>(static_cast<T>(l.val), static_cast<T>(r.val));
    return operation_result<T>(diff.value, valid, poison, diff.overflow);
  }
}

template <typename T, typename L, typename R>
__device__ __forceinline__ reg_val<T> eval_mul(reg_val<L> l, reg_val<R> r)
{
  bool const poison = l.overflow || r.overflow;
  bool const valid  = l.valid && r.valid;
  if constexpr (is_floating_point_v<T>) {
    return reg_val<T>{static_cast<T>(l.val) * static_cast<T>(r.val), valid, poison};
  } else {
    auto const prod = checked_mul<T>(static_cast<T>(l.val), static_cast<T>(r.val));
    return operation_result<T>(prod.value, valid, poison, prod.overflow);
  }
}

/**
 * @brief Whether an integer division or modulo of @p dividend by @p divisor is the lowest value by
 *        -1, for which DuckDB raises an overflow: the quotient is one past the largest value.
 */
template <typename T>
__device__ __forceinline__ bool divides_lowest_by_minus_one(T dividend, T divisor)
{
  if constexpr (is_signed_v<T>) {
    return dividend == limits<T>::lowest() && divisor == T{-1};
  } else {
    return false;
  }
}

/**
 * @brief The divisor an integer division or modulo of @p dividend by @p divisor runs with: 1 where
 *        the hardware operation is undefined (a zero divisor, the lowest value by -1), @p divisor
 *        elsewhere.
 *
 * A valid row with such operands carries Overflow Poison, so the substitution changes no result.
 */
template <typename T>
__device__ __forceinline__ T defined_divisor(T dividend, T divisor)
{
  return (divisor == T{0} || divides_lowest_by_minus_one(dividend, divisor)) ? T{1} : divisor;
}

/**
 * @brief SQL `//`, which raises for a zero divisor in every type: the row carries Overflow Poison,
 *        and CPU Replay returns DuckDB's error, or its NULL under `error_on_division_by_zero =
 *        false`.
 */
template <typename T, typename L, typename R>
__device__ __forceinline__ reg_val<T> eval_int_div(reg_val<L> l, reg_val<R> r)
{
  bool const poison = l.overflow || r.overflow;
  bool const valid  = l.valid && r.valid;
  T const lv        = static_cast<T>(l.val);
  T const rv        = static_cast<T>(r.val);
  if constexpr (is_floating_point_v<T>) {
    return operation_result<T>(
      (valid ? lv : T{1}) / (valid ? rv : T{1}), valid, poison, rv == T{0});
  } else {
    return operation_result<T>(static_cast<T>(lv / defined_divisor(lv, rv)),
                               valid,
                               poison,
                               rv == T{0} || divides_lowest_by_minus_one(lv, rv));
  }
}

/**
 * @brief SQL `/`. A DOUBLE or FLOAT quotient follows IEEE 754, so a zero divisor gives an infinity
 *        or NaN; any other type divides as `//` does.
 */
template <typename T, typename L, typename R>
__device__ __forceinline__ reg_val<T> eval_div(reg_val<L> l, reg_val<R> r)
{
  if constexpr (is_floating_point_v<T>) {
    bool const valid = l.valid && r.valid;
    // A NULL row divides 1 by 1, for the reason `eval_int_div` gives.
    return reg_val<T>{
      (valid ? static_cast<T>(l.val) : T{1}) / (valid ? static_cast<T>(r.val) : T{1}),
      valid,
      l.overflow || r.overflow};
  } else {
    return eval_int_div<T>(l, r);
  }
}

/**
 * @brief SQL `%`. A DOUBLE or FLOAT remainder follows IEEE 754; any other type raises for a zero
 *        divisor as `//` does.
 */
template <typename T, typename L, typename R>
__device__ __forceinline__ reg_val<T> eval_mod(reg_val<L> l, reg_val<R> r)
{
  bool const poison = l.overflow || r.overflow;
  bool const valid  = l.valid && r.valid;
  T const lv        = static_cast<T>(l.val);
  T const rv        = static_cast<T>(r.val);
  if constexpr (is_floating_point_v<T>) {
    if constexpr (sizeof(T) == 4) {
      return reg_val<T>{::fmodf(lv, rv), valid, poison};
    } else {
      return reg_val<T>{::fmod(lv, rv), valid, poison};
    }
  } else {
    return operation_result<T>(static_cast<T>(lv % defined_divisor(lv, rv)),
                               valid,
                               poison,
                               rv == T{0} || divides_lowest_by_minus_one(lv, rv));
  }
}

template <typename T, typename InT>
__device__ __forceinline__ reg_val<T> eval_negate(reg_val<InT> in)
{
  T const v = static_cast<T>(in.val);
  if constexpr (is_floating_point_v<T>) {
    return reg_val<T>{-v, in.valid, in.overflow};
  } else {
    using U = make_unsigned_t<T>;
    return operation_result<T>(
      static_cast<T>(U{0} - static_cast<U>(v)), in.valid, in.overflow, v == limits<T>::lowest());
  }
}

template <typename T, typename InT>
__device__ __forceinline__ reg_val<T> eval_abs(reg_val<InT> in)
{
  T const v = static_cast<T>(in.val);
  if constexpr (is_floating_point_v<T>) {
    if constexpr (sizeof(T) == 4) {
      return reg_val<T>{::fabsf(v), in.valid, in.overflow};
    } else {
      return reg_val<T>{::fabs(v), in.valid, in.overflow};
    }
  } else if constexpr (is_signed_v<T>) {
    using U         = make_unsigned_t<T>;
    T const negated = static_cast<T>(U{0} - static_cast<U>(v));
    return operation_result<T>(
      v < T{0} ? negated : v, in.valid, in.overflow, v == limits<T>::lowest());
  } else {
    return reg_val<T>{v, in.valid, in.overflow};
  }
}

//===----------DIVISION BY A POWER OF TEN----------===//
/** @brief The largest divisor `div_by_pow10` accepts: 10^38, the widest DECIMAL scale. */
inline constexpr double div_by_pow10_max = 1e38;

/**
 * @brief The correctly rounded quotient @p x / @p divisor, which is the result of the hardware
 *        division, for a literal divisor.
 *
 * The compiler folds the reciprocal of the literal, so the division costs a multiplication and two
 * fused multiply-adds instead of the double-precision divide sequence: the first one computes the
 * remainder of the estimated quotient exactly and the second one corrects the estimate by it. The
 * result is the correctly rounded quotient because the reciprocal is correctly rounded, no step
 * overflows or underflows for the operands the preconditions admit, and the divisor's significand
 * is not all ones (Markstein, "Computation of elementary functions on the IBM RISC System/6000
 * processor", 1990). Both fused multiply-adds are explicit `fma` calls, so the result is the same
 * whether or not the kernel is compiled with `CompileOptions::fuse_multiply_add`.
 *
 * @pre @p divisor is in [1, `div_by_pow10_max`] and its significand is not all ones. The doubles
 *         `translation_context` prints for 10^s, 0 <= s <= 38, qualify.
 * @pre @p x is +0, or it is finite with a magnitude of at least 1. A dividend of -0 would return
 *         +0.
 */
__device__ __forceinline__ double div_by_pow10(double x, double divisor)
{
  // The test folds for a literal divisor.
  if (divisor == 1.0) { return x; }
  double const reciprocal = 1.0 / divisor;
  double const estimate   = x * reciprocal;                // estimate ~ x / divisor
  double const remainder  = ::fma(-estimate, divisor, x);  // remainder = x - estimate * divisor
  return ::fma(
    remainder, reciprocal, estimate);  // corrected estimate = estimate + remainder / divisor
}

//===----------ROUND----------===//
__device__ __forceinline__ bool is_finite_double(double x) { return (x - x) == 0.0; }

/**
 * @brief DuckDB's `round(x, n)` (`RoundOperatorPrecision`) for the literal @p modifier 10^|n|,
 *        which is infinite for |n| >= 39.
 */
template <typename T, bool NegativePrecision>
__device__ __forceinline__ reg_val<T> eval_round(reg_val<T> in, double modifier)
{
  double const x = static_cast<double>(in.val);
  double rounded;
  if constexpr (NegativePrecision) {
    if (!is_finite_double(modifier)) {
      // Every finite value rounds to 0; an infinity and a NaN saturate.
      rounded = is_finite_double(x) ? 0.0 : x;
    } else {
      // A value too large to be rounded to this precision saturates (infinity and Nan).
      double const multiple = ::round(x / modifier) * modifier;  // Shift left, round, shift right.
      rounded               = is_finite_double(multiple) ? multiple : x;
    }
  } else {
    double const scaled = ::round(x * modifier);
    double const quotient =
      modifier > div_by_pow10_max
        ? scaled / modifier  // If the modifier is too large, use direct division.
        : (scaled == 0.0 ? scaled : div_by_pow10(scaled, modifier));
    rounded = is_finite_double(scaled) ? quotient : x;
  }
  return reg_val<T>{static_cast<T>(rounded), in.valid, in.overflow};
}

//===----------CAST----------===//
template <typename DstT, typename SrcT>
__device__ __forceinline__ reg_val<DstT> eval_cast_int_to_int(reg_val<SrcT> in)
{
  auto const out = checked_narrow<DstT>(in.val);
  return operation_result<DstT>(out.value, in.valid, in.overflow, out.overflow);
}

template <typename SrcT>
__device__ __forceinline__ reg_val<bool> eval_cast_to_bool(reg_val<SrcT> in)
{
  return reg_val<bool>{in.val != SrcT{0}, in.valid, in.overflow};
}

template <typename DstFloat, typename SrcFloat>
__device__ __forceinline__ reg_val<DstFloat> eval_cast_float_to_float(reg_val<SrcFloat> in)
{
  if constexpr (sizeof(DstFloat) < sizeof(SrcFloat)) {
    // A finite DOUBLE that leaves the FLOAT range overflows; an infinity or a NaN converts.
    double const d      = static_cast<double>(in.val);
    float const f       = static_cast<float>(d);
    bool const f_finite = (f - f) == 0.0f;
    return operation_result<DstFloat>(f, in.valid, in.overflow, is_finite_double(d) && !f_finite);
  } else {
    return reg_val<DstFloat>{static_cast<DstFloat>(in.val), in.valid, in.overflow};
  }
}

/**
 * @brief Float/double to integer cast.
 *
 * @note Follows DuckDB's @c TryCastWithOverflowCheckFloat: the value is rounded to the nearest
 *       integer with ties to even (@c std::nearbyint under the default rounding mode), and it is
 *       the rounded value that must lie in [lowest, max + 1); a non-finite input overflows.
 */
template <typename DstInt, typename SrcFloat>
__device__ __forceinline__ reg_val<DstInt> eval_cast_float_to_int(reg_val<SrcFloat> in)
{
  double const rounded          = ::rint(static_cast<double>(in.val));
  constexpr double one_past_max = static_cast<double>(limits<DstInt>::max()) + 1.0;
  constexpr double lo           = static_cast<double>(limits<DstInt>::lowest());
  bool const overflows          = !(rounded >= lo && rounded < one_past_max);
  return operation_result<DstInt>(
    static_cast<DstInt>(overflows ? 0.0 : rounded), in.valid, in.overflow, overflows);
}
