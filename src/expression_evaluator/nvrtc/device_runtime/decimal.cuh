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
 * @brief Decimal arithmetic, comparison operands and casts.
 *
 * Fragment 4 of the NVRTC expression evaluator device runtime.
 */

//===----------DECIMAL ARITHMETIC----------===//
/**
 * @brief @p v * @p factor for the power of ten @p factor that brings a decimal to a larger scale.
 */
__device__ __forceinline__ ovf_result<__int128_t> rescale(__int128_t v, __int128_t factor)
{
  // The factor is a literal at every call site, so this test folds and equal scales cost nothing.
  if (factor == 1) { return ovf_result<__int128_t>{v, false}; }
  return checked_mul128(v, factor);
}

/**
 * @brief Whether @p v lies outside [@p lowest, @p highest], the values DuckDB returns for a sum,
 *        difference or product of its SQL type.
 *
 * A DECIMAL holds fewer than 10^precision in magnitude, so DuckDB raises for a result with more
 * digits although it fits the carrier. A HUGEINT holds everything its carrier holds.
 */
template <typename Rep>
__device__ __forceinline__ bool leaves_type(Rep v, Rep lowest, Rep highest)
{
  return v < lowest || v > highest;
}

/**
 * @brief Decimal addition (@tparam Subtract false) or subtraction of operands brought to their
 *        common scale by the factors @p f_l and @p f_r, into a type that holds [@p lowest, @p
 *        highest].
 *
 * Namely, f_l = 10^(s_out - s_l) and f_r = 10^(s_out - s_r) -- the multipliers that bring the left
 * and right operands to the output's decimal scale -- for the output scale s_out and the operands'
 * scales s_l and s_r. The factors are literals at every call site, so the test folds.
 */
template <typename OutRep, bool Subtract, typename L, typename R>
__device__ __forceinline__ reg_val<OutRep> eval_dec_add_sub(
  reg_val<L> l, reg_val<R> r, __int128_t f_l, __int128_t f_r, OutRep lowest, OutRep highest)
{
  bool const poison = l.overflow || r.overflow;
  bool const valid  = l.valid && r.valid;
  if constexpr (sizeof(L) <= sizeof(OutRep) &&
                sizeof(R) <= sizeof(OutRep)) {  // Operands no wider than the result's carrier
    if (f_l == 1 && f_r == 1) {                 // Operands at the result's scale
      OutRep const lv = static_cast<OutRep>(l.val);
      OutRep const rv = static_cast<OutRep>(r.val);
      auto const out  = Subtract ? checked_sub<OutRep>(lv, rv) : checked_add<OutRep>(lv, rv);
      return operation_result<OutRep>(
        out.value, valid, poison, out.overflow || leaves_type(out.value, lowest, highest));
    }
  }
  if constexpr (sizeof(L) <= 8 && sizeof(R) <= 8) {
    // Rescaled by at most 10^18 (the widest DECIMAL64 scale) the operands stay inside 128 bits, so
    // only the final narrowing can overflow.
    __int128_t const sl = static_cast<__int128_t>(l.val) * f_l;
    __int128_t const sr = static_cast<__int128_t>(r.val) * f_r;
    auto const out      = checked_narrow<OutRep>(Subtract ? sl - sr : sl + sr);
    return operation_result<OutRep>(
      out.value, valid, poison, out.overflow || leaves_type(out.value, lowest, highest));
  } else {
    auto const sl  = rescale(static_cast<__int128_t>(l.val), f_l);
    auto const sr  = rescale(static_cast<__int128_t>(r.val), f_r);
    auto const sum = Subtract ? checked_sub<__int128_t>(sl.value, sr.value)
                              : checked_add<__int128_t>(sl.value, sr.value);
    auto const out = checked_narrow<OutRep>(sum.value);
    return operation_result<OutRep>(out.value,
                                    valid,
                                    poison,
                                    sl.overflow || sr.overflow || sum.overflow || out.overflow ||
                                      leaves_type(out.value, lowest, highest));
  }
}

template <typename OutRep, typename L, typename R>
__device__ __forceinline__ reg_val<OutRep> eval_dec_add(
  reg_val<L> l, reg_val<R> r, __int128_t f_l, __int128_t f_r, OutRep lowest, OutRep highest)
{
  return eval_dec_add_sub<OutRep, false>(l, r, f_l, f_r, lowest, highest);
}

template <typename OutRep, typename L, typename R>
__device__ __forceinline__ reg_val<OutRep> eval_dec_sub(
  reg_val<L> l, reg_val<R> r, __int128_t f_l, __int128_t f_r, OutRep lowest, OutRep highest)
{
  return eval_dec_add_sub<OutRep, true>(l, r, f_l, f_r, lowest, highest);
}

/** @brief Decimal multiplication into a type that holds [@p lowest, @p highest]. */
template <typename OutRep, typename L, typename R>
__device__ __forceinline__ reg_val<OutRep> eval_dec_mul(reg_val<L> l,
                                                        reg_val<R> r,
                                                        OutRep lowest,
                                                        OutRep highest)
{
  bool const poison = l.overflow || r.overflow;
  bool const valid  = l.valid && r.valid;
  OutRep const lv   = static_cast<OutRep>(l.val);
  OutRep const rv   = static_cast<OutRep>(r.val);
  // Only same-width product needs checked multiply.
  if constexpr (sizeof(L) < sizeof(OutRep) && sizeof(R) < sizeof(OutRep)) {
    OutRep const product = lv * rv;
    return operation_result<OutRep>(product, valid, poison, leaves_type(product, lowest, highest));
  } else {
    auto const product = checked_mul<OutRep>(lv, rv);
    return operation_result<OutRep>(
      product.value,
      valid,
      poison,
      product.overflow || leaves_type(product.value, lowest, highest));
  }
}

/**
 * @brief Decimal division producing output scale s_out (from @tparam OutRep)from operands of scales
 *        s_l (from @tparam L) and s_r (from @tparam R).
 *
 * The natural quotient has scale s_l - s_r; f_out = 10^|s_out - (s_l - s_r)|. When the output scale
 * is larger (i.e., @tparam ScaleUp = true) the dividend is scaled before dividing so the extra
 * digits are computed rather than zero-filled (1.00 / 3.00 -> DECIMAL(.., 4) is 0.3333).
 *
 * A zero divisor raises, as it does in `eval_int_div`.
 */
template <typename OutRep, bool ScaleUp = true, typename L, typename R>
__device__ __forceinline__ reg_val<OutRep> eval_dec_div(reg_val<L> l,
                                                        reg_val<R> r,
                                                        __int128_t f_out = 1)
{
  bool const poison   = l.overflow || r.overflow;
  bool const valid    = l.valid && r.valid;
  __int128_t lv       = static_cast<__int128_t>(l.val);
  __int128_t const rv = static_cast<__int128_t>(r.val);
  bool overflows      = false;
  if constexpr (ScaleUp) {
    auto const scaled = rescale(lv, f_out);
    overflows         = scaled.overflow;
    lv                = scaled.value;
  }
  overflows    = overflows || (lv == limits<__int128_t>::lowest() && rv == -1);
  __int128_t q = lv / defined_divisor(lv, rv);
  if constexpr (!ScaleUp) { q /= f_out; }
  auto const out = checked_narrow<OutRep>(q);
  return operation_result<OutRep>(out.value, valid, poison, rv == 0 || overflows || out.overflow);
}

/**
 * @brief Decimal modulo of operands brought to their common scale by the factors @p f_l and @p f_r.
 *
 * A zero divisor raises, as it does in `eval_int_div`, and so does the lowest 128-bit value modulo
 * -1, as DuckDB raises for it: a HUGEINT can hold that value, a DECIMAL cannot.
 */
template <typename OutRep, typename L, typename R>
__device__ __forceinline__ reg_val<OutRep> eval_dec_mod(reg_val<L> l,
                                                        reg_val<R> r,
                                                        __int128_t f_l,
                                                        __int128_t f_r)
{
  bool const poison = l.overflow || r.overflow;
  bool const valid  = l.valid && r.valid;
  auto const sl     = rescale(static_cast<__int128_t>(l.val), f_l);
  auto const sr     = rescale(static_cast<__int128_t>(r.val), f_r);
  auto const out    = checked_narrow<OutRep>(sl.value % defined_divisor(sl.value, sr.value));
  return operation_result<OutRep>(out.value,
                                  valid,
                                  poison,
                                  sl.overflow || sr.overflow || sr.value == 0 || out.overflow ||
                                    divides_lowest_by_minus_one(sl.value, sr.value));
}

//===----------DECIMAL COMPARISON----------===//
/**
 * @brief @p v * @p factor modulo 2^128, for comparing decimals of different scales.
 */
__device__ __forceinline__ __int128_t rescale_wrapping(__int128_t v, __int128_t factor)
{
  return static_cast<__int128_t>(static_cast<unsigned __int128>(v) *
                                 static_cast<unsigned __int128>(factor));
}

//===----------DECIMAL CAST----------===//
/**
 * @brief @p v / @p factor rounded half away from zero, as DuckDB rounds a DECIMAL cast that drops
 *        scale digits, including a cast to an integer or to HUGEINT (carried as a scale-0
 *        DECIMAL128).
 */
__device__ __forceinline__ __int128_t div_round_half_away(__int128_t v, __int128_t factor)
{
  __int128_t const q     = v / factor;
  __int128_t const r     = v - q * factor;  // reuses the quotient instead of a second division
  __int128_t const abs_r = r < 0 ? -r : r;
  // abs_r >= factor - abs_r is 2 * abs_r >= factor without overflowing for factor near 10^38.
  if (abs_r >= factor - abs_r) { return v < 0 ? q - 1 : q + 1; }
  return q;
}

template <typename DstRep, bool CanFitOne>
__device__ __forceinline__ reg_val<DstRep> eval_cast_bool_to_dec(reg_val<bool> in,
                                                                 DstRep one_scaled)
{
  return operation_result<DstRep>(
    in.val ? one_scaled : DstRep{0}, in.valid, in.overflow, in.val && !CanFitOne);
}

template <typename DstRep, typename SrcInt>
__device__ __forceinline__ reg_val<DstRep> eval_cast_int_to_dec(reg_val<SrcInt> in,
                                                                __int128_t int_limit,
                                                                __int128_t factor)
{
  __int128_t const v = static_cast<__int128_t>(in.val);
  auto const scaled  = rescale(v, factor);
  auto const out     = checked_narrow<DstRep>(scaled.value);
  return operation_result<DstRep>(
    out.value,
    in.valid,
    in.overflow,
    v <= -int_limit || v >= int_limit || scaled.overflow || out.overflow);
}

template <typename DstInt, typename SrcRep>
__device__ __forceinline__ reg_val<DstInt> eval_cast_dec_to_int(reg_val<SrcRep> in,
                                                                __int128_t factor)
{
  auto const out =
    checked_narrow<DstInt>(div_round_half_away(static_cast<__int128_t>(in.val), factor));
  return operation_result<DstInt>(out.value, in.valid, in.overflow, out.overflow);
}

template <typename DstRep, typename SrcRep>
__device__ __forceinline__ reg_val<DstRep> eval_cast_dec_scale_up(reg_val<SrcRep> in,
                                                                  __int128_t factor,
                                                                  __int128_t limit)
{
  auto const scaled = rescale(static_cast<__int128_t>(in.val), factor);
  auto const out    = checked_narrow<DstRep>(scaled.value);
  return operation_result<DstRep>(
    out.value,
    in.valid,
    in.overflow,
    scaled.overflow || scaled.value <= -limit || scaled.value >= limit || out.overflow);
}

template <typename DstRep, typename SrcRep>
__device__ __forceinline__ reg_val<DstRep> eval_cast_dec_scale_down(reg_val<SrcRep> in,
                                                                    __int128_t factor,
                                                                    __int128_t limit)
{
  __int128_t const q = div_round_half_away(static_cast<__int128_t>(in.val), factor);
  auto const out     = checked_narrow<DstRep>(q);
  return operation_result<DstRep>(
    out.value, in.valid, in.overflow, q <= -limit || q >= limit || out.overflow);
}

/**
 * @brief The DECIMAL's integer converted to the destination type and then divided by 10^scale, so
 *        the result is rounded twice.
 *
 * @note DuckDB returns the DOUBLE or FLOAT nearest to the exact decimal value. The two agree where
 *       the integer and 10^scale are exact in the destination type: up to a magnitude of 2^53 and a
 *       scale of 22 for DOUBLE, 2^24 and 10 for FLOAT. Beyond that the last bit can differ.
 */
template <typename DstFloat, typename SrcRep>
__device__ __forceinline__ reg_val<DstFloat> eval_cast_dec_to_float(reg_val<SrcRep> in,
                                                                    double scale_pow10)
{
  if constexpr (sizeof(DstFloat) == 4) {
    return reg_val<float>{
      static_cast<float>(in.val) / static_cast<float>(scale_pow10), in.valid, in.overflow};
  } else {
    return reg_val<double>{
      div_by_pow10(static_cast<double>(in.val), scale_pow10), in.valid, in.overflow};
  }
}

template <typename DstRep, typename SrcFloat>
__device__ __forceinline__ reg_val<DstRep> eval_cast_float_to_dec(reg_val<SrcFloat> in,
                                                                  double scale_pow10,
                                                                  double limit_pow10,
                                                                  __int128_t limit_128)
{
  double const scaled = static_cast<double>(in.val) * scale_pow10;
  bool const outside  = !(scaled > -limit_pow10 && scaled < limit_pow10);
  __int128_t const iv = static_cast<__int128_t>(outside ? 0.0 : scaled);
  auto const out      = checked_narrow<DstRep>(iv);
  return operation_result<DstRep>(out.value,
                                  in.valid,
                                  in.overflow,
                                  outside || iv <= -limit_128 || iv >= limit_128 || out.overflow);
}
