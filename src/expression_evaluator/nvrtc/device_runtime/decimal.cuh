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
 * @brief Decimal arithmetic and casts.
 *
 * Fragment 4 of the NVRTC expression evaluator device runtime.
 */

#include "codegen/stdint_shim.hpp"

//===----------DECIMAL ARITHMETIC----------===//
template <typename OutRep, typename L, typename R>
__device__ __forceinline__ reg_val<OutRep> eval_dec_add(reg_val<L> l,
                                                        reg_val<R> r,
                                                        __int128_t f_l,
                                                        __int128_t f_r)
{
  bool const ovf_in = l.overflow || r.overflow;
  bool const valid  = l.valid && r.valid;
  if (!valid || ovf_in) { return reg_val<OutRep>{OutRep{}, valid, ovf_in}; }
  // Operands of at most 64 bits rescaled by at most 10^18 (the widest DECIMAL64 scale) stay far
  // inside 128 bits, so only the final narrowing can overflow.
  if constexpr (sizeof(L) <= 8 && sizeof(R) <= 8) {
    __int128_t const sum =
      static_cast<__int128_t>(l.val) * f_l + static_cast<__int128_t>(r.val) * f_r;
    auto const out = overflow_cast<OutRep>(sum);
    return reg_val<OutRep>{out.overflow ? OutRep{} : out.value, true, out.overflow};
  }
  auto const sl = mul_overflow128(static_cast<__int128_t>(l.val), f_l);
  auto const sr = mul_overflow128(static_cast<__int128_t>(r.val), f_r);
  if (sl.overflow || sr.overflow) { return reg_val<OutRep>{OutRep{}, true, true}; }
  auto const sum = add_overflow128(sl.value, sr.value);
  if (sum.overflow) { return reg_val<OutRep>{OutRep{}, true, true}; }
  auto const out = overflow_cast<OutRep>(sum.value);
  return reg_val<OutRep>{out.overflow ? OutRep{} : out.value, true, out.overflow};
}

template <typename OutRep, typename L, typename R>
__device__ __forceinline__ reg_val<OutRep> eval_dec_sub(reg_val<L> l,
                                                        reg_val<R> r,
                                                        __int128_t f_l,
                                                        __int128_t f_r)
{
  bool const ovf_in = l.overflow || r.overflow;
  bool const valid  = l.valid && r.valid;
  if (!valid || ovf_in) { return reg_val<OutRep>{OutRep{}, valid, ovf_in}; }
  // Operands of at most 64 bits rescaled by at most 10^18 (the widest DECIMAL64 scale) stay far
  // inside 128 bits, so only the final narrowing can overflow.
  if constexpr (sizeof(L) <= 8 && sizeof(R) <= 8) {
    __int128_t const diff =
      static_cast<__int128_t>(l.val) * f_l - static_cast<__int128_t>(r.val) * f_r;
    auto const out = overflow_cast<OutRep>(diff);
    return reg_val<OutRep>{out.overflow ? OutRep{} : out.value, true, out.overflow};
  }
  auto const sl = mul_overflow128(static_cast<__int128_t>(l.val), f_l);
  auto const sr = mul_overflow128(static_cast<__int128_t>(r.val), f_r);
  if (sl.overflow || sr.overflow) { return reg_val<OutRep>{OutRep{}, true, true}; }
  auto const diff = sub_overflow128(sl.value, sr.value);
  if (diff.overflow) { return reg_val<OutRep>{OutRep{}, true, true}; }
  auto const out = overflow_cast<OutRep>(diff.value);
  return reg_val<OutRep>{out.overflow ? OutRep{} : out.value, true, out.overflow};
}

template <typename OutRep, typename L, typename R>
__device__ __forceinline__ reg_val<OutRep> eval_dec_mul(reg_val<L> l, reg_val<R> r)
{
  bool const ovf_in = l.overflow || r.overflow;
  bool const valid  = l.valid && r.valid;
  if (!valid || ovf_in) { return reg_val<OutRep>{OutRep{}, valid, ovf_in}; }
  // Two operands narrower than the result can never overflow it (the product of two 64-bit values
  // always fits in 128 bits); only a same-width product needs the checked multiply.
  if constexpr (sizeof(L) < sizeof(OutRep) && sizeof(R) < sizeof(OutRep)) {
    return reg_val<OutRep>{static_cast<OutRep>(l.val) * static_cast<OutRep>(r.val), true, false};
  } else {
    auto const prod = mul_overflow<OutRep>(static_cast<OutRep>(l.val), static_cast<OutRep>(r.val));
    return reg_val<OutRep>{prod.overflow ? OutRep{} : prod.value, true, prod.overflow};
  }
}

/**
 * @brief Decimal division producing output scale s_out (from @tparam OutRep)from operands of scales
 *        s_l (from @tparam L) and s_r (from @tparam R).
 *
 * The natural quotient has scale s_l - s_r; f_out = 10^|s_out - (s_l - s_r)|. When the output scale
 * is larger (i.e., @tparam ScaleUp = true) the dividend is scaled before dividing so the extra
 * digits are computed rather than zero-filled (1.00 / 3.00 -> DECIMAL(.., 4) is 0.3333).
 */
template <typename OutRep, bool ScaleUp = true, typename L, typename R>
__device__ __forceinline__ reg_val<OutRep> eval_dec_div(reg_val<L> l,
                                                        reg_val<R> r,
                                                        __int128_t f_out = 1)
{
  bool const ovf_in = l.overflow || r.overflow;
  bool const valid  = l.valid && r.valid;
  if (!valid || ovf_in) { return reg_val<OutRep>{OutRep{}, valid, ovf_in}; }
  __int128_t lv       = static_cast<__int128_t>(l.val);
  __int128_t const rv = static_cast<__int128_t>(r.val);
  if (rv == 0) { return reg_val<OutRep>{OutRep{}, false, false}; }
  if constexpr (ScaleUp) {
    if (f_out != 1) {
      auto const scaled = mul_overflow128(lv, f_out);
      if (scaled.overflow) { return reg_val<OutRep>{OutRep{}, true, true}; }
      lv = scaled.value;
    }
  }
  if (lv == limits<__int128_t>::lowest() && rv == -1) {
    return reg_val<OutRep>{OutRep{}, true, true};
  }
  __int128_t q = lv / rv;
  if constexpr (!ScaleUp) { q /= f_out; }
  auto const out = overflow_cast<OutRep>(q);
  return reg_val<OutRep>{out.overflow ? OutRep{} : out.value, true, out.overflow};
}

template <typename OutRep, typename L, typename R>
__device__ __forceinline__ reg_val<OutRep> eval_dec_mod(reg_val<L> l,
                                                        reg_val<R> r,
                                                        __int128_t f_l,
                                                        __int128_t f_r)
{
  bool const ovf_in = l.overflow || r.overflow;
  bool const valid  = l.valid && r.valid;
  if (!valid || ovf_in) { return reg_val<OutRep>{OutRep{}, valid, ovf_in}; }
  auto const sl = mul_overflow128(static_cast<__int128_t>(l.val), f_l);
  auto const sr = mul_overflow128(static_cast<__int128_t>(r.val), f_r);
  if (sl.overflow || sr.overflow) { return reg_val<OutRep>{OutRep{}, true, true}; }
  if (sr.value == 0) { return reg_val<OutRep>{OutRep{}, false, false}; }
  if (sl.value == limits<__int128_t>::lowest() && sr.value == -1) {
    return reg_val<OutRep>{OutRep{0}, true, false};
  }
  __int128_t const rem = sl.value % sr.value;
  auto const out       = overflow_cast<OutRep>(rem);
  return reg_val<OutRep>{out.overflow ? OutRep{} : out.value, true, out.overflow};
}

//===----------DECIMAL CAST----------===//
template <typename DstRep, bool CanFitOne>
__device__ __forceinline__ reg_val<DstRep> eval_cast_bool_to_dec(reg_val<bool> in,
                                                                 DstRep one_scaled)
{
  if (!in.valid || in.overflow) { return reg_val<DstRep>{DstRep{}, in.valid, in.overflow}; }
  if (!in.val) { return reg_val<DstRep>{DstRep{0}, true, false}; }
  if constexpr (CanFitOne) {
    return reg_val<DstRep>{one_scaled, true, false};
  } else {
    return reg_val<DstRep>{DstRep{}, true, true};
  }
}

template <typename DstRep, typename SrcInt>
__device__ __forceinline__ reg_val<DstRep> eval_cast_int_to_dec(reg_val<SrcInt> in,
                                                                __int128_t int_limit,
                                                                __int128_t factor)
{
  if (!in.valid || in.overflow) { return reg_val<DstRep>{DstRep{}, in.valid, in.overflow}; }
  __int128_t const v = static_cast<__int128_t>(in.val);
  if (v <= -int_limit || v >= int_limit) { return reg_val<DstRep>{DstRep{}, true, true}; }
  auto const scaled = mul_overflow128(v, factor);
  if (scaled.overflow) { return reg_val<DstRep>{DstRep{}, true, true}; }
  auto const out = overflow_cast<DstRep>(scaled.value);
  return reg_val<DstRep>{out.overflow ? DstRep{} : out.value, true, out.overflow};
}

template <typename DstInt, typename SrcRep>
__device__ __forceinline__ reg_val<DstInt> eval_cast_dec_to_int(reg_val<SrcRep> in,
                                                                __int128_t factor)
{
  if (!in.valid || in.overflow) { return reg_val<DstInt>{DstInt{}, in.valid, in.overflow}; }
  __int128_t const q = static_cast<__int128_t>(in.val) / factor;
  bool const fits    = in_range<DstInt>(q);
  return reg_val<DstInt>{fits ? static_cast<DstInt>(q) : DstInt{}, true, !fits};
}

template <typename DstRep, typename SrcRep>
__device__ __forceinline__ reg_val<DstRep> eval_cast_dec_scale_up(reg_val<SrcRep> in,
                                                                  __int128_t factor,
                                                                  __int128_t limit)
{
  if (!in.valid || in.overflow) { return reg_val<DstRep>{DstRep{}, in.valid, in.overflow}; }
  auto const scaled = mul_overflow128(static_cast<__int128_t>(in.val), factor);
  if (scaled.overflow || scaled.value <= -limit || scaled.value >= limit) {
    return reg_val<DstRep>{DstRep{}, true, true};
  }
  auto const out = overflow_cast<DstRep>(scaled.value);
  return reg_val<DstRep>{out.overflow ? DstRep{} : out.value, true, out.overflow};
}

template <typename DstRep, typename SrcRep>
__device__ __forceinline__ reg_val<DstRep> eval_cast_dec_scale_down(reg_val<SrcRep> in,
                                                                    __int128_t factor,
                                                                    __int128_t limit)
{
  if (!in.valid || in.overflow) { return reg_val<DstRep>{DstRep{}, in.valid, in.overflow}; }
  __int128_t const q = static_cast<__int128_t>(in.val) / factor;
  if (q <= -limit || q >= limit) { return reg_val<DstRep>{DstRep{}, true, true}; }
  auto const out = overflow_cast<DstRep>(q);
  return reg_val<DstRep>{out.overflow ? DstRep{} : out.value, true, out.overflow};
}

template <typename DstFloat, typename SrcRep>
__device__ __forceinline__ reg_val<DstFloat> eval_cast_dec_to_float(reg_val<SrcRep> in,
                                                                    double scale_pow10)
{
  if (!in.valid || in.overflow) { return reg_val<DstFloat>{DstFloat{}, in.valid, in.overflow}; }
  if constexpr (sizeof(DstFloat) == 4) {
    return reg_val<float>{
      static_cast<float>(in.val) / static_cast<float>(scale_pow10), true, false};
  } else {
    return reg_val<double>{static_cast<double>(in.val) / scale_pow10, true, false};
  }
}

template <typename DstRep, typename SrcFloat>
__device__ __forceinline__ reg_val<DstRep> eval_cast_float_to_dec(reg_val<SrcFloat> in,
                                                                  double scale_pow10,
                                                                  double limit_pow10,
                                                                  __int128_t limit_128)
{
  if (!in.valid || in.overflow) { return reg_val<DstRep>{DstRep{}, in.valid, in.overflow}; }
  double const d = static_cast<double>(in.val);
  if (!is_finite_double(d)) { return reg_val<DstRep>{DstRep{}, true, true}; }
  double const scaled = d * scale_pow10;
  if (!is_finite_double(scaled) || scaled <= -limit_pow10 || scaled >= limit_pow10) {
    return reg_val<DstRep>{DstRep{}, true, true};
  }
  __int128_t const iv = static_cast<__int128_t>(scaled);
  if (iv <= -limit_128 || iv >= limit_128) { return reg_val<DstRep>{DstRep{}, true, true}; }
  auto const out = overflow_cast<DstRep>(iv);
  return reg_val<DstRep>{out.overflow ? DstRep{} : out.value, true, out.overflow};
}
