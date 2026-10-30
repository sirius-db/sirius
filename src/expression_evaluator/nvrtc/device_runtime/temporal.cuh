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
 * @brief DATE <-> TIMESTAMP casts,
 *
 * @note Rounding contract (DuckDB's TryCastTimestampBase, Timestamp::GetDate):
 *   - coarse -> fine precision (@c eval_cast_ts_scale_up, @c eval_cast_date_to_timestamp): exact
 *     multiply, overflow is poisoned;
 *   - fine -> coarse precision (@c eval_cast_ts_round_down): round to the nearest coarse tick with
 *     ties away from the epoch, for every pair (ns -> us, us -> ms, us -> s, ...);
 *   - TIMESTAMP -> DATE (@c eval_cast_timestamp_to_date): floor, so 1969-12-31 23:00 is 1969-12-31;
 *   - @c date_trunc result reconciliation (@c eval_cast_ts_scale_down): floor division. It is only
 *     applied to values already truncated to a whole unit of the coarser type, so no rounding
 *     decision is involved.
 * @note +/-infinity (INT64_MAX and INT64_MIN) pass through every conversion unchanged.
 *
 * Fragment 6 of the NVRTC expression evaluator device runtime.
 */

#include "codegen/stdint_shim.hpp"

using extract_component = sirius::nvrtc::abi::extract_component;

//===----------DATE <-> TIMESTAMP CASTS----------===//
template <int64_t Scale>
__device__ __forceinline__ reg_val<int64_t> eval_cast_date_to_timestamp(reg_val<int32_t> in)
{
  if (!in.valid || in.overflow) { return reg_val<int64_t>{0, in.valid, in.overflow}; }
  constexpr int32_t date_inf       = limits<int32_t>::max();
  constexpr int64_t ts_inf         = limits<int64_t>::max();
  constexpr int64_t micros_per_day = 86400000000LL;
  constexpr int64_t max_days       = ts_inf / (Scale > micros_per_day ? Scale : micros_per_day);
  int32_t const day                = in.val;
  if (day == date_inf) { return reg_val<int64_t>{ts_inf, true, false}; }
  if (day == -date_inf) { return reg_val<int64_t>{-ts_inf, true, false}; }
  if (day < -max_days || day > max_days) { return reg_val<int64_t>{0, true, true}; }
  return reg_val<int64_t>{static_cast<int64_t>(day) * Scale, true, false};
}

template <int64_t Scale>
__device__ __forceinline__ reg_val<int64_t> eval_cast_ts_scale_up(reg_val<int64_t> in)
{
  if (!in.valid || in.overflow) { return reg_val<int64_t>{0, in.valid, in.overflow}; }
  constexpr int64_t ts_inf = limits<int64_t>::max();
  constexpr int64_t ts_min = limits<int64_t>::lowest();
  int64_t const t          = in.val;
  if (t == ts_inf || t == -ts_inf) { return reg_val<int64_t>{t, true, false}; }
  if (t < (ts_min / Scale) || t > (ts_inf / Scale)) { return reg_val<int64_t>{0, true, true}; }
  return reg_val<int64_t>{t * Scale, true, false};
}

template <int64_t Scale>
__device__ __forceinline__ reg_val<int64_t> eval_cast_ts_round_down(reg_val<int64_t> in)
{
  static_assert(Scale >= 10 && Scale % 2 == 0, "precision ratios are even powers of ten");
  if (!in.valid || in.overflow) { return reg_val<int64_t>{0, in.valid, in.overflow}; }
  constexpr int64_t ts_inf = limits<int64_t>::max();
  int64_t const t          = in.val;
  if (t == ts_inf || t == -ts_inf) { return reg_val<int64_t>{t, true, false}; }
  // Dividing by half the scale first keeps the +1/-1 tie adjustment inside int64.
  int64_t const half    = t / (Scale / 2);
  int64_t const rounded = half + (half < 0 ? -1LL : 1LL);
  return reg_val<int64_t>{rounded / 2LL, true, false};
}

template <int64_t Scale>
__device__ __forceinline__ reg_val<int64_t> eval_cast_ts_scale_down(reg_val<int64_t> in)
{
  if (!in.valid || in.overflow) { return reg_val<int64_t>{0, in.valid, in.overflow}; }
  constexpr int64_t ts_inf = limits<int64_t>::max();
  int64_t const t          = in.val;
  if (t == ts_inf || t == -ts_inf) { return reg_val<int64_t>{t, true, false}; }
  int64_t const q = (t >= 0) ? (t / Scale) : ((t - Scale + 1LL) / Scale);
  return reg_val<int64_t>{q, true, false};
}

template <int64_t Scale>
__device__ __forceinline__ reg_val<int32_t> eval_cast_timestamp_to_date(reg_val<int64_t> in)
{
  if (!in.valid || in.overflow) { return reg_val<int32_t>{0, in.valid, in.overflow}; }
  constexpr int32_t date_inf = limits<int32_t>::max();
  constexpr int64_t ts_inf   = limits<int64_t>::max();
  int64_t const t            = in.val;
  if (t == ts_inf) { return reg_val<int32_t>{date_inf, true, false}; }
  if (t == -ts_inf) { return reg_val<int32_t>{-date_inf, true, false}; }
  int64_t const days = (t >= 0) ? (t / Scale) : ((t - Scale + 1LL) / Scale);
  if (days < static_cast<int64_t>(limits<int32_t>::lowest()) ||
      days >= static_cast<int64_t>(date_inf)) {
    return reg_val<int32_t>{0, true, true};
  }
  return reg_val<int32_t>{static_cast<int32_t>(days), true, false};
}

//===----------DATE/TIMESTAMP EXTRACTION----------===//
/**
 * @brief Proleptic Gregorian date from a 32-bit count of days since 1970-01-01.
 *
 * This is @c cuda::std::chrono::year_month_day's algorithm (H. Hinnant's @c civil_from_days) with
 * the same 32-bit int arithmetic, so the +/-infinity sentinels wrap exactly as they do in the
 * cuDF-based evaluators; the wrapping steps use unsigned arithmetic to stay well defined.
 */
__device__ __forceinline__ void civil_from_days(int32_t days, int16_t& y, int16_t& m, int16_t& d)
{
  int32_t const z     = static_cast<int32_t>(static_cast<uint32_t>(days) + 719468u);
  int32_t const z_neg = static_cast<int32_t>(static_cast<uint32_t>(z) - 146096u);
  // `z >= 0` is tested as `days >= -719468`: that is how the optimizer folds the (formally
  // overflowing) test in the compiled chrono code, and it is what decides the sentinel's value.
  int32_t const era  = (days >= -719468 ? z : z_neg) / 146097;
  uint32_t const doe = static_cast<uint32_t>(z) - static_cast<uint32_t>(era) * 146097u;
  uint32_t const yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
  int32_t const yr   = static_cast<int32_t>(yoe) + era * 400;
  uint32_t const doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
  uint32_t const mp  = (5 * doy + 2) / 153;
  uint32_t const dd  = doy - (153 * mp + 2) / 5 + 1;
  uint32_t const mth = mp + (mp < 10 ? 3u : static_cast<uint32_t>(-9));
  // chrono::year holds a short and chrono::month / chrono::day an unsigned char.
  y = static_cast<int16_t>(yr + (mth <= 2 ? 1 : 0));
  m = static_cast<int16_t>(static_cast<uint8_t>(mth));
  d = static_cast<int16_t>(static_cast<uint8_t>(dd));
}

template <typename OutT, extract_component Comp>
__device__ __forceinline__ reg_val<OutT> eval_extract_date(reg_val<int32_t> in)
{
  if (!in.valid || in.overflow) { return reg_val<OutT>{OutT{}, in.valid, in.overflow}; }
  int32_t const days = in.val;
  if constexpr (Comp == extract_component::MILLISECOND || Comp == extract_component::MICROSECOND) {
    constexpr int32_t date_inf = limits<int32_t>::max();
    if (days == date_inf || days == -date_inf) { return reg_val<OutT>{OutT{}, false, false}; }
    return reg_val<OutT>{OutT{0}, true, false};
  } else if constexpr (Comp == extract_component::HOUR || Comp == extract_component::MINUTE ||
                       Comp == extract_component::SECOND) {
    return reg_val<OutT>{OutT{0}, true, false};
  } else {
    int16_t y = 0, m = 0, d = 0;
    civil_from_days(days, y, m, d);
    int16_t const res =
      (Comp == extract_component::YEAR) ? y : ((Comp == extract_component::MONTH) ? m : d);
    return reg_val<OutT>{static_cast<OutT>(res), true, false};
  }
}

template <typename OutT, int64_t TicksPerSec, extract_component Comp>
__device__ __forceinline__ reg_val<OutT> eval_extract_timestamp(reg_val<int64_t> in)
{
  if (!in.valid || in.overflow) { return reg_val<OutT>{OutT{}, in.valid, in.overflow}; }
  int64_t const ticks = in.val;
  if constexpr (Comp == extract_component::MILLISECOND || Comp == extract_component::MICROSECOND) {
    constexpr int64_t ts_inf = limits<int64_t>::max();
    if (ticks == ts_inf || ticks == -ts_inf) { return reg_val<OutT>{OutT{}, false, false}; }
  }
  constexpr int64_t ticks_per_day = 86400LL * TicksPerSec;
  if constexpr (Comp <= extract_component::SECOND) {
    // cuda::std::chrono::floor<days>: the day count is a 32-bit `int` (truncating division, then
    // one step back when the result overshoots), so the sentinel timestamps wrap exactly as in the
    // cuDF-based evaluators; finite values get the ordinary floor.
    int32_t day32 = static_cast<int32_t>(ticks / ticks_per_day);
    if (static_cast<int64_t>(day32) * ticks_per_day > ticks) {
      day32 = static_cast<int32_t>(static_cast<uint32_t>(day32) - 1u);
    }
    if constexpr (Comp <= extract_component::DAY) {
      int16_t y = 0, m = 0, d = 0;
      civil_from_days(day32, y, m, d);
      int16_t const res =
        (Comp == extract_component::YEAR) ? y : ((Comp == extract_component::MONTH) ? m : d);
      return reg_val<OutT>{static_cast<OutT>(res), true, false};
    } else {
      // cuDF's extract_component_operator: the time since midnight is moved forward by one day
      // when the wrapped day count leaves it negative, then split with truncating duration_casts.
      int64_t tsm = ticks - static_cast<int64_t>(day32) * ticks_per_day;
      if (tsm < 0) { tsm += ticks_per_day; }
      int64_t const hrs = tsm / (3600LL * TicksPerSec);
      if constexpr (Comp == extract_component::HOUR) {
        return reg_val<OutT>{static_cast<OutT>(static_cast<int16_t>(hrs)), true, false};
      } else if constexpr (Comp == extract_component::MINUTE) {
        return reg_val<OutT>{
          static_cast<OutT>(static_cast<int16_t>(tsm / (60LL * TicksPerSec) - hrs * 60LL)),
          true,
          false};
      } else {
        int64_t const mins = tsm / (60LL * TicksPerSec) - hrs * 60LL;
        return reg_val<OutT>{
          static_cast<OutT>(static_cast<int16_t>(tsm / TicksPerSec - hrs * 3600LL - mins * 60LL)),
          true,
          false};
      }
    }
  } else {
    int64_t rem = ticks % ticks_per_day;
    if (rem < 0) { rem += ticks_per_day; }
    int64_t const sec_of_day = rem / TicksPerSec;
    int64_t const subsec     = rem % TicksPerSec;
    int64_t const sec        = static_cast<int64_t>(static_cast<int16_t>(sec_of_day % 60LL));
    int64_t ms_part          = 0LL;
    if constexpr (TicksPerSec == 1000LL) {
      ms_part = static_cast<int64_t>(static_cast<int16_t>(subsec));
    } else if constexpr (TicksPerSec == 1000000LL) {
      ms_part = static_cast<int64_t>(static_cast<int16_t>(subsec / 1000LL));
    } else if constexpr (TicksPerSec == 1000000000LL) {
      ms_part = static_cast<int64_t>(static_cast<int16_t>(subsec / 1000000LL));
    }
    int64_t const ms_total = sec * 1000LL + ms_part;
    if constexpr (Comp == extract_component::MILLISECOND) {
      return reg_val<OutT>{static_cast<OutT>(ms_total), true, false};
    } else {
      int64_t us_part = 0LL;
      if constexpr (TicksPerSec == 1000000LL) {
        us_part = static_cast<int64_t>(static_cast<int16_t>(subsec % 1000LL));
      } else if constexpr (TicksPerSec == 1000000000LL) {
        us_part = static_cast<int64_t>(static_cast<int16_t>((subsec / 1000LL) % 1000LL));
      }
      return reg_val<OutT>{static_cast<OutT>(ms_total * 1000LL + us_part), true, false};
    }
  }
}

//===----------DATE TRUNCATION----------===//
template <int64_t Divisor>
__device__ __forceinline__ reg_val<int64_t> eval_date_trunc(reg_val<int64_t> in)
{
  if (!in.valid || in.overflow) { return reg_val<int64_t>{0, in.valid, in.overflow}; }
  if constexpr (Divisor <= 1LL) {
    return in;
  } else {
    int64_t const t = in.val;
    int64_t q       = t / Divisor;
    int64_t const r = t % Divisor;
    if (r < 0) { q -= 1LL; }
    return reg_val<int64_t>{q * Divisor, true, false};
  }
}
