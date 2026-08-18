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

// standard library
#include <cstddef>
#include <string_view>
#include <type_traits>

namespace sirius::utils {

/// Natural (digit-aware) name ordering: maximal digit runs compare by numeric value (a longer
/// run of significant digits is larger; equal-length runs compare lexicographically, which
/// equals numeric order), everything else compares byte-wise. So "part.2" < "part.10" while
/// plain lexicographic order would give "part.10" < "part.2". Used to pin multi-file datasets
/// in their logical part order regardless of readdir order.
inline bool natural_name_less(std::string_view lhs, std::string_view rhs)
{
  std::size_t i = 0, j = 0;
  auto is_digit = [](char c) { return c >= '0' && c <= '9'; };
  while (i < lhs.size() && j < rhs.size()) {
    if (is_digit(lhs[i]) && is_digit(rhs[j])) {
      // Skip leading zeros, then compare the significant digit runs.
      std::size_t li = i, rj = j;
      while (li < lhs.size() && lhs[li] == '0')
        ++li;
      while (rj < rhs.size() && rhs[rj] == '0')
        ++rj;
      std::size_t le = li, re = rj;
      while (le < lhs.size() && is_digit(lhs[le]))
        ++le;
      while (re < rhs.size() && is_digit(rhs[re]))
        ++re;
      auto const llen = le - li;
      auto const rlen = re - rj;
      if (llen != rlen) { return llen < rlen; }
      auto const lrun = lhs.substr(li, llen);
      auto const rrun = rhs.substr(rj, rlen);
      if (lrun != rrun) { return lrun < rrun; }
      // Equal numeric value: fewer leading zeros first, for a total deterministic order.
      if ((li - i) != (rj - j)) { return (li - i) < (rj - j); }
      i = le;
      j = re;
      continue;
    }
    if (lhs[i] != rhs[j]) { return lhs[i] < rhs[j]; }
    ++i;
    ++j;
  }
  return (lhs.size() - i) < (rhs.size() - j);
}

template <typename T>
inline constexpr T ceil_div(T a, T b)
{
  static_assert(std::is_integral<T>::value, "ceil_div requires an integral type");
  return (a + b - 1) / b;
}

template <typename T>
inline constexpr T ceil_div_8(T a)
{
  static_assert(std::is_integral<T>::value, "ceil_div_8 requires an integral type");
  static_assert(std::is_unsigned<T>::value, "ceil_div_8 requires an unsigned type");
  return (a + 7) >> 3;
}

template <typename T>
inline constexpr T div_8(T a)
{
  static_assert(std::is_integral<T>::value, "div_8 requires an integral type");
  static_assert(std::is_unsigned<T>::value, "div_8 requires an unsigned type");
  return a >> 3;
}

template <typename T>
inline constexpr T mul_8(T a)
{
  static_assert(std::is_integral<T>::value, "mul_8 requires an integral type");
  static_assert(std::is_unsigned<T>::value, "mul_8 requires an unsigned type");
  return a << 3;
}

template <typename T>
inline constexpr T mod_8(T a)
{
  static_assert(std::is_integral<T>::value, "mod_8 requires an integral type");
  static_assert(std::is_unsigned<T>::value, "mod_8 requires an unsigned type");
  return a & 7;
}

template <typename S, typename T>
inline constexpr S make_mask(T num_bits)
{
  static_assert(std::is_integral<T>::value, "make_mask requires an integral type for num_bits");
  static_assert(std::is_unsigned<T>::value, "make_mask requires an unsigned type for num_bits");
  static_assert(std::is_integral<S>::value, "make_mask requires an integral type for return");
  static_assert(std::is_unsigned<S>::value, "make_mask requires an unsigned type for return");
  return static_cast<S>((static_cast<S>(1) << num_bits) - 1);
}

}  // namespace sirius::utils
