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
#include <type_traits>

namespace sirius::utils {

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
