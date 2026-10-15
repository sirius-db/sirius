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

// decimal_divide_round(), the arithmetic sirius_decimal_divide() shares between the CPU function
// and the GPU kernel, on unscaled values. Expected quotients were computed with Python's decimal
// module (ROUND_HALF_UP, 200 digits of precision).

#include "helper/decimal_divide.hpp"

#include <catch.hpp>

#include <string>

namespace {

using sirius::decimal_divide_round;
using sirius::decimal_divide_status;

/// Parses a decimal integer literal into __int128 (there is no 128-bit literal syntax). The
/// magnitude is accumulated unsigned, so -2^127 parses too.
__int128_t i128(std::string const& digits)
{
  bool const negative         = !digits.empty() && digits[0] == '-';
  unsigned __int128 magnitude = 0;
  for (auto i = negative ? 1U : 0U; i < digits.size(); ++i) {
    magnitude = magnitude * 10 + static_cast<unsigned>(digits[i] - '0');
  }
  return static_cast<__int128_t>(negative ? ~magnitude + 1 : magnitude);
}

std::string to_string(__int128_t value)
{
  if (value == 0) { return "0"; }
  bool const negative = value < 0;
  auto const bits     = static_cast<unsigned __int128>(value);
  std::string digits;
  for (auto magnitude = negative ? ~bits + 1 : bits; magnitude != 0; magnitude /= 10) {
    digits.insert(digits.begin(), static_cast<char>('0' + static_cast<int>(magnitude % 10)));
  }
  return negative ? "-" + digits : digits;
}

/// The rounded quotient as a string, or the failure status.
std::string divide(std::string const& dividend, std::string const& divisor, int exponent)
{
  __int128_t quotient = 0;
  switch (decimal_divide_round(i128(dividend), i128(divisor), exponent, &quotient)) {
    case decimal_divide_status::ok: return to_string(quotient);
    case decimal_divide_status::divide_by_zero: return "divide_by_zero";
    case decimal_divide_status::overflow: return "overflow";
  }
  return "unreachable";
}

}  // namespace

TEST_CASE("decimal_divide_round rounds half away from zero", "[decimal_divide]")
{
  // DECIMAL(15,2) / DECIMAL(15,2) at scale 8: the dividend is scaled by 10^8.
  CHECK(divide("100", "300", 8) == "33333333");
  CHECK(divide("200", "300", 8) == "66666667");
  CHECK(divide("-200", "300", 8) == "-66666667");
  CHECK(divide("200", "-300", 8) == "-66666667");
  CHECK(divide("-200", "-300", 8) == "66666667");
  // 0.01 / 2000000.00 = 0.000000005 exactly: a tie at the last digit rounds away from zero, not to
  // even (0.05 / 2000000.00 = 0.000000025 gives 3, not 2).
  CHECK(divide("1", "200000000", 8) == "1");
  CHECK(divide("-1", "200000000", 8) == "-1");
  CHECK(divide("3", "200000000", 8) == "2");
  CHECK(divide("5", "200000000", 8) == "3");
  CHECK(divide("-5", "200000000", 8) == "-3");
  CHECK(divide("0", "500", 8) == "0");
  // Integers at scale 0.
  CHECK(divide("5", "2", 0) == "3");
  CHECK(divide("-5", "2", 0) == "-3");
  CHECK(divide("7", "-2", 0) == "-4");
  CHECK(divide("49", "100", 0) == "0");
}

TEST_CASE("decimal_divide_round gives divide_by_zero for a zero divisor", "[decimal_divide]")
{
  CHECK(divide("5", "0", 8) == "divide_by_zero");
  CHECK(divide("0", "0", 0) == "divide_by_zero");
  CHECK(divide("-99999999999999999999999999999999999999", "0", 38) == "divide_by_zero");
}

TEST_CASE("decimal_divide_round scales the divisor for a negative exponent", "[decimal_divide]")
{
  // DECIMAL(15,6) / 1 at scale 2: 1.235000 -> 1.24, 1.234999 -> 1.23.
  CHECK(divide("1235000", "1", -4) == "124");
  CHECK(divide("-1235000", "1", -4) == "-124");
  CHECK(divide("1234999", "1", -4) == "123");
  // 0.000005 / 0.00001 at scale 0 is exactly 0.5.
  CHECK(divide("5", "1", -1) == "1");
  // A divisor scaled past 2^128 exceeds twice any dividend: the quotient is 0.
  CHECK(divide("99999999999999999999999999999999999999",
               "10000000000000000000000000000000000000",
               -38) == "0");
}

TEST_CASE("decimal_divide_round keeps a scaled dividend wider than 128 bits exact",
          "[decimal_divide]")
{
  // 12345678901234567890123456789012.3456 (DECIMAL(38,4)) / 9999 at scale 10: the dividend times
  // 10^6 is about 1.2 * 10^41, past 2^128.
  CHECK(divide("123456789012345678901234567890123456", "9999", 6) ==
        "12346913592593827272850741863198665467");
  CHECK(divide("-123456789012345678901234567890123456", "9999", 6) ==
        "-12346913592593827272850741863198665467");
  // 10^40 / (2^41 * 5^23) = 381469726562.5: a tie in the long-division path.
  CHECK(divide("10000000000000000000000000000000000", "26214400000000000000000000000", 6) ==
        "381469726563");
  CHECK(divide("-10000000000000000000000000000000000", "26214400000000000000000000000", 6) ==
        "-381469726563");
  // DECIMAL(38,2) / DECIMAL(38,2) at scale 12.
  CHECK(divide("1234567890123456789012345678", "300", 12) ==
        "4115226300411522630041152260000000000");
  CHECK(divide("9999999999999999999999999999", "9999999999999999999999999998", 12) ==
        "1000000000000");
}

TEST_CASE("decimal_divide_round reports quotients beyond 38 digits", "[decimal_divide]")
{
  CHECK(divide("99999999999999999999999999999999999999", "1", 0) ==
        "99999999999999999999999999999999999999");
  CHECK(divide("-99999999999999999999999999999999999999", "-1", 0) ==
        "99999999999999999999999999999999999999");
  CHECK(divide("99999999999999999999999999999999999999", "1", 1) == "overflow");
  CHECK(divide("123456789012345678901234567890123456", "1000", 6) == "overflow");
  CHECK(divide("9999999999999999999999999999", "1", 12) == "overflow");
  // 10^37 * 10^76 needs more than 256 bits.
  CHECK(divide("10000000000000000000000000000000000000", "5", 76) == "overflow");
}

TEST_CASE("decimal_divide_round handles the minimum of each operand width", "[decimal_divide]")
{
  constexpr auto int32_min  = "-2147483648";
  constexpr auto int64_min  = "-9223372036854775808";
  constexpr auto int128_min = "-170141183460469231731687303715884105728";
  REQUIRE(to_string(i128(int128_min)) == int128_min);

  CHECK(divide(int32_min, "1", 0) == "-2147483648");
  CHECK(divide(int32_min, "-1", 0) == "2147483648");
  CHECK(divide(int32_min, "7", 8) == "-30678337828571429");
  CHECK(divide("1", int32_min, 10) == "-5");
  CHECK(divide(int64_min, "-1", 0) == "9223372036854775808");
  CHECK(divide(int64_min, "3", 8) == "-307445734561825860266666667");
  CHECK(divide("12345", int64_min, 30) == "-1338447581933355");
  // 2^127 needs 39 digits.
  CHECK(divide(int128_min, "-1", 0) == "overflow");
  CHECK(divide(int128_min, "1", 0) == "overflow");
  CHECK(divide(int128_min, "2", 0) == "-85070591730234615865843651857942052864");
  CHECK(divide(int128_min, "-3", 0) == "56713727820156410577229101238628035243");
  CHECK(divide(int128_min, "10000000000", -2) == "-170141183460469231731687304");
  CHECK(divide("1", int128_min, 0) == "0");
  CHECK(divide("10000000000000000000000000000000000000", int128_min, 2) == "-6");
  CHECK(divide("-99999999999999999999999999999999999999", int128_min, 0) == "1");
  CHECK(divide(int128_min, int128_min, 0) == "1");
  CHECK(divide(int128_min, int128_min, 38) == "overflow");
}

TEST_CASE("decimal_divide_round long division with a divisor above 64 bits", "[decimal_divide]")
{
  // d = 2^70 + 12345; the dividend times 10^20 is about 2^167. The remainders are (d - 1) / 2,
  // just under half, and (d + 1) / 2, just over.
  constexpr auto d = "1180591620717411315769";
  CHECK(divide("1000000000219591430245354663037", d, 20) == "84703294744030150656372924164");
  CHECK(divide("-1000000000219591430245354663037", d, 20) == "-84703294744030150656372924164");
  CHECK(divide("1000000000360550512485983935218", d, 20) == "84703294755969849343627075836");
  CHECK(divide("-1000000000360550512485983935218", d, 20) == "-84703294755969849343627075836");
}

TEST_CASE("decimal_divide_round long division at the 2^128 quotient bound", "[decimal_divide]")
{
  // a * 10^25 has a high half h = 36280689513667850293524. With d = h + 1 the long division runs
  // and its quotient (about 2^128 * h / d) fails the 38-digit check; with d = h the quotient needs
  // more than 128 bits and is refused before the division.
  constexpr auto a = "1234567890123456789012345678901234567";
  CHECK(divide(a, "36280689513667850293525", 25) == "overflow");
  CHECK(divide(a, "36280689513667850293524", 25) == "overflow");
  CHECK(divide("-1234567890123456789012345678901234567", "36280689513667850293524", 25) ==
        "overflow");
}

TEST_CASE("decimal_divide_round scales a wide divisor for a negative exponent", "[decimal_divide]")
{
  // The scaled divisor lies between 2^64 and 2^128, so the 128-bit division runs.
  CHECK(divide("1000000000000000000000000000007", "3000000000000000", -6) == "333333333");
  CHECK(divide("-10000000000000000000000000000123456789", "987654321987654321", -3) ==
        "-10124999989748438");
  CHECK(divide("50000000000000000000000000", "1000000000000000000000000", -1) == "5");
}

TEST_CASE("decimal_divide_round at the largest 38-digit quotient", "[decimal_divide]")
{
  constexpr auto max_quotient = "99999999999999999999999999999999999999";
  // Zero remainder, through the 128-bit division and through the long division.
  CHECK(divide(max_quotient, "1", 0) == max_quotient);
  CHECK(divide(max_quotient, "1000", 3) == max_quotient);
  // 10^38 - 1 + 1/11 rounds down.
  CHECK(divide("109999999999999999999999999999999999999", "11", 1) == max_quotient);
  // Exactly 10^38.
  CHECK(divide("100000000000000000000000000000000000000", "1", 0) == "overflow");
  CHECK(divide("10000000000000000000000000000000000000", "10", 2) == "overflow");
}
