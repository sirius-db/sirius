// Copyright 2026, Sirius Contributors. SPDX-License-Identifier: Apache-2.0

#include "catch.hpp"
#include "config/chrono.hpp"
#include "config/parser.hpp"
#include "config/size.hpp"
#include "scan_manager/config.hpp"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <string_view>
#include <vector>

using namespace sirius::config;
using namespace sirius::config::size_literals;
using namespace std::chrono_literals;

namespace {

template <class T>
T parse_ok(std::string_view text)
{
  T value{};
  auto const status = parser<T>{}(text, value);
  INFO("text: '" << text << "' error: " << status.error());
  REQUIRE(status);
  return value;
}

template <class T>
std::string parse_error(std::string_view text)
{
  T value{};
  auto const status = parser<T>{}(text, value);
  INFO("text: '" << text << "'");
  REQUIRE_FALSE(status);
  return status.error();
}

template <class T>
void require_round_trip(const T& value)
{
  auto const text = formatter<T>{}(value);
  INFO("formatted: '" << text << "'");
  REQUIRE(parse_ok<T>(text) == value);
}

enum class test_speed { slow, fast };

bool string_to_enum(std::string_view text, test_speed& out)
{
  if (text == "slow") {
    out = test_speed::slow;
    return true;
  }
  if (text == "fast" || text == "quick") {  // "quick" is an alias
    out = test_speed::fast;
    return true;
  }
  return false;
}

bool enum_to_string(test_speed value, std::string& out)
{
  out = value == test_speed::fast ? "fast" : "slow";
  return true;
}

enum class test_level : int { low = 1, high = 2 };

}  // namespace

static_assert(string_enum<test_speed>);
static_assert(int_enum<test_level>);
static_assert(parsable<bytes> && parsable<mebibytes> && parsable<std::chrono::milliseconds>);
static_assert(parsable<test_speed> && parsable<test_level> && parsable<double>);
static_assert(!parsable<std::vector<int>>);

TEST_CASE("basic_size converts, compares and adds across units", "[config][config_values]")
{
  STATIC_REQUIRE(1_GiB == 1024_MiB);
  STATIC_REQUIRE(1_KB < 1_KiB);
  STATIC_REQUIRE(1_GiB + 1_MiB == mebibytes(1025));
  STATIC_REQUIRE(size_cast<gibibytes>(1536_MiB).count() == 1);
  STATIC_REQUIRE(bytes(1_KiB).count() == 1024);
  STATIC_REQUIRE(bytes::zero() == 0_B);
  // Lossy conversions are not implicit.
  STATIC_REQUIRE(!std::is_convertible_v<bytes, kibibytes>);
  STATIC_REQUIRE(std::is_convertible_v<kibibytes, bytes>);
}

TEST_CASE("byte sizes parse exactly", "[config][config_values]")
{
  REQUIRE(parse_ok<bytes>("1.5 GiB") == bytes(1610612736));
  REQUIRE(parse_ok<bytes>("1 GB") == bytes(1000000000));
  REQUIRE(parse_ok<bytes>("0 B") == 0_B);
  REQUIRE(parse_ok<bytes>("0.001 KB") == 1_B);
  REQUIRE(parse_ok<bytes>("1.50 GiB") == bytes(1610612736));
  REQUIRE(parse_ok<bytes>("1.000000000000000000000000 GiB") == 1_GiB);
  REQUIRE(parse_ok<bytes>("9223372036854775807 B").count() ==
          std::numeric_limits<std::int64_t>::max());
  REQUIRE(parse_ok<mebibytes>("1 GiB").count() == 1024);

  REQUIRE_THAT(parse_error<bytes>("9223372036854775808 B"),
               Catch::Matchers::ContainsSubstring("out of range"));
  REQUIRE_THAT(parse_error<bytes>("999999999999999999999999 GiB"),
               Catch::Matchers::ContainsSubstring("out of range"));
  REQUIRE_THAT(parse_error<bytes>("0.1 B"),
               Catch::Matchers::ContainsSubstring("would need rounding"));
  REQUIRE_THAT(parse_error<bytes>("1.5"),
               Catch::Matchers::ContainsSubstring("would need rounding"));
  REQUIRE_THAT(parse_error<mebibytes>("1 KiB"),
               Catch::Matchers::ContainsSubstring("would need rounding"));
  REQUIRE(parse_error<bytes>("-1 B") == "byte value must be non-negative");
  REQUIRE(parse_error<bytes>("1 XB") == "unknown byte suffix: 'XB'");
  REQUIRE(parse_error<bytes>("1 s") == "unknown byte suffix: 's'");
  REQUIRE(parse_error<bytes>("1.2.3 B") == "unknown byte suffix: '.3 B'");
  REQUIRE(parse_error<bytes>("GiB") == "invalid byte value: 'GiB'");
  REQUIRE(parse_error<bytes>("") == "invalid byte value: ''");
}

TEST_CASE("byte sizes accept today's spellings", "[config][config_values]")
{
  REQUIRE(parse_ok<bytes>("1048576") == 1_MiB);  // bare: bytes
  REQUIRE(parse_ok<bytes>("2048 b") == 2_KiB);
  REQUIRE(parse_ok<bytes>("8Gi") == 8_GiB);
  REQUIRE(parse_ok<bytes>("512M") == 512_MB);
  REQUIRE(parse_ok<bytes>("1kb") == 1_KB);
  REQUIRE(parse_ok<bytes>("4 TiB") == 4_TiB);
  REQUIRE(parse_ok<bytes>("2T") == 2_TB);
  REQUIRE(parse_ok<bytes>("  64 MiB  ") == 64_MiB);
  REQUIRE(parse_ok<bytes>(".5 KiB") == 512_B);
  REQUIRE(parse_ok<bytes>("-0") == 0_B);
}

TEST_CASE("durations parse exactly", "[config][config_values]")
{
  using std::chrono::milliseconds;
  using std::chrono::seconds;
  REQUIRE(parse_ok<milliseconds>("1.5 s") == 1500ms);
  REQUIRE(parse_ok<milliseconds>("2min") == 120000ms);
  REQUIRE(parse_ok<std::chrono::microseconds>("0.5 ms").count() == 500);
  REQUIRE_THAT(parse_error<milliseconds>("0.5 ms"),
               Catch::Matchers::ContainsSubstring("not a whole number of ms"));
  REQUIRE_THAT(parse_error<seconds>("1.5s"),
               Catch::Matchers::ContainsSubstring("would need rounding"));
  REQUIRE(parse_error<milliseconds>("1 GB") == "unknown time suffix: 'GB'");
  REQUIRE(parse_error<milliseconds>("soon") == "invalid time value: 'soon'");
}

TEST_CASE("durations accept today's spellings", "[config][config_values]")
{
  using std::chrono::milliseconds;
  using std::chrono::seconds;
  REQUIRE(parse_ok<milliseconds>("500") == 500ms);  // bare: the field's own unit
  REQUIRE(parse_ok<seconds>("30") == 30s);
  REQUIRE(parse_ok<milliseconds>("1.0") == 1ms);
  REQUIRE(parse_ok<milliseconds>("10msec") == 10ms);
  REQUIRE(parse_ok<milliseconds>("1 SEC") == 1000ms);
  REQUIRE(parse_ok<seconds>("5 Minutes") == 300s);
  REQUIRE(parse_ok<seconds>("2hr") == 7200s);
  REQUIRE(parse_ok<seconds>("1 m") == 60s);
  REQUIRE(parse_ok<std::chrono::nanoseconds>("3 usec").count() == 3000);
  REQUIRE(parse_ok<milliseconds>("-5ms") == -5ms);
  REQUIRE_THAT(parse_error<milliseconds>("1.5"),
               Catch::Matchers::ContainsSubstring("would need rounding"));
}

TEST_CASE("sizes and durations format with the largest exact unit", "[config][config_values]")
{
  REQUIRE(formatter<bytes>{}(bytes(1610612736)) == "1536 MiB");
  REQUIRE(formatter<bytes>{}(1_GB) == "1 GB");
  REQUIRE(formatter<bytes>{}(1_GiB) == "1 GiB");
  REQUIRE(formatter<bytes>{}(bytes(1000)) == "1 KB");
  REQUIRE(formatter<bytes>{}(bytes(1023)) == "1023 B");
  REQUIRE(formatter<bytes>{}(0_B) == "0 B");
  REQUIRE(formatter<mebibytes>{}(mebibytes(3)) == "3 MiB");
  REQUIRE(formatter<std::chrono::milliseconds>{}(1500ms) == "1500 ms");
  REQUIRE(formatter<std::chrono::seconds>{}(120s) == "2 min");
  REQUIRE(formatter<std::chrono::milliseconds>{}(0ms) == "0 ms");
  REQUIRE(formatter<std::chrono::milliseconds>{}(-3s) == "-3 s");

  for (bytes value :
       {0_B, 1_B, bytes(1_KB), bytes(1_KiB), bytes(1610612736), bytes(7_TiB), bytes(999999937)}) {
    require_round_trip(value);
  }
  for (auto value : {std::chrono::milliseconds{0}, 1ms, 1500ms, 3600000ms, -250ms}) {
    require_round_trip(value);
  }
}

TEST_CASE("integers parse without wrapping", "[config][config_values]")
{
  REQUIRE(parse_ok<int>("42") == 42);
  REQUIRE(parse_ok<int>(" -7 ") == -7);
  REQUIRE(parse_ok<int>("+3") == 3);
  REQUIRE(parse_ok<std::int64_t>("-9223372036854775808") ==
          std::numeric_limits<std::int64_t>::min());
  REQUIRE(parse_ok<std::size_t>("-0") == 0);
  REQUIRE(parse_error<std::size_t>("-1") == "must be a nonnegative integer, got '-1'");
  REQUIRE(parse_error<int>("99999999999") == "integer out of range: '99999999999'");
  REQUIRE(parse_error<std::uint8_t>("256") == "integer out of range: '256'");
  REQUIRE(parse_error<int>("1.5") == "invalid integer value: '1.5'");
  REQUIRE(parse_error<int>("") == "invalid integer value: ''");
  REQUIRE(parse_error<int>("12abc") == "invalid integer value: '12abc'");
  for (auto value : {0, 1, -1, std::numeric_limits<int>::max(), std::numeric_limits<int>::min()}) {
    require_round_trip(value);
  }
}

TEST_CASE("floating point accepts decimal, exponent and YAML specials", "[config][config_values]")
{
  REQUIRE(parse_ok<double>("0.95") == 0.95);
  REQUIRE(parse_ok<double>("1e-3") == 0.001);
  REQUIRE(parse_ok<double>("+2.5") == 2.5);
  REQUIRE(parse_ok<double>(".inf") == std::numeric_limits<double>::infinity());
  REQUIRE(parse_ok<double>("-.Inf") == -std::numeric_limits<double>::infinity());
  REQUIRE(std::isnan(parse_ok<double>(".nan")));
  REQUIRE(parse_error<double>("1.5x") == "invalid number: '1.5x'");
  REQUIRE(parse_error<double>("--1") == "invalid number: '--1'");
  REQUIRE(formatter<double>{}(1.0) == "1.0");
  REQUIRE(formatter<double>{}(0.95) == "0.95");
  REQUIRE(formatter<double>{}(std::numeric_limits<double>::infinity()) == ".inf");
  for (auto value : {0.0, 0.1, 0.95, 1.0 / 3.0, 1e300, -2.5}) {
    require_round_trip(value);
  }
}

TEST_CASE("booleans accept yaml-cpp spellings", "[config][config_values]")
{
  REQUIRE(parse_ok<bool>("true"));
  REQUIRE(parse_ok<bool>("Yes"));
  REQUIRE(parse_ok<bool>("ON"));
  REQUIRE_FALSE(parse_ok<bool>("false"));
  REQUIRE_FALSE(parse_ok<bool>("n"));
  REQUIRE_FALSE(parse_ok<bool>("Off"));
  REQUIRE(parse_error<bool>("maybe") == "invalid boolean value: 'maybe'");
  require_round_trip(true);
  require_round_trip(false);
}

TEST_CASE("strings are taken verbatim", "[config][config_values]")
{
  REQUIRE(parse_ok<std::string>(" spaced ") == " spaced ");
  require_round_trip(std::string("telemetry_data"));
}

TEST_CASE("enums with converters use string_to_enum and enum_to_string", "[config][config_values]")
{
  REQUIRE(parse_ok<test_speed>("fast") == test_speed::fast);
  REQUIRE(parse_ok<test_speed>("quick") == test_speed::fast);    // alias reads
  REQUIRE(formatter<test_speed>{}(test_speed::fast) == "fast");  // canonical spelling writes
  REQUIRE(parse_error<test_speed>("sluggish") == "invalid enum value 'sluggish'");
  require_round_trip(test_speed::slow);

  // A Sirius enum, including its deprecated alias.
  using sirius::scan_manager::io_backend;
  REQUIRE(parse_ok<io_backend>("native") == io_backend::native);
  REQUIRE(parse_ok<io_backend>("sirius") == io_backend::native);
  REQUIRE(formatter<io_backend>{}(io_backend::native) == "native");
}

TEST_CASE("enums without converters use the underlying integer", "[config][config_values]")
{
  REQUIRE(parse_ok<test_level>("2") == test_level::high);
  REQUIRE(formatter<test_level>{}(test_level::low) == "1");
  REQUIRE(parse_error<test_level>("high") == "invalid integer value: 'high'");
}
