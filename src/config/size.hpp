// Copyright 2026, Sirius Contributors. SPDX-License-Identifier: Apache-2.0
#pragma once

#include "config/parser.hpp"
#include "config/quantity.hpp"

#include <compare>
#include <concepts>
#include <cstdint>
#include <numeric>
#include <ratio>
#include <string>
#include <string_view>
#include <type_traits>

namespace sirius::config {

/**
 * @brief A byte count with a compile-time unit, modelled on std::chrono::duration.
 *
 * Conversions that cannot lose information are implicit (GiB -> bytes); the others need
 * size_cast. Values of different units compare and add in their common unit.
 *
 * @tparam Rep the count type.
 * @tparam Period std::ratio: bytes per tick.
 */
template <class Rep, class Period = std::ratio<1>>
class basic_size;

namespace detail {

template <class>
struct is_basic_size : std::false_type {};
template <class R, class P>
struct is_basic_size<basic_size<R, P>> : std::true_type {};

template <class P1, class P2>
using common_size_period = std::ratio<std::gcd(P1::num, P2::num), std::lcm(P1::den, P2::den)>;

}  // namespace detail

template <class Rep, class Period>
class basic_size {
 public:
  using rep    = Rep;
  using period = typename Period::type;

  constexpr basic_size() = default;
  constexpr explicit basic_size(const Rep& count) : _count(count) {}

  /// Implicit only when every tick of @p other is a whole number of ticks of this unit.
  template <class R2, class P2>
    requires(std::floating_point<Rep> ||
             (std::ratio_divide<P2, Period>::den == 1 && !std::floating_point<R2>))
  constexpr basic_size(const basic_size<R2, P2>& other)  // NOLINT(google-explicit-constructor)
    : _count(static_cast<Rep>(other.count()) * std::ratio_divide<P2, Period>::num /
             std::ratio_divide<P2, Period>::den)
  {
  }

  [[nodiscard]] constexpr Rep count() const { return _count; }
  [[nodiscard]] static constexpr basic_size zero() { return basic_size(Rep{}); }

  constexpr basic_size& operator+=(const basic_size& other)
  {
    _count += other._count;
    return *this;
  }
  constexpr basic_size& operator-=(const basic_size& other)
  {
    _count -= other._count;
    return *this;
  }
  constexpr basic_size& operator*=(const Rep& factor)
  {
    _count *= factor;
    return *this;
  }

 private:
  Rep _count{};
};

template <class R1, class P1, class R2, class P2>
using common_size_t = basic_size<std::common_type_t<R1, R2>, detail::common_size_period<P1, P2>>;

template <class R1, class P1, class R2, class P2>
constexpr auto operator+(const basic_size<R1, P1>& lhs, const basic_size<R2, P2>& rhs)
{
  using C = common_size_t<R1, P1, R2, P2>;
  return C(C(lhs).count() + C(rhs).count());
}

template <class R1, class P1, class R2, class P2>
constexpr auto operator-(const basic_size<R1, P1>& lhs, const basic_size<R2, P2>& rhs)
{
  using C = common_size_t<R1, P1, R2, P2>;
  return C(C(lhs).count() - C(rhs).count());
}

template <class R1, class P1, class R2, class P2>
constexpr auto operator<=>(const basic_size<R1, P1>& lhs, const basic_size<R2, P2>& rhs)
{
  using C = common_size_t<R1, P1, R2, P2>;
  return C(lhs).count() <=> C(rhs).count();
}

template <class R1, class P1, class R2, class P2>
constexpr bool operator==(const basic_size<R1, P1>& lhs, const basic_size<R2, P2>& rhs)
{
  using C = common_size_t<R1, P1, R2, P2>;
  return C(lhs).count() == C(rhs).count();
}

/// Convert to @p To, truncating toward zero like std::chrono::duration_cast.
template <class To, class R, class P>
  requires detail::is_basic_size<To>::value
constexpr To size_cast(const basic_size<R, P>& size)
{
  using F  = std::ratio_divide<P, typename To::period>;
  using CT = std::common_type_t<typename To::rep, R, std::intmax_t>;
  return To(static_cast<typename To::rep>(static_cast<CT>(size.count()) * F::num / F::den));
}

using bytes     = basic_size<std::int64_t>;
using kilobytes = basic_size<std::int64_t, std::kilo>;
using megabytes = basic_size<std::int64_t, std::mega>;
using gigabytes = basic_size<std::int64_t, std::giga>;
using terabytes = basic_size<std::int64_t, std::tera>;
using kibibytes = basic_size<std::int64_t, std::ratio<1LL << 10>>;
using mebibytes = basic_size<std::int64_t, std::ratio<1LL << 20>>;
using gibibytes = basic_size<std::int64_t, std::ratio<1LL << 30>>;
using tebibytes = basic_size<std::int64_t, std::ratio<1LL << 40>>;

namespace size_literals {
// clang-format off
constexpr bytes     operator""_B(unsigned long long v)   { return bytes(static_cast<std::int64_t>(v)); }
constexpr kilobytes operator""_KB(unsigned long long v)  { return kilobytes(static_cast<std::int64_t>(v)); }
constexpr megabytes operator""_MB(unsigned long long v)  { return megabytes(static_cast<std::int64_t>(v)); }
constexpr gigabytes operator""_GB(unsigned long long v)  { return gigabytes(static_cast<std::int64_t>(v)); }
constexpr terabytes operator""_TB(unsigned long long v)  { return terabytes(static_cast<std::int64_t>(v)); }
constexpr kibibytes operator""_KiB(unsigned long long v) { return kibibytes(static_cast<std::int64_t>(v)); }
constexpr mebibytes operator""_MiB(unsigned long long v) { return mebibytes(static_cast<std::int64_t>(v)); }
constexpr gibibytes operator""_GiB(unsigned long long v) { return gibibytes(static_cast<std::int64_t>(v)); }
constexpr tebibytes operator""_TiB(unsigned long long v) { return tebibytes(static_cast<std::int64_t>(v)); }
// clang-format on
}  // namespace size_literals

/// "1.5 GiB", "512M", "1048576" (bare: ticks of the target unit). Exact: a value that is not a
/// whole number of ticks is rejected, as is a negative one.
template <class Rep, class Period>
struct parser<basic_size<Rep, Period>> {
  parse_status operator()(std::string_view text, basic_size<Rep, Period>& out) const
  {
    Rep count{};
    auto status = quantity::parse<Rep, Period>(text, count, quantity::size_kind);
    if (status) { out = basic_size<Rep, Period>(count); }
    return status;
  }
};

/// The largest unit that represents the value exactly: 1610612736 B -> "1536 MiB".
template <class Rep, class Period>
struct formatter<basic_size<Rep, Period>> {
  std::string operator()(const basic_size<Rep, Period>& value) const
  {
    return quantity::format<Rep, Period>(value.count(), quantity::size_kind);
  }
};

}  // namespace sirius::config
