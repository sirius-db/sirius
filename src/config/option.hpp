// Copyright 2026, Sirius Contributors. SPDX-License-Identifier: Apache-2.0
#pragma once

#include "config/chrono.hpp"
#include "config/node.hpp"
#include "config/parser.hpp"
#include "config/size.hpp"

#include <concepts>
#include <cstddef>
#include <functional>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

namespace sirius::config {

/// Where an option's current value came from.
enum class value_source {
  unset,     ///< not loaded, absent without a fallback, or skipped (for_prod outside prod)
  file,      ///< read from the configuration (or assigned as if it were)
  fallback,  ///< absent, so the declared fallback applies
};

/// The environment a configuration is loaded for; for_prod() options are skipped outside prod.
enum class environment { dev, prod };

/// When a setting may change: once per session (the engine reads it at startup), or per query
/// (a DuckDB SET can override it for a connection; the file value is the starting point).
enum class option_scope { session, query };

[[nodiscard]] constexpr std::string_view to_string(value_source source) noexcept
{
  switch (source) {
    case value_source::unset: return "unset";
    case value_source::file: return "file";
    case value_source::fallback: return "fallback";
  }
  return "unknown";
}

[[nodiscard]] constexpr std::string_view to_string(option_scope scope) noexcept
{
  switch (scope) {
    case option_scope::session: return "session";
    case option_scope::query: return "query";
  }
  return "unknown";
}

/// A configuration the user supplied cannot be used. The message names the dotted key:
/// `'<dotted.path>': <reason>`.
class config_error : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

[[nodiscard]] inline std::string join_path(std::string_view path, std::string_view key)
{
  if (path.empty()) { return std::string(key); }
  return std::string(path) + "." + std::string(key);
}

// ---- reading and writing a value of T from / to a node --------------------------------------

namespace detail {
template <class>
struct is_vector : std::false_type {};
template <class E, class A>
struct is_vector<std::vector<E, A>> : std::true_type {};
}  // namespace detail

/// A scalar type parser<T> handles, or a std::vector of one (read from a sequence).
template <class T>
concept config_value =
  parsable<T> || (detail::is_vector<T>::value && parsable<typename T::value_type>);

/// Read @p n into @p out: a scalar through parser<T>, a std::vector element-wise from a sequence.
/// Leaves @p out untouched on failure.
template <config_value T>
parse_status read_value(const node& n, T& out)
{
  if constexpr (detail::is_vector<T>::value) {
    if (!n.is_sequence()) {
      return parse_status::failure("expected a sequence, got " + std::string(n.kind_name()));
    }
    T result;
    result.reserve(n.elements().size());
    for (std::size_t i = 0; i < n.elements().size(); ++i) {
      auto const& element = n.elements()[i];
      typename T::value_type value{};
      auto status =
        element.is_scalar()
          ? parser<typename T::value_type>{}(element.text(), value)
          : parse_status::failure("expected a scalar, got " + std::string(element.kind_name()));
      if (!status) {
        return parse_status::failure("element " + std::to_string(i) + ": " + status.error());
      }
      result.push_back(std::move(value));
    }
    out = std::move(result);
    return {};
  } else {
    if (!n.is_scalar()) {
      return parse_status::failure("expected a scalar, got " + std::string(n.kind_name()));
    }
    return parser<T>{}(n.text(), out);
  }
}

/// The node read_value<T> reads back to @p value.
template <config_value T>
node write_value(const T& value)
{
  if constexpr (detail::is_vector<T>::value) {
    node::sequence_type elements;
    elements.reserve(value.size());
    for (auto const& element : value) {
      elements.push_back(node::scalar(formatter<typename T::value_type>{}(element)));
    }
    return node::sequence(std::move(elements));
  } else {
    return node::scalar(formatter<T>{}(value));
  }
}

// ---- validators ---------------------------------------------------------------------------

/// Accepts low <= value <= high.
template <class T>
struct between {
  T low, high;
  [[nodiscard]] bool operator()(const T& value) const { return low <= value && value <= high; }
  [[nodiscard]] std::string description() const
  {
    return "must be between " + formatter<T>{}(low) + " and " + formatter<T>{}(high);
  }
};

namespace detail {

template <class T, class... Ts>
struct first_convertible {
  using type = void;
};
template <class T, class U, class... Rest>
struct first_convertible<T, U, Rest...> {
  using type = std::
    conditional_t<std::is_convertible_v<T, U>, U, typename first_convertible<T, Rest...>::type>;
};

/// A value of T initializes exactly one alternative: its own type, else the only one T
/// implicitly converts to (1_MiB -> bytes, 42 -> std::size_t).
template <class T, class... Ts>
concept selects_one_of = (std::is_same_v<std::decay_t<T>, Ts> || ...) ||
                         ((static_cast<std::size_t>(std::is_convertible_v<T, Ts>) + ... + 0) == 1);

template <class T, class... Ts>
using selected_alternative_t = std::conditional_t<(std::is_same_v<std::decay_t<T>, Ts> || ...),
                                                  std::decay_t<T>,
                                                  typename first_convertible<T, Ts...>::type>;

template <class F>
concept described = requires(const F& f) {
  { f.description() } -> std::convertible_to<std::string>;
};

/// Wrap a validator into the stored form. A bool validator rejects with "value out of range",
/// plus its description() when it has one; a parse_status validator words its own rejection.
template <class T, class F>
std::function<parse_status(const T&)> make_validator(F fn, std::string description)
{
  if constexpr (std::is_invocable_r_v<parse_status, const F&, const T&> &&
                std::same_as<std::invoke_result_t<const F&, const T&>, parse_status>) {
    return [fn = std::move(fn)](const T& value) { return std::invoke(fn, value); };
  } else {
    static_assert(std::is_invocable_r_v<bool, const F&, const T&>,
                  "a validator takes const T& and returns bool or parse_status");
    if (description.empty()) {
      if constexpr (described<F>) { description = std::string(fn.description()); }
    }
    return [fn = std::move(fn), description = std::move(description)](const T& value) {
      if (std::invoke(fn, value)) { return parse_status{}; }
      std::string text = "value out of range";
      if constexpr (config_value<T> && !is_vector<T>::value) {
        text += ": " + config::detail::quoted(formatter<T>{}(value));
      }
      if (!description.empty()) { text += " (" + description + ")"; }
      return parse_status::failure(std::move(text));
    };
  }
}

}  // namespace detail

// ---- options ------------------------------------------------------------------------------

/// One spelling of an option: its key and how its value is read, validated and written.
template <class T>
struct option_alternative {
  using value_type = T;

  std::string key;
  std::function<parse_status(const node&, T&)> parser = [](const node& n, T& out) {
    return read_value(n, out);
  };
  std::function<node(const T&)> writer = [](const T& value) { return write_value(value); };
  std::function<parse_status(const T&)> validator;  ///< empty: every value is accepted
};

/**
 * @brief A typed configuration option: one value read from one of several mutually exclusive
 * keys (its alternatives, e.g. `capacity_bytes` as bytes or `capacity_fraction` as double).
 *
 * Loading rules:
 * - at most one alternative may be present; a present null counts as absent;
 * - a present value that does not parse or validate is an error, never a fallback;
 * - a required option must be present: a fallback does not satisfy it;
 * - absent and optional: the fallback (value_source::fallback) or nothing (value_source::unset);
 * - for_prod() options are left unset outside environment::prod;
 * - a failed load leaves the option unset.
 *
 * Declare options with option<T>(key)...build().
 */
template <class... Ts>
struct config_option {
  static_assert(sizeof...(Ts) >= 1, "an option has at least one alternative");

  using value_type = std::variant<Ts...>;
  using first_type = std::tuple_element_t<0, std::tuple<Ts...>>;

  std::tuple<option_alternative<Ts>...> alternatives;
  std::optional<value_type> fallback;
  std::optional<value_type> value;
  bool required      = false;
  bool prod_only     = false;
  option_scope scope = option_scope::session;
  std::string description;

  [[nodiscard]] value_source source() const noexcept { return _source; }
  [[nodiscard]] bool is_from_file() const noexcept { return _source == value_source::file; }
  [[nodiscard]] bool is_fallback() const noexcept { return _source == value_source::fallback; }
  [[nodiscard]] bool is_unset() const noexcept { return _source == value_source::unset; }
  [[nodiscard]] bool has_value() const noexcept { return value.has_value(); }

  /// The first alternative's key, which names the option.
  [[nodiscard]] const std::string& key() const { return std::get<0>(alternatives).key; }

  /// Every alternative's key, in declaration order.
  [[nodiscard]] std::vector<std::string_view> keys() const
  {
    return std::apply([](const auto&... alt) { return std::vector<std::string_view>{alt.key...}; },
                      alternatives);
  }

  /// The value when it holds alternative @p T, else nullptr.
  template <class T = first_type>
  [[nodiscard]] const T* get_if() const noexcept
  {
    return value ? std::get_if<T>(&*value) : nullptr;
  }

  /// The value as alternative @p T. @throws std::logic_error when unset or holding another one.
  template <class T = first_type>
  [[nodiscard]] const T& get() const
  {
    if (auto const* v = get_if<T>()) { return *v; }
    throw std::logic_error("config option '" + key() +
                           "': " + (value ? "holds another alternative" : "has no value"));
  }

  /// Set the value as if it had been read from the configuration.
  template <class T>
    requires detail::selects_one_of<T, Ts...>
  void assign(T&& v)
  {
    value.emplace(std::in_place_type<detail::selected_alternative_t<T, Ts...>>, std::forward<T>(v));
    _source = value_source::file;
  }

  /// Forget the value (value_source::unset).
  void reset() noexcept
  {
    value.reset();
    _source = value_source::unset;
  }

  /**
   * @brief Load from the mapping @p block, whose dotted path is @p path ("" at the root).
   * @throws config_error naming the offending key.
   */
  void load(const node& block, std::string_view path, environment env)
  {
    reset();
    if (prod_only && env != environment::prod) { return; }

    std::size_t matches = 0;
    std::string present;
    std::apply(
      [&](const auto&... alt) {
        auto count = [&](const auto& a) {
          if (!block.has_value(a.key)) { return; }
          present += (matches++ == 0 ? "'" : " and '") + a.key + "'";
        };
        (count(alt), ...);
      },
      alternatives);
    if (matches > 1) {
      throw config_error("'" + std::string(path) + "': " + present + " are mutually exclusive");
    }

    std::optional<value_type> candidate;
    std::apply(
      [&](const auto&... alt) {
        auto read = [&](const auto& a) {
          auto const* n = block.find(a.key);
          if (n == nullptr || n->is_null()) { return; }
          using T = typename std::decay_t<decltype(a)>::value_type;
          T parsed{};
          auto status = a.parser(*n, parsed);
          if (status && a.validator) { status = a.validator(parsed); }
          if (!status) {
            throw config_error("'" + join_path(path, a.key) + "': " + status.error());
          }
          candidate.emplace(std::in_place_type<T>, std::move(parsed));
        };
        (read(alt), ...);
      },
      alternatives);

    if (matches == 0 && required) {
      throw config_error("'" + join_path(path, key()) + "': required but missing");
    }
    if (candidate) {
      value   = std::move(candidate);
      _source = value_source::file;
    } else if (fallback) {
      value   = fallback;
      _source = value_source::fallback;
    }
  }

  /// Add this option to the mapping @p block: the key of the alternative it holds and its value.
  /// Writes a fallback only when @p include_fallbacks; writes nothing when unset.
  void write(node& block, bool include_fallbacks) const
  {
    if (!value || (_source == value_source::fallback && !include_fallbacks)) { return; }
    std::apply(
      [&](const auto&... alt) {
        auto emit = [&](const auto& a) {
          using T = typename std::decay_t<decltype(a)>::value_type;
          if (auto const* v = std::get_if<T>(&*value)) { block.set(a.key, a.writer(*v)); }
        };
        (emit(alt), ...);
      },
      alternatives);
  }

 private:
  value_source _source = value_source::unset;
};

/**
 * @brief Fluent declaration of a config_option. validate() and parser() apply to the most
 * recently added alternative; the rest applies to the whole option.
 *
 *     config_option<bytes, double> capacity =
 *       option<bytes>("capacity_bytes")
 *         .or_type<double>("capacity_fraction")
 *         .validate(between<double>{0.0, 1.0})
 *         .fallback(0.9)
 *         .build();
 */
template <class... Ts>
class option_builder {
 public:
  using option_type = config_option<Ts...>;

  explicit option_builder(option_type spec) : _spec(std::move(spec)) {}

  /// Validate the last alternative with a callable taking const T& and returning bool (rejected
  /// as "value out of range", plus the callable's description() when it has one) or parse_status.
  template <class F>
  option_builder& validate(F fn, std::string description = {})
  {
    auto& alt     = last();
    using T       = typename std::decay_t<decltype(alt)>::value_type;
    alt.validator = detail::make_validator<T>(std::move(fn), std::move(description));
    return *this;
  }

  /// Replace the last alternative's parser. Accepts
  /// `bool|parse_status (std::string_view, T&)` for a scalar or `parse_status (const node&, T&)`.
  template <class F>
  option_builder& parser(F fn)
  {
    auto& alt = last();
    using T   = typename std::decay_t<decltype(alt)>::value_type;
    if constexpr (std::is_invocable_v<const F&, const node&, T&>) {
      alt.parser = std::move(fn);
    } else {
      static_assert(std::is_invocable_v<const F&, std::string_view, T&>,
                    "a parser takes (std::string_view, T&) or (const node&, T&)");
      alt.parser = [fn = std::move(fn)](const node& n, T& out) -> parse_status {
        if (!n.is_scalar()) {
          return parse_status::failure("expected a scalar, got " + std::string(n.kind_name()));
        }
        if constexpr (std::same_as<std::invoke_result_t<const F&, std::string_view, T&>,
                                   parse_status>) {
          return std::invoke(fn, std::string_view(n.text()), out);
        } else {
          if (std::invoke(fn, std::string_view(n.text()), out)) { return {}; }
          return parse_status::failure("cannot parse " + detail::quoted(n.text()));
        }
      };
    }
    return *this;
  }

  /// Replace the last alternative's writer: `std::string (const T&)` for a scalar, or
  /// `node (const T&)`. Pair it with parser() when the custom spelling differs from formatter<T>.
  template <class F>
  option_builder& writer(F fn)
  {
    auto& alt = last();
    using T   = typename std::decay_t<decltype(alt)>::value_type;
    if constexpr (std::same_as<std::invoke_result_t<const F&, const T&>, node>) {
      alt.writer = std::move(fn);
    } else {
      alt.writer = [fn = std::move(fn)](const T& value) {
        return node::scalar(std::string(std::invoke(fn, value)));
      };
    }
    return *this;
  }

  /// Add a mutually exclusive alternative: the same setting spelled as another key and type.
  template <class U>
  [[nodiscard]] option_builder<Ts..., U> or_type(std::string key) const
  {
    static_assert((!std::is_same_v<U, Ts> && ...),
                  "each alternative of an option needs a distinct value type");
    option_alternative<U> added;
    added.key = std::move(key);
    config_option<Ts..., U> next;
    next.alternatives = std::tuple_cat(_spec.alternatives, std::make_tuple(std::move(added)));
    if (_spec.fallback) {
      next.fallback =
        std::visit([](const auto& v) { return typename config_option<Ts..., U>::value_type(v); },
                   *_spec.fallback);
    }
    next.required    = _spec.required;
    next.prod_only   = _spec.prod_only;
    next.scope       = _spec.scope;
    next.description = _spec.description;
    return option_builder<Ts..., U>(std::move(next));
  }

  /// The value used when no alternative is present (for an optional option). It initializes its
  /// own alternative type, else the only one it converts to.
  template <class T>
    requires detail::selects_one_of<T, Ts...>
  option_builder& fallback(T&& v)
  {
    _spec.fallback.emplace(std::in_place_type<detail::selected_alternative_t<T, Ts...>>,
                           std::forward<T>(v));
    return *this;
  }

  option_builder& as_mandatory()
  {
    _spec.required = true;
    return *this;
  }
  option_builder& as_optional()
  {
    _spec.required = false;
    return *this;
  }
  option_builder& for_prod()
  {
    _spec.prod_only = true;
    return *this;
  }
  option_builder& scope(option_scope s)
  {
    _spec.scope = s;
    return *this;
  }
  option_builder& describe(std::string text)
  {
    _spec.description = std::move(text);
    return *this;
  }

  /// The declared option. @throws std::logic_error when the fallback fails its alternative's
  /// validator (a declaration bug).
  [[nodiscard]] option_type build() const
  {
    if (_spec.fallback) {
      std::apply(
        [&](const auto&... alt) {
          auto check = [&](const auto& a) {
            using T       = typename std::decay_t<decltype(a)>::value_type;
            auto const* v = std::get_if<T>(&*_spec.fallback);
            if (v == nullptr || !a.validator) { return; }
            if (auto status = a.validator(*v); !status) {
              throw std::logic_error("config option '" + a.key +
                                     "': invalid fallback: " + status.error());
            }
          };
          (check(alt), ...);
        },
        _spec.alternatives);
    }
    return _spec;
  }

 private:
  auto& last() { return std::get<sizeof...(Ts) - 1>(_spec.alternatives); }

  option_type _spec;
};

/// Start declaring an option read from @p key as a T.
template <class T>
[[nodiscard]] option_builder<T> option(std::string key)
{
  config_option<T> spec;
  std::get<0>(spec.alternatives).key = std::move(key);
  return option_builder<T>(std::move(spec));
}

}  // namespace sirius::config
