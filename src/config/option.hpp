// Copyright 2026, Sirius Contributors. SPDX-License-Identifier: Apache-2.0
#pragma once

#include "config/chrono.hpp"  // IWYU pragma: keep (parser<> / formatter<> for durations)
#include "config/node.hpp"
#include "config/parser.hpp"
#include "config/size.hpp"  // IWYU pragma: keep (parser<> / formatter<> for sizes)

#include <concepts>
#include <cstddef>
#include <functional>
#include <initializer_list>
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
  unset,     ///< not loaded, or absent without a fallback
  user,      ///< defined by the user: read from the configuration, or assigned
  fallback,  ///< absent, so the declared fallback applies
};

[[nodiscard]] constexpr std::string_view to_string(value_source source) noexcept
{
  switch (source) {
    case value_source::unset: return "unset";
    case value_source::user: return "user";
    case value_source::fallback: return "fallback";
  }
  return "unknown";
}

/// A configuration the user supplied cannot be used. The message names the dotted key and, when
/// the option has a description, what it is for: `'<dotted.path>': <reason>[; <description>]`.
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

template <class F>
concept described = requires(const F& f) {
  { f.description() } -> std::convertible_to<std::string>;
};

/// A validator wording its own rejection.
template <class F, class T>
concept status_validator = requires(const F& f, const T& value) {
  { std::invoke(f, value) } -> std::convertible_to<parse_status>;
};

/// A validator answering yes or no (exactly bool: a type that merely converts to bool, such as
/// std::error_code, may mean the opposite).
template <class F, class T>
concept bool_validator = requires(const F& f, const T& value) {
  { std::invoke(f, value) } -> std::same_as<bool>;
};

template <class T, class F>
std::function<parse_status(const T&)> make_validator(F fn, std::string description)
{
  static_assert(status_validator<F, T> || bool_validator<F, T>,
                "a validator is callable as const F& with const T& and returns bool or "
                "parse_status");
  if constexpr (!bool_validator<F, T>) {
    // Rejected with its own reason, plus the description when one was given.
    return [fn = std::move(fn), description = std::move(description)](const T& value) {
      parse_status status = std::invoke(fn, value);
      if (status || description.empty()) { return status; }
      return parse_status::failure(status.error() + " (" + description + ")");
    };
  } else {
    // Rejected as "value out of range: '<value>' (<description>)"; an explicit description wins
    // over the callable's description().
    if (description.empty()) {
      if constexpr (described<F>) { description = std::string(fn.description()); }
    }
    return [fn = std::move(fn), description = std::move(description)](const T& value) {
      if (std::invoke(fn, value)) { return parse_status{}; }
      std::string text = "value out of range";
      if constexpr (parsable<T>) { text += ": " + config::detail::quoted(formatter<T>{}(value)); }
      if (!description.empty()) { text += " (" + description + ")"; }
      return parse_status::failure(std::move(text));
    };
  }
}

/// A text parser: called with the scalar's text as written (const std::string&, so callables
/// taking std::string_view work too).
template <class F, class T>
concept text_status_parser = requires(const F& f, const std::string& text, T& out) {
  { std::invoke(f, text, out) } -> std::convertible_to<parse_status>;
};
template <class F, class T>
concept text_bool_parser = requires(const F& f, const std::string& text, T& out) {
  { std::invoke(f, text, out) } -> std::same_as<bool>;
};
template <class F, class T>
concept node_parser = requires(const F& f, const node& n, T& out) {
  { std::invoke(f, n, out) } -> std::convertible_to<parse_status>;
};

template <class F, class T>
concept node_writer = requires(const F& f, const T& value) {
  { std::invoke(f, value) } -> std::convertible_to<node>;
};
template <class F, class T>
concept text_writer = requires(const F& f, const T& value) {
  { std::invoke(f, value) } -> std::convertible_to<std::string>;
};

/// From converts to To without losing information or meaning: the same type; a class type
/// conversion the class allows implicitly (mebibytes -> bytes, a string literal -> std::string);
/// or an arithmetic widening that keeps the kind (integral or floating) and, for integers, the
/// signedness. Nothing but bool converts to bool, and bool converts to nothing else.
template <class From, class To>
concept lossless_to =
  std::same_as<std::decay_t<From>, To> ||
  ((std::is_class_v<std::decay_t<From>> || std::is_class_v<To>) &&
   std::is_convertible_v<From, To>) ||
  (std::is_arithmetic_v<std::decay_t<From>> && std::is_arithmetic_v<To> &&
   !std::same_as<std::decay_t<From>, bool> && !std::same_as<To, bool> &&
   ((std::is_integral_v<std::decay_t<From>> && std::is_integral_v<To> &&
     std::is_signed_v<std::decay_t<From>> == std::is_signed_v<To>) ||
    (std::is_floating_point_v<std::decay_t<From>> && std::is_floating_point_v<To>)) &&
   sizeof(To) >= sizeof(std::decay_t<From>));

template <class T, class... Ts>
struct first_lossless {
  using type = void;
};
template <class T, class U, class... Rest>
struct first_lossless<T, U, Rest...> {
  using type = std::conditional_t<lossless_to<T, U>, U, typename first_lossless<T, Rest...>::type>;
};

/// A value of T initializes exactly one alternative: its own type, else the only one it converts
/// to losslessly (1_MiB -> bytes, 1.0f -> double; not 1 -> double, -1 -> std::size_t).
template <class T, class... Ts>
concept selects_one_of = (std::same_as<std::decay_t<T>, Ts> || ...) ||
                         ((static_cast<std::size_t>(lossless_to<T, Ts>) + ... + 0) == 1);

template <class T, class... Ts>
using selected_alternative_t = std::conditional_t<(std::same_as<std::decay_t<T>, Ts> || ...),
                                                  std::decay_t<T>,
                                                  typename first_lossless<T, Ts...>::type>;

/// `'<path>': <message>`, or the bare message at the root.
[[nodiscard]] inline std::string framed(std::string_view path, const std::string& message)
{
  return path.empty() ? message : "'" + std::string(path) + "': " + message;
}

}  // namespace detail

// ---- options ------------------------------------------------------------------------------

template <class... Ts>
class option_builder;

/// One spelling of an option: its key and how its value is read, validated and written. A type
/// without parser<T> / formatter<T> starts with neither and must be given both.
template <class T>
struct option_alternative {
  using value_type = T;

  std::string key;
  std::function<parse_status(const node&, T&)> parser = default_parser();
  std::function<node(const T&)> writer                = default_writer();
  std::function<parse_status(const T&)> validator;  ///< empty: every value is accepted

 private:
  static std::function<parse_status(const node&, T&)> default_parser()
  {
    if constexpr (config_value<T>) {
      return [](const node& n, T& out) { return read_value(n, out); };
    } else {
      return {};
    }
  }
  static std::function<node(const T&)> default_writer()
  {
    if constexpr (config_value<T>) {
      return [](const T& value) { return write_value(value); };
    } else {
      return {};
    }
  }
};

/**
 * @brief A typed configuration option: one value read from one of several mutually exclusive
 * keys (its alternatives, e.g. `capacity_bytes` as bytes or `capacity_fraction` as double).
 *
 * Loading rules:
 * - at most one alternative may be present; a present null counts as absent;
 * - a present value that does not parse or validate is an error, never a fallback;
 * - a required option must be present when its block is (a null block is an absent one, and
 *   then nothing is required); a fallback does not satisfy it;
 * - absent and optional: the fallback (value_source::fallback) or nothing (value_source::unset);
 * - a failed load leaves the option unset.
 *
 * Declare options with option<T>(key)...build(). Not synchronized: load, assign and reset must
 * not race with readers.
 */
template <class... Ts>
class config_option {
 public:
  static_assert(sizeof...(Ts) >= 1, "an option has at least one alternative");

  using value_type = std::variant<Ts...>;
  using first_type = std::tuple_element_t<0, std::tuple<Ts...>>;

  [[nodiscard]] value_source source() const noexcept { return _source; }
  [[nodiscard]] bool user_defined() const noexcept { return _source == value_source::user; }
  [[nodiscard]] bool is_fallback() const noexcept { return _source == value_source::fallback; }
  [[nodiscard]] bool is_unset() const noexcept { return _source == value_source::unset; }
  [[nodiscard]] bool has_value() const noexcept { return _value.has_value(); }

  /// The current value (user-defined or the fallback), if any.
  [[nodiscard]] const std::optional<value_type>& value() const noexcept { return _value; }
  /// The declared fallback, if any.
  [[nodiscard]] const std::optional<value_type>& fallback() const noexcept { return _fallback; }
  [[nodiscard]] bool is_required() const noexcept { return _required; }
  /// A developer-only option, as every option not marked production is: a configuration dump
  /// outside developer mode leaves it out.
  [[nodiscard]] bool is_developer() const noexcept { return _developer; }
  [[nodiscard]] const std::string& description() const noexcept { return _description; }

  /// The first alternative's key, which names the option.
  [[nodiscard]] const std::string& key() const { return std::get<0>(_alternatives).key; }

  /// Every alternative's key, in declaration order (views into this option).
  [[nodiscard]] std::vector<std::string_view> keys() const
  {
    return std::apply([](const auto&... alt) { return std::vector<std::string_view>{alt.key...}; },
                      _alternatives);
  }

  /// The value when it holds alternative @p T, else nullptr.
  template <class T = first_type>
  [[nodiscard]] const T* get_if() const noexcept
  {
    return _value ? std::get_if<T>(&*_value) : nullptr;
  }

  /// The value as alternative @p T. @throws std::logic_error when unset or holding another one.
  template <class T = first_type>
  [[nodiscard]] const T& get() const
  {
    if (auto const* v = get_if<T>()) { return *v; }
    throw std::logic_error("config option '" + key() +
                           "': " + (_value ? "holds another alternative" : "has no value"));
  }

  /**
   * @brief Set the value as if it had been read from the configuration: @p v initializes its own
   * alternative, or the only one it converts to losslessly, and must pass that alternative's
   * validator.
   * @throws config_error (`'<key>': <reason>`) when the validator rejects it; the option is then
   * unchanged.
   */
  template <class T>
    requires detail::selects_one_of<T, Ts...>
  void assign(T&& v)
  {
    using A = detail::selected_alternative_t<T, Ts...>;
    A candidate(std::forward<T>(v));
    auto const& alt = std::get<option_alternative<A>>(_alternatives);
    if (alt.validator) {
      if (auto status = alt.validator(candidate); !status) {
        fail("'" + alt.key + "': " + status.error());
      }
    }
    reset();  // a throwing emplace then leaves the option unset, not half-assigned
    _value.emplace(std::in_place_type<A>, std::move(candidate));
    _source = value_source::user;
  }

  /// Forget the value (value_source::unset).
  void reset() noexcept
  {
    _value.reset();
    _source = value_source::unset;
  }

  /**
   * @brief Load from the mapping @p block, whose dotted path is @p path ("" at the root); a null
   * @p block is an absent one.
   * @throws config_error naming the offending key; an exception from a custom parser or
   * validator is reported the same way. std::bad_alloc passes through.
   */
  void load(const node& block, std::string_view path)
  {
    reset();

    std::vector<std::string_view> present;
    std::apply(
      [&](const auto&... alt) {
        ((block.has_value(alt.key) ? present.push_back(alt.key) : void()), ...);
      },
      _alternatives);
    if (present.size() > 1) {
      fail(detail::framed(path, quote_list(present, " and ") + " are mutually exclusive"));
    }
    if (present.empty() && _required && !block.is_null()) {
      if constexpr (sizeof...(Ts) == 1) {
        fail("'" + join_path(path, key()) + "': required but missing");
      } else {
        fail(detail::framed(path, "one of " + quote_list(keys(), " or ") + " is required"));
      }
    }

    std::optional<value_type> candidate;
    std::apply([&](const auto&... alt) { (read(alt, block, path, candidate), ...); },
               _alternatives);
    if (candidate) {
      _value  = std::move(candidate);
      _source = value_source::user;
    } else if (_fallback) {
      _value  = _fallback;
      _source = value_source::fallback;
    }
  }

  /// Put this option into the mapping @p block: the key of the alternative it holds, with its
  /// value, replacing any alternative already there. A fallback is written only when
  /// @p include_fallbacks; an unset option writes nothing (and removes nothing).
  void write(node& block, bool include_fallbacks) const
  {
    if (!_value || (_source == value_source::fallback && !include_fallbacks)) { return; }
    std::apply(
      [&](const auto&... alt) {
        (block.erase(alt.key), ...);
        auto emit = [&](const auto& a) {
          using T = typename std::decay_t<decltype(a)>::value_type;
          if (auto const* v = std::get_if<T>(&*_value)) { block.set(a.key, a.writer(*v)); }
        };
        (emit(alt), ...);
      },
      _alternatives);
  }

 private:
  template <class...>
  friend class option_builder;

  static std::string quote_list(const std::vector<std::string_view>& keys, std::string_view sep)
  {
    std::string text;
    for (std::size_t i = 0; i < keys.size(); ++i) {
      if (i > 0) { text += i + 1 == keys.size() ? std::string(sep) : ", "; }
      text += "'" + std::string(keys[i]) + "'";
    }
    return text;
  }

  [[noreturn]] void fail(std::string message) const
  {
    if (!_description.empty()) { message += "; " + _description; }
    throw config_error(message);
  }

  template <class T>
  void read(const option_alternative<T>& alt,
            const node& block,
            std::string_view path,
            std::optional<value_type>& candidate) const
  {
    auto const* n = block.find(alt.key);
    if (n == nullptr || n->is_null()) { return; }
    T parsed{};
    parse_status status;
    try {
      status = alt.parser(*n, parsed);
      if (status && alt.validator) { status = alt.validator(parsed); }
    } catch (const std::bad_alloc&) {
      throw;
    } catch (const std::exception& e) {
      status = parse_status::failure(e.what());
    }
    if (!status) { fail("'" + join_path(path, alt.key) + "': " + status.error()); }
    candidate.emplace(std::in_place_type<T>, std::move(parsed));
  }

  std::tuple<option_alternative<Ts>...> _alternatives;
  std::optional<value_type> _fallback;
  std::optional<value_type> _value;
  bool _required  = false;
  bool _developer = true;
  std::string _description;
  value_source _source = value_source::unset;
};

/**
 * @brief Fluent declaration of a config_option. validate(), parser() and writer() apply to the
 * most recently added alternative; the rest applies to the whole option. Use it in one
 * expression ending in build(): the setters return a reference to the builder.
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

  /// An option read from @p key as the single alternative type; see option<T>().
  explicit option_builder(std::string key)
    requires(sizeof...(Ts) == 1)
  {
    std::get<0>(_spec._alternatives).key = std::move(key);
  }

  /// Validate the last alternative with a callable taking const T& and returning bool (rejected
  /// as "value out of range", with @p description or the callable's description()) or
  /// parse_status (its own reason, followed by @p description when given).
  template <class F>
  option_builder& validate(F fn, std::string description = {})
  {
    auto& alt     = last();
    using T       = typename std::decay_t<decltype(alt)>::value_type;
    alt.validator = detail::make_validator<T>(std::move(fn), std::move(description));
    return *this;
  }

  /// Replace the last alternative's parser with a callable taking (const node&, T&) and returning
  /// parse_status, or taking the scalar's text as written (std::string_view or
  /// const std::string&) and T&, and returning parse_status or bool. Leave @p out untouched on
  /// failure.
  template <class F>
  option_builder& parser(F fn)
  {
    auto& alt = last();
    using T   = typename std::decay_t<decltype(alt)>::value_type;
    if constexpr (detail::node_parser<F, T>) {
      alt.parser = [fn = std::move(fn)](const node& n, T& out) -> parse_status {
        return std::invoke(fn, n, out);
      };
    } else {
      static_assert(detail::text_status_parser<F, T> || detail::text_bool_parser<F, T>,
                    "a parser takes (const node&, T&) and returns parse_status, or takes the "
                    "text (std::string_view or const std::string&) and T& and returns "
                    "parse_status or bool");
      alt.parser = [fn = std::move(fn)](const node& n, T& out) -> parse_status {
        if (!n.is_scalar()) {
          return parse_status::failure("expected a scalar, got " + std::string(n.kind_name()));
        }
        if constexpr (detail::text_status_parser<F, T>) {
          return std::invoke(fn, n.text(), out);
        } else {
          if (std::invoke(fn, n.text(), out)) { return {}; }
          return parse_status::failure("cannot parse " + detail::quoted(n.text()));
        }
      };
    }
    return *this;
  }

  /// Replace the last alternative's writer with a callable taking const T& and returning a node,
  /// or the scalar's text. Pair it with parser() when the custom spelling differs from
  /// formatter<T>.
  template <class F>
  option_builder& writer(F fn)
  {
    auto& alt = last();
    using T   = typename std::decay_t<decltype(alt)>::value_type;
    if constexpr (detail::node_writer<F, T>) {
      alt.writer = [fn = std::move(fn)](const T& value) { return node(std::invoke(fn, value)); };
    } else {
      static_assert(detail::text_writer<F, T>,
                    "a writer takes const T& and returns a node or a string");
      alt.writer = [fn = std::move(fn)](const T& value) {
        return node::scalar(std::string(std::invoke(fn, value)));
      };
    }
    return *this;
  }

  /// Read and write the last alternative as one of @p table's spellings, for a type without
  /// string_to_enum / enum_to_string (e.g. a third-party enum, or std::optional of one). Matching
  /// is exact (case-sensitive, untrimmed); anything else is rejected with the list. The first
  /// spelling of a value is the one written; writing a value the table lacks throws
  /// std::logic_error.
  option_builder& choices(
    std::initializer_list<
      std::pair<std::string_view, std::tuple_element_t<sizeof...(Ts) - 1, std::tuple<Ts...>>>>
      table)
  {
    using T = std::tuple_element_t<sizeof...(Ts) - 1, std::tuple<Ts...>>;
    std::vector<std::pair<std::string, T>> entries;
    for (auto const& [spelling, value] : table) {
      entries.emplace_back(std::string(spelling), value);
    }
    auto& alt  = last();
    alt.parser = [entries](const node& n, T& out) -> parse_status {
      if (!n.is_scalar()) {
        return parse_status::failure("expected a scalar, got " + std::string(n.kind_name()));
      }
      std::string expected;
      for (auto const& [spelling, value] : entries) {
        if (n.text() == spelling) {
          out = value;
          return {};
        }
        expected += (expected.empty() ? "" : ", ") + spelling;
      }
      return parse_status::failure("invalid enum value " + detail::quoted(n.text()) +
                                   " (expected " + expected + ")");
    };
    alt.writer = [entries](const T& value) {
      for (auto const& [spelling, candidate] : entries) {
        if (candidate == value) { return node::scalar(spelling); }
      }
      throw std::logic_error("config option: value has no spelling among its choices");
    };
    return *this;
  }

  /// Add a mutually exclusive alternative: the same setting spelled as another key and type.
  /// Option-wide settings made so far carry over.
  template <class U>
  [[nodiscard]] option_builder<Ts..., U> or_type(std::string key) const&
  {
    return option_builder(*this).template or_type<U>(std::move(key));
  }
  template <class U>
  [[nodiscard]] option_builder<Ts..., U> or_type(std::string key) &&
  {
    static_assert((!std::is_same_v<U, Ts> && ...),
                  "each alternative of an option needs a distinct value type");
    option_alternative<U> added;
    added.key = std::move(key);
    config_option<Ts..., U> next;
    next._alternatives =
      std::tuple_cat(std::move(_spec._alternatives), std::make_tuple(std::move(added)));
    if (_spec._fallback) {
      next._fallback = std::visit(
        [](auto&& v) {
          using V = std::decay_t<decltype(v)>;
          return typename config_option<Ts..., U>::value_type(std::in_place_type<V>,
                                                              std::forward<decltype(v)>(v));
        },
        std::move(*_spec._fallback));
    }
    next._required    = _spec._required;
    next._developer   = _spec._developer;
    next._description = std::move(_spec._description);
    return option_builder<Ts..., U>(std::move(next));
  }

  /// The value used when no alternative is present (for an optional option). It initializes its
  /// own alternative type, else the only one it converts to losslessly.
  template <class T>
    requires detail::selects_one_of<T, Ts...>
  option_builder& fallback(T&& v)
  {
    _spec._fallback.emplace(std::in_place_type<detail::selected_alternative_t<T, Ts...>>,
                            std::forward<T>(v));
    return *this;
  }

  option_builder& mark_mandatory()
  {
    _spec._required = true;
    return *this;
  }
  option_builder& mark_optional()
  {
    _spec._required = false;
    return *this;
  }
  /// A production setting: written to every configuration dump. Options left unmarked are
  /// developer-only, kept out of a dump unless Sirius runs in developer mode.
  option_builder& mark_production()
  {
    _spec._developer = false;
    return *this;
  }
  /// What the option is for, in a few words; every error about the option ends with it.
  option_builder& describe(std::string text)
  {
    _spec._description = std::move(text);
    return *this;
  }

  /**
   * @brief The declared option.
   * @throws std::logic_error for a declaration bug: an empty or repeated key, an alternative
   * without a parser or writer (a type with no parser<T> / formatter<T> needs both), or a
   * fallback its alternative's validator rejects.
   */
  [[nodiscard]] option_type build() const& { return option_builder(*this).build(); }
  [[nodiscard]] option_type build() &&
  {
    auto const names = _spec.keys();
    for (std::size_t i = 0; i < names.size(); ++i) {
      if (names[i].empty()) { throw std::logic_error("config option: empty key"); }
      for (std::size_t j = 0; j < i; ++j) {
        if (names[i] == names[j]) {
          throw std::logic_error("config option '" + std::string(names[i]) + "': repeated key");
        }
      }
    }
    std::apply(
      [&](const auto&... alt) {
        auto check = [&](const auto& a) {
          if (!a.parser || !a.writer) {
            throw std::logic_error("config option '" + a.key +
                                   "': no parser or writer (give both for a type without "
                                   "parser<T> / formatter<T>)");
          }
          using T       = typename std::decay_t<decltype(a)>::value_type;
          auto const* v = _spec._fallback ? std::get_if<T>(&*_spec._fallback) : nullptr;
          if (v == nullptr || !a.validator) { return; }
          if (auto status = a.validator(*v); !status) {
            throw std::logic_error("config option '" + a.key +
                                   "': invalid fallback: " + status.error());
          }
        };
        (check(alt), ...);
      },
      _spec._alternatives);
    return std::move(_spec);
  }

 private:
  template <class...>
  friend class option_builder;

  explicit option_builder(option_type spec) : _spec(std::move(spec)) {}

  auto& last() { return std::get<sizeof...(Ts) - 1>(_spec._alternatives); }

  option_type _spec;
};

/// Start declaring an option read from @p key as a T.
template <class T>
[[nodiscard]] option_builder<T> option(std::string key)
{
  return option_builder<T>(std::move(key));
}

}  // namespace sirius::config
