// Copyright 2026, Sirius Contributors. SPDX-License-Identifier: Apache-2.0
#pragma once

#include "config/node.hpp"
#include "config/option.hpp"

#include <cstddef>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

namespace sirius::config {

template <class Derived>
class block_base;

/// What a configuration dump contains.
struct write_options {
  /// Also the fallbacks of options the user left alone, not only user-defined values.
  bool include_fallbacks = false;
  /// Also developer-only options (see config_option::is_developer()).
  bool include_developer = true;

  write_options() = default;
  /// The common case, `write(true)`: every value, developer ones included.
  write_options(bool fallbacks)
    : include_fallbacks(fallbacks) {}  // NOLINT(google-explicit-constructor)
  write_options(bool fallbacks, bool developer)
    : include_fallbacks(fallbacks), include_developer(developer)
  {
  }
};

/**
 * @brief A configuration block as file adapters see it: read from a node, written to a node.
 *
 * This is the type-erased surface. A file format (YAML today) converts its document to and from
 * node trees and calls these, without knowing any block's options or their C++ types. Copy and
 * assign the concrete block types; the base cannot be copied or assigned on its own (that would
 * slice).
 */
class config_block {
 public:
  virtual ~config_block() = default;

  /**
   * @brief Replace every value of this block with what @p n says. @p n is the block's mapping, or
   * null when the block is absent: every option then takes its fallback, and options declared
   * mandatory are not required (they are required when their block is present).
   *
   * Strong guarantee: when it throws, the block is unchanged.
   *
   * @param path the block's dotted path, used in messages ("sirius.executor.downgrade").
   * @throws config_error for a bad value, a missing required key, mutually exclusive keys, a
   * removed or unknown key, or a failed cross-field rule.
   * @throws std::logic_error for a declaration bug (an empty or repeated key).
   */
  virtual void read(const node& n, std::string_view path) = 0;

  /// The mapping read() reads back to the same values: user-defined values only, plus fallbacks
  /// when options.include_fallbacks, without developer-only options unless
  /// options.include_developer, in declaration order. Absent nested blocks and empty lists are
  /// omitted; a nested block the user defined is kept even when empty. Without developer-only
  /// options, a block or list left with nothing to write is omitted, as is a list with any such
  /// element. A dump with fallbacks is a snapshot to look at: reading it back makes every value,
  /// and every block it names, user-defined.
  [[nodiscard]] virtual node write(const write_options& options) const = 0;

  /// Throw std::logic_error for a declaration bug in this block or any block nested in it: an
  /// empty key, or a key claimed twice (by options, nested blocks, lists or removed keys).
  /// read() checks the same; this lets a unit test check every block without a file.
  virtual void check_declaration() const = 0;

  /// Whether the user defined this block: the last read() found it in the configuration (an
  /// empty mapping counts; a null value does not).
  [[nodiscard]] bool user_defined() const noexcept { return _user_defined; }

  /// Whether the user defined this block or any value in it, read or assigned: user_defined(),
  /// a user-defined option, a nested block with user values, or a non-empty list.
  [[nodiscard]] virtual bool has_user_values() const = 0;

 protected:
  config_block()                                   = default;
  config_block(const config_block&)                = default;
  config_block(config_block&&) noexcept            = default;
  config_block& operator=(const config_block&)     = default;
  config_block& operator=(config_block&&) noexcept = default;

 private:
  template <class>
  friend class block_base;

  bool _user_defined = false;
};

namespace detail {

template <class>
struct is_config_option : std::false_type {};
template <class... Ts>
struct is_config_option<config_option<Ts...>> : std::true_type {};

}  // namespace detail

/**
 * @brief Implements config_block for @p Derived from its field list.
 *
 *     struct pool_options : block_base<pool_options> {
 *       config_option<int> num_threads = option<int>("num_threads").fallback(4).build();
 *       nested_options nested;
 *       std::vector<item_options> items;
 *       auto fields()
 *       {
 *         return field_list(num_threads, sub("nested", nested), list("items", items),
 *                           removed("thread_name_prefix", "removed; names are internal"));
 *       }
 *       void validate(std::string_view path) const;  // optional
 *     };
 *
 * - fields() returns field_list(...) of the config_option members (by reference), sub(key, block)
 *   for nested blocks, list(key, blocks) for sequences of blocks and removed(key, reason) for keys
 *   no longer accepted. It is called on every read and write (also through a const block), so it
 *   must only tie members: cheap, and modifying nothing.
 * - validate(path) const, when declared, checks rules that span fields. It runs after every
 *   field is read, also for an absent block, so it must not assume an option has a value (use
 *   get_if() / has_value()). It throws config_error. A declaration with another signature does
 *   not compile.
 * - Reading a level checks removed keys first, reads the fields in order (nested blocks report
 *   their own errors first), rejects keys no field claims, then validates.
 * - An empty list reads and writes like an absent one; a null list element reads as an element
 *   with every fallback.
 * - A block holds no values until it is read: read a null node to get the fallbacks.
 * - Derived must name itself: `struct x : block_base<x>`. Options a class derived from Derived
 *   adds are not part of the block unless its own fields() lists them.
 */
template <class Derived>
class block_base : public config_block {
 public:
  void read(const node& n, std::string_view path) override
  {
    // Read into a copy and swap it in, so a failure leaves this block as it was.
    Derived next(self());
    next.read_in_place(n, path);
    self() = std::move(next);
  }

  [[nodiscard]] node write(const write_options& options) const override
  {
    node out = node::mapping();
    // fields() only ties references (see the class comment): nothing is modified through them.
    auto fields = const_cast<Derived&>(static_cast<const Derived&>(*this)).fields();
    std::apply([&](auto&... field) { (write_field(field, out, options), ...); }, fields);
    return out;
  }

  [[nodiscard]] bool has_user_values() const override
  {
    if (_user_defined) { return true; }
    auto fields = const_cast<Derived&>(static_cast<const Derived&>(*this)).fields();
    return std::apply([](const auto&... field) { return (holds_user_values(field) || ...); },
                      fields);
  }

  void check_declaration() const override
  {
    auto fields = const_cast<Derived&>(static_cast<const Derived&>(*this)).fields();
    (void)claimed_keys(fields);
    std::apply([](auto&... field) { (check_nested(field), ...); }, fields);
  }

  /// No cross-field rules; a Derived declares its own `void validate(std::string_view) const`.
  void validate(std::string_view /*path*/) const {}

 protected:
  /// A nested block under @p key.
  template <class Block>
  struct nested_block {
    std::string key;
    Block* block;
  };

  /// A sequence of blocks under @p key (e.g. sirius.space.gpu).
  template <class Block>
  struct block_list {
    std::string key;
    std::vector<Block>* blocks;
  };

  /// A key no longer accepted: its presence, even with a null value, fails with @p reason.
  struct removed_key {
    std::string key;
    std::string reason;
  };

  template <class Block>
    requires std::derived_from<Block, config_block>
  [[nodiscard]] static nested_block<Block> sub(std::string key, Block& block)
  {
    return {std::move(key), &block};
  }

  template <class Block>
    requires std::derived_from<Block, config_block>
  [[nodiscard]] static block_list<Block> list(std::string key, std::vector<Block>& blocks)
  {
    return {std::move(key), &blocks};
  }

  [[nodiscard]] static removed_key removed(std::string key, std::string reason)
  {
    return {std::move(key), std::move(reason)};
  }

  /// The tuple fields() returns: options by reference, descriptors by value.
  template <class... Fields>
  [[nodiscard]] static std::tuple<Fields...> field_list(Fields&&... fields)
  {
    (check_field_kind<Fields>(), ...);
    return std::tuple<Fields...>(std::forward<Fields>(fields)...);
  }

 private:
  friend Derived;

  block_base()                                 = default;
  block_base(const block_base&)                = default;
  block_base(block_base&&) noexcept            = default;
  block_base& operator=(const block_base&)     = default;
  block_base& operator=(block_base&&) noexcept = default;

  Derived& self() { return static_cast<Derived&>(*this); }

  template <class T>
  static constexpr bool is_nested = false;
  template <class B>
  static constexpr bool is_nested<nested_block<B>> = true;
  template <class T>
  static constexpr bool is_list = false;
  template <class B>
  static constexpr bool is_list<block_list<B>> = true;

  /// Options must be the block's own members (a non-const lvalue), descriptors made in place.
  template <class F>
  static constexpr void check_field_kind()
  {
    using plain = std::remove_cvref_t<F>;
    if constexpr (detail::is_config_option<plain>::value) {
      static_assert(std::is_lvalue_reference_v<F> && !std::is_const_v<std::remove_reference_t<F>>,
                    "field_list: pass a config_option member itself, not a copy, a temporary or "
                    "a const reference");
    } else {
      static_assert(!std::is_reference_v<F>,
                    "field_list: pass sub(), list() and removed() results directly");
      static_assert(std::is_same_v<plain, removed_key> || is_nested<plain> || is_list<plain>,
                    "field_list: fields are config_option members, sub(), list() or removed()");
    }
  }

  void read_in_place(const node& n, std::string_view path)
  {
    if (!n.is_null() && !n.is_mapping()) {
      throw config_error(
        detail::framed(path, "expected a mapping, got " + std::string(n.kind_name())));
    }
    auto fields       = self().fields();
    auto const claims = claimed_keys(fields);

    // Removed keys first: they explain themselves better than whatever else is wrong.
    std::apply([&](auto&... field) { (check_removed(field, n, path), ...); }, fields);
    std::apply([&](auto&... field) { (read_field(field, n, path), ...); }, fields);
    if (n.is_mapping()) {
      for (auto const& entry : n.entries()) {
        if (!contains(claims, entry.first)) {
          throw config_error("unknown config key: '" + entry.first + "'" +
                             (path.empty() ? std::string() : " in " + std::string(path)));
        }
      }
    }
    std::as_const(self()).validate(path);
    _user_defined = !n.is_null();
  }

  /// Every key a field claims, checked for declaration bugs.
  template <class Tuple>
  static std::vector<std::string_view> claimed_keys(const Tuple& fields)
  {
    std::vector<std::string_view> keys;
    std::apply([&](const auto&... field) { (add_keys(field, keys), ...); }, fields);
    for (std::size_t i = 0; i < keys.size(); ++i) {
      if (keys[i].empty()) { throw std::logic_error("config block: empty key"); }
      for (std::size_t j = 0; j < i; ++j) {
        if (keys[i] == keys[j]) {
          throw std::logic_error("config block: key '" + std::string(keys[i]) +
                                 "' is claimed by two fields");
        }
      }
    }
    return keys;
  }

  [[nodiscard]] static bool contains(const std::vector<std::string_view>& keys,
                                     std::string_view key)
  {
    for (auto const k : keys) {
      if (k == key) { return true; }
    }
    return false;
  }

  // -- options

  template <class... Ts>
  static void add_keys(const config_option<Ts...>& option, std::vector<std::string_view>& keys)
  {
    for (auto const k : option.keys()) {
      keys.push_back(k);
    }
  }
  template <class... Ts>
  static void check_removed(const config_option<Ts...>&, const node&, std::string_view)
  {
  }
  template <class... Ts>
  static void read_field(config_option<Ts...>& option, const node& n, std::string_view path)
  {
    option.load(n, path);
  }
  template <class... Ts>
  static void write_field(const config_option<Ts...>& option,
                          node& out,
                          const write_options& options)
  {
    if (option.is_developer() && !options.include_developer) { return; }
    option.write(out, options.include_fallbacks);
  }
  template <class... Ts>
  static void check_nested(const config_option<Ts...>&)
  {
  }
  template <class... Ts>
  static bool holds_user_values(const config_option<Ts...>& option)
  {
    return option.user_defined();
  }

  // -- nested blocks

  template <class B>
  static void add_keys(const nested_block<B>& field, std::vector<std::string_view>& keys)
  {
    keys.push_back(field.key);
  }
  template <class B>
  static void check_removed(const nested_block<B>&, const node&, std::string_view)
  {
  }
  template <class B>
  static void read_field(const nested_block<B>& field, const node& n, std::string_view path)
  {
    static const node absent;
    auto const* child = n.find(field.key);
    field.block->read(child != nullptr ? *child : absent, join_path(path, field.key));
  }
  template <class B>
  static void write_field(const nested_block<B>& field, node& out, const write_options& options)
  {
    auto child = field.block->write(options);
    if (!child.entries().empty() || (field.block->user_defined() && options.include_developer)) {
      out.set(field.key, std::move(child));
    }
  }
  template <class B>
  static void check_nested(const nested_block<B>& field)
  {
    field.block->check_declaration();
  }
  template <class B>
  static bool holds_user_values(const nested_block<B>& field)
  {
    return field.block->has_user_values();
  }

  // -- lists of blocks

  template <class B>
  static void add_keys(const block_list<B>& field, std::vector<std::string_view>& keys)
  {
    keys.push_back(field.key);
  }
  template <class B>
  static void check_removed(const block_list<B>&, const node&, std::string_view)
  {
  }
  template <class B>
  static void read_field(const block_list<B>& field, const node& n, std::string_view path)
  {
    auto const list_path = join_path(path, field.key);
    std::vector<B> blocks;
    auto const* child = n.find(field.key);
    if (child != nullptr && !child->is_null()) {
      if (!child->is_sequence()) {
        throw config_error("'" + list_path + "': expected a sequence, got " +
                           std::string(child->kind_name()));
      }
      blocks.reserve(child->elements().size());
      for (std::size_t i = 0; i < child->elements().size(); ++i) {
        blocks.emplace_back().read(child->elements()[i], list_path + "[" + std::to_string(i) + "]");
      }
    }
    *field.blocks = std::move(blocks);
  }
  template <class B>
  static void write_field(const block_list<B>& field, node& out, const write_options& options)
  {
    if (field.blocks->empty()) { return; }
    node::sequence_type elements;
    elements.reserve(field.blocks->size());
    for (auto const& block : *field.blocks) {
      elements.push_back(block.write(options));
      // An element without its developer-only settings would read back as another one.
      if (elements.back().entries().empty() && !options.include_developer) { return; }
    }
    out.set(field.key, node::sequence(std::move(elements)));
  }
  template <class B>
  static void check_nested(const block_list<B>&)
  {
    B{}.check_declaration();
  }
  template <class B>
  static bool holds_user_values(const block_list<B>& field)
  {
    return !field.blocks->empty();
  }

  // -- removed keys

  static void add_keys(const removed_key& field, std::vector<std::string_view>& keys)
  {
    keys.push_back(field.key);
  }
  static void check_removed(const removed_key& field, const node& n, std::string_view path)
  {
    if (n.find(field.key) != nullptr) {
      throw config_error("'" + join_path(path, field.key) + "': " + field.reason);
    }
  }
  static void read_field(const removed_key&, const node&, std::string_view) {}
  static void write_field(const removed_key&, node&, const write_options&) {}
  static void check_nested(const removed_key&) {}
  static bool holds_user_values(const removed_key&) { return false; }
};

}  // namespace sirius::config
