// Copyright 2026, Sirius Contributors. SPDX-License-Identifier: Apache-2.0
#pragma once

#include "config/node.hpp"
#include "config/option.hpp"

#include <concepts>
#include <cstddef>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

namespace sirius::config {

/**
 * @brief A configuration block as file adapters see it: read from a node, written to a node.
 *
 * This is the type-erased surface. A file format (YAML today) converts its document to and from
 * node trees and calls these, without knowing any block's options or their C++ types.
 */
class config_block {
 public:
  config_block()                               = default;
  config_block(const config_block&)            = default;
  config_block(config_block&&)                 = default;
  config_block& operator=(const config_block&) = default;
  config_block& operator=(config_block&&)      = default;
  virtual ~config_block()                      = default;

  /**
   * @brief Replace every value of this block with what @p n says. @p n is the block's mapping, or
   * null when the block is absent (every option then takes its fallback).
   * @param path the block's dotted path, used in messages ("sirius.executor.downgrade").
   * @throws config_error for a bad value, a missing required key, mutually exclusive keys, a
   * removed or unknown key, or a failed cross-field rule.
   */
  virtual void read(const node& n, std::string_view path, environment env) = 0;

  /// The mapping read() reads back to the same values: file values only, plus fallbacks when
  /// @p include_fallbacks. Absent nested blocks and empty lists are omitted.
  [[nodiscard]] virtual node write(bool include_fallbacks) const = 0;

  /// Whether the user defined this block: the last read() found it in the configuration (an
  /// empty mapping counts).
  [[nodiscard]] bool user_defined() const noexcept { return _user_defined; }

 protected:
  bool _user_defined = false;
};

// ---- field descriptors a block lists in fields() ------------------------------------------

/// A nested block under @p key.
template <class Block>
struct nested_block {
  std::string_view key;
  Block* block;
};

/// A sequence of blocks under @p key (e.g. sirius.space.gpu).
template <class Block>
struct block_list {
  std::string_view key;
  std::vector<Block>* blocks;
};

/// A key that is no longer accepted: its presence (even with a null value) fails with @p reason.
struct removed_key {
  std::string_view key;
  std::string_view reason;
};

template <class Block>
  requires std::derived_from<Block, config_block>
[[nodiscard]] nested_block<Block> sub(std::string_view key, Block& block) noexcept
{
  return {key, &block};
}

template <class Block>
  requires std::derived_from<Block, config_block>
[[nodiscard]] block_list<Block> list(std::string_view key, std::vector<Block>& blocks) noexcept
{
  return {key, &blocks};
}

[[nodiscard]] constexpr removed_key removed(std::string_view key, std::string_view reason) noexcept
{
  return {key, reason};
}

/// The tuple a block's fields() returns: options by reference, descriptors by value.
///
///     auto fields() { return field_list(num_threads, cpu_affinity, removed("strategy", "...")); }
template <class... Fields>
[[nodiscard]] std::tuple<Fields...> field_list(Fields&&... fields)
{
  return std::tuple<Fields...>(std::forward<Fields>(fields)...);
}

namespace detail {

template <class>
struct is_config_option : std::false_type {};
template <class... Ts>
struct is_config_option<config_option<Ts...>> : std::true_type {};

template <class B>
concept has_validate = requires(const B& block, std::string_view path) { block.validate(path); };

}  // namespace detail

/**
 * @brief Implements config_block for @p Derived from its field list.
 *
 * @p Derived provides `auto fields()` returning field_list(...) of its config_option members,
 * sub(key, block) for nested blocks, list(key, blocks) for sequences of blocks and
 * removed(key, reason) for keys that are no longer accepted. It may provide
 * `void validate(std::string_view path) const`, run after every field is read, for rules that span
 * several fields; it throws config_error.
 *
 * Reading checks removed keys first, then reads the fields in order, then rejects keys that no
 * field claims.
 */
template <class Derived>
class block_base : public config_block {
 public:
  void read(const node& n, std::string_view path, environment env) override
  {
    if (!n.is_null() && !n.is_mapping()) {
      throw config_error("'" + std::string(path) + "': expected a mapping, got " +
                         std::string(n.kind_name()));
    }
    _user_defined = !n.is_null();
    auto fields   = self().fields();

    // Removed keys first: they explain themselves better than whatever else is wrong.
    std::apply([&](auto&... field) { (check_removed(field, n, path), ...); }, fields);
    std::apply([&](auto&... field) { (read_field(field, n, path, env), ...); }, fields);

    if (n.is_mapping()) {
      for (auto const& [key, value] : n.entries()) {
        bool claimed = false;
        std::apply([&](auto&... field) { ((claimed = claimed || claims(field, key)), ...); },
                   fields);
        if (!claimed) {
          throw config_error("unknown config key: '" + key + "'" +
                             (path.empty() ? std::string() : " in " + std::string(path)));
        }
      }
    }
    if constexpr (detail::has_validate<Derived>) { std::as_const(self()).validate(path); }
  }

  [[nodiscard]] node write(bool include_fallbacks) const override
  {
    node out = node::mapping();
    // fields() only ties references; nothing is modified through them here.
    auto fields = const_cast<Derived&>(static_cast<const Derived&>(*this)).fields();
    std::apply([&](auto&... field) { (write_field(field, out, include_fallbacks), ...); }, fields);
    return out;
  }

 private:
  Derived& self() { return static_cast<Derived&>(*this); }

  // -- options

  template <class... Ts>
  static void check_removed(config_option<Ts...>&, const node&, std::string_view)
  {
  }
  template <class... Ts>
  static void read_field(config_option<Ts...>& option,
                         const node& n,
                         std::string_view path,
                         environment env)
  {
    option.load(n, path, env);
  }
  template <class... Ts>
  static bool claims(const config_option<Ts...>& option, std::string_view key)
  {
    for (auto const k : option.keys()) {
      if (k == key) { return true; }
    }
    return false;
  }
  template <class... Ts>
  static void write_field(const config_option<Ts...>& option, node& out, bool include_fallbacks)
  {
    option.write(out, include_fallbacks);
  }

  // -- nested blocks

  template <class B>
  static void check_removed(const nested_block<B>&, const node&, std::string_view)
  {
  }
  template <class B>
  static void read_field(const nested_block<B>& field,
                         const node& n,
                         std::string_view path,
                         environment env)
  {
    auto const* child = n.find(field.key);
    field.block->read(child != nullptr ? *child : node{}, join_path(path, field.key), env);
  }
  template <class B>
  static bool claims(const nested_block<B>& field, std::string_view key)
  {
    return field.key == key;
  }
  template <class B>
  static void write_field(const nested_block<B>& field, node& out, bool include_fallbacks)
  {
    auto child = field.block->write(include_fallbacks);
    if (!child.entries().empty() || field.block->user_defined()) {
      out.set(std::string(field.key), std::move(child));
    }
  }

  // -- lists of blocks

  template <class B>
  static void check_removed(const block_list<B>&, const node&, std::string_view)
  {
  }
  template <class B>
  static void read_field(const block_list<B>& field,
                         const node& n,
                         std::string_view path,
                         environment env)
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
        blocks.emplace_back().read(
          child->elements()[i], list_path + "[" + std::to_string(i) + "]", env);
      }
    }
    *field.blocks = std::move(blocks);
  }
  template <class B>
  static bool claims(const block_list<B>& field, std::string_view key)
  {
    return field.key == key;
  }
  template <class B>
  static void write_field(const block_list<B>& field, node& out, bool include_fallbacks)
  {
    if (field.blocks->empty()) { return; }
    node::sequence_type elements;
    elements.reserve(field.blocks->size());
    for (auto const& block : *field.blocks) {
      elements.push_back(block.write(include_fallbacks));
    }
    out.set(std::string(field.key), node::sequence(std::move(elements)));
  }

  // -- removed keys

  static void check_removed(const removed_key& field, const node& n, std::string_view path)
  {
    if (n.find(field.key) != nullptr) {
      throw config_error("'" + join_path(path, field.key) + "': " + std::string(field.reason));
    }
  }
  static void read_field(const removed_key&, const node&, std::string_view, environment) {}
  static bool claims(const removed_key& field, std::string_view key) { return field.key == key; }
  static void write_field(const removed_key&, node&, bool) {}
};

}  // namespace sirius::config
