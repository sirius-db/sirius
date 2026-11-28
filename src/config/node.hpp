// Copyright 2026, Sirius Contributors. SPDX-License-Identifier: Apache-2.0
#pragma once

#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

namespace sirius::config {

/**
 * @brief A format-neutral configuration value: null, a scalar, a sequence or a mapping.
 *
 * A file adapter (YAML today) turns a document into a node tree for the config blocks to read,
 * and turns the tree the blocks write back into a document. Scalars keep their source text;
 * converting text to C++ values is the options' job (parser<T>), so the adapter knows nothing
 * about option types. Mappings keep their entries in document order.
 */
class node {
 public:
  enum class kind { null, scalar, sequence, mapping };
  using sequence_type = std::vector<node>;
  using mapping_type  = std::vector<std::pair<std::string, node>>;

  /// A null node (an absent value, or YAML's `~` / empty value).
  node() = default;

  [[nodiscard]] static node scalar(std::string text)
  {
    node n;
    n._value = std::move(text);
    return n;
  }
  [[nodiscard]] static node sequence(sequence_type elements = {})
  {
    node n;
    n._value = std::move(elements);
    return n;
  }
  [[nodiscard]] static node mapping(mapping_type entries = {})
  {
    node n;
    n._value = std::move(entries);
    return n;
  }

  [[nodiscard]] kind type() const noexcept { return static_cast<kind>(_value.index()); }
  [[nodiscard]] bool is_null() const noexcept { return type() == kind::null; }
  [[nodiscard]] bool is_scalar() const noexcept { return type() == kind::scalar; }
  [[nodiscard]] bool is_sequence() const noexcept { return type() == kind::sequence; }
  [[nodiscard]] bool is_mapping() const noexcept { return type() == kind::mapping; }

  /// The scalar's text. @throws std::logic_error unless this is a scalar.
  [[nodiscard]] const std::string& text() const { return get<std::string>("a scalar"); }
  [[nodiscard]] const sequence_type& elements() const { return get<sequence_type>("a sequence"); }
  [[nodiscard]] sequence_type& elements() { return get<sequence_type>("a sequence"); }
  [[nodiscard]] const mapping_type& entries() const { return get<mapping_type>("a mapping"); }
  [[nodiscard]] mapping_type& entries() { return get<mapping_type>("a mapping"); }

  /// The value under @p key, or nullptr when this is not a mapping or has no such key.
  [[nodiscard]] const node* find(std::string_view key) const noexcept
  {
    auto const* entries = std::get_if<mapping_type>(&_value);
    if (entries == nullptr) { return nullptr; }
    for (auto const& [name, value] : *entries) {
      if (name == key) { return &value; }
    }
    return nullptr;
  }

  /// Whether @p key is present with a non-null value; a null value counts as absent.
  [[nodiscard]] bool has_value(std::string_view key) const noexcept
  {
    auto const* value = find(key);
    return value != nullptr && !value->is_null();
  }

  /// Add @p key to a mapping (a null node becomes an empty mapping first), replacing an existing
  /// entry in place. References and pointers into this mapping (e.g. from find()) are invalidated.
  /// @throws std::logic_error when this is a scalar or a sequence.
  void set(std::string key, node value)
  {
    if (is_null()) { _value = mapping_type{}; }
    auto& map = entries();
    for (auto& [name, existing] : map) {
      if (name == key) {
        existing = std::move(value);
        return;
      }
    }
    map.emplace_back(std::move(key), std::move(value));
  }

  /// Remove @p key from a mapping; a no-op when absent or when this is not a mapping.
  void erase(std::string_view key)
  {
    auto* map = std::get_if<mapping_type>(&_value);
    if (map == nullptr) { return; }
    std::erase_if(*map, [key](const auto& entry) { return entry.first == key; });
  }

  /// "null", "a scalar", "a sequence" or "a mapping", for messages.
  [[nodiscard]] static std::string_view kind_name(kind k) noexcept
  {
    switch (k) {
      case kind::null: return "null";
      case kind::scalar: return "a scalar";
      case kind::sequence: return "a sequence";
      case kind::mapping: return "a mapping";
    }
    return "unknown";
  }
  [[nodiscard]] std::string_view kind_name() const noexcept { return kind_name(type()); }

  friend bool operator==(const node&, const node&) = default;

 private:
  template <class T>
  [[nodiscard]] const T& get(std::string_view expected) const
  {
    if (auto const* value = std::get_if<T>(&_value)) { return *value; }
    throw std::logic_error("config::node: expected " + std::string(expected) + ", got " +
                           std::string(kind_name()));
  }
  template <class T>
  [[nodiscard]] T& get(std::string_view expected)
  {
    return const_cast<T&>(std::as_const(*this).get<T>(expected));
  }

  // Alternative order matches kind.
  std::variant<std::monostate, std::string, sequence_type, mapping_type> _value;
};

}  // namespace sirius::config
