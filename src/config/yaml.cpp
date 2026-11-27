// Copyright 2026, Sirius Contributors. SPDX-License-Identifier: Apache-2.0
#include "config/yaml.hpp"

#include "config/yaml_cpp.hpp"

#include <yaml-cpp/depthguard.h>
#include <yaml-cpp/yaml.h>

#include <cstddef>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_set>
#include <utility>
#include <vector>

namespace sirius::config {

namespace {

/// Deeper than any configuration, shallow enough for the stack; also stops an alias that refers
/// to the node containing it, which yaml-cpp loads as a cycle.
constexpr std::size_t max_depth = 64;

/// Far more values than any configuration has. Aliases are copied on conversion, so a small
/// document of nested aliases could otherwise expand without bound.
constexpr std::size_t max_values = 100'000;

/// Where @p mark is in the text; nothing for a node built in code, which has no position.
std::string at(const YAML::Mark& mark)
{
  if (mark.is_null()) { return {}; }
  return " (line " + std::to_string(mark.line + 1) + ", column " + std::to_string(mark.column + 1) +
         ")";
}

class converter {
 public:
  node convert(const YAML::Node& yaml, const std::string& path, std::size_t depth)
  {
    if (depth > max_depth) {
      throw yaml_syntax_error("YAML document nested more than " + std::to_string(max_depth) +
                              " levels deep, or an alias refers to the value containing it");
    }
    if (++_values > max_values) {
      throw yaml_syntax_error("YAML document has more than " + std::to_string(max_values) +
                              " values once its aliases are expanded");
    }
    switch (yaml.Type()) {
      case YAML::NodeType::Undefined:
      case YAML::NodeType::Null: return node{};
      case YAML::NodeType::Scalar: return node::scalar(yaml.Scalar());
      case YAML::NodeType::Sequence: {
        node::sequence_type elements;
        elements.reserve(yaml.size());
        std::size_t index = 0;
        for (auto const& element : yaml) {
          elements.push_back(
            convert(element, path + "[" + std::to_string(index++) + "]", depth + 1));
        }
        return node::sequence(std::move(elements));
      }
      case YAML::NodeType::Map: {
        node::mapping_type entries;
        entries.reserve(yaml.size());
        std::unordered_set<std::string> keys;
        keys.reserve(yaml.size());
        for (auto const& entry : yaml) {
          if (!entry.first.IsScalar()) {
            throw config_error(detail::framed(path, "mapping keys must be non-null scalars") +
                               at(entry.first.Mark()));
          }
          auto key = entry.first.Scalar();
          if (!keys.insert(key).second) {
            throw config_error("duplicate config key: '" + key + "'" +
                               (path.empty() ? std::string() : " in " + path) +
                               at(entry.first.Mark()));
          }
          auto value = convert(entry.second, join_path(path, key), depth + 1);
          entries.emplace_back(std::move(key), std::move(value));
        }
        return node::mapping(std::move(entries));
      }
    }
    return node{};
  }

 private:
  std::size_t _values = 0;
};

/// Text the emitter would write plain but a parser reads as something else: a document marker
/// at the start of a line.
bool needs_quotes(std::string_view text)
{
  return text.starts_with("---") || text.starts_with("...");
}

void emit_text(YAML::Emitter& out, const std::string& text)
{
  if (needs_quotes(text)) { out << YAML::DoubleQuoted; }
  out << text;
}

void emit(YAML::Emitter& out, const node& n)
{
  switch (n.type()) {
    case node::kind::null: out << YAML::Null; break;
    case node::kind::scalar: emit_text(out, n.text()); break;
    case node::kind::sequence:
      out << YAML::BeginSeq;
      for (auto const& element : n.elements()) {
        emit(out, element);
      }
      out << YAML::EndSeq;
      break;
    case node::kind::mapping:
      out << YAML::BeginMap;
      for (auto const& [key, value] : n.entries()) {
        out << YAML::Key;
        emit_text(out, key);
        out << YAML::Value;
        emit(out, value);
      }
      out << YAML::EndMap;
      break;
  }
}

}  // namespace

node parse_yaml(std::string_view text)
{
  try {
    auto const documents = YAML::LoadAll(std::string(text));
    // Trailing empty documents ("a: 1\n---\n") are harmless; a second real one is a mistake
    // that reading only the first would hide.
    std::size_t real = 0;
    for (auto const& document : documents) {
      if (!document.IsNull()) { ++real; }
    }
    if (real > 1) {
      throw yaml_syntax_error("expected one YAML document, got " + std::to_string(real));
    }
    if (documents.empty()) { return node{}; }
    return converter{}.convert(documents.front(), "", 0);
  } catch (const YAML::DeepRecursion&) {
    throw yaml_syntax_error("YAML document nested too deeply");
  } catch (const YAML::Exception& e) {
    throw yaml_syntax_error(e.what());
  }
}

node from_yaml_cpp(const YAML::Node& document)
{
  try {
    return converter{}.convert(document, "", 0);
  } catch (const YAML::Exception& e) {
    throw yaml_syntax_error(e.what());
  }
}

std::string emit_yaml(const node& n)
{
  YAML::Emitter out;
  emit(out, n);
  if (!out.good()) { throw std::logic_error("emit_yaml: " + out.GetLastError()); }
  return std::string(out.c_str(), out.size()) + "\n";
}

void read_yaml(config_block& block, std::string_view text, std::string_view root_key)
{
  read_yaml(block, parse_yaml(text), root_key);
}

void read_yaml(config_block& block, const node& document, std::string_view root_key)
{
  auto const missing = "missing top-level '" + std::string(root_key) + "' key";
  if (document.is_null()) { throw config_error(missing); }
  if (!document.is_mapping()) {
    throw config_error("expected a mapping at the top level, got " +
                       std::string(document.kind_name()));
  }
  for (auto const& [key, value] : document.entries()) {
    if (key != root_key) { throw config_error("unknown config key: '" + key + "'"); }
  }
  // A root key without a value is missing, as before: `sirius: {}` is the empty configuration.
  auto const* root = document.find(root_key);
  if (root == nullptr || root->is_null()) { throw config_error(missing); }
  block.read(*root, root_key);
}

std::string write_yaml(const config_block& block,
                       std::string_view root_key,
                       const write_options& options)
{
  node document = node::mapping();
  document.set(std::string(root_key), block.write(options));
  return emit_yaml(document);
}

}  // namespace sirius::config
