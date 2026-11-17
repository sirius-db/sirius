// Copyright 2026, Sirius Contributors. SPDX-License-Identifier: Apache-2.0
#include "config/yaml.hpp"

#include <yaml-cpp/yaml.h>

#include <string>
#include <string_view>
#include <unordered_set>
#include <utility>

namespace sirius::config {

namespace {

node from_yaml(const YAML::Node& yaml, const std::string& path)
{
  switch (yaml.Type()) {
    case YAML::NodeType::Undefined:
    case YAML::NodeType::Null: return node{};
    case YAML::NodeType::Scalar: return node::scalar(yaml.Scalar());
    case YAML::NodeType::Sequence: {
      node::sequence_type elements;
      elements.reserve(yaml.size());
      std::size_t index = 0;
      for (auto const& element : yaml) {
        elements.push_back(from_yaml(element, path + "[" + std::to_string(index++) + "]"));
      }
      return node::sequence(std::move(elements));
    }
    case YAML::NodeType::Map: {
      node::mapping_type entries;
      entries.reserve(yaml.size());
      std::unordered_set<std::string> seen;
      for (auto const& entry : yaml) {
        if (!entry.first.IsScalar()) {
          throw config_error("'" + path + "': mapping keys must be scalars");
        }
        auto key = entry.first.Scalar();
        if (!seen.insert(key).second) {
          throw config_error("duplicate config key: '" + key + "'" +
                             (path.empty() ? std::string() : " in " + path));
        }
        auto value = from_yaml(entry.second, join_path(path, key));
        entries.emplace_back(std::move(key), std::move(value));
      }
      return node::mapping(std::move(entries));
    }
  }
  return node{};
}

void emit(YAML::Emitter& out, const node& n)
{
  switch (n.type()) {
    case node::kind::null: out << YAML::Null; break;
    case node::kind::scalar: out << n.text(); break;
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
        out << YAML::Key << key << YAML::Value;
        emit(out, value);
      }
      out << YAML::EndMap;
      break;
  }
}

}  // namespace

node parse_yaml(std::string_view text)
{
  YAML::Node document;
  try {
    document = YAML::Load(std::string(text));
  } catch (const YAML::Exception& e) {
    throw yaml_syntax_error(e.what());
  }
  return from_yaml(document, "");
}

std::string emit_yaml(const node& n)
{
  YAML::Emitter out;
  emit(out, n);
  if (!out.good()) { throw std::logic_error("emit_yaml: " + out.GetLastError()); }
  return std::string(out.c_str()) + "\n";
}

void read_yaml(config_block& block,
               std::string_view text,
               std::string_view root_key,
               environment env)
{
  auto const document = parse_yaml(text);
  if (!document.is_mapping()) {
    throw config_error("expected a mapping at the top level, got " +
                       std::string(document.kind_name()));
  }
  for (auto const& [key, value] : document.entries()) {
    if (key != root_key) { throw config_error("unknown config key: '" + key + "'"); }
  }
  auto const* root = document.find(root_key);
  if (root == nullptr) {
    throw config_error("missing top-level '" + std::string(root_key) + "' key");
  }
  block.read(*root, root_key, env);
}

std::string write_yaml(const config_block& block, std::string_view root_key, bool include_fallbacks)
{
  node document = node::mapping();
  document.set(std::string(root_key), block.write(include_fallbacks));
  return emit_yaml(document);
}

}  // namespace sirius::config
