// Copyright 2026, Sirius Contributors. SPDX-License-Identifier: Apache-2.0
#include "c_api/error_internal.hpp"
#include "c_api/plan_internal.hpp"

#include <algorithm>
#include <array>
#include <set>
#include <stdexcept>
#include <string_view>
#include <unordered_map>

namespace {
using namespace sirius::c_api::plan;
void require(bool condition, const char* message)
{
  if (!condition) { throw std::invalid_argument(message); }
}
template <class F>
sirius_status checked(sirius_error** error, F&& operation) noexcept
{
  return sirius::c_api::invoke(error, [&] {
    try {
      operation();
      return sirius_status{SIRIUS_SUCCESS};
    } catch (const std::invalid_argument& e) {
      return sirius::c_api::fail(SIRIUS_INVALID_ARGUMENT, e.what(), error);
    }
  });
}
template <class T, class F>
sirius_status create(T** out, sirius_error** error, F&& operation) noexcept
{
  if (out) { *out = nullptr; }
  return checked(error, [&] {
    require(out != nullptr, "Output pointer is NULL");
    *out = operation();
  });
}
std::string bytes(const char* data, size_t size)
{
  require(data && size, "Expected nonempty bytes");
  std::string result(data, size);
  require(result.find('\0') == std::string::npos, "Embedded NUL is not allowed");
  return result;
}
bool has_uri_scheme(std::string_view path)
{
  auto alpha = [](char c) { return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z'); };
  auto colon = path.find(':');
  if (colon == std::string_view::npos || colon == 0 || !alpha(path.front())) { return false; }
  return std::all_of(path.begin() + 1, path.begin() + colon, [&](char c) {
    return alpha(c) || (c >= '0' && c <= '9') || c == '+' || c == '-' || c == '.';
  });
}
bool utf8(std::string_view text)
{
  for (size_t i = 0; i < text.size();) {
    auto first = static_cast<unsigned char>(text[i++]);
    if (first < 0x80) { continue; }
    unsigned count   = 0;
    uint32_t code    = 0;
    uint32_t minimum = 0;
    if (first >= 0xc2 && first <= 0xdf) {
      count   = 1;
      code    = first & 0x1f;
      minimum = 0x80;
    } else if (first >= 0xe0 && first <= 0xef) {
      count   = 2;
      code    = first & 0x0f;
      minimum = 0x800;
    } else if (first >= 0xf0 && first <= 0xf4) {
      count   = 3;
      code    = first & 0x07;
      minimum = 0x10000;
    } else {
      return false;
    }
    if (count > text.size() - i) { return false; }
    for (unsigned j = 0; j < count; ++j) {
      auto next = static_cast<unsigned char>(text[i++]);
      if ((next & 0xc0) != 0x80) { return false; }
      code = (code << 6) | (next & 0x3f);
    }
    if (code < minimum || code > 0x10ffff || (code >= 0xd800 && code <= 0xdfff)) { return false; }
  }
  return true;
}
std::string quoted(std::string_view value)
{
  constexpr char hex[] = "0123456789abcdef";
  std::string result   = "\"";
  for (unsigned char c : value) {
    if (c == '"' || c == '\\') {
      result += '\\';
      result += static_cast<char>(c);
    } else if (c < 0x20 || c >= 0x7f) {
      result += "\\x";
      result += hex[c >> 4];
      result += hex[c & 15];
    } else {
      result += static_cast<char>(c);
    }
  }
  return result + '"';
}
// Print shared expression nodes once; expanding a shared DAG can grow exponentially.
std::string expression_text(const sirius_expression& expression,
                            std::unordered_map<const sirius_expression*, size_t>& ids,
                            std::string& definitions)
{
  auto [position, inserted] = ids.emplace(&expression, ids.size());
  auto label                = "e" + std::to_string(position->second);
  if (!inserted) { return label; }
  std::string value;
  if (auto literal = std::get_if<int64_t>(&expression.value)) {
    value = std::to_string(*literal);
  } else if (auto literal = std::get_if<bool>(&expression.value)) {
    value = *literal ? "true" : "false";
  } else if (auto reference = std::get_if<column>(&expression.value)) {
    value = "$" + std::to_string(reference->index);
  } else {
    const auto& comparison = std::get<sirius::c_api::plan::comparison>(expression.value);
    constexpr std::array symbols{"=", "!=", "<", "<=", ">", ">="};
    auto left  = expression_text(*comparison.left, ids, definitions);
    auto right = expression_text(*comparison.right, ids, definitions);
    value      = left + " " + symbols[comparison.operation] + " " + right;
  }
  definitions += label + " = " + value + "\n";
  return label;
}
std::string describe(const sirius_plan_node& root)
{
  std::string result;
  std::string definitions;
  std::unordered_map<const sirius_expression*, size_t> expressions;
  auto* node         = &root;
  size_t indentation = 0;
  while (node) {
    result.append(indentation, ' ');
    if (auto scan = std::get_if<parquet_scan>(&node->value)) {
      result += "ParquetScan(path=" + quoted(scan->path) + ")\n";
      node = nullptr;
    } else {
      const auto& value = std::get<filter>(node->value);
      result += "Filter(" + expression_text(*value.predicate, expressions, definitions) + ")\n";
      node = value.input.get();
    }
    indentation += 2;
  }
  result += "Output: ";
  for (size_t i = 0; i < root.schema->fields.size(); ++i) {
    const auto& field = root.schema->fields[i];
    if (i) { result += ", "; }
    result += quoted(field.name) + ": " + (field.type == SIRIUS_TYPE_INT64 ? "Int64" : "Boolean") +
              (field.nullable ? "?" : "");
  }
  return definitions + result + '\n';
}
}  // namespace

sirius_status sirius_schema_create(const sirius_field* fields,
                                   size_t count,
                                   sirius_schema** out,
                                   sirius_error** error) noexcept
{
  return create(out, error, [&] {
    require(fields && count, "Expected a nonempty schema");
    std::vector<field> result;
    std::set<std::string> names;
    for (size_t i = 0; i < count; ++i) {
      auto name = bytes(fields[i].name, fields[i].name_size);
      require(utf8(name), "Field name is not UTF-8");
      require(names.insert(name).second, "Duplicate field name");
      require(fields[i].type == SIRIUS_TYPE_BOOLEAN || fields[i].type == SIRIUS_TYPE_INT64,
              "Unsupported field type");
      require(fields[i].nullable <= 1, "Nullability must be 0 or 1");
      result.push_back({std::move(name), fields[i].type, fields[i].nullable != 0});
    }
    return new sirius_schema(std::move(result));
  });
}
size_t sirius_schema_field_count(const sirius_schema* schema) noexcept
{
  return schema ? schema->fields.size() : 0;
}
sirius_status sirius_schema_field(const sirius_schema* schema,
                                  size_t index,
                                  sirius_field* out,
                                  sirius_error** error) noexcept
{
  if (out) { *out = {}; }
  return checked(error, [&] {
    require(schema && out, "Schema or output pointer is NULL");
    require(index < schema->fields.size(), "Field index is out of range");
    const auto& field = schema->fields[index];
    *out = {field.name.c_str(), field.name.size(), field.type, uint32_t{field.nullable}};
  });
}
sirius_status sirius_plan_scan_parquet(const char* path,
                                       size_t size,
                                       const sirius_schema* schema,
                                       sirius_plan_node** out,
                                       sirius_error** error) noexcept
{
  return create(out, error, [&] {
    require(schema != nullptr, "Schema is NULL");
    auto name = bytes(path, size);
    require(!has_uri_scheme(name), "Expected a local file path without a URI scheme");
    return new sirius_plan_node(acquire(schema), parquet_scan{std::move(name)}, 1);
  });
}
sirius_status sirius_expression_column(const sirius_plan_node* input,
                                       size_t index,
                                       sirius_expression** out,
                                       sirius_error** error) noexcept
{
  return create(out, error, [&] {
    require(input != nullptr, "Input node is NULL");
    require(index < input->schema->fields.size(), "Column index is out of range");
    const auto& field = input->schema->fields[index];
    return new sirius_expression(field.type, field.nullable, acquire(input), column{index}, 1);
  });
}
sirius_status sirius_expression_literal_int64(int64_t value,
                                              sirius_expression** out,
                                              sirius_error** error) noexcept
{
  return create(
    out, error, [&] { return new sirius_expression(SIRIUS_TYPE_INT64, false, {}, value, 1); });
}
sirius_status sirius_expression_literal_boolean(uint32_t value,
                                                sirius_expression** out,
                                                sirius_error** error) noexcept
{
  return create(out, error, [&] {
    require(value <= 1, "Boolean literal must be 0 or 1");
    return new sirius_expression(SIRIUS_TYPE_BOOLEAN, false, {}, value != 0, 1);
  });
}
sirius_status sirius_expression_compare(const sirius_expression* left,
                                        sirius_comparison operation,
                                        const sirius_expression* right,
                                        sirius_expression** out,
                                        sirius_error** error) noexcept
{
  return create(out, error, [&] {
    require(left && right, "Comparison operand is NULL");
    require(operation <= SIRIUS_COMPARISON_GREATER_EQUAL, "Unknown comparison");
    require(left->type == right->type, "Comparison types differ");
    require(left->type != SIRIUS_TYPE_BOOLEAN || operation <= SIRIUS_COMPARISON_NOT_EQUAL,
            "Boolean comparisons support only equality and inequality");
    require(!left->scope || !right->scope || left->scope.get() == right->scope.get(),
            "Comparison operands belong to different input scopes");
    auto depth = 1 + std::max(left->depth, right->depth);
    require(depth <= max_depth, "Expression depth exceeds 256");
    auto scope = acquire(left->scope ? left->scope.get() : right->scope.get());
    return new sirius_expression(SIRIUS_TYPE_BOOLEAN,
                                 left->nullable || right->nullable,
                                 std::move(scope),
                                 comparison{acquire(left), acquire(right), operation},
                                 depth);
  });
}
sirius_status sirius_plan_filter(const sirius_plan_node* input,
                                 const sirius_expression* predicate,
                                 sirius_plan_node** out,
                                 sirius_error** error) noexcept
{
  return create(out, error, [&] {
    require(input && predicate, "Filter input or predicate is NULL");
    require(predicate->type == SIRIUS_TYPE_BOOLEAN, "Filter predicate is not Boolean");
    require(!predicate->scope || predicate->scope.get() == input,
            "Filter predicate belongs to another input scope");
    require(input->depth < max_depth, "Plan node depth exceeds 256");
    return new sirius_plan_node(
      acquire(input->schema.get()), filter{acquire(input), acquire(predicate)}, input->depth + 1);
  });
}
sirius_data_type sirius_expression_type(const sirius_expression* value) noexcept
{
  return value ? value->type : 0;
}
uint32_t sirius_expression_nullable(const sirius_expression* value) noexcept
{
  return value && value->nullable;
}
sirius_plan_node_kind sirius_plan_node_get_kind(const sirius_plan_node* node) noexcept
{
  return node ? (std::holds_alternative<parquet_scan>(node->value) ? SIRIUS_NODE_PARQUET_SCAN
                                                                   : SIRIUS_NODE_FILTER)
              : 0;
}
sirius_plan_node* sirius_plan_node_input(const sirius_plan_node* node) noexcept
{
  if (node) {
    if (auto value = std::get_if<filter>(&node->value)) { return value->input.get(); }
  }
  return nullptr;
}
sirius_schema* sirius_plan_node_schema(const sirius_plan_node* node) noexcept
{
  return node ? node->schema.get() : nullptr;
}
sirius_status sirius_plan_from_root(const sirius_plan_node* root,
                                    sirius_plan** out,
                                    sirius_error** error) noexcept
{
  return create(out, error, [&] {
    require(root != nullptr, "Root is NULL");
    return new sirius_plan(acquire(root), describe(*root));
  });
}
sirius_plan_node* sirius_plan_root(const sirius_plan* plan) noexcept
{
  return plan ? plan->root.get() : nullptr;
}
const char* sirius_plan_describe(const sirius_plan* plan) noexcept
{
  return plan ? plan->description.c_str() : "";
}
void sirius_schema_retain(sirius_schema* value) noexcept { retain(value); }
void sirius_schema_release(sirius_schema* value) noexcept { release<sirius_schema>{}(value); }
void sirius_expression_retain(sirius_expression* value) noexcept { retain(value); }
void sirius_expression_release(sirius_expression* value) noexcept
{
  release<sirius_expression>{}(value);
}
void sirius_plan_node_retain(sirius_plan_node* value) noexcept { retain(value); }
void sirius_plan_node_release(sirius_plan_node* value) noexcept
{
  release<sirius_plan_node>{}(value);
}
void sirius_plan_retain(sirius_plan* value) noexcept { retain(value); }
void sirius_plan_release(sirius_plan* value) noexcept { release<sirius_plan>{}(value); }
