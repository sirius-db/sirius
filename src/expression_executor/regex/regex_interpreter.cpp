/*
 * Copyright 2025, Sirius Contributors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "expression_executor/regex/regex_interpreter.hpp"

#include "duckdb/common/exception.hpp"

#include <cstddef>
#include <functional>
#include <memory>
#include <set>
#include <sstream>
#include <utility>
#include <vector>

namespace sirius {
namespace expression {
namespace {

enum class NodeKind {
  START_ANCHOR,
  END_ANCHOR,
  LITERAL,
  DOT,
  CHAR_CLASS,
  SEQUENCE,
  QUANTIFIED,
  GROUP,
  NON_CAPTURING_GROUP
};

struct Node {
  explicit Node(NodeKind kind_p) : kind(kind_p) {}
  virtual ~Node() = default;

  NodeKind kind;
};

struct StartAnchor : Node {
  StartAnchor() : Node(NodeKind::START_ANCHOR) {}
};

struct EndAnchor : Node {
  EndAnchor() : Node(NodeKind::END_ANCHOR) {}
};

struct Literal : Node {
  explicit Literal(std::string text_p) : Node(NodeKind::LITERAL), text(std::move(text_p)) {}
  std::string text;
};

struct Dot : Node {
  Dot() : Node(NodeKind::DOT) {}
};

struct CharClass : Node {
  CharClass(std::vector<char> chars_p, bool negated_p)
      : Node(NodeKind::CHAR_CLASS), chars(std::move(chars_p)), negated(negated_p) {}

  std::vector<char> chars;
  bool negated;
};

struct Sequence : Node {
  Sequence() : Node(NodeKind::SEQUENCE) {}
  std::vector<std::unique_ptr<Node>> children;
};

struct Quantified : Node {
  Quantified(std::unique_ptr<Node> child_p, char quant_p)
      : Node(NodeKind::QUANTIFIED), child(std::move(child_p)), quant(quant_p) {}

  std::unique_ptr<Node> child;
  char quant;
};

struct Group : Node {
  Group(std::unique_ptr<Node> child_p, bool capturing_p, std::optional<int> index_p)
      : Node(NodeKind::GROUP), child(std::move(child_p)), capturing(capturing_p), group_index(index_p) {}

  std::unique_ptr<Node> child;
  bool capturing;
  std::optional<int> group_index;
};

struct NonCapturingGroup : Node {
  explicit NonCapturingGroup(std::unique_ptr<Node> child_p)
      : Node(NodeKind::NON_CAPTURING_GROUP), child(std::move(child_p)) {}

  std::unique_ptr<Node> child;
};

struct CodegenContext {
  void Emit(const std::string& line = "") { lines.push_back(std::string(indent * 4, ' ') + line); }
  void IndentMore() { ++indent; }
  void IndentLess() { --indent; }

  int indent = 0;
  std::vector<std::string> lines;
};

std::string EnsureWrapped(const std::string& value) {
  if (value.size() >= 2 && value.front() == '(' && value.back() == ')') {
    return value;
  }
  return "(" + value + ")";
}

std::unique_ptr<CharClass> ParseCharClass(const std::string& src, std::size_t& pos) {
  if (src[pos] != '[') {
    throw duckdb::InvalidInputException("Expected '[' when parsing character class");
  }

  auto closing = src.find(']', pos + 1);
  if (closing == std::string::npos) {
    throw duckdb::InvalidInputException("Unterminated character class");
  }

  auto body = src.substr(pos + 1, closing - pos - 1);
  bool negated = false;
  std::vector<char> chars;
  std::size_t idx = 0;
  if (!body.empty() && body[0] == '^') {
    negated = true;
    idx = 1;
  }

  while (idx < body.size()) {
    auto c = body[idx];
    if (c == '\\' && idx + 1 < body.size()) {
      chars.push_back(body[idx + 1]);
      idx += 2;
    } else {
      chars.push_back(c);
      ++idx;
    }
  }

  pos = closing + 1;
  return std::make_unique<CharClass>(std::move(chars), negated);
}

std::unique_ptr<Sequence> ParseGroupBody(const std::string& src) {
  auto result = std::make_unique<Sequence>();
  std::size_t i = 0;
  const auto n = src.size();

  while (i < n) {
    char c = src[i];
    std::unique_ptr<Node> node;

    if (c == '\\') {
      if (i + 1 >= n) {
        throw duckdb::InvalidInputException("Dangling escape in group");
      }
      node = std::make_unique<Literal>(src.substr(i + 1, 1));
      i += 2;
    } else if (c == '.') {
      node = std::make_unique<Dot>();
      ++i;
    } else if (c == '[') {
      node = ParseCharClass(src, i);
    } else {
      node = std::make_unique<Literal>(src.substr(i, 1));
      ++i;
    }

    if (i < n && (src[i] == '?' || src[i] == '+' || src[i] == '*')) {
      node = std::make_unique<Quantified>(std::move(node), src[i]);
      ++i;
    }

    result->children.push_back(std::move(node));
  }

  return result;
}

std::unique_ptr<Sequence> ParsePattern(const std::string& src) {
  if (src.size() < 2) {
    throw duckdb::InvalidInputException("Regex pattern too short");
  }

  const auto inner = src.substr(1, src.size() - 2);
  auto result = std::make_unique<Sequence>();

  std::size_t i = 0;
  const auto n = inner.size();
  if (i < n && inner[i] == '^') {
    result->children.push_back(std::make_unique<StartAnchor>());
    ++i;
  }

  int group_count = 0;

  while (i < n) {
    char c = inner[i];
    std::unique_ptr<Node> node;

    if (c == '$' && i == n - 1) {
      result->children.push_back(std::make_unique<EndAnchor>());
      ++i;
      break;
    }

    if (c == '\\') {
      if (i + 1 >= n) {
        throw duckdb::InvalidInputException("Dangling escape in pattern");
      }
      node = std::make_unique<Literal>(inner.substr(i + 1, 1));
      i += 2;
    } else if (c == '.') {
      node = std::make_unique<Dot>();
      ++i;
    } else if (c == '[') {
      node = ParseCharClass(inner, i);
    } else if (c == '(') {
      auto closing = inner.find(')', i + 1);
      if (closing == std::string::npos) {
        throw duckdb::InvalidInputException("Unterminated group");
      }

      auto body = inner.substr(i + 1, closing - i - 1);
      if (body.rfind("?:", 0) == 0) {
        auto child = ParseGroupBody(body.substr(2));
        node = std::make_unique<NonCapturingGroup>(std::move(child));
      } else {
        ++group_count;
        auto child = ParseGroupBody(body);
        node = std::make_unique<Group>(std::move(child), true, group_count);
      }
      i = closing + 1;
    } else {
      node = std::make_unique<Literal>(inner.substr(i, 1));
      ++i;
    }

    if (i < n && (inner[i] == '?' || inner[i] == '+' || inner[i] == '*')) {
      node = std::make_unique<Quantified>(std::move(node), inner[i]);
      ++i;
    }

    result->children.push_back(std::move(node));
  }

  return result;
}

std::unique_ptr<Node> CoalesceLiterals(std::unique_ptr<Node> node) {
  if (!node) {
    return node;
  }

  switch (node->kind) {
  case NodeKind::SEQUENCE: {
    auto* seq = static_cast<Sequence*>(node.get());
    std::vector<std::unique_ptr<Node>> new_children;
    std::string buf;
    for (auto& child : seq->children) {
      child = CoalesceLiterals(std::move(child));
      if (child && child->kind == NodeKind::LITERAL) {
        buf += static_cast<Literal*>(child.get())->text;
      } else {
        if (!buf.empty()) {
          new_children.push_back(std::make_unique<Literal>(buf));
          buf.clear();
        }
        new_children.push_back(std::move(child));
      }
    }
    if (!buf.empty()) {
      new_children.push_back(std::make_unique<Literal>(buf));
    }
    seq->children = std::move(new_children);
    return node;
  }
  case NodeKind::GROUP: {
    auto* grp = static_cast<Group*>(node.get());
    grp->child = CoalesceLiterals(std::move(grp->child));
    return node;
  }
  case NodeKind::NON_CAPTURING_GROUP: {
    auto* grp = static_cast<NonCapturingGroup*>(node.get());
    grp->child = CoalesceLiterals(std::move(grp->child));
    return node;
  }
  case NodeKind::QUANTIFIED: {
    auto* quant = static_cast<Quantified*>(node.get());
    quant->child = CoalesceLiterals(std::move(quant->child));
    return node;
  }
  default:
    return node;
  }
}

void CollectGroups(const Node& node, std::set<int>& out) {
  switch (node.kind) {
  case NodeKind::GROUP: {
    auto const& grp = static_cast<const Group&>(node);
    if (grp.capturing && grp.group_index.has_value()) {
      out.insert(*grp.group_index);
      CollectGroups(*grp.child, out);
    }
    break;
  }
  case NodeKind::SEQUENCE: {
    auto const& seq = static_cast<const Sequence&>(node);
    for (auto const& child : seq.children) {
      CollectGroups(*child, out);
    }
    break;
  }
  case NodeKind::NON_CAPTURING_GROUP: {
    auto const& grp = static_cast<const NonCapturingGroup&>(node);
    CollectGroups(*grp.child, out);
    break;
  }
  case NodeKind::QUANTIFIED: {
    auto const& quant = static_cast<const Quantified&>(node);
    CollectGroups(*quant.child, out);
    break;
  }
  default:
    break;
  }
}

void EmitMismatch(CodegenContext& ctx) {
  ctx.Emit("{");
  ctx.IndentMore();
  ctx.Emit("*out = url;");
  ctx.Emit("return;");
  ctx.IndentLess();
  ctx.Emit("}");
}

std::string EmitCharLiteral(char ch) {
  if (ch == '\\') {
    return "'\\\\'";
  }
  if (ch == '\'') {
    return "'\\''";
  }
  return std::string("'") + ch + "'";
}

void EmitCharClassMatch(CodegenContext& ctx, const CharClass& cls) {
  if (cls.negated) {
    if (cls.chars.size() != 1) {
      throw duckdb::NotImplementedException("Only single-char negated classes are supported");
    }
    auto ch = EmitCharLiteral(cls.chars[0]);
    ctx.Emit("if (pos >= static_cast<int32_t>(len) || url[pos] == " + ch + ")");
    EmitMismatch(ctx);
    ctx.Emit("++pos;");
    return;
  }

  std::vector<std::string> conds;
  conds.reserve(cls.chars.size());
  for (auto ch : cls.chars) {
    conds.emplace_back("url[pos] == " + EmitCharLiteral(ch));
  }
  std::ostringstream cond;
  for (std::size_t idx = 0; idx < conds.size(); ++idx) {
    if (idx > 0) {
      cond << " || ";
    }
    cond << conds[idx];
  }

  ctx.Emit("if (pos >= static_cast<int32_t>(len) || !(" + cond.str() + "))");
  EmitMismatch(ctx);
  ctx.Emit("++pos;");
}

void GenerateQuantified(CodegenContext& ctx, const Quantified& q) {
  auto* child = q.child.get();
  char quant = q.quant;
  ctx.Emit(std::string("// Quantifier ") + quant);

  if (quant == '?') {
    ctx.Emit("{");
    ctx.IndentMore();
    ctx.Emit("int32_t save_pos = pos;");

    if (child->kind == NodeKind::LITERAL) {
      auto const& lit = static_cast<const Literal&>(*child);
      const auto& text = lit.text;
      if (text.size() == 1) {
        auto ch = EmitCharLiteral(text[0]);
        ctx.Emit("if (pos < static_cast<int32_t>(len) && url[pos] == " + ch + ") {");
        ctx.IndentMore();
        ctx.Emit("++pos;");
        ctx.IndentLess();
        ctx.Emit("} else {");
        ctx.IndentMore();
        ctx.Emit("pos = save_pos;");
        ctx.IndentLess();
        ctx.Emit("}");
      } else {
        const auto n = text.size();
        std::vector<std::string> conds;
        conds.reserve(n);
        for (std::size_t i = 0; i < n; ++i) {
          conds.emplace_back("url[pos + " + std::to_string(i) + "] == " + EmitCharLiteral(text[i]));
        }
        std::ostringstream cond_join;
        for (std::size_t i = 0; i < conds.size(); ++i) {
          if (i > 0) {
            cond_join << " && ";
          }
          cond_join << conds[i];
        }
        ctx.Emit("if (len - pos >= " + std::to_string(n) + " && ");
        ctx.Emit("    " + cond_join.str() + ") {");
        ctx.IndentMore();
        ctx.Emit("pos += " + std::to_string(n) + ";");
        ctx.IndentLess();
        ctx.Emit("} else {");
        ctx.IndentMore();
        ctx.Emit("pos = save_pos;");
        ctx.IndentLess();
        ctx.Emit("}");
      }
    } else if (child->kind == NodeKind::NON_CAPTURING_GROUP) {
      auto const& grp = static_cast<const NonCapturingGroup&>(*child);
      if (grp.child->kind != NodeKind::SEQUENCE) {
        throw duckdb::NotImplementedException("Only literal non-capturing groups are supported for '?'");
      }

      std::string lit;
      for (auto const& inner_child : static_cast<const Sequence&>(*grp.child).children) {
        if (inner_child->kind != NodeKind::LITERAL) {
          throw duckdb::NotImplementedException("Only literal non-capturing groups are supported for '?'");
        }
        lit += static_cast<const Literal&>(*inner_child).text;
      }

      auto n = lit.size();
      std::vector<std::string> conds;
      conds.reserve(n);
      for (std::size_t i = 0; i < n; ++i) {
        conds.emplace_back("url[pos + " + std::to_string(i) + "] == " + EmitCharLiteral(lit[i]));
      }
      std::ostringstream cond_join;
      for (std::size_t i = 0; i < conds.size(); ++i) {
        if (i > 0) {
          cond_join << " && ";
        }
        cond_join << conds[i];
      }
      ctx.Emit("if (len - pos >= " + std::to_string(n) + " && ");
      ctx.Emit("    " + cond_join.str() + ") {");
      ctx.IndentMore();
      ctx.Emit("pos += " + std::to_string(n) + ";");
      ctx.IndentLess();
      ctx.Emit("} else {");
      ctx.IndentMore();
      ctx.Emit("pos = save_pos;");
      ctx.IndentLess();
      ctx.Emit("}");
    } else {
      throw duckdb::NotImplementedException("Unsupported child for '?' quantifier in this prototype");
    }

    ctx.IndentLess();
    ctx.Emit("}");
    return;
  }

  if (quant == '+' || quant == '*') {
    if (child->kind == NodeKind::DOT) {
      ctx.Emit("auto newline_pos = url.find('\\n', pos);");
      if (quant == '+') {
        ctx.Emit("if (pos >= static_cast<int32_t>(len) || newline_pos == pos)");
        EmitMismatch(ctx);
      }
      ctx.Emit("if (newline_pos == cudf::string_view::npos) {");
      ctx.IndentMore();
      ctx.Emit("pos = static_cast<int32_t>(len);");
      ctx.IndentLess();
      ctx.Emit("} else {");
      ctx.IndentMore();
      ctx.Emit("pos = static_cast<int32_t>(newline_pos);");
      ctx.IndentLess();
      ctx.Emit("}");
      return;
    }

    if (child->kind == NodeKind::CHAR_CLASS) {
      auto const& cls = static_cast<const CharClass&>(*child);
      if (!(cls.negated && cls.chars.size() == 1)) {
        throw duckdb::NotImplementedException("Unsupported child for '+' or '*'");
      }
      auto ch = EmitCharLiteral(cls.chars[0]);
      ctx.Emit("auto stop_pos = url.find(" + ch + ", pos);");
      if (quant == '+') {
        ctx.Emit("if (pos >= static_cast<int32_t>(len) || stop_pos == pos)");
        EmitMismatch(ctx);
      }
      ctx.Emit("if (stop_pos == cudf::string_view::npos) {");
      ctx.IndentMore();
      ctx.Emit("pos = static_cast<int32_t>(len);");
      ctx.IndentLess();
      ctx.Emit("} else {");
      ctx.IndentMore();
      ctx.Emit("pos = static_cast<int32_t>(stop_pos);");
      ctx.IndentLess();
      ctx.Emit("}");
      return;
    }

    if (child->kind == NodeKind::LITERAL) {
      auto const& lit = static_cast<const Literal&>(*child);
      if (lit.text.size() != 1) {
        throw duckdb::NotImplementedException("Unsupported child for '+' or '*'");
      }
      auto ch = EmitCharLiteral(lit.text[0]);
      if (quant == '+') {
        ctx.Emit("if (pos >= static_cast<int32_t>(len) || url[pos] != " + ch + ")");
        EmitMismatch(ctx);
      }
      ctx.Emit("auto next_pos = url.find_first_not_of(" + ch + ", pos);");
      ctx.Emit("if (next_pos == cudf::string_view::npos) {");
      ctx.IndentMore();
      ctx.Emit("pos = static_cast<int32_t>(len);");
      ctx.IndentLess();
      ctx.Emit("} else {");
      ctx.IndentMore();
      ctx.Emit("pos = static_cast<int32_t>(next_pos);");
      ctx.IndentLess();
      ctx.Emit("}");
      return;
    }

    throw duckdb::NotImplementedException("Unsupported child for '+' or '*' in this prototype");
  }

  throw duckdb::NotImplementedException("Unknown quantifier");
}

void GenerateNode(CodegenContext& ctx, const Node& node) {
  switch (node.kind) {
  case NodeKind::START_ANCHOR: {
    ctx.Emit("// ^ start anchor");
    break;
  }
  case NodeKind::END_ANCHOR: {
    ctx.Emit("// $ end anchor");
    ctx.Emit("if (pos != static_cast<int32_t>(len))");
    EmitMismatch(ctx);
    break;
  }
  case NodeKind::LITERAL: {
    auto const& lit = static_cast<const Literal&>(node);
    auto const& text = lit.text;
    auto n = text.size();
    ctx.Emit("// Literal \"" + text + "\"");
    std::vector<std::string> conds;
    conds.reserve(n);
    for (std::size_t i = 0; i < n; ++i) {
      conds.emplace_back("url[pos + " + std::to_string(i) + "] == " + EmitCharLiteral(text[i]));
    }
    std::ostringstream cond_join;
    for (std::size_t i = 0; i < conds.size(); ++i) {
      if (i > 0) {
        cond_join << " && ";
      }
      cond_join << conds[i];
    }
    ctx.Emit("if (!(len - pos >= " + std::to_string(n) + " && ");
    ctx.Emit("      " + cond_join.str() + "))");
    EmitMismatch(ctx);
    ctx.Emit("pos += " + std::to_string(n) + ";");
    break;
  }
  case NodeKind::QUANTIFIED: {
    auto const& quant = static_cast<const Quantified&>(node);
    GenerateQuantified(ctx, quant);
    break;
  }
  case NodeKind::NON_CAPTURING_GROUP: {
    auto const& grp = static_cast<const NonCapturingGroup&>(node);
    ctx.Emit("// Non-capturing group");
    GenerateNode(ctx, *grp.child);
    break;
  }
  case NodeKind::GROUP: {
    auto const& grp = static_cast<const Group&>(node);
    if (grp.child->kind != NodeKind::SEQUENCE) {
      throw duckdb::NotImplementedException("Only sequence groups are supported in this prototype");
    }
    int gid = grp.group_index.value_or(-1);
    ctx.Emit("// Capturing group " + std::to_string(gid));
    ctx.Emit("g" + std::to_string(gid) + "_start = pos;");
    auto const& seq = static_cast<const Sequence&>(*grp.child);
    for (auto const& child : seq.children) {
      GenerateNode(ctx, *child);
    }
    ctx.Emit("g" + std::to_string(gid) + "_end = pos;");
    break;
  }
  case NodeKind::CHAR_CLASS: {
    auto const& cls = static_cast<const CharClass&>(node);
    ctx.Emit("// Character class");
    EmitCharClassMatch(ctx, cls);
    break;
  }
  case NodeKind::DOT: {
    ctx.Emit("// Dot");
    ctx.Emit("if (pos >= static_cast<int32_t>(len) || url[pos] == '\\n')");
    EmitMismatch(ctx);
    ctx.Emit("++pos;");
    break;
  }
  default:
    throw duckdb::NotImplementedException("Unsupported node");
  }
}

std::string GenerateCudaUdf(const std::string& fn_name, const Sequence& pattern_ast, const std::string& replacement) {
  CodegenContext ctx;
  ctx.Emit("__device__ void " + fn_name + "(cudf::string_view* out, "
           "cuda::std::optional<cudf::string_view> const url_opt) {");
  ctx.IndentMore();
  ctx.Emit("// Skip null");
  ctx.Emit("if (!url_opt.has_value()) {");
  ctx.IndentMore();
  ctx.Emit("return;");
  ctx.IndentLess();
  ctx.Emit("}");
  ctx.Emit("cudf::string_view url = url_opt.value();");
  ctx.Emit("auto len = url.length();");
  ctx.Emit("int32_t pos = 0;");

  std::set<int> group_ids;
  CollectGroups(pattern_ast, group_ids);
  for (int gid : group_ids) {
    ctx.Emit("int32_t g" + std::to_string(gid) + "_start = -1;");
    ctx.Emit("int32_t g" + std::to_string(gid) + "_end = -1;");
  }

  for (auto const& node : pattern_ast.children) {
    GenerateNode(ctx, *node);
  }

  ctx.Emit("// Build replacement on success");
  if (replacement.size() < 2) {
    throw duckdb::InvalidInputException("Replacement is too short for regex interpreter");
  }
  auto replacement_inner = replacement.substr(1, replacement.size() - 2);
  if (replacement_inner == "\\1" && group_ids.count(1) > 0) {
    ctx.Emit("if (g1_start >= 0 && g1_end >= g1_start) {");
    ctx.IndentMore();
    ctx.Emit("*out = url.substr(g1_start, g1_end - g1_start);");
    ctx.IndentLess();
    ctx.Emit("} else {");
    ctx.IndentMore();
    ctx.Emit("*out = url;");
    ctx.IndentLess();
    ctx.Emit("}");
  } else {
    ctx.Emit("*out = url; // TODO: general replacement handling");
  }

  ctx.IndentLess();
  ctx.Emit("}");

  std::ostringstream ss;
  for (std::size_t i = 0; i < ctx.lines.size(); ++i) {
    ss << ctx.lines[i];
    if (i + 1 < ctx.lines.size()) {
      ss << "\n";
    }
  }
  return ss.str();
}

std::string MakeKey(const std::string& pattern, const std::string& replacement) {
  return pattern + "\x1f" + replacement;
}

std::string MakeFunctionName(const std::string& pattern, const std::string& replacement) {
  std::size_t hash_value = std::hash<std::string>{}(pattern + "|" + replacement);
  std::ostringstream ss;
  ss << "regex_udf_" << std::hex << hash_value;
  return ss.str();
}

} // namespace

RegexUdf RegexInterpreter::Generate(std::string pattern, std::string replacement) const {
  auto normalized_pattern = EnsureWrapped(pattern);
  auto normalized_replacement = EnsureWrapped(replacement);

  auto ast = ParsePattern(normalized_pattern);
  auto coalesced = CoalesceLiterals(std::move(ast));
  auto* seq_ptr = dynamic_cast<Sequence*>(coalesced.release());
  if (!seq_ptr) {
    throw duckdb::InvalidInputException("Failed to build regex AST");
  }
  std::unique_ptr<Sequence> seq(seq_ptr);

  auto fn_name = MakeFunctionName(normalized_pattern, normalized_replacement);
  auto source = GenerateCudaUdf(fn_name, *seq, normalized_replacement);
  return RegexUdf{fn_name, std::move(source)};
}

RegexUdfCache& RegexUdfCache::Instance() {
  static RegexUdfCache cache;
  return cache;
}

const RegexUdf& RegexUdfCache::GetOrCreate(const std::string& pattern, const std::string& replacement) {
  auto normalized_pattern = EnsureWrapped(pattern);
  auto normalized_replacement = EnsureWrapped(replacement);
  const auto key = MakeKey(normalized_pattern, normalized_replacement);

  {
    std::lock_guard<std::mutex> guard(mutex_);
    auto it = cache_.find(key);
    if (it != cache_.end()) {
      return it->second;
    }
  }

  auto udf = interpreter_.Generate(normalized_pattern, normalized_replacement);

  std::lock_guard<std::mutex> guard(mutex_);
  auto it = cache_.find(key);
  if (it == cache_.end()) {
    auto inserted = cache_.emplace(key, std::move(udf));
    return inserted.first->second;
  }
  return it->second;
}

} // namespace expression
} // namespace sirius
