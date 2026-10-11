/*
 * Copyright 2026, Sirius Contributors.
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

#include "expression/gpu_regexp_replace.hpp"

#include <cudf/strings/regex/regex_program.hpp>
#include <cudf/utilities/error.hpp>

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <string>
#include <variant>
#include <vector>

namespace sirius {

namespace {

// RE2 rejects nested counted repetitions whose counts multiply past this (its kMaxRepeat).
constexpr int max_repeat = 1000;
// RE2 rejects deep nesting; cuDF's limit is unknown, so stay far below it.
constexpr int max_depth = 100;

// Characters with a meaning in cuDF's regex syntax, outside and inside a character class.
constexpr std::string_view metacharacters       = R"(\.+*?()|[]{}^$)";
constexpr std::string_view class_metacharacters = R"(\]^-[)";

// RE2's Perl classes are ASCII-only, while cuDF's follow Unicode; RE2's \s also excludes \v.
// These spell out RE2's sets as cuDF class contents.
constexpr std::string_view digit_set = "0-9";
constexpr std::string_view space_set = "\t\n\f\r ";
constexpr std::string_view word_set  = "0-9A-Za-z_";

// Under case folding RE2 also matches these with k and s (Unicode simple case folding).
constexpr char32_t kelvin_sign = U'K';
constexpr char32_t long_s      = U'ſ';

void append_utf8(std::string& out, char32_t c)
{
  if (c < 0x80) {
    out += static_cast<char>(c);
  } else if (c < 0x800) {
    out += static_cast<char>(0xC0 | (c >> 6));
    out += static_cast<char>(0x80 | (c & 0x3F));
  } else if (c < 0x10000) {
    out += static_cast<char>(0xE0 | (c >> 12));
    out += static_cast<char>(0x80 | ((c >> 6) & 0x3F));
    out += static_cast<char>(0x80 | (c & 0x3F));
  } else {
    out += static_cast<char>(0xF0 | (c >> 18));
    out += static_cast<char>(0x80 | ((c >> 12) & 0x3F));
    out += static_cast<char>(0x80 | ((c >> 6) & 0x3F));
    out += static_cast<char>(0x80 | (c & 0x3F));
  }
}

bool is_metacharacter(char32_t c, std::string_view set)
{
  return c < 0x80 && set.find(static_cast<char>(c)) != std::string_view::npos;
}

// What a translated subexpression can match.
struct parsed {
  bool nullable;   // matches the empty string
  int repeat;      // largest product of nested repetition counts, as RE2 checks it
  bool assertion;  // zero-width anchor
  bool consumes;   // matches a nonempty string
};

// Translates a subset of RE2 syntax whose cuDF meaning matches RE2's into cuDF syntax. Every
// parse function returns nullopt for syntax outside the subset.
class translator {
 public:
  translator(std::string_view pattern, bool case_insensitive)
    : _in(pattern), _case_insensitive(case_insensitive)
  {
  }

  std::optional<parsed> regex()
  {
    auto result = alternation(0);
    // A stray ')' stops the parse before the end.
    if (!result || !at_end()) { return std::nullopt; }
    return result;
  }

  // The 'l' option: the pattern is a literal string.
  std::optional<parsed> literal()
  {
    while (!at_end()) {
      auto const c = code_point();
      if (!c || !emit_literal(*c)) { return std::nullopt; }
    }
    return parsed{_in.empty(), 1, false, !_in.empty()};
  }

  std::string const& out() const { return _out; }
  int groups() const { return _groups; }
  // Matches only at the start of the string, so at most once per row.
  bool anchored() const
  {
    return !_top_level_alternation && (_out.starts_with('^') || _out.starts_with("\\A"));
  }

 private:
  bool at_end() const { return _pos >= _in.size(); }
  char peek() const { return _in[_pos]; }

  std::optional<char32_t> code_point()
  {
    auto const lead  = static_cast<unsigned char>(_in[_pos]);
    auto const width = lead < 0x80             ? 1
                       : (lead & 0xE0) == 0xC0 ? 2
                       : (lead & 0xF0) == 0xE0 ? 3
                       : (lead & 0xF8) == 0xF0 ? 4
                                               : 0;
    if (width == 0 || _pos + width > _in.size()) { return std::nullopt; }
    char32_t c = width == 1 ? lead : lead & (0x7F >> width);
    for (int i = 1; i < width; ++i) {
      auto const next = static_cast<unsigned char>(_in[_pos + i]);
      if ((next & 0xC0) != 0x80) { return std::nullopt; }
      c = (c << 6) | (next & 0x3F);
    }
    _pos += width;
    return c;
  }

  // A one-character escape that stands for a literal: \t \n \r \f, \xHH and escaped punctuation.
  static std::optional<char32_t> escaped_literal(char c, std::string_view hex)
  {
    switch (c) {
      case 't': return U'\t';
      case 'n': return U'\n';
      case 'r': return U'\r';
      case 'f': return U'\f';
      default: break;
    }
    if (c == 'x') {
      // RE2 reads \xHH as a code point. Above 0x7F cuDF's reading is unknown.
      if (hex.size() < 2 || !std::isxdigit(static_cast<unsigned char>(hex[0])) ||
          !std::isxdigit(static_cast<unsigned char>(hex[1]))) {
        return std::nullopt;
      }
      auto const value = std::stoi(std::string(hex.substr(0, 2)), nullptr, 16);
      if (value >= 0x80) { return std::nullopt; }
      return static_cast<char32_t>(value);
    }
    if (std::ispunct(static_cast<unsigned char>(c))) { return static_cast<char32_t>(c); }
    return std::nullopt;
  }

  // Consumes the escape after a backslash: a literal, or a Perl class in `class_set`.
  std::optional<char32_t> escape(std::string_view& class_set)
  {
    if (at_end()) { return std::nullopt; }
    auto const c = _in[_pos++];
    switch (c) {
      case 'd': class_set = digit_set; return std::nullopt;
      case 's': class_set = space_set; return std::nullopt;
      // Case folding widens RE2's \w beyond ASCII.
      case 'w':
        if (!_case_insensitive) { class_set = word_set; }
        return std::nullopt;
      default: break;
    }
    auto const literal = escaped_literal(c, _in.substr(_pos));
    if (literal && c == 'x') { _pos += 2; }
    return literal;
  }

  bool emit_literal(char32_t c)
  {
    if (_case_insensitive && c >= 0x80) { return false; }
    if (_case_insensitive && std::isalpha(static_cast<int>(c))) {
      _out += '[';
      emit_class_range(c, c);
      _out += ']';
      return true;
    }
    if (is_metacharacter(c, metacharacters)) { _out += '\\'; }
    append_utf8(_out, c);
    return true;
  }

  void emit_class_char(char32_t c)
  {
    if (is_metacharacter(c, class_metacharacters)) { _out += '\\'; }
    append_utf8(_out, c);
  }

  // Emits [lo, hi] and, under case folding, the ASCII range's case-folded counterparts.
  void emit_class_range(char32_t lo, char32_t hi)
  {
    auto const emit = [this](char32_t from, char32_t to) {
      emit_class_char(from);
      if (to != from) {
        _out += '-';
        emit_class_char(to);
      }
    };
    emit(lo, hi);
    if (!_case_insensitive) { return; }
    auto const fold = [&](char32_t first, char32_t last) {
      auto const from = std::max(lo, first);
      auto const to   = std::min(hi, last);
      if (from <= to) { emit(from ^ 0x20, to ^ 0x20); }
    };
    fold(U'a', U'z');
    fold(U'A', U'Z');
    auto const contains = [&](char32_t c) {
      return (lo <= c && c <= hi) || (lo <= (c ^ 0x20) && (c ^ 0x20) <= hi);
    };
    if (contains(U'k')) { append_utf8(_out, kelvin_sign); }
    if (contains(U's')) { append_utf8(_out, long_s); }
  }

  std::optional<parsed> alternation(int depth)
  {
    if (depth > max_depth) { return std::nullopt; }
    auto result = concatenation(depth);
    while (result && !at_end() && peek() == '|') {
      if (depth == 0) { _top_level_alternation = true; }
      ++_pos;
      _out += '|';
      auto const branch = concatenation(depth);
      if (!branch) { return std::nullopt; }
      result = parsed{result->nullable || branch->nullable,
                      std::max(result->repeat, branch->repeat),
                      false,
                      result->consumes || branch->consumes};
    }
    return result;
  }

  std::optional<parsed> concatenation(int depth)
  {
    parsed result{true, 1, false, false};
    while (!at_end() && peek() != '|' && peek() != ')') {
      auto const next = piece(depth);
      if (!next) { return std::nullopt; }
      result.nullable = result.nullable && next->nullable;
      result.repeat   = std::max(result.repeat, next->repeat);
      result.consumes = result.consumes || next->consumes;
    }
    return result;
  }

  // {n}, {n,} or {n,m}, as {min, max} with max -1 for no bound.
  std::optional<std::pair<int, int>> counted_repetition()
  {
    auto const close = _in.find('}', _pos);
    if (close == std::string_view::npos) { return std::nullopt; }
    auto const body   = _in.substr(_pos + 1, close - _pos - 1);
    auto const number = [](std::string_view digits) -> std::optional<int> {
      if (digits.empty() || digits.size() > 4 ||
          !std::all_of(
            digits.begin(), digits.end(), [](unsigned char c) { return std::isdigit(c); })) {
        return std::nullopt;
      }
      return std::stoi(std::string(digits));
    };
    auto const comma = body.find(',');
    auto const min   = number(body.substr(0, comma));
    auto const max   = comma == std::string_view::npos ? min
                       : comma + 1 == body.size()      ? std::optional<int>{-1}
                                                       : number(body.substr(comma + 1));
    // RE2 reads other braces as literals; this subset rejects them instead.
    if (!min || !max || *min > max_repeat || *max > max_repeat || *max == 0 ||
        (*max >= 0 && *max < *min)) {
      return std::nullopt;
    }
    _out += _in.substr(_pos, close + 1 - _pos);
    _pos = close + 1;
    return std::pair{*min, *max};
  }

  std::optional<parsed> piece(int depth)
  {
    auto result = atom(depth);
    if (!result || at_end()) { return result; }
    int min = 1;
    switch (peek()) {
      case '*':
      case '?': min = 0; [[fallthrough]];
      case '+':
        _out += peek();
        ++_pos;
        break;
      case '{': {
        auto const counts = counted_repetition();
        if (!counts) { return std::nullopt; }
        min               = counts->first;
        auto const factor = counts->second >= 0 ? counts->second : counts->first;
        if (factor > 0) { result->repeat *= factor; }
        if (result->repeat > max_repeat) { return std::nullopt; }
        break;
      }
      default: return result;
    }
    if (result->assertion) { return std::nullopt; }
    if (!at_end() && peek() == '?') {
      _out += '?';
      ++_pos;
    }
    // RE2 rejects a repeated repetition such as a**.
    if (!at_end() && std::string_view("*+?{").find(peek()) != std::string_view::npos) {
      return std::nullopt;
    }
    result->nullable = result->nullable || min == 0;
    return result;
  }

  std::optional<parsed> atom(int depth)
  {
    switch (peek()) {
      case '(': return group(depth);
      case '[': return character_class();
      case '.':
        ++_pos;
        _out += '.';
        return parsed{false, 1, false, true};
      case '^':
        ++_pos;
        _out += '^';
        return parsed{true, 1, true, false};
      case '$':
        // RE2's $ is the end of the text; cuDF's also matches before a final newline.
        ++_pos;
        _out += "\\Z";
        return parsed{true, 1, true, false};
      case '\\': return escape_atom();
      // RE2 rejects a repetition with nothing to repeat and reads a bare brace as a literal.
      case '*':
      case '+':
      case '?':
      case '{': return std::nullopt;
      default: {
        auto const c = code_point();
        if (!c || !emit_literal(*c)) { return std::nullopt; }
        return parsed{false, 1, false, true};
      }
    }
  }

  std::optional<parsed> group(int depth)
  {
    ++_pos;
    if (_in.substr(_pos).starts_with("?:")) {
      _pos += 2;
      _out += "(?:";
    } else if (!at_end() && peek() == '?') {
      // Flags and named groups.
      return std::nullopt;
    } else {
      _out += '(';
      ++_groups;
    }
    auto const inner = alternation(depth + 1);
    if (!inner || at_end() || peek() != ')') { return std::nullopt; }
    ++_pos;
    _out += ')';
    return parsed{inner->nullable, inner->repeat, false, inner->consumes};
  }

  std::optional<parsed> escape_atom()
  {
    ++_pos;
    if (!at_end()) {
      auto const negated_set = [this](std::string_view set) {
        _pos++;
        _out += "[^";
        _out += set;
        _out += ']';
        return parsed{false, 1, false, true};
      };
      switch (peek()) {
        case 'D': return negated_set(digit_set);
        case 'S': return negated_set(space_set);
        case 'W':
          if (_case_insensitive) { return std::nullopt; }
          return negated_set(word_set);
        case 'A':
          _pos++;
          _out += "\\A";
          return parsed{true, 1, true, false};
        case 'z':
          _pos++;
          _out += "\\Z";
          return parsed{true, 1, true, false};
        default: break;
      }
    }
    std::string_view set;
    auto const literal = escape(set);
    if (!set.empty()) {
      _out += '[';
      _out += set;
      _out += ']';
      return parsed{false, 1, false, true};
    }
    if (!literal || !emit_literal(*literal)) { return std::nullopt; }
    return parsed{false, 1, false, true};
  }

  std::optional<parsed> character_class()
  {
    ++_pos;
    _out += '[';
    if (!at_end() && peek() == '^') {
      ++_pos;
      _out += '^';
    }
    // RE2 reads a leading ']' as a literal.
    for (bool first = true;; first = false) {
      if (at_end()) { return std::nullopt; }
      if (peek() == ']' && !first) {
        ++_pos;
        break;
      }
      // POSIX classes such as [:alpha:].
      if (_in.substr(_pos).starts_with("[:")) { return std::nullopt; }
      auto const low = class_item();
      if (!low) { return std::nullopt; }
      if (!*low) { continue; }  // a Perl class, already emitted
      auto high = *low;
      if (_pos + 1 < _in.size() && peek() == '-' && _in[_pos + 1] != ']') {
        ++_pos;
        auto const item = class_item();
        if (!item || !*item || **item < **low) { return std::nullopt; }
        high = *item;
      }
      if (_case_insensitive && *high >= 0x80) { return std::nullopt; }
      emit_class_range(**low, *high);
    }
    _out += ']';
    return parsed{false, 1, false, true};
  }

  // A class member: a code point, or an empty optional for a Perl class it emitted.
  std::optional<std::optional<char32_t>> class_item()
  {
    if (peek() != '\\') {
      auto const c = code_point();
      if (!c) { return std::nullopt; }
      return std::optional<char32_t>{*c};
    }
    ++_pos;
    std::string_view set;
    auto const literal = escape(set);
    if (!set.empty()) {
      _out += set;
      return std::optional<char32_t>{};
    }
    if (!literal) { return std::nullopt; }
    return std::optional<char32_t>{*literal};
  }

  std::string_view _in;
  bool _case_insensitive;
  std::size_t _pos{0};
  std::string _out;
  int _groups{0};
  bool _top_level_alternation{false};
};

// A replacement in RE2's rewrite syntax: literal text and group references \0 to \9.
using rewrite = std::vector<std::variant<std::string, int>>;

std::optional<rewrite> parse_rewrite(std::string_view replacement)
{
  rewrite parts;
  std::string literal;
  for (std::size_t i = 0; i < replacement.size(); ++i) {
    if (replacement[i] != '\\') {
      literal += replacement[i];
      continue;
    }
    // RE2 rejects any other escape, and a trailing backslash.
    if (++i == replacement.size()) { return std::nullopt; }
    auto const c = replacement[i];
    if (c == '\\') {
      literal += '\\';
    } else if (std::isdigit(static_cast<unsigned char>(c))) {
      if (!literal.empty()) { parts.emplace_back(std::move(literal)); }
      literal.clear();
      parts.emplace_back(c - '0');
    } else {
      return std::nullopt;
    }
  }
  if (!literal.empty()) { parts.emplace_back(std::move(literal)); }
  return parts;
}

}  // namespace

std::optional<gpu_regexp_replace> translate_gpu_regexp_replace(std::string_view pattern,
                                                               std::string_view replacement,
                                                               std::string_view options)
{
  // DuckDB's RE2 options: later letters override earlier ones.
  bool case_insensitive = false;
  bool literal          = false;
  bool dot_all          = false;
  bool global           = false;
  for (auto const option : options) {
    switch (option) {
      case 'c': case_insensitive = false; break;
      case 'i': case_insensitive = true; break;
      case 'l': literal = true; break;
      case 'm':
      case 'n':
      case 'p': dot_all = false; break;
      case 's': dot_all = true; break;
      case 'g': global = true; break;
      case ' ':
      case '\t':
      case '\n': break;
      default: return std::nullopt;
    }
  }

  translator regex(pattern, case_insensitive);
  auto const shape = literal ? regex.literal() : regex.regex();
  if (!shape) { return std::nullopt; }
  // cuDF and RE2 place empty matches after the first differently. The first match of a pattern
  // that can match the empty string is at the start, where cuDF finds it too, unless the pattern
  // matches only the empty string: cuDF never replaces with such a pattern.
  if (shape->nullable && (global || !shape->consumes)) { return std::nullopt; }

  auto const parts = parse_rewrite(replacement);
  if (!parts) { return std::nullopt; }
  auto max_group = -1;
  for (auto const& part : *parts) {
    if (auto const* group = std::get_if<int>(&part)) { max_group = std::max(max_group, *group); }
  }
  // DuckDB rejects a replacement that names a missing group.
  if (max_group > regex.groups()) { return std::nullopt; }

  gpu_regexp_replace result;
  result.flags     = dot_all ? cudf::strings::regex_flags::DOTALL : cudf::strings::regex_flags{};
  auto cudf_groups = regex.groups();
  if (max_group < 0) {
    result.pattern = regex.out();
    for (auto const& part : *parts) {
      result.replacement += std::get<std::string>(part);
    }
    if (!global) { result.max_replace_count = 1; }
  } else {
    result.backrefs = true;
    // replace_with_backrefs replaces every match. For the first match alone, a lazy prefix
    // anchors the pattern at its leftmost match, as RE2 itself searches an unanchored pattern.
    auto const prefixed = !global && !regex.anchored();
    auto const shift    = prefixed ? 2 : 0;
    if (prefixed) {
      result.pattern     = "\\A([\\s\\S]*?)(" + regex.out() + ")";
      result.replacement = "${1}";
      cudf_groups += 2;
    } else {
      result.pattern = regex.out();
    }
    for (auto const& part : *parts) {
      if (auto const* group = std::get_if<int>(&part)) {
        result.replacement += "${" + std::to_string(*group + shift) + "}";
        continue;
      }
      // cuDF would read these as group references.
      auto const& text = std::get<std::string>(part);
      if (text.find("${") != std::string::npos) { return std::nullopt; }
      for (std::size_t i = 0; i + 1 < text.size(); ++i) {
        if (text[i] == '\\' && std::isdigit(static_cast<unsigned char>(text[i + 1]))) {
          return std::nullopt;
        }
      }
      result.replacement += text;
    }
  }

  // cuDF rejects some patterns RE2 accepts, such as large repetition counts.
  try {
    auto const program = cudf::strings::regex_program::create(result.pattern, result.flags);
    if (program->groups_count() != cudf_groups) { return std::nullopt; }
  } catch (cudf::logic_error const&) {
    return std::nullopt;
  }
  return result;
}

}  // namespace sirius
