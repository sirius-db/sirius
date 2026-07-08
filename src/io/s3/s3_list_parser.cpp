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

#include "io/s3/s3_list_parser.hpp"

#include <cctype>
#include <optional>
#include <stdexcept>

namespace sirius::io::s3 {

namespace {

// Single-pass unescape of the five predefined XML entities. Unknown sequences
// (e.g. numeric character references, which S3 does not emit for keys) pass
// through verbatim.
std::string xml_unescape(std::string_view s)
{
  std::string out;
  out.reserve(s.size());
  for (std::size_t i = 0; i < s.size();) {
    if (s[i] == '&') {
      if (s.compare(i, 5, "&amp;") == 0) {
        out += '&';
        i += 5;
        continue;
      }
      if (s.compare(i, 4, "&lt;") == 0) {
        out += '<';
        i += 4;
        continue;
      }
      if (s.compare(i, 4, "&gt;") == 0) {
        out += '>';
        i += 4;
        continue;
      }
      if (s.compare(i, 6, "&quot;") == 0) {
        out += '"';
        i += 6;
        continue;
      }
      if (s.compare(i, 6, "&apos;") == 0) {
        out += '\'';
        i += 6;
        continue;
      }
    }
    out += s[i];
    ++i;
  }
  return out;
}

std::string_view trim(std::string_view s)
{
  std::size_t b = 0;
  std::size_t e = s.size();
  while (b < e && std::isspace(static_cast<unsigned char>(s[b])) != 0) {
    ++b;
  }
  while (e > b && std::isspace(static_cast<unsigned char>(s[e - 1])) != 0) {
    --e;
  }
  return s.substr(b, e - b);
}

// Raw text between the first `<tag>` and its `</tag>` (these S3 elements carry
// no attributes), searching from @p from. nullopt when the element is absent.
std::optional<std::string_view> element_text(std::string_view xml,
                                             std::string_view tag,
                                             std::size_t from = 0)
{
  std::string const open  = "<" + std::string{tag} + ">";
  std::string const close = "</" + std::string{tag} + ">";
  auto const o            = xml.find(open, from);
  if (o == std::string_view::npos) { return std::nullopt; }
  auto const s = o + open.size();
  auto const c = xml.find(close, s);
  if (c == std::string_view::npos) { return std::nullopt; }
  return xml.substr(s, c - s);
}

std::uint64_t parse_size(std::string_view raw)
{
  auto const text = trim(raw);
  if (text.empty()) {
    throw std::runtime_error("parse_list_objects_v2: empty <Size> in <Contents>");
  }
  std::uint64_t value = 0;
  for (char const c : text) {
    if (c < '0' || c > '9') {
      throw std::runtime_error("parse_list_objects_v2: non-numeric <Size> '" + std::string{text} +
                               "' in <Contents>");
    }
    value = value * 10 + static_cast<std::uint64_t>(c - '0');
  }
  return value;
}

}  // namespace

list_objects_v2_page parse_list_objects_v2(std::string_view xml)
{
  if (xml.find("<ListBucketResult") == std::string_view::npos) {
    throw std::runtime_error(
      "parse_list_objects_v2: not a ListObjectsV2 response (no <ListBucketResult>)");
  }

  list_objects_v2_page page;

  // Object entries: the <Key>/<Size> inside each <Contents> block. Scoping to
  // <Contents> (rather than scanning every <Key> in the document) keeps a stray
  // <Key> that an S3-compatible service might place elsewhere — or a future
  // response extension — from being mistaken for an object key. <CommonPrefixes>
  // rollups use <Prefix>, so they are excluded for free. A <Contents> without a
  // parseable <Size> throws: downstream opens rely on the LIST-provided size to
  // skip their size-discovery round-trip, so a silent zero would corrupt reads.
  constexpr std::string_view k_contents_open  = "<Contents>";
  constexpr std::string_view k_contents_close = "</Contents>";
  for (std::size_t pos = 0;;) {
    auto const co = xml.find(k_contents_open, pos);
    if (co == std::string_view::npos) { break; }
    auto const block_begin = co + k_contents_open.size();
    auto const ce          = xml.find(k_contents_close, block_begin);
    if (ce == std::string_view::npos) { break; }
    auto const block = xml.substr(block_begin, ce - block_begin);
    if (auto const key = element_text(block, "Key"); key.has_value()) {
      auto const size = element_text(block, "Size");
      if (!size.has_value()) {
        throw std::runtime_error("parse_list_objects_v2: <Contents> without <Size> for key '" +
                                 xml_unescape(*key) + "'");
      }
      page.entries.push_back({xml_unescape(*key), parse_size(*size)});
    }
    pos = ce + k_contents_close.size();
  }

  if (auto const truncated = element_text(xml, "IsTruncated"); truncated.has_value()) {
    page.is_truncated = trim(*truncated) == "true";
  }
  if (auto const token = element_text(xml, "NextContinuationToken"); token.has_value()) {
    page.next_continuation_token = xml_unescape(trim(*token));
  }

  return page;
}

}  // namespace sirius::io::s3
