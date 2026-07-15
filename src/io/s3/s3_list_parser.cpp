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
#include <limits>
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
    auto const digit = static_cast<std::uint64_t>(c - '0');
    // Guard the accumulation: a wrapped-small size is later trusted to skip the
    // HEAD in the known-size open, so an overflow would silently truncate reads.
    if (value > (std::numeric_limits<std::uint64_t>::max() - digit) / 10) {
      throw std::runtime_error("parse_list_objects_v2: <Size> '" + std::string{text} +
                               "' overflows uint64 in <Contents>");
    }
    value = value * 10 + digit;
  }
  return value;
}

}  // namespace

list_objects_v2_page parse_list_objects_v2(std::string_view xml)
{
  constexpr std::string_view k_root_open  = "<ListBucketResult";
  constexpr std::string_view k_root_close = "</ListBucketResult>";
  // The root open tag must be exactly <ListBucketResult, terminated by '>' or
  // XML whitespace (attributes follow) — a prefix match alone would accept a
  // bogus root like <ListBucketResultBogus> and parse its content as a real
  // listing.
  auto const is_xml_space = [](char c) { return c == ' ' || c == '\t' || c == '\r' || c == '\n'; };
  std::size_t root_open   = std::string_view::npos;
  for (auto cand = xml.find(k_root_open); cand != std::string_view::npos;
       cand      = xml.find(k_root_open, cand + 1)) {
    auto const after = cand + k_root_open.size();
    if (after < xml.size() && (xml[after] == '>' || is_xml_space(xml[after]))) {
      root_open = cand;
      break;
    }
  }
  if (root_open == std::string_view::npos) {
    throw std::runtime_error(
      "parse_list_objects_v2: not a ListObjectsV2 response (no <ListBucketResult>)");
  }
  // Only whitespace and one optional <?xml ...?> prologue may precede the
  // root: any other leading content (a wrapper element, a comment, a second
  // document) marks a body this parser does not understand — fail closed
  // rather than silently skipping it.
  auto const pre = trim(xml.substr(0, root_open));
  if (!pre.empty()) {
    bool prologue_only = false;
    if (pre.substr(0, 5) == "<?xml") {
      auto const end = pre.find("?>");
      prologue_only  = end != std::string_view::npos && trim(pre.substr(end + 2)).empty();
    }
    if (!prologue_only) {
      throw std::runtime_error(
        "parse_list_objects_v2: unexpected content before <ListBucketResult>");
    }
  }
  // Window every scan below to the root element's CONTENT. Elements placed
  // outside the root (a <Contents>, <IsTruncated> or continuation token after
  // the close tag) must never be honored as results — an injected token could
  // even steer the pagination loop. The window starts after the open tag's '>'
  // (tolerating the xmlns attribute and any prologue before the root), and the
  // close is searched AFTER the open, so a close-before-open body is rejected
  // as truncated for free.
  auto const root_open_end = xml.find('>', root_open + k_root_open.size());
  if (root_open_end == std::string_view::npos) {
    throw std::runtime_error(
      "parse_list_objects_v2: truncated ListObjectsV2 response (unterminated <ListBucketResult> "
      "open tag)");
  }
  // Require the root close tag: a body truncated before it (a malformed /
  // partial 200 from an S3-compatible gateway) would otherwise parse as a
  // "valid" page with is_truncated defaulting to false — the paged loop would
  // believe the listing is complete and a glob would silently drop every object
  // past the cut. Real S3/MinIO always close the root.
  auto const root_close = xml.find(k_root_close, root_open_end + 1);
  if (root_close == std::string_view::npos) {
    throw std::runtime_error(
      "parse_list_objects_v2: truncated ListObjectsV2 response (missing </ListBucketResult>)");
  }
  auto const body = xml.substr(root_open_end + 1, root_close - root_open_end - 1);

  list_objects_v2_page page;

  // Object entries: the <Key>/<Size> inside each <Contents> block. Scoping to
  // <Contents> (rather than scanning every <Key> in the document) keeps a stray
  // <Key> that an S3-compatible service might place elsewhere — or a future
  // response extension — from being mistaken for an object key. <CommonPrefixes>
  // rollups use <Prefix>, so they are excluded for free. A <Contents> without a
  // present, closed, non-empty <Key> or without a parseable <Size> throws: AWS
  // documents both as always present, so their absence means a mangled body —
  // silently skipping the entry would make a glob drop the object with no
  // error, and downstream opens rely on the LIST-provided size to skip their
  // size-discovery round-trip, so a silent zero would corrupt reads.
  constexpr std::string_view k_contents_open  = "<Contents>";
  constexpr std::string_view k_contents_close = "</Contents>";
  for (std::size_t pos = 0;;) {
    auto const co = body.find(k_contents_open, pos);
    if (co == std::string_view::npos) { break; }  // no more objects — clean end
    auto const block_begin = co + k_contents_open.size();
    auto const ce          = body.find(k_contents_close, block_begin);
    if (ce == std::string_view::npos) {
      // An opened <Contents> with no close is a mid-block truncation — throw
      // rather than silently returning the objects parsed so far.
      throw std::runtime_error(
        "parse_list_objects_v2: truncated ListObjectsV2 page (unclosed <Contents>)");
    }
    auto const block = body.substr(block_begin, ce - block_begin);
    auto const key   = element_text(block, "Key");
    if (!key.has_value()) {
      throw std::runtime_error(
        "parse_list_objects_v2: malformed ListObjectsV2 page (<Contents> without <Key>)");
    }
    auto object_key = xml_unescape(*key);
    if (object_key.empty()) {
      throw std::runtime_error(
        "parse_list_objects_v2: malformed ListObjectsV2 page (empty <Key> in <Contents>)");
    }
    auto const size = element_text(block, "Size");
    if (!size.has_value()) {
      throw std::runtime_error("parse_list_objects_v2: <Contents> without <Size> for key '" +
                               object_key + "'");
    }
    page.entries.push_back({std::move(object_key), parse_size(*size)});
    pos = ce + k_contents_close.size();
  }

  // <IsTruncated> is mandatory and strictly boolean. AWS documents it as always
  // present; a page missing it (or carrying anything but true/false) is a
  // mangled body, and defaulting to "not truncated" would end the paged loop
  // early — a glob would silently treat a partial listing as complete.
  auto const truncated = element_text(body, "IsTruncated");
  if (!truncated.has_value()) {
    throw std::runtime_error(
      "parse_list_objects_v2: malformed ListObjectsV2 page (missing <IsTruncated>)");
  }
  auto const truncated_text = trim(*truncated);
  if (truncated_text == "true") {
    page.is_truncated = true;
  } else if (truncated_text == "false") {
    page.is_truncated = false;
  } else {
    throw std::runtime_error("parse_list_objects_v2: invalid <IsTruncated> value '" +
                             std::string{truncated_text} + "'");
  }

  if (auto const token = element_text(body, "NextContinuationToken"); token.has_value()) {
    page.next_continuation_token = xml_unescape(trim(*token));
  } else if (page.is_truncated) {
    // A truncated page must carry the token that fetches the next page; without
    // it the listing cannot be completed and must not be treated as complete.
    // (An empty <NextContinuationToken></NextContinuationToken> passes here and
    // is rejected by the paged caller — the parser validates body shape only.)
    throw std::runtime_error(
      "parse_list_objects_v2: truncated ListObjectsV2 page without a continuation token "
      "(<NextContinuationToken> missing)");
  }

  // Reject non-whitespace content after the root close. Checked LAST so a
  // malformed window (missing <IsTruncated>, token-less truncated page) reports
  // its own, more specific error first; nothing here was honored either way —
  // every scan above is window-scoped.
  if (!trim(xml.substr(root_close + k_root_close.size())).empty()) {
    throw std::runtime_error("parse_list_objects_v2: unexpected content after </ListBucketResult>");
  }

  return page;
}

}  // namespace sirius::io::s3
