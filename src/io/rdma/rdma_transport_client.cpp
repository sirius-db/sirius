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

#include "io/rdma/rdma_transport_client.hpp"

#include <cctype>

namespace sirius::io::rdma {

bool non_empty_reply_tag(std::string_view tag) noexcept { return !tag.empty(); }

namespace {

constexpr std::string_view k_token_label = "x-amz-rdma-token";

bool label_matches(std::string_view text, size_t at) noexcept
{
  if (at + k_token_label.size() > text.size()) { return false; }
  for (size_t i = 0; i < k_token_label.size(); ++i) {
    if (std::tolower(static_cast<unsigned char>(text[at + i])) != k_token_label[i]) {
      return false;
    }
  }
  return true;
}

}  // namespace

std::string redact_rdma_tokens(std::string_view text)
{
  std::string out;
  out.reserve(text.size());
  size_t i = 0;
  while (i < text.size()) {
    if (!label_matches(text, i)) {
      out.push_back(text[i]);
      ++i;
      continue;
    }
    // Keep the label itself (diagnostics stay attributable), drop the value.
    out.append(k_token_label);
    i += k_token_label.size();
    // Copy the immediate separator run (": ", "=", quotes, whitespace)
    // verbatim, then drop EVERYTHING up to the header-value boundary (CR/LF
    // or end of text).  Token formats are opaque, so no value-alphabet
    // guessing: over-redacting same-line text is the accepted trade against
    // leaking any token fragment.
    while (i < text.size() && (text[i] == ':' || text[i] == '=' || text[i] == ' ' ||
                               text[i] == '\t' || text[i] == '"' || text[i] == '\'')) {
      out.push_back(text[i]);
      ++i;
    }
    const size_t value_begin = i;
    while (i < text.size() && text[i] != '\r' && text[i] != '\n') {
      ++i;
    }
    if (i > value_begin) { out.append("[REDACTED]"); }
  }
  return out;
}

}  // namespace sirius::io::rdma
