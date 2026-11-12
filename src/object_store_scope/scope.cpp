// Copyright 2026 Sirius Contributors
// SPDX-License-Identifier: Apache-2.0
#include "object_store_scope/scope.hpp"

#include "io/s3/scope_policy.hpp"

#include <utility>

namespace sirius::object_store_scope {

using sirius::io::s3::key_defeats_prefix_match;
using sirius::io::s3::lowercase_scheme;
using sirius::io::s3::redacted_uri;

std::string_view token_of(refusal_kind kind) noexcept
{
  switch (kind) {
    case refusal_kind::unconfigured: return "object_store_unconfigured";
    case refusal_kind::unmatchable_key: return "object_store_scope";
    case refusal_kind::out_of_scope: return "object_store_scope";
    case refusal_kind::route_unavailable: return "object_store_route_unavailable";
  }
  return {};
}

resolution resolve(const scope_set& set, std::string_view uri)
{
  if (set.empty()) { return {}; }
  if (key_defeats_prefix_match(uri)) {
    return {.hit     = nullptr,
            .refused = refusal{.kind = refusal_kind::unmatchable_key, .item = redacted_uri(uri)}};
  }
  auto const lowered = lowercase_scheme(uri);
  for (const auto& candidate : set) {
    if (candidate->value->uri_prefix.empty() || lowered.starts_with(candidate->value->uri_prefix)) {
      return {.hit = candidate, .refused = std::nullopt};
    }
  }
  return {};
}

std::optional<refusal> classify_remote_reads(const scope_set& set,
                                             std::span<const std::string> remote_items)
{
  if (remote_items.empty()) { return std::nullopt; }
  if (set.empty()) {
    return refusal{.kind = refusal_kind::unconfigured, .item = redacted_uri(remote_items.front())};
  }
  for (const auto& item : remote_items) {
    auto resolved = resolve(set, item);
    if (resolved.refused) { return std::move(resolved.refused); }
    if (!resolved.hit) {
      return refusal{.kind = refusal_kind::out_of_scope, .item = redacted_uri(item)};
    }
  }
  return refusal{.kind = refusal_kind::route_unavailable,
                 .item = redacted_uri(remote_items.front())};
}

}  // namespace sirius::object_store_scope
