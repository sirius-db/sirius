// Copyright 2026 Sirius Contributors
// SPDX-License-Identifier: Apache-2.0
#include "object_store_scope/scope.hpp"

#include "io/s3/scope_policy.hpp"

#include <algorithm>
#include <utility>

namespace sirius::object_store_scope {

using sirius::io::s3::has_s3_scheme;
using sirius::io::s3::key_defeats_prefix_match;
using sirius::io::s3::lowercase_scheme;
using sirius::io::s3::redacted_s3_location;
using sirius::io::s3::redacted_uri;

namespace {

std::string refusal_item(std::string_view uri)
{
  return has_s3_scheme(uri) ? std::string{redacted_s3_location} : redacted_uri(uri);
}

}  // namespace

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

std::shared_ptr<const scope_set> with_entry(const scope_set& current,
                                            std::shared_ptr<const entry> replacement)
{
  auto next = std::make_shared<scope_set>(current);
  std::erase_if(
    *next, [&](const auto& e) { return e->value->uri_prefix == replacement->value->uri_prefix; });
  next->push_back(std::move(replacement));
  std::stable_sort(next->begin(), next->end(), [](const auto& a, const auto& b) {
    return a->value->uri_prefix.size() > b->value->uri_prefix.size();
  });
  return next;
}

std::shared_ptr<const scope_set> without_entry(const scope_set& current, std::string_view key)
{
  auto next = std::make_shared<scope_set>(current);
  std::erase_if(*next, [&](const auto& e) { return e->value->uri_prefix == key; });
  return next;
}

resolution resolve(const scope_set& set, std::string_view uri)
{
  if (set.empty()) { return {}; }
  if (key_defeats_prefix_match(uri)) {
    return {.hit     = nullptr,
            .refused = refusal{.kind = refusal_kind::unmatchable_key, .item = refusal_item(uri)}};
  }
  auto const lowered = lowercase_scheme(uri);
  for (const auto& candidate : set) {
    auto const& prefix = candidate->value->uri_prefix;
    if (prefix.empty() ? has_s3_scheme(uri) : lowered.starts_with(prefix)) {
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
    return refusal{.kind = refusal_kind::unconfigured, .item = refusal_item(remote_items.front())};
  }
  for (const auto& item : remote_items) {
    auto resolved = resolve(set, item);
    if (resolved.refused) { return std::move(resolved.refused); }
    if (!resolved.hit) {
      return refusal{.kind = refusal_kind::out_of_scope, .item = refusal_item(item)};
    }
  }
  return refusal{.kind = refusal_kind::route_unavailable,
                 .item = refusal_item(remote_items.front())};
}

}  // namespace sirius::object_store_scope
