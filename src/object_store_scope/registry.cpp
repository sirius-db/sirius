// Copyright 2026 Sirius Contributors
// SPDX-License-Identifier: Apache-2.0
#include "object_store_scope/registry.hpp"

#include <algorithm>
#include <utility>
#include <vector>

namespace sirius::object_store_scope {

namespace {

// A sorted copy of `current` with the entry of `key` replaced by `replacement` (or removed when
// null).
std::shared_ptr<const scope_set> with_key(const scope_set& current,
                                          std::string_view key,
                                          std::shared_ptr<const entry> replacement)
{
  auto next = std::make_shared<scope_set>(current);
  std::erase_if(*next, [&](const auto& e) { return e->value->uri_prefix == key; });
  if (replacement) { next->push_back(std::move(replacement)); }
  std::stable_sort(next->begin(), next->end(), [](const auto& a, const auto& b) {
    return a->value->uri_prefix.size() > b->value->uri_prefix.size();
  });
  return next;
}

}  // namespace

registry::registry() : _set(std::make_shared<const scope_set>()) {}

std::shared_ptr<const scope_set> registry::capture() const
{
  std::lock_guard lock(_mutex);
  return _set;
}

resolution registry::lookup(std::string_view uri) const { return resolve(*capture(), uri); }

// Lock order: update, then publication. The candidate entry and the retired set are owned by
// this frame, outside both locks: a candidate whose publication fails and a set that was
// replaced are released only after the locks are gone.
registry::outcome registry::publish(std::shared_ptr<const settings> value,
                                    std::shared_ptr<void> owner,
                                    const capabilities& backend)
{
  if (auto refused = check_settings(*value, backend)) {
    return {.refused = refused, .published = nullptr};
  }
  auto const candidate = std::make_shared<const entry>(entry{std::move(value), std::move(owner)});
  std::shared_ptr<const scope_set> retired;
  std::shared_ptr<const scope_set> published;
  {
    std::lock_guard update(_update_mutex);
    auto next = with_key(*capture(), candidate->value->uri_prefix, candidate);
    published = next;
    std::lock_guard lock(_mutex);
    retired = std::exchange(_set, std::move(next));
  }
  return {.refused = std::nullopt, .published = std::move(published)};
}

registry::outcome registry::drop(std::string_view uri_prefix)
{
  if (auto refused = check_uri_prefix(uri_prefix)) {
    return {.refused = refused, .published = nullptr};
  }
  std::shared_ptr<const scope_set> retired;
  std::shared_ptr<const scope_set> published;
  {
    std::lock_guard update(_update_mutex);
    auto const current = capture();
    bool const present = std::any_of(current->begin(), current->end(), [&](const auto& e) {
      return e->value->uri_prefix == uri_prefix;
    });
    if (!present) { return {}; }
    auto next = with_key(*current, uri_prefix, nullptr);
    published = next;
    std::lock_guard lock(_mutex);
    retired = std::exchange(_set, std::move(next));
  }
  return {.refused = std::nullopt, .published = std::move(published)};
}

}  // namespace sirius::object_store_scope
