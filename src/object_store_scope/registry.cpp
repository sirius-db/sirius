// Copyright 2026 Sirius Contributors
// SPDX-License-Identifier: Apache-2.0
#include "object_store_scope/registry.hpp"

#include <algorithm>
#include <utility>
#include <vector>

namespace sirius::object_store_scope {

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
    auto next = with_entry(*capture(), candidate);
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
    auto next = without_entry(*current, uri_prefix);
    published = next;
    std::lock_guard lock(_mutex);
    retired = std::exchange(_set, std::move(next));
  }
  return {.refused = std::nullopt, .published = std::move(published)};
}

}  // namespace sirius::object_store_scope
