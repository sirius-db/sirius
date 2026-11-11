// Copyright 2026 Sirius Contributors
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "object_store/scope.hpp"
#include "object_store/settings.hpp"
#include "object_store/validation.hpp"

#include <memory>
#include <mutex>
#include <optional>
#include <string_view>

namespace sirius::object_store {

/// The published scope set of one context, replaced as a whole (copy-on-write). Writers hold an
/// update lock across copy, edit and publish, so concurrent `publish()` / `drop()` compose like
/// sequential ones; `lookup()` and `capture()` take only the publication lock and never the
/// update lock. No owner and no retired set is released while either lock is held, so a
/// caller-supplied deleter may call back into the registry. Thread-safe.
class registry {
 public:
  /// Outcome of a write: the violation that refused it, or the set it published. Both are empty
  /// when a drop found nothing to remove.
  struct outcome {
    std::optional<violation> refused;
    std::shared_ptr<const scope_set> published;
  };

  registry();
  registry(const registry&)            = delete;
  registry& operator=(const registry&) = delete;

  /// Checks @p value against @p backend (`check_settings()`) and publishes it for the scope its
  /// `uri_prefix` names, replacing an entry already published for that key and keeping the other
  /// entries. @p value must not be null. @p owner is kept alive with the entry and returned by
  /// `lookup()`. A refused or failed publication leaves the published set as it was.
  [[nodiscard]] outcome publish(std::shared_ptr<const settings> value,
                                std::shared_ptr<void> owner,
                                const capabilities& backend);

  /// Removes the entry published for @p uri_prefix (exact bytes; empty = the default scope). A
  /// malformed key is refused; an absent key publishes nothing.
  [[nodiscard]] outcome drop(std::string_view uri_prefix);

  /// Resolves @p uri against the set published now; see `resolve()`.
  [[nodiscard]] resolution lookup(std::string_view uri) const;

  /// The set published now. It never changes and may outlive the registry.
  [[nodiscard]] std::shared_ptr<const scope_set> capture() const;

 private:
  std::mutex _update_mutex;
  mutable std::mutex _mutex;
  std::shared_ptr<const scope_set> _set;
};

}  // namespace sirius::object_store
