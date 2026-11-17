// Copyright 2026 Sirius Contributors
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "object_store_scope/settings.hpp"

#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace sirius::object_store_scope {

/// One published scope: its settings and the publisher's own object, which is kept alive with
/// the entry and handed back on lookup. The module never looks inside `owner`.
struct entry {
  std::shared_ptr<const settings> value;
  std::shared_ptr<void> owner;
};

/// The published scopes, ordered longest uri_prefix first (the default scope, empty prefix,
/// last). Immutable once published: a publication builds a new set, so a captured set never
/// changes.
using scope_set = std::vector<std::shared_ptr<const entry>>;

/// Why a read is not served.
enum class refusal_kind : std::uint8_t {
  unconfigured,
  unmatchable_key,
  out_of_scope,
  route_unavailable,
};

/// A refused item. `item` is the offending URI in redacted form.
struct refusal {
  refusal_kind kind;
  std::string item;
};

/// The stable reason token a refusal message starts with.
[[nodiscard]] std::string_view token_of(refusal_kind kind) noexcept;

/// Outcome of resolving one URI: a hit, a refusal, or neither when no scope serves it.
struct resolution {
  std::shared_ptr<const entry> hit;
  std::optional<refusal> refused;
};

/// The entry serving @p uri: the first (longest) prefix the scheme-lowered URI starts with; the
/// default scope matches every URI. Neither a hit nor a refusal when nothing matches or @p set
/// is empty. A key that cannot be matched byte-for-byte (an empty, "." or ".." segment, or a
/// raw `%` byte) is refused before any prefix is consulted — including when only the default
/// scope would serve it: the store or libcurl may normalize such a key into a published
/// prefix's namespace, which the default scope must not serve. Key bytes are never rewritten
/// here (only the scheme is case-folded) and a refused key is never retried, so a per-URI
/// lookup and the execution guard answer alike.
[[nodiscard]] resolution resolve(const scope_set& set, std::string_view uri);

/// Classifies the remote items of one plan against @p set: nothing when there is no item;
/// `unconfigured` when @p set is empty; otherwise the first item, in order, whose key is
/// refused or that no scope serves; and `route_unavailable` when every item is in scope. Every
/// item is checked before any is served: a plan is refused whole.
[[nodiscard]] std::optional<refusal> classify_remote_reads(
  const scope_set& set, std::span<const std::string> remote_items);

}  // namespace sirius::object_store_scope
