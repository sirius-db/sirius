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

#pragma once

#include <cstddef>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>

namespace sirius::io::rest {
struct config;

namespace redirect_admission {

/// Pure admission result. Counters, diagnostics and delegation remain in the reactor.
struct redirect_decision {
  bool follow{false};
  std::string reason;
};

/// Evaluate the existing one-hop policy using the URL actually sent on the wire.
/// already_delegated describes this physical operation, not the logical object's URL.
/// The caller retains and uses Location verbatim; this function never rewrites it.
redirect_decision evaluate_cluster_redirect(config const& cfg,
                                            std::string_view request_url,
                                            bool already_delegated,
                                            long status,
                                            std::string_view location);

namespace detail {
/// Shared lexical parser for admission and reactor diagnostics. Views borrow url.
/// It deliberately does not normalize default ports or validate host spelling.
struct url_authority {
  std::string_view scheme;
  std::string_view authority;
};
std::optional<url_authority> split_authority(std::string_view url) noexcept;
}  // namespace detail

}  // namespace redirect_admission
namespace redirect_routes {

/// The retained immutable string object is the publication identity. Equal URL
/// bytes in distinct allocations are different publications, as in the reactor.
using snapshot = std::shared_ptr<const std::string>;

/// One slot per open, visible only to its live admitting context. An expired or
/// foreign owner does not free an occupied slot. Keep the selected snapshot alive
/// through the attempt so invalidation cannot erase a newer publication.
class route_cache {
 public:
  [[nodiscard]] snapshot route(std::shared_ptr<const void> const& owner) const;
  bool seed_route(snapshot route, std::weak_ptr<const void> owner) const;
  void drop_route(snapshot const& stale) const;

  /// Optional publication after the adapter's admission/validation/eligibility
  /// checks. Allocation and locking failure must not fail a validated read.
  bool try_seed_route(std::string_view route, std::weak_ptr<const void> owner) const noexcept;

 private:
  mutable std::mutex _route_mutex;
  mutable snapshot _route;
  mutable std::weak_ptr<const void> _route_owner;
};

bool route_reuse_enabled(config const& cfg) noexcept;
/// Longer Locations remain usable for one delegation, but are never stored.
inline constexpr std::size_t k_max_route_bytes = 2048;

/// Request-state part of the first-submission gate. The adapter additionally
/// requires a REST open with a strong validator before looking up its route.
/// Call only at the existing slot-owned, cancellation-checked submission hook.
bool first_submission_eligible(config const& cfg,
                               std::string_view delegated_url,
                               bool refreshed,
                               std::size_t attempt,
                               std::size_t auth_attempt) noexcept;

}  // namespace redirect_routes
}  // namespace sirius::io::rest
