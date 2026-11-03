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

#include "io/rest/redirect_policy.hpp"

#include "io/rest/config.hpp"

#include <cctype>
#include <utility>

namespace sirius::io::rest::redirect_admission {
namespace {
char ascii_lower(char c) { return static_cast<char>(std::tolower(static_cast<unsigned char>(c))); }

/// ASCII case-insensitive equality.  Used for schemes and for `host[:port]`
/// authorities, where the port is compared literally: the configured spelling
/// has to match the server's, default ports are not normalized.
bool ascii_iequals(std::string_view a, std::string_view b) noexcept
{
  if (a.size() != b.size()) { return false; }
  for (size_t i = 0; i < a.size(); ++i) {
    if (ascii_lower(a[i]) != ascii_lower(b[i])) { return false; }
  }
  return true;
}

}  // namespace

namespace detail {
std::optional<url_authority> split_authority(std::string_view url) noexcept
{
  auto const sep = url.find("://");
  if (sep == std::string_view::npos) { return std::nullopt; }
  std::string_view const scheme = url.substr(0, sep);
  if (!ascii_iequals(scheme, "http") && !ascii_iequals(scheme, "https")) { return std::nullopt; }
  std::string_view const rest      = url.substr(sep + 3);
  auto const end                   = rest.find_first_of("/?#");
  std::string_view const authority = end == std::string_view::npos ? rest : rest.substr(0, end);
  if (authority.empty()) { return std::nullopt; }
  return url_authority{scheme, authority};
}

}  // namespace detail

using detail::split_authority;

redirect_decision evaluate_cluster_redirect(config const& cfg,
                                            std::string_view request_url,
                                            bool already_delegated,
                                            long status,
                                            std::string_view location)
{
  auto const& policy = cfg.cluster_redirect;
  if (!policy.enabled) { return {false, "redirects are not followed"}; }
  if (status != 307) { return {false, "only 307 is delegated"}; }
  if (already_delegated) { return {false, "the storage node redirected again"}; }
  if (location.empty()) { return {false, "307 without Location"}; }
  auto const from = split_authority(request_url);
  if (!from || !ascii_iequals(from->authority, policy.origin)) {
    return {false, "the request did not go to the configured cluster origin"};
  }
  auto const to = split_authority(location);
  if (!to) { return {false, "Location is not an absolute http(s) URL"}; }
  // The Location goes to curl as-is, so what curl would reject must not count
  // as a delegation: no whitespace or control bytes anywhere in it.
  for (unsigned char const c : location) {
    if (c <= 0x20 || c == 0x7f) { return {false, "Location contains whitespace or control bytes"}; }
  }
  if (to->authority.find('@') != std::string_view::npos ||
      location.find('#') != std::string_view::npos) {
    return {false, "Location carries userinfo or a fragment"};
  }
  if (ascii_iequals(from->scheme, "https") && ascii_iequals(to->scheme, "http")) {
    return {false, "Location downgrades https to http"};
  }
  for (auto const& allowed : policy.allowed_targets) {
    if (ascii_iequals(to->authority, allowed)) { return {true, {}}; }
  }
  return {false, "Location names a storage node outside allowed_targets"};
}

}  // namespace sirius::io::rest::redirect_admission

namespace sirius::io::rest::redirect_routes {

snapshot route_cache::route(std::shared_ptr<const void> const& owner) const
{
  std::lock_guard lock(_route_mutex);
  if (_route == nullptr) { return nullptr; }
  auto const live = _route_owner.lock();
  return live != nullptr && live == owner ? _route : nullptr;
}

bool route_cache::seed_route(snapshot route, std::weak_ptr<const void> owner) const
{
  std::lock_guard lock(_route_mutex);
  if (_route != nullptr) { return false; }
  _route       = std::move(route);
  _route_owner = std::move(owner);
  return true;
}

void route_cache::drop_route(snapshot const& stale) const
{
  std::lock_guard lock(_route_mutex);
  if (_route == stale) { _route.reset(); }
}

bool route_cache::try_seed_route(std::string_view route,
                                 std::weak_ptr<const void> owner) const noexcept
{
  try {
    return seed_route(std::make_shared<const std::string>(route), std::move(owner));
  } catch (...) {
    return false;
  }
}

bool route_reuse_enabled(config const& cfg) noexcept
{
  return cfg.cluster_redirect.enabled &&
         cfg.cluster_redirect.route_reuse == redirect_route_reuse::same_open;
}

bool first_submission_eligible(config const& cfg,
                               std::string_view delegated_url,
                               bool refreshed,
                               std::size_t attempt,
                               std::size_t auth_attempt) noexcept
{
  return route_reuse_enabled(cfg) && delegated_url.empty() && !refreshed && attempt == 0 &&
         auth_attempt == 0;
}

}  // namespace sirius::io::rest::redirect_routes

namespace sirius::io::rest::redirect_retention_policy {

std::size_t connection_retention(bool enabled,
                                 redirect_retention mode,
                                 std::size_t per_host,
                                 std::size_t targets) noexcept
{
  constexpr std::size_t cap = 4096;
  if (!enabled || mode == redirect_retention::legacy || per_host >= cap) { return per_host; }
  if (per_host == 0) { return 0; }
  // With positive C and T >= cap the mathematical product exceeds the cap.
  // This check also makes 1 + T safe without constructing an overflowing sum.
  if (targets >= cap) { return cap; }
  auto const authorities = targets + 1;
  if (per_host > cap / authorities) { return cap; }
  return per_host * authorities;
}

std::size_t connection_retention(config const& cfg) noexcept
{
  return connection_retention(cfg.cluster_redirect.enabled,
                              cfg.cluster_redirect.retention,
                              cfg.max_connections,
                              cfg.cluster_redirect.allowed_targets.size());
}

char const* retention_name(config const& cfg) noexcept
{
  return cfg.cluster_redirect.retention == redirect_retention::expanded ? "expanded" : "legacy";
}

}  // namespace sirius::io::rest::redirect_retention_policy
