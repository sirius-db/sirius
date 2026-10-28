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
}  // namespace sirius::io::rest
