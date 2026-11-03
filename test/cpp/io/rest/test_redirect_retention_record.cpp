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

#include "io/rest/config.hpp"
#include "io/rest/redirect_policy.hpp"

#include <catch.hpp>

#include <array>
#include <format>
#include <limits>
#include <type_traits>

namespace policy = sirius::io::rest::redirect_retention_policy;
using sirius::io::rest::redirect_retention;

TEST_CASE("retention guards and cap arithmetic preserve the mathematical allowance",
          "[redirect_retention_policy]")
{
  constexpr auto max = std::numeric_limits<std::size_t>::max();
  std::array<std::size_t, 14> const capacities{
    0, 1, 2, 3, 63, 64, 2047, 2048, 4094, 4095, 4096, 4097, max - 1, max};
  std::array<std::size_t, 14> const targets{
    0, 1, 2, 3, 63, 64, 2047, 2048, 4094, 4095, 4096, 4097, max - 1, max};
  for (bool enabled : {false, true}) {
    for (auto mode : {redirect_retention::legacy, redirect_retention::expanded}) {
      for (auto c : capacities) {
        for (auto t : targets) {
          CAPTURE(enabled, static_cast<int>(mode), c, t);
          std::size_t expected = c;
          if (enabled && mode == redirect_retention::expanded && c < 4096) {
            // Independent wider arithmetic: even SIZE_MAX + 1 and its product
            // with the guarded C fit, so the expected value does not wrap.
            unsigned __int128 product =
              static_cast<unsigned __int128>(c) * (static_cast<unsigned __int128>(t) + 1);
            expected = product > 4096 ? 4096 : static_cast<std::size_t>(product);
          }
          CHECK(policy::connection_retention(enabled, mode, c, t) == expected);
        }
      }
    }
  }
}

TEST_CASE("retention config keeps requested mode separate from effective allowance",
          "[redirect_retention_policy]")
{
  sirius::io::rest::config cfg;
  cfg.max_connections                  = 2;
  cfg.cluster_redirect.allowed_targets = {"a", "b"};
  cfg.cluster_redirect.retention       = redirect_retention::expanded;
  CHECK(std::string_view(policy::retention_name(cfg)) == "expanded");
  CHECK(policy::connection_retention(cfg) == 2);
  cfg.cluster_redirect.enabled = true;
  CHECK(policy::connection_retention(cfg) == 6);
  cfg.cluster_redirect.retention = redirect_retention::legacy;
  CHECK(std::string_view(policy::retention_name(cfg)) == "legacy");
  CHECK(policy::connection_retention(cfg) == 2);
  cfg.cluster_redirect.retention = redirect_retention::expanded;
  cfg.max_connections            = 4096;
  CHECK(std::string_view(policy::retention_name(cfg)) == "expanded");
  CHECK(policy::connection_retention(cfg) == 4096);
}

TEST_CASE("transfer representation preserves field order and unknown observations",
          "[redirect_transfer_record]")
{
  using sirius::io::rest::redirect_records::transfer_record;
  static_assert(std::is_trivially_copyable_v<transfer_record>);
  static_assert(std::is_aggregate_v<transfer_record>);
  auto render = [](transfer_record const& r) {
    return std::format(transfer_record::format,
                       r.method,
                       r.authority,
                       r.range,
                       r.status,
                       r.outcome,
                       r.bytes,
                       r.attempt,
                       r.delegated,
                       r.elapsed_ms,
                       r.route,
                       r.connects);
  };
  transfer_record record{
    "GET", "target:8080", "bytes=13-28", 206, "ok", 16, 2, true, 7, "cached", "0"};
  CHECK(
    render(record) ==
    "rest_reactor: transfer method=GET authority=target:8080 range=bytes=13-28 "
    "status=206 curl=ok bytes=16 attempt=2 delegated=true elapsed_ms=7 route=cached connects=0");
  record = {"GET", "target:8080", "bytes=13-28", 0, "aborted-local", 5, 0, true, 9, "fresh", "-"};
  CHECK(render(record) ==
        "rest_reactor: transfer method=GET authority=target:8080 range=bytes=13-28 "
        "status=0 curl=aborted-local bytes=5 attempt=0 delegated=true elapsed_ms=9 route=fresh "
        "connects=-");
  record = {"HEAD", "-", "-", 200, "ok", 0, 0, false, 0, "origin", "1"};
  CHECK(render(record) ==
        "rest_reactor: transfer method=HEAD authority=- range=- status=200 curl=ok "
        "bytes=0 attempt=0 delegated=false elapsed_ms=0 route=origin connects=1");
}
