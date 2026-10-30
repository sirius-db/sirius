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

#include <string>
#include <vector>

using sirius::io::rest::config;
using sirius::io::rest::redirect_admission::evaluate_cluster_redirect;

namespace {
config policy_config()
{
  config cfg;
  cfg.cluster_redirect.enabled         = true;
  cfg.cluster_redirect.origin          = "gateway:8080";
  cfg.cluster_redirect.allowed_targets = {"target:8081", "other:8082"};
  return cfg;
}
}  // namespace

TEST_CASE("redirect admission preserves the reference rule table", "[redirect_policy]")
{
  struct rule {
    char const* id;
    std::string source;
    bool delegated;
    long status;
    std::string location;
    std::string reason;
  };
  std::vector<rule> const rules{
    {"A01-allowed", "http://gateway:8080/b/key", false, 307, "http://target:8081/b/key", ""},
    {"A02-rewrite",
     "http://gateway:8080/b/key",
     false,
     307,
     "http://target:8081/s3/b//key%2F..?pid=fixture&utm=1",
     ""},
    {"A03-case", "HTTP://GATEWAY:8080/b/key", false, 307, "hTtP://TARGET:8081/b/key", ""},
    {"A04-upgrade", "http://gateway:8080/b/key", false, 307, "https://other:8082/b/key", ""},
    {"A05-same-https", "https://gateway:8080/b/key", false, 307, "https://target:8081/b/key", ""},
    {"A06-disallowed",
     "http://gateway:8080/b/key",
     false,
     307,
     "http://elsewhere/b/key",
     "Location names a storage node outside allowed_targets"},
    {"A07-missing", "http://gateway:8080/b/key", false, 307, "", "307 without Location"},
    {"A08-relative",
     "http://gateway:8080/b/key",
     false,
     307,
     "/s3/b/key",
     "Location is not an absolute http(s) URL"},
    {"A09-protocol-relative",
     "http://gateway:8080/b/key",
     false,
     307,
     "//target:8081/b/key",
     "Location is not an absolute http(s) URL"},
    {"A10-other-scheme",
     "http://gateway:8080/b/key",
     false,
     307,
     "ftp://target:8081/b/key",
     "Location is not an absolute http(s) URL"},
    {"A11-empty-authority",
     "http://gateway:8080/b/key",
     false,
     307,
     "http:///b/key",
     "Location is not an absolute http(s) URL"},
    {"A12-userinfo",
     "http://gateway:8080/b/key",
     false,
     307,
     "http://u@target:8081/b/key",
     "Location carries userinfo or a fragment"},
    {"A13-fragment",
     "http://gateway:8080/b/key",
     false,
     307,
     "http://target:8081/b/key#",
     "Location carries userinfo or a fragment"},
    {"A14-downgrade",
     "https://gateway:8080/b/key",
     false,
     307,
     "http://target:8081/b/key",
     "Location downgrades https to http"},
    {"A15-second-hop",
     "http://target:8081/b/key",
     true,
     307,
     "http://other:8082/b/key",
     "the storage node redirected again"},
    {"A16-actual-source",
     "http://target:8081/b/key",
     false,
     307,
     "http://target:8081/b/key",
     "the request did not go to the configured cluster origin"},
    {"A17-relative-source",
     "/b/key",
     false,
     307,
     "http://target:8081/b/key",
     "the request did not go to the configured cluster origin"},
    {"A18-at-in-path", "http://gateway:8080/b/key", false, 307, "http://target:8081/b/@key", ""},
    {"A19-escaped-fragment",
     "http://gateway:8080/b/key",
     false,
     307,
     "http://target:8081/b/%23",
     ""},
  };
  for (auto const& row : rules) {
    DYNAMIC_SECTION(row.id)
    {
      auto const cfg = policy_config();
      auto const decision =
        evaluate_cluster_redirect(cfg, row.source, row.delegated, row.status, row.location);
      CHECK(decision.follow == row.reason.empty());
      CHECK(decision.reason == row.reason);
    }
  }
}

TEST_CASE("redirect admission preserves rejection precedence", "[redirect_policy]")
{
  auto cfg                     = policy_config();
  cfg.cluster_redirect.enabled = false;
  CHECK(evaluate_cluster_redirect(cfg, "bad", true, 302, "").reason ==
        "redirects are not followed");
  cfg.cluster_redirect.enabled = true;
  for (long status : {0L, 200L, 301L, 302L, 303L, 304L, 308L, 399L, 401L, 403L, 999L}) {
    CAPTURE(status);
    auto const decision = evaluate_cluster_redirect(cfg, "bad", true, status, "");
    CHECK_FALSE(decision.follow);
    CHECK(decision.reason == "only 307 is delegated");
  }
  CHECK(evaluate_cluster_redirect(cfg, "bad", true, 307, "").reason ==
        "the storage node redirected again");
  CHECK(evaluate_cluster_redirect(cfg, "bad", false, 307, "").reason == "307 without Location");
  CHECK(evaluate_cluster_redirect(cfg, "bad", false, 307, "bad").reason ==
        "the request did not go to the configured cluster origin");
  CHECK(evaluate_cluster_redirect(cfg, "https://gateway:8080", false, 307, "bad \n#").reason ==
        "Location is not an absolute http(s) URL");
  CHECK(evaluate_cluster_redirect(cfg, "https://gateway:8080", false, 307, "http://u@unknown/x #")
          .reason == "Location contains whitespace or control bytes");
  CHECK(evaluate_cluster_redirect(cfg, "https://gateway:8080", false, 307, "http://u@unknown/x#")
          .reason == "Location carries userinfo or a fragment");
  CHECK(
    evaluate_cluster_redirect(cfg, "https://gateway:8080", false, 307, "http://unknown/x").reason ==
    "Location downgrades https to http");
}

TEST_CASE("redirect admission rejects every control byte without rewriting Location",
          "[redirect_policy]")
{
  auto const cfg = policy_config();
  for (int byte = 0; byte <= 0x7f; ++byte) {
    if (byte > 0x20 && byte != 0x7f) { continue; }
    DYNAMIC_SECTION("A20-control-" << byte)
    {
      std::string location = "http://target:8081/s3/b/key?pid=1&utm=";
      location.push_back(static_cast<char>(byte));
      auto const before = location;
      auto const decision =
        evaluate_cluster_redirect(cfg, "http://gateway:8080", false, 307, location);
      CHECK_FALSE(decision.follow);
      CHECK(decision.reason == "Location contains whitespace or control bytes");
      CHECK(location == before);
    }
  }
}

TEST_CASE("redirect admission keeps literal ports and borrowed URL components", "[redirect_policy]")
{
  auto cfg                             = policy_config();
  cfg.cluster_redirect.origin          = "gateway";
  cfg.cluster_redirect.allowed_targets = {"target"};
  CHECK(evaluate_cluster_redirect(cfg, "http://gateway/x", false, 307, "http://target/x").follow);
  CHECK(
    evaluate_cluster_redirect(cfg, "http://gateway:80/x", false, 307, "http://target/x").reason ==
    "the request did not go to the configured cluster origin");
  CHECK(
    evaluate_cluster_redirect(cfg, "http://gateway/x", false, 307, "http://target:80/x").reason ==
    "Location names a storage node outside allowed_targets");
  cfg.cluster_redirect.allowed_targets.clear();
  CHECK_FALSE(
    evaluate_cluster_redirect(cfg, "http://gateway/x", false, 307, "http://target/x").follow);
  cfg.cluster_redirect.origin          = "[::1]:8080";
  cfg.cluster_redirect.allowed_targets = {"[::1]:8081"};
  CHECK(evaluate_cluster_redirect(cfg, "http://[::1]:8080/x", false, 307, "http://[::1]:8081/x")
          .follow);

  std::string const location = "HtTp://TARGET:8081/s3/b//%2F?pid=fixture&utm=1";
  auto const parsed = sirius::io::rest::redirect_admission::detail::split_authority(location);
  REQUIRE(parsed);
  CHECK(parsed->scheme == "HtTp");
  CHECK(parsed->authority == "TARGET:8081");
  CHECK(parsed->scheme.data() == location.data());
  CHECK(parsed->authority.data() == location.data() + 7);
}
