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

#include <atomic>
#include <limits>
#include <semaphore>
#include <thread>

namespace routes = sirius::io::rest::redirect_routes;

TEST_CASE("route cache preserves open and live context scope", "[redirect_route_state]")
{
  routes::route_cache first;
  routes::route_cache second;
  auto owner   = std::make_shared<int>(1);
  auto foreign = std::make_shared<int>(2);
  auto grant   = std::make_shared<const std::string>("http://target/same-object");
  REQUIRE(first.seed_route(grant, owner));
  CHECK(first.route(owner) == grant);
  CHECK(second.route(owner) == nullptr);
  CHECK(first.route(foreign) == nullptr);
  CHECK_FALSE(first.seed_route(grant, foreign));
  CHECK(first.route(owner) == grant);

  owner.reset();
  CHECK(first.route(foreign) == nullptr);
  CHECK_FALSE(first.seed_route(grant, foreign));
  first.drop_route(grant);
  REQUIRE(first.seed_route(grant, foreign));
  CHECK(first.route(foreign) == grant);

  SECTION("ownerless publication remains occupied and invisible")
  {
    routes::route_cache cache;
    REQUIRE(cache.seed_route(grant, {}));
    CHECK(cache.route({}) == nullptr);
    CHECK(cache.route(foreign) == nullptr);
    CHECK_FALSE(cache.seed_route(grant, foreign));
  }
  SECTION("nonowning alias has no live owner")
  {
    routes::route_cache cache;
    int value = 3;
    std::shared_ptr<const void> nonowning(std::shared_ptr<int>{}, &value);
    REQUIRE(cache.seed_route(grant, nonowning));
    CHECK(cache.route(nonowning) == nullptr);
    CHECK_FALSE(cache.seed_route(grant, foreign));
  }
  SECTION("null publication preserves the original empty-slot behavior")
  {
    routes::route_cache cache;
    CHECK(cache.seed_route(nullptr, foreign));
    CHECK(cache.route(foreign) == nullptr);
    CHECK(cache.seed_route(grant, foreign));
    CHECK(cache.route(foreign) == grant);
  }
}

TEST_CASE("stale attempt cannot invalidate a newer equal-URL publication", "[redirect_route_state]")
{
  routes::route_cache cache;
  auto owner = std::make_shared<int>(1);
  auto older = std::make_shared<const std::string>("http://target/same-url");
  auto newer = std::make_shared<const std::string>(*older);
  REQUIRE(older != newer);
  REQUIRE(cache.seed_route(older, owner));
  std::binary_semaphore selected(0);
  std::binary_semaphore published(0);
  std::atomic<bool> selected_old{false};
  std::thread stale_attempt([&] {
    auto snapshot = cache.route(owner);
    selected_old  = snapshot == older;
    selected.release();
    published.acquire();
    cache.drop_route(snapshot);
  });
  selected.acquire();
  cache.drop_route(older);
  bool const did_publish = cache.seed_route(newer, owner);
  published.release();
  stale_attempt.join();
  CHECK(selected_old.load());
  CHECK(did_publish);
  CHECK(cache.route(owner) == newer);
  cache.drop_route(std::make_shared<const std::string>(*newer));
  CHECK(cache.route(owner) == newer);
  cache.drop_route(newer);
  CHECK(cache.route(owner) == nullptr);
}

TEST_CASE("route selection requires an untouched first submission", "[redirect_route_state]")
{
  using sirius::io::rest::redirect_route_reuse;
  for (unsigned bits = 0; bits < 64; ++bits) {
    DYNAMIC_SECTION("eligibility-mask=" << bits)
    {
      sirius::io::rest::config cfg;
      cfg.cluster_redirect.enabled = (bits & 1) != 0;
      cfg.cluster_redirect.route_reuse =
        (bits & 2) ? redirect_route_reuse::same_open : redirect_route_reuse::off;
      auto const delegated           = (bits & 4) ? "http://target/old" : "";
      bool const refreshed           = (bits & 8) != 0;
      std::size_t const attempt      = (bits & 16) ? 1 : 0;
      std::size_t const auth_attempt = (bits & 32) ? 1 : 0;
      CHECK(routes::first_submission_eligible(cfg, delegated, refreshed, attempt, auth_attempt) ==
            (bits == 3));
    }
  }
  sirius::io::rest::config cfg;
  cfg.cluster_redirect.enabled     = true;
  cfg.cluster_redirect.route_reuse = redirect_route_reuse::same_open;
  CHECK_FALSE(
    routes::first_submission_eligible(cfg, "", false, std::numeric_limits<std::size_t>::max(), 0));
  CHECK_FALSE(
    routes::first_submission_eligible(cfg, "", false, 0, std::numeric_limits<std::size_t>::max()));
}

TEST_CASE("optional publication preserves the first snapshot", "[redirect_route_state]")
{
  routes::route_cache cache;
  auto owner = std::make_shared<int>(1);
  REQUIRE(cache.try_seed_route("http://target/validated", owner));
  auto const selected = cache.route(owner);
  REQUIRE(selected != nullptr);
  CHECK(*selected == "http://target/validated");
  CHECK_FALSE(cache.try_seed_route("http://target/new", owner));
  CHECK(cache.route(owner) == selected);
  cache.drop_route(selected);
  CHECK(cache.try_seed_route("http://target/new", owner));
  CHECK(cache.route(owner) != selected);
  static_assert(noexcept(cache.try_seed_route("", {})));
}
