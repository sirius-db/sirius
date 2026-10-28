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

#include "sirius/exception.hpp"
#include "sirius_context.hpp"

#include <catch.hpp>
#include <duckdb.hpp>
#include <utils/sirius_test_env.hpp>

#include <memory>

using window = duckdb::SiriusContext::StandaloneQueryScope;

TEST_CASE("distributed queries are numbered on one connection and tied to their window",
          "[exchange][query_windows]")
{
  REQUIRE(sirius::test::g_integration_env != nullptr);
  if (!sirius::test::g_integration_env->is_active()) { sirius::test::g_integration_env->resume(); }
  auto owner =
    std::make_unique<duckdb::Connection>(sirius::test::g_integration_env->make_connection());
  auto other =
    std::make_unique<duckdb::Connection>(sirius::test::g_integration_env->make_connection());
  auto ctx = owner->context->registered_state->Get<duckdb::SiriusContext>("sirius_state");
  REQUIRE(ctx);

  // Planned before its window opens, as transparent execution does: the window adopts it.
  auto const first = ctx->next_distributed_query(*owner->context);
  sirius::query_id_t first_window{};
  {
    window w(*ctx, *owner->context, "distributed_first");
    first_window = w.query_id();
    CHECK(ctx->distributed_query_of(first_window) == first);
  }
  CHECK(ctx->distributed_query_of(first_window) == 0);

  // Planned while its window is open: tied to that window at once.
  {
    window w(*ctx, *owner->context, "distributed_second");
    auto const second = ctx->next_distributed_query(*owner->context);
    CHECK(second == first + 1);
    CHECK(ctx->distributed_query_of(w.query_id()) == second);
  }

  // Another connection's statements would number differently on each rank.
  CHECK_THROWS_AS(ctx->next_distributed_query(*other->context), sirius::invalid_input_exception);

  // Closing the connection hands the role on; numbering continues where it stopped.
  owner.reset();
  auto const third = ctx->next_distributed_query(*other->context);
  CHECK(third == first + 2);
  auto later =
    std::make_unique<duckdb::Connection>(sirius::test::g_integration_env->make_connection());
  CHECK_THROWS_AS(ctx->next_distributed_query(*later->context), sirius::invalid_input_exception);
}
