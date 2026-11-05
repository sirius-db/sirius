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

// A plan built in one statement and run in another: a prepared statement's validated plan.

#include "sirius_context.hpp"

#include <catch.hpp>
#include <duckdb.hpp>
#include <utils/sirius_test_env.hpp>

using window = duckdb::SiriusContext::StandaloneQueryScope;

TEST_CASE("a distributed query planned in an earlier statement", "[exchange][query_windows]")
{
  REQUIRE(sirius::test::g_integration_env != nullptr);
  if (!sirius::test::g_integration_env->is_active()) { sirius::test::g_integration_env->resume(); }
  auto con = sirius::test::g_integration_env->make_connection();
  auto ctx = con.context->registered_state->Get<duckdb::SiriusContext>("sirius_state");
  REQUIRE(ctx);
  auto const planned = ctx->next_distributed_query(*con.context);
  // The next statement starts: what the last one planned is no longer pending.
  REQUIRE_FALSE(con.Query("SELECT 42")->HasError());

  SECTION("is not adopted by the next statement's window")
  {
    window w(*ctx, *con.context, "next_statement");
    CHECK(ctx->distributed_query_of(w.query_id()) == 0);
  }

  SECTION("is adopted by the window that reuses its plan")
  {
    sirius::query_id_t id{};
    {
      window w(*ctx, *con.context, "reuse");
      id = w.query_id();
      ctx->adopt_distributed_query(*con.context, planned);
      CHECK(ctx->distributed_query_of(id) == planned);
    }
    CHECK(ctx->distributed_query_of(id) == 0);
  }
}
