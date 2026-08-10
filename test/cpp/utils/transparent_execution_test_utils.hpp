/*
 * Copyright 2025, Sirius Contributors.
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

#include "catch.hpp"
#include "sirius_context.hpp"

#include <duckdb.hpp>

namespace sirius::test {

inline duckdb::shared_ptr<duckdb::SiriusContext> get_registered_sirius_context(
  duckdb::Connection& con)
{
  auto sirius_ctx = con.context->registered_state->Get<duckdb::SiriusContext>("sirius_state");
  REQUIRE(sirius_ctx != nullptr);
  return sirius_ctx;
}

inline duckdb::SiriusContext::transparent_execution_stats get_transparent_execution_stats(
  duckdb::Connection& con)
{
  return get_registered_sirius_context(con)->get_transparent_execution_stats();
}

inline duckdb::SiriusContext::compressed_materialization_stats get_compressed_materialization_stats(
  duckdb::Connection& con)
{
  return get_registered_sirius_context(con)->get_compressed_materialization_stats();
}

inline duckdb::SiriusContext::size_estimation_stats get_size_estimation_stats(
  duckdb::Connection& con)
{
  return get_registered_sirius_context(con)->get_size_estimation_stats();
}

inline void require_transparent_execution_delta(
  const duckdb::SiriusContext::transparent_execution_stats& before,
  const duckdb::SiriusContext::transparent_execution_stats& after,
  uint64_t expected_rebind_delta,
  uint64_t expected_fallback_delta,
  uint64_t expected_execution_delta,
  uint64_t expected_runtime_fallback_delta = 0)
{
  REQUIRE(after.successful_rebinds == before.successful_rebinds + expected_rebind_delta);
  REQUIRE(after.fallbacks == before.fallbacks + expected_fallback_delta);
  REQUIRE(after.executions == before.executions + expected_execution_delta);
  REQUIRE(after.runtime_fallbacks == before.runtime_fallbacks + expected_runtime_fallback_delta);
}

}  // namespace sirius::test
