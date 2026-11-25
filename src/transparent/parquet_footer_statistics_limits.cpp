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

#include "sirius_context.hpp"
#include "transparent/parquet_footer_statistics.hpp"

#include <duckdb/common/types/value.hpp>
#include <duckdb/main/client_context.hpp>

#include <cstdint>

// Apart from parquet_footer_statistics.cpp: the engine's headers reach <linux/fs.h>, whose
// BLOCK_SIZE macro breaks DuckDB's Parquet headers.
namespace sirius::transparent {

namespace {

std::int64_t limit_of(duckdb::ClientContext& context, char const* setting, std::int64_t engine)
{
  duckdb::Value value;
  if (context.TryGetCurrentSetting(setting, value) && !value.IsNull()) {
    return value.GetValue<int64_t>();
  }
  return engine;
}

}  // namespace

footer_read_limits footer_read_limits_of(duckdb::ClientContext& context)
{
  auto const limits = [&](operator_params const& engine) {
    return footer_read_limits{limit_of(context,
                                       "parquet_footer_statistics_max_remote_reads",
                                       engine.parquet_footer_statistics_max_remote_reads),
                              limit_of(context,
                                       "parquet_footer_statistics_max_local_reads",
                                       engine.parquet_footer_statistics_max_local_reads)};
  };
  if (auto const runtime = context.registered_state->Get<duckdb::SiriusContext>("sirius_state")) {
    return limits(runtime->get_config().get_operator_params());
  }
  return limits(operator_params{});
}

}  // namespace sirius::transparent
