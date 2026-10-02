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

#include <duckdb/common/unique_ptr.hpp>

#include <cstdint>

namespace duckdb {
class ClientContext;
class LogicalOperator;
}  // namespace duckdb

namespace sirius::planner {

/// Eager aggregation pushdown (Yan & Larson): when a grouped aggregate sits
/// on an equi-join — directly or through one pure pass-through projection — and
/// every aggregate input comes from one join side,
/// pre-aggregate that side by its join keys below the join and combine the
/// partial results above it. The join then consumes one row per distinct key
/// instead of one row per input row (TPC-H q13: the customer⋈orders join input
/// shrinks from |filtered orders| rows to |distinct custkeys| partial counts).
///
/// Returns a REWRITTEN COPY of @p plan in `plan` when at least one provable
/// candidate was found and rewritten, nullptr otherwise. @p plan itself is never
/// modified, so the caller can fall back to it if the rewritten plan fails any
/// later planning stage (see sirius_physical_plan_generator::create_plan).
///
/// Correctness gates (all provable at plan time, fail closed — see the .cpp
/// header comment for the full soundness argument):
///   - the aggregate sits on the join directly or through ONE pure
///     pass-through projection (every slot a plain column ref — DuckDB's
///     column pruning inserts one on some shapes); references are traced
///     through it;
///   - single grouping set, no GROUPING() calls, groups are plain column refs
///     that do not touch the pushed side;
///   - every aggregate is a single-column-ref COUNT / SUM /
///     MIN / MAX without DISTINCT / FILTER / ORDER BY, and every aggregate
///     input comes from the pushed side. SUM over FLOAT / DOUBLE is excluded:
///     floating-point addition is not associative, so summing per-key partials
///     is not value-identical to the single-pass sum the original plan
///     computes;
///   - the join is a plain comparison join (INNER, or LEFT/RIGHT pushing into
///     the non-preserved side) whose conditions are all `=` with a plain
///     column ref on the pushed side and no residual predicate;
///   - COUNT's 0-vs-NULL mismatch on outer joins is repaired with a
///     COALESCE(combined, 0) projection above the aggregate, and any
///     combine-type widening (SUM over BIGINT partials returns HUGEINT) is
///     cast back so the plan's output schema is byte-identical.
///
/// Benefit gate (heuristic only — never affects correctness), two independent
/// refusals:
///   - the non-pushed side must be a bare, unfiltered table scan (modulo
///     projections), i.e. the join is not expected to discard most of the
///     pushed side's rows;
///   - the pushed side's join keys must not be PROVABLY unique (a primary key
///     of the scanned base table). A unique key makes the lower aggregate emit
///     one row per input row, so it pays a full partition + merge for no
///     reduction at all. Only provable uniqueness is used: no distinct-count
///     statistics are available at this point in planning, so a merely
///     near-unique key is not caught.
///
/// Settings (both read from the active connection's Sirius operator params, so
/// they are visible in `current_setting` and settable per connection):
///   enable_eager_agg_pushdown = false   kill switch, pass never fires
///   eager_agg_pushdown_force  = true    TEST ONLY (requires
///                                       SIRIUS_ENABLE_TEST_OPTIONS): bypass
///                                       the BENEFIT gate; the correctness
///                                       gates above always apply
struct eager_agg_pushdown_result {
  /// Rewritten copy of the input plan, or nullptr when the pass refused.
  duckdb::unique_ptr<duckdb::LogicalOperator> plan;
  /// How many candidates `plan` contains rewrites for. The caller records this
  /// only once the rewritten plan has survived every later planning stage, so
  /// "applied" always means "a rewritten plan was executed".
  std::uint64_t applied = 0;
};

[[nodiscard]] eager_agg_pushdown_result try_eager_aggregation_pushdown(
  duckdb::LogicalOperator& plan, duckdb::ClientContext& context);

}  // namespace sirius::planner
