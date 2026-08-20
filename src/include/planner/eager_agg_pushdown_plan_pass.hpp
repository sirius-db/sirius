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
/// directly on an equi-join and every aggregate input comes from one join side,
/// pre-aggregate that side by its join keys below the join and combine the
/// partial results above it. The join then consumes one row per distinct key
/// instead of one row per input row (TPC-H q13: the customer⋈orders join input
/// shrinks from |filtered orders| rows to |distinct custkeys| partial counts).
///
/// Returns a REWRITTEN COPY of @p plan when at least one provable candidate was
/// found and rewritten, nullptr otherwise. @p plan itself is never modified, so
/// the caller can fall back to it if the rewritten plan fails any later
/// planning stage (see sirius_physical_plan_generator::create_plan).
///
/// Correctness gates (all provable at plan time, fail closed — see the .cpp
/// header comment for the full soundness argument):
///   - the aggregate sits on the join directly or through ONE pure
///     pass-through projection (every slot a plain column ref — DuckDB's
///     column pruning inserts one on some shapes); references are traced
///     through it;
///   - single grouping set, no GROUPING() calls, groups are plain column refs
///     that do not touch the pushed side;
///   - every aggregate is a single-column-ref COUNT / SUM / SUM_NO_OVERFLOW /
///     MIN / MAX without DISTINCT / FILTER / ORDER BY, and every aggregate
///     input comes from the pushed side;
///   - the join is a plain comparison join (INNER, or LEFT/RIGHT pushing into
///     the non-preserved side) whose conditions are all `=` with a plain
///     column ref on the pushed side and no residual predicate;
///   - COUNT's 0-vs-NULL mismatch on outer joins is repaired with a
///     COALESCE(combined, 0) projection above the aggregate, and any
///     combine-type widening (SUM over BIGINT partials returns HUGEINT) is
///     cast back so the plan's output schema is byte-identical.
///
/// Benefit gate (heuristic only — never affects correctness): the rewrite is
/// applied only when the non-pushed side is a bare, unfiltered table scan
/// (modulo projections), i.e. the join is not expected to discard most of the
/// pushed side's rows. When optimizer cardinality estimates are present on the
/// join they refine this decision (estimated join output must be at least
/// SIRIUS_EAGER_AGG_MIN_RATIO — default 0.5 — of the pushed side's input).
///
/// Environment switches (read per call, so tests can A/B in-process):
///   SIRIUS_EAGER_AGG_PUSHDOWN=0   kill switch, pass never fires
///   SIRIUS_EAGER_AGG_FORCE=1      bypass the BENEFIT gate (correctness gates
///                                 always apply)
///   SIRIUS_EAGER_AGG_MIN_RATIO=x  estimate-ratio threshold (default 0.5)
[[nodiscard]] duckdb::unique_ptr<duckdb::LogicalOperator> try_eager_aggregation_pushdown(
  duckdb::LogicalOperator& plan, duckdb::ClientContext& context);

/// Process-wide count of applied rewrites; lets tests assert the pass actually
/// fired (a silently-refused rewrite would make GPU-vs-CPU tests vacuous).
[[nodiscard]] std::uint64_t eager_agg_pushdown_applied_count();

}  // namespace sirius::planner
