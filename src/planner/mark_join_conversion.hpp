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

#include <duckdb/common/common.hpp>
#include <duckdb/common/unique_ptr.hpp>
#include <duckdb/planner/logical_operator.hpp>

#include <cstddef>

namespace duckdb {
class ClientContext;
}

namespace sirius::planner {

/// Turn `FILTER(mark)` over a MARK join into a SEMI join, and `FILTER(NOT mark)` into an ANTI join,
/// on Sirius's own copy of the logical plan (never the plan DuckDB executes on fallback).
///
/// Why: with `delim_join_as_cte = true`, DuckDB wraps joins in identity projections that still
/// reference the mark column when its filter pushdown runs, so its own MARK-to-SEMI/ANTI conversion
/// is switched off (`convert_mark_to_semi = false`). Column pruning drops those references later,
/// but nothing re-enables the conversion. A MARK join cannot publish a dynamic filter; a SEMI join
/// can, because the probe rows such a filter drops have no match and SEMI drops them anyway.
///
/// An ANTI join cannot publish: it returns exactly the probe rows without a match. Converting it
/// still spares the GPU a mark for every probe row and the filter over it.
///
/// DuckDB picks build sides while the join is still MARK, which has no flipped form, so a converted
/// join builds on its subquery and its filter reaches only the outer rows. The pass therefore makes
/// that choice itself. It flips the join to RIGHT_SEMI or RIGHT_ANTI, which builds on the outer
/// rows and publishes their keys into the subquery's scans, when:
///   - a right-family join can publish (`right_family_publication_enabled`);
///   - every condition is the same kind of equality (all `=` or all `IS NOT DISTINCT FROM`) and one
///     compares INTEGER or BIGINT columns, the keys early publication takes, so that the flipped
///     join is a plain hash join (a right-family join has no nested-loop fallback);
///   - the outer side has filter or semi-join build evidence and reads no CTE (a CTE read there is
///     the delim rewrite's join-back, whose outer-row read must stay reachable from joins above);
///   - the outer side's estimated row count is at most a third of the subquery's, the ratio
///     DuckDB's build-side optimizer uses for semi joins.
/// A flipped join keeps its keys as they are.
///
/// The rules are DuckDB's own (`FilterPushdown::PushdownMarkJoin`):
///   - the filter is exactly the mark column, or exactly NOT(mark column);
///   - NOT(mark) needs every comparison to be null-safe, so the mark is never NULL;
///   - nothing else in the plan reads the mark column;
/// plus a positional-consumer guard: every operator between the filter and the next projection or
/// aggregate must address its input by binding, so dropping the (last) mark column shifts nothing.
///
/// Acts only when `delim_join_as_cte` is true, the setting under which the CTE rewrite adds those
/// projections, whether or not the delim CTE is still in the plan; under `false` the plan is
/// returned untouched.
///
/// Must run before column-binding resolution. Returns the number of joins converted.
std::size_t convert_mark_joins(duckdb::ClientContext& context,
                               duckdb::unique_ptr<duckdb::LogicalOperator>& plan);

}  // namespace sirius::planner
