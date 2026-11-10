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

namespace duckdb {
class LogicalComparisonJoin;
}  // namespace duckdb

namespace sirius::planner {

/**
 * @brief Removes scan filters on @p join's keys that the join implies and that keep nearly every
 * row
 *
 * DuckDB narrows each side of an INNER or SEMI equality join to the other side's key range, e.g.
 * TPC-H q18's `c_custkey <= 149999999`, however few rows that removes. The join drops those rows
 * anyway, and the filter makes the build side look selective to the dynamic-filter wiring. A
 * filter goes when the other scan's Parquet column statistics prove it passes every key the other
 * side can supply, and its own column's statistics show it keeps at least 99% of the key range,
 * which keeps about as many of the rows when the keys are unique or evenly spread.
 *
 * @pre @p join still owns both logical children, with resolved types and positional references.
 */
void drop_join_implied_filters(duckdb::LogicalComparisonJoin& join);

}  // namespace sirius::planner
