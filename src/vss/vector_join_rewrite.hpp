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

#include "duckdb/common/unique_ptr.hpp"

#include <cstddef>

namespace duckdb {
class Binder;
class ClientContext;
class LogicalOperator;
}  // namespace duckdb

namespace sirius::vss {

/**
 * @brief Turn plain-SQL vector joins in an optimized DuckDB plan into the vector-join operator.
 *
 * A threshold join written as SQL -- `FROM q, r WHERE array_cosine_similarity(q.v, r.v) >= s`,
 * or the same with array_distance / array_cosine_distance -- reaches the optimizer as an
 * ANY_JOIN on that predicate, or as a FILTER over a join tree the probe relation is crossed into.
 * Either way DuckDB evaluates the predicate on every pair. The rewrite replaces it with
 * sirius_knn_join_rel over the two sides as child relations: the probe relation, and the corpus
 * as whatever relational subtree DuckDB already built around it, so any join or filter on the
 * corpus runs before the vector join rather than after it. Columns above the join are remapped to
 * the operator's outputs and the distance expression to its score. Plans the rewrite cannot prove
 * equivalent are left unchanged.
 *
 * @param inlined_ctes set when a materialized CTE carrying vectors was put back in place to expose
 *        the join; with nothing rewritten the caller should keep its own copy of the plan
 * @return how many joins were rewritten
 */
std::size_t rewrite_plain_sql_vector_joins(duckdb::ClientContext& context,
                                           duckdb::Binder& binder,
                                           duckdb::unique_ptr<duckdb::LogicalOperator>& plan,
                                           bool* inlined_ctes = nullptr);

/// Whether @p plan computes a vector distance anywhere: the only plans the rewrite can change.
bool plan_has_vector_distance(duckdb::LogicalOperator& plan);

}  // namespace sirius::vss
