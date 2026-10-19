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

#include "duckdb/common/types/value.hpp"
#include "duckdb/function/function.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace sirius::vss {

enum class vector_join_mode : std::uint8_t {
  global_top_k,
  per_row_top_k,
  threshold,
};

/// - exact:      brute force, L2 computed Unexpanded, no GEMM.
/// - exact_gemm: brute force, L2 computed Expanded (with GEMM). Exact for
///               normalized / moderate-magnitude vectors.
/// - approx:     not implemented yet
enum class vector_join_search_mode : std::uint8_t {
  exact,
  exact_gemm,
  approx,
};

/// The type of score to emit for each result pair, and which value space the join
/// selects/thresholds in.
/// - similarity: higher is closer. For cosine this is the inner product
///   (1 - distance). Threshold semantics are score >= eps.
/// - distance:   lower is closer. The natural output for L2. Threshold semantics
///   are score <= eps.
enum class vector_join_output_type : std::uint8_t {
  similarity,
  distance,
};

struct vector_join_side {
  std::string catalog;                      ///< resolved catalog of the pinned table
  std::string schema;                       ///< resolved schema
  std::string table;                        ///< base table
  std::string column;                       ///< vector column
  std::vector<std::string> output_columns;  ///< base-table columns to emit in order
  /// Where each of @c output_columns sits in the side's batches. Empty: column c at c + 1, right
  /// after the vector column, as a side's own scan lays them out.
  std::vector<std::size_t> output_positions;
  /// A VIEW (named subquery) rather than a base table; only valid on a scanned corpus side.
  /// The planner binds and streams it, and nothing looks for a pin.
  bool is_view{false};
  /// The side is a relation the join takes as a child plan (the probe of sirius_knn_join_rel, or
  /// a corpus the plain-SQL rewrite built), so there is no table behind it: its columns are
  /// @c relation_columns, positionally, and @c column / @c output_columns name among them.
  bool from_relation{false};
  std::vector<std::string> relation_columns;
};

/// `column <op> constant` on a corpus column, taken from a filter over the join's output that
/// DuckDB offered the table function. Evaluated against the corpus before it is searched.
struct corpus_predicate {
  enum class op : std::uint8_t { eq, ne, lt, le, gt, ge };
  std::string column;
  op cmp{op::eq};
  duckdb::Value value;
};

struct vector_join_request {
  vector_join_side left;
  vector_join_side right;
  vector_join_mode mode;
  std::string metric;  ///< distance metric
  vector_join_search_mode search_mode{vector_join_search_mode::exact_gemm};
  std::int64_t k{0};           ///< top-k
  std::int64_t n_clusters{0};  ///< number of clusters of k-means cluster
  std::int64_t n_probes{1};    ///< nearest clusters each point is assigned to
  std::int64_t dim{0};         ///< vector dimensionality
  double eps{0.0};             ///< distance/similarity threshold
  vector_join_output_type output_type{vector_join_output_type::distance};
  /// Whether anything reads the score. Set by the planner from the columns the query reads; a
  /// threshold join that no one reads the distance of only has to decide membership.
  bool score_read{true};
  /// The statement only counts the pairs: `SELECT count(*)` directly over a threshold join. The
  /// operator then counts inside the search and emits one BIGINT row per probe batch instead of
  /// the pairs, which past a few hundred million would be written to host memory only to be
  /// counted. Set by push_count_into_vector_joins; the planner then reads no column and no score.
  bool count_only{false};
  /// `keep => 'best'`: a threshold join that emits, per probe row, only its closest pair within
  /// the threshold. The operator folds the pairs into one per row as it finds them, so a row with
  /// thousands of pairs costs neither their exact re-scoring nor an aggregate over them.
  bool best_only{false};
  /// `earlier_by => '<column>'` on a threshold join: only pairs whose right row has a smaller
  /// value of this column than the left row count, which over a self-join is each pair once.
  /// Both tables carry the column, and both are pinned.
  std::string earlier_by;
  /// `other_by => '<column>'` instead: the column (held in earlier_by) only has to differ, which
  /// in a self-join on a unique column is "any row but itself".
  bool key_differs{false};
  /// The column is `rowid`: a self-join ordered by the rows' positions in the pinned table, which
  /// needs no column and is the order the cluster lists already keep.
  bool key_is_rowid{false};
  /// Take the corpus from a child scan materialized by the build phase instead of from a
  /// pinned catalog table. Temporary A/B switch (`build_source => 'scan'`) while the build
  /// phase is verified against the pin path; the SQL surface that replaces it is the
  /// LATERAL-rewrite rule, which will set this unconditionally.
  bool build_from_scan{false};
  /// Same for the probe side (`probe_source => 'scan'`). Independent of build_from_scan so each
  /// side can be A/B'd against its pinned path on its own.
  bool probe_from_scan{false};
  /// Set by the plan generator when both sides scan the same table's vector column: the probe
  /// reads the corpus side's batches instead of a scan and a copy of its own.
  bool probe_shares_build{false};
  /// Name of a clustering trained by `sirius_kmeans_fit`, shared by both sides. Empty means an
  /// exhaustive join. When set, each probe batch visits only the corpus chunks holding clusters
  /// its rows were assigned to, which is what makes the join approximate.
  std::string clustering;
  /// Column on the corpus table holding each row's cluster id, as emitted by
  /// `sirius_kmeans_assign`. Pruning is only effective when the corpus is stored in cluster
  /// order, since a chunk is skipped on its [min, max] cluster range.
  std::string build_cluster_column;
  /// Conjuncts over right-side output columns pushed into the join. Only ever set where
  /// filtering the corpus first gives the same pairs as filtering the join's output: a
  /// threshold join, whose pairs are independent of every other corpus row.
  std::vector<corpus_predicate> right_predicates;
  /// The probe relation replaces a scalar subquery (the plain-SQL rewrite unwrapped it), so it
  /// must hold exactly one row, as the subquery had to.
  bool probe_scalar{false};
};

struct SiriusVectorJoinBindData : public duckdb::TableFunctionData {
  vector_join_request req;
  /// Pinned row counts of both sides, captured at bind time so the cardinality
  /// callback needs no catalog access. Zero if the side resolved to no pin.
  std::uint64_t left_rows{0};
  std::uint64_t right_rows{0};
  /// True for the relational surface, whose probe is a bound subquery. Its row count is the
  /// child's and is unknown here, which is why the cardinality callback declines for it.
  bool probe_is_relation{false};
};

}  // namespace sirius::vss
