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

#include "vss/vector_join_rewrite.hpp"

#include "duckdb/catalog/catalog.hpp"
#include "duckdb/catalog/catalog_entry/table_catalog_entry.hpp"
#include "duckdb/catalog/catalog_entry/table_function_catalog_entry.hpp"
#include "duckdb/common/types.hpp"
#include "duckdb/function/function_binder.hpp"
#include "duckdb/function/scalar/struct_utils.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/optimizer/column_binding_replacer.hpp"
#include "duckdb/optimizer/cte_inlining.hpp"
#include "duckdb/parallel/task_scheduler.hpp"
#include "duckdb/parser/keyword_helper.hpp"
#include "duckdb/planner/binder.hpp"
#include "duckdb/planner/column_binding_map.hpp"
#include "duckdb/planner/expression/bound_aggregate_expression.hpp"
#include "duckdb/planner/expression/bound_between_expression.hpp"
#include "duckdb/planner/expression/bound_case_expression.hpp"
#include "duckdb/planner/expression/bound_cast_expression.hpp"
#include "duckdb/planner/expression/bound_columnref_expression.hpp"
#include "duckdb/planner/expression/bound_comparison_expression.hpp"
#include "duckdb/planner/expression/bound_conjunction_expression.hpp"
#include "duckdb/planner/expression/bound_constant_expression.hpp"
#include "duckdb/planner/expression/bound_function_expression.hpp"
#include "duckdb/planner/expression/bound_unnest_expression.hpp"
#include "duckdb/planner/expression_iterator.hpp"
#include "duckdb/planner/filter/conjunction_filter.hpp"
#include "duckdb/planner/filter/constant_filter.hpp"
#include "duckdb/planner/logical_operator_deep_copy.hpp"
#include "duckdb/planner/logical_operator_visitor.hpp"
#include "duckdb/planner/operator/logical_aggregate.hpp"
#include "duckdb/planner/operator/logical_any_join.hpp"
#include "duckdb/planner/operator/logical_comparison_join.hpp"
#include "duckdb/planner/operator/logical_cross_product.hpp"
#include "duckdb/planner/operator/logical_cte.hpp"
#include "duckdb/planner/operator/logical_cteref.hpp"
#include "duckdb/planner/operator/logical_delim_get.hpp"
#include "duckdb/planner/operator/logical_filter.hpp"
#include "duckdb/planner/operator/logical_get.hpp"
#include "duckdb/planner/operator/logical_join.hpp"
#include "duckdb/planner/operator/logical_projection.hpp"
#include "duckdb/planner/operator/logical_top_n.hpp"
#include "duckdb/planner/operator/logical_unnest.hpp"
#include "duckdb/planner/table_filter.hpp"
#include "duckdb/storage/statistics/numeric_stats.hpp"
#include "log/logging.hpp"
#include "sirius_context.hpp"
#include "vss/access_path_cost.hpp"
#include "vss/cluster_lists.hpp"
#include "vss/device_rates.hpp"
#include "vss/kmeans_functions.hpp"
#include "vss/vector_join.hpp"
#include "vss/vector_join_binding.hpp"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <limits>
#include <optional>
#include <string>
#include <vector>

namespace sirius::vss {

namespace {

using duckdb::ColumnBinding;
using duckdb::Expression;
using duckdb::ExpressionClass;
using duckdb::ExpressionType;
using duckdb::LogicalOperator;
using duckdb::LogicalOperatorType;
using duckdb::unique_ptr;

enum class distance_kind { l2, cosine_similarity, cosine_distance };

struct distance_call {
  distance_kind kind;
  ColumnBinding a;
  ColumnBinding b;
  duckdb::idx_t dim;
};

std::optional<duckdb::idx_t> float_array_width(const duckdb::LogicalType& t)
{
  if (t.id() != duckdb::LogicalTypeId::ARRAY) { return std::nullopt; }
  if (duckdb::ArrayType::GetChildType(t).id() != duckdb::LogicalTypeId::FLOAT) {
    return std::nullopt;
  }
  return duckdb::ArrayType::GetSize(t);
}

std::optional<distance_call> as_distance_call(const Expression& e)
{
  if (e.GetExpressionClass() != ExpressionClass::BOUND_FUNCTION) { return std::nullopt; }
  auto const& f = e.Cast<duckdb::BoundFunctionExpression>();
  distance_kind kind;
  if (f.function.name == "array_distance") {
    kind = distance_kind::l2;
  } else if (f.function.name == "array_cosine_similarity") {
    kind = distance_kind::cosine_similarity;
  } else if (f.function.name == "array_cosine_distance") {
    kind = distance_kind::cosine_distance;
  } else {
    return std::nullopt;
  }
  if (f.children.size() != 2) { return std::nullopt; }
  for (auto const& c : f.children) {
    if (c->GetExpressionClass() != ExpressionClass::BOUND_COLUMN_REF) { return std::nullopt; }
  }
  auto const wa = float_array_width(f.children[0]->return_type);
  auto const wb = float_array_width(f.children[1]->return_type);
  if (!wa || !wb || *wa != *wb) { return std::nullopt; }
  return distance_call{kind,
                       f.children[0]->Cast<duckdb::BoundColumnRefExpression>().binding,
                       f.children[1]->Cast<duckdb::BoundColumnRefExpression>().binding,
                       *wa};
}

bool same_pair(const distance_call& c, const ColumnBinding& a, const ColumnBinding& b)
{
  return (c.a == a && c.b == b) || (c.a == b && c.b == a);
}

/// `distance(a, b) <= eps` (or `<`), `similarity(a, b) >= s` (or `>`), either operand order.
struct threshold_predicate {
  distance_call call;
  bool strict;
  double bound;
};

std::optional<threshold_predicate> as_threshold(const Expression& e)
{
  if (e.GetExpressionClass() != ExpressionClass::BOUND_COMPARISON) { return std::nullopt; }
  auto const& cmp              = e.Cast<duckdb::BoundComparisonExpression>();
  auto type                    = cmp.GetExpressionType();
  const Expression* fn_side    = cmp.left.get();
  const Expression* const_side = cmp.right.get();
  if (const_side->GetExpressionClass() != ExpressionClass::BOUND_CONSTANT) {
    std::swap(fn_side, const_side);
    type = duckdb::FlipComparisonExpression(type);
  }
  if (const_side->GetExpressionClass() != ExpressionClass::BOUND_CONSTANT) { return std::nullopt; }
  auto const call = as_distance_call(*fn_side);
  if (!call) { return std::nullopt; }
  auto const& value = const_side->Cast<duckdb::BoundConstantExpression>().value;
  if (value.IsNull() || !value.type().IsNumeric()) { return std::nullopt; }
  auto const bound         = value.GetValue<double>();
  bool const is_similarity = call->kind == distance_kind::cosine_similarity;
  switch (type) {
    case ExpressionType::COMPARE_LESSTHAN:
    case ExpressionType::COMPARE_LESSTHANOREQUALTO:
      if (is_similarity) { return std::nullopt; }
      break;
    case ExpressionType::COMPARE_GREATERTHAN:
    case ExpressionType::COMPARE_GREATERTHANOREQUALTO:
      if (!is_similarity) { return std::nullopt; }
      break;
    default: return std::nullopt;
  }
  bool const strict =
    type == ExpressionType::COMPARE_LESSTHAN || type == ExpressionType::COMPARE_GREATERTHAN;
  return threshold_predicate{*call, strict, bound};
}

void split_conjunction(unique_ptr<Expression> e, std::vector<unique_ptr<Expression>>& out)
{
  if (e->GetExpressionType() == ExpressionType::CONJUNCTION_AND) {
    auto& conj = e->Cast<duckdb::BoundConjunctionExpression>();
    for (auto& child : conj.children) {
      split_conjunction(std::move(child), out);
    }
    return;
  }
  // DuckDB folds two bounds on one expression into a BETWEEN; on a distance call that hides the
  // threshold, so it goes back to the two comparisons it stands for.
  if (e->GetExpressionClass() == ExpressionClass::BOUND_BETWEEN) {
    auto& between = e->Cast<duckdb::BoundBetweenExpression>();
    if (as_distance_call(*between.input)) {
      out.push_back(duckdb::make_uniq<duckdb::BoundComparisonExpression>(
        between.lower_inclusive ? ExpressionType::COMPARE_GREATERTHANOREQUALTO
                                : ExpressionType::COMPARE_GREATERTHAN,
        between.input->Copy(),
        std::move(between.lower)));
      out.push_back(duckdb::make_uniq<duckdb::BoundComparisonExpression>(
        between.upper_inclusive ? ExpressionType::COMPARE_LESSTHANOREQUALTO
                                : ExpressionType::COMPARE_LESSTHAN,
        std::move(between.input),
        std::move(between.upper)));
      return;
    }
  }
  out.push_back(std::move(e));
}

/// True when DuckDB's statistics say the table column behind @p b can hold a NULL vector. A NULL
/// vector has no distance, so DuckDB orders or filters it in ways the join does not reproduce;
/// declining leaves the query to a plan that can fall back to the CPU at run time. Only a column
/// read straight from a table scan is checked; any other source reports false.
bool vector_may_hold_nulls(duckdb::ClientContext& context,
                           LogicalOperator& subtree,
                           const ColumnBinding& b)
{
  if (subtree.type == LogicalOperatorType::LOGICAL_GET) {
    auto& scan = subtree.Cast<duckdb::LogicalGet>();
    if (scan.table_index != b.table_index) { return false; }
    auto table = scan.GetTable();
    if (!table) { return false; }
    auto const pos  = scan.projection_ids.empty() ? b.column_index
                      : b.column_index < scan.projection_ids.size()
                        ? scan.projection_ids[b.column_index]
                        : scan.GetColumnIds().size();
    auto const& ids = scan.GetColumnIds();
    if (pos >= ids.size() || ids[pos].IsRowIdColumn() || ids[pos].IsEmptyColumn()) { return false; }
    try {
      auto const stats = table->GetStatistics(context, ids[pos].GetPrimaryIndex());
      return stats && stats->CanHaveNull();
    } catch (std::exception&) {
      return false;
    }
  }
  for (auto& child : subtree.children) {
    if (vector_may_hold_nulls(context, *child, b)) { return true; }
  }
  return false;
}

bool has_binding(LogicalOperator& op, const ColumnBinding& b)
{
  auto const bindings = op.GetColumnBindings();
  return std::find(bindings.begin(), bindings.end(), b) != bindings.end();
}

bool has_projection_map(const LogicalOperator& op)
{
  if (op.type == LogicalOperatorType::LOGICAL_FILTER) {
    return !op.Cast<duckdb::LogicalFilter>().projection_map.empty();
  }
  switch (op.type) {
    case LogicalOperatorType::LOGICAL_COMPARISON_JOIN:
    case LogicalOperatorType::LOGICAL_ANY_JOIN:
    case LogicalOperatorType::LOGICAL_DELIM_JOIN:
    case LogicalOperatorType::LOGICAL_ASOF_JOIN: {
      auto const& join = op.Cast<duckdb::LogicalJoin>();
      return !join.left_projection_map.empty() || !join.right_projection_map.empty();
    }
    default: return false;
  }
}

/// Every column binding an expression tree reads.
void collect_bindings(const Expression& e, duckdb::column_binding_set_t& out)
{
  if (e.GetExpressionClass() == ExpressionClass::BOUND_COLUMN_REF) {
    out.insert(e.Cast<duckdb::BoundColumnRefExpression>().binding);
  }
  duckdb::ExpressionIterator::EnumerateChildren(
    e, [&](const Expression& child) { collect_bindings(child, out); });
}

/// Walk the plan outside @p skip, handing @p fn every expression slot.
void for_each_expression_outside(LogicalOperator& op,
                                 const LogicalOperator* skip,
                                 const std::function<void(unique_ptr<Expression>&)>& fn)
{
  if (&op == skip) { return; }
  duckdb::LogicalOperatorVisitor::EnumerateExpressions(op,
                                                       [&](unique_ptr<Expression>* e) { fn(*e); });
  for (auto& child : op.children) {
    if (child) { for_each_expression_outside(*child, skip, fn); }
  }
}

/// Whether an expression reads either vector binding other than through a distance call on the
/// pair. Those reads would need the vectors themselves above the join.
bool reads_vectors_raw(const Expression& e, const ColumnBinding& a, const ColumnBinding& b)
{
  if (auto const call = as_distance_call(e); call && same_pair(*call, a, b)) { return false; }
  if (e.GetExpressionClass() == ExpressionClass::BOUND_COLUMN_REF) {
    auto const& binding = e.Cast<duckdb::BoundColumnRefExpression>().binding;
    if (binding == a || binding == b) { return true; }
  }
  bool raw = false;
  duckdb::ExpressionIterator::EnumerateChildren(
    e, [&](const Expression& child) { raw = raw || reads_vectors_raw(child, a, b); });
  return raw;
}

/// A distance expression over the pair, in terms of the join's score column.
unique_ptr<Expression> score_expression(duckdb::ClientContext& context,
                                        distance_kind requested,
                                        distance_kind joined,
                                        const ColumnBinding& score)
{
  auto ref = duckdb::make_uniq<duckdb::BoundColumnRefExpression>(duckdb::LogicalType::FLOAT, score);
  bool const joined_cosine = joined != distance_kind::l2;
  if (requested == distance_kind::l2) {
    if (joined_cosine) { return nullptr; }
    return ref;
  }
  if (!joined_cosine) { return nullptr; }
  // The cosine join emits similarity.
  if (requested == distance_kind::cosine_similarity) { return ref; }
  duckdb::vector<unique_ptr<Expression>> args;
  args.push_back(duckdb::make_uniq<duckdb::BoundConstantExpression>(duckdb::Value::FLOAT(1.0F)));
  args.push_back(std::move(ref));
  duckdb::ErrorData error;
  duckdb::FunctionBinder binder(context);
  auto result =
    binder.BindScalarFunction(DEFAULT_SCHEMA, "-", std::move(args), error, /*is_operator=*/true);
  return result;
}

/// Replace every distance call over the pair in @p e with its score form. False if one cannot
/// be expressed through the score (a different metric than the join's).
bool replace_distance_calls(duckdb::ClientContext& context,
                            unique_ptr<Expression>& e,
                            const ColumnBinding& a,
                            const ColumnBinding& b,
                            distance_kind joined,
                            const ColumnBinding& score)
{
  if (auto const call = as_distance_call(*e); call && same_pair(*call, a, b)) {
    auto replacement = score_expression(context, call->kind, joined, score);
    if (!replacement) { return false; }
    if (replacement->return_type != e->return_type) {
      replacement =
        duckdb::BoundCastExpression::AddCastToType(context, std::move(replacement), e->return_type);
    }
    e = std::move(replacement);
    return true;
  }
  bool ok = true;
  duckdb::ExpressionIterator::EnumerateChildren(*e, [&](unique_ptr<Expression>& child) {
    ok = ok && replace_distance_calls(context, child, a, b, joined, score);
  });
  return ok;
}

/// Where the probe relation sits inside a filtered join tree: the slot holding the cross
/// product it was crossed in through, and which child of that cross product it is.
struct lifted_relation {
  unique_ptr<LogicalOperator>* cross_slot{nullptr};
  std::size_t probe_child{0};
};

/// Share of all pairs a band keeps, from the min/max statistics DuckDB's statistics propagation
/// left on the join (one pair per condition, kept only if every condition got one). One condition
/// bounds an expression from below and the other from above; the band's width is the gap between
/// the bounds' midpoints, and its share is that width over the bounded expression's range -- a
/// uniform spread assumed, so skew moves it, but by a factor, not by DuckDB's ~100x.
std::optional<double> band_selectivity(duckdb::LogicalComparisonJoin const& join)
{
  if (join.join_stats.size() != 2 * join.conditions.size()) { return std::nullopt; }
  auto range_of = [](duckdb::BaseStatistics const& st) -> std::optional<std::pair<double, double>> {
    if (st.GetStatsType() != duckdb::StatisticsType::NUMERIC_STATS ||
        !duckdb::NumericStats::HasMinMax(st)) {
      return std::nullopt;
    }
    try {
      return std::pair{
        duckdb::NumericStats::Min(st).DefaultCastAs(duckdb::LogicalType::DOUBLE).GetValue<double>(),
        duckdb::NumericStats::Max(st)
          .DefaultCastAs(duckdb::LogicalType::DOUBLE)
          .GetValue<double>()};
    } catch (std::exception&) {
      return std::nullopt;
    }
  };
  // Per range condition: which side bounds the other from below (left <= right means the left
  // expression is a lower bound of the right one).
  struct bound {
    std::pair<double, double> left, right;
    bool left_is_lower;
  };
  std::vector<bound> bounds;
  for (std::size_t i = 0; i < join.conditions.size(); ++i) {
    auto const cmp = join.conditions[i].comparison;
    bool const lower =
      cmp == ExpressionType::COMPARE_LESSTHAN || cmp == ExpressionType::COMPARE_LESSTHANOREQUALTO;
    bool const upper = cmp == ExpressionType::COMPARE_GREATERTHAN ||
                       cmp == ExpressionType::COMPARE_GREATERTHANOREQUALTO;
    if (!lower && !upper) { continue; }
    auto const l = range_of(*join.join_stats[2 * i]);
    auto const r = range_of(*join.join_stats[2 * i + 1]);
    if (!l || !r) { return std::nullopt; }
    bounds.push_back({*l, *r, lower});
  }
  if (bounds.size() != 2 || bounds[0].left_is_lower == bounds[1].left_is_lower) {
    return std::nullopt;
  }
  auto const& lo = bounds[0].left_is_lower ? bounds[0] : bounds[1];
  auto const& hi = bounds[0].left_is_lower ? bounds[1] : bounds[0];
  auto mid       = [](std::pair<double, double> const& r) { return (r.first + r.second) / 2; };
  auto span      = [](std::pair<double, double> const& r) { return r.second - r.first; };
  // Either the left side brackets the right (lo.left <= right <= hi.left) or the right brackets
  // the left (hi.right <= left <= lo.right); the reading with a positive width is the band.
  double const width_on_right = mid(hi.left) - mid(lo.left);
  double const width_on_left  = mid(lo.right) - mid(hi.right);
  double const width          = std::max(width_on_right, width_on_left);
  double const range          = width_on_right >= width_on_left ? span(lo.right) : span(lo.left);
  if (!(width > 0) || !(range > 0)) { return std::nullopt; }
  return std::min(1.0, width / range);
}

/// An inner join on `<>` conditions and range conditions that, under a threshold, is cheaper as
/// the cross product with its conditions as filters (the vector join first). One range condition
/// (`a.k < b.k`, as a self-join uses to count each pair once) keeps about half the pairs, so it
/// always is. A band (two range conditions) is when it keeps enough pairs. Joining by the band
/// first runs on DuckDB's CPU, which pays per band pair mostly to gather both vectors: on Vec-H
/// images (d = 1152, 24 threads) 8.7 us a pair (a 0.01% price band, 2.9M pairs: 25.3 s), against
/// the A5000's 0.73 ns a pair over all 1.15e10 (8.4 s). So there the vector join wins once the band
/// keeps more than about 1/12,000 of the pairs; both sides scale with d, so the ratio does not. The
/// CPU side is scaled by DuckDB's threads and the GPU side by the device's FP32 rate. The share
/// comes from band_selectivity when the join carries min/max statistics for both sides of its
/// conditions; DuckDB's own range-join estimate is the fallback, and it runs far low (that band:
/// 0.017%). SIRIUS_VSS_BAND_VECTOR_FIRST=1 takes every band vector first, =0 none. Equality joins
/// are always left alone.
bool is_inequality_join(duckdb::ClientContext& context, LogicalOperator& op)
{
  if (op.type != LogicalOperatorType::LOGICAL_COMPARISON_JOIN) { return false; }
  auto const& join = op.Cast<duckdb::LogicalComparisonJoin>();
  if (join.join_type != duckdb::JoinType::INNER || join.conditions.empty()) { return false; }
  int ranges = 0;
  for (auto const& c : join.conditions) {
    switch (c.comparison) {
      case ExpressionType::COMPARE_NOTEQUAL:
      case ExpressionType::COMPARE_DISTINCT_FROM: break;
      case ExpressionType::COMPARE_LESSTHAN:
      case ExpressionType::COMPARE_GREATERTHAN:
      case ExpressionType::COMPARE_LESSTHANOREQUALTO:
      case ExpressionType::COMPARE_GREATERTHANOREQUALTO: ++ranges; break;
      default: return false;
    }
  }
  if (ranges <= 1) { return true; }
  if (ranges > 2) { return false; }
  auto const* env = std::getenv("SIRIUS_VSS_BAND_VECTOR_FIRST");
  if (env != nullptr && std::strcmp(env, "0") == 0) { return false; }
  if (env != nullptr && std::strcmp(env, "1") == 0) { return true; }
  double const pairs = static_cast<double>(op.children[0]->EstimateCardinality(context)) *
                       static_cast<double>(op.children[1]->EstimateCardinality(context));
  auto const from_stats = band_selectivity(join);
  double const share    = from_stats  ? *from_stats
                          : pairs > 0 ? static_cast<double>(op.EstimateCardinality(context)) / pairs
                                      : 0.0;
  auto const threads =
    std::max<double>(1.0, duckdb::TaskScheduler::GetScheduler(context).NumberOfThreads());
  double const cpu_per_pair = 8.7e-6 * 24.0 / threads;
  double const gpu_per_pair = 0.73e-9 / current_device_scale().fp32;
  bool const vector_first   = share * cpu_per_pair >= gpu_per_pair;
  SIRIUS_LOG_INFO("[vector_join_rewrite] band join keeps ~{:.2g} of {:.3g} pairs ({}): {}",
                  share,
                  pairs,
                  from_stats ? "column statistics" : "DuckDB's estimate",
                  vector_first ? "vector join first" : "left to the band join");
  return vector_first;
}

/// Find a CROSS_PRODUCT (or an inequality join) under @p slot, through operators without
/// projection maps, with the binding @p probe_vec on exactly one side and @p corpus_vec on the
/// other.
std::optional<lifted_relation> find_crossed_relation(duckdb::ClientContext& context,
                                                     unique_ptr<LogicalOperator>& slot,
                                                     const ColumnBinding& probe_vec,
                                                     const ColumnBinding& corpus_vec)
{
  auto& op = *slot;
  if (op.type == LogicalOperatorType::LOGICAL_CROSS_PRODUCT || is_inequality_join(context, op)) {
    for (std::size_t side = 0; side < 2; ++side) {
      if (has_binding(*op.children[side], probe_vec) &&
          has_binding(*op.children[1 - side], corpus_vec)) {
        return lifted_relation{&slot, side};
      }
    }
  }
  if (has_projection_map(op)) { return std::nullopt; }
  switch (op.type) {
    case LogicalOperatorType::LOGICAL_CROSS_PRODUCT:
    case LogicalOperatorType::LOGICAL_COMPARISON_JOIN:
    case LogicalOperatorType::LOGICAL_ANY_JOIN:
    case LogicalOperatorType::LOGICAL_FILTER: break;
    default: return std::nullopt;
  }
  for (auto& child : op.children) {
    if (has_binding(*child, probe_vec)) {
      return find_crossed_relation(context, child, probe_vec, corpus_vec);
    }
  }
  return std::nullopt;
}

void resolve_types_bottom_up(LogicalOperator& op)
{
  for (auto& child : op.children) {
    resolve_types_bottom_up(*child);
  }
  op.ResolveOperatorTypes();
}

bool rewrite_enabled(duckdb::ClientContext& context)
{
  auto const* env = std::getenv("SIRIUS_VSS_SQL_REWRITE");
  if (env != nullptr && std::strcmp(env, "0") == 0) { return false; }
  duckdb::Value setting;
  if (!context.TryGetCurrentSetting("gpu_execution", setting) || setting.IsNull() ||
      !setting.GetValue<bool>()) {
    return false;
  }
  return true;
}

class rewriter {
 public:
  rewriter(duckdb::ClientContext& context,
           duckdb::Binder& binder,
           unique_ptr<LogicalOperator>& root)
    : _context(context), _binder(binder), _root(root)
  {
  }

  [[nodiscard]] bool inlined() const { return _inlined; }

  std::size_t run()
  {
    std::size_t rewrites = 0;
    if (std::getenv("SIRIUS_VSS_REWRITE_DUMP") != nullptr) {
      SIRIUS_LOG_INFO("[vector_join_rewrite] plan before:\n{}", _root->ToString());
    }
    inline_vector_ctes(_root);
    // One pattern per pass: a rewrite moves subtrees, so the search restarts from the root.
    while (rewrites < 16 && try_one(_root)) {
      ++rewrites;
    }
    if (rewrites > 0) { resolve_types_bottom_up(*_root); }
    if (rewrites > 0 && std::getenv("SIRIUS_VSS_REWRITE_DUMP") != nullptr) {
      SIRIUS_LOG_INFO("[vector_join_rewrite] plan after:\n{}", _root->ToString());
    }
    return rewrites;
  }

 private:
  /// DuckDB turns a subplan that appears twice into a materialized CTE (COMMON_SUBPLAN), so a
  /// self-join written as `images JOIN part ... JOIN images JOIN part` reaches the matcher with
  /// both sides as CTE scans, and a scan of a materialized CTE has no vectors to search. A CTE that
  /// carries an ARRAY column is put back in place, a copy per scan, as DuckDB's own CTE inlining
  /// does; one it would refuse to inline (volatile functions and the like) is left. The caller
  /// keeps DuckDB's plan when nothing is rewritten after all.
  void inline_vector_ctes(unique_ptr<LogicalOperator>& op)
  {
    for (auto& child : op->children) {
      inline_vector_ctes(child);
    }
    if (op->type != LogicalOperatorType::LOGICAL_MATERIALIZED_CTE) { return; }
    auto& cte        = op->Cast<duckdb::LogicalCTE>();
    auto& definition = cte.children[0];
    definition->ResolveOperatorTypes();
    bool const vector =
      std::any_of(definition->types.begin(), definition->types.end(), [](auto const& t) {
        return t.id() == duckdb::LogicalTypeId::ARRAY;
      });
    if (!vector) { return; }
    duckdb::PreventInlining prevent;
    prevent.VisitOperator(*definition);
    if (prevent.prevent_inlining) { return; }
    bool copied_all = true;
    std::function<void(unique_ptr<LogicalOperator>&)> fill =
      [&](unique_ptr<LogicalOperator>& node) {
        if (node->type == LogicalOperatorType::LOGICAL_CTE_REF) {
          auto& ref = node->Cast<duckdb::LogicalCTERef>();
          if (ref.cte_index != cte.table_index) { return; }
          unique_ptr<LogicalOperator> copy;
          try {
            duckdb::LogicalOperatorDeepCopy deep_copy(_binder, nullptr);
            copy = deep_copy.DeepCopy(definition);
          } catch (std::exception&) {
            copied_all = false;
            return;
          }
          copy->ResolveOperatorTypes();
          duckdb::vector<unique_ptr<Expression>> columns;
          auto const bindings = copy->GetColumnBindings();
          for (std::size_t i = 0; i < bindings.size(); ++i) {
            columns.push_back(
              duckdb::make_uniq<duckdb::BoundColumnRefExpression>(copy->types[i], bindings[i]));
          }
          auto projection =
            duckdb::make_uniq<duckdb::LogicalProjection>(ref.table_index, std::move(columns));
          projection->children.push_back(std::move(copy));
          node = std::move(projection);
          return;
        }
        for (auto& child : node->children) {
          fill(child);
        }
      };
    auto body = std::move(cte.children[1]);
    fill(body);
    if (!copied_all) {
      cte.children[1] = std::move(body);  // the scans left are still served by the CTE
      _inlined        = true;
      return;
    }
    op       = std::move(body);
    _inlined = true;
  }

  /// What a column of the replaced operator becomes: the score, a corpus column, or -- for a
  /// LATERAL correlated on more than its vector -- a probe column.
  struct join_col {
    bool is_score;
    ColumnBinding corpus_col;
    distance_kind kind;
    ColumnBinding probe_col{};
  };

  /// A projection a rewrite installed in place of the operator it replaced.
  struct installed_projection {
    duckdb::idx_t index;
    LogicalOperator* op;
  };

  /// Installs @p replacement in @p slot under a projection that emits the replaced operator's
  /// columns in their original positions, so everything above that reads columns by position --
  /// projection maps, and the scans of a CTE or a delim join the operator fed -- reads what it
  /// read before. @p exprs has one expression per column of the replaced operator, over
  /// @p replacement's outputs; a null one, for a column nothing reads, becomes a typed NULL.
  installed_projection install(unique_ptr<LogicalOperator>& slot,
                               unique_ptr<LogicalOperator> replacement,
                               duckdb::vector<unique_ptr<Expression>> exprs)
  {
    for (std::size_t i = 0; i < exprs.size(); ++i) {
      if (!exprs[i]) {
        // The NULL only holds the position. A nested one (usually the vector column, unread
        // under a count(*)) is a constant the GPU cannot build, and any scalar NULL holds it
        // as well.
        auto const& type = _slot_types[i];
        exprs[i]         = duckdb::make_uniq<duckdb::BoundConstantExpression>(
          duckdb::Value(type.IsNested() ? duckdb::LogicalType::INTEGER : type));
      }
    }
    auto const index = _binder.GenerateTableIndex();
    auto proj        = duckdb::make_uniq<duckdb::LogicalProjection>(index, std::move(exprs));
    proj->children.push_back(std::move(replacement));
    proj->ResolveOperatorTypes();
    auto* op = proj.get();
    slot     = std::move(proj);
    return {index, op};
  }

  /// Points the column references above @p p at it instead of at the replaced operator.
  void remap_above(const installed_projection& p)
  {
    duckdb::ColumnBindingReplacer replacer;
    replacer.stop_operator = p.op;
    for (std::size_t i = 0; i < _slot_bindings.size(); ++i) {
      replacer.replacement_bindings.emplace_back(
        _slot_bindings[i], ColumnBinding{p.index, i}, p.op->types[i]);
    }
    replacer.VisitOperator(*_root);
  }

  /// Where @p b sat in the replaced operator's output, if it was there.
  std::optional<std::size_t> slot_position(const ColumnBinding& b) const
  {
    for (std::size_t i = 0; i < _slot_bindings.size(); ++i) {
      if (_slot_bindings[i] == b) { return i; }
    }
    return std::nullopt;
  }

  bool try_one(unique_ptr<LogicalOperator>& slot)
  {
    // What the operator a rewrite may replace emits, taken before the rewrite takes it apart.
    switch (slot->type) {
      case LogicalOperatorType::LOGICAL_ANY_JOIN:
      case LogicalOperatorType::LOGICAL_FILTER:
      case LogicalOperatorType::LOGICAL_DELIM_JOIN:
      case LogicalOperatorType::LOGICAL_TOP_N:
      case LogicalOperatorType::LOGICAL_CROSS_PRODUCT:
        _slot_bindings = slot->GetColumnBindings();
        _slot_types    = slot->types;
        break;
      default: break;
    }
    // Top-down: a top-k or threshold shape above a cross product has to claim it before the
    // cross product alone would be read as an unbounded all-pairs join.
    bool const rewritten = [&] {
      switch (slot->type) {
        case LogicalOperatorType::LOGICAL_ANY_JOIN: return try_any_join(slot);
        case LogicalOperatorType::LOGICAL_FILTER: return try_filter(slot);
        case LogicalOperatorType::LOGICAL_DELIM_JOIN: return try_lateral_topk(slot);
        case LogicalOperatorType::LOGICAL_TOP_N: return try_global_topk(slot);
        case LogicalOperatorType::LOGICAL_CROSS_PRODUCT: return try_scalar_distance(slot);
        default: return false;
      }
    }();
    if (rewritten) { return true; }
    for (auto& child : slot->children) {
      if (child && try_one(child)) { return true; }
    }
    return false;
  }

  /// `distance(t.v, (SELECT v FROM q))` computed for every row: a cross product with a one-row
  /// scalar subquery whose vector is read only through distance calls on the pair. That is a join
  /// with no bound, so it becomes a threshold join every pair passes.
  bool try_scalar_distance(unique_ptr<LogicalOperator>& slot)
  {
    auto& cross = *slot;
    for (std::size_t pi = 0; pi < 2; ++pi) {
      auto& probe_slot = cross.children[pi];
      // Find the scalar vector this side exposes, and the distance calls reading it above.
      if (probe_slot->type != LogicalOperatorType::LOGICAL_PROJECTION) { continue; }
      auto const probe_bindings = probe_slot->GetColumnBindings();
      if (probe_bindings.size() != 1) { continue; }
      auto const probe_vec = probe_bindings[0];
      std::optional<distance_call> pair;
      bool ok = true;
      for_each_expression_outside(*_root, slot.get(), [&](unique_ptr<Expression>& e) {
        std::function<void(const Expression&)> walk = [&](const Expression& x) {
          if (auto const c = as_distance_call(x); c && (c->a == probe_vec || c->b == probe_vec)) {
            if (pair && (!same_pair(*pair, c->a, c->b) ||
                         (pair->kind == distance_kind::l2) != (c->kind == distance_kind::l2))) {
              ok = false;
            }
            if (!pair) { pair = c; }
            return;
          }
          if (x.GetExpressionClass() == ExpressionClass::BOUND_COLUMN_REF &&
              x.Cast<duckdb::BoundColumnRefExpression>().binding == probe_vec) {
            ok = false;
          }
          duckdb::ExpressionIterator::EnumerateChildren(x, walk);
        };
        walk(*e);
      });
      if (!ok || !pair) { continue; }
      auto const corpus_vec = pair->a == probe_vec ? pair->b : pair->a;
      if (!has_binding(*cross.children[1 - pi], corpus_vec)) { continue; }
      auto probe        = std::move(probe_slot);
      auto probe_vec_in = probe_vec;
      if (!unwrap_scalar_subquery(probe, probe_vec_in)) {
        probe_slot = std::move(probe);  // not a scalar subquery: an unbounded join, leave it
        continue;
      }
      if (reads_vectors_raw_anywhere(slot.get(), corpus_vec, probe_vec)) {
        return false;  // the corpus vector is also read raw above; keep DuckDB's plan
      }
      duckdb::column_binding_set_t used;
      for_each_expression_outside(
        *_root, slot.get(), [&](unique_ptr<Expression>& e) { collect_bindings(*e, used); });
      auto corpus = std::move(cross.children[1 - pi]);
      probe->ResolveOperatorTypes();
      corpus->ResolveOperatorTypes();
      duckdb::vector<ColumnBinding> passthrough;
      std::vector<join_col> cols;
      for (auto const& b : corpus->GetColumnBindings()) {
        if (used.count(b) && b != corpus_vec) {
          passthrough.push_back(b);
          cols.push_back({false, b, pair->kind});
        }
      }
      if (!build_topk(slot,
                      std::move(probe),
                      std::move(corpus),
                      probe_vec_in,
                      corpus_vec,
                      *pair,
                      /*k=*/1,
                      passthrough,
                      cols,
                      used,
                      vector_join_mode::threshold,
                      /*probe_scalar=*/true)) {
        return false;
      }
      return true;
    }
    return false;
  }

  /// Whether anything above @p node reads @p v raw (not inside a distance call on (v, other)).
  bool reads_vectors_raw_anywhere(const LogicalOperator* node,
                                  const ColumnBinding& v,
                                  const ColumnBinding& other)
  {
    bool raw = false;
    for_each_expression_outside(*_root, node, [&](unique_ptr<Expression>& e) {
      if (reads_vectors_raw(*e, v, other)) { raw = true; }
    });
    return raw;
  }

  /// Substitute references to the projections in @p projs (by table index) with the expressions
  /// they project, recursively, so an expression reads the operators below them.
  static unique_ptr<Expression> inline_projections(
    const Expression& e, const std::vector<duckdb::LogicalProjection*>& projs)
  {
    if (e.GetExpressionClass() == ExpressionClass::BOUND_COLUMN_REF) {
      auto const& b = e.Cast<duckdb::BoundColumnRefExpression>().binding;
      for (auto* proj : projs) {
        if (proj->table_index == b.table_index && b.column_index < proj->expressions.size()) {
          return inline_projections(*proj->expressions[b.column_index], projs);
        }
      }
    }
    auto copy = e.Copy();
    duckdb::ExpressionIterator::EnumerateChildren(
      *copy, [&](unique_ptr<Expression>& child) { child = inline_projections(*child, projs); });
    return copy;
  }

  /// `SET vector_join_probes = p` makes a join that can go through cluster lists approximate, as
  /// `ivfflat.probes` does in pgvector: each probe row searches its p nearest clusters. The lists
  /// are those of the clustering `vector_join_clustering` names, when it is one of this column's,
  /// or else lists that would answer exactly. With no such lists the setting does not apply.
  bool use_approximate_lists(vector_join_request& req,
                             duckdb::SiriusContext& ctx,
                             const std::string& catalog,
                             const std::string& schema,
                             const std::string& table,
                             const std::string& column,
                             std::uint64_t rows)
  {
    duckdb::Value setting;
    if (!_context.TryGetCurrentSetting("vector_join_probes", setting) || setting.IsNull()) {
      return false;
    }
    auto const probes = setting.GetValue<std::int64_t>();
    if (probes <= 0) { return false; }
    bool const cosine = req.metric == "cosine";
    auto const n_rows = static_cast<std::int64_t>(rows);
    std::optional<exact_lists_choice> choice;
    if (_context.TryGetCurrentSetting("vector_join_clustering", setting) && !setting.IsNull() &&
        !setting.ToString().empty()) {
      choice =
        find_named_lists(ctx, setting.ToString(), catalog, schema, table, column, cosine, n_rows);
    }
    if (!choice) { choice = find_exact_lists(ctx, catalog, schema, table, column, cosine, n_rows); }
    if (!choice) { return false; }
    req.search_mode = vector_join_search_mode::approx;
    req.clustering  = choice->clustering;
    req.n_probes    = std::min(probes, choice->n_clusters);
    SIRIUS_LOG_INFO(
      "[vector_join_rewrite] searching '{}' through {} of the {} clusters of '{}' "
      "(vector_join_probes)",
      table,
      req.n_probes,
      choice->n_clusters,
      choice->clustering);
    return true;
  }

  /// Picks how a join over a pinned corpus runs. Under vector_join_probes, approximately through
  /// cluster lists (use_approximate_lists). Otherwise exactly: brute force over the pinned rows,
  /// or every cluster of its lists searched (the same answer, from fewer bytes and on tensor
  /// cores), by access_path_cost on the estimated probe rows. With no lists it builds them first
  /// (kept for later queries, like an index) when building and searching beats brute force,
  /// unless SIRIUS_VSS_BUILD_IN_QUERY=0. SIRIUS_VSS_ACCESS_PATH=lists|brute forces the choice;
  /// SIRIUS_VSS_REWRITE_LISTS=0 is brute.
  bool use_lists(vector_join_request& req,
                 duckdb::TableCatalogEntry& table,
                 const std::string& column,
                 std::uint64_t rows,
                 double probe_rows)
  {
    auto const* env = std::getenv("SIRIUS_VSS_REWRITE_LISTS");
    if (env != nullptr && std::strcmp(env, "0") == 0) { return false; }
    auto const* forced = std::getenv("SIRIUS_VSS_ACCESS_PATH");
    std::string_view const force{forced != nullptr ? forced : "cost"};
    if (force == "brute") { return false; }
    auto sirius_ctx = _context.registered_state->Get<duckdb::SiriusContext>("sirius_state");
    if (!sirius_ctx) { return false; }
    auto const& catalog = table.ParentCatalog().GetName();
    auto const& schema  = table.ParentSchema().name;
    if (use_approximate_lists(req, *sirius_ctx, catalog, schema, table.name, column, rows)) {
      return true;
    }
    bool const cosine = req.metric == "cosine";
    auto choice       = find_exact_lists(
      *sirius_ctx, catalog, schema, table.name, column, cosine, static_cast<std::int64_t>(rows));
    auto pin = sirius_ctx->get_scan_manager().find_pinned_entry_for_duckdb_table(
      catalog, schema, table.name);
    access_path_shape shape;
    shape.probe_rows       = std::max(probe_rows, 1.0);
    shape.corpus_rows      = static_cast<double>(rows);
    shape.dim              = static_cast<double>(req.dim);
    shape.corpus_on_device = pin != nullptr && pin->tier == cucascade::memory::Tier::GPU;
    access_path_cost const cost{current_device_scale()};
    double const brute = cost.brute(shape);
    auto describe      = [&](exact_lists_choice const& c) {
      shape.list_bytes_per_value = c.encoding == list_encoding::uint8     ? 1
                                        : c.encoding == list_encoding::float16 ? 2
                                                                               : 4;
      shape.lists_on_device      = c.on_device;
      shape.n_clusters           = static_cast<double>(c.n_clusters);
      shape.inexact_unseeded     = c.encoding == list_encoding::float16 && !c.seeded;
    };
    if (!choice) {
      auto const* build_env = std::getenv("SIRIUS_VSS_BUILD_IN_QUERY");
      if (force == "cost" && build_env != nullptr && std::strcmp(build_env, "0") == 0) {
        return false;
      }
      // Lists that would answer exactly and beat brute force: UINT8 if every value is a byte,
      // FLOAT16 otherwise (FP32 lists hold the same bytes as the pin and never pay).
      if (req.dim % 16 != 0) { return false; }
      shape.list_bytes_per_value = cosine ? 2 : 1;
      shape.lists_on_device      = true;
      shape.n_clusters           = 1024;
      double const built         = cost.build(shape) + cost.lists(shape);
      if (force == "cost" && built >= brute) {
        SIRIUS_LOG_INFO(
          "[vector_join_rewrite] '{}': brute force ({:.3f} s est.) over building lists ({:.3f} s)",
          table.name,
          brute,
          built);
        return false;
      }
      choice = build_lists_in_query(*sirius_ctx, req, catalog, schema, table.name, column, rows);
      if (!choice) { return false; }
    }
    describe(*choice);
    double const listed = cost.lists(shape);
    bool const fp32     = choice->encoding == list_encoding::float32;
    if (force == "cost" && (fp32 || listed >= brute)) {
      SIRIUS_LOG_INFO(
        "[vector_join_rewrite] '{}': brute force ({:.3f} s est.) over the lists of '{}' ({:.3f} s)",
        table.name,
        brute,
        choice->clustering,
        listed);
      return false;
    }
    req.search_mode = vector_join_search_mode::approx;
    req.clustering  = choice->clustering;
    req.n_probes    = choice->n_clusters;
    SIRIUS_LOG_INFO(
      "[vector_join_rewrite] searching '{}' through the lists of clustering '{}' ({:.3f} s est. vs "
      "brute force {:.3f} s)",
      table.name,
      choice->clustering,
      listed,
      brute);
    return true;
  }

  /// Fits a clustering of the pinned column and writes its lists, under a name later queries
  /// find them by: UINT8 when every value is a byte (the build checks), else FLOAT16, which
  /// keeps the FP32 rows to re-score against. Nullopt, and the join runs by brute force, when the
  /// build is refused (no room for the lists or their FP32 copy).
  std::optional<exact_lists_choice> build_lists_in_query(duckdb::SiriusContext& ctx,
                                                         vector_join_request const& req,
                                                         const std::string& catalog,
                                                         const std::string& schema,
                                                         const std::string& table,
                                                         const std::string& column,
                                                         std::uint64_t rows)
  {
    bool const cosine      = req.metric == "cosine";
    std::string const name = "__sirius_auto_" + table + "_" + column + (cosine ? "_cosine" : "");
    try {
      kmeans_fit_request fit;
      fit.name            = name;
      fit.catalog         = catalog;
      fit.schema          = schema;
      fit.table           = table;
      fit.column          = column;
      fit.dim             = req.dim;
      fit.spec.n_clusters = std::clamp<std::int64_t>(
        static_cast<std::int64_t>(std::sqrt(static_cast<double>(rows))), 1, 1024);
      run_kmeans_fit(ctx, fit);
      kmeans_assign_request lists;
      lists.clustering = name;
      lists.catalog    = catalog;
      lists.schema     = schema;
      lists.table      = table;
      lists.column     = column;
      lists.dim        = req.dim;
      run_kmeans_build_lists(
        ctx, lists, cosine ? list_storage::float16 : list_storage::exact, false, cosine);
    } catch (std::exception const& e) {
      SIRIUS_LOG_INFO(
        "[vector_join_rewrite] building lists for '{}' declined: {}", table, e.what());
      return std::nullopt;
    }
    SIRIUS_LOG_INFO(
      "[vector_join_rewrite] built lists '{}' for '{}' inside the query", name, table);
    return find_exact_lists(
      ctx, catalog, schema, table, column, cosine, static_cast<std::int64_t>(rows));
  }

  /// DuckDB's join order can push an inner join or a filter from above a LATERAL into its subquery
  /// side: DELIM_JOIN(probe, JOIN(lateral, t)). When none of them reads the probe, the same rows
  /// come from JOIN(DELIM_JOIN(probe, lateral), t), where the LATERAL is again the delim join's
  /// whole subquery side. Moves them there, under a projection that keeps the delim join's column
  /// layout, and leaves the LATERAL itself to the next pass.
  bool hoist_out_of_lateral(unique_ptr<LogicalOperator>& slot, std::size_t sub_child)
  {
    auto& dj = *slot;
    duckdb::column_binding_set_t probe_set;
    for (auto const& b : dj.children[1 - sub_child]->GetColumnBindings()) {
      probe_set.insert(b);
    }
    std::function<bool(LogicalOperator&)> has_delim_get = [&](LogicalOperator& op) {
      if (op.type == LogicalOperatorType::LOGICAL_DELIM_GET) { return true; }
      return std::any_of(
        op.children.begin(), op.children.end(), [&](auto& c) { return c && has_delim_get(*c); });
    };
    unique_ptr<LogicalOperator>* at = &dj.children[sub_child];
    std::vector<LogicalOperator*> path_ops;
    while (true) {
      auto& op         = **at;
      std::size_t next = 0;
      if (op.type == LogicalOperatorType::LOGICAL_FILTER && op.children.size() == 1) {
        next = 0;
      } else if (op.type == LogicalOperatorType::LOGICAL_COMPARISON_JOIN &&
                 op.Cast<duckdb::LogicalComparisonJoin>().join_type == duckdb::JoinType::INNER) {
        bool const in0 = has_delim_get(*op.children[0]);
        if (in0 == has_delim_get(*op.children[1])) { return false; }
        next = in0 ? 0 : 1;
      } else {
        break;
      }
      bool reads_probe = false;
      duckdb::LogicalOperatorVisitor::EnumerateExpressions(op, [&](unique_ptr<Expression>* e) {
        duckdb::column_binding_set_t reads;
        collect_bindings(**e, reads);
        for (auto const& b : reads) {
          if (probe_set.count(b)) { reads_probe = true; }
        }
      });
      if (reads_probe) { return false; }
      path_ops.push_back(&op);
      at = &op.children[next];
    }
    if (at == &dj.children[sub_child]) { return false; }
    auto const* top = at->get();
    while (top->type == LogicalOperatorType::LOGICAL_PROJECTION) {
      top = top->children[0].get();
    }
    if (top->type != LogicalOperatorType::LOGICAL_UNNEST &&
        top->type != LogicalOperatorType::LOGICAL_AGGREGATE_AND_GROUP_BY) {
      return false;
    }

    // Projection maps select child columns by position, and every child on the path changes
    // shape. They only prune, and the projection installed above restores the delim join's
    // columns by binding, so they are dropped on the path. The delim join itself keeps exactly
    // what that projection and the moved operators read: an unread vector left in its output
    // would need a NULL of the vector's type as a placeholder, which the GPU cannot build.
    duckdb::column_binding_set_t needed(_slot_bindings.begin(), _slot_bindings.end());
    for (auto* op : path_ops) {
      duckdb::LogicalOperatorVisitor::EnumerateExpressions(
        *op, [&](unique_ptr<Expression>* e) { collect_bindings(**e, needed); });
    }
    std::array<duckdb::vector<duckdb::idx_t>, 2> maps;
    for (std::size_t side = 0; side < 2; ++side) {
      auto const binds =
        side == sub_child ? (*at)->GetColumnBindings() : dj.children[side]->GetColumnBindings();
      for (std::size_t i = 0; i < binds.size(); ++i) {
        if (needed.count(binds[i])) { maps[side].push_back(i); }
      }
      if (maps[side].empty()) { return false; }
      if (maps[side].size() == binds.size()) { maps[side].clear(); }
    }
    for (auto* op : path_ops) {
      if (op->type == LogicalOperatorType::LOGICAL_FILTER) {
        op->Cast<duckdb::LogicalFilter>().projection_map.clear();
      } else {
        op->Cast<duckdb::LogicalJoin>().left_projection_map.clear();
        op->Cast<duckdb::LogicalJoin>().right_projection_map.clear();
      }
    }
    auto& dj_join                = dj.Cast<duckdb::LogicalJoin>();
    dj_join.left_projection_map  = std::move(maps[0]);
    dj_join.right_projection_map = std::move(maps[1]);
    auto lateral                 = std::move(*at);
    auto path                    = std::move(dj.children[sub_child]);
    dj.children[sub_child]       = std::move(lateral);
    *at                          = std::move(slot);
    slot                         = std::move(path);
    resolve_types_bottom_up(*slot);
    duckdb::vector<unique_ptr<Expression>> exprs;
    for (std::size_t i = 0; i < _slot_bindings.size(); ++i) {
      exprs.push_back(
        duckdb::make_uniq<duckdb::BoundColumnRefExpression>(_slot_types[i], _slot_bindings[i]));
    }
    auto hoisted = std::move(slot);
    remap_above(install(slot, std::move(hoisted), std::move(exprs)));
    SIRIUS_LOG_DEBUG("[vector_join_rewrite] moved a join/filter out of a LATERAL's subquery side");
    return true;
  }

  /// `probe, LATERAL (SELECT ... FROM corpus ORDER BY distance(probe.v, corpus.v) LIMIT k)`, as
  /// DuckDB decorrelates it: DELIM_JOIN(probe, PROJECTION* <- UNNEST <- AGGREGATE[arg_min(
  /// struct_pack(fields), distance, k) GROUP BY v] <- PROJECTION* <- CROSS(corpus, DELIM_GET)).
  /// That is a per-row top-k join, so it becomes one.
  bool try_lateral_topk(unique_ptr<LogicalOperator>& slot)
  {
    auto decline = [](int where) {
      SIRIUS_LOG_DEBUG("[vector_join_rewrite] LATERAL top-k shape not matched (check {})", where);
      return false;
    };
    auto& dj = slot->Cast<duckdb::LogicalComparisonJoin>();
    // A projection map only selects which child columns the join passes up; the replacement
    // projection emits exactly the columns the plan above reads, so it does not matter here.
    // The LATERAL is correlated on the probe vector and possibly on other probe columns too.
    if (dj.join_type != duckdb::JoinType::INNER || dj.duplicate_eliminated_columns.empty() ||
        dj.conditions.size() != dj.duplicate_eliminated_columns.size() || dj.children.size() != 2) {
      return decline(1);
    }
    std::vector<ColumnBinding> delim_cols;
    for (auto const& e : dj.duplicate_eliminated_columns) {
      if (e->GetExpressionClass() != ExpressionClass::BOUND_COLUMN_REF) { return decline(2); }
      delim_cols.push_back(e->Cast<duckdb::BoundColumnRefExpression>().binding);
    }
    std::size_t pi = 2;
    for (std::size_t c = 0; c < 2; ++c) {
      if (std::all_of(delim_cols.begin(), delim_cols.end(), [&](auto const& b) {
            return has_binding(*dj.children[c], b);
          })) {
        pi = c;
      }
    }
    if (pi == 2) { return decline(3); }
    auto& sub_slot = dj.children[1 - pi];

    // Subquery side, top-down: projections, then either the unnest of the aggregate's list of k
    // rows or -- LIMIT 1 -- the aggregate's single value directly.
    std::vector<duckdb::LogicalProjection*> above;
    LogicalOperator* cur = sub_slot.get();
    while (cur->type == LogicalOperatorType::LOGICAL_PROJECTION) {
      above.push_back(&cur->Cast<duckdb::LogicalProjection>());
      cur = cur->children[0].get();
    }
    duckdb::LogicalUnnest* unnest = nullptr;
    LogicalOperator* agg_op       = cur;
    if (cur->type == LogicalOperatorType::LOGICAL_UNNEST && cur->expressions.size() == 1) {
      unnest = &cur->Cast<duckdb::LogicalUnnest>();
      agg_op = unnest->children[0].get();
    } else if (cur->type != LogicalOperatorType::LOGICAL_AGGREGATE_AND_GROUP_BY) {
      return hoist_out_of_lateral(slot, 1 - pi) || decline(4);
    }
    if (agg_op->type != LogicalOperatorType::LOGICAL_AGGREGATE_AND_GROUP_BY) { return decline(5); }
    auto& agg = agg_op->Cast<duckdb::LogicalAggregate>();
    if (agg.groups.size() != delim_cols.size() || agg.expressions.size() != 1 ||
        agg.grouping_sets.size() > 1) {
      return decline(6);
    }
    auto const& aggr  = agg.expressions[0]->Cast<duckdb::BoundAggregateExpression>();
    auto const& aname = aggr.function.name;
    // arg_min/arg_max(value, key[, k]); for LIMIT 1 over the key alone, min/max(key).
    bool const by_key = aname == "min" || aname == "max";
    bool const ascending =
      aname == "arg_min" || aname == "arg_min_nulls_last" || aname == "min_by" || aname == "min";
    bool const descending =
      aname == "arg_max" || aname == "arg_max_nulls_last" || aname == "max_by" || aname == "max";
    auto const arity = unnest ? std::size_t{3} : by_key ? std::size_t{1} : std::size_t{2};
    if ((!ascending && !descending) || (unnest && by_key) || aggr.children.size() != arity ||
        aggr.filter || aggr.aggr_type != duckdb::AggregateType::NON_DISTINCT || aggr.order_bys) {
      return decline(7);
    }
    // The column the subquery's values are read from: the unnested row, or the aggregate itself.
    auto const value_binding =
      unnest ? ColumnBinding(unnest->unnest_index, 0) : ColumnBinding(agg.aggregate_index, 0);
    if (unnest) {
      // The unnest must read the aggregate's list.
      auto const& un = unnest->expressions[0]->Cast<duckdb::BoundUnnestExpression>();
      if (un.child->GetExpressionClass() != ExpressionClass::BOUND_COLUMN_REF ||
          un.child->Cast<duckdb::BoundColumnRefExpression>().binding !=
            ColumnBinding(agg.aggregate_index, 0)) {
        return decline(8);
      }
    }

    // Below the aggregate: projections, then CROSS(corpus, DELIM_GET).
    std::vector<duckdb::LogicalProjection*> below;
    unique_ptr<LogicalOperator>* below_slot = &agg.children[0];
    while ((*below_slot)->type == LogicalOperatorType::LOGICAL_PROJECTION) {
      below.push_back(&(*below_slot)->Cast<duckdb::LogicalProjection>());
      below_slot = &(*below_slot)->children[0];
    }
    auto& cross = **below_slot;
    if (cross.type != LogicalOperatorType::LOGICAL_CROSS_PRODUCT) { return decline(9); }
    std::size_t dgi = 2;
    for (std::size_t c = 0; c < 2; ++c) {
      if (cross.children[c]->type == LogicalOperatorType::LOGICAL_DELIM_GET) { dgi = c; }
    }
    if (dgi == 2) { return decline(10); }
    // The delim get emits the correlated columns in the delim join's order.
    auto const delim_bindings = cross.children[dgi]->GetColumnBindings();
    if (delim_bindings.size() != delim_cols.size()) { return decline(11); }

    auto const ordering = inline_projections(*aggr.children[by_key ? 0 : 1], below);
    auto const call     = as_distance_call(*ordering);
    auto const vec_at   = [&](const ColumnBinding& b) {
      return static_cast<std::size_t>(std::find(delim_bindings.begin(), delim_bindings.end(), b) -
                                      delim_bindings.begin());
    };
    if (!call) { return decline(12); }
    auto const vi = std::min(vec_at(call->a), vec_at(call->b));
    if (vi >= delim_bindings.size()) { return decline(12); }
    auto const delim_vec  = delim_bindings[vi];
    auto const probe_vec  = delim_cols[vi];
    auto const corpus_vec = call->a == delim_vec ? call->b : call->a;
    if ((call->kind == distance_kind::cosine_similarity) != descending) { return decline(13); }
    if (vector_may_hold_nulls(_context, *dj.children[pi], probe_vec) ||
        vector_may_hold_nulls(_context, *cross.children[1 - dgi], corpus_vec)) {
      return decline(19);
    }
    // The aggregate groups by exactly the correlated columns; group g is delim column
    // group_delim[g], which the probe has as delim_cols[group_delim[g]].
    std::vector<std::size_t> group_delim;
    for (auto const& g : agg.groups) {
      auto const group = inline_projections(*g, below);
      if (group->GetExpressionClass() != ExpressionClass::BOUND_COLUMN_REF) { return decline(14); }
      auto const at = vec_at(group->Cast<duckdb::BoundColumnRefExpression>().binding);
      if (at >= delim_bindings.size() ||
          std::find(group_delim.begin(), group_delim.end(), at) != group_delim.end()) {
        return decline(14);
      }
      group_delim.push_back(at);
    }
    std::int64_t k = 1;
    if (unnest) {
      auto const& kexpr = *aggr.children[2];
      if (kexpr.GetExpressionClass() != ExpressionClass::BOUND_CONSTANT) { return decline(15); }
      k = kexpr.Cast<duckdb::BoundConstantExpression>().value.GetValue<std::int64_t>();
      if (k <= 0) { return decline(16); }
    }

    // The aggregate's value: struct_pack(fields) or a single field, each a corpus column or the
    // distance itself.
    auto value = inline_projections(*aggr.children[0], below);
    std::vector<unique_ptr<Expression>> fields;
    // One selected column is aggregated as itself; several are packed into a struct.
    bool const single =
      !(value->GetExpressionClass() == ExpressionClass::BOUND_FUNCTION &&
        value->Cast<duckdb::BoundFunctionExpression>().function.name == "struct_pack");
    if (single) {
      fields.push_back(value->Copy());
    } else {
      for (auto& f : value->Cast<duckdb::BoundFunctionExpression>().children) {
        fields.push_back(f->Copy());
      }
    }
    auto& corpus_slot          = cross.children[1 - dgi];
    auto const corpus_bindings = corpus_slot->GetColumnBindings();
    duckdb::column_binding_set_t corpus_set;
    for (auto const& b : corpus_bindings) {
      corpus_set.insert(b);
    }
    for (auto const& f : fields) {
      if (auto const fc = as_distance_call(*f); fc && same_pair(*fc, delim_vec, corpus_vec)) {
        continue;
      }
      if (f->GetExpressionClass() != ExpressionClass::BOUND_COLUMN_REF ||
          !corpus_set.count(f->Cast<duckdb::BoundColumnRefExpression>().binding) ||
          f->Cast<duckdb::BoundColumnRefExpression>().binding == corpus_vec) {
        return decline(18);
      }
    }

    // What each column the subquery emits is, in terms of the unnested struct.
    auto const sub_bindings = sub_slot->GetColumnBindings();
    std::vector<join_col> sub_cols;
    for (auto const& b : sub_bindings) {
      duckdb::BoundColumnRefExpression ref(duckdb::LogicalType::ANY, b);
      auto e = inline_projections(ref, above);
      if (single && e->GetExpressionClass() == ExpressionClass::BOUND_COLUMN_REF &&
          e->Cast<duckdb::BoundColumnRefExpression>().binding == value_binding) {
        auto const& f = *fields[0];
        if (auto const fc = as_distance_call(f)) {
          sub_cols.push_back({true, ColumnBinding{}, fc->kind});
        } else {
          sub_cols.push_back(
            {false, f.Cast<duckdb::BoundColumnRefExpression>().binding, call->kind});
        }
        continue;
      }
      if (e->GetExpressionClass() == ExpressionClass::BOUND_COLUMN_REF) {
        // A group column is the probe's own value of that correlated column (the vector
        // included); the whole struct is only fine if nothing reads it.
        auto const& b = e->Cast<duckdb::BoundColumnRefExpression>().binding;
        if (b.table_index == agg.group_index && b.column_index < group_delim.size()) {
          sub_cols.push_back(
            {false, ColumnBinding{}, call->kind, delim_cols[group_delim[b.column_index]]});
          continue;
        }
      }
      if (e->GetExpressionClass() != ExpressionClass::BOUND_FUNCTION) {
        sub_cols.push_back({false, ColumnBinding{}, call->kind});
        continue;
      }
      auto& fn = e->Cast<duckdb::BoundFunctionExpression>();
      if ((fn.function.name != "struct_extract" && fn.function.name != "struct_extract_at") ||
          !fn.bind_info || fn.children.empty() ||
          fn.children[0]->GetExpressionClass() != ExpressionClass::BOUND_COLUMN_REF ||
          fn.children[0]->Cast<duckdb::BoundColumnRefExpression>().binding != value_binding) {
        return decline(19);
      }
      auto const idx = fn.bind_info->Cast<duckdb::StructExtractBindData>().index;
      if (idx >= fields.size()) { return decline(20); }
      auto const& f = *fields[idx];
      if (auto const fc = as_distance_call(f)) {
        sub_cols.push_back({true, ColumnBinding{}, fc->kind});
      } else {
        sub_cols.push_back({false, f.Cast<duckdb::BoundColumnRefExpression>().binding, call->kind});
      }
    }

    // Which subquery and probe columns the plan above reads.
    duckdb::column_binding_set_t used;
    for_each_expression_outside(
      *_root, slot.get(), [&](unique_ptr<Expression>& e) { collect_bindings(*e, used); });
    for (std::size_t j = 0; j < sub_bindings.size(); ++j) {
      if (used.count(sub_bindings[j]) && !sub_cols[j].is_score &&
          sub_cols[j].corpus_col.table_index == duckdb::DConstants::INVALID_INDEX &&
          sub_cols[j].probe_col.table_index == duckdb::DConstants::INVALID_INDEX) {
        return decline(21);  // the probe vector or the raw struct is read above
      }
    }

    auto probe  = std::move(dj.children[pi]);
    auto corpus = std::move(corpus_slot);
    probe->ResolveOperatorTypes();
    corpus->ResolveOperatorTypes();
    return build_topk(slot,
                      std::move(probe),
                      std::move(corpus),
                      probe_vec,
                      corpus_vec,
                      *call,
                      k,
                      sub_bindings,
                      sub_cols,
                      used);
  }

  /// A scalar subquery giving the vector plans as PROJECTION(CASE WHEN count > 1 THEN error()
  /// ELSE first(v) END) <- AGGREGATE(first(v), count_star()). The GPU plan cannot run first() over
  /// an array, and the join needs no aggregate: probe with the subquery's input instead and let the
  /// operator enforce the one-row contract. Leaves @p probe untouched when the shape differs.
  static bool unwrap_scalar_subquery(unique_ptr<LogicalOperator>& probe, ColumnBinding& probe_vec)
  {
    if (probe->type != LogicalOperatorType::LOGICAL_PROJECTION) { return false; }
    auto& proj = probe->Cast<duckdb::LogicalProjection>();
    if (probe_vec.table_index != proj.table_index ||
        probe_vec.column_index >= proj.expressions.size()) {
      return false;
    }
    auto const& e = *proj.expressions[probe_vec.column_index];
    if (e.GetExpressionClass() != ExpressionClass::BOUND_CASE) { return false; }
    auto const& cs = e.Cast<duckdb::BoundCaseExpression>();
    if (!cs.else_expr || cs.else_expr->GetExpressionClass() != ExpressionClass::BOUND_COLUMN_REF ||
        proj.children[0]->type != LogicalOperatorType::LOGICAL_AGGREGATE_AND_GROUP_BY) {
      return false;
    }
    auto& agg = proj.children[0]->Cast<duckdb::LogicalAggregate>();
    if (!agg.groups.empty()) { return false; }
    auto const first_ref = cs.else_expr->Cast<duckdb::BoundColumnRefExpression>().binding;
    if (first_ref.table_index != agg.aggregate_index ||
        first_ref.column_index >= agg.expressions.size()) {
      return false;
    }
    auto const& first =
      agg.expressions[first_ref.column_index]->Cast<duckdb::BoundAggregateExpression>();
    if (first.function.name != "first" || first.children.size() != 1 ||
        first.children[0]->GetExpressionClass() != ExpressionClass::BOUND_COLUMN_REF) {
      return false;
    }
    probe_vec  = first.children[0]->Cast<duckdb::BoundColumnRefExpression>().binding;
    auto input = std::move(agg.children[0]);
    probe      = std::move(input);
    return true;
  }

  /// `ORDER BY distance(a.v, b.v) LIMIT k` over a cross product of the two sides (a scalar
  /// subquery vector is one): TOP_N <- PROJECTION* <- CROSS_PRODUCT. The cross product becomes a
  /// global top-k join producing the k nearest pairs; the TOP_N stays to order them.
  bool try_global_topk(unique_ptr<LogicalOperator>& slot)
  {
    auto& topn = slot->Cast<duckdb::LogicalTopN>();
    if (topn.orders.empty() || topn.limit == 0) { return false; }
    std::vector<duckdb::LogicalProjection*> projs;
    unique_ptr<LogicalOperator>* cur = &topn.children[0];
    while ((*cur)->type == LogicalOperatorType::LOGICAL_PROJECTION) {
      projs.push_back(&(*cur)->Cast<duckdb::LogicalProjection>());
      cur = &(*cur)->children[0];
    }
    if ((*cur)->type != LogicalOperatorType::LOGICAL_CROSS_PRODUCT) { return false; }
    auto& cross_slot = *cur;
    auto const key   = inline_projections(*topn.orders[0].expression, projs);
    auto const call  = as_distance_call(*key);
    if (!call) { return false; }
    bool const asc = topn.orders[0].type == duckdb::OrderType::ASCENDING;
    if ((call->kind == distance_kind::cosine_similarity) == asc) { return false; }
    std::size_t ai = 2;
    for (std::size_t c = 0; c < 2; ++c) {
      if (has_binding(*cross_slot->children[c], call->a) &&
          has_binding(*cross_slot->children[1 - c], call->b)) {
        ai = c;
      }
    }
    if (ai == 2) { return false; }
    auto const est_a      = cross_slot->children[ai]->EstimateCardinality(_context);
    auto const est_b      = cross_slot->children[1 - ai]->EstimateCardinality(_context);
    std::size_t const pi  = est_a <= est_b ? ai : 1 - ai;
    auto const probe_vec  = pi == ai ? call->a : call->b;
    auto const corpus_vec = pi == ai ? call->b : call->a;
    if (vector_may_hold_nulls(_context, *cross_slot->children[pi], probe_vec) ||
        vector_may_hold_nulls(_context, *cross_slot->children[1 - pi], corpus_vec)) {
      return false;
    }
    auto const k = static_cast<std::int64_t>(topn.limit + topn.offset);

    // Nothing above may read the vectors raw; distance calls on the pair become the score.
    bool ok = true;
    for_each_expression_outside(*_root, cross_slot.get(), [&](unique_ptr<Expression>& e) {
      if (reads_vectors_raw(*e, call->a, call->b)) { ok = false; }
    });
    if (!ok) { return false; }
    duckdb::column_binding_set_t used;
    for_each_expression_outside(
      *_root, cross_slot.get(), [&](unique_ptr<Expression>& e) { collect_bindings(*e, used); });

    // The cross product is what the join replaces, not the top-N above it.
    _slot_bindings    = cross_slot->GetColumnBindings();
    _slot_types       = cross_slot->types;
    auto probe        = std::move(cross_slot->children[pi]);
    auto corpus       = std::move(cross_slot->children[1 - pi]);
    auto probe_vec_in = probe_vec;
    bool const scalar = unwrap_scalar_subquery(probe, probe_vec_in);
    probe->ResolveOperatorTypes();
    corpus->ResolveOperatorTypes();
    auto const probe_bindings  = probe->GetColumnBindings();
    auto const corpus_bindings = corpus->GetColumnBindings();
    std::vector<ColumnBinding> corpus_cols;
    for (auto const& b : corpus_bindings) {
      if (used.count(b) && b != corpus_vec) { corpus_cols.push_back(b); }
    }
    // Reuse the top-k builder: every used corpus column is "emitted by the subquery" as itself.
    duckdb::vector<ColumnBinding> passthrough;
    std::vector<join_col> cols;
    for (auto const& b : corpus_cols) {
      passthrough.push_back(b);
      cols.push_back({false, b, call->kind});
    }
    if (!build_topk(cross_slot,
                    std::move(probe),
                    std::move(corpus),
                    probe_vec_in,
                    corpus_vec,
                    *call,
                    k,
                    passthrough,
                    cols,
                    used,
                    vector_join_mode::global_top_k,
                    scalar)) {
      return false;
    }
    return true;
  }

  bool build_topk(unique_ptr<LogicalOperator>& slot,
                  unique_ptr<LogicalOperator> probe,
                  unique_ptr<LogicalOperator> corpus,
                  const ColumnBinding& probe_vec,
                  const ColumnBinding& corpus_vec,
                  const distance_call& call,
                  std::int64_t k,
                  const duckdb::vector<ColumnBinding>& sub_bindings,
                  const std::vector<join_col>& sub_cols,
                  const duckdb::column_binding_set_t& used,
                  vector_join_mode mode = vector_join_mode::per_row_top_k,
                  bool probe_scalar     = false)
  {
    SiriusVectorJoinBindData bind;
    auto& req        = bind.req;
    bool const cos   = call.kind != distance_kind::l2;
    req.mode         = mode;
    req.probe_scalar = probe_scalar;
    if (mode == vector_join_mode::threshold) {
      // Unbounded: every pair passes (a cosine similarity is never below -1).
      req.eps = call.kind == distance_kind::l2
                  ? static_cast<double>(std::numeric_limits<float>::max())
                  : -2.0;
    }
    req.metric      = cos ? "cosine" : "l2";
    req.search_mode = vector_join_search_mode::exact_gemm;
    req.k           = k;
    req.dim         = static_cast<std::int64_t>(call.dim);
    req.output_type = cos ? vector_join_output_type::similarity : vector_join_output_type::distance;
    req.probe_from_scan = true;

    auto const probe_bindings  = probe->GetColumnBindings();
    auto const corpus_bindings = corpus->GetColumnBindings();
    duckdb::vector<duckdb::LogicalType> returned_types;
    duckdb::vector<std::string> returned_names;
    std::vector<std::pair<ColumnBinding, std::size_t>> probe_pos;  // probe binding -> output pos

    // Probe columns read above, directly or as the subquery's copy of a correlated column.
    auto emit = used;
    for (std::size_t j = 0; j < sub_bindings.size(); ++j) {
      if (used.count(sub_bindings[j]) &&
          sub_cols[j].probe_col.table_index != duckdb::DConstants::INVALID_INDEX) {
        emit.insert(sub_cols[j].probe_col);
      }
    }
    // The probe vector is emitted only when something above reads it other than through a
    // distance call on the pair, which reads the score instead.
    bool vector_out = reads_vectors_raw_anywhere(slot.get(), probe_vec, corpus_vec);
    for (std::size_t j = 0; j < sub_bindings.size(); ++j) {
      vector_out =
        vector_out || (used.count(sub_bindings[j]) && sub_cols[j].probe_col == probe_vec);
    }
    req.left.from_relation = true;
    for (std::size_t i = 0; i < probe_bindings.size(); ++i) {
      auto const name = "p" + std::to_string(i);
      req.left.relation_columns.push_back(name);
      if (probe_bindings[i] == probe_vec) { req.left.column = name; }
      if (!emit.count(probe_bindings[i]) || (probe_bindings[i] == probe_vec && !vector_out)) {
        continue;
      }
      req.left.output_columns.push_back(name);
      probe_pos.emplace_back(probe_bindings[i], returned_types.size());
      returned_types.push_back(probe->types[i]);
      returned_names.push_back("left_" + name);
    }
    if (req.left.column.empty()) { return false; }

    // Corpus columns the subquery emits and something above reads.
    std::vector<ColumnBinding> corpus_cols;
    for (std::size_t j = 0; j < sub_bindings.size(); ++j) {
      if (!used.count(sub_bindings[j]) || sub_cols[j].is_score ||
          sub_cols[j].probe_col.table_index != duckdb::DConstants::INVALID_INDEX) {
        continue;
      }
      if (std::find(corpus_cols.begin(), corpus_cols.end(), sub_cols[j].corpus_col) ==
          corpus_cols.end()) {
        corpus_cols.push_back(sub_cols[j].corpus_col);
      }
    }

    // A bare pinned table is searched from its pin; anything else (a filtered or joined corpus)
    // streams in as a relation, so its filter applies before the top-k, as LATERAL's does.
    auto sirius_ctx = _context.registered_state->Get<duckdb::SiriusContext>("sirius_state");
    bool pinned     = false;
    std::uint64_t right_rows = 0;
    std::vector<std::size_t> corpus_out_pos;
    auto const right_base = returned_types.size();
    if (sirius_ctx && corpus->type == LogicalOperatorType::LOGICAL_GET) {
      auto& scan = corpus->Cast<duckdb::LogicalGet>();
      auto table = scan.GetTable();
      if (table && scan.table_filters.filters.empty()) {
        auto const& column_ids = scan.GetColumnIds();
        auto name_of           = [&](const ColumnBinding& b) -> std::optional<std::string> {
          auto const pos =
            scan.projection_ids.empty() ? b.column_index : scan.projection_ids[b.column_index];
          if (pos >= column_ids.size() || !column_ids[pos].HasPrimaryIndex() ||
              column_ids[pos].IsRowIdColumn()) {
            return std::nullopt;
          }
          return scan.names[column_ids[pos].GetPrimaryIndex()];
        };
        std::vector<std::string> out_names;
        bool names_ok = name_of(corpus_vec).has_value();
        for (auto const& b : corpus_cols) {
          auto const n = name_of(b);
          names_ok     = names_ok && n.has_value();
          if (n) { out_names.push_back(*n); }
        }
        if (names_ok) {
          duckdb::vector<duckdb::LogicalType> rt;
          duckdb::vector<std::string> rn;
          vector_join_side side;
          try {
            auto const qualified =
              duckdb::KeywordHelper::WriteQuoted(table->ParentCatalog().GetName(), '"') + "." +
              duckdb::KeywordHelper::WriteQuoted(table->ParentSchema().name, '"') + "." +
              duckdb::KeywordHelper::WriteQuoted(table->name, '"');
            resolve_vector_join_side(_context,
                                     *sirius_ctx,
                                     "right",
                                     qualified,
                                     *name_of(corpus_vec),
                                     table->ParentSchema().name,
                                     out_names,
                                     /*require_pin=*/true,
                                     side,
                                     rt,
                                     rn,
                                     right_rows);
            if (out_names.empty()) {
              side.output_columns.clear();
              rt.clear();
              rn.clear();
            }
            req.right = side;
            for (std::size_t j = 0; j < rt.size(); ++j) {
              returned_types.push_back(rt[j]);
              returned_names.push_back(rn[j]);
            }
            pinned = true;
            use_lists(req,
                      *table,
                      *name_of(corpus_vec),
                      right_rows,
                      static_cast<double>(probe->EstimateCardinality(_context)));
          } catch (std::exception& e) {
            SIRIUS_LOG_DEBUG("[vector_join_rewrite] top-k pinned corpus declined: {}", e.what());
          }
        }
      }
    }
    if (!pinned) {
      req.build_from_scan     = true;
      req.right.from_relation = true;
      for (std::size_t i = 0; i < corpus_bindings.size(); ++i) {
        auto const name = "c" + std::to_string(i);
        req.right.relation_columns.push_back(name);
        if (corpus_bindings[i] == corpus_vec) { req.right.column = name; }
      }
      if (req.right.column.empty()) { return false; }
      for (auto const& b : corpus_cols) {
        auto const i = static_cast<std::size_t>(
          std::find(corpus_bindings.begin(), corpus_bindings.end(), b) - corpus_bindings.begin());
        if (i >= corpus_bindings.size()) { return false; }
        req.right.output_columns.push_back("c" + std::to_string(i));
        returned_types.push_back(corpus->types[i]);
        returned_names.push_back("right_c" + std::to_string(i));
      }
      right_rows = corpus->EstimateCardinality(_context);
    }
    auto const score_pos = returned_types.size();
    returned_types.push_back(duckdb::LogicalType::FLOAT);
    returned_names.push_back(cos ? "similarity" : "distance");

    bind.left_rows         = probe->EstimateCardinality(_context);
    bind.right_rows        = right_rows;
    bind.probe_is_relation = true;
    auto const probe_rows  = bind.left_rows;

    auto& entry = duckdb::Catalog::GetEntry<duckdb::TableFunctionCatalogEntry>(
      _context, SYSTEM_CATALOG, DEFAULT_SCHEMA, "sirius_knn_join_rel");
    auto function          = entry.functions.GetFunctionByOffset(0);
    auto const table_index = _binder.GenerateTableIndex();
    auto get               = duckdb::make_uniq<duckdb::LogicalGet>(
      table_index,
      function,
      duckdb::make_uniq<SiriusVectorJoinBindData>(std::move(bind)),
      returned_types,
      returned_names);
    for (std::size_t i = 0; i < returned_types.size(); ++i) {
      get->AddColumnId(i);
    }
    get->input_table_types = probe->types;
    for (auto const& name :
         get->bind_data->Cast<SiriusVectorJoinBindData>().req.left.relation_columns) {
      get->input_table_names.push_back(name);
    }
    get->children.push_back(std::move(probe));
    if (!pinned) { get->children.push_back(std::move(corpus)); }
    get->SetEstimatedCardinality(std::max<duckdb::idx_t>(probe_rows, 1) *
                                 static_cast<duckdb::idx_t>(k));
    get->ResolveOperatorTypes();
    auto* get_ptr = get.get();

    // The replaced operator's columns keep their positions: probe columns pass through, subquery
    // columns become corpus columns or the score, and one of the pair's vectors -- which only
    // distance calls above read -- carries the score those calls read from now on.
    ColumnBinding const score{table_index, score_pos};
    duckdb::vector<unique_ptr<Expression>> exprs;
    std::optional<std::size_t> score_at;
    for (auto const& b : _slot_bindings) {
      unique_ptr<Expression> e;
      auto const in_probe = std::find_if(
        probe_pos.begin(), probe_pos.end(), [&](auto const& bp) { return bp.first == b; });
      auto const j = static_cast<std::size_t>(
        std::find(sub_bindings.begin(), sub_bindings.end(), b) - sub_bindings.begin());
      if (b == call.a || b == call.b) {
        e = duckdb::make_uniq<duckdb::BoundColumnRefExpression>(duckdb::LogicalType::FLOAT, score);
        if (!score_at) { score_at = exprs.size(); }
      } else if (in_probe != probe_pos.end()) {
        e = duckdb::make_uniq<duckdb::BoundColumnRefExpression>(
          returned_types[in_probe->second], ColumnBinding{table_index, in_probe->second});
      } else if (j < sub_bindings.size() && used.count(b)) {
        auto const as_probe = std::find_if(probe_pos.begin(), probe_pos.end(), [&](auto const& bp) {
          return bp.first == sub_cols[j].probe_col;
        });
        if (sub_cols[j].probe_col.table_index != duckdb::DConstants::INVALID_INDEX) {
          if (as_probe == probe_pos.end()) {
            throw duckdb::InternalException(
              "sirius vector-join rewrite: a correlated probe column is not an output of the join");
          }
          e = duckdb::make_uniq<duckdb::BoundColumnRefExpression>(
            returned_types[as_probe->second], ColumnBinding{table_index, as_probe->second});
        } else if (sub_cols[j].is_score) {
          e = score_expression(_context, sub_cols[j].kind, call.kind, score);
          if (!e) { return false; }
        } else {
          auto const c = static_cast<std::size_t>(
            std::find(corpus_cols.begin(), corpus_cols.end(), sub_cols[j].corpus_col) -
            corpus_cols.begin());
          auto const pos = right_base + c;
          e              = duckdb::make_uniq<duckdb::BoundColumnRefExpression>(returned_types[pos],
                                                                  ColumnBinding{table_index, pos});
        }
      } else if (used.count(b)) {
        throw duckdb::InternalException(
          "sirius vector-join rewrite: a column read above is not an output of the join");
      }
      exprs.push_back(std::move(e));
    }
    auto const p = install(slot, std::move(get), std::move(exprs));
    if (score_at) {
      bool ok = true;
      for_each_expression_outside(*_root, p.op, [&](unique_ptr<Expression>& e) {
        ok = ok && replace_distance_calls(
                     _context, e, call.a, call.b, call.kind, ColumnBinding{p.index, *score_at});
      });
      if (!ok) {
        throw duckdb::InternalException("sirius vector-join rewrite: distance call left unmapped");
      }
    }
    remap_above(p);
    SIRIUS_LOG_INFO("[vector_join_rewrite] {} {} rewritten to sirius_knn_join_rel ({})",
                    cos ? "cosine" : "l2",
                    mode == vector_join_mode::per_row_top_k  ? "LATERAL top-" + std::to_string(k)
                    : mode == vector_join_mode::global_top_k ? "global top-" + std::to_string(k)
                                                             : std::string("all-pairs distance"),
                    pinned ? "pinned corpus" : "corpus relation");
    return true;
  }

  /// Picks the threshold conjunct, if any, whose pair splits across @p probe_side / corpus.
  static std::optional<std::size_t> pick_threshold(
    const std::vector<unique_ptr<Expression>>& conjuncts,
    const std::function<bool(const distance_call&)>& splits)
  {
    for (std::size_t i = 0; i < conjuncts.size(); ++i) {
      auto const pred = as_threshold(*conjuncts[i]);
      if (pred && splits(pred->call)) { return i; }
    }
    return std::nullopt;
  }

  bool try_any_join(unique_ptr<LogicalOperator>& slot)
  {
    auto& join = slot->Cast<duckdb::LogicalAnyJoin>();
    if (join.join_type != duckdb::JoinType::INNER || has_projection_map(join) || !join.condition) {
      return false;
    }
    std::vector<unique_ptr<Expression>> conjuncts;
    split_conjunction(join.condition->Copy(), conjuncts);
    auto& c0       = *join.children[0];
    auto& c1       = *join.children[1];
    auto const idx = pick_threshold(conjuncts, [&](const distance_call& call) {
      return (has_binding(c0, call.a) && has_binding(c1, call.b)) ||
             (has_binding(c0, call.b) && has_binding(c1, call.a));
    });
    if (!idx) { return false; }
    auto const pred = *as_threshold(*conjuncts[*idx]);
    // The smaller side probes: the search is one GEMM per probe batch against the corpus.
    auto const est0      = c0.EstimateCardinality(_context);
    auto const est1      = c1.EstimateCardinality(_context);
    std::size_t const pi = est0 <= est1 ? 0 : 1;
    auto const probe_vec = has_binding(*join.children[pi], pred.call.a) ? pred.call.a : pred.call.b;
    auto const corpus_vec = probe_vec == pred.call.a ? pred.call.b : pred.call.a;
    if (vector_may_hold_nulls(_context, *join.children[pi], probe_vec) ||
        vector_may_hold_nulls(_context, *join.children[1 - pi], corpus_vec)) {
      return false;
    }
    if (!validate_above(slot.get(), conjuncts, *idx, pred)) { return false; }

    auto probe  = std::move(join.children[pi]);
    auto corpus = std::move(join.children[1 - pi]);
    return build(slot,
                 std::move(probe),
                 std::move(corpus),
                 probe_vec,
                 corpus_vec,
                 pred,
                 std::move(conjuncts),
                 *idx);
  }

  bool try_filter(unique_ptr<LogicalOperator>& slot)
  {
    // A projection map only selects which columns the filter passes up; the replacement emits
    // exactly those, by binding.
    auto& filter = slot->Cast<duckdb::LogicalFilter>();
    if (filter.children.size() != 1) { return false; }
    std::vector<unique_ptr<Expression>> conjuncts;
    for (auto& e : filter.expressions) {
      split_conjunction(e->Copy(), conjuncts);
    }
    auto& tree = filter.children[0];
    std::optional<lifted_relation> lifted;
    ColumnBinding probe_vec, corpus_vec;
    auto const idx = pick_threshold(conjuncts, [&](const distance_call& call) {
      for (auto const& [p, c] : {std::pair{call.a, call.b}, std::pair{call.b, call.a}}) {
        if (auto found = find_crossed_relation(_context, tree, p, c)) {
          lifted     = found;
          probe_vec  = p;
          corpus_vec = c;
          return true;
        }
      }
      return false;
    });
    if (!idx || !lifted) { return false; }
    if (vector_may_hold_nulls(_context, *tree, probe_vec) ||
        vector_may_hold_nulls(_context, *tree, corpus_vec)) {
      return false;
    }
    auto const pred = *as_threshold(*conjuncts[*idx]);

    // The lifted relation must feed nothing inside the tree but the cross product, or removing it
    // from the tree would strand a reference. An inequality join's own conditions do read it;
    // they move above the join with the other conjuncts.
    auto& cross       = **lifted->cross_slot;
    auto& probe_rel   = *cross.children[lifted->probe_child];
    auto const pbinds = probe_rel.GetColumnBindings();
    duckdb::column_binding_set_t probe_bindings;
    for (auto const& b : pbinds) {
      probe_bindings.insert(b);
    }
    bool stranded                               = false;
    std::function<void(LogicalOperator&)> visit = [&](LogicalOperator& op) {
      if (&op == &probe_rel) { return; }
      if (&op != &cross) {
        duckdb::LogicalOperatorVisitor::EnumerateExpressions(op, [&](unique_ptr<Expression>* e) {
          duckdb::column_binding_set_t used;
          collect_bindings(**e, used);
          for (auto const& b : used) {
            if (probe_bindings.count(b)) { stranded = true; }
          }
        });
      }
      for (auto& child : op.children) {
        if (child) { visit(*child); }
      }
    };
    visit(*tree);
    if (stranded) { return false; }
    if (cross.type == LogicalOperatorType::LOGICAL_COMPARISON_JOIN) {
      for (auto const& c : cross.Cast<duckdb::LogicalComparisonJoin>().conditions) {
        conjuncts.push_back(duckdb::make_uniq<duckdb::BoundComparisonExpression>(
          c.comparison, c.left->Copy(), c.right->Copy()));
      }
    }
    if (!validate_above(slot.get(), conjuncts, *idx, pred)) { return false; }

    auto probe          = std::move(cross.children[lifted->probe_child]);
    auto rest           = std::move(cross.children[1 - lifted->probe_child]);
    *lifted->cross_slot = std::move(rest);
    auto corpus         = std::move(filter.children[0]);
    return build(slot,
                 std::move(probe),
                 std::move(corpus),
                 probe_vec,
                 corpus_vec,
                 pred,
                 std::move(conjuncts),
                 *idx);
  }

  /// Nothing above the join, and no residual conjunct, may read the vectors other than through a
  /// distance call on the pair (the join does not emit them), and every such call must be
  /// expressible through the score.
  bool validate_above(const LogicalOperator* node,
                      const std::vector<unique_ptr<Expression>>& conjuncts,
                      std::size_t picked,
                      const threshold_predicate& pred)
  {
    auto const& a  = pred.call.a;
    auto const& b  = pred.call.b;
    bool ok        = true;
    bool const cos = pred.call.kind != distance_kind::l2;
    auto check     = [&](const Expression& e) {
      if (reads_vectors_raw(e, a, b)) { ok = false; }
      std::function<void(const Expression&)> metric_ok = [&](const Expression& x) {
        if (auto const call = as_distance_call(x); call && same_pair(*call, a, b)) {
          if ((call->kind != distance_kind::l2) != cos) { ok = false; }
          return;
        }
        duckdb::ExpressionIterator::EnumerateChildren(x, metric_ok);
      };
      metric_ok(e);
    };
    for_each_expression_outside(*_root, node, [&](unique_ptr<Expression>& e) { check(*e); });
    for (std::size_t i = 0; i < conjuncts.size(); ++i) {
      if (i != picked) { check(*conjuncts[i]); }
    }
    return ok;
  }

  /// The threshold forms' last step. Inside @p replacement (the residual filter, and the corpus
  /// subtree when the join sits in place of its scan) the old columns and distance calls are read
  /// from the join directly. Above it, each column the replaced operator emitted keeps its
  /// position: an output of the join, a column still produced by the corpus subtree, or -- for one
  /// of the pair's vectors, which only distance calls read -- the score, which those calls then
  /// read.
  void finish_threshold(unique_ptr<LogicalOperator>& slot,
                        unique_ptr<LogicalOperator> replacement,
                        LogicalOperator* get,
                        const std::vector<std::pair<ColumnBinding, std::size_t>>& remap,
                        const ColumnBinding& score,
                        const distance_call& call,
                        const duckdb::column_binding_set_t& used)
  {
    auto const& get_types = get->Cast<duckdb::LogicalGet>().returned_types;
    bool ok               = true;
    for_each_expression_outside(*replacement, get, [&](unique_ptr<Expression>& e) {
      ok = ok && replace_distance_calls(_context, e, call.a, call.b, call.kind, score);
    });
    duckdb::ColumnBindingReplacer inner;
    inner.stop_operator = get;
    for (auto const& [old_binding, pos] : remap) {
      inner.replacement_bindings.emplace_back(old_binding, ColumnBinding{score.table_index, pos});
    }
    inner.VisitOperator(*replacement);

    duckdb::vector<unique_ptr<Expression>> exprs;
    std::optional<std::size_t> score_at;
    for (std::size_t i = 0; i < _slot_bindings.size(); ++i) {
      auto const& b = _slot_bindings[i];
      auto const it =
        std::find_if(remap.begin(), remap.end(), [&](auto const& r) { return r.first == b; });
      unique_ptr<Expression> e;
      if (b == call.a || b == call.b) {
        e = duckdb::make_uniq<duckdb::BoundColumnRefExpression>(duckdb::LogicalType::FLOAT, score);
        if (!score_at) { score_at = i; }
      } else if (it != remap.end()) {
        e = duckdb::make_uniq<duckdb::BoundColumnRefExpression>(
          get_types[it->second], ColumnBinding{score.table_index, it->second});
      } else if (used.count(b)) {
        e = duckdb::make_uniq<duckdb::BoundColumnRefExpression>(_slot_types[i], b);
      }
      exprs.push_back(std::move(e));
    }
    auto const p = install(slot, std::move(replacement), std::move(exprs));
    if (score_at) {
      for_each_expression_outside(*_root, p.op, [&](unique_ptr<Expression>& e) {
        ok = ok && replace_distance_calls(
                     _context, e, call.a, call.b, call.kind, ColumnBinding{p.index, *score_at});
      });
    }
    if (!ok) {
      // validate_above ruled this out; a failure here means a shape it did not anticipate.
      throw duckdb::InternalException("sirius vector-join rewrite: distance call left unmapped");
    }
    remap_above(p);
  }

  bool build(unique_ptr<LogicalOperator>& slot,
             unique_ptr<LogicalOperator> probe,
             unique_ptr<LogicalOperator> corpus,
             const ColumnBinding& probe_vec,
             const ColumnBinding& corpus_vec,
             const threshold_predicate& pred,
             std::vector<unique_ptr<Expression>> conjuncts,
             std::size_t picked)
  {
    auto* node = slot.get();
    probe->ResolveOperatorTypes();
    corpus->ResolveOperatorTypes();
    if (build_on_pinned_table(
          slot, probe, corpus, probe_vec, corpus_vec, pred, conjuncts, picked)) {
      return true;
    }
    auto const probe_bindings  = probe->GetColumnBindings();
    auto const corpus_bindings = corpus->GetColumnBindings();

    // Columns the rest of the plan reads, which are the only ones the join has to emit.
    duckdb::column_binding_set_t used;
    for_each_expression_outside(
      *_root, node, [&](unique_ptr<Expression>& e) { collect_bindings(*e, used); });
    for (std::size_t i = 0; i < conjuncts.size(); ++i) {
      if (i != picked) { collect_bindings(*conjuncts[i], used); }
    }

    SiriusVectorJoinBindData bind;
    auto& req       = bind.req;
    bool const cos  = pred.call.kind != distance_kind::l2;
    req.mode        = vector_join_mode::threshold;
    req.metric      = cos ? "cosine" : "l2";
    req.search_mode = vector_join_search_mode::exact_gemm;
    req.k           = 10;
    req.dim         = static_cast<std::int64_t>(pred.call.dim);
    req.output_type = cos ? vector_join_output_type::similarity : vector_join_output_type::distance;
    // A cosine-distance bound d is the similarity bound 1 - d.
    req.eps = pred.call.kind == distance_kind::cosine_distance ? 1.0 - pred.bound : pred.bound;
    req.probe_from_scan = true;
    req.build_from_scan = true;

    duckdb::vector<duckdb::LogicalType> returned_types;
    duckdb::vector<std::string> returned_names;
    std::vector<std::pair<ColumnBinding, std::size_t>> remap;  // old binding -> output position

    auto describe_side = [&](const char* prefix,
                             LogicalOperator& rel,
                             const duckdb::vector<ColumnBinding>& bindings,
                             const ColumnBinding& vec,
                             vector_join_side& side,
                             const char* out_prefix) {
      side.from_relation = true;
      for (std::size_t i = 0; i < bindings.size(); ++i) {
        auto const name = std::string(prefix) + std::to_string(i);
        side.relation_columns.push_back(name);
        if (bindings[i] == vec) {
          side.column = name;
          continue;
        }
        if (!used.count(bindings[i])) { continue; }
        side.output_columns.push_back(name);
        remap.emplace_back(bindings[i], returned_types.size());
        returned_types.push_back(rel.types[i]);
        returned_names.push_back(std::string(out_prefix) + name);
      }
    };
    describe_side("p", *probe, probe_bindings, probe_vec, req.left, "left_");
    describe_side("c", *corpus, corpus_bindings, corpus_vec, req.right, "right_");
    if (req.left.column.empty() || req.right.column.empty()) { return false; }
    auto const score_pos = returned_types.size();
    returned_types.push_back(duckdb::LogicalType::FLOAT);
    returned_names.push_back(cos ? "similarity" : "distance");

    auto const probe_rows  = probe->EstimateCardinality(_context);
    auto const corpus_rows = corpus->EstimateCardinality(_context);
    bind.left_rows         = probe_rows;
    bind.right_rows        = corpus_rows;
    bind.probe_is_relation = true;

    auto& entry = duckdb::Catalog::GetEntry<duckdb::TableFunctionCatalogEntry>(
      _context, SYSTEM_CATALOG, DEFAULT_SCHEMA, "sirius_knn_join_rel");
    auto function = entry.functions.GetFunctionByOffset(0);

    auto const table_index = _binder.GenerateTableIndex();
    auto get               = duckdb::make_uniq<duckdb::LogicalGet>(
      table_index,
      function,
      duckdb::make_uniq<SiriusVectorJoinBindData>(std::move(bind)),
      returned_types,
      returned_names);
    for (std::size_t i = 0; i < returned_types.size(); ++i) {
      get->AddColumnId(i);
    }
    get->input_table_types = probe->types;
    for (auto const& name :
         get->bind_data->Cast<SiriusVectorJoinBindData>().req.left.relation_columns) {
      get->input_table_names.push_back(name);
    }
    get->children.push_back(std::move(probe));
    get->children.push_back(std::move(corpus));
    get->SetEstimatedCardinality(std::max<duckdb::idx_t>(probe_rows, 1) * 10);
    get->ResolveOperatorTypes();
    auto* get_ptr = get.get();

    ColumnBinding const score{table_index, score_pos};
    // Residual conjuncts: everything but the threshold, plus the strict edge the inclusive
    // search admits (distance == bound), all over the join's outputs. One that reads only corpus
    // columns filters the corpus instead, before the search.
    std::vector<unique_ptr<Expression>> residual;
    std::vector<unique_ptr<Expression>> corpus_only;
    duckdb::column_binding_set_t corpus_set;
    for (auto const& cb : corpus_bindings) {
      corpus_set.insert(cb);
    }
    for (std::size_t i = 0; i < conjuncts.size(); ++i) {
      if (i == picked) { continue; }
      duckdb::column_binding_set_t reads;
      collect_bindings(*conjuncts[i], reads);
      bool const only_corpus =
        !reads.empty() &&
        std::all_of(reads.begin(), reads.end(), [&](auto& r) { return corpus_set.count(r) > 0; });
      (only_corpus ? corpus_only : residual).push_back(std::move(conjuncts[i]));
    }
    if (pred.strict) {
      auto const edge = static_cast<float>(req_eps_in_score_space(pred));
      residual.push_back(duckdb::make_uniq<duckdb::BoundComparisonExpression>(
        cos ? ExpressionType::COMPARE_GREATERTHAN : ExpressionType::COMPARE_LESSTHAN,
        duckdb::make_uniq<duckdb::BoundColumnRefExpression>(duckdb::LogicalType::FLOAT, score),
        duckdb::make_uniq<duckdb::BoundConstantExpression>(duckdb::Value::FLOAT(edge))));
    }

    if (!corpus_only.empty()) {
      auto corpus_filter = duckdb::make_uniq<duckdb::LogicalFilter>();
      for (auto& e : corpus_only) {
        corpus_filter->expressions.push_back(std::move(e));
      }
      corpus_filter->children.push_back(std::move(get->children[1]));
      corpus_filter->ResolveOperatorTypes();
      get->children[1] = std::move(corpus_filter);
    }
    unique_ptr<LogicalOperator> replacement = std::move(get);
    if (!residual.empty()) {
      auto filter = duckdb::make_uniq<duckdb::LogicalFilter>();
      for (auto& e : residual) {
        filter->expressions.push_back(std::move(e));
      }
      filter->children.push_back(std::move(replacement));
      replacement = std::move(filter);
    }
    finish_threshold(slot, std::move(replacement), get_ptr, remap, score, pred.call, used);
    SIRIUS_LOG_INFO(
      "[vector_join_rewrite] {} threshold join rewritten to sirius_knn_join_rel "
      "(probe ~{} rows, corpus ~{} rows)",
      req_metric_name(pred),
      probe_rows,
      corpus_rows);
    return true;
  }

  /// The slot holding the LogicalGet whose table_index is @p table_index, under @p slot.
  static unique_ptr<LogicalOperator>* find_get(unique_ptr<LogicalOperator>& slot,
                                               duckdb::idx_t table_index)
  {
    if (slot->type == LogicalOperatorType::LOGICAL_GET &&
        slot->Cast<duckdb::LogicalGet>().table_index == table_index) {
      return &slot;
    }
    for (auto& child : slot->children) {
      if (auto* found = find_get(child, table_index)) { return found; }
    }
    return nullptr;
  }

  /// Vector-first: search the pinned table the corpus vector is scanned from, in place of that
  /// scan, and leave the corpus's relational operators above it. Worth it when the probe side is
  /// small -- a pinned corpus is searched at memory bandwidth, while filtering it first drags every
  /// row's vector through the relational join -- or when the corpus is that table alone. False when
  /// it does not apply; nothing has been changed then.
  /// The probe side read from its pin when @p probe is a scan of a pinned table with no filters:
  /// fills @p side and the probe columns read above (@p used) into the returned columns. False,
  /// leaving everything as it was, when it is anything else or the table is not pinned.
  bool pinned_probe_side(const unique_ptr<LogicalOperator>& probe,
                         const ColumnBinding& probe_vec,
                         const duckdb::column_binding_set_t& used,
                         vector_join_side& side,
                         duckdb::vector<duckdb::LogicalType>& returned_types,
                         duckdb::vector<std::string>& returned_names,
                         std::vector<std::pair<ColumnBinding, std::size_t>>& remap,
                         std::uint64_t& rows)
  {
    auto sirius_ctx = _context.registered_state->Get<duckdb::SiriusContext>("sirius_state");
    if (!sirius_ctx || probe->type != LogicalOperatorType::LOGICAL_GET) { return false; }
    auto& scan = probe->Cast<duckdb::LogicalGet>();
    auto table = scan.GetTable();
    if (!table || !scan.table_filters.filters.empty() ||
        probe_vec.table_index != scan.table_index) {
      return false;
    }
    auto const& column_ids = scan.GetColumnIds();
    auto const bindings    = scan.GetColumnBindings();
    auto name_of           = [&](std::size_t binding_col) -> std::optional<std::string> {
      auto const pos = scan.projection_ids.empty() ? binding_col : scan.projection_ids[binding_col];
      if (pos >= column_ids.size() || !column_ids[pos].HasPrimaryIndex() ||
          column_ids[pos].IsRowIdColumn()) {
        return std::nullopt;
      }
      return scan.names[column_ids[pos].GetPrimaryIndex()];
    };
    auto const vec_name = name_of(probe_vec.column_index);
    if (!vec_name) { return false; }
    std::vector<std::string> out;
    std::vector<ColumnBinding> old;
    for (std::size_t i = 0; i < bindings.size(); ++i) {
      if (!used.count(bindings[i]) || bindings[i] == probe_vec) { continue; }
      auto const name = name_of(i);
      if (!name) { return false; }
      if (std::find(out.begin(), out.end(), *name) == out.end()) {
        out.push_back(*name);
        old.push_back(bindings[i]);
      }
    }
    vector_join_side resolved;
    duckdb::vector<duckdb::LogicalType> types;
    duckdb::vector<std::string> names;
    try {
      auto const qualified =
        duckdb::KeywordHelper::WriteQuoted(table->ParentCatalog().GetName(), '"') + "." +
        duckdb::KeywordHelper::WriteQuoted(table->ParentSchema().name, '"') + "." +
        duckdb::KeywordHelper::WriteQuoted(table->name, '"');
      resolve_vector_join_side(_context,
                               *sirius_ctx,
                               "left",
                               qualified,
                               *vec_name,
                               table->ParentSchema().name,
                               out,
                               /*require_pin=*/true,
                               resolved,
                               types,
                               names,
                               rows);
    } catch (std::exception& e) {
      SIRIUS_LOG_DEBUG("[vector_join_rewrite] pinned probe declined: {}", e.what());
      return false;
    }
    // The resolver reads "no output columns" as "all of them".
    if (out.empty()) {
      resolved.output_columns.clear();
      types.clear();
      names.clear();
    }
    side = std::move(resolved);
    for (std::size_t j = 0; j < out.size(); ++j) {
      remap.emplace_back(old[j], returned_types.size());
      returned_types.push_back(types[j]);
      returned_names.push_back(names[j]);
    }
    return true;
  }

  bool build_on_pinned_table(unique_ptr<LogicalOperator>& slot,
                             unique_ptr<LogicalOperator>& probe,
                             unique_ptr<LogicalOperator>& corpus,
                             const ColumnBinding& probe_vec,
                             const ColumnBinding& corpus_vec,
                             const threshold_predicate& pred,
                             std::vector<unique_ptr<Expression>>& conjuncts,
                             std::size_t picked)
  {
    auto sirius_ctx = _context.registered_state->Get<duckdb::SiriusContext>("sirius_state");
    if (!sirius_ctx) { return false; }
    auto* get_slot = find_get(corpus, corpus_vec.table_index);
    if (get_slot == nullptr) { return false; }
    auto& scan = (*get_slot)->Cast<duckdb::LogicalGet>();
    auto table = scan.GetTable();
    if (!table) { return false; }
    // Projection maps between the corpus root and the scan select child columns by position,
    // and the join that replaces the scan lays its columns out differently. They only prune, and
    // everything above refers to columns by binding, so they are dropped on that path.
    std::function<bool(LogicalOperator&)> clear_maps_to_scan = [&](LogicalOperator& op) {
      if (&op == get_slot->get()) { return true; }
      for (auto& child : op.children) {
        if (child && clear_maps_to_scan(*child)) {
          if (op.type == LogicalOperatorType::LOGICAL_FILTER) {
            op.Cast<duckdb::LogicalFilter>().projection_map.clear();
          } else if (op.type == LogicalOperatorType::LOGICAL_COMPARISON_JOIN ||
                     op.type == LogicalOperatorType::LOGICAL_ANY_JOIN ||
                     op.type == LogicalOperatorType::LOGICAL_DELIM_JOIN ||
                     op.type == LogicalOperatorType::LOGICAL_ASOF_JOIN) {
            op.Cast<duckdb::LogicalJoin>().left_projection_map.clear();
            op.Cast<duckdb::LogicalJoin>().right_projection_map.clear();
          }
          return true;
        }
      }
      return false;
    };
    auto const probe_rows      = probe->EstimateCardinality(_context);
    auto const* env            = std::getenv("SIRIUS_VSS_REWRITE_VECTOR_FIRST_PROBES");
    auto const max_probes      = env != nullptr ? std::strtoull(env, nullptr, 10) : 256ULL;
    bool const corpus_is_table = get_slot == &corpus;
    if (!corpus_is_table && probe_rows > max_probes) { return false; }

    // The scan's output columns, by table column name.
    auto const& column_ids   = scan.GetColumnIds();
    auto const scan_bindings = scan.GetColumnBindings();
    auto name_of             = [&](std::size_t binding_col) -> std::optional<std::string> {
      auto const pos = scan.projection_ids.empty() ? binding_col : scan.projection_ids[binding_col];
      if (pos >= column_ids.size() || !column_ids[pos].HasPrimaryIndex() ||
          column_ids[pos].IsRowIdColumn()) {
        return std::nullopt;
      }
      return scan.names[column_ids[pos].GetPrimaryIndex()];
    };
    auto const vec_name = name_of(corpus_vec.column_index);
    if (!vec_name) { return false; }

    duckdb::column_binding_set_t used;
    for_each_expression_outside(
      *_root, slot.get(), [&](unique_ptr<Expression>& e) { collect_bindings(*e, used); });
    for_each_expression_outside(
      *corpus, get_slot->get(), [&](unique_ptr<Expression>& e) { collect_bindings(*e, used); });
    for (std::size_t i = 0; i < conjuncts.size(); ++i) {
      if (i != picked) { collect_bindings(*conjuncts[i], used); }
    }
    // Filters DuckDB pushed into the scan still have to hold; they become a filter over the join,
    // which therefore also emits the columns they read.
    std::vector<std::pair<std::size_t, const duckdb::TableFilter*>> pushed;
    for (auto const& [key, filter] : scan.table_filters.filters) {
      pushed.emplace_back(key, filter.get());
    }

    std::vector<std::string> right_out;
    std::vector<ColumnBinding> right_old;
    auto want_right = [&](std::size_t binding_col) {
      auto const name = name_of(binding_col);
      if (!name || *name == *vec_name) { return name.has_value(); }
      if (std::find(right_out.begin(), right_out.end(), *name) == right_out.end()) {
        right_out.push_back(*name);
        right_old.push_back(scan_bindings[binding_col]);
      }
      return true;
    };
    for (std::size_t i = 0; i < scan_bindings.size(); ++i) {
      if (used.count(scan_bindings[i]) && !want_right(i)) { return false; }
    }
    std::vector<std::pair<std::string, const duckdb::TableFilter*>> pushed_by_name;
    for (auto const& [key, filter] : pushed) {
      // table_filters are keyed by the table's column index (TableFilterSet::PushFilter keys by
      // the ColumnIndex's primary index), not by position in column_ids.
      if (key >= scan.names.size()) { return false; }
      auto const name = scan.names[key];
      pushed_by_name.emplace_back(name, filter);
      if (std::find(right_out.begin(), right_out.end(), name) == right_out.end()) {
        right_out.push_back(name);
        right_old.push_back(ColumnBinding{});  // not referenced above, only filtered on
      }
    }

    SiriusVectorJoinBindData bind;
    auto& req = bind.req;
    duckdb::vector<duckdb::LogicalType> right_types;
    duckdb::vector<std::string> right_names;
    std::uint64_t right_rows = 0;
    try {
      auto const qualified =
        duckdb::KeywordHelper::WriteQuoted(table->ParentCatalog().GetName(), '"') + "." +
        duckdb::KeywordHelper::WriteQuoted(table->ParentSchema().name, '"') + "." +
        duckdb::KeywordHelper::WriteQuoted(table->name, '"');
      resolve_vector_join_side(_context,
                               *sirius_ctx,
                               "right",
                               qualified,
                               *vec_name,
                               table->ParentSchema().name,
                               right_out,
                               /*require_pin=*/true,
                               req.right,
                               right_types,
                               right_names,
                               right_rows);
    } catch (std::exception& e) {
      SIRIUS_LOG_DEBUG("[vector_join_rewrite] vector-first declined: {}", e.what());
      return false;
    }
    // The resolver reads "no output columns" as "all of them", the table function's default.
    // Nothing above reads the corpus columns here, so the join emits none.
    if (right_out.empty()) {
      req.right.output_columns.clear();
      right_types.clear();
      right_names.clear();
    }

    bool const cos  = pred.call.kind != distance_kind::l2;
    req.mode        = vector_join_mode::threshold;
    req.metric      = cos ? "cosine" : "l2";
    req.search_mode = vector_join_search_mode::exact_gemm;
    req.k           = 10;
    req.dim         = static_cast<std::int64_t>(pred.call.dim);
    req.output_type = cos ? vector_join_output_type::similarity : vector_join_output_type::distance;
    req.eps = pred.call.kind == distance_kind::cosine_distance ? 1.0 - pred.bound : pred.bound;
    duckdb::vector<duckdb::LogicalType> returned_types;
    duckdb::vector<std::string> returned_names;
    std::vector<std::pair<ColumnBinding, std::size_t>> remap;
    // A probe that is a pinned table read whole -- a self-join over it, say -- is searched from its
    // pin like the corpus rather than streamed through the plan, which would put a second copy of
    // the table on the device beside the pin (2.4M reviews x 1024: out of room).
    std::uint64_t left_rows = 0;
    bool const probe_pinned = pinned_probe_side(
      probe, probe_vec, used, req.left, returned_types, returned_names, remap, left_rows);
    req.probe_from_scan = !probe_pinned;
    if (!probe_pinned) {
      // Probe side: the lifted relation, as in the filter-first form.
      auto const probe_bindings = probe->GetColumnBindings();
      req.left.from_relation    = true;
      for (std::size_t i = 0; i < probe_bindings.size(); ++i) {
        auto const name = "p" + std::to_string(i);
        req.left.relation_columns.push_back(name);
        if (probe_bindings[i] == probe_vec) {
          req.left.column = name;
          continue;
        }
        if (!used.count(probe_bindings[i])) { continue; }
        req.left.output_columns.push_back(name);
        remap.emplace_back(probe_bindings[i], returned_types.size());
        returned_types.push_back(probe->types[i]);
        returned_names.push_back("left_" + name);
      }
      if (req.left.column.empty()) { return false; }
    }
    auto const right_base = returned_types.size();
    for (std::size_t j = 0; j < right_out.size(); ++j) {
      if (right_old[j].table_index != duckdb::DConstants::INVALID_INDEX) {
        remap.emplace_back(right_old[j], right_base + j);
      }
      returned_types.push_back(right_types[j]);
      returned_names.push_back(right_names[j]);
    }
    auto const score_pos = returned_types.size();
    returned_types.push_back(duckdb::LogicalType::FLOAT);
    returned_names.push_back(cos ? "similarity" : "distance");
    bind.left_rows         = probe_pinned ? left_rows : probe_rows;
    bind.right_rows        = right_rows;
    bind.probe_is_relation = !probe_pinned;

    auto& entry = duckdb::Catalog::GetEntry<duckdb::TableFunctionCatalogEntry>(
      _context, SYSTEM_CATALOG, DEFAULT_SCHEMA, "sirius_knn_join_rel");
    auto function          = entry.functions.GetFunctionByOffset(0);
    auto const table_index = _binder.GenerateTableIndex();
    auto get               = duckdb::make_uniq<duckdb::LogicalGet>(
      table_index,
      function,
      duckdb::make_uniq<SiriusVectorJoinBindData>(std::move(bind)),
      returned_types,
      returned_names);
    for (std::size_t i = 0; i < returned_types.size(); ++i) {
      get->AddColumnId(i);
    }
    if (!probe_pinned) {
      get->input_table_types = probe->types;
      for (auto const& name :
           get->bind_data->Cast<SiriusVectorJoinBindData>().req.left.relation_columns) {
        get->input_table_names.push_back(name);
      }
      get->children.push_back(std::move(probe));
    }
    get->SetEstimatedCardinality(std::max<duckdb::idx_t>(probe_rows, 1) * 10);
    get->ResolveOperatorTypes();
    auto* get_ptr = get.get();

    unique_ptr<LogicalOperator> in_place = std::move(get);
    // A constant comparison the operator evaluates itself, masking the corpus before the search
    // (as for the table function's own pushdown); anything else filters the join's output, and so
    // does everything when the search goes through the lists, which take no corpus predicates.
    bool const via_lists = use_lists(
      get_ptr->bind_data->Cast<SiriusVectorJoinBindData>().req,
      *table,
      *vec_name,
      right_rows,
      static_cast<double>(get_ptr->bind_data->Cast<SiriusVectorJoinBindData>().left_rows));
    auto device_filter = [&](const duckdb::TableFilter& f, const duckdb::LogicalType& column) {
      if (via_lists) { return false; }
      auto comparable = [](const duckdb::LogicalType& t) {
        return (t.IsIntegral() && t.id() != duckdb::LogicalTypeId::HUGEINT &&
                t.id() != duckdb::LogicalTypeId::UHUGEINT) ||
               t.id() == duckdb::LogicalTypeId::FLOAT || t.id() == duckdb::LogicalTypeId::DOUBLE ||
               t.id() == duckdb::LogicalTypeId::BOOLEAN || t.id() == duckdb::LogicalTypeId::VARCHAR;
      };
      auto constant_ok = [&](const duckdb::TableFilter& c) {
        if (c.filter_type != duckdb::TableFilterType::CONSTANT_COMPARISON) { return false; }
        auto const& value = c.Cast<duckdb::ConstantFilter>().constant;
        return !value.IsNull() && comparable(value.type());
      };
      if (!comparable(column)) { return false; }
      if (f.filter_type == duckdb::TableFilterType::CONJUNCTION_AND) {
        auto const& children = f.Cast<duckdb::ConjunctionAndFilter>().child_filters;
        return std::all_of(
          children.begin(), children.end(), [&](auto const& c) { return constant_ok(*c); });
      }
      return constant_ok(f);
    };
    auto filter = duckdb::make_uniq<duckdb::LogicalFilter>();
    for (auto const& [name, table_filter] : pushed_by_name) {
      auto const j = static_cast<std::size_t>(std::find(right_out.begin(), right_out.end(), name) -
                                              right_out.begin());
      if (device_filter(*table_filter, right_types[j])) {
        get_ptr->table_filters.PushFilter(duckdb::ColumnIndex(right_base + j),
                                          table_filter->Copy());
        continue;
      }
      duckdb::BoundColumnRefExpression column(right_types[j],
                                              ColumnBinding{table_index, right_base + j});
      filter->expressions.push_back(table_filter->ToExpression(column));
    }
    if (!filter->expressions.empty()) {
      filter->children.push_back(std::move(in_place));
      filter->ResolveOperatorTypes();
      in_place = std::move(filter);
    }
    clear_maps_to_scan(*corpus);
    *get_slot = std::move(in_place);

    ColumnBinding const score{table_index, score_pos};
    std::vector<unique_ptr<Expression>> residual;
    for (std::size_t i = 0; i < conjuncts.size(); ++i) {
      if (i != picked) { residual.push_back(std::move(conjuncts[i])); }
    }
    if (pred.strict) {
      residual.push_back(duckdb::make_uniq<duckdb::BoundComparisonExpression>(
        cos ? ExpressionType::COMPARE_GREATERTHAN : ExpressionType::COMPARE_LESSTHAN,
        duckdb::make_uniq<duckdb::BoundColumnRefExpression>(duckdb::LogicalType::FLOAT, score),
        duckdb::make_uniq<duckdb::BoundConstantExpression>(
          duckdb::Value::FLOAT(static_cast<float>(req_eps_in_score_space(pred))))));
    }
    unique_ptr<LogicalOperator> replacement = std::move(corpus);
    if (!residual.empty()) {
      auto filter = duckdb::make_uniq<duckdb::LogicalFilter>();
      for (auto& e : residual) {
        filter->expressions.push_back(std::move(e));
      }
      filter->children.push_back(std::move(replacement));
      replacement = std::move(filter);
    }
    finish_threshold(slot, std::move(replacement), get_ptr, remap, score, pred.call, used);
    SIRIUS_LOG_INFO(
      "[vector_join_rewrite] {} threshold join rewritten vector-first over pinned '{}' "
      "(probe ~{} rows{})",
      req_metric_name(pred),
      table->name,
      probe_rows,
      probe_pinned ? ", read from its pin" : "");
    return true;
  }

  static double req_eps_in_score_space(const threshold_predicate& pred)
  {
    return pred.call.kind == distance_kind::cosine_distance ? 1.0 - pred.bound : pred.bound;
  }

  static const char* req_metric_name(const threshold_predicate& pred)
  {
    return pred.call.kind == distance_kind::l2 ? "l2" : "cosine";
  }

  duckdb::ClientContext& _context;
  bool _inlined{false};
  duckdb::Binder& _binder;
  unique_ptr<LogicalOperator>& _root;
  /// The columns and types of the operator try_one last handed to a pattern.
  duckdb::vector<ColumnBinding> _slot_bindings;
  duckdb::vector<duckdb::LogicalType> _slot_types;
};

}  // namespace

namespace {

bool expression_has_vector_distance(Expression& e)
{
  if (e.GetExpressionClass() == duckdb::ExpressionClass::BOUND_FUNCTION) {
    auto const& name = e.Cast<duckdb::BoundFunctionExpression>().function.name;
    if (name == "array_distance" || name == "array_cosine_similarity" ||
        name == "array_cosine_distance") {
      return true;
    }
  }
  bool found = false;
  duckdb::ExpressionIterator::EnumerateChildren(
    e, [&](Expression& child) { found = found || expression_has_vector_distance(child); });
  return found;
}

}  // namespace

bool plan_has_vector_distance(LogicalOperator& plan)
{
  bool found = false;
  duckdb::LogicalOperatorVisitor::EnumerateExpressions(
    plan, [&](unique_ptr<Expression>* e) { found = found || expression_has_vector_distance(**e); });
  for (auto& child : plan.children) {
    found = found || (child && plan_has_vector_distance(*child));
  }
  return found;
}

std::size_t rewrite_plain_sql_vector_joins(duckdb::ClientContext& context,
                                           duckdb::Binder& binder,
                                           duckdb::unique_ptr<duckdb::LogicalOperator>& plan,
                                           bool* inlined_ctes)
{
  if (!plan || !rewrite_enabled(context)) { return 0; }
  try {
    rewriter r(context, binder, plan);
    auto const n = r.run();
    if (inlined_ctes != nullptr) { *inlined_ctes = r.inlined(); }
    return n;
  } catch (duckdb::InternalException&) {
    throw;
  } catch (std::exception& e) {
    SIRIUS_LOG_DEBUG("[vector_join_rewrite] declined: {}", e.what());
    return 0;
  }
}

}  // namespace sirius::vss
