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

/**
 * @file plan_shape_test_utils.hpp
 * @brief Shared scaffolding for the plan-shape suites (test_plan_tree_shape.cpp,
 *        test_eager_agg_pushdown.cpp): an on-disk scratch database, an RAII
 *        setting override, SQL -> optimized-logical / Sirius-physical plan
 *        builders, and delim-join-aware tree traversal helpers.
 */

#include "op/sirius_physical_delim_join.hpp"
#include "op/sirius_physical_operator.hpp"
#include "planner/sirius_physical_plan_generator.hpp"

#include <catch.hpp>
#include <duckdb.hpp>
#include <duckdb/execution/column_binding_resolver.hpp>
#include <duckdb/main/config.hpp>
#include <duckdb/optimizer/optimizer.hpp>
#include <duckdb/parser/parser.hpp>
#include <duckdb/planner/logical_operator.hpp>
#include <duckdb/planner/planner.hpp>
#include <unistd.h>

#include <cstddef>
#include <cstdio>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace sirius::test {

/// RAII on-disk DuckDB path: the GPU-native seq_scan ingestible refuses non-single-file
/// block managers, so these tests need an on-disk database rather than :memory:.
class scoped_temp_db_path {
 public:
  /// @p prefix names the scratch file so concurrent suites never collide.
  explicit scoped_temp_db_path(const std::string& prefix)
  {
    std::string tmpl = "/tmp/" + prefix + "_XXXXXX";
    int fd           = ::mkstemp(tmpl.data());
    REQUIRE(fd >= 0);
    ::close(fd);
    ::unlink(tmpl.c_str());
    _path = tmpl;
  }

  ~scoped_temp_db_path()
  {
    if (!_path.empty()) {
      std::remove(_path.c_str());
      std::remove((_path + ".wal").c_str());
    }
  }

  scoped_temp_db_path(const scoped_temp_db_path&)            = delete;
  scoped_temp_db_path& operator=(const scoped_temp_db_path&) = delete;

  const std::string& path() const { return _path; }

 private:
  std::string _path;
};

/// RAII Sirius setting override: the setters write into the shared operator
/// params, so a test must put the previous value back.
class scoped_setting {
 public:
  scoped_setting(duckdb::Connection& con, std::string name, const std::string& value)
    : _con(con), _name(std::move(name))
  {
    auto current = con.Query("SELECT current_setting('" + _name + "')");
    REQUIRE(current);
    REQUIRE_FALSE(current->HasError());
    _original  = current->GetValue(0, 0).ToString();
    auto apply = con.Query("SET " + _name + " = " + value);
    REQUIRE(apply);
    REQUIRE_FALSE(apply->HasError());
  }
  ~scoped_setting() { _con.Query("SET " + _name + " = " + _original); }
  scoped_setting(const scoped_setting&)            = delete;
  scoped_setting& operator=(const scoped_setting&) = delete;

 private:
  duckdb::Connection& _con;
  std::string _name;
  std::string _original;
};

/// Knobs the two plan-shape suites disagree on; the defaults are the
/// eager-agg-pushdown suite's (plain optimizer output, bindings left unresolved).
struct plan_generation_options {
  /// Disable the shape-sensitive optimizers. The plan-tree-shape suite needs
  /// this so the deliminator keeps the DELIM_JOINs it asserts on.
  bool disable_shape_sensitive_optimizers = false;
  /// Run `ResolveOperatorTypes` + `ColumnBindingResolver` before `create_plan`.
  /// The eager-agg pass matches bound column refs, which the resolver would have
  /// rewritten away, so its suite hands `create_plan` the UNRESOLVED plan —
  /// exactly what the transparent capture path provides (create_plan resolves
  /// them itself).
  bool resolve_column_bindings = false;
};

/// Parse + bind + optimize @p query, then hand the logical plan to @p consume
/// while the transaction is still open (catalog lookups below `create_plan`
/// need it). Throws on any failure, after rolling back and restoring the
/// optimizer settings, so a planner regression fails the test instead of
/// silently skipping it.
template <typename Fn>
auto with_optimized_logical_plan(duckdb::Connection& con,
                                 const std::string& query,
                                 const plan_generation_options& options,
                                 const Fn& consume)
{
  auto& context = *con.context;

  auto original_disabled = duckdb::DBConfig::GetConfig(context).options.disabled_optimizers;
  if (options.disable_shape_sensitive_optimizers) {
    auto& disabled = duckdb::DBConfig::GetConfig(context).options.disabled_optimizers;
    disabled.insert(duckdb::OptimizerType::IN_CLAUSE);
    disabled.insert(duckdb::OptimizerType::COMPRESSED_MATERIALIZATION);
    disabled.insert(duckdb::OptimizerType::STATISTICS_PROPAGATION);
  }

  con.Query("BEGIN TRANSACTION");

  decltype(consume(std::declval<duckdb::unique_ptr<duckdb::LogicalOperator>>())) result;
  try {
    duckdb::Parser parser(context.GetParserOptions());
    parser.ParseQuery(query);
    REQUIRE(parser.statements.size() == 1);

    duckdb::Planner planner(context);
    planner.CreatePlan(std::move(parser.statements[0]));
    REQUIRE(planner.plan);

    auto plan = std::move(planner.plan);
    if (context.config.enable_optimizer) {
      duckdb::Optimizer optimizer(*planner.binder, context);
      plan = optimizer.Optimize(std::move(plan));
    }

    if (options.resolve_column_bindings) {
      plan->ResolveOperatorTypes();
      duckdb::ColumnBindingResolver resolver;
      duckdb::ColumnBindingResolver::Verify(*plan);
      resolver.VisitOperator(*plan);
    }

    result = consume(std::move(plan));
  } catch (...) {
    con.Query("ROLLBACK");
    duckdb::DBConfig::GetConfig(context).options.disabled_optimizers = original_disabled;
    throw;
  }

  con.Query("COMMIT");
  duckdb::DBConfig::GetConfig(context).options.disabled_optimizers = original_disabled;
  return result;
}

/// The optimizer output itself — the exact tree the eager-agg pass matches
/// against, so a test can assert on the shape the pass depends on.
inline duckdb::unique_ptr<duckdb::LogicalOperator> generate_optimized_logical_plan(
  duckdb::Connection& con, const std::string& query, const plan_generation_options& options = {})
{
  return with_optimized_logical_plan(
    con, query, options, [](duckdb::unique_ptr<duckdb::LogicalOperator> plan) { return plan; });
}

/// Generate a Sirius physical plan from a SQL query string.
inline duckdb::unique_ptr<sirius::op::sirius_physical_operator> generate_sirius_plan(
  duckdb::Connection& con, const std::string& query, const plan_generation_options& options = {})
{
  return with_optimized_logical_plan(
    con, query, options, [&](duckdb::unique_ptr<duckdb::LogicalOperator> plan) {
      sirius::planner::sirius_physical_plan_generator gen(*con.context);
      return gen.create_plan(std::move(plan));
    });
}

//===----------------------------------------------------------------------===//
// Logical tree helpers
//===----------------------------------------------------------------------===//

template <typename Fn>
void for_each_logical_operator(duckdb::LogicalOperator* root, const Fn& fn)
{
  if (!root) { return; }
  fn(root);
  for (auto& child : root->children) {
    for_each_logical_operator(child.get(), fn);
  }
}

inline duckdb::LogicalOperator* find_first_logical(duckdb::LogicalOperator* root,
                                                   duckdb::LogicalOperatorType type)
{
  duckdb::LogicalOperator* found = nullptr;
  for_each_logical_operator(root, [&](duckdb::LogicalOperator* op) {
    if (found == nullptr && op->type == type) { found = op; }
  });
  return found;
}

//===----------------------------------------------------------------------===//
// Physical tree helpers
//===----------------------------------------------------------------------===//

/// Visit every operator in the tree, including DELIM JOIN internal `join`/`distinct_root`
/// subtrees (owned outside `children[]`).
template <typename Fn>
void for_each_operator(sirius::op::sirius_physical_operator* root, const Fn& fn)
{
  if (!root) { return; }
  fn(root);
  for (auto& child : root->children) {
    for_each_operator(child.get(), fn);
  }
  if (root->type == sirius::op::SiriusPhysicalOperatorType::LEFT_DELIM_JOIN ||
      root->type == sirius::op::SiriusPhysicalOperatorType::RIGHT_DELIM_JOIN) {
    auto& delim = root->Cast<sirius::op::sirius_physical_delim_join>();
    for_each_operator(delim.join.get(), fn);
    for_each_operator(delim.distinct_root.get(), fn);
  }
}

inline std::vector<sirius::op::sirius_physical_operator*> collect(
  sirius::op::sirius_physical_operator* root, sirius::op::SiriusPhysicalOperatorType type)
{
  std::vector<sirius::op::sirius_physical_operator*> out;
  for_each_operator(root, [&](sirius::op::sirius_physical_operator* op) {
    if (op->type == type) { out.push_back(op); }
  });
  return out;
}

inline std::size_t count_ops(sirius::op::sirius_physical_operator* root,
                             sirius::op::SiriusPhysicalOperatorType type)
{
  std::size_t count = 0;
  for_each_operator(root, [&](sirius::op::sirius_physical_operator* op) {
    if (op->type == type) { count++; }
  });
  return count;
}

inline sirius::op::sirius_physical_operator* find_first(sirius::op::sirius_physical_operator* root,
                                                        sirius::op::SiriusPhysicalOperatorType type)
{
  sirius::op::sirius_physical_operator* found = nullptr;
  for_each_operator(root, [&](sirius::op::sirius_physical_operator* op) {
    if (found == nullptr && op->type == type) { found = op; }
  });
  return found;
}

inline bool contains(sirius::op::sirius_physical_operator* root,
                     const sirius::op::sirius_physical_operator* target)
{
  bool found = false;
  for_each_operator(root, [&](sirius::op::sirius_physical_operator* op) {
    if (op == target) { found = true; }
  });
  return found;
}

/// Render the tree (including delim-join internals) for failure diagnostics.
inline void tree_to_string(sirius::op::sirius_physical_operator* root,
                           int depth,
                           std::ostringstream& out)
{
  if (!root) { return; }
  out << std::string(static_cast<size_t>(depth) * 2, ' ')
      << sirius::op::SiriusPhysicalOperatorToString(root->type) << "\n";
  for (auto& child : root->children) {
    tree_to_string(child.get(), depth + 1, out);
  }
  if (root->type == sirius::op::SiriusPhysicalOperatorType::LEFT_DELIM_JOIN ||
      root->type == sirius::op::SiriusPhysicalOperatorType::RIGHT_DELIM_JOIN) {
    auto& delim = root->Cast<sirius::op::sirius_physical_delim_join>();
    out << std::string(static_cast<size_t>(depth + 1) * 2, ' ') << "(join)\n";
    tree_to_string(delim.join.get(), depth + 2, out);
    out << std::string(static_cast<size_t>(depth + 1) * 2, ' ') << "(distinct_root)\n";
    tree_to_string(delim.distinct_root.get(), depth + 2, out);
  }
}

inline std::string tree_to_string(sirius::op::sirius_physical_operator* root)
{
  std::ostringstream out;
  tree_to_string(root, 0, out);
  return out.str();
}

}  // namespace sirius::test
