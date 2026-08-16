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
 * @file plan_test_harness.hpp
 * @brief SQL -> Sirius physical plan harness shared by the planner shape suites
 *        (test_plan_tree_shape.cpp, test_twin_scan_fusion.cpp): an on-disk temp database
 *        (the GPU-native seq_scan ingestible refuses non-single-file block managers), plan
 *        generation via Parser -> Planner -> Optimizer ->
 *        `sirius_physical_plan_generator::create_plan`, and tree walkers that descend into
 *        DELIM JOIN internal `join`/`distinct_root` subtrees.
 */

#include "op/sirius_physical_delim_join.hpp"
#include "op/sirius_physical_operator.hpp"
#include "planner/sirius_physical_plan_generator.hpp"
#include "planner/sirius_plan_twin_scan_fusion.hpp"

#include <catch.hpp>
#include <duckdb.hpp>
#include <duckdb/execution/column_binding_resolver.hpp>
#include <duckdb/main/config.hpp>
#include <duckdb/optimizer/optimizer.hpp>
#include <duckdb/parser/parser.hpp>
#include <duckdb/planner/planner.hpp>
#include <unistd.h>

#include <cstdio>
#include <sstream>
#include <string>
#include <vector>

namespace sirius::test {

/// RAII on-disk DuckDB path: the GPU-native seq_scan ingestible refuses non-single-file
/// block managers, so plan-shape tests need an on-disk database rather than :memory:.
class scoped_temp_db_path {
 public:
  scoped_temp_db_path()
  {
    char tmpl[] = "/tmp/sirius_plan_test_XXXXXX";
    int fd      = ::mkstemp(tmpl);
    REQUIRE(fd >= 0);
    ::close(fd);
    ::unlink(tmpl);
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

/// Generate a Sirius physical plan from a SQL query string. Throws on any failure (after
/// rolling back and restoring the optimizer settings) so a planner regression fails the
/// test instead of silently skipping it. When @p fusion_report is non-null it receives the
/// generator's `twin_scan_report()`. The defaulted optimizer mask keeps IN_CLAUSE,
/// COMPRESSED_MATERIALIZATION, and STATISTICS_PROPAGATION disabled: both suites are
/// shape-sensitive, and disabling STATISTICS_PROPAGATION lets the deliminator retain the
/// DELIM_JOINs they assert on.
inline duckdb::unique_ptr<sirius::op::sirius_physical_operator> generate_sirius_plan(
  duckdb::Connection& con,
  const std::string& query,
  sirius::planner::twin_scan_fusion_report* fusion_report                = nullptr,
  const duckdb::vector<duckdb::OptimizerType>& extra_disabled_optimizers = {
    duckdb::OptimizerType::IN_CLAUSE,
    duckdb::OptimizerType::COMPRESSED_MATERIALIZATION,
    duckdb::OptimizerType::STATISTICS_PROPAGATION})
{
  auto& context = *con.context;

  auto original_disabled = duckdb::DBConfig::GetConfig(context).options.disabled_optimizers;
  auto& disabled         = duckdb::DBConfig::GetConfig(context).options.disabled_optimizers;
  for (auto const optimizer : extra_disabled_optimizers) {
    disabled.insert(optimizer);
  }

  con.Query("BEGIN TRANSACTION");

  duckdb::unique_ptr<sirius::op::sirius_physical_operator> result;
  try {
    duckdb::Parser parser(context.GetParserOptions());
    parser.ParseQuery(query);
    REQUIRE(!parser.statements.empty());

    duckdb::Planner planner(context);
    planner.CreatePlan(std::move(parser.statements[0]));
    REQUIRE(planner.plan);

    auto plan = std::move(planner.plan);

    if (context.config.enable_optimizer) {
      duckdb::Optimizer optimizer(*planner.binder, context);
      plan = optimizer.Optimize(std::move(plan));
    }

    plan->ResolveOperatorTypes();

    duckdb::ColumnBindingResolver resolver;
    duckdb::ColumnBindingResolver::Verify(*plan);
    resolver.VisitOperator(*plan);

    sirius::planner::sirius_physical_plan_generator gen(context);
    result = gen.create_plan(std::move(plan));
    if (fusion_report != nullptr) { *fusion_report = gen.twin_scan_report(); }
  } catch (...) {
    con.Query("ROLLBACK");
    duckdb::DBConfig::GetConfig(context).options.disabled_optimizers = original_disabled;
    throw;
  }

  con.Query("COMMIT");
  duckdb::DBConfig::GetConfig(context).options.disabled_optimizers = original_disabled;
  return result;
}

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

inline sirius::op::sirius_physical_operator* find_first(sirius::op::sirius_physical_operator* root,
                                                        sirius::op::SiriusPhysicalOperatorType type)
{
  auto all = collect(root, type);
  return all.empty() ? nullptr : all.front();
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
