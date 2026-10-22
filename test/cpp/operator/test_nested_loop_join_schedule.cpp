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

#include "helper/type_conversions.hpp"
#include "op/sirius_physical_concat.hpp"
#include "op/sirius_physical_nested_loop_join.hpp"
#include "operator_test_utils.hpp"
#include "pipeline/sirius_pipeline.hpp"

#include <catch.hpp>
#include <cucascade/data/disk_data_representation.hpp>
#include <cucascade/data/disk_io_backend.hpp>
#include <duckdb/planner/expression/bound_reference_expression.hpp>
#include <duckdb/planner/operator/logical_comparison_join.hpp>

#include <array>
#include <map>
#include <set>
#include <thread>

namespace {
using namespace sirius::op;
using duckdb::JoinType;
using batch_ptr = std::shared_ptr<cucascade::data_batch>;

class producer_pipeline : public sirius::pipeline::sirius_pipeline {
 public:
  producer_pipeline() : sirius_pipeline(sirius::pipeline::pipeline_build_context{nullptr, true}) {}
  bool is_pipeline_finished() const override { return finished; }
  bool finished = false;
};

// Metadata-only scheduler tests must not initialize the default backend's CUDA pinned pool.
class no_io_backend : public cucascade::idisk_io_backend {
 public:
  void write(std::filesystem::path const&,
             void const*,
             std::size_t,
             std::size_t,
             ::cuda::stream_ref) override
  {
    throw std::logic_error("Unexpected GPU write");
  }
  void read(
    std::filesystem::path const&, void*, std::size_t, std::size_t, ::cuda::stream_ref) override
  {
    throw std::logic_error("Unexpected GPU read");
  }
  void write(std::filesystem::path const&, void const*, std::size_t, std::size_t) override
  {
    throw std::logic_error("Unexpected host write");
  }
  void read(std::filesystem::path const&, void*, std::size_t, std::size_t) override
  {
    throw std::logic_error("Unexpected host read");
  }
};

struct fixture {
  // Metadata-only disk batches exercise scheduling and slicing without CUDA or actual disk I/O.
  cucascade::memory::memory_space space{
    cucascade::memory::disk_memory_space_config{
      .disk_id = 0, .memory_capacity = 1024 * 1024, .mount_paths = "/tmp"},
    std::make_shared<no_io_backend>()};
  duckdb::LogicalComparisonJoin logical;
  std::unique_ptr<sirius_physical_nested_loop_join> join;
  std::array<std::shared_ptr<producer_pipeline>, 2> producers;
  std::array<cucascade::shared_data_repository, 2> repos;

  explicit fixture(JoinType type                     = JoinType::INNER,
                   bool cross                        = false,
                   duckdb::ExpressionType comparison = duckdb::ExpressionType::COMPARE_LESSTHAN)
    : logical(type)
  {
    logical.types = {duckdb::LogicalType::INTEGER};
    if (type == JoinType::MARK) {
      logical.types.push_back(duckdb::LogicalType::BOOLEAN);
    } else if (type != JoinType::SEMI && type != JoinType::ANTI) {
      logical.types.push_back(duckdb::LogicalType::INTEGER);
    }
    auto child = [] {
      return duckdb::make_uniq<sirius_physical_concat>(
        sirius::from_duckdb_vec(duckdb::vector<duckdb::LogicalType>{duckdb::LogicalType::INTEGER}),
        0,
        false,
        1024);
    };
    duckdb::vector<duckdb::JoinCondition> conditions;
    if (!cross) {
      conditions.emplace_back(
        duckdb::make_uniq<duckdb::BoundReferenceExpression>(duckdb::LogicalType::INTEGER, 0),
        duckdb::make_uniq<duckdb::BoundReferenceExpression>(duckdb::LogicalType::INTEGER, 0),
        comparison);
    }
    join = std::make_unique<sirius_physical_nested_loop_join>(
      logical, child(), child(), sirius::wrap_join_conditions(std::move(conditions)), type, 0);
    join->operator_id = 0;
    sirius::pipeline::sirius_pipeline_build_state state;
    for (std::size_t i = 0; i < 2; ++i) {
      producers[i] = std::make_shared<producer_pipeline>();
      state.add_pipeline_operator(*producers[i], *join->children[i]);
      auto port          = std::make_unique<sirius_physical_operator::port>();
      port->type         = join->input_barrier_for(*join->children[i]);
      port->repo         = &repos[i];
      port->src_pipeline = producers[i];
      join->add_port(i == 0 ? "default" : "build", std::move(port));
    }
  }

  batch_ptr batch(std::size_t rows = 4)
  {
    cucascade::memory::column_metadata col{};
    col.num_rows = rows;
    auto table   = std::make_unique<cucascade::memory::disk_table_allocation>(
      "", std::vector<cucascade::memory::column_metadata>{col}, rows * 4);
    return cucascade::data_batch::make(
      sirius::get_next_batch_id(),
      std::make_unique<cucascade::disk_data_representation>(std::move(table), space));
  }
  batch_ptr push(std::size_t side, std::size_t rows = 4)
  {
    auto b = batch(rows);
    repos[side].add_data_batch(b);
    return b;
  }
  void finish()
  {
    for (auto& p : producers) {
      p->finished = true;
    }
  }
};

std::array<uint64_t, 2> pair_ids(operator_data const& data)
{
  auto const& batches = dynamic_cast<pipelineable_operator_data const&>(data).get_data_batches();
  REQUIRE(batches.size() == 2);
  return {batches[0]->get_batch_id(), batches[1]->get_batch_id()};
}
}  // namespace

TEST_CASE("NLJ streams all late-arriving pairs exactly once", "[nlj_schedule]")
{
  fixture f;
  CHECK(f.join->get_port("build")->type == MemoryBarrierType::PARTIAL);
  CHECK(f.join->get_port("default")->type == MemoryBarrierType::PARTIAL);
  auto b1   = f.push(1);
  auto hint = f.join->get_next_task_hint();
  REQUIRE(hint);
  CHECK(hint->producer == f.join->children[0].get());
  auto p1 = f.push(0);
  REQUIRE(f.join->get_next_task_hint()->hint == TaskCreationHint::READY);
  auto first = f.join->get_next_task_input_data();
  REQUIRE(first);
  CHECK(pair_ids(*first) == std::array<uint64_t, 2>{p1->get_batch_id(), b1->get_batch_id()});
  CHECK_FALSE(f.join->get_next_task_input_data());
  CHECK(f.repos[0].total_size() == 1);
  CHECK(f.repos[1].total_size() == 1);

  auto p2 = f.push(0);
  auto b2 = f.push(1);
  std::set<std::array<uint64_t, 2>> pairs{pair_ids(*first)};
  while (auto task = f.join->get_next_task_input_data()) {
    CHECK(pairs.insert(pair_ids(*task)).second);
  }
  CHECK(pairs.size() == 4);
  f.producers[1]->finished = true;
  (void)f.join->get_next_task_hint();
  CHECK(f.repos[0].total_size() == 0);
  CHECK(f.repos[1].total_size() == 2);
  // A late probe still pairs with build batches retained after their producer closed.
  f.push(0);
  while (auto task = f.join->get_next_task_input_data()) {
    CHECK(pairs.insert(pair_ids(*task)).second);
  }
  CHECK(pairs.size() == 6);
  f.finish();
  CHECK_FALSE(f.join->get_next_task_hint());
  CHECK(f.join->all_ports_empty());
  // In-flight tasks retain valid ownership after repository retirement.
  CHECK(pair_ids(*first)[0] == p1->get_batch_id());
}

TEST_CASE("NLJ slices remain schedulable after producers finish", "[nlj_schedule]")
{
  fixture f(JoinType::INNER, true);
  f.join->set_cross_join_task_bytes(8);
  f.push(0, 4);
  f.push(1, 1);
  auto first = f.join->get_next_task_input_data();
  REQUIRE(first);
  auto const* slice = dynamic_cast<cross_join_slice_data const*>(first.get());
  REQUIRE(slice);
  REQUIRE(slice->num_slices == 4);
  f.finish();
  for (std::size_t i = 1; i < 4; ++i) {
    REQUIRE(f.join->get_next_task_hint()->hint == TaskCreationHint::READY);
    CHECK_FALSE(f.join->all_ports_empty());
    auto task = f.join->get_next_task_input_data();
    REQUIRE(task);
    auto const* next = dynamic_cast<cross_join_slice_data const*>(task.get());
    REQUIRE(next);
    CHECK(next->slice == i);
    CHECK(next->num_slices == 4);
    CHECK(pair_ids(*task) == pair_ids(*first));
  }
  CHECK_FALSE(f.join->get_next_task_hint());
  CHECK(f.join->all_ports_empty());
}

TEST_CASE("NLJ empty producers emit each orphan once and preserve side on rebuild",
          "[nlj_schedule]")
{
  auto type = GENERATE(JoinType::INNER,
                       JoinType::LEFT,
                       JoinType::RIGHT,
                       JoinType::OUTER,
                       JoinType::SEMI,
                       JoinType::ANTI,
                       JoinType::MARK);
  auto side = GENERATE(0, 1);
  fixture f(type);
  auto original = f.push(side);
  CHECK_FALSE(f.join->get_next_task_input_data());
  f.producers[1 - side]->finished = true;
  CHECK_FALSE(f.join->get_next_task_input_data());
  f.finish();
  REQUIRE(f.join->get_next_task_hint()->hint == TaskCreationHint::READY);
  auto task    = f.join->get_next_task_input_data();
  auto* orphan = dynamic_cast<nested_loop_join_orphan_data*>(task.get());
  REQUIRE(orphan);
  CHECK(orphan->present_is_build == (side == 1));
  auto rebuilt         = orphan->with_batches_preserving_rows({f.batch()});
  auto* rebuilt_orphan = dynamic_cast<nested_loop_join_orphan_data*>(rebuilt.get());
  REQUIRE(rebuilt_orphan);
  CHECK(rebuilt_orphan->present_is_build == (side == 1));
  CHECK(rebuilt->original_batch_ids() == std::vector<uint64_t>{original->get_batch_id()});
  CHECK_FALSE(f.join->get_next_task_input_data());
  CHECK_FALSE(f.join->get_next_task_hint());
  CHECK(f.join->all_ports_empty());
}

TEST_CASE("NLJ rejects fragmented whole sides even without a new hint", "[nlj_schedule]")
{
  auto type = GENERATE(JoinType::LEFT,
                       JoinType::RIGHT,
                       JoinType::OUTER,
                       JoinType::SEMI,
                       JoinType::ANTI,
                       JoinType::MARK);
  fixture f(type);
  f.push(0);
  f.push(1);
  REQUIRE(f.join->get_next_task_input_data());
  f.push(type == JoinType::RIGHT ? 0 : 1);
  CHECK_THROWS(f.join->get_next_task_input_data());
}

TEST_CASE("NLJ concurrent creators claim each pair once", "[nlj_schedule]")
{
  fixture f;
  for (int i = 0; i < 4; ++i) {
    f.push(0);
    f.push(1);
  }
  f.finish();
  std::array<std::vector<std::array<uint64_t, 2>>, 2> results;
  auto drain = [&](std::size_t i) {
    while (auto task = f.join->get_next_task_input_data()) {
      auto const& batches =
        dynamic_cast<pipelineable_operator_data const&>(*task).get_data_batches();
      results[i].push_back({batches[0]->get_batch_id(), batches[1]->get_batch_id()});
    }
  };
  std::thread a(drain, 0), b(drain, 1);
  a.join();
  b.join();
  std::set<std::array<uint64_t, 2>> unique;
  for (auto const& result : results) {
    for (auto const& pair : result) {
      CHECK(unique.insert(pair).second);
    }
  }
  CHECK(unique.size() == 16);
  CHECK_FALSE(f.join->get_next_task_hint());
  CHECK(f.join->all_ports_empty());
}

namespace {
using cell = std::optional<int32_t>;
using row  = std::vector<cell>;
using bag  = std::multiset<row>;

std::optional<bool> compare(cell l, cell r, duckdb::ExpressionType comparison)
{
  if (comparison == duckdb::ExpressionType::COMPARE_NOT_DISTINCT_FROM) { return l == r; }
  if (comparison == duckdb::ExpressionType::COMPARE_DISTINCT_FROM) { return l != r; }
  if (!l || !r) { return std::nullopt; }
  return comparison == duckdb::ExpressionType::COMPARE_EQUAL ? *l == *r : *l < *r;
}

bag reference_join(std::vector<cell> const& left,
                   std::vector<cell> const& right,
                   JoinType type,
                   duckdb::ExpressionType comparison)
{
  bag result;
  std::vector<bool> right_matched(right.size(), false);
  for (auto l : left) {
    bool matched = false, unknown = false;
    for (std::size_t i = 0; i < right.size(); ++i) {
      auto truth = compare(l, right[i], comparison);
      unknown |= !truth.has_value();
      if (truth != true) { continue; }
      matched          = true;
      right_matched[i] = true;
      if (type == JoinType::INNER || type == JoinType::LEFT || type == JoinType::RIGHT ||
          type == JoinType::OUTER) {
        result.insert(row{l, right[i]});
      }
    }
    if (!matched && (type == JoinType::LEFT || type == JoinType::OUTER)) {
      result.insert(row{l, std::nullopt});
    }
    if ((matched && type == JoinType::SEMI) || (!matched && type == JoinType::ANTI)) {
      result.insert(row{l});
    }
    if (type == JoinType::MARK) {
      result.insert(row{l, matched ? cell{1} : unknown ? cell{} : cell{0}});
    }
  }
  if (type == JoinType::RIGHT || type == JoinType::OUTER) {
    for (std::size_t i = 0; i < right.size(); ++i) {
      if (!right_matched[i]) { result.insert(row{std::nullopt, right[i]}); }
    }
  }
  return result;
}

std::vector<batch_ptr> gpu_batches(std::vector<cell> const& values, bool whole, bool emit_empty)
{
  using namespace sirius::test::operator_utils;
  std::vector<batch_ptr> batches;
  auto emit = [&](std::vector<cell> const& group) {
    std::vector<int32_t> data;
    std::vector<bool> valid;
    for (auto value : group) {
      data.push_back(value.value_or(0));
      valid.push_back(value.has_value());
    }
    batches.push_back(
      make_numeric_batch_with_nulls(*get_default_gpu_space(), data, valid, cudf::type_id::INT32));
  };
  if (values.empty()) {
    if (emit_empty) { emit({}); }
  } else if (whole) {
    emit(values);
  } else {
    for (auto value : values) {
      emit({value});
    }
  }
  return batches;
}

void append_result(bag& result, operator_data const& output)
{
  using namespace sirius::test::operator_utils;
  for (auto const& batch :
       dynamic_cast<pipelineable_operator_data const&>(output).get_data_batches()) {
    auto view = sirius::get_cudf_table_view(*batch);
    std::vector<row> rows(view.num_rows(), row(view.num_columns()));
    for (cudf::size_type col = 0; col < view.num_columns(); ++col) {
      auto column   = view.column(col);
      auto validity = copy_validity_to_host(column);
      std::vector<int32_t> values;
      if (column.type().id() == cudf::type_id::BOOL8) {
        auto bools = copy_column_to_host<bool>(column);
        values.assign(bools.begin(), bools.end());
      } else {
        values = copy_column_to_host<int32_t>(column);
      }
      for (cudf::size_type i = 0; i < view.num_rows(); ++i) {
        if (validity[i]) { rows[i][col] = values[i]; }
      }
    }
    result.insert(rows.begin(), rows.end());
  }
}
}  // namespace

TEST_CASE("NLJ incremental execution preserves duplicates NULLs and empty-side semantics",
          "[nlj_streaming_results][nested_loop_join]")
{
  auto type       = GENERATE(JoinType::INNER,
                       JoinType::LEFT,
                       JoinType::RIGHT,
                       JoinType::OUTER,
                       JoinType::SEMI,
                       JoinType::ANTI,
                       JoinType::MARK);
  auto comparison = GENERATE(duckdb::ExpressionType::COMPARE_EQUAL,
                             duckdb::ExpressionType::COMPARE_LESSTHAN,
                             duckdb::ExpressionType::COMPARE_NOT_DISTINCT_FROM,
                             duckdb::ExpressionType::COMPARE_DISTINCT_FROM);
  // Zero batches and a real zero-row batch are separate paths through scheduling.
  auto empty_case = GENERATE(0, 1, 2, 3, 4, 5);
  fixture f(type, false, comparison);
  std::vector<cell> left{std::nullopt, 1, 1, 3, 5}, right{std::nullopt, 2, 2, 1};
  if (empty_case == 1 || empty_case == 3 || empty_case == 4) { left.clear(); }
  if (empty_case == 2 || empty_case == 3 || empty_case == 5) { right.clear(); }
  auto probe =
    gpu_batches(left, type == JoinType::RIGHT || type == JoinType::OUTER, empty_case == 4);
  auto build =
    gpu_batches(right, type != JoinType::INNER && type != JoinType::RIGHT, empty_case == 5);
  std::vector<std::unique_ptr<operator_data>> tasks;
  auto drain = [&] {
    while (auto task = f.join->get_next_task_input_data()) {
      tasks.push_back(std::move(task));
    }
  };
  for (std::size_t i = 0; i < std::max(probe.size(), build.size()); ++i) {
    if (i < probe.size()) { f.repos[0].add_data_batch(probe[i]); }
    if (i < build.size()) { f.repos[1].add_data_batch(build[i]); }
    drain();
    if (i == 0 && !probe.empty() && !build.empty()) { REQUIRE_FALSE(tasks.empty()); }
  }
  f.finish();
  drain();
  CHECK_FALSE(f.join->get_next_task_hint());
  CHECK(f.join->all_ports_empty());
  // Execute only after repository retirement: task references must own every input they need.
  bag actual;
  for (auto const& task : tasks) {
    auto output = f.join->execute(*task, cudf::get_default_stream());
    append_result(actual, *output);
  }
  CHECK(actual == reference_join(left, right, type, comparison));
}

TEST_CASE("NLJ completes empty inputs and asymmetric batch grids", "[nlj_schedule]")
{
  auto probe_count = GENERATE(0, 1, 4);
  auto build_count = GENERATE(0, 1, 4);
  fixture f;
  for (int i = 0; i < probe_count; ++i) {
    f.push(0);
  }
  for (int i = 0; i < build_count; ++i) {
    f.push(1);
  }
  f.finish();
  int pairs = 0, orphans = 0;
  while (auto task = f.join->get_next_task_input_data()) {
    if (dynamic_cast<nested_loop_join_orphan_data*>(task.get())) {
      ++orphans;
    } else {
      ++pairs;
    }
  }
  CHECK(pairs == probe_count * build_count);
  CHECK(orphans == (probe_count == 0 || build_count == 0 ? probe_count + build_count : 0));
  CHECK_FALSE(f.join->get_next_task_hint());
  CHECK(f.join->all_ports_empty());
}

TEST_CASE("NLJ discovers a new batch while an earlier cross pair has pending slices",
          "[nlj_schedule]")
{
  fixture f(JoinType::INNER, true);
  f.join->set_cross_join_task_bytes(8);
  f.push(0, 4);
  f.push(1, 1);
  auto first = f.join->get_next_task_input_data();
  REQUIRE(first);
  f.push(1, 1);
  f.finish();
  std::map<std::array<uint64_t, 2>, std::set<std::size_t>> slices;
  auto record = [&](operator_data const& input) {
    auto const& slice = dynamic_cast<cross_join_slice_data const&>(input);
    CHECK(slice.num_slices == 4);
    CHECK(slices[pair_ids(input)].insert(slice.slice).second);
  };
  record(*first);
  while (auto task = f.join->get_next_task_input_data()) {
    record(*task);
  }
  CHECK(slices.size() == 2);
  for (auto const& [pair, indices] : slices) {
    CHECK(indices.size() == 4);
  }
  CHECK_FALSE(f.join->get_next_task_hint());
  CHECK(f.join->all_ports_empty());
}
