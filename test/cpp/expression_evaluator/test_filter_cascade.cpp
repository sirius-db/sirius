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

// Tests for the cheap-conjunct filter cascade in expression_evaluator::select().
//
// The cascade splits a top-level AND into cheap fixed-width prefilter conjuncts
// and an expensive (string-carried) residual, evaluates the cheap group first,
// and runs the residual only on surviving rows when the prefilter is selective
// enough. These tests assert two things for every decision branch:
//   1. the decision taken (via last_filter_cascade_decision_for_testing), and
//   2. byte-identical results against the monolithic single-mask path
//      (cascade disabled), including under NULLs and Kleene AND semantics.
// The conjunct classifier is additionally exercised arm-by-arm through
// filter_cascade_internal.hpp.
//
// Config knobs are process-global; config_guard saves/restores them so these
// tests cannot leak thresholds into other test files.

// test
#include "ast_test_support.hpp"

#include <catch.hpp>

// sirius
#include <config.hpp>
#include <cucascade/cudf/gpu_data_representation.hpp>
#include <cucascade/memory/reservation_manager_configurator.hpp>
#include <data/sirius_converter_registry.hpp>
#include <expression/ast/node.hpp>
#include <expression/value.hpp>
#include <expression_evaluator/expression_evaluator.hpp>
#include <expression_evaluator/filter_cascade_internal.hpp>
#include <helper/logical_type.hpp>
#include <memory/sirius_memory_reservation_manager.hpp>

// test utils
#include <operator/operator_type_traits.hpp>
#include <utils/data_utils.hpp>

// cudf
#include <cudf/column/column_factories.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <cuda_runtime_api.h>

// standard library
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

using namespace cucascade;
using namespace cucascade::memory;
using namespace sirius::expr_test;

namespace {

namespace test_utils = ::sirius::test::operator_utils;

using memory_mgr = ::sirius::memory::sirius_memory_reservation_manager;
using decision   = ::sirius::expression_evaluator::filter_cascade_decision;

std::unique_ptr<memory_mgr> initialize_memory_manager()
{
  ::sirius::converter_registry::reset_for_testing();
  reservation_manager_configurator builder;
  auto constexpr gpu_capacity  = 256ull << 20;  // 256MB
  auto constexpr host_capacity = 512ull << 20;  // 512MB
  auto constexpr limit_ratio   = 0.75;
  builder.set_number_of_gpus(1)
    .set_gpu_usage_limit(gpu_capacity)
    .set_reservation_fraction_per_gpu(limit_ratio)
    .set_per_numa_region_capacity(host_capacity)
    .use_gpu_id_as_host_id()
    .set_reservation_fraction_per_numa_region(limit_ratio);
  auto configs = builder.build();
  auto manager = std::make_unique<memory_mgr>(std::move(configs));
  ::sirius::converter_registry::initialize();
  return manager;
}

memory_space* get_default_gpu_space()
{
  static auto manager = initialize_memory_manager();
  return const_cast<memory_space*>(manager->get_memory_space(Tier::GPU, 0));
}

rmm::device_async_resource_ref get_resource_ref(memory_space& space)
{
  return space.get_default_allocator();
}

/// Save/restore the cascade Config knobs around each test.
struct config_guard {
  bool enabled           = duckdb::Config::FILTER_CASCADE_CHEAP_CONJUNCTS;
  std::uint64_t min_rows = duckdb::Config::FILTER_CASCADE_MIN_ROWS;
  double max_pass_rate   = duckdb::Config::FILTER_CASCADE_MAX_PASS_RATE;
  ~config_guard()
  {
    duckdb::Config::FILTER_CASCADE_CHEAP_CONJUNCTS = enabled;
    duckdb::Config::FILTER_CASCADE_MIN_ROWS        = min_rows;
    duckdb::Config::FILTER_CASCADE_MAX_PASS_RATE   = max_pass_rate;
  }
};

/// Null out exactly the given rows of @p col (explicit rows, not stride coincidence).
void set_nulls_at(cudf::column& col,
                  std::vector<cudf::size_type> const& rows,
                  rmm::cuda_stream_view stream,
                  rmm::device_async_resource_ref mr)
{
  auto mask      = cudf::create_null_mask(col.size(), cudf::mask_state::ALL_VALID, stream, mr);
  auto* mask_ptr = static_cast<cudf::bitmask_type*>(mask.data());
  for (auto const row : rows) {
    cudf::set_null_mask(mask_ptr, row, row + 1, false, stream);
  }
  col.set_null_mask(std::move(mask), static_cast<cudf::size_type>(rows.size()));
}

/// What make_mixed_table appends after its two predicate columns.
enum class extra_column : std::uint8_t {
  none,
  string_payload,  ///< col2 STRING "payload-<i>" — expensive to gather, referenced by nothing
  int_payload,     ///< col2 INT32 — fixed-width, referenceable by a cheap conjunct
};

/// Input fixture: col0 INT32 = 0..n-1 (optionally with nulls), col1 STRING cycling
/// {"AIR", "TRUCK", "MAIL", "SHIP"} (optionally with nulls), plus an optional payload column
/// that no test predicate's residual references — the column the gather-scope guard must not
/// gather.
std::unique_ptr<cudf::table> make_mixed_table(memory_space& space,
                                              cudf::size_type n,
                                              bool with_nulls,
                                              extra_column extra = extra_column::none)
{
  auto mr     = get_resource_ref(space);
  auto stream = cudf::get_default_stream();

  std::vector<int32_t> ints(n);
  std::vector<std::string> strs(n);
  static std::vector<std::string> const kModes{"AIR", "TRUCK", "MAIL", "SHIP"};
  for (cudf::size_type i = 0; i < n; ++i) {
    ints[i] = i;
    strs[i] = kModes[i % kModes.size()];
  }

  auto int_col =
    ::sirius::test::vector_to_cudf_column<test_utils::gpu_type_traits<int32_t>>(ints, stream, mr);
  auto str_col =
    ::sirius::test::vector_to_cudf_column<test_utils::gpu_type_traits<test_utils::string_tag>>(
      strs, stream, mr);

  if (with_nulls) {
    // Null out every 7th int row and every 5th string row (offset so the sets differ).
    auto strided_rows = [n](cudf::size_type stride, cudf::size_type offset) {
      std::vector<cudf::size_type> rows;
      for (cudf::size_type i = offset; i < n; i += stride) {
        rows.push_back(i);
      }
      return rows;
    };
    set_nulls_at(*int_col, strided_rows(7, 3), stream, mr);
    set_nulls_at(*str_col, strided_rows(5, 1), stream, mr);
  }

  std::vector<std::unique_ptr<cudf::column>> cols;
  cols.push_back(std::move(int_col));
  cols.push_back(std::move(str_col));
  if (extra == extra_column::string_payload) {
    std::vector<std::string> payload(n);
    for (cudf::size_type i = 0; i < n; ++i) {
      payload[i] = "payload-" + std::to_string(i);
    }
    cols.push_back(
      ::sirius::test::vector_to_cudf_column<test_utils::gpu_type_traits<test_utils::string_tag>>(
        payload, stream, mr));
  } else if (extra == extra_column::int_payload) {
    std::vector<int32_t> payload(n);
    for (cudf::size_type i = 0; i < n; ++i) {
      payload[i] = n - i;
    }
    cols.push_back(::sirius::test::vector_to_cudf_column<test_utils::gpu_type_traits<int32_t>>(
      payload, stream, mr));
  }
  return std::make_unique<cudf::table>(std::move(cols));
}

/// AND[col0 < int_bound (cheap), col1 = 'AIR' (expensive)].
std::unique_ptr<ast_node> make_mixed_and(int32_t int_bound)
{
  std::vector<std::unique_ptr<ast_node>> conjuncts;
  conjuncts.push_back(
    make_cmp(sirius::comparison_type::lt,
             make_ref_typed(0, sirius::logical_type::make(sirius::type_id::INTEGER)),
             make_int_const(int_bound)));
  conjuncts.push_back(
    make_cmp(sirius::comparison_type::equal,
             make_ref_typed(1, sirius::logical_type::make(sirius::type_id::VARCHAR)),
             make_str_const("AIR")));
  return make_conj(sirius::ast::conjunction::kind::op_and, std::move(conjuncts));
}

struct select_run {
  std::unique_ptr<cudf::table> result;
  decision taken;
};

select_run run_select(memory_space& space,
                      ast_node const& expr,
                      cudf::table_view input,
                      ::sirius::expression_evaluator_strategy strategy,
                      std::vector<cudf::size_type> const& output_indices = {})
{
  ::sirius::expression_evaluator evaluator(
    expr, get_resource_ref(space), cudf::get_default_stream(), strategy);
  auto result =
    output_indices.empty() ? evaluator.select(input) : evaluator.select(input, output_indices);
  return {std::move(result), evaluator.last_filter_cascade_decision_for_testing()};
}

/// Assert two tables (already on GPU) are element-identical, including validity.
void expect_tables_equal(cudf::table_view lhs, cudf::table_view rhs)
{
  REQUIRE(lhs.num_columns() == rhs.num_columns());
  REQUIRE(lhs.num_rows() == rhs.num_rows());
  for (cudf::size_type c = 0; c < lhs.num_columns(); ++c) {
    auto const& l = lhs.column(c);
    auto const& r = rhs.column(c);
    REQUIRE(l.type().id() == r.type().id());
    REQUIRE(copy_valids_to_host(l) == copy_valids_to_host(r));
    if (l.type().id() == cudf::type_id::STRING) {
      REQUIRE(copy_string_column_to_host(l) == copy_string_column_to_host(r));
    } else {
      REQUIRE(copy_column_to_host<int32_t>(l) == copy_column_to_host<int32_t>(r));
    }
  }
}

/// Run @p expr twice — cascade enabled then disabled — asserting the enabled run takes
/// @p expected and both results are byte-identical. Returns the enabled run for extra checks.
select_run expect_decision_and_baseline_equality(
  memory_space& space,
  ast_node const& expr,
  cudf::table_view input,
  ::sirius::expression_evaluator_strategy strategy,
  decision expected,
  std::vector<cudf::size_type> const& output_indices = {})
{
  duckdb::Config::FILTER_CASCADE_CHEAP_CONJUNCTS = true;
  auto enabled = run_select(space, expr, input, strategy, output_indices);
  REQUIRE(enabled.taken == expected);

  duckdb::Config::FILTER_CASCADE_CHEAP_CONJUNCTS = false;
  auto baseline = run_select(space, expr, input, strategy, output_indices);
  REQUIRE(baseline.taken == decision::not_applicable);
  expect_tables_equal(baseline.result->view(), enabled.result->view());

  duckdb::Config::FILTER_CASCADE_CHEAP_CONJUNCTS = true;
  return enabled;
}

struct mat_strategy {
  static constexpr auto value = ::sirius::expression_evaluator_strategy::MATERIALIZE;
};
struct ast_interpret_strategy {
  static constexpr auto value = ::sirius::expression_evaluator_strategy::AST_INTERPRET;
};
struct ast_jit_strategy {
  static constexpr auto value = ::sirius::expression_evaluator_strategy::AST_JIT;
};

constexpr auto kInterpret = ::sirius::expression_evaluator_strategy::AST_INTERPRET;

}  // namespace

TEMPLATE_TEST_CASE("filter cascade selects the same rows as the monolithic path",
                   "[expression_evaluator][filter_cascade]",
                   mat_strategy,
                   ast_interpret_strategy,
                   ast_jit_strategy)
{
  constexpr auto strategy = TestType::value;
  auto* space             = get_default_gpu_space();
  config_guard guard;

  duckdb::Config::FILTER_CASCADE_MIN_ROWS      = 1;
  duckdb::Config::FILTER_CASCADE_MAX_PASS_RATE = 0.75;

  for (bool with_nulls : {false, true}) {
    auto input = make_mixed_table(*space, 1000, with_nulls);
    // col0 < 300 passes 30% -> below the 0.75 pass-rate cap -> cascaded.
    auto expr = make_mixed_and(300);

    auto cascaded = expect_decision_and_baseline_equality(
      *space, *expr, input->view(), strategy, decision::cascaded);
    REQUIRE(cascaded.result->num_rows() > 0);
  }
}

TEST_CASE("filter cascade honors output_indices projection",
          "[expression_evaluator][filter_cascade]")
{
  auto* space = get_default_gpu_space();
  config_guard guard;

  duckdb::Config::FILTER_CASCADE_MIN_ROWS      = 1;
  duckdb::Config::FILTER_CASCADE_MAX_PASS_RATE = 0.75;

  auto input = make_mixed_table(*space, 1000, /*with_nulls=*/true);
  auto expr  = make_mixed_and(300);
  std::vector<cudf::size_type> const project_int_only{0};

  auto cascaded = expect_decision_and_baseline_equality(
    *space, *expr, input->view(), kInterpret, decision::cascaded, project_int_only);
  REQUIRE(cascaded.result->num_columns() == 1);
}

TEMPLATE_TEST_CASE("unselective cheap prefilter combines masks without a gather",
                   "[expression_evaluator][filter_cascade]",
                   mat_strategy,
                   ast_interpret_strategy,
                   ast_jit_strategy)
{
  constexpr auto strategy = TestType::value;
  auto* space             = get_default_gpu_space();
  config_guard guard;

  duckdb::Config::FILTER_CASCADE_MIN_ROWS      = 1;
  duckdb::Config::FILTER_CASCADE_MAX_PASS_RATE = 0.75;

  auto input = make_mixed_table(*space, 1000, /*with_nulls=*/true);
  // col0 < 990 passes ~99% -> above the 0.75 cap -> combined_masks.
  auto expr = make_mixed_and(990);

  expect_decision_and_baseline_equality(
    *space, *expr, input->view(), strategy, decision::combined_masks);
}

TEMPLATE_TEST_CASE("cheap prefilter that drops every row short-circuits the residual",
                   "[expression_evaluator][filter_cascade]",
                   mat_strategy,
                   ast_interpret_strategy,
                   ast_jit_strategy)
{
  constexpr auto strategy = TestType::value;
  auto* space             = get_default_gpu_space();
  config_guard guard;

  duckdb::Config::FILTER_CASCADE_MIN_ROWS      = 1;
  duckdb::Config::FILTER_CASCADE_MAX_PASS_RATE = 0.75;

  auto input = make_mixed_table(*space, 1000, /*with_nulls=*/false);
  auto expr  = make_mixed_and(-1);  // col0 < -1 passes nothing

  auto run = expect_decision_and_baseline_equality(
    *space, *expr, input->view(), strategy, decision::short_circuited);
  REQUIRE(run.result->num_rows() == 0);
  REQUIRE(run.result->num_columns() == 2);
  REQUIRE(run.result->view().column(1).type().id() == cudf::type_id::STRING);
}

TEST_CASE("short-circuit under projection synthesizes a typed empty table",
          "[expression_evaluator][filter_cascade]")
{
  auto* space = get_default_gpu_space();
  config_guard guard;

  duckdb::Config::FILTER_CASCADE_MIN_ROWS      = 1;
  duckdb::Config::FILTER_CASCADE_MAX_PASS_RATE = 0.75;

  auto input = make_mixed_table(*space, 1000, /*with_nulls=*/false);
  auto expr  = make_mixed_and(-1);  // col0 < -1 passes nothing
  std::vector<cudf::size_type> const project_int_only{0};

  // The one place a projected empty table is synthesized (empty_like of the projection) rather
  // than gathered.
  auto run = expect_decision_and_baseline_equality(
    *space, *expr, input->view(), kInterpret, decision::short_circuited, project_int_only);
  REQUIRE(run.result->num_rows() == 0);
  REQUIRE(run.result->num_columns() == 1);
  REQUIRE(run.result->view().column(0).type().id() == cudf::type_id::INT32);
}

TEST_CASE("cascade does not engage without a mixed AND or above min rows",
          "[expression_evaluator][filter_cascade]")
{
  auto* space = get_default_gpu_space();
  config_guard guard;

  duckdb::Config::FILTER_CASCADE_CHEAP_CONJUNCTS = true;
  duckdb::Config::FILTER_CASCADE_MIN_ROWS        = 1;
  duckdb::Config::FILTER_CASCADE_MAX_PASS_RATE   = 0.75;

  auto input = make_mixed_table(*space, 1000, /*with_nulls=*/false);

  SECTION("all-cheap conjunction")
  {
    std::vector<std::unique_ptr<ast_node>> conjuncts;
    conjuncts.push_back(make_cmp(sirius::comparison_type::lt, make_ref(0), make_int_const(300)));
    conjuncts.push_back(make_cmp(sirius::comparison_type::ge, make_ref(0), make_int_const(10)));
    auto expr = make_conj(sirius::ast::conjunction::kind::op_and, std::move(conjuncts));
    auto run  = run_select(*space, *expr, input->view(), kInterpret);
    REQUIRE(run.taken == decision::not_applicable);
  }

  SECTION("OR root")
  {
    std::vector<std::unique_ptr<ast_node>> disjuncts;
    disjuncts.push_back(make_cmp(sirius::comparison_type::lt, make_ref(0), make_int_const(300)));
    disjuncts.push_back(
      make_cmp(sirius::comparison_type::equal,
               make_ref_typed(1, sirius::logical_type::make(sirius::type_id::VARCHAR)),
               make_str_const("AIR")));
    auto expr = make_conj(sirius::ast::conjunction::kind::op_or, std::move(disjuncts));
    auto run  = run_select(*space, *expr, input->view(), kInterpret);
    REQUIRE(run.taken == decision::not_applicable);
  }

  SECTION("below min rows")
  {
    duckdb::Config::FILTER_CASCADE_MIN_ROWS = 100000;
    auto expr                               = make_mixed_and(300);
    auto run                                = run_select(*space, *expr, input->view(), kInterpret);
    REQUIRE(run.taken == decision::not_applicable);
  }

  SECTION("disabled by knob")
  {
    duckdb::Config::FILTER_CASCADE_CHEAP_CONJUNCTS = false;
    auto expr                                      = make_mixed_and(300);
    auto run = run_select(*space, *expr, input->view(), kInterpret);
    REQUIRE(run.taken == decision::not_applicable);
  }
}

TEST_CASE("empty input refuses the cascade without throwing",
          "[expression_evaluator][filter_cascade]")
{
  auto* space = get_default_gpu_space();
  config_guard guard;

  duckdb::Config::FILTER_CASCADE_CHEAP_CONJUNCTS = true;
  // A zero-row batch must refuse on the num_rows > 0 guard even when the row floor is 0.
  duckdb::Config::FILTER_CASCADE_MIN_ROWS      = 0;
  duckdb::Config::FILTER_CASCADE_MAX_PASS_RATE = 0.75;

  auto input = make_mixed_table(*space, 0, /*with_nulls=*/false);
  auto expr  = make_mixed_and(300);

  auto run = run_select(*space, *expr, input->view(), kInterpret);
  REQUIRE(run.taken == decision::not_applicable);
  REQUIRE(run.result->num_rows() == 0);
  REQUIRE(run.result->num_columns() == 2);
}

TEST_CASE("min rows boundary: equal row count engages, one fewer refuses",
          "[expression_evaluator][filter_cascade]")
{
  auto* space = get_default_gpu_space();
  config_guard guard;

  duckdb::Config::FILTER_CASCADE_MIN_ROWS      = 1000;
  duckdb::Config::FILTER_CASCADE_MAX_PASS_RATE = 0.75;

  auto expr = make_mixed_and(300);

  SECTION("num_rows == min_rows engages")
  {
    auto input = make_mixed_table(*space, 1000, /*with_nulls=*/false);
    expect_decision_and_baseline_equality(
      *space, *expr, input->view(), kInterpret, decision::cascaded);
  }

  SECTION("num_rows == min_rows - 1 refuses")
  {
    auto input = make_mixed_table(*space, 999, /*with_nulls=*/false);
    duckdb::Config::FILTER_CASCADE_CHEAP_CONJUNCTS = true;
    auto run = run_select(*space, *expr, input->view(), kInterpret);
    REQUIRE(run.taken == decision::not_applicable);
  }
}

TEST_CASE("pass rate exactly at the knob gathers; one row above combines masks",
          "[expression_evaluator][filter_cascade]")
{
  auto* space = get_default_gpu_space();
  config_guard guard;

  duckdb::Config::FILTER_CASCADE_MIN_ROWS      = 1;
  duckdb::Config::FILTER_CASCADE_MAX_PASS_RATE = 0.75;

  // No nulls, n = 1000: col0 < bound passes exactly `bound` rows.
  auto input = make_mixed_table(*space, 1000, /*with_nulls=*/false);

  SECTION("passed = 750 -> rate exactly 0.750 -> cascaded (comparison is inclusive)")
  {
    auto expr = make_mixed_and(750);
    expect_decision_and_baseline_equality(
      *space, *expr, input->view(), kInterpret, decision::cascaded);
  }

  SECTION("passed = 751 -> rate 0.751 -> combined_masks")
  {
    auto expr = make_mixed_and(751);
    expect_decision_and_baseline_equality(
      *space, *expr, input->view(), kInterpret, decision::combined_masks);
  }
}

TEST_CASE("all-pass prefilter and the pass-rate knob's degenerate ends",
          "[expression_evaluator][filter_cascade]")
{
  auto* space = get_default_gpu_space();
  config_guard guard;

  duckdb::Config::FILTER_CASCADE_MIN_ROWS = 1;

  auto input = make_mixed_table(*space, 1000, /*with_nulls=*/false);

  SECTION("100% cheap pass rate under the default cap combines masks")
  {
    duckdb::Config::FILTER_CASCADE_MAX_PASS_RATE = 0.75;
    auto expr                                    = make_mixed_and(2000);  // passes every row
    expect_decision_and_baseline_equality(
      *space, *expr, input->view(), kInterpret, decision::combined_masks);
  }

  SECTION("max_pass_rate = 1.0 always gathers, even at a 100% pass rate")
  {
    duckdb::Config::FILTER_CASCADE_MAX_PASS_RATE = 1.0;
    auto expr                                    = make_mixed_and(2000);
    expect_decision_and_baseline_equality(
      *space, *expr, input->view(), kInterpret, decision::cascaded);
  }

  SECTION("max_pass_rate = 0.0 never gathers")
  {
    duckdb::Config::FILTER_CASCADE_MAX_PASS_RATE = 0.0;
    auto expr                                    = make_mixed_and(300);  // selective: 30%
    expect_decision_and_baseline_equality(
      *space, *expr, input->view(), kInterpret, decision::combined_masks);
  }
}

TEST_CASE("Kleene partition invariance: the 3VL matrix selects identical rows on every route",
          "[expression_evaluator][filter_cascade]")
{
  auto* space = get_default_gpu_space();
  config_guard guard;

  duckdb::Config::FILTER_CASCADE_MIN_ROWS = 1;

  // Nine explicit rows covering all of {TRUE, FALSE, NULL}^2 for the (cheap, expensive) pair:
  // cheap is col0 = 1 and expensive is col1 = 'AIR', so row i encodes the pair
  // (T,T),(T,F),(T,N),(F,T),(F,F),(F,N),(N,T),(N,F),(N,N). Only the (T,T) row may survive; a
  // NULL in the cheap group must drop the row before the residual runs (TRUE AND NULL != TRUE)
  // and a NULL residual must drop a cheap-TRUE row in the residual stage.
  auto mr     = get_resource_ref(*space);
  auto stream = cudf::get_default_stream();

  std::vector<int32_t> const ints{1, 1, 1, 0, 0, 0, 0, 0, 0};
  std::vector<std::string> const strs{"AIR", "XYZ", "", "AIR", "XYZ", "", "AIR", "XYZ", ""};
  auto int_col =
    ::sirius::test::vector_to_cudf_column<test_utils::gpu_type_traits<int32_t>>(ints, stream, mr);
  auto str_col =
    ::sirius::test::vector_to_cudf_column<test_utils::gpu_type_traits<test_utils::string_tag>>(
      strs, stream, mr);
  set_nulls_at(*int_col, {6, 7, 8}, stream, mr);
  set_nulls_at(*str_col, {2, 5, 8}, stream, mr);

  std::vector<std::unique_ptr<cudf::column>> cols;
  cols.push_back(std::move(int_col));
  cols.push_back(std::move(str_col));
  cudf::table input(std::move(cols));

  std::vector<std::unique_ptr<ast_node>> conjuncts;
  conjuncts.push_back(
    make_cmp(sirius::comparison_type::equal,
             make_ref_typed(0, sirius::logical_type::make(sirius::type_id::INTEGER)),
             make_int_const(1)));
  conjuncts.push_back(
    make_cmp(sirius::comparison_type::equal,
             make_ref_typed(1, sirius::logical_type::make(sirius::type_id::VARCHAR)),
             make_str_const("AIR")));
  auto expr = make_conj(sirius::ast::conjunction::kind::op_and, std::move(conjuncts));

  SECTION("cascaded branch")
  {
    // The cheap conjunct passes 3 of 9 rows (rate 0.333) -> gathered.
    duckdb::Config::FILTER_CASCADE_MAX_PASS_RATE = 0.75;
    auto run                                     = expect_decision_and_baseline_equality(
      *space, *expr, input.view(), kInterpret, decision::cascaded);
    REQUIRE(run.result->num_rows() == 1);
  }

  SECTION("combined_masks branch")
  {
    duckdb::Config::FILTER_CASCADE_MAX_PASS_RATE = 0.0;
    auto run                                     = expect_decision_and_baseline_equality(
      *space, *expr, input.view(), kInterpret, decision::combined_masks);
    REQUIRE(run.result->num_rows() == 1);
  }
}

TEST_CASE("cascade groups multiple cheap and expensive conjuncts",
          "[expression_evaluator][filter_cascade]")
{
  auto* space = get_default_gpu_space();
  config_guard guard;

  duckdb::Config::FILTER_CASCADE_MIN_ROWS      = 1;
  duckdb::Config::FILTER_CASCADE_MAX_PASS_RATE = 0.75;

  auto input = make_mixed_table(*space, 1000, /*with_nulls=*/true);

  // AND[col0 >= 100, col0 < 400, col1 IN ('AIR','MAIL'), col1 <> 'SHIP'] —
  // two cheap conjuncts and two expensive ones, mirroring q19's part-scan shape
  // (numeric range + two string groups).
  auto build = [] {
    auto varchar = sirius::logical_type::make(sirius::type_id::VARCHAR);
    std::vector<std::unique_ptr<ast_node>> conjuncts;
    conjuncts.push_back(make_cmp(sirius::comparison_type::ge, make_ref(0), make_int_const(100)));
    conjuncts.push_back(make_cmp(sirius::comparison_type::lt, make_ref(0), make_int_const(400)));
    std::vector<std::unique_ptr<ast_node>> in_values;
    in_values.push_back(make_str_const("AIR"));
    in_values.push_back(make_str_const("MAIL"));
    conjuncts.push_back(make_in(make_ref_typed(1, varchar), std::move(in_values), false));
    conjuncts.push_back(make_cmp(
      sirius::comparison_type::not_equal, make_ref_typed(1, varchar), make_str_const("SHIP")));
    return make_conj(sirius::ast::conjunction::kind::op_and, std::move(conjuncts));
  };

  auto expr = build();
  expect_decision_and_baseline_equality(
    *space, *expr, input->view(), kInterpret, decision::cascaded);
}

TEST_CASE("out-of-scope survivor gather refuses the cascade",
          "[expression_evaluator][filter_cascade]")
{
  auto* space = get_default_gpu_space();
  config_guard guard;

  duckdb::Config::FILTER_CASCADE_CHEAP_CONJUNCTS = true;
  duckdb::Config::FILTER_CASCADE_MIN_ROWS        = 1;
  duckdb::Config::FILTER_CASCADE_MAX_PASS_RATE   = 0.75;

  // col2 ("payload-<i>" strings) is neither projected nor referenced by the residual: the
  // cascaded gather would materialize it pointlessly, and a cascade that may never gather can
  // only lose to the monolithic kernel, so the guard refuses outright (not combined_masks) even
  // though the prefilter is selective (30% << 0.75).
  auto input = make_mixed_table(*space, 1000, /*with_nulls=*/true, extra_column::string_payload);
  auto expr  = make_mixed_and(300);

  SECTION("projection excludes an unreferenced column -> not_applicable")
  {
    std::vector<cudf::size_type> const project_first_two{0, 1};
    auto refused = expect_decision_and_baseline_equality(
      *space, *expr, input->view(), kInterpret, decision::not_applicable, project_first_two);
    REQUIRE(refused.result->num_columns() == 2);
  }

  SECTION("projection covers every column -> gather stays in scope")
  {
    std::vector<cudf::size_type> const project_all{0, 1, 2};
    expect_decision_and_baseline_equality(
      *space, *expr, input->view(), kInterpret, decision::cascaded, project_all);
  }

  SECTION("all-columns select overload is always in scope")
  {
    expect_decision_and_baseline_equality(
      *space, *expr, input->view(), kInterpret, decision::cascaded);
  }

  SECTION("a column referenced only by the cheap group does not extend the scope")
  {
    // col2 is INT32 here and referenced by a cheap conjunct, but it is dead after the prefilter
    // (neither projected nor residual-referenced), so gathering it would be waste: the needed
    // set is output_indices union referenced(residual) — deliberately excluding cheap
    // references — and the guard must refuse.
    auto cheap_ref_input =
      make_mixed_table(*space, 1000, /*with_nulls=*/true, extra_column::int_payload);
    std::vector<std::unique_ptr<ast_node>> conjuncts;
    conjuncts.push_back(
      make_cmp(sirius::comparison_type::lt,
               make_ref_typed(0, sirius::logical_type::make(sirius::type_id::INTEGER)),
               make_int_const(300)));
    conjuncts.push_back(
      make_cmp(sirius::comparison_type::ge,
               make_ref_typed(2, sirius::logical_type::make(sirius::type_id::INTEGER)),
               make_int_const(0)));
    conjuncts.push_back(
      make_cmp(sirius::comparison_type::equal,
               make_ref_typed(1, sirius::logical_type::make(sirius::type_id::VARCHAR)),
               make_str_const("AIR")));
    auto cheap_ref_expr = make_conj(sirius::ast::conjunction::kind::op_and, std::move(conjuncts));

    std::vector<cudf::size_type> const project_first_two{0, 1};
    expect_decision_and_baseline_equality(*space,
                                          *cheap_ref_expr,
                                          cheap_ref_input->view(),
                                          kInterpret,
                                          decision::not_applicable,
                                          project_first_two);
  }
}

TEST_CASE("classifier arms: cheap means an elementwise fixed-width-carried subtree",
          "[expression_evaluator][filter_cascade]")
{
  auto* space = get_default_gpu_space();
  auto mr     = get_resource_ref(*space);
  auto stream = cudf::get_default_stream();

  // Column carriers are all that matters to the classifier: col0 INT32, col1 STRING,
  // col2 BOOL8 (the carrier a decode-time predicate substitution leaves behind).
  std::vector<std::unique_ptr<cudf::column>> cols;
  cols.push_back(cudf::make_numeric_column(
    cudf::data_type{cudf::type_id::INT32}, 4, cudf::mask_state::UNALLOCATED, stream, mr));
  cols.push_back(
    ::sirius::test::vector_to_cudf_column<test_utils::gpu_type_traits<test_utils::string_tag>>(
      {"a", "b", "c", "d"}, stream, mr));
  cols.push_back(cudf::make_numeric_column(
    cudf::data_type{cudf::type_id::BOOL8}, 4, cudf::mask_state::UNALLOCATED, stream, mr));
  cudf::table table(std::move(cols));
  auto const input = table.view();

  auto integer = sirius::logical_type::make(sirius::type_id::INTEGER);
  auto varchar = sirius::logical_type::make(sirius::type_id::VARCHAR);

  auto is_cheap = [&](std::unique_ptr<ast_node> const& n) {
    return sirius::detail::is_cheap_prefilter_conjunct(*n, input);
  };

  SECTION("references classify by runtime carrier and bounds")
  {
    CHECK(is_cheap(make_ref(0)));    // INT32 carrier
    CHECK(!is_cheap(make_ref(1)));   // STRING carrier
    CHECK(is_cheap(make_ref(2)));    // BOOL8 carrier — the tier-2 composition guarantee
    CHECK(!is_cheap(make_ref(99)));  // out-of-bounds column index
  }

  SECTION("constants classify by payload type")
  {
    CHECK(is_cheap(make_int_const(5)));
    CHECK(!is_cheap(make_str_const("x")));
  }

  SECTION("elementwise interior nodes are cheap over cheap operands")
  {
    CHECK(is_cheap(make_cmp(sirius::comparison_type::lt, make_ref(0), make_int_const(5))));
    CHECK(!is_cheap(
      make_cmp(sirius::comparison_type::equal, make_ref_typed(1, varchar), make_str_const("A"))));

    CHECK(is_cheap(make_between(make_ref(0), make_int_const(1), make_int_const(5), true, true)));

    std::vector<std::unique_ptr<ast_node>> int_values;
    int_values.push_back(make_int_const(1));
    int_values.push_back(make_int_const(2));
    CHECK(is_cheap(make_in(make_ref(0), std::move(int_values), false)));

    std::vector<std::unique_ptr<ast_node>> str_values;
    str_values.push_back(make_str_const("A"));
    CHECK(!is_cheap(make_in(make_ref_typed(1, varchar), std::move(str_values), false)));
  }

  SECTION("nested conjunctions are cheap iff every child is")
  {
    auto cheap_or = [&] {
      std::vector<std::unique_ptr<ast_node>> children;
      children.push_back(make_cmp(sirius::comparison_type::lt, make_ref(0), make_int_const(5)));
      children.push_back(make_cmp(sirius::comparison_type::ge, make_ref(0), make_int_const(7)));
      return make_conj(sirius::ast::conjunction::kind::op_or, std::move(children));
    };
    CHECK(is_cheap(cheap_or()));

    std::vector<std::unique_ptr<ast_node>> and_children;
    and_children.push_back(make_cmp(sirius::comparison_type::lt, make_ref(0), make_int_const(5)));
    and_children.push_back(
      make_between(make_ref(0), make_int_const(1), make_int_const(3), true, true));
    CHECK(is_cheap(make_conj(sirius::ast::conjunction::kind::op_and, std::move(and_children))));

    std::vector<std::unique_ptr<ast_node>> mixed_children;
    mixed_children.push_back(make_cmp(sirius::comparison_type::lt, make_ref(0), make_int_const(5)));
    mixed_children.push_back(
      make_cmp(sirius::comparison_type::equal, make_ref_typed(1, varchar), make_str_const("A")));
    CHECK(!is_cheap(make_conj(sirius::ast::conjunction::kind::op_or, std::move(mixed_children))));
  }

  SECTION("unary operators: NOT / IS NULL / IS NOT NULL only, over cheap children")
  {
    CHECK(
      is_cheap(make_unary(sirius::ast::unary_op::kind::op_not,
                          make_cmp(sirius::comparison_type::lt, make_ref(0), make_int_const(5)))));
    CHECK(is_cheap(make_unary(sirius::ast::unary_op::kind::op_is_null, make_ref(0))));
    CHECK(is_cheap(make_unary(sirius::ast::unary_op::kind::op_is_not_null, make_ref(0))));
    CHECK(!is_cheap(make_unary(sirius::ast::unary_op::kind::op_is_null, make_ref(1))));
    CHECK(!is_cheap(make_unary(
      sirius::ast::unary_op::kind::op_not,
      make_cmp(sirius::comparison_type::equal, make_ref_typed(1, varchar), make_str_const("A")))));
    // Any other unary kind is expensive regardless of its child.
    CHECK(
      !is_cheap(make_unary(sirius::ast::unary_op::kind::op_try,
                           make_cmp(sirius::comparison_type::lt, make_ref(0), make_int_const(5)))));
  }

  SECTION("AST breakers are unconditionally expensive")
  {
    // A fixed-width cast is elementwise-cheap in principle; classifying it expensive is the
    // documented conservatism (widen only with measured evidence).
    CHECK(!is_cheap(make_cast(make_ref_typed(0, integer),
                              sirius::logical_type::make(sirius::type_id::BIGINT),
                              /*try_cast=*/false)));
  }
}
