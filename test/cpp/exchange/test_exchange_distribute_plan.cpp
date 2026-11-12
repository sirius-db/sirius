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

// distribute_plan on plans the plan generator built from SQL in one process, each rank's copy
// rewritten on its own thread over a loopback exchange: where the exchanges go, what they route
// on, what it refuses, and a shape mismatch between ranks.

#include "exchange/exchange_service.hpp"
#include "exchange/loopback_fabric.hpp"
#include "exec/query_lifecycle_registry.hpp"
#include "op/sirius_physical_exchange_sink.hpp"
#include "op/sirius_physical_exchange_source.hpp"
#include "op/sirius_physical_hash_join.hpp"
#include "op/sirius_physical_partition.hpp"
#include "op/sirius_physical_sort_sample.hpp"
#include "planner/distributed_plan.hpp"
#include "planner/sirius_physical_plan_generator.hpp"
#include "sirius/exception.hpp"
#include "sirius_context.hpp"

#include <catch.hpp>
#include <duckdb.hpp>
#include <duckdb/execution/column_binding_resolver.hpp>
#include <duckdb/planner/logical_operator.hpp>
#include <utils/parquet_fixture_utils.hpp>
#include <utils/sirius_test_env.hpp>

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <memory>
#include <string>
#include <thread>
#include <vector>

namespace fs = std::filesystem;

namespace {

using sirius::exchange::exchange_service;
using sirius::op::exchange_spec;
using sirius::op::sirius_physical_operator;
using sirius::test::exchange_utils::frame_ring;
using sirius::test::exchange_utils::loopback_hub;
using sirius::test::exchange_utils::rank_t;
using T    = sirius::op::SiriusPhysicalOperatorType;
using plan = duckdb::unique_ptr<sirius_physical_operator>;

constexpr std::size_t KiB = std::size_t{1} << 10;

void write_config(fs::path const& yaml_path)
{
  std::ofstream f(yaml_path);
  f << "sirius:\n"
       "  topology:\n"
       "    num_gpus: 1\n"
       "  memory:\n"
       "    gpu:\n"
       "      usage_limit_fraction: 0.2\n"
       "      reservation_limit_fraction: 1.0\n"
       "    host:\n"
       "      capacity_bytes: 4000000000\n"
       "      initial_number_pools: 2\n"
       "      pool_size: 512\n"
       "      block_size: 1048576\n"
       "  executor:\n"
       "    pipeline:\n"
       "      num_threads: 2\n"
       "    task_creator:\n"
       "      num_threads: 1\n"
       "    downgrade:\n"
       "      num_threads: 1\n";
}

void pause_shared_envs()
{
  for (auto* env : {sirius::test::g_shared_env,
                    sirius::test::g_integration_env,
                    sirius::test::g_integration_env_2gpu}) {
    if (env && env->is_active()) { env->pause(); }
  }
}

class physical_planner : public sirius::planner::sirius_physical_plan_generator {
 public:
  using sirius_physical_plan_generator::sirius_physical_plan_generator;

  plan lowered(duckdb::unique_ptr<duckdb::LogicalOperator> logical)
  {
    logical->ResolveOperatorTypes();
    duckdb::ColumnBindingResolver resolver;
    resolver.VisitOperator(*logical);
    auto p = create_plan(*logical);
    insert_gpu_pipeline_operators(p);
    return p;
  }
};

/// Tables `big` and `big2` (100,000 rows each) and `small` (10 rows), with columns k and v, as
/// Parquet files planned through read_parquet, and one-process plans of SQL over them.
struct fixture {
  sirius::test::scratch_dir scratch{"distribute_plan"};
  std::unique_ptr<sirius::test::shared_test_env> env;
  std::unique_ptr<duckdb::Connection> con;

  fixture()
  {
    pause_shared_envs();
    auto const config = scratch.path() / "sirius.yaml";
    write_config(config);
    {
      sirius::test::scoped_sirius_disable disable_sirius;
      duckdb::DuckDB db(nullptr);
      duckdb::Connection gen(db);
      for (auto const& [name, rows] : std::vector<std::pair<std::string, int>>{
             {"big", 100'000}, {"big2", 100'000}, {"small", 10}}) {
        auto r = gen.Query("COPY (SELECT range AS k, range % 7 AS v FROM range(" +
                           std::to_string(rows) + ")) TO " + path(name) + " (FORMAT PARQUET)");
        REQUIRE_FALSE(r->HasError());
      }
    }
    env = std::make_unique<sirius::test::shared_test_env>(config);
    con = std::make_unique<duckdb::Connection>(env->make_connection());
    REQUIRE_FALSE(con->Query("SET gpu_execution = false")->HasError());
    for (auto const* name : {"big", "big2", "small"}) {
      REQUIRE_FALSE(con
                      ->Query(std::string{"CREATE VIEW "} + name +
                              " AS SELECT * FROM read_parquet(" + path(name) + ")")
                      ->HasError());
    }
  }

  [[nodiscard]] std::string path(std::string const& name) const
  {
    return sirius::test::sql_literal((scratch.path() / (name + ".parquet")).string());
  }

  plan lowered(std::string const& sql)
  {
    REQUIRE_FALSE(con->Query("BEGIN TRANSACTION READ ONLY")->HasError());
    physical_planner planner(*con->context);
    auto p = planner.lowered(con->ExtractPlan(sql));
    REQUIRE_FALSE(con->Query("ROLLBACK")->HasError());
    return p;
  }

  [[nodiscard]] cucascade::memory::memory_space& gpu() const
  {
    auto ctx = con->context->registered_state->Get<duckdb::SiriusContext>("sirius_state");
    return *ctx->get_memory_manager().get_memory_space(cucascade::memory::Tier::GPU, 0);
  }
};

/// Ranks in one process, each with an exchange service over the loopback fabric.
struct ranks {
  struct one {
    std::unique_ptr<frame_ring> send;
    std::unique_ptr<frame_ring> recv;
    std::unique_ptr<sirius::exchange::fabric> fab;
    std::shared_ptr<exchange_service> service;
  };
  loopback_hub hub;
  std::vector<one> all;
  sirius::exec::query_lifecycle_registry lifecycle;

  ranks(rank_t world, cucascade::memory::memory_space& gpu) : hub(world), all(world)
  {
    for (rank_t r = 0; r < world; ++r) {
      auto& o   = all[r];
      o.send    = std::make_unique<frame_ring>(4, 64 * KiB);
      o.recv    = std::make_unique<frame_ring>(8, 64 * KiB);
      o.fab     = hub.attach(r, *o.send, *o.recv);
      o.service = std::make_shared<exchange_service>(*o.fab, *o.send, *o.recv, gpu, nullptr);
    }
  }
  ~ranks()
  {
    for (auto& o : all) {
      o.service.reset();
    }
  }
};

/// Rewrites plans[r] as rank r of @p c, each on its own thread; returns each rank's error. The
/// plans hold the ranks' services: declare @p c before them, so it outlives them.
std::vector<std::string> distribute(std::vector<plan>& plans,
                                    ranks& c,
                                    fixture& f,
                                    std::uint64_t query = 7)
{
  auto const world = static_cast<rank_t>(plans.size());
  std::vector<std::string> errors(world);
  std::vector<std::thread> threads;
  for (rank_t r = 0; r < world; ++r) {
    threads.emplace_back([&, r] {
      try {
        sirius::planner::distribute_plan(plans[r], c.all[r].service, f.gpu(), query, c.lifecycle);
      } catch (std::exception const& e) {
        errors[r] = e.what();
      }
    });
  }
  for (auto& t : threads) {
    t.join();
  }
  return errors;
}

std::vector<plan> plan_on_every_rank(fixture& f, std::string const& sql, rank_t world)
{
  std::vector<plan> plans;
  for (rank_t r = 0; r < world; ++r) {
    plans.push_back(f.lowered(sql));
  }
  return plans;
}

sirius_physical_operator* find(sirius_physical_operator& node, T type)
{
  if (node.type == type) { return &node; }
  for (auto& child : node.children) {
    if (auto* found = find(*child, type)) { return found; }
  }
  return nullptr;
}

/// The exchange sink directly below @p partition, or null when its input is not exchanged.
sirius::op::sirius_physical_exchange_sink const* sink_below(sirius_physical_operator& partition)
{
  auto& input = *partition.children.at(0);
  if (input.type != T::EXCHANGE_SOURCE) { return nullptr; }
  return &input.children.at(0)->Cast<sirius::op::sirius_physical_exchange_sink>();
}

/// A hash join's probe and build PARTITIONs (each side is CONCAT → PARTITION → input).
std::pair<sirius_physical_operator*, sirius_physical_operator*> join_sides(
  sirius_physical_operator& join)
{
  return {join.children.at(0)->children.at(0).get(), join.children.at(1)->children.at(0).get()};
}

}  // namespace

TEST_CASE("distribute_plan broadcasts a small build and hashes a large one", "[exchange][planner]")
{
  fixture f;

  SECTION("a small build is broadcast and the probe stays")
  {
    ranks c(2, f.gpu());
    auto plans = plan_on_every_rank(f, "SELECT b.v FROM big b JOIN small s ON b.k = s.k", 2);
    REQUIRE(distribute(plans, c, f) == std::vector<std::string>(2));
    auto* join = find(*plans[0], T::HASH_JOIN);
    REQUIRE(join);
    auto [probe, build] = join_sides(*join);
    REQUIRE(sink_below(*build));
    CHECK(sink_below(*build)->spec().routing == exchange_spec::mode::broadcast);
    CHECK_FALSE(sink_below(*probe));
  }

  SECTION("builds as large as their probe are hashed on the join keys, on 3 ranks")
  {
    ranks c(3, f.gpu());
    auto plans = plan_on_every_rank(f, "SELECT b.v, c.v FROM big b JOIN big2 c ON b.k = c.k", 3);
    REQUIRE(distribute(plans, c, f) == std::vector<std::string>(3));
    auto* join = find(*plans[0], T::HASH_JOIN);
    REQUIRE(join);
    auto [probe, build] = join_sides(*join);
    for (auto* side : {probe, build}) {
      REQUIRE(sink_below(*side));
      auto const& spec = sink_below(*side)->spec();
      CHECK(spec.routing == exchange_spec::mode::hash);
      CHECK(spec.key_columns ==
            side->Cast<sirius::op::sirius_physical_partition>().partition_keys());
      CHECK_FALSE(spec.null_keys_to_every_rank);
    }
  }

  SECTION("a hashed MARK join sends its NULL build keys to every rank")
  {
    ranks c(3, f.gpu());
    auto plans = plan_on_every_rank(f, "SELECT k, k IN (SELECT k FROM big2) FROM big", 3);
    REQUIRE(distribute(plans, c, f) == std::vector<std::string>(3));
    auto* join = find(*plans[0], T::HASH_JOIN);
    REQUIRE(join);
    REQUIRE(join->Cast<sirius::op::sirius_physical_hash_join>().join_type ==
            duckdb::JoinType::MARK);
    auto [probe, build] = join_sides(*join);
    REQUIRE(sink_below(*probe));
    REQUIRE(sink_below(*build));
    CHECK_FALSE(sink_below(*probe)->spec().null_keys_to_every_rank);
    CHECK(sink_below(*build)->spec().null_keys_to_every_rank);
  }

  SECTION("a join that emits unmatched build rows is hashed however small its build")
  {
    ranks c(2, f.gpu());
    auto plans =
      plan_on_every_rank(f, "SELECT b.v, s.v FROM big b FULL OUTER JOIN small s ON b.k = s.k", 2);
    REQUIRE(distribute(plans, c, f) == std::vector<std::string>(2));
    auto* join = find(*plans[0], T::HASH_JOIN);
    REQUIRE(join);
    auto [probe, build] = join_sides(*join);
    for (auto* side : {probe, build}) {
      REQUIRE(sink_below(*side));
      CHECK(sink_below(*side)->spec().routing == exchange_spec::mode::hash);
    }
  }
}

TEST_CASE("distribute_plan range-exchanges an ORDER BY on the sample's order",
          "[exchange][planner]")
{
  fixture f;
  ranks c(2, f.gpu());
  auto plans = plan_on_every_rank(f, "SELECT k, v FROM big ORDER BY v DESC NULLS FIRST, k", 2);
  REQUIRE(distribute(plans, c, f) == std::vector<std::string>(2));
  // The result is gathered to rank 0 in rank order.
  REQUIRE(plans[0]->type == T::EXCHANGE_SOURCE);
  CHECK(plans[0]->Cast<sirius::op::sirius_physical_exchange_source>().ordered());
  CHECK(
    plans[0]->children.at(0)->Cast<sirius::op::sirius_physical_exchange_sink>().spec().routing ==
    exchange_spec::mode::gather);
  auto* partition = find(*plans[0], T::SORT_PARTITION);
  REQUIRE(partition);
  REQUIRE(sink_below(*partition));
  auto const& spec = sink_below(*partition)->spec();
  CHECK(spec.routing == exchange_spec::mode::range);
  CHECK(spec.key_columns == std::vector<int>{1, 0});
  CHECK(spec.column_order ==
        std::vector<cudf::order>{cudf::order::DESCENDING, cudf::order::ASCENDING});
  // cuDF orders NULL as the smallest value under BEFORE: last when descending.
  CHECK(spec.null_precedence.at(0) == cudf::null_order::AFTER);
  auto* sample = find(*plans[0], T::SORT_SAMPLE);
  REQUIRE(sample);
  CHECK(sample->runs_without_input());
}

TEST_CASE("distribute_plan refuses what it cannot place", "[exchange][planner]")
{
  fixture f;
  // A refusal throws before the shape check, so one rank is enough.
  auto const refused = [&](std::string const& sql) {
    ranks c(2, f.gpu());
    auto p = f.lowered(sql);
    try {
      sirius::planner::distribute_plan(p, c.all[0].service, f.gpu(), 7, c.lifecycle);
    } catch (sirius::not_implemented_exception const& e) {
      return std::string{e.what()};
    }
    return std::string{};
  };

  SECTION("COUNT(DISTINCT) with grouping sets")
  {
    CHECK_THAT(refused("SELECT v, count(DISTINCT k) FROM big GROUP BY GROUPING SETS ((v), ())"),
               Catch::Matchers::ContainsSubstring("COUNT(DISTINCT) with grouping sets"));
  }

  SECTION("a nested column crossing an exchange")
  {
    // An ungrouped COUNT(DISTINCT)'s partial is a LIST; a grouped STDDEV_SAMP's a STRUCT.
    auto const sql = GENERATE(as<std::string>{},
                              "SELECT count(DISTINCT k) FROM big",
                              "SELECT v, stddev_samp(k) FROM big GROUP BY v");
    CHECK_THAT(refused(sql), Catch::Matchers::ContainsSubstring("an exchange of a column"));
  }

  SECTION("an operator it has no placement for")
  {
    CHECK_THAT(refused("SELECT k FROM big UNION SELECT k FROM big2"),
               Catch::Matchers::ContainsSubstring("distributed execution does not support"));
  }
}

TEST_CASE("a rank whose rewrite fails aborts the query on every rank, and the next query plans",
          "[exchange][planner]")
{
  fixture f;
  ranks c(2, f.gpu());
  {
    // Rank 0 refuses its plan; rank 1 plans a supported one and waits for rank 0's shape.
    std::vector<plan> plans;
    plans.push_back(f.lowered("SELECT k FROM big UNION SELECT k FROM big2"));
    plans.push_back(f.lowered("SELECT k FROM big"));
    auto const start  = std::chrono::steady_clock::now();
    auto const errors = distribute(plans, c, f);
    // Not the shape all-gather's own timeout, which is minutes.
    CHECK(std::chrono::steady_clock::now() - start < std::chrono::seconds{60});
    CHECK_THAT(errors[0],
               Catch::Matchers::ContainsSubstring("distributed execution does not support"));
    CHECK_THAT(errors[1], Catch::Matchers::ContainsSubstring("failed on rank 0"));
    CHECK_THAT(errors[1],
               Catch::Matchers::ContainsSubstring("distributed execution does not support"));
  }
  auto plans = plan_on_every_rank(f, "SELECT k FROM big", 2);
  CHECK(distribute(plans, c, f, 8) == std::vector<std::string>(2));
}

TEST_CASE("distribute_plan fails on every rank when the ranks planned differently",
          "[exchange][planner]")
{
  fixture f;
  ranks c(2, f.gpu());
  std::vector<plan> plans;
  plans.push_back(f.lowered("SELECT k FROM big"));
  plans.push_back(f.lowered("SELECT k + 1 FROM big"));
  auto const errors = distribute(plans, c, f);
  for (auto const& error : errors) {
    CHECK_THAT(error, Catch::Matchers::ContainsSubstring("planned this query differently"));
  }
}
