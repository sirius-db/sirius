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

// Per-query scan state (sirius_scan_manager::query_scan_manager_state).
//
// The scan manager used to keep one query's worth of state in bare members, with reset()
// wiping all of it. Two things broke the moment two queries could overlap:
//
//   - the dispatcher was shared, so finishing query A called request_stop() on the dispatcher
//     running query B's scans — B's sequencer died and its split_connector closed mid-scan,
//     which the consumer reads as end-of-stream (silent truncation, not an error);
//   - the coalescer was shared and its slot map is keyed by scan_op->get_operator_id(), and
//     operator ids restart at 0 for every query — so two queries' first scan operators
//     collided on one slot and each could be fed the other's splits.
//
// These gates pin both, plus the lifecycle arithmetic (register / reset / reset_all and the
// no-op paths) that sirius_context's cleanup and failure backstop depend on.

#include "catch.hpp"
#include "memory/topology_index.hpp"
#include "op/scan/parquet_gpu_ingestible.hpp"
#include "op/scan/sirius_gpu_scan_operator.hpp"
#include "op/scan/sirius_gpu_scan_operator_data.hpp"
#include "pipeline/repository_wiring.hpp"
#include "pipeline/sirius_pipeline.hpp"
#include "planner/query.hpp"
#include "scan/test_utils.hpp"
#include "scan_manager/sirius_scan_manager.hpp"
#include "scan_manager/split_connector.hpp"
#include "utils/telemetry_utils.hpp"

#include <cucascade/memory/topology_discovery.hpp>

#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace {

namespace scan = sirius::op::scan;
using sirius::scan_manager::sirius_scan_manager;

std::filesystem::path project_root()
{
#ifdef SIRIUS_PROJECT_ROOT
  return std::filesystem::path{SIRIUS_PROJECT_ROOT};
#else
  return std::filesystem::current_path();
#endif
}

/// Local-file scan config: no S3 backend, no prefetch cache, one worker. The pool the manager
/// builds is num_threads + k_max_concurrent_queries, which is what the two-query gates lean on.
sirius::scan_manager::scan_manager_config make_local_config()
{
  sirius::scan_manager::scan_manager_config cfg;
  cfg.thread_pool.num_threads = 2;
  cfg.uring_n_reactors        = 1;
  cfg.enable_prefetch_cache   = false;
  return cfg;
}

std::unique_ptr<scan::parquet_ingestible_table_info> make_table_info(std::string const& file)
{
  auto info                 = std::make_unique<scan::parquet_ingestible_table_info>();
  info->resolved_file_paths = {
    (project_root() / "test/cpp/integration/data/parquet" / file).string()};
  info->names = {"c0"};
  info->returned_types.push_back(sirius::logical_type::make(sirius::type_id::INTEGER));
  info->column_ids.push_back(duckdb::ColumnIndex(0));
  info->scan_output_arity = 1;
  return info;
}

/// One query, one pipeline, one GPU scan operator over @p file. Operator ids are assigned per
/// query (restarting at 0), which is exactly the collision the coalescer slot map used to hit.
struct query_context {
  duckdb::shared_ptr<sirius::pipeline::sirius_pipeline> pipeline;
  std::unique_ptr<scan::sirius_gpu_scan_operator> scan_op;
  std::shared_ptr<const sirius::telemetry::telemetry_context> tctx;
  duckdb::shared_ptr<sirius::planner::query> query;
};

query_context make_query(sirius::query_id_t query_id, std::string const& file)
{
  query_context ctx;
  const sirius::pipeline::pipeline_build_context build_ctx{nullptr, true};
  ctx.pipeline = duckdb::make_shared_ptr<sirius::pipeline::sirius_pipeline>(build_ctx);
  ctx.pipeline->set_pipeline_id(1);
  ctx.pipeline->set_query_id(query_id);

  auto ingestible = scan::make_ingestible(make_table_info(file));
  ctx.scan_op     = std::make_unique<scan::sirius_gpu_scan_operator>(
    duckdb::vector<sirius::logical_type>{sirius::logical_type::make(sirius::type_id::INTEGER)},
    0,
    std::move(ingestible));

  sirius::pipeline::sirius_pipeline_build_state build_state;
  build_state.set_pipeline_source(*ctx.pipeline, *ctx.scan_op);
  // Also register it as a pipeline operator: query::build_indices only calls set_pipeline() on
  // operators in get_operators(), and the coalescer's register_pipeline dereferences
  // scan_op->get_pipeline() to read the pipeline id. assign_operator_ids walks the same list.
  build_state.add_pipeline_operator(*ctx.pipeline, *ctx.scan_op);

  duckdb::vector<duckdb::shared_ptr<sirius::pipeline::sirius_pipeline>> pipelines{ctx.pipeline};
  sirius::pipeline::assign_operator_ids(pipelines);

  ctx.tctx = sirius::test::make_test_telemetry_context();
  sirius::telemetry::query_telemetry_info tinfo{
    ctx.tctx->engine_id(), ctx.tctx->worker_id(), query_id};
  ctx.query = duckdb::make_shared_ptr<sirius::planner::query>(
    pipelines, ctx.tctx->context(), query_id, tinfo);
  return ctx;
}

cucascade::memory::system_topology_info single_gpu_topology()
{
  cucascade::memory::system_topology_info topology;
  topology.num_gpus = 1;
  cucascade::memory::gpu_topology_info gpu;
  gpu.id        = 0;
  gpu.numa_node = 0;
  topology.gpus.push_back(std::move(gpu));
  return topology;
}

struct fixture {
  std::unique_ptr<sirius::memory::sirius_memory_reservation_manager> memory =
    initialize_memory_manager(1);
  std::shared_ptr<const sirius::memory::topology_index> topology =
    std::make_shared<sirius::memory::topology_index>(single_gpu_topology(), std::vector<int>{0});
};

}  // namespace

TEST_CASE("two queries register independently and reset drops only one",
          "[scan_manager][query_state]")
{
  fixture f;
  sirius_scan_manager manager{make_local_config(), *f.memory, f.topology};

  auto a = make_query(sirius::make_query_id(1), "nation.parquet");
  auto b = make_query(sirius::make_query_id(2), "supplier.parquet");

  manager.prepare_for_query(*a.query, /*enable_pinned_zone_map_pruning=*/true, {});
  REQUIRE(manager.num_active_queries() == 1);
  manager.prepare_for_query(*b.query, /*enable_pinned_zone_map_pruning=*/true, {});
  REQUIRE(manager.num_active_queries() == 2);

  // The gate: tearing down A must not touch B. Before per-query dispatchers this called
  // request_stop() on the one dispatcher driving both, killing B's sequencer.
  manager.reset(a.query->query_id());
  REQUIRE(manager.num_active_queries() == 1);

  // B is still live and still serving: its connector yields its splits and closes normally
  // (a closed-by-A connector would have ended the stream with no splits at all).
  std::size_t b_splits = 0;
  while (auto split = b.scan_op->get_split_connector().get_next_split()) {
    ++b_splits;
  }
  REQUIRE(b_splits > 0);

  manager.reset(b.query->query_id());
  REQUIRE(manager.num_active_queries() == 0);
}

TEST_CASE("concurrent queries do not collide on operator id", "[scan_manager][query_state]")
{
  fixture f;
  sirius_scan_manager manager{make_local_config(), *f.memory, f.topology};

  // Both scan operators are id 0 — ids restart per query. A shared coalescer keys its slot map
  // by that id, so one query's splits would land in the other's slot.
  auto a = make_query(sirius::make_query_id(1), "nation.parquet");
  auto b = make_query(sirius::make_query_id(2), "supplier.parquet");
  REQUIRE(a.scan_op->get_operator_id() == b.scan_op->get_operator_id());

  manager.prepare_for_query(*a.query, true, {});
  manager.prepare_for_query(*b.query, true, {});

  // Each operator receives splits for ITS OWN file, not the other's. The coalescer emits
  // parquet_split_info (row-group slices, possibly spanning files), not the per-file
  // parquet_file_scan_info the pre-coalescer metadata pass produces.
  auto drain_paths = [](scan::sirius_gpu_scan_operator& op) {
    std::vector<std::string> paths;
    while (auto split = op.get_split_connector().get_next_split()) {
      auto* input = dynamic_cast<scan::scan_operator_input*>(split->get());
      if (input == nullptr || !input->has_scan_metadata()) { continue; }
      auto const* split_info =
        dynamic_cast<scan::parquet_split_info const*>(&input->get_scan_info());
      if (split_info == nullptr) { continue; }
      for (auto const& slice : split_info->rg_slices) {
        paths.push_back(slice.file_path);
      }
    }
    return paths;
  };

  auto a_paths = drain_paths(*a.scan_op);
  auto b_paths = drain_paths(*b.scan_op);

  REQUIRE_FALSE(a_paths.empty());
  REQUIRE_FALSE(b_paths.empty());
  for (auto const& p : a_paths) {
    REQUIRE(p.find("nation.parquet") != std::string::npos);
  }
  for (auto const& p : b_paths) {
    REQUIRE(p.find("supplier.parquet") != std::string::npos);
  }

  manager.reset_all();
  REQUIRE(manager.num_active_queries() == 0);
}

TEST_CASE("reset is a no-op for unknown and already-reset queries", "[scan_manager][query_state]")
{
  fixture f;
  sirius_scan_manager manager{make_local_config(), *f.memory, f.topology};

  // Never prepared: sirius_context's failure backstop calls reset(query_id) unconditionally,
  // so an unknown id has to be harmless.
  REQUIRE_NOTHROW(manager.reset(sirius::make_query_id(99)));
  REQUIRE(manager.num_active_queries() == 0);

  auto a = make_query(sirius::make_query_id(1), "nation.parquet");
  manager.prepare_for_query(*a.query, true, {});
  REQUIRE(manager.num_active_queries() == 1);

  manager.reset(a.query->query_id());
  // Double reset: run_mandatory_cleanup and drop_query_runtime_state_best_effort can both
  // fire for the same query on the failure path.
  REQUIRE_NOTHROW(manager.reset(a.query->query_id()));
  REQUIRE(manager.num_active_queries() == 0);
}

TEST_CASE("reset_all drops every query and stop() still returns", "[scan_manager][query_state]")
{
  fixture f;
  sirius_scan_manager manager{make_local_config(), *f.memory, f.topology};

  auto a = make_query(sirius::make_query_id(1), "nation.parquet");
  auto b = make_query(sirius::make_query_id(2), "supplier.parquet");
  manager.prepare_for_query(*a.query, true, {});
  manager.prepare_for_query(*b.query, true, {});
  REQUIRE(manager.num_active_queries() == 2);

  manager.reset_all();
  REQUIRE(manager.num_active_queries() == 0);

  // stop() drains again then stops the pool; a sequencer left parked on a dequeue that will
  // never be satisfied would hang here.
  REQUIRE_NOTHROW(manager.stop());
}

TEST_CASE("a query with no GPU scan operators registers nothing", "[scan_manager][query_state]")
{
  fixture f;
  sirius_scan_manager manager{make_local_config(), *f.memory, f.topology};

  auto tctx           = sirius::test::make_test_telemetry_context();
  auto const query_id = sirius::make_query_id(7);
  sirius::telemetry::query_telemetry_info tinfo{tctx->engine_id(), tctx->worker_id(), query_id};
  sirius::planner::query empty{
    duckdb::vector<duckdb::shared_ptr<sirius::pipeline::sirius_pipeline>>{},
    tctx->context(),
    query_id,
    tinfo};

  manager.prepare_for_query(empty, true, {});
  // Nothing to tear down, so nothing is registered and the matching reset is a no-op.
  REQUIRE(manager.num_active_queries() == 0);
  REQUIRE_NOTHROW(manager.reset(query_id));
}
