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

// These tests exercise the contract between the parquet metadata / gpu scan
// operators and sirius_pipeline's task counters:
//
//   * task_creator's manager loop (src/creator/task_creator.cpp — else branch)
//     runs this for every gpu source:
//
//         while (!node->all_ports_empty()) {
//             pipeline->mark_task_created();
//             auto input = node->get_next_task_input_data();
//             if (!input) { pipeline->mark_task_completed(); break; }
//             ... dispatch to executor, which later calls mark_task_completed()
//         }
//
//   * update_pipeline_status() closes the pipeline only when
//         first_node->is_source_pipeline_finished() && all_ports_empty()
//         && tasks_created == tasks_completed
//
// The tests simulate that loop inline (no executor), invoking
// mark_task_completed() synchronously after each dispatch to mimic a finished
// task. That lets us assert the exact counter values and the finished-state
// transitions for both the metadata pipeline (pipeline 1) and the downstream
// gpu scan pipeline (pipeline 2) — including the multi-file case where
// max_file_processed = 1 forces one metadata task per file.

// test
#include <catch.hpp>
#include <scan/test_utils.hpp>
#include <utils/utils.hpp>

// sirius
#include <helper/type_conversions.hpp>
#include <op/scan/parquet_scan_operator_data.hpp>
#include <op/scan/sirius_gpu_parquet_scan_operator.hpp>
#include <op/scan/sirius_parquet_metadata_scan_operator.hpp>
#include <op/sirius_physical_operator.hpp>
#include <pipeline/sirius_pipeline.hpp>
#include <sirius_engine.hpp>
#include <sirius_interface.hpp>

// cucascade
#include <cucascade/memory/memory_reservation_manager.hpp>

// cudf
#include <cudf/utilities/default_stream.hpp>

// duckdb
#include <duckdb/common/types.hpp>

// standard library
#include <cstddef>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

using namespace sirius::scan_test_utils;
using namespace cucascade::memory;

namespace {

//===----------------------------------------------------------------------===//
// Fixture
//===----------------------------------------------------------------------===//

/// Owns everything needed to construct real sirius_pipeline instances —
/// engine, memory manager, gpu memory space. One fixture per test case so
/// each run starts from a clean state.
struct pipeline_task_counting_fixture {
  pipeline_task_counting_fixture()
    : db(nullptr),
      con(db),
      sirius_iface(*con.context),
      memory_manager(initialize_memory_manager()),
      engine(*con.context, sirius_iface),
      gpu_space(get_space(*memory_manager, Tier::GPU))
  {
    REQUIRE(gpu_space != nullptr);
  }

  duckdb::DuckDB db;
  duckdb::Connection con;
  sirius::sirius_interface sirius_iface;
  std::unique_ptr<sirius::memory::sirius_memory_reservation_manager> memory_manager;
  sirius::sirius_engine engine;
  cucascade::memory::memory_space* gpu_space;
};

//===----------------------------------------------------------------------===//
// Parquet helpers (duplicated from test_metadata_gpu_scan_operators.cpp to
// keep this file self-contained; the scan tests don't expose them)
//===----------------------------------------------------------------------===//

std::filesystem::path write_parquet(duckdb::Connection& con,
                                    std::string const& table_name,
                                    std::size_t row_group_size = 0)
{
  auto path = std::filesystem::temp_directory_path() / (table_name + ".parquet");
  std::string sql;
  if (row_group_size != 0) {
    sql = "COPY " + table_name + " TO '" + path.string() +
          "' (FORMAT PARQUET, COMPRESSION zstd, ROW_GROUP_SIZE " + std::to_string(row_group_size) +
          ")";
  } else {
    sql = "COPY " + table_name + " TO '" + path.string() + "' (FORMAT PARQUET, COMPRESSION zstd)";
  }
  auto result = con.Query(sql);
  REQUIRE(result);
  REQUIRE(!result->HasError());
  return path;
}

struct parquet_file_cleanup {
  std::vector<std::filesystem::path> paths;
  ~parquet_file_cleanup()
  {
    for (auto const& p : paths) {
      std::filesystem::remove(p);
    }
  }
};

struct schema_info {
  duckdb::vector<duckdb::ColumnIndex> column_ids;
  duckdb::vector<std::string> names;
  duckdb::vector<duckdb::LogicalType> types;
};

schema_info synthetic_schema()
{
  return {{duckdb::ColumnIndex(0),
           duckdb::ColumnIndex(1),
           duckdb::ColumnIndex(2),
           duckdb::ColumnIndex(3)},
          {"id", "value", "price", "name"},
          {duckdb::LogicalType::INTEGER,
           duckdb::LogicalType::BIGINT,
           duckdb::LogicalType::DOUBLE,
           duckdb::LogicalType::VARCHAR}};
}

//===----------------------------------------------------------------------===//
// Pipeline wiring
//===----------------------------------------------------------------------===//

struct pipeline_pair {
  duckdb::shared_ptr<sirius::pipeline::sirius_pipeline> metadata_pipeline;
  duckdb::shared_ptr<sirius::pipeline::sirius_pipeline> gpu_pipeline;
};

/// Build the two-pipeline skeleton the same way sirius_pipeline_converter would
/// for a parquet scan, stripped down to the bits update_pipeline_status()
/// actually inspects:
///
///   metadata_pipeline:
///     - source = sink = metadata_op  (single-op self-pipeline)
///     - no ports on metadata_op → is_source_pipeline_finished() trivially true
///
///   gpu_pipeline:
///     - source = sink = gpu_op
///     - gpu_op has a "handoff" port whose src_pipeline points at
///       metadata_pipeline — identical to converter.cpp's PARTIAL scheduling
///       port — so gpu_op.is_source_pipeline_finished() returns true only
///       after metadata_pipeline has finished.
///
///   metadata_pipeline.parents = { gpu_pipeline }  — required so that
///     update_pipeline_status() on metadata_pipeline notifies the gpu pipeline
///     once it's done.
pipeline_pair build_pipelines(sirius::sirius_engine& engine,
                              sirius::op::scan::sirius_parquet_metadata_scan_operator& metadata_op,
                              sirius::op::scan::sirius_gpu_parquet_scan_operator& gpu_op)
{
  using namespace sirius::pipeline;

  auto metadata_pipeline = duckdb::make_shared_ptr<sirius_pipeline>(engine);
  auto gpu_pipeline      = duckdb::make_shared_ptr<sirius_pipeline>(engine);

  sirius_pipeline_build_state build_state;
  build_state.set_pipeline_source(*metadata_pipeline, metadata_op);
  build_state.set_pipeline_sink(*metadata_pipeline, &metadata_op, 0);

  build_state.set_pipeline_source(*gpu_pipeline, gpu_op);
  build_state.set_pipeline_sink(*gpu_pipeline, &gpu_op, 0);

  gpu_op.add_port("handoff",
                  std::make_unique<sirius::op::sirius_physical_operator::port>(
                    sirius::op::MemoryBarrierType::PARTIAL,
                    /*repo=*/nullptr,
                    metadata_pipeline,
                    gpu_pipeline));

  metadata_op.set_pipeline(metadata_pipeline);
  gpu_op.set_pipeline(gpu_pipeline);

  // add_dependency() simultaneously records the dependency on gpu_pipeline's
  // side and appends gpu_pipeline to metadata_pipeline->parents, which is what
  // notify_downstream_pipelines() walks when metadata_pipeline closes.
  gpu_pipeline->add_dependency(metadata_pipeline);
  return {metadata_pipeline, gpu_pipeline};
}

//===----------------------------------------------------------------------===//
// Simulated task-creator loop
//===----------------------------------------------------------------------===//

/// Mirrors the else-branch of task_creator::manager_loop() but runs each
/// "task" synchronously in-line so we can observe counters deterministically.
///
/// Returns the number of successful (non-null) task inputs dispatched. The
/// caller can compare that against get_tasks_created() to confirm the counter
/// matches what the real task_creator would have produced.
template <typename TaskFn>
std::size_t simulate_task_creator_loop(sirius::pipeline::sirius_pipeline& pipe,
                                       sirius::op::sirius_physical_operator& node,
                                       TaskFn&& run_task)
{
  std::size_t dispatched = 0;
  while (!node.all_ports_empty()) {
    pipe.mark_task_created();
    auto input = node.get_next_task_input_data();
    if (!input) {
      pipe.mark_task_completed();
      break;
    }
    run_task(*input);
    pipe.mark_task_completed();
    ++dispatched;
  }
  return dispatched;
}

}  // namespace

//===----------------------------------------------------------------------===//
// Tests
//===----------------------------------------------------------------------===//

TEST_CASE("pipeline task counting - single file, default max_file_processed (one metadata task)",
          "[pipeline_task_counting][shared_context]")
{
  pipeline_task_counting_fixture fixture;

  constexpr std::size_t NUM_ROWS  = 2000;
  constexpr std::size_t ROW_GROUP = 500;  // → 4 row groups, so ~4 partitions
  create_synthetic_table(fixture.con, "task_count_single", NUM_ROWS);
  auto path = write_parquet(fixture.con, "task_count_single", ROW_GROUP);
  parquet_file_cleanup cleanup{{path}};

  auto schema                    = synthetic_schema();
  std::vector<std::string> files = {path.string()};
  duckdb::vector<duckdb::idx_t> no_projection;

  sirius::op::scan::sirius_gpu_parquet_scan_operator gpu_op(sirius::from_duckdb_vec(schema.types),
                                                            0);
  sirius::op::scan::sirius_parquet_metadata_scan_operator metadata_op(
    &gpu_op,
    sirius::from_duckdb_vec(schema.types),
    sirius::from_duckdb_vec(schema.types),
    0,
    files,
    schema.column_ids,
    no_projection,
    schema.names);

  auto [metadata_pipeline, gpu_pipeline] = build_pipelines(fixture.engine, metadata_op, gpu_op);

  // ----------- pipeline 1: metadata scan -----------
  auto metadata_tasks = simulate_task_creator_loop(
    *metadata_pipeline, metadata_op, [&](sirius::op::operator_data& input) {
      auto output = metadata_op.execute(input, cudf::get_default_stream());
      REQUIRE(output);
      metadata_op.sink(*output, cudf::get_default_stream());
    });

  // Default max_file_processed = 8 and only 1 file → exactly one metadata task.
  REQUIRE(metadata_tasks == 1);
  REQUIRE(metadata_pipeline->get_tasks_created() == 1);
  REQUIRE(metadata_pipeline->get_tasks_completed() == 1);
  REQUIRE(metadata_pipeline->is_pipeline_finished());

  // ----------- pipeline 2: gpu parquet scan -----------
  std::size_t gpu_tasks =
    simulate_task_creator_loop(*gpu_pipeline, gpu_op, [&](sirius::op::operator_data& input) {
      input.prepare_for_processing(fixture.gpu_space, cudf::get_default_stream());
      auto output = gpu_op.execute(input, cudf::get_default_stream());
      REQUIRE(output);
    });

  REQUIRE(gpu_tasks >= 1);  // at least one partition for a non-empty table
  REQUIRE(gpu_pipeline->get_tasks_created() == gpu_tasks);
  REQUIRE(gpu_pipeline->get_tasks_completed() == gpu_tasks);
  REQUIRE(gpu_pipeline->is_pipeline_finished());
}

TEST_CASE("pipeline task counting - multi-file metadata scan with max_file_processed = 1",
          "[pipeline_task_counting][multi_file][shared_context]")
{
  pipeline_task_counting_fixture fixture;

  // Four small tables, each written to its own parquet file. With
  // max_file_processed = 1 the metadata scan has to emit one task per file,
  // so task_creator's loop should call get_next_task_input_data() 4 times
  // successfully (plus one trailing null once all_ports_empty()).
  constexpr std::size_t NUM_FILES = 4;
  constexpr std::size_t ROWS_PER  = 300;
  constexpr std::size_t ROW_GROUP = 100;
  std::vector<std::filesystem::path> paths;
  std::vector<std::string> files;
  for (std::size_t i = 0; i < NUM_FILES; ++i) {
    auto name = "task_count_multi_" + std::to_string(i);
    create_synthetic_table(fixture.con, name, ROWS_PER);
    auto path = write_parquet(fixture.con, name, ROW_GROUP);
    paths.push_back(path);
    files.push_back(path.string());
  }
  parquet_file_cleanup cleanup{std::move(paths)};

  auto schema = synthetic_schema();
  duckdb::vector<duckdb::idx_t> no_projection;

  sirius::op::scan::sirius_gpu_parquet_scan_operator gpu_op(sirius::from_duckdb_vec(schema.types),
                                                            0);
  sirius::op::scan::sirius_parquet_metadata_scan_operator metadata_op(
    &gpu_op,
    sirius::from_duckdb_vec(schema.types),
    sirius::from_duckdb_vec(schema.types),
    /*estimated_cardinality=*/0,
    files,
    schema.column_ids,
    no_projection,
    schema.names,
    /*table_filter_set=*/nullptr,
    /*partition_indices=*/{},
    sirius::config::DEFAULT_SCAN_TASK_BATCH_SIZE,
    /*max_file_processed=*/1);

  REQUIRE(metadata_op.get_max_file_processed() == 1);
  REQUIRE(metadata_op.get_total_files() == NUM_FILES);

  auto [metadata_pipeline, gpu_pipeline] = build_pipelines(fixture.engine, metadata_op, gpu_op);

  // ----------- pipeline 1: per-file metadata tasks -----------
  // Each successful iteration of task_creator's loop advances the file
  // counter by max_file_processed (= 1 here). After the Nth iteration
  // all_ports_empty() flips true and the loop exits, leaving
  // tasks_created == tasks_completed == NUM_FILES.
  std::size_t observed_max_created = 0;
  auto metadata_tasks              = simulate_task_creator_loop(
    *metadata_pipeline, metadata_op, [&](sirius::op::operator_data& input) {
      auto output = metadata_op.execute(input, cudf::get_default_stream());
      REQUIRE(output);
      metadata_op.sink(*output, cudf::get_default_stream());
      observed_max_created = std::max(observed_max_created, metadata_pipeline->get_tasks_created());
    });

  REQUIRE(metadata_tasks == NUM_FILES);
  REQUIRE(observed_max_created == NUM_FILES);
  REQUIRE(metadata_pipeline->get_tasks_created() == NUM_FILES);
  REQUIRE(metadata_pipeline->get_tasks_completed() == NUM_FILES);
  REQUIRE(metadata_pipeline->is_pipeline_finished());

  // The metadata pipeline is finished; the gpu op detects this via the
  // handoff port's src_pipeline->is_pipeline_finished().
  REQUIRE(gpu_op.is_source_pipeline_finished());

  // ----------- pipeline 2: gpu scan over all accumulated partitions -----------
  std::size_t gpu_tasks =
    simulate_task_creator_loop(*gpu_pipeline, gpu_op, [&](sirius::op::operator_data& input) {
      input.prepare_for_processing(fixture.gpu_space, cudf::get_default_stream());
      auto output = gpu_op.execute(input, cudf::get_default_stream());
      REQUIRE(output);
    });

  REQUIRE(gpu_tasks >= NUM_FILES);  // at least one partition per file
  REQUIRE(gpu_pipeline->get_tasks_created() == gpu_tasks);
  REQUIRE(gpu_pipeline->get_tasks_completed() == gpu_tasks);
  REQUIRE(gpu_pipeline->is_pipeline_finished());
}
