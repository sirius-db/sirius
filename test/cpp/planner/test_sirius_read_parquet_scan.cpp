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

// Regression test: the plan generator must plan `sirius_read_parquet` — the internal
// rewrite target for read_parquet('s3://…') inside gpu_execution.
// Before the fix, wrap_table_scan_source threw "Unsupported scan function:
// sirius_read_parquet", so S3 SQL failed during planning (and S3 has no CPU fallback;
// the CPU-side table function only throws). The scan is URI-agnostic — the resolved
// URI travels in parameters[0] — so a local parquet file exercises the exact same plan
// path without an S3 harness. A passing query also proves the GPU path served it: the
// DuckDB-side execute callback for this function unconditionally throws.

#include <cudf/cudf_utils.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <catch.hpp>
#include <duckdb.hpp>
#include <duckdb/common/multi_file/multi_file_reader.hpp>
#include <duckdb/execution/column_binding_resolver.hpp>
#include <duckdb/planner/logical_operator.hpp>
#include <op/dynamic_filter/sirius_dynamic_filter.hpp>
#include <op/input_free_operator_data.hpp>
#include <op/scan/parquet_gpu_ingestible.hpp>
#include <op/scan/sirius_gpu_scan_operator.hpp>
#include <op/sirius_physical_gpu_values.hpp>
#include <op/sirius_physical_table_scan.hpp>
#include <planner/sirius_physical_plan_generator.hpp>
#include <sirius_context.hpp>
#include <utils/local_ioctx.hpp>
#include <utils/parquet_fixture_utils.hpp>
#include <utils/sirius_test_env.hpp>
#include <utils/table_filter_test_utils.hpp>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <optional>
#include <string>

namespace fs = std::filesystem;

namespace {

constexpr std::int64_t kRows = 100'000;

// Throwaway, Sirius-disabled DuckDB writes the parquet so the extension callback does
// not build a SiriusContext on it.
void generate_parquet(fs::path const& path)
{
  sirius::test::scoped_sirius_disable disable_sirius;
  duckdb::DuckDB gen_db(nullptr);
  duckdb::Connection gen(gen_db);
  auto r = gen.Query("COPY (SELECT range AS k, range * 2 AS v FROM range(" + std::to_string(kRows) +
                     ")) TO " + sirius::test::sql_literal(path.string()) + " (FORMAT PARQUET);");
  REQUIRE(r);
  REQUIRE_FALSE(r->HasError());
}

void write_config(fs::path const& yaml_path)
{
  std::ofstream f(yaml_path);
  f << "sirius:\n"
       "  topology:\n"
       "    num_gpus: 1\n"
       "  memory:\n"
       "    gpu:\n"
       "      usage_limit_fraction: 0.4\n"
       "      reservation_limit_fraction: 1.0\n"
       "    host:\n"
       "      capacity_bytes: 32000000000\n"
       "      initial_number_pools: 10\n"
       "      pool_size: 512\n"
       "      block_size: 1048576\n"
       "  executor:\n"
       "    pipeline:\n"
       "      num_threads: 4\n"
       "    task_creator:\n"
       "      num_threads: 2\n"
       "    downgrade:\n"
       "      num_threads: 1\n"
       "      monitor_period: 10ms\n"
       "  operator_params:\n"
       "    scan_task_batch_size: 100000000\n"
       "    max_sort_partition_bytes: 0\n"
       "    hash_partition_bytes: 100000000\n"
       "    concat_batch_bytes: 100000000\n"
       "    max_build_hash_table_bytes: 90000000\n";
}

void pause_shared_envs()
{
  for (auto* env : {sirius::test::g_shared_env,
                    sirius::test::g_integration_env,
                    sirius::test::g_integration_env_2gpu}) {
    if (env && env->is_active()) { env->pause(); }
  }
}

// Stop at the same boundary as create_plan(unique_ptr), before lineage and scan lowering,
// so the test can supply channels whose producers have already been registered.
class scan_layout_planner : public sirius::planner::sirius_physical_plan_generator {
 public:
  using sirius_physical_plan_generator::sirius_physical_plan_generator;

  duckdb::unique_ptr<sirius::op::sirius_physical_operator> before_lowering(
    duckdb::unique_ptr<duckdb::LogicalOperator> logical)
  {
    logical->ResolveOperatorTypes();
    duckdb::ColumnBindingResolver resolver;
    resolver.VisitOperator(*logical);
    return create_plan(*logical);
  }
};

sirius::op::sirius_physical_operator* find_scan(sirius::op::sirius_physical_operator& root,
                                                sirius::op::SiriusPhysicalOperatorType type)
{
  if (root.type == type) { return &root; }
  for (auto& child : root.children) {
    if (auto* found = find_scan(*child, type)) { return found; }
  }
  return nullptr;
}

}  // namespace

TEST_CASE("parquet all-pruned lowering owns a typed empty source without a scan",
          "[planner][parquet][source_pruning][tree_pipeline]")
{
  using op_type             = sirius::op::SiriusPhysicalOperatorType;
  auto const narrow_carrier = GENERATE(false, true);
  pause_shared_envs();
  sirius::test::scratch_dir scratch{"parquet_empty_source"};
  auto const path = scratch.path() / "data.parquet";
  generate_parquet(path);
  auto const config = scratch.path() / "scan.yaml";
  write_config(config);
  sirius::test::shared_test_env local_env(config);
  auto con = local_env.make_connection();
  REQUIRE_FALSE(con.Query("SET gpu_execution=false")->HasError());
  REQUIRE_FALSE(con.Query("BEGIN TRANSACTION READ ONLY")->HasError());
  scan_layout_planner planner(*con.context);
  auto plan = planner.before_lowering(
    con.ExtractPlan("SELECT k, filename, file_index, file_row_number, k FROM read_parquet(" +
                    sirius::test::sql_literal(path.string()) + ")"));
  auto* node = find_scan(*plan, op_type::TABLE_SCAN);
  REQUIRE(node);
  auto& scan          = node->Cast<sirius::op::sirius_physical_table_scan>();
  auto const expected = scan.types;
  std::vector<cudf::data_type> physical;
  if (narrow_carrier) {
    for (auto const& type : expected) {
      physical.push_back(sirius::require_cudf_type(type));
    }
    REQUIRE(physical.front().id() == cudf::type_id::INT64);
    physical.front() = cudf::data_type{cudf::type_id::INT8};
    scan.set_physical_types(physical);
  }
  // Inject after DuckDB optimization to prove Sirius performs source pruning.
  scan.table_filters = duckdb::make_uniq<duckdb::TableFilterSet>();
  bool injected      = false;
  for (size_t i = 0; i < scan.column_ids.size(); ++i) {
    if (scan.column_ids[i].GetPrimaryIndex() ==
        duckdb::MultiFileReader::COLUMN_IDENTIFIER_FILE_INDEX) {
      scan.table_filters->PushFilter(
        duckdb::ProjectionIndex(i),
        sirius::test::constant_filter(duckdb::ExpressionType::COMPARE_EQUAL,
                                      duckdb::Value::UBIGINT(99)));
      injected = true;
    }
  }
  REQUIRE(injected);
  planner.insert_gpu_pipeline_operators(plan);
  CHECK_FALSE(find_scan(*plan, op_type::GPU_SCAN));
  CHECK_FALSE(find_scan(*plan, op_type::TABLE_SCAN));
  auto* empty = find_scan(*plan, op_type::GPU_VALUES);
  REQUIRE(empty);
  CHECK(empty->types == expected);
  CHECK(empty->estimated_cardinality == 0);
  if (narrow_carrier) { CHECK(empty->get_physical_types() == physical); }
  // The original bind/ingestible is gone; this source owns its schema and emits
  // exactly one GPU task descriptor, with no scan or metadata task descriptor.
  REQUIRE(empty->get_next_task_hint());
  auto input = empty->get_next_task_input_data();
  REQUIRE(dynamic_cast<sirius::op::input_free_operator_data*>(input.get()));
  auto state        = con.context->registered_state->Get<duckdb::SiriusContext>("sirius_state");
  auto const* space = state->get_memory_manager().get_memory_space(cucascade::memory::Tier::GPU, 0);
  REQUIRE(space);
  input->prepare_for_processing(space, cudf::get_default_stream());
  auto output = empty->execute(*input, cudf::get_default_stream());
  auto const& batches =
    dynamic_cast<sirius::op::pipelineable_operator_data&>(*output).get_read_only_batches();
  REQUIRE(batches.size() == 1);
  auto view =
    batches.front().get_data()->cast<cucascade::gpu_table_representation>().get_table_view();
  CHECK(view.num_rows() == 0);
  REQUIRE(view.num_columns() == expected.size());
  for (size_t i = 0; i < expected.size(); ++i) {
    CHECK(view.column(i).type() ==
          (narrow_carrier ? physical[i] : sirius::require_cudf_type(expected[i])));
  }
  CHECK_FALSE(empty->get_next_task_input_data());
  CHECK(empty->all_ports_empty());
  CHECK_FALSE(empty->get_next_task_hint());
  REQUIRE_FALSE(con.Query("ROLLBACK")->HasError());
}

// NB: no [integration]/[shared_context] tag — this TEST_CASE builds its own SiriusContext
// and manages (pauses) the shared envs itself, mirroring the isolated-context pin tests.
TEST_CASE("tree pipeline build plans sirius_read_parquet scans",
          "[planner][sirius_read_parquet][tree_pipeline]")
{
  pause_shared_envs();

  sirius::test::scratch_dir scratch{"srp_tree"};
  auto const& tmp = scratch.path();

  auto parquet_path = tmp / "kv.parquet";
  generate_parquet(parquet_path);
  auto collision_path = tmp / "physical_names.parquet";
  {
    sirius::test::scoped_sirius_disable disable;
    duckdb::DuckDB db(nullptr);
    duckdb::Connection gen(db);
    auto result = gen.Query(
      "COPY (SELECT range AS k, CASE WHEN range = 0 THEN NULL ELSE 'physical' END AS filename, "
      "77::BIGINT AS file_index, 88::INTEGER AS file_row_number FROM range(2)) TO " +
      sirius::test::sql_literal(collision_path.string()) + " (FORMAT PARQUET)");
    REQUIRE(result);
    REQUIRE_FALSE(result->HasError());
  }

  auto yaml_path = tmp / "srp_tree.yaml";
  write_config(yaml_path);
  REQUIRE(fs::exists(yaml_path));

  {
    sirius::test::shared_test_env local_env(yaml_path);
    auto con = local_env.make_connection();

    // A planning failure must surface as an error, not a CPU replay (which would throw
    // the function's "internal rewrite target" error anyway).
    auto fb = con.Query("SET enable_duckdb_fallback = false;");
    REQUIRE(fb);
    REQUIRE_FALSE(fb->HasError());

    auto res = con.Query("SELECT max(k), count(*) FROM sirius_read_parquet(" +
                         sirius::test::sql_literal(parquet_path.string()) + ");");
    REQUIRE(res);
    if (res->HasError()) { UNSCOPED_INFO("sirius_read_parquet query error: " << res->GetError()); }
    REQUIRE_FALSE(res->HasError());  // pre-fix: "Unsupported scan function: sirius_read_parquet"
    REQUIRE(res->Collection().GetValue(0, 0).GetValue<int64_t>() == kRows - 1);
    REQUIRE(res->Collection().GetValue(1, 0).GetValue<int64_t>() == kRows);

    auto const scan =
      "sirius_read_parquet(" + sirius::test::sql_literal(parquet_path.string()) + ")";
    auto virtuals = con.Query("SELECT filename, file_index, file_row_number, k FROM " + scan +
                              " WHERE k IN (1, 8193, 99999) ORDER BY k");
    REQUIRE(virtuals);
    INFO((virtuals->HasError() ? virtuals->GetError() : ""));
    REQUIRE_FALSE(virtuals->HasError());
    REQUIRE(virtuals->RowCount() == 3);
    CHECK(virtuals->GetTypes()[0] == duckdb::LogicalType::VARCHAR);
    CHECK(virtuals->GetTypes()[1] == duckdb::LogicalType::UBIGINT);
    CHECK(virtuals->GetTypes()[2] == duckdb::LogicalType::BIGINT);
    for (duckdb::idx_t row = 0; row < virtuals->RowCount(); ++row) {
      CHECK(virtuals->Collection().GetValue(0, row).ToString() == parquet_path.string());
      CHECK(virtuals->Collection().GetValue(1, row).GetValue<uint64_t>() == 0);
      CHECK(virtuals->Collection().GetValue(2, row).GetValue<int64_t>() ==
            virtuals->Collection().GetValue(3, row).GetValue<int64_t>());
    }

    auto filtered = con.Query("SELECT k FROM " + scan + " WHERE file_row_number = 8193");
    REQUIRE(filtered);
    REQUIRE_FALSE(filtered->HasError());
    REQUIRE(filtered->RowCount() == 1);
    CHECK(filtered->Collection().GetValue(0, 0).GetValue<int64_t>() == 8193);

    // Virtual columns are non-null, including when they are only referenced by
    // a predicate. Exercise the residual path rather than assuming the
    // planner's IS_NOT_NULL exemption also handles virtual bindings.
    for (auto const* name : {"filename", "file_index", "file_row_number"}) {
      CAPTURE(name);
      auto non_null = con.Query("SELECT count(*) FROM " + scan + " WHERE " + name + " IS NOT NULL");
      REQUIRE(non_null);
      INFO((non_null->HasError() ? non_null->GetError() : ""));
      REQUIRE_FALSE(non_null->HasError());
      CHECK(non_null->Collection().GetValue(0, 0).GetValue<int64_t>() == kRows);
      auto only_null = con.Query("SELECT k FROM " + scan + " WHERE " + name + " IS NULL");
      REQUIRE(only_null);
      REQUIRE_FALSE(only_null->HasError());
      CHECK(only_null->RowCount() == 0);
    }

    auto empty = con.Query("SELECT filename, file_index, file_row_number FROM " + scan +
                           " WHERE file_index = 1");
    REQUIRE(empty);
    REQUIRE_FALSE(empty->HasError());
    CHECK(empty->RowCount() == 0);

    auto star = con.Query("SELECT * FROM " + scan + " LIMIT 1");
    REQUIRE(star);
    REQUIRE_FALSE(star->HasError());
    CHECK(star->ColumnCount() == 2);

    // Names alone do not establish virtual identity: real columns win binding,
    // retain their physical types, and may contain nulls.
    auto const collision_scan =
      "sirius_read_parquet(" + sirius::test::sql_literal(collision_path.string()) + ")";
    auto physical = con.Query("SELECT filename, file_index, file_row_number FROM " +
                              collision_scan + " ORDER BY k");
    REQUIRE(physical);
    INFO((physical->HasError() ? physical->GetError() : ""));
    REQUIRE_FALSE(physical->HasError());
    REQUIRE(physical->RowCount() == 2);
    CHECK(physical->GetTypes()[1] == duckdb::LogicalType::BIGINT);
    CHECK(physical->GetTypes()[2] == duckdb::LogicalType::INTEGER);
    CHECK(physical->Collection().GetValue(0, 0).IsNull());
    CHECK(physical->Collection().GetValue(0, 1).ToString() == "physical");
    for (duckdb::idx_t row = 0; row < physical->RowCount(); ++row) {
      CHECK(physical->Collection().GetValue(1, row).GetValue<int64_t>() == 77);
      CHECK(physical->Collection().GetValue(2, row).GetValue<int32_t>() == 88);
    }
    auto physical_nulls = con.Query("SELECT k FROM " + collision_scan + " WHERE filename IS NULL");
    REQUIRE(physical_nulls);
    REQUIRE_FALSE(physical_nulls->HasError());
    REQUIRE(physical_nulls->RowCount() == 1);
    CHECK(physical_nulls->Collection().GetValue(0, 0).GetValue<int64_t>() == 0);
  }
}

TEST_CASE("parquet lineage and lowering agree on registered dynamic-filter channels",
          "[planner][parquet][virtual_columns][metadata_only][tree_pipeline]")
{
  namespace scan    = sirius::op::scan;
  using op_type     = sirius::op::SiriusPhysicalOperatorType;
  auto const nested = GENERATE(false, true);
  // No channel, an unused channel, and a registered producer that has not published.
  auto const channel_state = GENERATE(0, 1, 2);
  CAPTURE(nested, channel_state);
  pause_shared_envs();
  sirius::test::scratch_dir scratch{"parquet_channel_layout"};
  auto const path = scratch.file("data.parquet");
  {
    sirius::test::scoped_sirius_disable disable;
    duckdb::DuckDB db(nullptr);
    duckdb::Connection writer(db);
    auto result = writer.Query(
      "COPY (SELECT " + std::string{nested ? "[range, range + 1]" : "range"} +
      " AS payload FROM range(19)) TO " + sirius::test::sql_literal(path) + " (FORMAT PARQUET)");
    REQUIRE(result);
    REQUIRE_FALSE(result->HasError());
  }
  auto const config = scratch.path() / "scan.yaml";
  write_config(config);
  sirius::test::shared_test_env local_env(config);
  auto con = local_env.make_connection();
  REQUIRE_FALSE(con.Query("SET gpu_execution=false")->HasError());
  REQUIRE_FALSE(con.Query("BEGIN TRANSACTION READ ONLY")->HasError());
  scan_layout_planner planner(*con.context);
  auto plan = planner.before_lowering(con.ExtractPlan("SELECT file_row_number FROM read_parquet(" +
                                                      sirius::test::sql_literal(path) + ")"));
  auto* unwrapped = find_scan(*plan, op_type::TABLE_SCAN);
  REQUIRE(unwrapped);
  auto& scan_op = unwrapped->Cast<sirius::op::sirius_physical_table_scan>();
  std::optional<sirius::op::sirius_dynamic_filter_set::producer> producer;
  if (channel_state != 0) {
    scan_op.sirius_dynamic_filters = std::make_shared<sirius::op::sirius_dynamic_filter_set>();
    if (channel_state == 2) {
      producer.emplace(scan_op.sirius_dynamic_filters->register_producer({0}));
      REQUIRE_FALSE(scan_op.sirius_dynamic_filters->has_filters());
    }
  }
  if (nested && channel_state == 2) {
    // A possible dynamic filter requires real rows; a nested-only schema has no supported
    // physical carrier. Both lineage and the reader must retain this refusal boundary.
    REQUIRE_THROWS_AS(planner.insert_gpu_pipeline_operators(plan), duckdb::NotImplementedException);
  } else {
    planner.insert_gpu_pipeline_operators(plan);
    auto* lowered = find_scan(*plan, op_type::GPU_SCAN);
    REQUIRE(lowered);
    auto& gpu = lowered->Cast<scan::sirius_gpu_scan_operator>();
    CHECK(bool(gpu.consumer()) == (channel_state == 2));
    auto& reader     = dynamic_cast<scan::parquet_gpu_ingestible&>(gpu.get_ingestible());
    auto const& bind = static_cast<scan::parquet_ingestible_table_info const&>(reader.table_info());
    CHECK(bool(bind.sirius_dynamic_filters) == (channel_state == 2));
    auto ioctx = sirius::test::make_test_local_ioctx();
    auto task  = reader.next_split_provider([&](std::string_view) { return ioctx; });
    REQUIRE(task);
    auto coalescer = reader.create_batch_coalescer();
    auto splits    = coalescer->push(task({}));
    auto tail      = coalescer->flush();
    for (auto& split : tail) {
      splits.push_back(std::move(split));
    }
    REQUIRE(splits.size() == 1);
    auto const& split = dynamic_cast<scan::parquet_split_info const&>(*splits.front());
    CHECK(split.read_mode == (channel_state == 2 ? scan::parquet_read_mode::READER
                                                 : scan::parquet_read_mode::METADATA_ONLY));
    auto state = con.context->registered_state->Get<duckdb::SiriusContext>("sirius_state");
    REQUIRE(state);
    auto const* space =
      state->get_memory_manager().get_memory_space(cucascade::memory::Tier::GPU, 0);
    REQUIRE(space);
    auto decoded = reader.materialize_metadata_to_table(
      split, *space, cudf::get_default_stream(), false, nullptr);
    CHECK(decoded.table.num_rows() == 19);
    CHECK(reader.reader_calls() == (channel_state == 2 ? 1 : 0));
  }
  REQUIRE_FALSE(con.Query("ROLLBACK")->HasError());
}
