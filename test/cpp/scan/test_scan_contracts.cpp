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

#include "exec/stream_bind_catalog.hpp"
#include "exec/stream_plan_bindings.hpp"
#include "helper/type_conversions.hpp"
#include "op/scan/table_scan/bound_read_view.hpp"
#include "op/scan/table_scan/scan_contract.hpp"
#include "op/sirius_physical_streaming_source.hpp"
#include "pipeline/sirius_pipeline_converter.hpp"
#include "planner/connector_registry.hpp"
#include "planner/sirius_physical_plan_generator.hpp"
#include "sirius_extension.hpp"
#include "transparent/read_view_registry.hpp"
#include "utils/child_process_environment.hpp"
#include "utils/log_test_utils.hpp"
#include "utils/parquet_fixture_utils.hpp"
#include "utils/pipeline_conversion_test_utils.hpp"
#include "utils/sirius_test_env.hpp"

#include <catch.hpp>
#include <duckdb/catalog/catalog.hpp>
#include <duckdb/catalog/catalog_entry/table_catalog_entry.hpp>
#include <duckdb/catalog/catalog_entry/table_function_catalog_entry.hpp>
#include <duckdb/common/multi_file/multi_file_function.hpp>
#include <duckdb/common/multi_file/multi_file_list.hpp>
#include <duckdb/function/table/table_scan.hpp>
#include <duckdb/main/config.hpp>
#include <duckdb/main/extension/extension_loader.hpp>
#include <duckdb/main/extension_helper.hpp>
#include <duckdb/main/extension_manager.hpp>
#include <duckdb/planner/operator/logical_get.hpp>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <csignal>
#include <set>
#include <string>
#include <thread>
#include <vector>

namespace {
using namespace sirius::op::scan;

bound_read_view file_view(
  std::vector<std::string> const& paths,
  std::string selector                      = "",
  duckdb::vector<duckdb::LogicalType> types = {duckdb::LogicalType::INTEGER},
  duckdb::vector<std::string> names         = {"id"})
{
  bound_read_identity identity;
  identity.source      = {"read_parquet", source_kind::parquet_local, "duckdb.read_parquet.v1"};
  identity.data_view   = file_inventory{static_cast<uint32_t>(paths.size())};
  identity.bound_types = std::move(types);
  identity.bound_names = std::move(names);
  identity.selector    = std::move(selector);
  bound_read_view view;
  view.identity = make_bound_read_identity(std::move(identity), paths);
  return view;
}

duckdb::LogicalGet& first_get(duckdb::LogicalOperator& node)
{
  if (node.type == duckdb::LogicalOperatorType::LOGICAL_GET) {
    return node.Cast<duckdb::LogicalGet>();
  }
  for (auto& child : node.children) {
    try {
      return first_get(*child);
    } catch (std::out_of_range const&) {
    }
  }
  throw std::out_of_range("No LogicalGet");
}

void fake_scan(duckdb::ClientContext&, duckdb::TableFunctionInput&, duckdb::DataChunk&) {}

duckdb::unique_ptr<duckdb::GlobalTableFunctionState> fake_global(duckdb::ClientContext&,
                                                                 duckdb::TableFunctionInitInput&)
{
  throw duckdb::InvalidInputException("replacement global initializer");
}
duckdb::unique_ptr<duckdb::LocalTableFunctionState> fake_local(duckdb::ExecutionContext&,
                                                               duckdb::TableFunctionInitInput&,
                                                               duckdb::GlobalTableFunctionState*)
{
  throw duckdb::InvalidInputException("replacement local initializer");
}
void fake_serialize(duckdb::Serializer&,
                    duckdb::optional_ptr<duckdb::FunctionData>,
                    duckdb::TableFunction const&)
{
  throw duckdb::NotImplementedException("replacement serializer");
}
duckdb::unique_ptr<duckdb::FunctionData> fake_deserialize(duckdb::Deserializer&,
                                                          duckdb::TableFunction&)
{
  throw duckdb::NotImplementedException("replacement deserializer");
}

void replace_callback(duckdb::TableFunction& function, std::string_view phase)
{
  if (phase.ends_with("init_global"))
    function.init_global = fake_global;
  else if (phase.ends_with("init_local"))
    function.init_local = fake_local;
  else if (phase.ends_with("_deserialize"))
    function.deserialize = fake_deserialize;
  else if (phase.ends_with("_serialize"))
    function.serialize = fake_serialize;
  else
    function.function = fake_scan;
}

void require_registered_callbacks(duckdb::TableFunction const& actual,
                                  duckdb::TableFunction const& expected)
{
  REQUIRE(actual.function == expected.function);
  REQUIRE(actual.init_global == expected.init_global);
  REQUIRE(actual.init_local == expected.init_local);
  REQUIRE(actual.serialize == expected.serialize);
  REQUIRE(actual.deserialize == expected.deserialize);
}

duckdb::TableCatalogEntry* replacement_table = nullptr;

duckdb::unique_ptr<duckdb::FunctionData> replacement_bind(
  duckdb::ClientContext&,
  duckdb::TableFunctionBindInput& input,
  duckdb::vector<duckdb::LogicalType>& types,
  duckdb::vector<std::string>& names)
{
  types            = {duckdb::LogicalType::INTEGER};
  names            = {"id"};
  auto const& name = input.table_function.name;
  if (name == "seq_scan") return duckdb::make_uniq<duckdb::TableScanBindData>(*replacement_table);
  if (name == "sirius_read_parquet")
    return duckdb::make_uniq<duckdb::SiriusReadParquetBindData>("s3://bucket/a.parquet", 1);
  if (name == "sirius_stream_source")
    return duckdb::make_uniq<sirius::exec::stream_source_bind_data>(1);
  return duckdb::make_uniq<duckdb::MultiFileBindData>();
}

duckdb::unique_ptr<duckdb::FunctionData> fake_bind(duckdb::ClientContext&,
                                                   duckdb::TableFunctionBindInput&,
                                                   duckdb::vector<duckdb::LogicalType>&,
                                                   duckdb::vector<std::string>&)
{
  return nullptr;
}
duckdb::unique_ptr<duckdb::MultiFileReader> fake_reader(duckdb::TableFunction const&)
{
  return nullptr;
}
struct registry_test_generator : sirius::planner::sirius_physical_plan_generator {
  using sirius_physical_plan_generator::create_plan;
  using sirius_physical_plan_generator::sirius_physical_plan_generator;
};
}  // namespace

TEST_CASE("Read-view equality ignores independent evidence", "[scan][contracts]")
{
  auto base = file_view({"one.parquet"});
  std::vector<bound_read_view> views{base};
  for (int field = 0; field < 2; ++field) {
    for (int value = 0; value < 3; ++value) {
      auto view                      = file_view({"one.parquet"});
      auto evidence                  = std::make_shared<file_evidence_arrays>();
      evidence->size                 = {value == 2 ? 1024 : 512};
      evidence->size_present         = {static_cast<uint8_t>(field == 0 && value != 0)};
      evidence->etag                 = {field == 1 && value != 0 ? (value == 2 ? "v2" : "v1") : ""};
      view.evidence                  = std::move(evidence);
      view.transaction_id            = 100 + value;
      view.logical_selector_evidence = std::to_string(value);
      views.push_back(std::move(view));
    }
  }
  for (auto const& a : views) {
    for (auto const& b : views) {
      for (auto const& c : views) {
        REQUIRE(canonical_read_view_text(a) == canonical_read_view_text(b));
        REQUIRE(canonical_read_view_text(b) == canonical_read_view_text(c));
        REQUIRE(canonical_read_view_text(c) == canonical_read_view_text(a));
        REQUIRE(a.identity->fingerprint.hash == b.identity->fingerprint.hash);
      }
    }
  }
}

TEST_CASE("Read-view file inventory is a collision-safe multiset", "[scan][contracts]")
{
  auto a = file_view({"a|b", "'\n", ""});
  REQUIRE(canonical_read_view_text(a) == canonical_read_view_text(file_view({"", "'\n", "a|b"})));
  REQUIRE(canonical_read_view_text(a) !=
          canonical_read_view_text(file_view({"a|b", "'\n", "", ""})));
  REQUIRE(canonical_read_view_text(file_view({"a", "b"})) !=
          canonical_read_view_text(file_view({"a|b"})));
  REQUIRE(canonical_read_view_text(file_view({""})) != canonical_read_view_text(file_view({})));
  REQUIRE(canonical_read_view_text(file_view({"a", "b"})) !=
          canonical_read_view_text(file_view({"a"})));
  REQUIRE(canonical_read_view_text(file_view({std::string("a\0b", 3)})) !=
          canonical_read_view_text(file_view({"a"})));
}

TEST_CASE("Read-view path encoding preserves the length-delimited bytes", "[scan][contracts]")
{
  for (auto paths : std::vector<std::vector<std::string>>{{},
                                                          {""},
                                                          {"a", "b", "a"},
                                                          {std::string("a\0b", 3), "'\n", "|:"},
                                                          {std::string(9, 'a'),
                                                           std::string(10, 'b'),
                                                           std::string(99, 'c'),
                                                           std::string(100, 'd')}}) {
    auto view = file_view(paths);
    std::sort(paths.begin(), paths.end());
    // file_view always uses the same source, schema, and selector. Derive the
    // fixed header/tail from the empty inventory and encode the paths independently.
    auto empty = canonical_read_view_text(file_view({}));
    // Use the known variant index and locate its following count without relying on
    // delimiter searches inside arbitrary path bytes.
    auto empty_identity = file_view({}).identity;
    auto encode_number  = [](std::size_t value) {
      auto digits = std::to_string(value);
      return "u" + std::to_string(digits.size()) + ":" + digits;
    };
    auto source_prefix = std::string("v18:sirius.read-view.1s12:read_parquet") +
                         encode_number(static_cast<uint8_t>(source_kind::parquet_local)) +
                         "s22:duckdb.read_parquet.v1" +
                         encode_number(empty_identity->data_view.index());
    REQUIRE(empty.starts_with(source_prefix + encode_number(0)));
    auto expected = source_prefix + encode_number(paths.size());
    for (auto const& path : paths) {
      expected += "p" + std::to_string(path.size()) + ":";
      expected += path;
    }
    expected += empty.substr(source_prefix.size() + encode_number(0).size());
    CHECK(canonical_read_view_text(view) == expected);
  }
}

TEST_CASE("Read-view identity changes with bound schema", "[scan][contracts]")
{
  auto base = file_view({"a"});
  REQUIRE(canonical_read_view_text(base) !=
          canonical_read_view_text(file_view({"a"}, "", {duckdb::LogicalType::BIGINT})));
  REQUIRE(
    canonical_read_view_text(base) !=
    canonical_read_view_text(file_view({"a"}, "", {duckdb::LogicalType::INTEGER}, {"renamed"})));
}

TEST_CASE("Parquet identity includes bound explicit cardinality but not optimizer estimates",
          "[scan][contracts][shared_context]")
{
  REQUIRE(sirius::test::g_shared_env);
  auto con = sirius::test::g_shared_env->make_connection();
  REQUIRE_FALSE(con.Query("SET gpu_execution=false")->HasError());
  REQUIRE_FALSE(con.Query("BEGIN")->HasError());
  auto const parquet =
    std::string(SIRIUS_PROJECT_ROOT) + "/test/cpp/integration/data/parquet/lineitem.parquet";

  auto plan_100 =
    con.ExtractPlan("SELECT * FROM read_parquet('" + parquet + "', explicit_cardinality=100)");
  auto& get_100        = first_get(*plan_100);
  auto const bound_100 = capture_bound_read_view(get_100, *con.context);
  CHECK(bound_100.metrics.file_count == 1);
  CHECK(bound_100.metrics.canonical_capacity >= canonical_read_view_text(bound_100).size() + 1);
  CHECK(bound_100.metrics.evidence_capacity > 0);
  CHECK(bound_100.metrics.transient_path_capacity >= parquet.size() + 1);
  CHECK(bound_100.metrics.sort_index_capacity == 0);

  // An optimizer estimate is consumption/planning state, not part of the bound
  // read options. Changing it alone must leave the identity stable.
  ++get_100.estimated_cardinality;
  auto const reestimated_100 = capture_bound_read_view(get_100, *con.context);
  CHECK(canonical_read_view_text(bound_100) == canonical_read_view_text(reestimated_100));

  // explicit_cardinality is different: DuckDB serializes it as a bound parquet
  // option, so r18 requires it to remain in the identity.
  auto plan_200 =
    con.ExtractPlan("SELECT * FROM read_parquet('" + parquet + "', explicit_cardinality=200)");
  auto const bound_200 = capture_bound_read_view(first_get(*plan_200), *con.context);
  CHECK(canonical_read_view_text(bound_100) != canonical_read_view_text(bound_200));
  REQUIRE_FALSE(con.Query("ROLLBACK")->HasError());
}

TEST_CASE("Parquet captures preserve sorted evidence and independent inventories",
          "[scan][contracts][shared_context]")
{
  REQUIRE(sirius::test::g_shared_env);
  auto con = sirius::test::g_shared_env->make_connection();
  REQUIRE_FALSE(con.Query("SET gpu_execution=false")->HasError());
  REQUIRE_FALSE(con.Query("BEGIN")->HasError());
  auto const parquet =
    std::string(SIRIUS_PROJECT_ROOT) + "/test/cpp/integration/data/parquet/lineitem.parquet";
  auto plan  = con.ExtractPlan("SELECT * FROM read_parquet('" + parquet + "')");
  auto& get  = first_get(*plan);
  auto& bind = get.bind_data->Cast<duckdb::MultiFileBindData>();

  duckdb::vector<duckdb::OpenFileInfo> files;
  for (int i = 0; i < 3; ++i) {
    duckdb::OpenFileInfo file(std::string(100, static_cast<char>('a' + i)));
    if (i != 1) {
      file.extended_info       = duckdb::make_shared_ptr<duckdb::ExtendedOpenFileInfo>();
      auto& options            = file.extended_info->options;
      options["file_size"]     = duckdb::Value::BIGINT(100 + i);
      options["last_modified"] = duckdb::Value::TIMESTAMP(duckdb::timestamp_t(200 + i));
      options["etag"]          = duckdb::Value(std::string(50, static_cast<char>('x' + i)));
    }
    files.push_back(std::move(file));
  }
  std::vector<std::string> paths{files[0].path, files[1].path, files[2].path};
  std::vector<int> order{0, 1, 2};
  bound_read_view previous;
  do {
    duckdb::vector<duckdb::OpenFileInfo> shuffled;
    for (auto i : order)
      shuffled.push_back(files[i]);
    bind.file_list = duckdb::make_shared_ptr<duckdb::SimpleMultiFileList>(std::move(shuffled));
    auto resolved  = sirius::planner::resolve_parquet_scan_file_paths(
      get.function.name, get.bind_data.get(), get.parameters);
    REQUIRE(resolved.size() == order.size());
    for (std::size_t i = 0; i < order.size(); ++i)
      CHECK(resolved[i] == paths[order[i]]);
    // Resolving the candidate's paths must not move from the actual bound inventory.
    CHECK(sirius::planner::resolve_parquet_scan_file_paths(
            get.function.name, get.bind_data.get(), get.parameters) == resolved);
    auto view     = capture_bound_read_view(get, *con.context);
    auto expected = make_bound_read_identity(*view.identity, paths);
    CHECK(view.identity->fingerprint.canonical == expected->fingerprint.canonical);
    CHECK(view.identity->fingerprint.hash == expected->fingerprint.hash);
    REQUIRE(view.evidence);
    CHECK(view.evidence->size == std::vector<int64_t>{100, 0, 102});
    CHECK(view.evidence->last_modified == std::vector<int64_t>{200, 0, 202});
    CHECK(view.evidence->size_present == std::vector<uint8_t>{1, 0, 1});
    CHECK(view.evidence->last_modified_present == std::vector<uint8_t>{1, 0, 1});
    CHECK(view.evidence->etag ==
          std::vector<std::string>{std::string(50, 'x'), "", std::string(50, 'z')});
    CHECK(view.depth == evidence_depth::path_size_and_tag);
    if (std::is_sorted(order.begin(), order.end())) {
      CHECK(view.metrics.sort_index_capacity == 0);
    } else {
      CHECK(view.metrics.sort_index_capacity == 3 * sizeof(std::size_t));
    }
    if (previous.identity) {
      CHECK(previous.identity.get() != view.identity.get());
      CHECK(previous.evidence.get() != view.evidence.get());
    }
    previous = std::move(view);
  } while (std::next_permutation(order.begin(), order.end()));

  // ExtendedOpenFileInfo is shared by the snapshot; captured evidence must still own its tags.
  files[0].extended_info->options["etag"] = duckdb::Value("changed");
  auto changed                            = capture_bound_read_view(get, *con.context);
  CHECK(changed.evidence->etag[0] == "changed");
  CHECK(previous.evidence->etag[0] == std::string(50, 'x'));
  CHECK(previous.identity->fingerprint == changed.identity->fingerprint);

  // Duplicate paths remain a multiset, and a later capture cannot reuse an old inventory.
  files.push_back(files[0]);
  bind.file_list = duckdb::make_shared_ptr<duckdb::SimpleMultiFileList>(files);
  paths.push_back(paths[0]);
  auto duplicate = capture_bound_read_view(get, *con.context);
  CHECK(duplicate.metrics.file_count == 4);
  CHECK(duplicate.evidence->size == std::vector<int64_t>{100, 100, 0, 102});
  CHECK(duplicate.identity->fingerprint ==
        make_bound_read_identity(*duplicate.identity, paths)->fingerprint);
  CHECK_FALSE(duplicate.identity->fingerprint == previous.identity->fingerprint);

  // Modification times are retained evidence, but do not establish tag availability.
  for (bool has_size : {false, true}) {
    CAPTURE(has_size);
    duckdb::OpenFileInfo file("timestamp.parquet");
    file.extended_info = duckdb::make_shared_ptr<duckdb::ExtendedOpenFileInfo>();
    auto& options      = file.extended_info->options;
    if (has_size) { options["file_size"] = duckdb::Value::BIGINT(100); }
    bind.file_list = duckdb::make_shared_ptr<duckdb::SimpleMultiFileList>(
      duckdb::vector<duckdb::OpenFileInfo>{file});
    auto const without_timestamp = capture_bound_read_view(get, *con.context);

    options["last_modified"]  = duckdb::Value::TIMESTAMP(duckdb::timestamp_t(200));
    auto const with_timestamp = capture_bound_read_view(get, *con.context);
    REQUIRE(with_timestamp.evidence);
    CHECK(with_timestamp.evidence->last_modified == std::vector<int64_t>{200});
    CHECK(with_timestamp.evidence->last_modified_present == std::vector<uint8_t>{1});
    CHECK(with_timestamp.evidence->etag == std::vector<std::string>{""});
    CHECK(with_timestamp.depth ==
          (has_size ? evidence_depth::path_and_size : evidence_depth::path));
    CHECK(with_timestamp.depth == without_timestamp.depth);
    CHECK(with_timestamp.identity->fingerprint == without_timestamp.identity->fingerprint);
  }

  bind.file_list =
    duckdb::make_shared_ptr<duckdb::SimpleMultiFileList>(duckdb::vector<duckdb::OpenFileInfo>{});
  auto empty = capture_bound_read_view(get, *con.context);
  CHECK(empty.metrics.file_count == 0);
  CHECK(empty.metrics.sort_index_capacity == 0);
  CHECK(empty.evidence->size.empty());
  CHECK(empty.identity->fingerprint == make_bound_read_identity(*empty.identity, {})->fingerprint);
  CHECK(empty.replay_policy.permits_cpu_replay);
  bind.file_list =
    duckdb::make_shared_ptr<duckdb::SimpleMultiFileList>(duckdb::vector<duckdb::OpenFileInfo>{
      duckdb::OpenFileInfo("local.parquet"), duckdb::OpenFileInfo("S3://bucket/hidden.parquet")});
  auto remote = capture_bound_read_view(get, *con.context);
  CHECK_FALSE(remote.replay_policy.permits_cpu_replay);
  CHECK(remote.replay_policy.reason == "s3");
  CHECK(remote.replay_policy.source == sirius::transparent::byte_source_class::sirius_owned_s3);
  REQUIRE_FALSE(con.Query("ROLLBACK")->HasError());
}

TEST_CASE("Native read-view identity includes catalog schema and table incarnation",
          "[scan][contracts]")
{
  auto make = [](duckdb::idx_t catalog_oid, std::string schema, duckdb::idx_t table_oid) {
    bound_read_identity identity;
    identity.source = {"seq_scan", source_kind::duckdb_native, "duckdb.seq_scan.v1"};
    identity.data_view =
      native_table_identity{"db", catalog_oid, std::move(schema), "t", table_oid, "db.db"};
    identity.bound_types = {duckdb::LogicalType::INTEGER};
    identity.bound_names = {"id"};
    bound_read_view view;
    view.identity = make_bound_read_identity(std::move(identity), {});
    return canonical_read_view_text(view);
  };
  REQUIRE(make(1, "main", 2) != make(2, "main", 2));
  REQUIRE(make(1, "main", 2) != make(1, "other", 2));
  REQUIRE(make(1, "main", 2) != make(1, "main", 3));
}

TEST_CASE("Scan registry verifies all six catalog functions", "[scan][contracts][shared_context]")
{
  REQUIRE(sirius::test::g_shared_env);
  auto con = sirius::test::g_shared_env->make_connection();
  REQUIRE_FALSE(con.Query("SET gpu_execution=false")->HasError());
  REQUIRE_FALSE(con.Query("LOAD iceberg")->HasError());
  REQUIRE_FALSE(con.Query("CREATE TEMP TABLE r1_registry_native(id INTEGER)")->HasError());
  REQUIRE_FALSE(con.Query("BEGIN")->HasError());
  auto native      = con.ExtractPlan("SELECT * FROM r1_registry_native");
  auto& native_get = first_get(*native);
  REQUIRE(sirius::planner::lookup_connector(native_get, *con.context));

  std::set<std::string> names;
  for (auto const& entry : sirius::planner::registered_connectors())
    names.insert(entry.function_name);
  REQUIRE(names == std::set<std::string>{"seq_scan",
                                         "parquet_scan",
                                         "read_parquet",
                                         "sirius_read_parquet",
                                         "iceberg_scan",
                                         "sirius_stream_source"});
  for (auto const& name : names) {
    auto& catalog_entry =
      duckdb::Catalog::GetSystemCatalog(*con.context)
        .GetEntry<duckdb::TableFunctionCatalogEntry>(*con.context, DEFAULT_SCHEMA, name);
    for (auto function : catalog_entry.functions.functions) {
      duckdb::unique_ptr<duckdb::FunctionData> bind;
      if (name == "seq_scan") {
        bind = native_get.bind_data->Copy();
      } else if (name == "sirius_read_parquet") {
        bind = duckdb::make_uniq<duckdb::SiriusReadParquetBindData>("s3://bucket/a.parquet", 1);
      } else if (name == "sirius_stream_source") {
        bind = duckdb::make_uniq<sirius::exec::stream_source_bind_data>(1);
      } else {
        bind = duckdb::make_uniq<duckdb::MultiFileBindData>();
      }
      duckdb::LogicalGet get(1, function, std::move(bind), {}, {});
      REQUIRE(sirius::planner::lookup_connector(get, *con.context));
      get.function.function = fake_scan;
      REQUIRE_FALSE(sirius::planner::lookup_connector(get, *con.context));
      get.function      = function;
      get.function.bind = fake_bind;
      REQUIRE_FALSE(sirius::planner::lookup_connector(get, *con.context));
      get.function                       = function;
      get.function.get_multi_file_reader = fake_reader;
      REQUIRE_FALSE(sirius::planner::lookup_connector(get, *con.context));
      get.function           = function;
      get.function.arguments = {duckdb::LogicalType::BLOB, duckdb::LogicalType::BLOB};
      REQUIRE_FALSE(sirius::planner::lookup_connector(get, *con.context));
      get.function = function;
      get.bind_data.reset();
      REQUIRE_FALSE(sirius::planner::lookup_connector(get, *con.context));
    }
  }
  native_get.function.name = "r1_unknown_scan";
  REQUIRE_FALSE(sirius::planner::lookup_connector(native_get, *con.context));
  registry_test_generator generator(*con.context);
  REQUIRE_THROWS_WITH(
    generator.create_plan(native_get),
    Catch::Matchers::Contains("Table function 'r1_unknown_scan' is not supported in Sirius"));

  REQUIRE_FALSE(con.Query("ROLLBACK")->HasError());
}

TEST_CASE("Scan consumption fields do not alter read identity", "[scan][contracts][shared_context]")
{
  REQUIRE(sirius::test::g_shared_env);
  auto con = sirius::test::g_shared_env->make_connection();
  REQUIRE_FALSE(con.Query("SET gpu_execution=false")->HasError());
  REQUIRE_FALSE(con.Query("BEGIN")->HasError());
  auto const parquet =
    std::string(SIRIUS_PROJECT_ROOT) + "/test/cpp/integration/data/parquet/nation.parquet";
  auto first =
    con.ExtractPlan("SELECT n_name FROM read_parquet('" + parquet + "') WHERE n_nationkey > 5");
  auto second = con.ExtractPlan("SELECT n_regionkey FROM read_parquet('" + parquet +
                                "') WHERE n_nationkey < 10");
  auto& a     = first_get(*first);
  auto& b     = first_get(*second);
  REQUIRE(a.GetColumnIds() != b.GetColumnIds());
  REQUIRE(a.types != b.types);
  REQUIRE_FALSE(a.table_filters.filters.empty());
  REQUIRE_FALSE(b.table_filters.filters.empty());
  REQUIRE(a.table_filters.filters.begin()->second->DebugToString() !=
          b.table_filters.filters.begin()->second->DebugToString());
  auto const first_view  = capture_bound_read_view(a, *con.context);
  auto const second_view = capture_bound_read_view(b, *con.context);
  REQUIRE(first_view.identity != second_view.identity);
  CHECK(canonical_read_view_text(first_view) == canonical_read_view_text(second_view));
  CHECK(first_view.identity->fingerprint == second_view.identity->fingerprint);
  REQUIRE_FALSE(con.Query("ROLLBACK")->HasError());
}

TEST_CASE("Parquet identity captures bound file options", "[scan][contracts][shared_context]")
{
  REQUIRE(sirius::test::g_shared_env);
  auto con = sirius::test::g_shared_env->make_connection();
  REQUIRE_FALSE(con.Query("SET gpu_execution=false")->HasError());
  sirius::test::scratch_dir files("bound_options");
  auto const directory = files.path() / "part=one";
  std::filesystem::create_directories(directory);
  auto const path = sirius::test::sql_literal((directory / "data.parquet").string());
  REQUIRE_FALSE(con.Query("COPY (SELECT 1 AS id) TO " + path + " (FORMAT PARQUET)")->HasError());
  REQUIRE_FALSE(con.Query("BEGIN")->HasError());
  for (auto const* option : {"hive_partitioning", "union_by_name", "filename"}) {
    CAPTURE(option);
    auto capture = [&](bool enabled) {
      auto plan = con.ExtractPlan("SELECT * FROM read_parquet(" + path + ", " + option + "=" +
                                  (enabled ? "true" : "false") + ")");
      return capture_bound_read_view(first_get(*plan), *con.context);
    };
    auto const disabled = capture(false);
    auto const enabled  = capture(true);
    REQUIRE(disabled.identity != enabled.identity);
    CHECK(canonical_read_view_text(disabled) != canonical_read_view_text(enabled));
  }
  REQUIRE_FALSE(con.Query("ROLLBACK")->HasError());
}

TEST_CASE("Logical scan copy retains table index and verified source",
          "[scan][contracts][shared_context]")
{
  REQUIRE(sirius::test::g_shared_env);
  auto con = sirius::test::g_shared_env->make_connection();
  REQUIRE_FALSE(con.Query("SET gpu_execution=false")->HasError());
  REQUIRE_FALSE(
    con.Query("CREATE TEMP TABLE r1_copy_native(id INTEGER, value BIGINT)")->HasError());
  REQUIRE_FALSE(con.Query("INSERT INTO r1_copy_native VALUES (0, 1), (10, 2)")->HasError());
  REQUIRE_FALSE(con.Query("BEGIN")->HasError());
  auto plan        = con.ExtractPlan("SELECT value FROM r1_copy_native WHERE id > 5");
  auto& get        = first_get(*plan);
  auto copy        = get.Copy(*con.context);
  auto& copied_get = copy->Cast<duckdb::LogicalGet>();
  REQUIRE(copied_get.table_index == get.table_index);
  REQUIRE(sirius::planner::lookup_connector(copied_get, *con.context));
  REQUIRE_FALSE(con.Query("ROLLBACK")->HasError());
}

TEST_CASE("Physical scan lowering records its finalize generation",
          "[scan][contracts][shared_context]")
{
  REQUIRE(sirius::test::g_shared_env);
  auto con = sirius::test::g_shared_env->make_connection();
  REQUIRE_FALSE(con.Query("SET gpu_execution=false")->HasError());
  REQUIRE_FALSE(con.Query("BEGIN")->HasError());

  auto const parquet =
    std::string(SIRIUS_PROJECT_ROOT) + "/test/cpp/integration/data/parquet/lineitem.parquet";
  auto logical = con.ExtractPlan("SELECT * FROM read_parquet('" + parquet + "')");
  registry_test_generator generator(*con.context,
                                    sirius::planner::scan_contract_provenance{std::nullopt, 77});
  REQUIRE(generator.create_plan(std::move(logical)) != nullptr);
  REQUIRE(generator.read_views->entries().size() == 1);
  CHECK_FALSE(generator.read_views->entries().front().window_id.has_value());
  CHECK(generator.read_views->entries().front().finalize_generation == 77);
  REQUIRE_FALSE(con.Query("ROLLBACK")->HasError());
}

TEST_CASE("Physical stream lowering records its window without file checks",
          "[scan][contracts][shared_context]")
{
  REQUIRE(sirius::test::g_shared_env);
  auto con = sirius::test::g_shared_env->make_connection();
  REQUIRE_FALSE(con.Query("SET gpu_execution=false")->HasError());
  REQUIRE_FALSE(con.Query("BEGIN")->HasError());

  auto catalog = duckdb::make_shared_ptr<sirius::exec::stream_bind_catalog>();
  con.context->registered_state->Insert(sirius::exec::stream_bind_catalog::kStateKey, catalog);
  auto registered_catalog = sirius::exec::catalog_for(*con.context);
  registered_catalog->declare(
    1,
    sirius::exec::stream_input_binding{
      {"a"},
      sirius::from_duckdb_vec(duckdb::vector<duckdb::LogicalType>{duckdb::LogicalType::INTEGER}),
      std::make_shared<cucascade::shared_data_repository>(),
      {0},
      nullptr});

  auto& entry = duckdb::Catalog::GetSystemCatalog(*con.context)
                  .GetEntry<duckdb::TableFunctionCatalogEntry>(
                    *con.context, DEFAULT_SCHEMA, "sirius_stream_source");
  auto function = entry.functions.functions.front();
  duckdb::LogicalGet get(1,
                         std::move(function),
                         duckdb::make_uniq<sirius::exec::stream_source_bind_data>(1),
                         {duckdb::LogicalType::INTEGER},
                         {"a"});
  get.SetColumnIds({duckdb::ColumnIndex(0)});
  get.estimated_cardinality = 1;

  registry_test_generator generator(*con.context,
                                    sirius::planner::scan_contract_provenance{{42}, 0});
  auto physical = generator.create_plan(get);
  REQUIRE(physical != nullptr);
  REQUIRE(physical->type == sirius::op::SiriusPhysicalOperatorType::STREAMING_SOURCE);
  auto const& source         = physical->Cast<sirius::op::sirius_physical_streaming_source>();
  auto const& contract_entry = source.read_views()->entry(source.contract_id());
  CHECK(contract_entry.window_id == 42);
  CHECK(contract_entry.finalize_generation == 0);
  CHECK(contract_entry.eligibility.later_checks.empty());
  CHECK_FALSE(contract_entry.contract.view->replay_policy.permits_cpu_replay);
  CHECK(contract_entry.contract.view->replay_policy.reason == "stream");
  sirius::pipeline::pipeline_build_context build_context(nullptr);
  auto pipeline = std::make_shared<sirius::pipeline::sirius_pipeline>(build_context);
  sirius::pipeline::sirius_pipeline_build_state state;
  state.set_pipeline_source(*pipeline, *physical);
  sirius::pipeline::pipeline_conversion_result result{{pipeline}, {}, 1};
  for (auto const& dump : {sirius::pipeline::dump_pipeline_conversion_result(result),
                           sirius::pipeline::dump_pipeline_schedule_raw(result)}) {
    CHECK(dump.find("pushdown_mode=none scan_cpu_replay=forbidden scan_replay_veto=stream") !=
          std::string::npos);
    CHECK(dump.find("plan_cpu_replay=forbidden veto=stream") != std::string::npos);
  }
  REQUIRE_FALSE(con.Query("ROLLBACK")->HasError());
}

TEST_CASE("Canonical values distinguish NULL empty text and logical type", "[scan][contracts]")
{
  REQUIRE(canonical_value_text(duckdb::Value(duckdb::LogicalType::VARCHAR)) !=
          canonical_value_text(duckdb::Value("")));
  REQUIRE(canonical_value_text(duckdb::Value(duckdb::LogicalType::VARCHAR)) !=
          canonical_value_text(duckdb::Value(duckdb::LogicalType::INTEGER)));
  auto first  = file_view({"a"}).identity->fingerprint;
  auto second = file_view({"b"}).identity->fingerprint;
  second.hash = first.hash;
  REQUIRE_FALSE(first == second);
}

TEST_CASE("Dropping and recreating a native table changes read identity",
          "[scan][contracts][shared_context]")
{
  REQUIRE(sirius::test::g_shared_env);
  auto con = sirius::test::g_shared_env->make_connection();
  REQUIRE_FALSE(con.Query("SET gpu_execution=false")->HasError());
  auto identity = [&] {
    REQUIRE_FALSE(con.Query("BEGIN")->HasError());
    auto plan = con.ExtractPlan("SELECT * FROM r1_recreated_native");
    auto view = capture_bound_read_view(first_get(*plan), *con.context);
    REQUIRE_FALSE(con.Query("COMMIT")->HasError());
    return view;
  };
  REQUIRE_FALSE(con.Query("CREATE TEMP TABLE r1_recreated_native(id INTEGER)")->HasError());
  auto before = identity();
  REQUIRE_FALSE(con.Query("DROP TABLE r1_recreated_native")->HasError());
  REQUIRE_FALSE(con.Query("CREATE TEMP TABLE r1_recreated_native(id INTEGER)")->HasError());
  auto after = identity();
  CHECK(std::get<native_table_identity>(before.identity->data_view).table_oid !=
        std::get<native_table_identity>(after.identity->data_view).table_oid);
  CHECK_FALSE(before.identity->fingerprint == after.identity->fingerprint);
}

TEST_CASE("Scan registry rejects registered replacements before or after first lookup",
          "[scan][contracts][isolated_context]")
{
  auto const config = std::filesystem::path(SIRIUS_PROJECT_ROOT) / "test" / "cpp" / "integration" /
                      "s3" / "sirius.yaml";
  REQUIRE(std::filesystem::is_regular_file(config));
  for (auto const* phase : {"cold",
                            "warm",
                            "cold_same",
                            "warm_same",
                            "cold_init_global",
                            "cold_init_local",
                            "cold_serialize",
                            "cold_deserialize",
                            "warm_init_global",
                            "warm_init_local",
                            "warm_serialize",
                            "warm_deserialize",
                            "preload_init_global",
                            "preload_init_local",
                            "preload_serialize",
                            "preload_deserialize",
                            "preload_dynamic_init_global",
                            "preload_dynamic_init_local",
                            "preload_dynamic_serialize",
                            "preload_dynamic_deserialize",

                            "preload",
                            "preload_dynamic",
                            "iceberg_first",
                            "iceberg_first_same",
                            "iceberg_last",
                            "iceberg_first_dynamic",
                            "iceberg_last_dynamic",
                            "iceberg_unavailable",
                            "iceberg_disabled"}) {
    INFO(phase);
    bool const iceberg = std::string_view(phase).starts_with("iceberg_");
    bool const preload = iceberg || std::string_view(phase).starts_with("preload");
    // The parent test process may leave integration.yaml in the environment after pausing its
    // shared database.  A child must not reserve that config's 50% GPU pool beside the parent;
    // callback verification needs only a small Sirius context.
    sirius::test::child_process_environment environment{
      {{"SIRIUS_REGISTRY_TRUST_PHASE", phase},
       {"SIRIUS_TEST_SHARED_CONFIG_OVERRIDE", config.string()},
       {"SIRIUS_REGISTRY_PRELOAD_CHILD", preload ? "1" : "0"}}};
    std::string executable = "sirius_unittest";
    std::string filter     = iceberg ? "Iceberg trust bootstrap load order child"
                             : preload ? "Scan registry rejects replacement before Sirius load child"
                                       : "Scan registry first lookup child";
    char* arguments[]      = {executable.data(), filter.data(), nullptr};
    pid_t pid{};
    REQUIRE(
      ::posix_spawn(&pid, "/proc/self/exe", nullptr, nullptr, arguments, environment.data()) == 0);
    int status{};
    auto const deadline = std::chrono::steady_clock::now() + std::chrono::seconds(60);
    pid_t waited{};
    while ((waited = ::waitpid(pid, &status, WNOHANG)) == 0 &&
           std::chrono::steady_clock::now() < deadline)
      std::this_thread::sleep_for(std::chrono::milliseconds(20));
    if (waited == 0) {
      ::kill(pid, SIGKILL);
      ::waitpid(pid, &status, 0);
      FAIL("registry lookup child timed out");
    }
    REQUIRE(waited == pid);
    REQUIRE(WIFEXITED(status));
    CHECK(WEXITSTATUS(status) == 0);
  }
}

TEST_CASE("Iceberg trust bootstrap load order child", "[.][registry_trust_child]")
{
  REQUIRE(sirius::test::g_shared_env == nullptr);
  auto const phase          = std::string(std::getenv("SIRIUS_REGISTRY_TRUST_PHASE"));
  bool const iceberg_first  = phase.find("last") == std::string::npos;
  bool const dynamic        = phase.ends_with("_dynamic");
  bool const same_signature = phase.ends_with("_same");
  bool const unavailable    = phase.ends_with("_unavailable");
  bool const disabled       = phase.ends_with("_disabled");
  bool const trace          = std::getenv("SIRIUS_REGISTRY_IO_TRACE") != nullptr;
  auto marker               = [&](std::string const& value) {
    if (!trace) return;
    auto line = "R1_BOOTSTRAP " + value + "\n";
    REQUIRE(::write(STDERR_FILENO, line.data(), line.size()) == static_cast<ssize_t>(line.size()));
  };

  duckdb::DBConfig config;
  config.options.load_extensions = false;
  config.SetOptionByName("allow_unsigned_extensions", duckdb::Value::BOOLEAN(true));
  duckdb::DuckDB database(nullptr, &config);
  duckdb::ExtensionHelper::LoadExtension(database, "parquet");
  duckdb::ExtensionHelper::LoadExtension(database, "core_functions");
  duckdb::Connection con(database);
  duckdb::ExtensionLoader loader(*database.instance, "iceberg_bootstrap_test");
  auto load_iceberg = [&] {
    marker("BEGIN iceberg_load");
    auto result = con.Query("LOAD iceberg");
    marker("END iceberg_load");
    INFO((result->HasError() ? result->GetError() : "loaded"));
    REQUIRE_FALSE(result->HasError());
  };
  auto load_sirius = [&] {
    setenv("SIRIUS_CONFIG_FILE", std::getenv("SIRIUS_TEST_SHARED_CONFIG_OVERRIDE"), 1);
    if (disabled)
      setenv("SIRIUS_DISABLE", "1", 1);
    else
      unsetenv("SIRIUS_DISABLE");
    marker("BEGIN sirius_load");
    if (dynamic) {
      auto const executable = std::filesystem::canonical("/proc/self/exe");
      auto const extension =
        executable.parent_path().parent_path().parent_path() / "sirius.duckdb_extension";
      auto result = con.Query("LOAD " + sirius::test::sql_literal(extension.string()));
      INFO((result->HasError() ? result->GetError() : "loaded"));
      REQUIRE_FALSE(result->HasError());
    } else {
      database.LoadStaticExtension<duckdb::SiriusExtension>();
    }
    marker("END sirius_load");
    REQUIRE_FALSE(con.Query("SET gpu_execution=false")->HasError());
  };

  if (iceberg_first)
    load_iceberg();
  else
    load_sirius();

  duckdb::TableFunction original;
  if (iceberg_first) original = loader.GetTableFunction("iceberg_scan").functions.functions.front();
  auto replacement =
    same_signature ? original
                   : duckdb::TableFunction(
                       "iceberg_scan", {duckdb::LogicalType::INTEGER}, fake_scan, replacement_bind);
  replacement.function = fake_scan;
  if (same_signature) {
    duckdb::CreateTableFunctionInfo replace(replacement);
    replace.on_conflict = duckdb::OnCreateConflict::REPLACE_ON_CONFLICT;
    loader.RegisterFunction(std::move(replace));
  } else {
    loader.RegisterFunction(replacement);
  }

  // A missing trusted definition must stay unverified during lookup, even when the
  // library becomes locatable later; lookup may not retry cold initialization.
  auto info = duckdb::ExtensionManager::Get(*database.instance).GetExtensionInfo("iceberg");
  duckdb::ExtensionInstallInfo saved;
  if (unavailable) {
    REQUIRE(info);
    REQUIRE(info->install_info);
    saved                         = *info->install_info;
    info->install_info->mode      = duckdb::ExtensionInstallMode::NOT_INSTALLED;
    info->install_info->full_path = "/nonexistent/sirius-registry-bootstrap-test";
  }
  if (iceberg_first)
    load_sirius();
  else
    load_iceberg();
  if (unavailable) *info->install_info = saved;

  sirius::test::scoped_recording_log_sink logs;
  REQUIRE_FALSE(con.Query("BEGIN")->HasError());
  duckdb::MultiFileBindData bind;
  if (!dynamic) {
    marker("BEGIN first_lookup");
    auto const* rejected = sirius::planner::lookup_connector(replacement, &bind, *con.context);
    marker("END first_lookup");
    CHECK_FALSE(rejected);
  }
  // Restore the standard overload when testing replacement of its exact signature.
  if (same_signature) {
    duckdb::CreateTableFunctionInfo restore(original);
    restore.on_conflict = duckdb::OnCreateConflict::REPLACE_ON_CONFLICT;
    loader.RegisterFunction(std::move(restore));
  }
  if (!dynamic) {
    for (auto const& function : loader.GetTableFunction("iceberg_scan").functions.functions) {
      if (function.function == fake_scan) continue;
      marker("BEGIN genuine_lookup");
      auto const* trusted = sirius::planner::lookup_connector(function, &bind, *con.context);
      marker("END genuine_lookup");
      CHECK(static_cast<bool>(trusted) == (!unavailable && !disabled));
    }
  }
  REQUIRE_FALSE(con.Query("ROLLBACK")->HasError());
  if (!dynamic) {
    auto const records  = logs.records();
    auto const warnings = std::count_if(records.begin(), records.end(), [](auto const& record) {
      return record.level == sirius::log::level::warn &&
             record.message.find(
               "GPU scan source 'iceberg_scan' has no trusted reference definition") !=
               std::string::npos;
    });
    CHECK(warnings == (unavailable || disabled ? 1 : 0));
  }
  if (unavailable || disabled) return;

  // Exercise the actual loaded Sirius module too: its private cache differs from the
  // statically linked test executable's cache when LOAD uses the shared extension.
  REQUIRE_FALSE(con.Query("SET gpu_execution=true")->HasError());
  REQUIRE_FALSE(con.Query("SET enable_duckdb_fallback=false")->HasError());
  if (!same_signature) {
    auto rejected = con.Query("SELECT * FROM iceberg_scan(42)");
    INFO((rejected->HasError() ? rejected->GetError() : "unexpected admission"));
    REQUIRE(rejected->HasError());
    CHECK(rejected->GetError().find("unverified callbacks") != std::string::npos);
  }
  REQUIRE_FALSE(con.Query("SET sirius_test_inject_transparent_gpu_error='t6'")->HasError());
  auto const path =
    std::string(SIRIUS_PROJECT_ROOT) + "/test/cpp/integration/data/iceberg_snapshot_deletes";
  auto genuine = con.Query("SELECT sum(count) FROM iceberg_scan('" + path +
                           "', snapshot_from_id=9400000000000002)");
  INFO((genuine->HasError() ? genuine->GetError() : "missing injected error"));
  REQUIRE(genuine->HasError());
  CHECK(genuine->GetError().find("injected transparent GPU failure: t6") != std::string::npos);
}

TEST_CASE("Scan registry rejects replacement before Sirius load child", "[.][registry_trust_child]")
{
  REQUIRE(sirius::test::g_shared_env == nullptr);
  duckdb::DBConfig config;
  config.options.load_extensions = false;
  config.SetOptionByName("allow_unsigned_extensions", duckdb::Value::BOOLEAN(true));
  duckdb::DuckDB database(nullptr, &config);
  duckdb::ExtensionHelper::LoadExtension(database, "parquet");
  sirius::test::scratch_dir files("registry_preload");
  {
    duckdb::Connection setup(database);
    REQUIRE_FALSE(setup
                    .Query("COPY (SELECT 1::INTEGER AS i) TO " +
                           files.file_literal("input.parquet") + " (FORMAT PARQUET)")
                    ->HasError());
  }
  duckdb::ExtensionLoader loader(*database.instance, "registry_preload_test");
  auto original    = loader.GetTableFunction("read_parquet").functions.functions.front();
  auto replacement = original;
  auto const phase = std::string(std::getenv("SIRIUS_REGISTRY_TRUST_PHASE"));
  replace_callback(replacement, phase);
  duckdb::CreateTableFunctionInfo info(replacement);
  info.on_conflict = duckdb::OnCreateConflict::REPLACE_ON_CONFLICT;
  loader.RegisterFunction(std::move(info));
  require_registered_callbacks(loader.GetTableFunction("read_parquet").functions.functions.front(),
                               replacement);

  // Explicitly load Sirius only after the replacement is committed to the catalog.
  setenv("SIRIUS_CONFIG_FILE", std::getenv("SIRIUS_TEST_SHARED_CONFIG_OVERRIDE"), 1);
  unsetenv("SIRIUS_DISABLE");
  bool const dynamic = phase.starts_with("preload_dynamic");
  if (dynamic) {
    auto const executable = std::filesystem::canonical("/proc/self/exe");
    auto const extension =
      executable.parent_path().parent_path().parent_path() / "sirius.duckdb_extension";
    duckdb::Connection load(database);
    auto result = load.Query("LOAD " + sirius::test::sql_literal(extension.string()));
    INFO((result->HasError() ? result->GetError() : "loaded"));
    REQUIRE_FALSE(result->HasError());
  } else {
    database.LoadStaticExtension<duckdb::SiriusExtension>();
  }
  duckdb::Connection con(database);
  REQUIRE_FALSE(con.Query("SET gpu_execution=false")->HasError());
  if (phase.ends_with("init_global") || phase.ends_with("init_local")) {
    auto cpu = con.Query("SELECT * FROM read_parquet(" + files.file_literal("input.parquet") + ")");
    REQUIRE(cpu);
    REQUIRE(cpu->HasError());
    CHECK(cpu->GetError().find(phase.ends_with("init_global")
                                 ? "replacement global initializer"
                                 : "replacement local initializer") != std::string::npos);
  }
  REQUIRE_FALSE(con.Query("BEGIN")->HasError());
  duckdb::MultiFileBindData bind;
  CHECK_FALSE(sirius::planner::lookup_connector(replacement, &bind, *con.context));
  auto plan =
    con.ExtractPlan("SELECT * FROM read_parquet(" + files.file_literal("input.parquet") + ")");
  auto& get = first_get(*plan);
  require_registered_callbacks(get.function, replacement);
  registry_test_generator generator(*con.context);
  CHECK_THROWS_WITH(generator.create_plan(get), Catch::Matchers::Contains("unverified callbacks"));
  REQUIRE_FALSE(con.Query("COMMIT")->HasError());
  REQUIRE_FALSE(con.Query("SET gpu_execution=true")->HasError());
  REQUIRE_FALSE(con.Query("SET enable_duckdb_fallback=false")->HasError());
  auto rejected =
    con.Query("SELECT i FROM read_parquet(" + files.file_literal("input.parquet") + ")");
  INFO((rejected->HasError() ? rejected->GetError() : "unexpected admission"));
  REQUIRE(rejected->HasError());
  CHECK(rejected->GetError().find("unverified callbacks") != std::string::npos);
  REQUIRE_FALSE(con.Query("BEGIN")->HasError());
  duckdb::CreateTableFunctionInfo restore(original);
  restore.on_conflict = duckdb::OnCreateConflict::REPLACE_ON_CONFLICT;
  loader.RegisterFunction(std::move(restore));
  CHECK(sirius::planner::lookup_connector(original, &bind, *con.context));
  REQUIRE_FALSE(con.Query("ROLLBACK")->HasError());
}

TEST_CASE("Scan registry first lookup child", "[.][registry_trust_child][shared_context]")
{
  auto const* phase = std::getenv("SIRIUS_REGISTRY_TRUST_PHASE");
  if (!phase) return;
  REQUIRE(sirius::test::g_shared_env);
  auto con = sirius::test::g_shared_env->make_connection();
  REQUIRE_FALSE(con.Query("SET gpu_execution=false")->HasError());
  REQUIRE_FALSE(con.Query("LOAD iceberg")->HasError());
  REQUIRE_FALSE(con.Query("CREATE TEMP TABLE r1_replacement_native(id INTEGER)")->HasError());
  REQUIRE_FALSE(con.Query("INSERT INTO r1_replacement_native VALUES (1)")->HasError());
  REQUIRE_FALSE(con.Query("BEGIN")->HasError());
  auto native       = con.ExtractPlan("SELECT * FROM r1_replacement_native");
  replacement_table = &first_get(*native).bind_data->Cast<duckdb::TableScanBindData>().table;
  duckdb::ExtensionLoader loader(*con.context->db, "registry_replacement_test");
  for (auto const& source : sirius::planner::registered_connectors()) {
    INFO(source.function_name);
    auto& catalog = duckdb::Catalog::GetSystemCatalog(*con.context);
    auto original = catalog
                      .GetEntry<duckdb::TableFunctionCatalogEntry>(
                        *con.context, DEFAULT_SCHEMA, source.function_name)
                      .functions.functions.front();
    duckdb::unique_ptr<duckdb::FunctionData> bind;
    if (source.function_name == "seq_scan")
      bind = first_get(*native).bind_data->Copy();
    else if (source.function_name == "sirius_read_parquet")
      bind = duckdb::make_uniq<duckdb::SiriusReadParquetBindData>("s3://bucket/a.parquet", 1);
    else if (source.function_name == "sirius_stream_source")
      bind = duckdb::make_uniq<sirius::exec::stream_source_bind_data>(1);
    else
      bind = duckdb::make_uniq<duckdb::MultiFileBindData>();
    if (std::string_view(phase).starts_with("warm"))
      REQUIRE(sirius::planner::lookup_connector(original, bind.get(), *con.context));
    if (std::string_view(phase).ends_with("_same") ||
        std::string_view(phase).find("init_") != std::string_view::npos ||
        std::string_view(phase).ends_with("serialize")) {
      auto replacement = original;
      replace_callback(replacement, phase);
      duckdb::CreateTableFunctionInfo info(replacement);
      info.on_conflict = duckdb::OnCreateConflict::REPLACE_ON_CONFLICT;
      loader.RegisterFunction(std::move(info));
      auto registered = catalog
                          .GetEntry<duckdb::TableFunctionCatalogEntry>(
                            *con.context, DEFAULT_SCHEMA, source.function_name)
                          .functions.functions.front();
      require_registered_callbacks(registered, replacement);
      CHECK_FALSE(sirius::planner::lookup_connector(registered, bind.get(), *con.context));
      CHECK_FALSE(sirius::planner::lookup_connector(original, bind.get(), *con.context));
      duckdb::CreateTableFunctionInfo restore(original);
      restore.on_conflict = duckdb::OnCreateConflict::REPLACE_ON_CONFLICT;
      loader.RegisterFunction(std::move(restore));
    } else {
      duckdb::TableFunction replacement(
        source.function_name, {duckdb::LogicalType::INTEGER}, fake_scan, replacement_bind);
      loader.RegisterFunction(replacement);
      auto plan = con.ExtractPlan("SELECT * FROM " + source.function_name + "(42)");
      auto& get = first_get(*plan);
      REQUIRE(get.function.function == fake_scan);
      REQUIRE(source.bind_data_matches(get.bind_data.get()));
      CHECK_FALSE(sirius::planner::lookup_connector(get, *con.context));
      registry_test_generator generator(*con.context);
      CHECK_THROWS_WITH(generator.create_plan(get),
                        Catch::Matchers::Contains("unverified callbacks"));
    }
    CHECK(sirius::planner::lookup_connector(original, bind.get(), *con.context));
  }
  REQUIRE_FALSE(con.Query("ROLLBACK")->HasError());
  replacement_table = nullptr;
}

TEST_CASE("Scan diagnostics distinguish pushdown from replay permission",
          "[scan][contracts][shared_context]")
{
  REQUIRE(sirius::test::g_shared_env);
  auto con = sirius::test::g_shared_env->make_connection();
  auto path =
    std::string(SIRIUS_PROJECT_ROOT) + "/test/cpp/integration/data/parquet/lineitem.parquet";
  auto dump = sirius::test::convert_query_to_raw_schedule(
    con, "SELECT sum(l_orderkey) FROM read_parquet('" + path + "')");
  INFO(dump);
  CHECK(dump.find(" pushdown_mode=") != std::string::npos);
  CHECK(dump.find(" scan_cpu_replay=permitted") != std::string::npos);
  CHECK(dump.find("plan_cpu_replay=permitted veto=none") != std::string::npos);
  CHECK(dump.find(" policy=") == std::string::npos);
}

TEST_CASE("Read-view evidence indexes preserve file order and duplicates", "[scan][contracts]")
{
  std::vector<std::string> paths{"c", "a", "b", "a"};
  auto index = make_read_view_evidence_index(paths);
  REQUIRE(index.size() == paths.size());
  auto sorted = paths;
  std::sort(sorted.begin(), sorted.end());
  std::set<std::size_t> positions;
  for (std::size_t i = 0; i < paths.size(); ++i) {
    REQUIRE(index[i] < paths.size());
    CHECK(sorted[index[i]] == paths[i]);
    positions.insert(index[i]);
  }
  CHECK(positions.size() == paths.size());
  CHECK(paths == std::vector<std::string>{"c", "a", "b", "a"});
  CHECK(make_read_view_evidence_index({}).empty());
}
