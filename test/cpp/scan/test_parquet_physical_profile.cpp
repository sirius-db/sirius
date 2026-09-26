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

#include <cudf/io/experimental/hybrid_scan.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/io/parquet_io_utils.hpp>

// R2a T5: RED_FEATURE after commit (b).
#include <catch.hpp>
#include <io/parquet_helpers.hpp>
#include <op/scan/parquet_metadata.hpp>
#include <op/scan/table_scan/parquet_physical_profile.hpp>
#include <scan_manager/sirius_scan_manager.hpp>
#include <sirius_context.hpp>
#include <utils/gpu_execution_fixture.hpp>
#include <utils/parquet_fixture_utils.hpp>

#include <condition_variable>
#include <filesystem>
#include <fstream>
#include <numeric>

namespace {
using namespace sirius::op::scan;
namespace pq = cudf::io::parquet;
std::filesystem::path corpus()
{
  if (auto path = std::getenv("SIRIUS_PARQUET_PROFILE_FIXTURES")) {
    REQUIRE(std::filesystem::exists(std::filesystem::path(path) / "manifest.json"));
    return path;
  }
  // CI regenerates from the checked-in recipe; local evidence can reuse an
  // explicit fixture directory without making the tests depend on .ssw.
  static sirius::test::scratch_dir generated("parquet_profile_corpus");
  static bool const ready = [] {
    auto quote = [](std::string const& value) {
      std::string result = "'";
      for (char c : value)
        result += c == '\'' ? "'\\''" : std::string(1, c);
      return result + "'";
    };
    auto script = std::filesystem::path(SIRIUS_PROJECT_ROOT) /
                  "test/cpp/scan/data/physical_profile/generate.py";
    auto command =
      "python3 -B " + quote(script.string()) + " --output " + quote(generated.path().string());
    REQUIRE(std::system(command.c_str()) == 0);
    return true;
  }();
  (void)ready;
  return generated.path();
}
struct footer_fixture {
  pq::FileMetaData metadata;
  parquet_encryption_evidence encryption;
  bound_table_scan contract;
  effective_reader_projection projection;
  std::vector<std::size_t> retained;
  explicit footer_fixture(std::string const& file = "int-SNAPPY-PLAIN.parquet")
  {
    auto source  = cudf::io::datasource::create((corpus() / file).string());
    auto bytes   = fetch_plaintext_parquet_footer(*source, 73, file);
    encryption   = inspect_parquet_encryption({bytes->data(), bytes->size()});
    auto options = cudf::io::parquet_reader_options::builder().build();
    pq::experimental::hybrid_scan_reader reader(
      cudf::host_span<uint8_t const>(bytes->data(), bytes->size()), options);
    metadata    = reader.parquet_metadata();
    auto schema = sirius::io::parquet_helpers::extract_schema(metadata);
    projection.names.assign(schema.names.begin(), schema.names.end());
    projection.bound_types.assign(schema.types.begin(), schema.types.end());
    for (std::size_t i = 0; i < projection.names.size(); ++i)
      projection.projected.push_back(i);
    retained.resize(metadata.row_groups.size());
    std::iota(retained.begin(), retained.end(), 0);
    contract.contract_id = 73;
    contract.profiles    = std::make_shared<physical_profile_table>();
  }
  physical_profile_result check(leaf_set semantic = {true})
  {
    return check_parquet_split_profile(
      metadata, encryption, contract, projection, retained, semantic);
  }
};
}  // namespace

TEST_CASE("Parquet raw footer crypto evidence is complete and bounded", "[scan][parquet][profile]")
{
  footer_fixture plain;
  CHECK(plain.encryption.complete);
  CHECK_FALSE(plain.encryption.columns_encrypted);
  auto source   = cudf::io::datasource::create((corpus() / "encrypted-columns.parquet").string());
  auto bytes    = fetch_plaintext_parquet_footer(*source, 73, "columns");
  auto evidence = inspect_parquet_encryption({bytes->data(), bytes->size()});
  CHECK(evidence.columns_encrypted);
  CHECK(evidence.complete);
  plain.encryption = evidence;
  CHECK(plain.check().reason == verdict_reason::parquet_encrypted);
  auto encrypted = cudf::io::datasource::create((corpus() / "encrypted-footer.parquet").string());
  try {
    fetch_plaintext_parquet_footer(*encrypted, 73, "footer-file");
    FAIL("PARE accepted");
  } catch (unsupported_physical_input const& error) {
    CHECK(error.contract == 73);
    CHECK(error.reason == verdict_reason::parquet_encrypted);
    CHECK(error.input_identity == "footer-file");
  }
  auto plain_source =
    cudf::io::datasource::create((corpus() / "int-SNAPPY-PLAIN.parquet").string());
  bytes = fetch_plaintext_parquet_footer(*plain_source, 73, "plain");
  for (std::size_t size = 0; size < bytes->size(); ++size) {
    CHECK_FALSE(inspect_parquet_encryption({bytes->data(), size}).complete);
  }
  plain.encryption = {};
  auto missing     = plain.check();
  CHECK_FALSE(missing.approved);
  CHECK(missing.reason == verdict_reason::parquet_encryption_evidence_missing);
  CHECK(missing.profile == 0);
}

TEST_CASE("Parquet profiles inspect only retained decoded chunks", "[scan][parquet][profile]")
{
  footer_fixture f;
  auto ok = f.check();
  REQUIRE(ok.approved);
  REQUIRE(ok.profile != 0);
  CHECK(ok.validation.test(static_cast<unsigned>(later_check::profile_per_file)));
  CHECK(f.contract.profiles->get(ok.profile).columns.size() == 2);
  f.metadata.row_groups[0].columns[0].meta_data.codec = pq::Compression::LZO;
  CHECK(f.check().reason == verdict_reason::parquet_codec_unsupported);
  f.retained = {1};
  CHECK(f.check().approved);
  f.retained = {0};
  f.projection.projected.clear();
  CHECK(f.check().approved);
  f.projection.natural_read = true;
  CHECK(f.check().reason == verdict_reason::parquet_codec_unsupported);
  f.retained.clear();
  CHECK(f.check().approved);
}

TEST_CASE("Parquet footer encoding union accepts levels and empty lists",
          "[scan][parquet][profile]")
{
  footer_fixture f;
  for (auto encodings : std::vector<std::vector<pq::Encoding>>{
         {}, {pq::Encoding::PLAIN, pq::Encoding::BIT_PACKED}, {pq::Encoding::RLE}}) {
    for (auto& group : f.metadata.row_groups)
      group.columns[0].meta_data.encodings = encodings;
    CHECK(f.check().approved);
  }
  f.metadata.row_groups[0].columns[0].meta_data.encodings = {static_cast<pq::Encoding>(63)};
  CHECK(f.check().reason == verdict_reason::parquet_codec_unsupported);
}

TEST_CASE("Parquet D6 requires both export-only usage and an actual export cast",
          "[scan][parquet][profile]")
{
  footer_fixture f;
  f.projection.bound_types[0] = duckdb::LogicalType::BIGINT;
  auto exported               = f.check({false});
  REQUIRE(exported.approved);
  CHECK(exported.type_mismatches == 1);
  CHECK(f.contract.profiles->get(exported.profile).columns[0].type_mismatch);
  CHECK(f.check({true}).reason == verdict_reason::parquet_type_unqualified);
  CHECK(f.check({}).reason ==
        verdict_reason::parquet_type_unqualified);  // direct pin: no exception
  f.projection.projected.clear();
  f.projection.filter = {0};
  CHECK(f.check({true}).reason == verdict_reason::parquet_type_unqualified);
  f.projection.bound_types[0] = duckdb::LogicalType::LIST(duckdb::LogicalType::INTEGER);
  CHECK(f.check({false}).reason == verdict_reason::parquet_type_unqualified);
}

TEST_CASE("Parquet D6 preserves decoded temporal units and rejects duration exports",
          "[scan][parquet][profile]")
{
  footer_fixture f;
  auto& element = f.metadata.schema[1];
  element.type  = pq::Type::INT64;
  element.logical_type.reset();
  element.converted_type      = pq::ConvertedType::TIMESTAMP_MILLIS;
  f.projection.bound_types[0] = duckdb::LogicalType::TIMESTAMP;
  CHECK(f.check({true}).reason == verdict_reason::parquet_type_unqualified);
  CHECK(f.check({false}).approved);
  element.converted_type      = pq::ConvertedType::TIME_MICROS;
  f.projection.bound_types[0] = duckdb::LogicalType::BIGINT;
  CHECK(f.check({false}).reason == verdict_reason::parquet_type_unqualified);
  element.converted_type.reset();
  element.arrow_type = cudf::type_id::DURATION_MICROSECONDS;
  CHECK(f.check({false}).reason == verdict_reason::parquet_type_unqualified);
  element.arrow_type.reset();
  element.logical_type                            = pq::LogicalType(pq::LogicalType::TIMESTAMP);
  element.logical_type->timestamp_type            = pq::TimestampType{};
  element.logical_type->timestamp_type->unit.type = pq::TimeUnit::MICROS;
  f.projection.bound_types[0]                     = duckdb::LogicalType::TIMESTAMP;
  CHECK(sirius::io::parquet_helpers::leaf_schema_type(element) ==
        duckdb::LogicalType::TIMESTAMP_TZ);
  CHECK(f.check({false}).reason == verdict_reason::parquet_type_unqualified);
  element.logical_type = pq::LogicalType(pq::LogicalType::UNDEFINED);
  CHECK(f.check({false}).reason == verdict_reason::parquet_type_unqualified);
  element.logical_type.reset();
  element.type                = pq::Type::INT96;
  f.projection.bound_types[0] = duckdb::LogicalType::TIMESTAMP_NS;
  CHECK(f.check({true}).approved);
  f.projection.bound_types[0] = duckdb::LogicalType::TIMESTAMP;
  CHECK(f.check({true}).reason == verdict_reason::parquet_type_unqualified);
  CHECK(f.check({false}).approved);
}

TEST_CASE("Parquet D6 refuses nested name order count and ARRAY child drift",
          "[scan][parquet][profile]")
{
  footer_fixture f;
  auto leaf          = f.metadata.schema[1];
  auto group         = leaf;
  group.type         = pq::Type::UNDEFINED;
  group.num_children = 2;
  group.logical_type.reset();
  group.converted_type.reset();
  leaf.name         = "a";
  auto second       = leaf;
  second.name       = "b";
  f.metadata.schema = {f.metadata.schema[0], group, leaf, second};
  for (auto& rg : f.metadata.row_groups) {
    auto column                            = rg.columns[0];
    rg.columns[0].meta_data.path_in_schema = {group.name, "a"};
    rg.columns[0].schema_idx               = 2;
    column.meta_data.path_in_schema        = {group.name, "b"};
    column.schema_idx                      = 3;
    rg.columns.push_back(column);
  }
  using T = duckdb::LogicalType;
  for (auto const& type : {T::STRUCT({{"b", T::INTEGER}, {"a", T::INTEGER}}),
                           T::STRUCT({{"a", T::INTEGER}}),
                           T::STRUCT({{"a", T::INTEGER}, {"c", T::INTEGER}}),
                           T::ARRAY(T::BIGINT, 2)}) {
    f.projection.bound_types[0] = type;
    CHECK(f.check({false}).reason == verdict_reason::parquet_type_unqualified);
  }
  f.projection.bound_types[0] = T::STRUCT({{"a", T::BIGINT}, {"b", T::INTEGER}});
  CHECK(f.check({false}).approved);
  CHECK(f.check({true}).reason == verdict_reason::parquet_type_unqualified);
}

TEST_CASE("Iceberg per-file schema comparison preserves field ids and reasons",
          "[scan][parquet][profile][iceberg_schema]")
{
  footer_fixture f;
  auto& field    = f.metadata.schema[1];
  field.field_id = 1;
  iceberg_table_schema table{{{"x", 1, "INTEGER"}}};
  auto check = [&] { return check_iceberg_file_schema(f.metadata, table, "fixture.parquet"); };
  REQUIRE(check().approved);
  field.field_id.reset();
  CHECK(check().reason == verdict_reason::iceberg_schema_no_field_ids);
  field.field_id = 2;
  CHECK(check().reason == verdict_reason::iceberg_schema_missing_field);
  field.field_id       = 1;
  table.fields[0].type = "BIGINT";
  CHECK(check().reason == verdict_reason::iceberg_schema_promoted_type);
  CHECK(check().text.find("fixture.parquet") != std::string::npos);
  table.fields[0].type = "INTEGER";
  auto extra           = field;
  extra.name           = "y";
  extra.field_id       = 2;
  f.metadata.schema.push_back(extra);
  CHECK(check().reason == verdict_reason::iceberg_schema_field_count);
  table.fields.push_back({"y", 2, "INTEGER"});
  CHECK(check().approved);
  std::swap(table.fields[0], table.fields[1]);
  CHECK(check().reason == verdict_reason::iceberg_schema_physical_order);
  table.fields.clear();
  CHECK(check().approved);
  f.metadata.schema.clear();
  table.fields.push_back({"x", 1, "INTEGER"});
  CHECK(check().reason == verdict_reason::iceberg_schema_no_rows);
}

TEST_CASE_METHOD(sirius::test::GpuExecutionFixture,
                 "Parquet constructible codec encoding corpus decodes identically",
                 "[scan][parquet][profile][integration]")
{
  std::size_t count = 0;
  for (auto const& entry : std::filesystem::directory_iterator(corpus())) {
    if (entry.path().extension() != ".parquet" ||
        entry.path().filename().string().starts_with("encrypted"))
      continue;
    INFO(entry.path().filename().string());
    compare_gpu_vs_cpu("SELECT x FROM read_parquet(" +
                       sirius::test::sql_literal(entry.path().string()) + ")");
    ++count;
  }
  CHECK(count == 62);
}

TEST_CASE_METHOD(sirius::test::GpuExecutionFixture,
                 "Parquet metadata-task physical refusals replay unchanged local files",
                 "[scan][parquet][profile][integration]")
{
  auto query = "SELECT x FROM read_parquet(" +
               sirius::test::sql_literal((corpus() / "int-SNAPPY-PLAIN.parquet").string()) + ")";
  for (auto strip : {false, true}) {
    run_ok(std::string("SET sirius_test_strip_encryption_evidence=") + (strip ? "true" : "false"));
    run_ok(std::string("SET sirius_test_synthetic_parquet_codec='") + (strip ? "" : "LZO") + "'");
    run_ok("SET gpu_execution=false");
    auto cpu = con->Query(query);
    REQUIRE_FALSE(cpu->HasError());
    run_ok("SET gpu_execution=true");
    auto before = sirius::test::get_transparent_execution_stats(*con);
    auto gpu    = con->Query(query);
    if (gpu->HasError()) INFO(gpu->GetError());
    REQUIRE_FALSE(gpu->HasError());
    CHECK(collect_rows(*gpu) == collect_rows(*cpu));
    auto after = sirius::test::get_transparent_execution_stats(*con);
    CHECK(after.runtime_fallbacks == before.runtime_fallbacks + 1);
    CHECK(after.parquet_reader_calls == before.parquet_reader_calls);
    auto reason =
      static_cast<std::size_t>(strip ? verdict_reason::parquet_encryption_evidence_missing
                                     : verdict_reason::parquet_codec_unsupported);
    CHECK(after.split_physical_rejections[reason] == before.split_physical_rejections[reason] + 1);
  }
}

TEST_CASE_METHOD(sirius::test::GpuExecutionFixture,
                 "Parquet D6 pruned first file preserves export casts and refuses semantic drift",
                 "[scan][parquet][profile][integration]")
{
  sirius::test::scratch_dir directory("parquet_d6");
  run_ok("SET gpu_execution=false");
  run_ok("COPY (SELECT 0::INTEGER keep, 0::INTEGER x) TO " + directory.file_literal("a.parquet") +
         " (FORMAT PARQUET)");
  run_ok("COPY (SELECT 1::INTEGER keep, x::DOUBLE x FROM (VALUES (1.2),(1.4),(2.0)) t(x)) TO " +
         directory.file_literal("b.parquet") + " (FORMAT PARQUET)");
  auto source = "read_parquet(" + directory.file_literal("*.parquet") + ")";
  auto before = sirius::test::get_transparent_execution_stats(*con);
  compare_gpu_vs_cpu("SELECT x FROM " + source + " WHERE keep=1");
  auto after = sirius::test::get_transparent_execution_stats(*con);
  CHECK(after.parquet_type_mismatch_observed == before.parquet_type_mismatch_observed + 1);
  CHECK(after.parquet_type_refusals == before.parquet_type_refusals);
  for (auto const& query : std::vector<std::string>{
         "SELECT x FROM " + source + " WHERE keep=1 AND x=1",
         "SELECT x, count(*) FROM " + source + " WHERE keep=1 GROUP BY x",
         "SELECT count(x) FROM " + source + " WHERE keep=1",
         "SELECT x+1 FROM " + source + " WHERE keep=1",
         "SELECT x FROM " + source + " JOIN (VALUES (1),(2)) v(y) ON x=y WHERE keep=1",
         "WITH t AS MATERIALIZED (SELECT * FROM " + source + " WHERE keep=1) SELECT x FROM t"}) {
    INFO(query);
    run_ok("SET gpu_execution=false");
    auto cpu = con->Query(query);
    REQUIRE_FALSE(cpu->HasError());
    run_ok("SET gpu_execution=true");
    before      = sirius::test::get_transparent_execution_stats(*con);
    auto result = con->Query(query);
    if (result->HasError()) INFO(result->GetError());
    REQUIRE_FALSE(result->HasError());
    CHECK(collect_rows(*result) == collect_rows(*cpu));
    after = sirius::test::get_transparent_execution_stats(*con);
    CHECK(after.runtime_fallbacks == before.runtime_fallbacks + 1);
    CHECK(after.parquet_type_refusals == before.parquet_type_refusals + 1);
    CHECK(after.parquet_reader_calls[directory.file("b.parquet")] ==
          before.parquet_reader_calls[directory.file("b.parquet")]);
  }
  run_ok("SET sirius_test_lineage_unmodelled=true");
  auto query = "SELECT x FROM " + source + " WHERE keep=1";
  run_ok("SET gpu_execution=false");
  auto cpu = con->Query(query);
  REQUIRE_FALSE(cpu->HasError());
  run_ok("SET gpu_execution=true");
  before            = sirius::test::get_transparent_execution_stats(*con);
  auto conservative = con->Query(query);
  REQUIRE_FALSE(conservative->HasError());
  CHECK(collect_rows(*conservative) == collect_rows(*cpu));
  after = sirius::test::get_transparent_execution_stats(*con);
  CHECK(after.parquet_type_refusals == before.parquet_type_refusals + 1);
  CHECK(after.runtime_fallbacks == before.runtime_fallbacks + 1);
  CHECK(after.parquet_reader_calls == before.parquet_reader_calls);
  run_ok("SET sirius_test_lineage_unmodelled=false");
}

TEST_CASE_METHOD(sirius::test::GpuExecutionFixture,
                 "Parquet nested string drift is refused before unsafe export",
                 "[scan][parquet][profile][integration]")
{
  sirius::test::scratch_dir directory("parquet_nested_d6");
  run_ok("SET gpu_execution=false");
  run_ok("COPY (SELECT 0::INTEGER keep, {'child': 0::INTEGER} x) TO " +
         directory.file_literal("a.parquet") + " (FORMAT PARQUET)");
  run_ok("COPY (SELECT 1::INTEGER keep, {'child': '12'} x) TO " +
         directory.file_literal("b.parquet") + " (FORMAT PARQUET)");
  auto query =
    "SELECT x FROM read_parquet(" + directory.file_literal("*.parquet") + ") WHERE keep=1";
  auto cpu = con->Query(query);
  REQUIRE_FALSE(cpu->HasError());
  run_ok("SET gpu_execution=true");
  auto before = sirius::test::get_transparent_execution_stats(*con);
  auto result = con->Query(query);
  if (result->HasError()) INFO(result->GetError());
  REQUIRE_FALSE(result->HasError());
  CHECK(collect_rows(*result) == collect_rows(*cpu));
  auto after = sirius::test::get_transparent_execution_stats(*con);
  CHECK(after.runtime_fallbacks == before.runtime_fallbacks + 1);
  CHECK(after.parquet_type_refusals == before.parquet_type_refusals + 1);
  CHECK(after.parquet_reader_calls[directory.file("b.parquet")] == 0);
}

TEST_CASE_METHOD(sirius::test::GpuExecutionFixture,
                 "Parquet PARE keeps metadata-task bind and pin failure stages",
                 "[scan][parquet][profile][integration]")
{
  std::ifstream key_file(corpus() / "encrypted-footer.key");
  std::string key;
  REQUIRE(static_cast<bool>(std::getline(key_file, key)));
  auto file  = sirius::test::sql_literal((corpus() / "encrypted-footer.parquet").string());
  auto query = "SELECT x FROM read_parquet(" + file +
               ", encryption_config={footer_key_value:from_base64('" + key + "')})";
  run_ok("SET gpu_execution=false");
  auto cpu = con->Query(query);
  if (cpu->HasError()) UNSCOPED_INFO(cpu->GetError());
  REQUIRE_FALSE(cpu->HasError());
  run_ok("SET gpu_execution=true");
  auto before = sirius::test::get_transparent_execution_stats(*con);
  auto result = con->Query(query);
  if (result->HasError()) UNSCOPED_INFO(result->GetError());
  REQUIRE_FALSE(result->HasError());
  CHECK(collect_rows(*result) == collect_rows(*cpu));
  auto after  = sirius::test::get_transparent_execution_stats(*con);
  auto reason = static_cast<std::size_t>(verdict_reason::parquet_encrypted);
  CHECK(after.runtime_fallbacks == before.runtime_fallbacks + 1);
  CHECK(after.split_physical_rejections[reason] == before.split_physical_rejections[reason] + 1);
  CHECK(after.parquet_reader_calls == before.parquet_reader_calls);
  for (auto const& sql : {"SELECT * FROM sirius_read_parquet(" + file + ")",
                          "CALL pin_table(" + file + ", name='r2a_encrypted_pin', tier='gpu')"}) {
    before     = sirius::test::get_transparent_execution_stats(*con);
    auto error = con->Query(sql);
    REQUIRE(error->HasError());
    CHECK(error->GetError().find("Encrypted Parquet footer") != std::string::npos);
    after = sirius::test::get_transparent_execution_stats(*con);
    CHECK(after.runtime_fallbacks == before.runtime_fallbacks);
    CHECK(after.split_physical_rejections == before.split_physical_rejections);
  }
}

TEST_CASE_METHOD(sirius::test::GpuExecutionFixture,
                 "Parquet column crypto survives both store producers and cache reuse",
                 "[scan][parquet][profile][integration]")
{
  sirius::test::scratch_dir directory("parquet_crypto_cache");
  auto context  = sirius::test::get_registered_sirius_context(*con);
  auto& manager = context->get_scan_manager();
  for (bool describe_first : {false, true}) {
    auto path = directory.file(describe_first ? "describe.parquet" : "scan.parquet");
    std::filesystem::copy_file(corpus() / "encrypted-columns.parquet", path);
    if (describe_first) {
      auto description = manager.describe_parquet(path);
      REQUIRE(description.names.size() == 1);
    }
    auto query    = "SELECT x FROM read_parquet(" + sirius::test::sql_literal(path) + ")";
    auto prepared = con->Prepare(query);
    REQUIRE_FALSE(prepared->HasError());  // bind succeeds; encrypted pages cannot be read
    auto before = sirius::test::get_transparent_execution_stats(*con);
    auto result = prepared->Execute();
    REQUIRE(result->HasError());  // CPU replay cannot supply column keys either
    auto after  = sirius::test::get_transparent_execution_stats(*con);
    auto reason = static_cast<std::size_t>(verdict_reason::parquet_encrypted);
    CHECK(after.runtime_fallbacks == before.runtime_fallbacks + 1);
    CHECK(after.split_physical_rejections[reason] == before.split_physical_rejections[reason] + 1);
    CHECK(after.parquet_reader_calls == before.parquet_reader_calls);
    auto source = manager.create_datasource(path);
    REQUIRE(source);
    auto stored = std::dynamic_pointer_cast<parquet_metadata>(source->metadata());
    REQUIRE(stored);
    CHECK(stored->encryption_evidence.columns_encrypted);
    before = after;
    result = con->Query(query);
    REQUIRE(result->HasError());
    after = sirius::test::get_transparent_execution_stats(*con);
    CHECK(after.split_physical_rejections[reason] == before.split_physical_rejections[reason] + 1);
    CHECK(after.parquet_reader_calls == before.parquet_reader_calls);
  }
}

TEST_CASE_METHOD(sirius::test::GpuExecutionFixture,
                 "Parquet first split decodes while a later footer is held",
                 "[scan][parquet][profile][integration]")
{
  sirius::test::scratch_dir directory("parquet_pipeline_profile");
  run_ok("SET gpu_execution=false");
  for (auto const* name : {"a.parquet", "b.parquet"})
    run_ok("COPY (SELECT i::INTEGER x FROM range(4096) t(i)) TO " + directory.file_literal(name) +
           " (FORMAT PARQUET, ROW_GROUP_SIZE 2048)");
  auto batch_setting = con->Query("SELECT current_setting('scan_task_batch_size')");
  REQUIRE_FALSE(batch_setting->HasError());
  auto previous_batch = batch_setting->GetValue(0, 0).ToString();
  run_ok("SET scan_task_batch_size=1");
  run_ok("SET gpu_execution=true");
  auto state = con->context->registered_state->Get<duckdb::SiriusContext>("sirius_state");
  REQUIRE(state);
  auto counters = state->physical_counters();
  struct phase_guard {
    std::shared_ptr<physical_check_counters> counters;
    ~phase_guard() { counters->parquet_phase_for_testing = {}; }
  } guard{counters};
  std::mutex mutex;
  std::condition_variable changed;
  bool held = false, decoded = false, observed = false;
  counters->parquet_phase_for_testing = [&](std::string const& file, bool footer) {
    std::unique_lock lock(mutex);
    if (footer && file == directory.file("b.parquet")) {
      held = true;
      changed.notify_all();
      if (!changed.wait_for(lock, std::chrono::seconds(20), [&] { return decoded; }))
        throw std::runtime_error("first Parquet split did not decode before later footer");
    } else if (!footer && file == directory.file("a.parquet")) {
      if (!changed.wait_for(lock, std::chrono::seconds(20), [&] { return held; }))
        throw std::runtime_error("later footer was not held");
      observed = true;
      decoded  = true;
      changed.notify_all();
    }
  };
  auto before = sirius::test::get_transparent_execution_stats(*con);
  auto result =
    con->Query("SELECT sum(x) FROM read_parquet(" + directory.file_literal("*.parquet") + ")");
  REQUIRE_FALSE(result->HasError());
  REQUIRE(result->RowCount() == 1);
  CHECK(result->GetValue(0, 0).ToString() == "16773120");
  auto after = sirius::test::get_transparent_execution_stats(*con);
  CHECK(after.runtime_fallbacks == before.runtime_fallbacks);
  CHECK(observed);
  run_ok("SET scan_task_batch_size=" + previous_batch);
}

TEST_CASE_METHOD(sirius::test::GpuExecutionFixture,
                 "Parquet metadata producer ignores unsupported codecs in pruned row groups",
                 "[scan][parquet][profile][integration]")
{
  sirius::test::scratch_dir directory("parquet_pruned_codec");
  run_ok("SET gpu_execution=false");
  run_ok("COPY (SELECT i::INTEGER x FROM range(4096) t(i)) TO " +
         directory.file_literal("two_groups.parquet") + " (FORMAT PARQUET, ROW_GROUP_SIZE 2048)");
  run_ok("SET sirius_test_synthetic_parquet_codec='LZO:first_row_group'");
  auto before = sirius::test::get_transparent_execution_stats(*con);
  compare_gpu_vs_cpu("SELECT x FROM read_parquet(" + directory.file_literal("two_groups.parquet") +
                     ") WHERE x>=2048");
  auto after = sirius::test::get_transparent_execution_stats(*con);
  CHECK(after.split_physical_rejections == before.split_physical_rejections);
  CHECK(after.parquet_reader_calls[directory.file("two_groups.parquet")] ==
        before.parquet_reader_calls[directory.file("two_groups.parquet")] + 1);
}
