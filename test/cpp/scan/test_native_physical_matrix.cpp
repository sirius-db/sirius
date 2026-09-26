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

// R2a T4: RED_FEATURE after commit (d); producer subset required by commit (b).
#include <catch.hpp>
#include <duckdb/catalog/catalog.hpp>
#include <duckdb/catalog/catalog_entry/duck_table_entry.hpp>
#include <duckdb/storage/data_table.hpp>
#include <duckdb/storage/storage_manager.hpp>
#include <helper/type_conversions.hpp>
#include <op/scan/duckdb_native_metadata.hpp>
#include <scan_manager/sirius_scan_manager.hpp>
#include <utils/gpu_execution_fixture.hpp>
#include <utils/parquet_fixture_utils.hpp>

#include <fstream>
#include <set>
#include <sstream>

using namespace sirius::op::scan;

TEST_CASE("Native matrix rejects codecs that reach a different decoder family",
          "[scan][native][matrix]")
{
  using T = duckdb::LogicalTypeId;
  using C = duckdb::CompressionType;
  for (auto type :
       {T::BOOLEAN, T::INTEGER, T::BIGINT, T::DOUBLE, T::DATE, T::TIMESTAMP, T::DECIMAL}) {
    for (auto codec : {C::COMPRESSION_DICTIONARY,
                       C::COMPRESSION_FSST,
                       C::COMPRESSION_DICT_FSST,
                       C::COMPRESSION_ZSTD,
                       C::COMPRESSION_ROARING})
      CHECK_FALSE(native_matrix_supports(type, codec, C::COMPRESSION_CONSTANT));
    CHECK(native_matrix_supports(type, C::COMPRESSION_UNCOMPRESSED, C::COMPRESSION_ROARING));
    CHECK_FALSE(native_matrix_supports(type, C::COMPRESSION_CONSTANT, C::COMPRESSION_FSST));
  }
  CHECK_FALSE(native_matrix_supports(T::VARCHAR, C::COMPRESSION_CONSTANT, C::COMPRESSION_EMPTY));
  CHECK(native_matrix_supports(T::VARCHAR, C::COMPRESSION_DICT_FSST, C::COMPRESSION_EMPTY));
  CHECK_FALSE(native_matrix_supports(T::INTEGER, C::COMPRESSION_ALP, C::COMPRESSION_EMPTY));
  CHECK(native_matrix_supports(T::DOUBLE, C::COMPRESSION_ALP, C::COMPRESSION_EMPTY));
}

TEST_CASE_METHOD(sirius::test::GpuExecutionFixture,
                 "Native synthetic matrix negatives refuse in the range walk",
                 "[scan][native][matrix][integration]")
{
  run_ok("CREATE TABLE native_matrix_negative AS SELECT i::INTEGER x FROM range(4096) t(i)");
  run_ok("CHECKPOINT");
  run_ok("SET gpu_execution=true");
  for (auto codec : {"dictionary", "fsst", "dict_fsst", "outside"}) {
    INFO(codec);
    run_ok(std::string("SET sirius_test_synthetic_native_segment='") + codec + "'");
    auto before = sirius::test::get_transparent_execution_stats(*con);
    auto result = con->Query("SELECT sum(x) FROM native_matrix_negative");
    if (result->HasError()) INFO(result->GetError());
    REQUIRE_FALSE(result->HasError());
    CHECK(result->GetValue(0, 0).ToString() == "8386560");
    auto after = sirius::test::get_transparent_execution_stats(*con);
    CHECK(after.runtime_fallbacks == before.runtime_fallbacks + 1);
    CHECK(after.native_decoder_calls == before.native_decoder_calls);
    auto reason = static_cast<std::size_t>(verdict_reason::native_segment_codec);
    CHECK(after.split_physical_rejections[reason] == before.split_physical_rejections[reason] + 1);
  }
}

TEST_CASE_METHOD(sirius::test::GpuExecutionFixture,
                 "Native constructible storage versions record actual compression and decode",
                 "[scan][native][matrix][integration]")
{
  sirius::test::scratch_dir directory("native_matrix");
  struct cell {
    std::string type, expression;
  };
  std::vector<cell> cells = {{"BOOLEAN", "i%3=0"},
                             {"INTEGER", "i%997"},
                             {"BIGINT", "i*100000001"},
                             {"DOUBLE", "i/4.0"},
                             {"DATE", "DATE '2020-01-01' + (i%31)::INTEGER"},
                             {"TIMESTAMP", "TIMESTAMP '2020-01-01' + i * INTERVAL '1 SECOND'"},
                             {"DECIMAL(18,3)", "i/8.0"},
                             {"VARCHAR", "'prefix-' || (i%997)::VARCHAR"},
                             {"INTEGER[3]", "[CASE WHEN i%7=0 THEN NULL ELSE i%97 END,i%73,i%31]"}};
  std::multiset<std::string> observed;
  std::ofstream manifest;
  if (auto path = std::getenv("SIRIUS_NATIVE_MATRIX_MANIFEST")) {
    manifest.open(path);
    manifest << "version,logical_type,requested,segment_type,compression,rows\n";
  }
  for (auto version : {"v0.10.2", "v1.2.0", "v1.5.0"}) {
    auto file = directory.file_literal(std::string(version) + ".duckdb");
    run_ok("ATTACH " + file + " AS matrix_db (STORAGE_VERSION '" + version + "')");
    run_ok("USE matrix_db");
    run_ok("BEGIN TRANSACTION");
    auto const storage_version = duckdb::Catalog::GetCatalog(*con->context, "matrix_db")
                                   .GetAttached()
                                   .GetStorageManager()
                                   .GetStorageVersion();
    CHECK(storage_version == (std::string(version) == "v0.10.2"  ? 1
                              : std::string(version) == "v1.2.0" ? 4
                                                                 : 7));
    run_ok("COMMIT");
    for (auto const& cell : cells) {
      auto codecs =
        cell.type == "VARCHAR"
          ? std::vector<std::string>{"uncompressed", "dictionary", "fsst", "dict_fsst", "zstd"}
          : std::vector<std::string>{
              "uncompressed", "rle", "bitpacking", "alp", "alprd", "roaring"};
      for (auto const& codec : codecs) {
        INFO(version << " " << cell.type << " " << codec);
        run_ok("SET gpu_execution=false");
        run_ok("SET force_compression='" + codec + "'");
        auto expression = "(" + cell.expression + ")::" + cell.type;
        if (cell.type != "INTEGER[3]")
          expression = "CASE WHEN i%7=0 THEN NULL ELSE " + expression + " END";
        run_ok("CREATE TABLE matrix_t AS SELECT " + expression + " AS x FROM range(8192) t(i)");
        run_ok("CHECKPOINT matrix_db");
        auto actual = con->Query(
          "SELECT segment_type, compression, count FROM pragma_storage_info('matrix_t')");
        REQUIRE_FALSE(actual->HasError());
        bool negative = false;
        for (duckdb::idx_t row = 0; row < actual->RowCount(); ++row) {
          auto segment     = actual->GetValue(0, row).ToString();
          auto compression = actual->GetValue(1, row).ToString();
          std::ostringstream observation;
          observation << version << ',' << '"' << cell.type << '"' << ',' << codec << ',' << segment
                      << ',' << compression << ',' << actual->GetValue(2, row).ToString();
          observed.insert(observation.str());
          if (manifest) manifest << observation.str() << '\n';
          negative |= (segment == "VARCHAR" && compression == "ZSTD") ||
                      (segment == "BOOLEAN" && compression == "Roaring");
        }
        if (negative) {
          auto cpu = con->Query("SELECT x FROM matrix_t");
          REQUIRE_FALSE(cpu->HasError());
          run_ok("SET gpu_execution=true");
          auto before = sirius::test::get_transparent_execution_stats(*con);
          auto result = con->Query("SELECT x FROM matrix_t");
          REQUIRE_FALSE(result->HasError());
          CHECK(collect_rows(*result) == collect_rows(*cpu));
          auto after = sirius::test::get_transparent_execution_stats(*con);
          CHECK(after.runtime_fallbacks == before.runtime_fallbacks + 1);
          CHECK(after.native_decoder_calls == before.native_decoder_calls);
          auto reason = static_cast<std::size_t>(verdict_reason::native_segment_codec);
          CHECK(after.split_physical_rejections[reason] ==
                before.split_physical_rejections[reason] + 1);
        } else {
          compare_gpu_vs_cpu("SELECT x FROM matrix_t");
        }
        run_ok("DROP TABLE matrix_t");
      }
    }
    run_ok("USE " + attach_alias);
    run_ok("DETACH matrix_db");
  }
  run_ok("SET force_compression='auto'");
  std::ifstream expected_file(std::filesystem::path(SIRIUS_PROJECT_ROOT) /
                              "test/cpp/scan/data/native_physical_matrix.csv");
  REQUIRE(expected_file.good());
  std::string line;
  std::getline(expected_file, line);  // header
  std::multiset<std::string> expected;
  while (std::getline(expected_file, line))
    expected.insert(line);
  CHECK(observed == expected);
}

TEST_CASE_METHOD(sirius::test::GpuExecutionFixture,
                 "Native pin entries retain identity and layout evidence across queries",
                 "[scan][native][matrix][integration]")
{
  run_ok("CREATE TABLE matrix_pin AS SELECT i::INTEGER x FROM range(128) t(i)");
  run_ok("CHECKPOINT");
  auto state = sirius::test::get_registered_sirius_context(*con);
  for (auto tier : {"gpu", "host"}) {
    run_ok(std::string("CALL pin_table(format='duckdb', name='matrix_pin', tier='") + tier + "')");
    auto check = [&] {
      bool found = false;
      state->get_scan_manager().visit_pinned_entries([&](auto name, auto const& entry) {
        if (name == "matrix_pin") {
          found = true;
          CHECK(entry.identity_evidence.applies);
          CHECK(entry.identity_evidence.passed);
          CHECK(entry.layout_evidence.applies);
          CHECK(entry.layout_evidence.passed);
        }
        return true;
      });
      REQUIRE(found);
    };
    check();
    compare_gpu_vs_cpu("SELECT x FROM matrix_pin");
    check();
    run_ok("INSERT INTO matrix_pin VALUES (999)");
    compare_gpu_vs_cpu("SELECT x FROM matrix_pin");
    check();
    run_ok("CALL unpin_table('matrix_pin')");
    run_ok("CHECKPOINT");
  }
}

TEST_CASE_METHOD(sirius::test::GpuExecutionFixture,
                 "Pinned query rejects stale resident evidence before publication",
                 "[scan][native][matrix][certificate][consumption][integration]")
{
  run_ok("CREATE TABLE admission_pin AS SELECT i::INTEGER x FROM range(128) t(i)");
  run_ok("CHECKPOINT");
  run_ok("CALL pin_table(format='duckdb', name='admission_pin', tier='host')");
  compare_gpu_vs_cpu("SELECT x FROM admission_pin");
  run_ok("SET sirius_test_invalidate_pin_witness=true");
  auto before = sirius::test::get_transparent_execution_stats(*con);
  auto result = con->Query("SELECT sum(x) FROM admission_pin");
  REQUIRE_FALSE(result->HasError());
  CHECK(result->GetValue(0, 0).ToString() == "8128");
  auto after = sirius::test::get_transparent_execution_stats(*con);
  CHECK(after.runtime_fallbacks == before.runtime_fallbacks + 1);
  CHECK(after.certificate_incompletes == before.certificate_incompletes + 1);
  CHECK(after.certificate_mismatches == before.certificate_mismatches);
  run_ok("SET enable_duckdb_fallback=false");
  before = sirius::test::get_transparent_execution_stats(*con);
  result = con->Query("SELECT sum(x) FROM admission_pin");
  REQUIRE(result->HasError());
  CHECK(result->GetError().find("resident certificate incomplete: query token mismatch") !=
        std::string::npos);
  after = sirius::test::get_transparent_execution_stats(*con);
  CHECK(after.runtime_fallbacks == before.runtime_fallbacks);
  CHECK(after.certificate_incompletes == before.certificate_incompletes + 1);
  run_ok("SET enable_duckdb_fallback=true");
  run_ok("SET sirius_test_invalidate_pin_witness=false");
  compare_gpu_vs_cpu("SELECT x FROM admission_pin");
  run_ok("CALL unpin_table('admission_pin')");
}
