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

// GPU-vs-CPU correctness for NULL materialization through the parquet reader.
//
// The parquet reader encodes nullability via definition levels, not a simple
// validity bitmap. Each nullable encoding path needs its own exercise:
//
//   Fixture                       | Encoding path
//   ------------------------------|--------------------------------------------
//   ParquetNullFixture            | optional columns, wholly-null AND
//                                 | partially-null, one of each per physical
//                                 | encoding: INT32/INT64/DOUBLE/FLOAT/BOOLEAN/
//                                 | BYTE_ARRAY, DATE (INT32), TIMESTAMP (INT64),
//                                 | and DECIMAL at both INT32 and INT64 widths
//   ParquetDictNullFixture        | dictionary-encoded column with null entries
//   ParquetMultiRowGroupFixture   | nulls spanning row groups, ragged tail
//   ParquetDenseColFixture        | dense zero-NULL column baseline
//   ParquetNullRunFixture         | constant-valid run then all-NULL run; long
//                                 | NULL prefix then valid suffix
//   ParquetFlbaDecimalNullFixture | FIXED_LEN_BYTE_ARRAY decimal with nulls
//                                 | (reader-side filter pushdown disabled)
//
// The three parquet decimal widths are all covered: precision <= 9 -> INT32,
// <= 18 -> INT64, > 18 -> FIXED_LEN_BYTE_ARRAY. Precision <= 4 is deliberately
// avoided -- DuckDB stores it as INT16 and Sirius throws for that case
// (cudf_utils.hpp get_cudf_type), forcing a CPU fallback that would make
// compare_gpu_vs_cpu fail for reasons unrelated to NULLs.
//
// Fixtures that depend on a specific on-disk encoding assert it via
// parquet_schema() / parquet_metadata() at construction time, so the tests fail
// loudly rather than silently exercising a different decode path if DuckDB's
// writer choices change.
//
// Every query goes through compare_gpu_vs_cpu: run on GPU (asserting no
// fallback), run on DuckDB CPU, compare.

//
// KNOWN GAPS -- both need a fixture from a non-DuckDB writer (e.g. pyarrow),
// because DuckDB's COPY cannot produce either shape:
//
//   INT96 timestamps. parquet_helpers.cpp maps the deprecated INT96 physical
//   type to TIMESTAMP, but DuckDB only ever writes the INT64 logical form,
//   which is what is covered here.
//
//   REQUIRED (non-nullable) columns. ParquetWriter defaults can_have_nulls to
//   true for every top-level column, so a NOT NULL table constraint does not
//   carry through and everything is written OPTIONAL. Reading a column with no
//   definition-level stream is therefore untested; ParquetDenseColFixture
//   covers the nearest reachable shape (OPTIONAL with zero NULLs).

#include <catch.hpp>
#include <duckdb.hpp>
#include <unistd.h>
#include <utils/gpu_execution_fixture.hpp>

#include <atomic>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <system_error>
#include <vector>

namespace fs = std::filesystem;

namespace {

// ---------------------------------------------------------------------------
// Helper: generate a parquet file with a plain DuckDB (Sirius disabled).
// ---------------------------------------------------------------------------

/// Sets SIRIUS_DISABLE=1 for the duration of a scope, then restores whatever
/// was there before -- including the harness's own "1".
///
/// A bare setenv/unsetenv pair is wrong twice over: the harness deliberately
/// keeps SIRIUS_DISABLE=1 so untagged tests' DuckDB instances do not
/// auto-initialize a SiriusContext (see scoped_sirius_disable_clear in
/// test_plan_printer.cpp), so unsetting it leaks a changed global into later
/// tests; and the REQUIREs below throw on failure, which would skip the unset
/// entirely. Restoring in a destructor fixes both.
struct scoped_sirius_disable_set {
  scoped_sirius_disable_set()
  {
    if (char const* val = ::getenv("SIRIUS_DISABLE")) { _saved = val; }
    ::setenv("SIRIUS_DISABLE", "1", 1);
  }
  ~scoped_sirius_disable_set()
  {
    if (_saved) {
      ::setenv("SIRIUS_DISABLE", _saved->c_str(), 1);
    } else {
      ::unsetenv("SIRIUS_DISABLE");
    }
  }
  scoped_sirius_disable_set(scoped_sirius_disable_set const&)            = delete;
  scoped_sirius_disable_set& operator=(scoped_sirius_disable_set const&) = delete;

 private:
  std::optional<std::string> _saved;
};

struct ParquetFileGuard {
  fs::path dir;

  /// Creates the scratch directory and ONE Sirius-disabled DuckDB for the
  /// fixture's lifetime.
  ///
  /// Catch2 rebuilds a fixture for every TEST_CASE_METHOD that uses it, and the
  /// constructors here run ~20 metadata assertions each. Opening a database per
  /// assertion meant several hundred in-memory DuckDB initialisations for a
  /// single fixture; one connection, reused, makes it one per construction.
  ///
  /// SIRIUS_DISABLE only matters while the instance is being created — that is
  /// when the extension callback would attach a SiriusContext — so the guard
  /// does not need to outlive the constructor.
  explicit ParquetFileGuard(std::string tag)
  {
    static std::atomic<unsigned> ctr{0};
    dir = fs::temp_directory_path() / ("sirius_pq_nulls_" + tag + "_" + std::to_string(::getpid()) +
                                       "_" + std::to_string(ctr.fetch_add(1)));
    // PID + a process-local counter repeats once PIDs are reused (PID 1 in a
    // container makes that likely), and create_directories accepts an existing
    // directory — leaving a previous run's parquet files in place to be read
    // instead of the ones about to be written. Clear it first.
    std::error_code ec;
    fs::remove_all(dir, ec);
    fs::create_directories(dir);

    scoped_sirius_disable_set disable_guard;
    db_  = std::make_unique<duckdb::DuckDB>(nullptr);
    con_ = std::make_unique<duckdb::Connection>(*db_);
  }
  ~ParquetFileGuard()
  {
    // Best-effort: a destructor is noexcept, so a leftover temp dir must not
    // become a std::terminate.
    std::error_code ec;
    fs::remove_all(dir, ec);
  }

  // Run one or more SQL statements in a SIRIUS_DISABLE=1 DuckDB connection.
  // Connection::Query() only executes the first statement in a string, so
  // callers pass a vector when they need sequential execution (e.g. CREATE
  // TABLE + INSERT + COPY).
  void write(const std::vector<std::string>& stmts) const
  {
    for (auto const& sql : stmts) {
      auto r = con_->Query(sql);
      REQUIRE(r);
      if (r->HasError()) { UNSCOPED_INFO("parquet generation error: " << r->GetError()); }
      REQUIRE_FALSE(r->HasError());
    }
  }

  // Convenience overload for a single statement.
  void write(const std::string& stmt) const { write(std::vector<std::string>{stmt}); }

  std::string path(const std::string& name) const { return (dir / name).string(); }

  /// A single-quoted SQL literal with embedded quotes doubled. TMPDIR may
  /// legally contain one, which would otherwise terminate the literal early and
  /// break every generated statement.
  static std::string sql_literal(std::string const& value)
  {
    std::string out = "'";
    for (char const c : value) {
      if (c == '\'') { out.push_back('\''); }
      out.push_back(c);
    }
    out.push_back('\'');
    return out;
  }

  // Return a read_parquet(...) SQL expression for a single file.
  std::string scan(const std::string& name) const
  {
    return "read_parquet(" + sql_literal(path(name)) + ")";
  }

  // Assert one leaf column's repetition_type ('REQUIRED' / 'OPTIONAL').
  //
  // Leaves are selected with `num_children IS NULL`. Filtering by name would be
  // wrong: DuckDB names the schema root 'duckdb_schema' (not 'schema'), and it
  // sets the root's repetition_type to 'REQUIRED', so a name-based filter both
  // fails to exclude the root and lets it trivially satisfy a REQUIRED check.
  void assert_repetition_type(const std::string& pq_path,
                              const std::string& col_name,
                              const std::string& expected) const
  {
    auto r = con_->Query("SELECT repetition_type FROM parquet_schema(" + sql_literal(pq_path) +
                         ") WHERE name = '" + col_name + "' AND num_children IS NULL LIMIT 1");
    REQUIRE(r);
    REQUIRE_FALSE(r->HasError());
    if (r->RowCount() == 0) {
      UNSCOPED_INFO("parquet_schema found no leaf column '" << col_name << "'");
      REQUIRE(r->RowCount() > 0);
    }
    auto actual = r->GetValue(0, 0).ToString();
    if (actual != expected) {
      UNSCOPED_INFO("column '" << col_name << "' repetition_type is '" << actual << "', expected '"
                               << expected << "'");
      REQUIRE(actual == expected);
    }
  }

  // Assert that the named column was written with a specific encoding (checked
  // as a substring of the encodings list cast to VARCHAR, e.g. "DICTIONARY").
  void assert_column_encoding(const std::string& pq_path,
                              const std::string& col_name,
                              const std::string& encoding_substr) const
  {
    auto r =
      con_->Query("SELECT CAST(encodings AS VARCHAR) FROM parquet_metadata(" +
                  sql_literal(pq_path) + ") WHERE path_in_schema = '" + col_name + "' LIMIT 1");
    REQUIRE(r);
    REQUIRE_FALSE(r->HasError());
    if (r->RowCount() == 0) {
      UNSCOPED_INFO("parquet_metadata returned no rows for column '" << col_name << "'");
      REQUIRE(r->RowCount() > 0);
    }
    auto encodings_str = r->GetValue(0, 0).ToString();
    if (encodings_str.find(encoding_substr) == std::string::npos) {
      UNSCOPED_INFO("column '" << col_name << "' encodings '" << encodings_str
                               << "' do not contain '" << encoding_substr << "'");
      REQUIRE(encodings_str.find(encoding_substr) != std::string::npos);
    }
  }

  // Assert the file contains at least @p expected row groups.
  //
  // DuckDB's ROW_GROUP_SIZE is NOT a hard cut: ParquetWriteSink appends whole
  // DataChunks (up to STANDARD_VECTOR_SIZE = 2048 rows) to a buffer and only
  // then checks `buffer.Count() >= row_group_size`, and Flush writes the entire
  // buffer as ONE row group. So the effective minimum row group is the arriving
  // chunk size (2048) regardless of how small ROW_GROUP_SIZE is set -- asking
  // for 50 on a 203-row table yields a single 203-row row group. Multi-row-group
  // fixtures must therefore write well over 2048 rows AND assert the result.
  void assert_min_row_groups(const std::string& pq_path, std::size_t expected) const
  {
    auto r = con_->Query("SELECT COUNT(DISTINCT row_group_id) FROM parquet_metadata(" +
                         sql_literal(pq_path) + ")");
    REQUIRE(r);
    REQUIRE_FALSE(r->HasError());
    REQUIRE(r->RowCount() == 1);
    auto const actual = r->GetValue(0, 0).GetValue<std::int64_t>();
    if (actual < static_cast<std::int64_t>(expected)) {
      UNSCOPED_INFO("file has " << actual << " row group(s), expected at least " << expected);
      REQUIRE(actual >= static_cast<std::int64_t>(expected));
    }
  }

  // Assert the NULL population a column actually carries, independently of any
  // query result.
  //
  // Every result assertion in this file is a GPU-vs-CPU comparison of the SAME
  // file, so a generation bug that produced the wrong nulls would leave both
  // sides agreeing and the tests green while proving nothing about NULL
  // handling. These read the file's own statistics and its decoded contents and
  // compare them against numbers derived by hand from the fixture SQL.
  void assert_null_population(const std::string& pq_path,
                              const std::string& col_name,
                              std::int64_t expected_nulls,
                              std::int64_t expected_rows) const
  {
    // 1. The file's own per-row-group statistics, summed.
    auto stats =
      con_->Query("SELECT SUM(stats_null_count), SUM(num_values) FROM parquet_metadata(" +
                  sql_literal(pq_path) + ") WHERE path_in_schema = " + sql_literal(col_name));
    REQUIRE(stats);
    REQUIRE_FALSE(stats->HasError());
    REQUIRE(stats->RowCount() == 1);
    if (!stats->GetValue(0, 0).IsNull()) {
      auto const null_count = stats->GetValue(0, 0).GetValue<std::int64_t>();
      if (null_count != expected_nulls) {
        UNSCOPED_INFO("column '" << col_name << "' stats null_count is " << null_count
                                 << ", expected " << expected_nulls);
        REQUIRE(null_count == expected_nulls);
      }
    }
    auto const num_values = stats->GetValue(1, 0).GetValue<std::int64_t>();
    REQUIRE(num_values == expected_rows);

    // 2. The decoded contents, in case the statistics themselves are wrong.
    auto rows = con_->Query("SELECT COUNT(*), COUNT(" + col_name + ") FROM read_parquet(" +
                            sql_literal(pq_path) + ")");
    REQUIRE(rows);
    REQUIRE_FALSE(rows->HasError());
    auto const total    = rows->GetValue(0, 0).GetValue<std::int64_t>();
    auto const non_null = rows->GetValue(1, 0).GetValue<std::int64_t>();
    if (total != expected_rows || non_null != expected_rows - expected_nulls) {
      UNSCOPED_INFO("column '" << col_name << "': " << total << " rows, " << non_null
                               << " non-NULL; expected " << expected_rows << " and "
                               << (expected_rows - expected_nulls));
      REQUIRE(total == expected_rows);
      REQUIRE(non_null == expected_rows - expected_nulls);
    }
  }

  // Assert a column's parquet logical/converted annotation, and for decimals its
  // precision and scale.
  //
  // A physical type alone is ambiguous: INT32 covers INTEGER, DATE and a narrow
  // DECIMAL; INT64 covers BIGINT, TIMESTAMP and a wider one. It also matters for
  // FLBA specifically, whose pushdown-disable branch in parquet_gpu_ingestible
  // keys off the DECIMAL annotation rather than the physical type.
  void assert_logical_type(const std::string& pq_path,
                           const std::string& col_name,
                           const std::string& expected_converted,
                           std::optional<int> precision = std::nullopt,
                           std::optional<int> scale     = std::nullopt) const
  {
    auto r = con_->Query("SELECT converted_type, precision, scale FROM parquet_schema(" +
                         sql_literal(pq_path) + ") WHERE name = '" + col_name +
                         "' AND num_children IS NULL LIMIT 1");
    REQUIRE(r);
    REQUIRE_FALSE(r->HasError());
    REQUIRE(r->RowCount() == 1);

    auto const converted = r->GetValue(0, 0).ToString();
    if (converted != expected_converted) {
      UNSCOPED_INFO("column '" << col_name << "' converted_type is '" << converted
                               << "', expected '" << expected_converted << "'");
      REQUIRE(converted == expected_converted);
    }
    if (precision) {
      auto const actual = r->GetValue(1, 0).GetValue<int>();
      if (actual != *precision) {
        UNSCOPED_INFO("column '" << col_name << "' precision " << actual << ", expected "
                                 << *precision);
        REQUIRE(actual == *precision);
      }
    }
    if (scale) {
      auto const actual = r->GetValue(2, 0).GetValue<int>();
      if (actual != *scale) {
        UNSCOPED_INFO("column '" << col_name << "' scale " << actual << ", expected " << *scale);
        REQUIRE(actual == *scale);
      }
    }
  }

  // Assert a column's NULL count per row group, in order.
  //
  // Stronger than the totals: it pins WHICH rows landed in which group, so a
  // reordered write cannot pass. assert_null_population and
  // assert_row_group_sizes both check aggregates and would not notice.
  void assert_null_counts_per_row_group(const std::string& pq_path,
                                        const std::string& col_name,
                                        std::vector<std::int64_t> const& expected) const
  {
    auto r = con_->Query("SELECT stats_null_count FROM parquet_metadata(" + sql_literal(pq_path) +
                         ") WHERE path_in_schema = '" + col_name + "' ORDER BY row_group_id");
    REQUIRE(r);
    REQUIRE_FALSE(r->HasError());
    if (r->RowCount() != expected.size()) {
      UNSCOPED_INFO("column '" << col_name << "' spans " << r->RowCount()
                               << " row group(s), expected " << expected.size());
      REQUIRE(r->RowCount() == expected.size());
    }
    for (duckdb::idx_t i = 0; i < r->RowCount(); i++) {
      // Absent statistics are legal in the parquet spec, but DuckDB always
      // writes them and this assertion exists to pin the row-to-group mapping.
      // Skipping on absence would quietly turn the whole check into a no-op.
      if (r->GetValue(0, i).IsNull()) {
        UNSCOPED_INFO("column '" << col_name << "' row group " << i
                                 << " has no null_count statistic, so the layout cannot be "
                                    "verified");
        REQUIRE_FALSE(r->GetValue(0, i).IsNull());
      }
      auto const actual = r->GetValue(0, i).GetValue<std::int64_t>();
      if (actual != expected[i]) {
        UNSCOPED_INFO("column '" << col_name << "' row group " << i << " has " << actual
                                 << " NULLs, expected " << expected[i]);
        REQUIRE(actual == expected[i]);
      }
    }
  }

  // Assert the exact per-row-group row counts, in order.
  //
  // A minimum count does not establish a layout: it cannot tell four full groups
  // plus a ragged tail from two uneven ones, so a test that targets the tail by
  // row id would be aiming at the wrong rows.
  void assert_row_group_sizes(const std::string& pq_path,
                              std::vector<std::int64_t> const& expected) const
  {
    auto r = con_->Query("SELECT row_group_num_rows FROM parquet_metadata(" + sql_literal(pq_path) +
                         ") GROUP BY row_group_id, row_group_num_rows ORDER BY row_group_id");
    REQUIRE(r);
    REQUIRE_FALSE(r->HasError());
    if (r->RowCount() != expected.size()) {
      UNSCOPED_INFO("file has " << r->RowCount() << " row group(s), expected " << expected.size());
      REQUIRE(r->RowCount() == expected.size());
    }
    for (duckdb::idx_t i = 0; i < r->RowCount(); i++) {
      auto const actual = r->GetValue(0, i).GetValue<std::int64_t>();
      if (actual != expected[i]) {
        UNSCOPED_INFO("row group " << i << " has " << actual << " rows, expected " << expected[i]);
        REQUIRE(actual == expected[i]);
      }
    }
  }

  // Assert the on-disk parquet physical type of a column (e.g.
  // 'FIXED_LEN_BYTE_ARRAY', 'INT64', 'BOOLEAN'). DuckDB's decimal physical type
  // is precision-dependent (INT32 / INT64 / FLBA), so tests that mean to
  // exercise a specific decode path must pin it rather than assume it.
  void assert_physical_type(const std::string& pq_path,
                            const std::string& col_name,
                            const std::string& expected_type) const
  {
    auto r = con_->Query("SELECT type FROM parquet_schema(" + sql_literal(pq_path) +
                         ") WHERE name = '" + col_name + "' LIMIT 1");
    REQUIRE(r);
    REQUIRE_FALSE(r->HasError());
    if (r->RowCount() == 0) {
      UNSCOPED_INFO("parquet_schema returned no rows for column '" << col_name << "'");
      REQUIRE(r->RowCount() > 0);
    }
    auto actual = r->GetValue(0, 0).ToString();
    if (actual != expected_type) {
      UNSCOPED_INFO("column '" << col_name << "' physical type is '" << actual << "', expected '"
                               << expected_type << "'");
      REQUIRE(actual == expected_type);
    }
  }

 private:
  // Reused by every helper; see the constructor.
  std::unique_ptr<duckdb::DuckDB> db_;
  std::unique_ptr<duckdb::Connection> con_;
};

// ---------------------------------------------------------------------------
// Fixture 1 — wholly-NULL and partially-NULL columns across every flat type
//
//   id      INTEGER        always valid (row identity)
//   n_*                    wholly NULL, one per type
//   p_*                    partially NULL (valid on even rows), one per type
//
// The two shapes decode differently: a wholly-NULL column has every definition
// level 0 (and may carry no value page at all), whereas a partially-NULL column
// carries a real per-row definition level stream that must be zipped back
// against the values. Covering only INTEGER for the partial case would leave
// the per-type definition-level decode paths (DECIMAL / DOUBLE / DATE / VARCHAR
// / BOOLEAN / FLOAT / TIMESTAMP) untested.
//
// BOOLEAN is notable: its values are bit-packed, so validity and value bit
// streams are separately packed and easy to misalign.
// ---------------------------------------------------------------------------

class ParquetNullFixture : public sirius::test::GpuExecutionFixture {
 public:
  ParquetNullFixture()
  {
    auto const pq_path = pq_.path("nulls.parquet");

    // Column types are declared on the TABLE rather than inferred from a bare
    // COPY (SELECT ...). Inference would silently collapse this matrix: DuckDB's
    // range() yields BIGINT (so an unqualified `i` gives INT64, not INT32) and a
    // literal like 1.5 is DECIMAL (so `i * 1.5` gives DECIMAL, not DOUBLE).
    // Declaring the types makes the INSERT cast to them and pins what is written.
    pq_.write({
      "CREATE TABLE nt ("
      "  id      INTEGER,"
      // Wholly-NULL, one per type / physical encoding.
      "  n_int   INTEGER,"
      "  n_big   BIGINT,"
      "  n_dbl   DOUBLE,"
      "  n_flt   FLOAT,"
      "  n_dec32 DECIMAL(9,2),"   // precision <= 9  -> parquet INT32
      "  n_dec64 DECIMAL(18,2),"  // precision <= 18 -> parquet INT64
      "  n_date  DATE,"
      "  n_ts    TIMESTAMP,"
      "  n_bool  BOOLEAN,"
      "  n_str   VARCHAR,"
      // Partially-NULL (valid on even rows), one per type / physical encoding.
      "  p_int   INTEGER,"
      "  p_big   BIGINT,"
      "  p_dbl   DOUBLE,"
      "  p_flt   FLOAT,"
      "  p_dec32 DECIMAL(9,2),"
      "  p_dec64 DECIMAL(18,2),"
      "  p_date  DATE,"
      "  p_ts    TIMESTAMP,"
      "  p_bool  BOOLEAN,"
      "  p_str   VARCHAR)",

      "INSERT INTO nt SELECT"
      "  i,"
      "  NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL,"
      "  CASE WHEN i % 2 = 0 THEN i END,"
      "  CASE WHEN i % 2 = 0 THEN i * 1000 END,"
      "  CASE WHEN i % 2 = 0 THEN i * 1.5 END,"
      "  CASE WHEN i % 2 = 0 THEN i * 0.25 END,"
      "  CASE WHEN i % 2 = 0 THEN i END,"
      "  CASE WHEN i % 2 = 0 THEN i END,"
      // range() yields BIGINT and DuckDB only defines +(DATE, INTEGER), so the
      // day offset must be narrowed explicitly. (The INTERVAL form below needs
      // no cast: `INTERVAL (x) HOUR` is to_hours(x), which takes BIGINT.)
      "  CASE WHEN i % 2 = 0 THEN DATE '2021-01-01' + CAST(i AS INTEGER) END,"
      "  CASE WHEN i % 2 = 0 THEN TIMESTAMP '2021-01-01 00:00:00' + INTERVAL (i) HOUR END,"
      "  CASE WHEN i % 2 = 0 THEN (i % 4 = 0) END,"
      "  CASE WHEN i % 2 = 0 THEN 'v' || CAST(i AS VARCHAR) END"
      "  FROM range(1, 17) AS t(i)",

      "COPY nt TO " + pq_.sql_literal(pq_path) + " (FORMAT PARQUET)",
    });

    // Pin every physical encoding this fixture claims to exercise, so a change
    // in DuckDB's writer cannot silently collapse two columns onto one path.
    pq_.assert_physical_type(pq_path, "p_int", "INT32");
    pq_.assert_physical_type(pq_path, "p_big", "INT64");
    pq_.assert_physical_type(pq_path, "p_dbl", "DOUBLE");
    pq_.assert_physical_type(pq_path, "p_flt", "FLOAT");
    pq_.assert_physical_type(pq_path, "p_dec32", "INT32");
    pq_.assert_physical_type(pq_path, "p_dec64", "INT64");
    pq_.assert_physical_type(pq_path, "p_date", "INT32");
    pq_.assert_physical_type(pq_path, "p_ts", "INT64");
    pq_.assert_physical_type(pq_path, "p_bool", "BOOLEAN");
    pq_.assert_physical_type(pq_path, "p_str", "BYTE_ARRAY");

    // A physical type alone is ambiguous, so pin the annotations that
    // distinguish same-width columns from one another.
    pq_.assert_logical_type(pq_path, "p_date", "DATE");
    pq_.assert_logical_type(pq_path, "p_ts", "TIMESTAMP_MICROS");
    pq_.assert_logical_type(pq_path, "p_str", "UTF8");
    pq_.assert_logical_type(pq_path, "p_dec32", "DECIMAL", /*precision=*/9, /*scale=*/2);
    pq_.assert_logical_type(pq_path, "p_dec64", "DECIMAL", /*precision=*/18, /*scale=*/2);

    // 16 rows; every n_* wholly NULL, every p_* valid on the 8 even ids.
    for (auto const* col : {"n_int",
                            "n_big",
                            "n_dbl",
                            "n_flt",
                            "n_dec32",
                            "n_dec64",
                            "n_date",
                            "n_ts",
                            "n_bool",
                            "n_str"}) {
      pq_.assert_null_population(pq_path, col, /*nulls=*/16, /*rows=*/16);
    }
    for (auto const* col : {"p_int",
                            "p_big",
                            "p_dbl",
                            "p_flt",
                            "p_dec32",
                            "p_dec64",
                            "p_date",
                            "p_ts",
                            "p_bool",
                            "p_str"}) {
      pq_.assert_null_population(pq_path, col, /*nulls=*/8, /*rows=*/16);
    }
    pq_.assert_null_population(pq_path, "id", /*nulls=*/0, /*rows=*/16);

    scan_ = pq_.scan("nulls.parquet");
  }

 protected:
  ParquetFileGuard pq_{"basic"};
  std::string scan_;
};

// ---------------------------------------------------------------------------
// Fixture 2 — dictionary-encoded column with null entries
//
// DuckDB's COPY TO dictionary-encodes low-cardinality string columns. The
// encoding NAME depends on the parquet version: PLAIN_DICTIONARY under V1
// (DuckDB's default) and RLE_DICTIONARY under V2. The assertion below matches
// the shared 'DICTIONARY' substring so it holds under either, while still
// failing loudly if the column stops being dictionary-encoded altogether and
// silently falls back to PLAIN.
//
// Note parquet_metadata().encodings lists one entry per data page, so repeats
// (e.g. "PLAIN_DICTIONARY, PLAIN_DICTIONARY") are expected on multi-page
// chunks -- another reason to match a substring rather than compare equality.
// ---------------------------------------------------------------------------

class ParquetDictNullFixture : public sirius::test::GpuExecutionFixture {
 public:
  ParquetDictNullFixture()
  {
    auto const pq_path = pq_.path("dict_nulls.parquet");
    pq_.write(
      "COPY ("
      "  SELECT"
      "    i AS id,"
      "    CASE"
      "      WHEN i % 3 = 0 THEN NULL"
      "      WHEN i % 3 = 1 THEN 'alpha'"
      "      ELSE                 'beta'"
      "    END AS cat,"
      "    CASE WHEN i % 5 = 0 THEN NULL ELSE i * 10 END AS val"
      "  FROM range(1, 31) AS t(i)"
      ") TO " +
      pq_.sql_literal(pq_path) + " (FORMAT PARQUET);");

    // Assert that `cat` was actually written with dictionary encoding.
    // parquet_metadata().encodings is a list; cast to VARCHAR and check for
    // the 'DICTIONARY' substring, which covers both RLE_DICTIONARY and the
    // legacy PLAIN_DICTIONARY encoding names.
    pq_.assert_column_encoding(pq_path, "cat", "DICTIONARY");

    // 30 rows: cat NULL on the 10 multiples of 3, val on the 6 multiples of 5.
    pq_.assert_null_population(pq_path, "cat", /*nulls=*/10, /*rows=*/30);
    pq_.assert_null_population(pq_path, "val", /*nulls=*/6, /*rows=*/30);

    scan_ = pq_.scan("dict_nulls.parquet");
  }

 protected:
  ParquetFileGuard pq_{"dict"};
  std::string scan_;
};

// ---------------------------------------------------------------------------
// Fixture 3 — nullable column spanning multiple row groups, ragged tail
//
// The row count must exceed DuckDB's effective row-group granularity for this
// fixture to mean anything: ROW_GROUP_SIZE is not a hard cut, and the writer's
// real granularity is the arriving DataChunk size (STANDARD_VECTOR_SIZE, 2048).
// See assert_min_row_groups for the mechanism.
//
// 4 * 2048 + 203 = 8395 rows with ROW_GROUP_SIZE 2048 therefore yields 4 full
// 2048-row row groups plus a ragged 203-row tail. 203 is deliberately not a
// multiple of 8 (the validity-bitmask word granularity), which is where
// off-by-one errors in the final partial mask word surface. The row-group count
// is asserted so a future writer change cannot silently collapse this back to a
// single row group.
// ---------------------------------------------------------------------------

class ParquetMultiRowGroupFixture : public sirius::test::GpuExecutionFixture {
 public:
  ParquetMultiRowGroupFixture()
  {
    auto const pq_path = pq_.path("multi_rg.parquet");
    pq_.write(
      "COPY ("
      "  SELECT"
      "    CAST(i AS INTEGER) AS id,"
      "    CAST(NULL AS INTEGER) AS n_int,"
      "    CAST(CASE WHEN i % 7 <> 0 THEN i ELSE NULL END AS INTEGER) AS part"
      "  FROM range(1, 8396) AS t(i)"
      // ORDER BY, not incidental ordering: preserve_insertion_order is a global
      // setting DuckDB is explicitly allowed to ignore for queries without one,
      // and every claim below about WHICH rows land in WHICH row group depends
      // on the order. Matches test_pin_table_zone_map_pruning.cpp.
      "  ORDER BY i"
      ") TO " +
      pq_.sql_literal(pq_path) + " (FORMAT PARQUET, ROW_GROUP_SIZE 2048);");

    // 8395 rows / 2048 == 4 full row groups + a 203-row tail.
    // The exact layout, not just a count: the ragged-tail test below targets
    // rows by id, which only isolates the final group if the split is as claimed.
    pq_.assert_row_group_sizes(pq_path, {2048, 2048, 2048, 2048, 203});
    // 8395 rows; part NULL on the 1199 multiples of 7 in [1, 8395].
    //
    // 7, not 4: the row-group size 2048 is a multiple of 4, so a period of 4
    // would give every group an identical validity mask and identical null
    // count. A decoder reusing the first group's mask for the rest would then
    // satisfy every count, sum and row comparison. 7 is co-prime with 2048, so
    // each group starts at a different phase and holds a different number of
    // NULLs.
    pq_.assert_null_population(pq_path, "n_int", /*nulls=*/8395, /*rows=*/8395);
    pq_.assert_null_population(pq_path, "part", /*nulls=*/1199, /*rows=*/8395);
    // Per group, which pins the row->group mapping the ragged-tail test relies
    // on. The counts differ between groups precisely because the period does not
    // divide the group size — that difference is what makes a reused mask
    // detectable.
    pq_.assert_null_counts_per_row_group(pq_path, "part", {292, 293, 292, 293, 29});
    pq_.assert_null_counts_per_row_group(pq_path, "n_int", {2048, 2048, 2048, 2048, 203});

    scan_ = pq_.scan("multi_rg.parquet");
  }

 protected:
  ParquetFileGuard pq_{"mrg"};
  std::string scan_;
};

// ---------------------------------------------------------------------------
// Fixture 4 — dense (zero-NULL) columns baseline
//
// An OPTIONAL column that happens to contain no NULLs at all: every definition
// level is 1, and the decoder must produce an all-valid column rather than
// inventing NULLs. This is the shape of most real data (TPC-H etc.), so it is
// the baseline the NULL-bearing fixtures are measured against.
//
// This is NOT the REQUIRED (no definition-level stream) case, which DuckDB
// cannot write -- see KNOWN GAPS above. The OPTIONAL repetition is asserted so
// the fixture states what it actually exercises.
// ---------------------------------------------------------------------------

class ParquetDenseColFixture : public sirius::test::GpuExecutionFixture {
 public:
  ParquetDenseColFixture()
  {
    auto const pq_path = pq_.path("dense.parquet");

    pq_.write({
      "CREATE TABLE dense (id INTEGER, val INTEGER, s VARCHAR)",
      "INSERT INTO dense SELECT i, i * 2, 'str_' || CAST(i AS VARCHAR) FROM range(1, 33) AS t(i)",
      "COPY dense TO " + pq_.sql_literal(pq_path) + " (FORMAT PARQUET)",
    });

    // Pin what DuckDB actually emits: OPTIONAL, with zero nulls present.
    pq_.assert_repetition_type(pq_path, "id", "OPTIONAL");
    pq_.assert_repetition_type(pq_path, "val", "OPTIONAL");
    pq_.assert_repetition_type(pq_path, "s", "OPTIONAL");

    // The point of the fixture: OPTIONAL columns carrying no NULLs at all.
    for (auto const* col : {"id", "val", "s"}) {
      pq_.assert_null_population(pq_path, col, /*nulls=*/0, /*rows=*/32);
    }

    scan_ = pq_.scan("dense.parquet");
  }

 protected:
  ParquetFileGuard pq_{"dense"};
  std::string scan_;
};

// ---------------------------------------------------------------------------
// Fixture 5 — run-shaped NULL layouts
//
// Ports the run-layout cases from test_gpu_execution_allnull_scan.cpp to the
// parquet reader. Both columns hold long single-valued runs, which is exactly
// where a reader may take a constant/RLE fast path and mask the wrong side:
//
//   c_run  constant valid run (7) for the first half, then all-NULL
//   c_pre  long NULL prefix, then a valid suffix
//
// If the valid run were wrongly masked, COUNT/SUM would fall below the
// expected totals -- the assertions below pin those.
// ---------------------------------------------------------------------------

class ParquetNullRunFixture : public sirius::test::GpuExecutionFixture {
 public:
  ParquetNullRunFixture()
  {
    pq_.write(
      "COPY ("
      "  SELECT"
      "    i AS id,"
      "    CASE WHEN i <  500  THEN 7 ELSE NULL END AS c_run,"
      "    CASE WHEN i >= 7000 THEN i ELSE NULL END AS c_pre"
      "  FROM range(8000) AS t(i)"
      // The runs are only contiguous if the rows are written in order; see the
      // multi-row-group fixture.
      "  ORDER BY i"
      ") TO " +
      pq_.sql_literal(pq_.path("runs.parquet")) + " (FORMAT PARQUET);");
    // 8000 rows (ids 0..7999): c_run valid on the first 500, c_pre on the last
    // 1000. Pinned because a run-shaped column is exactly where a generation
    // slip would leave both GPU and CPU agreeing on the wrong data.
    pq_.assert_null_population(pq_.path("runs.parquet"), "c_run", /*nulls=*/7500, /*rows=*/8000);
    pq_.assert_null_population(pq_.path("runs.parquet"), "c_pre", /*nulls=*/7000, /*rows=*/8000);

    scan_ = pq_.scan("runs.parquet");
  }

 protected:
  ParquetFileGuard pq_{"runs"};
  std::string scan_;
};

// ---------------------------------------------------------------------------
// Fixture 6 — FLBA / BYTE_ARRAY decimal with nulls
//
// DuckDB picks a decimal's parquet physical type by precision: INT32 up to 9,
// INT64 up to 18, and FIXED_LEN_BYTE_ARRAY beyond that. Sirius disables
// reader-side row-group filter pushdown when an FLBA / BYTE_ARRAY decimal is
// among the scanned columns, because cudf's stats filter cannot compare a
// fixed_point_scalar literal against those stats -- the filter is instead
// applied post-decode (parquet_gpu_ingestible.cpp).
//
// Every other fixture here uses DECIMAL(10,2), which lands on INT32/INT64 and
// therefore never reaches that branch. DECIMAL(38,4) forces FLBA; the fixture
// asserts the physical type so this does not silently regress into testing the
// INT64 path. Filters over the FLBA decimal exercise the post-decode path with
// NULLs present.
// ---------------------------------------------------------------------------

class ParquetFlbaDecimalNullFixture : public sirius::test::GpuExecutionFixture {
 public:
  ParquetFlbaDecimalNullFixture()
  {
    auto const pq_path = pq_.path("flba_dec.parquet");
    pq_.write(
      "COPY ("
      "  SELECT"
      "    i AS id,"
      // Precision 38 forces FIXED_LEN_BYTE_ARRAY.
      "    CASE WHEN i % 3 <> 0 THEN CAST(i AS DECIMAL(38,4)) ELSE NULL END AS big_dec,"
      "    CAST(NULL AS DECIMAL(38,4)) AS n_big_dec"
      "  FROM range(1, 41) AS t(i)"
      ") TO " +
      pq_.sql_literal(pq_path) + " (FORMAT PARQUET);");

    // Pin the physical type -- the whole point of this fixture is the FLBA
    // decode + disabled-pushdown path, not decimals in general.
    pq_.assert_physical_type(pq_path, "big_dec", "FIXED_LEN_BYTE_ARRAY");
    pq_.assert_physical_type(pq_path, "n_big_dec", "FIXED_LEN_BYTE_ARRAY");
    // FLBA alone is not what disables reader-side pushdown -- the DECIMAL
    // annotation is what parquet_gpu_ingestible keys off -- so pin that too.
    pq_.assert_logical_type(pq_path, "big_dec", "DECIMAL", /*precision=*/38, /*scale=*/4);
    pq_.assert_logical_type(pq_path, "n_big_dec", "DECIMAL", /*precision=*/38, /*scale=*/4);

    // 40 rows; big_dec NULL on the 13 multiples of 3 in [1, 40].
    pq_.assert_null_population(pq_path, "big_dec", /*nulls=*/13, /*rows=*/40);
    pq_.assert_null_population(pq_path, "n_big_dec", /*nulls=*/40, /*rows=*/40);

    scan_ = pq_.scan("flba_dec.parquet");
  }

 protected:
  ParquetFileGuard pq_{"flba"};
  std::string scan_;
};

}  // namespace

// ===========================================================================
// Tests: wholly-NULL columns
// ===========================================================================

TEST_CASE_METHOD(ParquetNullFixture,
                 "parquet nulls — wholly-NULL column projection preserves NULLs",
                 "[integration][gpu_execution][scan][nulls][parquet]")
{
  // Every typed NULL column must come back as NULL on the GPU, not as
  // sentinel / garbage values.
  compare_gpu_vs_cpu(
    "SELECT id, n_int, n_big, n_dbl, n_flt, n_dec32, n_dec64, n_date, n_ts, n_bool, n_str "
    "FROM " +
    scan_);
}

TEST_CASE_METHOD(ParquetNullFixture,
                 "parquet nulls — COUNT(*) vs COUNT(col) on wholly-NULL columns",
                 "[integration][gpu_execution][scan][nulls][parquet]")
{
  // COUNT(*) == 16; every COUNT(n_*) == 0 because all values are NULL.
  compare_gpu_vs_cpu(
    "SELECT COUNT(*), COUNT(n_int), COUNT(n_big), COUNT(n_dbl), COUNT(n_flt), COUNT(n_dec32), "
    "COUNT(n_dec64), "
    "COUNT(n_date), COUNT(n_ts), COUNT(n_bool), COUNT(n_str) FROM " +
    scan_);
}

TEST_CASE_METHOD(ParquetNullFixture,
                 "parquet nulls — aggregates skip NULLs on wholly-NULL columns",
                 "[integration][gpu_execution][scan][nulls][parquet]")
{
  // SUM/MIN/MAX over a wholly-NULL column must return NULL, not 0 or a sentinel.
  compare_gpu_vs_cpu("SELECT SUM(n_int), MIN(n_int), MAX(n_int) FROM " + scan_);
  compare_gpu_vs_cpu("SELECT SUM(n_dec32), MIN(n_dec32), MAX(n_dec32) FROM " + scan_);
  compare_gpu_vs_cpu("SELECT SUM(n_dec64), MIN(n_dec64), MAX(n_dec64) FROM " + scan_);
  compare_gpu_vs_cpu("SELECT MIN(n_date), MAX(n_date), MIN(n_ts), MAX(n_ts) FROM " + scan_);
}

TEST_CASE_METHOD(ParquetNullFixture,
                 "parquet nulls — IS NULL / IS NOT NULL on wholly-NULL column",
                 "[integration][gpu_execution][scan][nulls][parquet]")
{
  // Every row is NULL: IS NULL keeps all 16 rows, IS NOT NULL keeps none.
  compare_gpu_vs_cpu("SELECT COUNT(*) FROM " + scan_ + " WHERE n_int IS NULL");
  compare_gpu_vs_cpu("SELECT COUNT(*) FROM " + scan_ + " WHERE n_int IS NOT NULL");
  compare_gpu_vs_cpu("SELECT COUNT(*) FROM " + scan_ + " WHERE n_date IS NULL");
  compare_gpu_vs_cpu("SELECT COUNT(*) FROM " + scan_ + " WHERE n_bool IS NULL");
  compare_gpu_vs_cpu("SELECT id FROM " + scan_ + " WHERE n_str IS NULL ORDER BY id");
}

// A wholly-NULL column selected ON ITS OWN. Every other case keeps `id` or
// aggregates over the full schema, which can mask a column-pruning bug: an
// error in leaf selection / column-name plumbing shows up only when the NULL
// column is the sole projected column.
TEST_CASE_METHOD(ParquetNullFixture,
                 "parquet nulls — column-pruned scan projecting only a wholly-NULL column",
                 "[integration][gpu_execution][scan][nulls][parquet]")
{
  compare_gpu_vs_cpu("SELECT n_int FROM " + scan_);
  compare_gpu_vs_cpu("SELECT n_str FROM " + scan_);
  compare_gpu_vs_cpu("SELECT n_dec32 FROM " + scan_);
  compare_gpu_vs_cpu("SELECT n_dec64 FROM " + scan_);
  compare_gpu_vs_cpu("SELECT n_bool FROM " + scan_);
  compare_gpu_vs_cpu("SELECT n_ts FROM " + scan_);
}

TEST_CASE_METHOD(ParquetNullFixture,
                 "parquet nulls — column-pruned scan projecting only a partially-NULL column",
                 "[integration][gpu_execution][scan][nulls][parquet]")
{
  compare_gpu_vs_cpu("SELECT p_int FROM " + scan_);
  compare_gpu_vs_cpu("SELECT p_str FROM " + scan_);
  compare_gpu_vs_cpu("SELECT p_bool FROM " + scan_);
}

// ===========================================================================
// Tests: partially-NULL columns (per-row definition levels, every type)
// ===========================================================================

TEST_CASE_METHOD(ParquetNullFixture,
                 "parquet nulls — partially-NULL columns keep their valid rows across all types",
                 "[integration][gpu_execution][scan][nulls][parquet]")
{
  // Each p_* column is valid only on even ids. A definition-level decode bug
  // for any one type shows up as a wrong value or a wrongly-placed NULL here.
  compare_gpu_vs_cpu(
    "SELECT id, p_int, p_big, p_dbl, p_flt, p_dec32, p_dec64, p_date, p_ts, p_bool, p_str "
    "FROM " +
    scan_);
}

TEST_CASE_METHOD(ParquetNullFixture,
                 "parquet nulls — aggregates over partially-NULL columns across all types",
                 "[integration][gpu_execution][scan][nulls][parquet]")
{
  // COUNT must be 8 (even ids in [1,16]) for every partially-NULL column.
  compare_gpu_vs_cpu(
    "SELECT COUNT(p_int), COUNT(p_big), COUNT(p_dbl), COUNT(p_flt), COUNT(p_dec32), "
    "COUNT(p_dec64), "
    "COUNT(p_date), COUNT(p_ts), COUNT(p_bool), COUNT(p_str) FROM " +
    scan_);
  compare_gpu_vs_cpu("SELECT SUM(p_int), MIN(p_int), MAX(p_int) FROM " + scan_);
  compare_gpu_vs_cpu("SELECT SUM(p_big), MIN(p_big), MAX(p_big) FROM " + scan_);
  compare_gpu_vs_cpu("SELECT SUM(p_dec32), MIN(p_dec32), MAX(p_dec32) FROM " + scan_);
  compare_gpu_vs_cpu("SELECT SUM(p_dec64), MIN(p_dec64), MAX(p_dec64) FROM " + scan_);
  compare_gpu_vs_cpu("SELECT MIN(p_date), MAX(p_date), MIN(p_ts), MAX(p_ts) FROM " + scan_);
  compare_gpu_vs_cpu("SELECT MIN(p_str), MAX(p_str) FROM " + scan_);
}

// Floating point aggregation order differs between GPU reduction and CPU serial
// summation, so DOUBLE/FLOAT sums compare with a relative tolerance.
TEST_CASE_METHOD(ParquetNullFixture,
                 "parquet nulls — floating-point aggregates over partially-NULL columns",
                 "[integration][gpu_execution][scan][nulls][parquet]")
{
  compare_gpu_vs_cpu_approx("SELECT SUM(p_dbl), SUM(p_flt) FROM " + scan_, {0, 1});
  compare_gpu_vs_cpu("SELECT MIN(p_dbl), MAX(p_dbl), MIN(p_flt), MAX(p_flt) FROM " + scan_);
}

TEST_CASE_METHOD(ParquetNullFixture,
                 "parquet nulls — IS NULL / IS NOT NULL across partially-NULL types",
                 "[integration][gpu_execution][scan][nulls][parquet]")
{
  compare_gpu_vs_cpu("SELECT id FROM " + scan_ + " WHERE p_int IS NULL ORDER BY id");
  compare_gpu_vs_cpu("SELECT id FROM " + scan_ + " WHERE p_int IS NOT NULL ORDER BY id");
  compare_gpu_vs_cpu("SELECT id FROM " + scan_ + " WHERE p_dec32 IS NOT NULL ORDER BY id");
  compare_gpu_vs_cpu("SELECT id FROM " + scan_ + " WHERE p_dec64 IS NOT NULL ORDER BY id");
  compare_gpu_vs_cpu("SELECT id FROM " + scan_ + " WHERE p_date IS NOT NULL ORDER BY id");
  compare_gpu_vs_cpu("SELECT id FROM " + scan_ + " WHERE p_ts IS NOT NULL ORDER BY id");
  compare_gpu_vs_cpu("SELECT id FROM " + scan_ + " WHERE p_str IS NOT NULL ORDER BY id");
  // BOOLEAN is bit-packed: a validity/value bit misalignment shows up as rows
  // shifting between the NULL and non-NULL sets.
  compare_gpu_vs_cpu("SELECT id, p_bool FROM " + scan_ + " WHERE p_bool IS NOT NULL ORDER BY id");
  compare_gpu_vs_cpu("SELECT id FROM " + scan_ + " WHERE p_bool ORDER BY id");
}

TEST_CASE_METHOD(ParquetNullFixture,
                 "parquet nulls — COALESCE on partially-NULL column",
                 "[integration][gpu_execution][scan][nulls][parquet]")
{
  compare_gpu_vs_cpu("SELECT id, COALESCE(p_int, -1) AS v FROM " + scan_);
  compare_gpu_vs_cpu("SELECT id, COALESCE(p_str, 'none') AS v FROM " + scan_);
}

TEST_CASE_METHOD(ParquetNullFixture,
                 "parquet nulls — CASE expression over partially-NULL column",
                 "[integration][gpu_execution][scan][nulls][parquet]")
{
  compare_gpu_vs_cpu("SELECT id, CASE WHEN p_int IS NULL THEN 0 ELSE p_int END AS v FROM " + scan_);
}

// ===========================================================================
// Tests: dictionary-encoded column with nulls
// ===========================================================================

TEST_CASE_METHOD(ParquetDictNullFixture,
                 "parquet nulls — dictionary-encoded string column with nulls projection",
                 "[integration][gpu_execution][scan][nulls][parquet]")
{
  compare_gpu_vs_cpu("SELECT id, cat FROM " + scan_);
}

TEST_CASE_METHOD(ParquetDictNullFixture,
                 "parquet nulls — IS NULL on dictionary-encoded column",
                 "[integration][gpu_execution][scan][nulls][parquet]")
{
  compare_gpu_vs_cpu("SELECT COUNT(*) FROM " + scan_ + " WHERE cat IS NULL");
  compare_gpu_vs_cpu("SELECT id FROM " + scan_ + " WHERE cat IS NULL ORDER BY id");
}

TEST_CASE_METHOD(ParquetDictNullFixture,
                 "parquet nulls — GROUP BY dictionary-encoded column with nulls",
                 "[integration][gpu_execution][scan][nulls][parquet]")
{
  // NULL group key must form its own group (SQL NULL semantics for GROUP BY).
  compare_gpu_vs_cpu("SELECT cat, COUNT(*), SUM(val) FROM " + scan_ + " GROUP BY cat");
}

TEST_CASE_METHOD(ParquetDictNullFixture,
                 "parquet nulls — numeric column with nulls aggregates",
                 "[integration][gpu_execution][scan][nulls][parquet]")
{
  compare_gpu_vs_cpu("SELECT COUNT(*), COUNT(val), SUM(val), MIN(val), MAX(val) FROM " + scan_);
}

TEST_CASE_METHOD(ParquetDictNullFixture,
                 "parquet nulls — filter on nullable numeric column",
                 "[integration][gpu_execution][scan][nulls][parquet]")
{
  compare_gpu_vs_cpu("SELECT id, val FROM " + scan_ + " WHERE val > 50 ORDER BY id");
  // Comparison with NULL in predicate: val = NULL must return false, not true.
  compare_gpu_vs_cpu("SELECT id FROM " + scan_ +
                     " WHERE val IS NOT DISTINCT FROM NULL ORDER BY id");
}

// ===========================================================================
// Tests: multi-row-group parquet with nulls
// ===========================================================================

TEST_CASE_METHOD(ParquetMultiRowGroupFixture,
                 "parquet nulls — wholly-NULL column across multiple row groups",
                 "[integration][gpu_execution][scan][nulls][parquet]")
{
  // COUNT(n_int) must be 0 across every row group; SUM must be NULL.
  compare_gpu_vs_cpu("SELECT COUNT(*), COUNT(n_int), SUM(n_int) FROM " + scan_);
  // Aggregated rather than row-by-row: the fixture is deliberately large
  // (8395 rows) to force multiple row groups, and a full row comparison here
  // would be slow without testing anything the counts do not already pin.
  compare_gpu_vs_cpu("SELECT COUNT(*) FROM " + scan_ + " WHERE n_int IS NULL");
  compare_gpu_vs_cpu("SELECT COUNT(*) FROM " + scan_ + " WHERE n_int IS NOT NULL");
}

TEST_CASE_METHOD(ParquetMultiRowGroupFixture,
                 "parquet nulls — partially-NULL column across multiple row groups",
                 "[integration][gpu_execution][scan][nulls][parquet]")
{
  // `part` is null for id % 7 == 0 rows; row group boundaries must not
  // corrupt the validity bitmask of adjacent rows.
  compare_gpu_vs_cpu("SELECT COUNT(*), COUNT(part), SUM(part) FROM " + scan_);
  compare_gpu_vs_cpu("SELECT COUNT(*) FROM " + scan_ + " WHERE part IS NULL");
  compare_gpu_vs_cpu("SELECT COUNT(*) FROM " + scan_ + " WHERE part IS NOT NULL");
  // Row-level, at every group boundary. The aggregates above cannot see a
  // validity mask shifted across a boundary: the null count stays 1199 and the
  // SUM can be preserved by a shift that swaps which ids are masked. Only
  // comparing (id, value) pairs in order catches that, and it has to be done at
  // each of 2048 / 4096 / 6144 as well as the ragged tail.
  for (auto const boundary : {2048, 4096, 6144}) {
    compare_gpu_vs_cpu_ordered("SELECT id, part FROM " + scan_ + " WHERE id BETWEEN " +
                               std::to_string(boundary - 8) + " AND " +
                               std::to_string(boundary + 8) + " ORDER BY id");
  }
  compare_gpu_vs_cpu_ordered("SELECT id, part FROM " + scan_ + " WHERE id > 8192 ORDER BY id");
}

// ===========================================================================
// Tests: dense (zero-NULL) columns baseline
// ===========================================================================

TEST_CASE_METHOD(ParquetDenseColFixture,
                 "parquet nulls — dense zero-NULL columns produce no spurious NULLs",
                 "[integration][gpu_execution][scan][nulls][parquet]")
{
  // Every definition level is 1; the reader must not synthesize any NULLs.
  compare_gpu_vs_cpu("SELECT id, val, s FROM " + scan_);
  compare_gpu_vs_cpu("SELECT COUNT(*), COUNT(val), COUNT(s) FROM " + scan_);
  // IS NULL must return no rows.
  compare_gpu_vs_cpu("SELECT COUNT(*) FROM " + scan_ + " WHERE val IS NULL");
  compare_gpu_vs_cpu("SELECT COUNT(*) FROM " + scan_ + " WHERE s IS NULL");
}

// ===========================================================================
// Tests: run-shaped NULL layouts (ported from test_gpu_execution_allnull_scan)
// ===========================================================================

TEST_CASE_METHOD(ParquetNullRunFixture,
                 "parquet nulls — constant-valid run followed by an all-NULL run",
                 "[integration][gpu_execution][scan][nulls][parquet]")
{
  // c_run is 7 for ids [0,500) and NULL after. If the constant valid run were
  // wrongly masked, COUNT(c_run)/SUM(c_run) would drop below 500/3500.
  compare_gpu_vs_cpu("SELECT COUNT(*), COUNT(c_run), SUM(c_run), MIN(c_run), MAX(c_run) FROM " +
                     scan_);
  compare_gpu_vs_cpu("SELECT COUNT(*) FROM " + scan_ + " WHERE c_run IS NOT NULL");
  compare_gpu_vs_cpu("SELECT COUNT(*) FROM " + scan_ + " WHERE c_run IS NULL");
  // c_run is the CONSTANT 7, so ANY equal-count corruption of its validity mask
  // preserves COUNT, SUM, MIN and MAX. A window around the transition would
  // only catch a shift at that one point, so compare every row: 8000 ordered
  // pairs is cheap next to being unable to see the corruption at all.
  compare_gpu_vs_cpu_ordered("SELECT id, c_run FROM " + scan_ + " ORDER BY id");
  // The valid set stated outright, so a corruption that happened to be
  // symmetric across the comparison still fails.
  compare_gpu_vs_cpu_ordered("SELECT id FROM " + scan_ + " WHERE c_run IS NOT NULL ORDER BY id");
}

TEST_CASE_METHOD(ParquetNullRunFixture,
                 "parquet nulls — long NULL prefix followed by a valid suffix",
                 "[integration][gpu_execution][scan][nulls][parquet]")
{
  // c_pre is NULL for ids [0,7000) and valid after. COUNT(c_pre)=1000 and the
  // SUM over [7000,8000) only hold if the trailing valid rows were not masked.
  compare_gpu_vs_cpu("SELECT COUNT(*), COUNT(c_pre), SUM(c_pre), MIN(c_pre), MAX(c_pre) FROM " +
                     scan_);
  compare_gpu_vs_cpu("SELECT COUNT(*) FROM " + scan_ + " WHERE c_pre IS NOT NULL");
  // Same reasoning as c_run, though c_pre's payload varies so SUM would catch
  // some shifts; compare every row regardless.
  compare_gpu_vs_cpu_ordered("SELECT id, c_pre FROM " + scan_ + " ORDER BY id");
  compare_gpu_vs_cpu_ordered("SELECT id FROM " + scan_ + " WHERE c_pre IS NOT NULL ORDER BY id");
}

TEST_CASE_METHOD(ParquetNullRunFixture,
                 "parquet nulls — run-shaped columns projected alone",
                 "[integration][gpu_execution][scan][nulls][parquet]")
{
  // Column-pruned: the run column is the only projected column.
  compare_gpu_vs_cpu("SELECT COUNT(c_run) FROM " + scan_);
  compare_gpu_vs_cpu("SELECT COUNT(c_pre) FROM " + scan_);
}

// ===========================================================================
// Tests: FLBA / BYTE_ARRAY decimal with NULLs
// ===========================================================================

TEST_CASE_METHOD(ParquetFlbaDecimalNullFixture,
                 "parquet nulls — FLBA decimal projection preserves NULLs",
                 "[integration][gpu_execution][scan][nulls][parquet]")
{
  compare_gpu_vs_cpu("SELECT id, big_dec, n_big_dec FROM " + scan_);
  compare_gpu_vs_cpu("SELECT COUNT(*), COUNT(big_dec), COUNT(n_big_dec) FROM " + scan_);
}

TEST_CASE_METHOD(ParquetFlbaDecimalNullFixture,
                 "parquet nulls — FLBA decimal aggregates skip NULLs",
                 "[integration][gpu_execution][scan][nulls][parquet]")
{
  compare_gpu_vs_cpu("SELECT SUM(big_dec), MIN(big_dec), MAX(big_dec) FROM " + scan_);
  // Wholly-NULL FLBA decimal: SUM/MIN/MAX must all be NULL.
  compare_gpu_vs_cpu("SELECT SUM(n_big_dec), MIN(n_big_dec), MAX(n_big_dec) FROM " + scan_);
}

// Filters over an FLBA decimal take the post-decode path, because Sirius
// disables reader-side row-group pushdown when such a column is scanned. The
// NULL rows must be excluded by the comparison (three-valued logic) rather
// than compared as sentinel values.
TEST_CASE_METHOD(ParquetFlbaDecimalNullFixture,
                 "parquet nulls — filter on FLBA decimal with NULLs (post-decode path)",
                 "[integration][gpu_execution][scan][nulls][parquet]")
{
  compare_gpu_vs_cpu("SELECT id, big_dec FROM " + scan_ + " WHERE big_dec > 20 ORDER BY id");
  compare_gpu_vs_cpu("SELECT id FROM " + scan_ + " WHERE big_dec IS NULL ORDER BY id");
  compare_gpu_vs_cpu("SELECT id FROM " + scan_ + " WHERE big_dec IS NOT NULL ORDER BY id");
  // A comparison against NULL rows must yield no match, not a sentinel hit.
  compare_gpu_vs_cpu("SELECT COUNT(*) FROM " + scan_ + " WHERE n_big_dec > 0");
}

TEST_CASE_METHOD(ParquetFlbaDecimalNullFixture,
                 "parquet nulls — FLBA decimal projected alone",
                 "[integration][gpu_execution][scan][nulls][parquet]")
{
  compare_gpu_vs_cpu("SELECT big_dec FROM " + scan_);
  compare_gpu_vs_cpu("SELECT n_big_dec FROM " + scan_);
}
