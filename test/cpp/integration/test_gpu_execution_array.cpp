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

/**
 * @file test_gpu_execution_array.cpp
 * @brief End-to-end tests for the fixed-size ARRAY data type through gpu_execution.
 *
 * Exercises the full ARRAY data path added alongside the type:
 *   - the GPU native scan path that decodes a stored ARRAY column into a
 *     cuDF LIST column (array-level validity + stride offsets + child values)
 *   - the read-back path that copies the cuDF LIST column into a DuckDB ARRAY
 *     vector (host_table_chunk_reader::copy_array).
 *
 * Each test runs the same query through transparent GPU execution and through
 * DuckDB CPU, then compares results. compare_gpu_vs_cpu() also asserts the
 * query genuinely ran on the GPU (exactly one execution, zero fallbacks), so a
 * silent CPU fallback for ARRAY columns will fail the test rather than pass it.
 */

#include <catch.hpp>
#include <duckdb.hpp>
#include <utils/gpu_execution_fixture.hpp>

#include <string>

// The file-backed native-scan fixture + GPU-vs-CPU comparator are shared with
// the other gpu_execution integration suites; keep the historical name locally.
using ArrayFixture = sirius::test::GpuExecutionFixture;

namespace {

/// Appends @p extra to the live `disabled_optimizers` mask and restores the exact prior mask on
/// scope exit. Sirius publishes a global mask at extension load (`in_clause`,
/// `compressed_materialization`, `late_materialization`) that transparent execution depends on,
/// so a test must never RESET this setting -- that wipes the mask and breaks every later test on
/// the shared database.
class disabled_optimizers_guard {
 public:
  disabled_optimizers_guard(duckdb::Connection& con, const std::string& extra) : _con(con)
  {
    auto current = _con.Query("SELECT current_setting('disabled_optimizers');");
    REQUIRE(current);
    REQUIRE_FALSE(current->HasError());
    _saved       = current->GetValue(0, 0).ToString();
    auto updated = _con.Query("SET disabled_optimizers = '" + _saved + (_saved.empty() ? "" : ",") +
                              extra + "';");
    REQUIRE(updated);
    REQUIRE_FALSE(updated->HasError());
  }
  ~disabled_optimizers_guard()
  {
    auto restored = _con.Query("SET disabled_optimizers = '" + _saved + "';");
    // CHECK (not REQUIRE): destructors must not throw; a failed restore still fails the test.
    CHECK(restored);
    if (restored) { CHECK_FALSE(restored->HasError()); }
  }
  disabled_optimizers_guard(const disabled_optimizers_guard&)            = delete;
  disabled_optimizers_guard& operator=(const disabled_optimizers_guard&) = delete;

 private:
  duckdb::Connection& _con;
  std::string _saved;
};

}  // namespace

//===----------------------------------------------------------------------===//
// Scan + projection passthrough
//===----------------------------------------------------------------------===//
TEST_CASE_METHOD(ArrayFixture,
                 "gpu_execution array - scan INTEGER[3] passthrough",
                 "[integration][gpu_execution][array][scan]")
{
  run_ok("CREATE TABLE arr_int (id INTEGER, a INTEGER[3]);");
  run_ok("INSERT INTO arr_int VALUES (1, [10, 20, 30]), (2, [40, 50, 60]), (3, [70, 80, 90]);");
  run_ok("CHECKPOINT;");  // Persist data to disk for native GPU scan

  // Project both the scalar key and the array column.
  compare_gpu_vs_cpu("SELECT id, a FROM arr_int;");
  // Project the array column alone.
  compare_gpu_vs_cpu("SELECT a FROM arr_int;");
}

TEST_CASE_METHOD(ArrayFixture,
                 "gpu_execution array - fully NULL array column",
                 "[integration][gpu_execution][array][scan]")
{
  // Whole-column NULLs checkpoint to CONSTANT all-null array-level validity;
  // the decoder must synthesize the null mask rather than treat it as valid.
  run_ok("CREATE TABLE arr_allnull (id INTEGER, a INTEGER[3]);");
  run_ok("INSERT INTO arr_allnull SELECT range::INTEGER, NULL::INTEGER[3] FROM range(3000);");
  run_ok("CHECKPOINT;");
  compare_gpu_vs_cpu("SELECT id, a FROM arr_allnull ORDER BY id;");
  compare_gpu_vs_cpu("SELECT count(*), count(a) FROM arr_allnull;");
}

TEST_CASE_METHOD(ArrayFixture,
                 "gpu_execution array - arrays of all-NULL elements",
                 "[integration][gpu_execution][array][scan]")
{
  // Present arrays whose elements are all NULL: CONSTANT all-null child
  // validity, decoded via the synthesized zero-bit path.
  run_ok("CREATE TABLE arr_nullelems (id INTEGER, a INTEGER[3]);");
  run_ok(
    "INSERT INTO arr_nullelems SELECT range::INTEGER, "
    "[NULL, NULL, NULL]::INTEGER[3] FROM range(3000);");
  run_ok("CHECKPOINT;");
  compare_gpu_vs_cpu("SELECT id, a FROM arr_nullelems ORDER BY id;");
}

TEST_CASE_METHOD(ArrayFixture,
                 "gpu_execution array - element types DOUBLE and BIGINT",
                 "[integration][gpu_execution][array][scan]")
{
  run_ok("CREATE TABLE arr_dbl (id INTEGER, a DOUBLE[2]);");
  run_ok("INSERT INTO arr_dbl VALUES (1, [1.5, 2.5]), (2, [3.25, 4.75]);");
  run_ok("CHECKPOINT;");
  compare_gpu_vs_cpu("SELECT id, a FROM arr_dbl;");

  run_ok("CREATE TABLE arr_big (id INTEGER, a BIGINT[4]);");
  run_ok("INSERT INTO arr_big VALUES (1, [100, 200, 300, 400]), (2, [500, 600, 700, 800]);");
  run_ok("CHECKPOINT;");
  compare_gpu_vs_cpu("SELECT id, a FROM arr_big;");
}

TEST_CASE_METHOD(ArrayFixture,
                 "gpu_execution array - signed integer element widths 1/2/4/8 bytes",
                 "[integration][gpu_execution][array][scan]")
{
  // The synthesized stride is sizeof(element)-sensitive: a byte-offset-vs-
  // element-offset confusion stays hidden at 4/8-byte strides (the INTEGER/
  // BIGINT cases above) but surfaces at 1/2 bytes. Sweep every signed width.
  // Each row carries the type max, -1 (all bits set), and 0 so the full byte
  // width must round-trip, not just the low bits.
  run_ok("CREATE TABLE arr_i8 (id INTEGER, a TINYINT[3]);");
  run_ok("INSERT INTO arr_i8 VALUES (1, [1, 2, 3]), (2, [127, -1, 0]);");
  run_ok("CHECKPOINT;");
  compare_gpu_vs_cpu("SELECT id, a FROM arr_i8;");

  run_ok("CREATE TABLE arr_i16 (id INTEGER, a SMALLINT[4]);");
  run_ok("INSERT INTO arr_i16 VALUES (1, [10, -20, 30, -40]), (2, [32767, -1, 0, -100]);");
  run_ok("CHECKPOINT;");
  compare_gpu_vs_cpu("SELECT id, a FROM arr_i16;");

  run_ok("CREATE TABLE arr_i32 (id INTEGER, a INTEGER[3]);");
  run_ok("INSERT INTO arr_i32 VALUES (1, [100, -200, 300]), (2, [2147483647, -1, 0]);");
  run_ok("CHECKPOINT;");
  compare_gpu_vs_cpu("SELECT id, a FROM arr_i32;");

  run_ok("CREATE TABLE arr_i64 (id INTEGER, a BIGINT[2]);");
  run_ok("INSERT INTO arr_i64 VALUES (1, [9223372036854775807, -1]), (2, [0, -100]);");
  run_ok("CHECKPOINT;");
  compare_gpu_vs_cpu("SELECT id, a FROM arr_i64;");
}

TEST_CASE_METHOD(ArrayFixture,
                 "gpu_execution array - unsigned integer element widths 1/2/4/8 bytes",
                 "[integration][gpu_execution][array][scan]")
{
  // Unsigned children take a distinct readback path (cuDF UINT8/16/32/64 mapped
  // back to DuckDB U* types). The type max sets the high bit, so a signed misread
  // would surface as a negative value and fail the GPU-vs-CPU comparison.
  run_ok("CREATE TABLE arr_u8 (id INTEGER, a UTINYINT[3]);");
  run_ok("INSERT INTO arr_u8 VALUES (1, [0, 1, 255]), (2, [128, 200, 254]);");
  run_ok("CHECKPOINT;");
  compare_gpu_vs_cpu("SELECT id, a FROM arr_u8;");

  run_ok("CREATE TABLE arr_u16 (id INTEGER, a USMALLINT[2]);");
  run_ok("INSERT INTO arr_u16 VALUES (1, [0, 65535]), (2, [32768, 40000]);");
  run_ok("CHECKPOINT;");
  compare_gpu_vs_cpu("SELECT id, a FROM arr_u16;");

  run_ok("CREATE TABLE arr_u32 (id INTEGER, a UINTEGER[2]);");
  run_ok("INSERT INTO arr_u32 VALUES (1, [0, 4294967295]), (2, [2147483648, 3000000000]);");
  run_ok("CHECKPOINT;");
  compare_gpu_vs_cpu("SELECT id, a FROM arr_u32;");

  run_ok("CREATE TABLE arr_u64 (id INTEGER, a UBIGINT[2]);");
  run_ok(
    "INSERT INTO arr_u64 VALUES "
    "(1, [0, 18446744073709551615]), (2, [9223372036854775808, 10000000000000000000]);");
  run_ok("CHECKPOINT;");
  compare_gpu_vs_cpu("SELECT id, a FROM arr_u64;");
}

TEST_CASE_METHOD(ArrayFixture,
                 "gpu_execution array - FLOAT elements (embedding-style vectors)",
                 "[integration][gpu_execution][array][scan]")
{
  // FLOAT (4-byte single precision) is the element type for vector-search
  // embeddings, so it gets dedicated coverage. Use exactly-representable binary
  // fractions (multiples of 1/2, 1/4, 1/8) so GPU and CPU stringify identically
  // and the comparison can't fail on float-formatting skew rather than on a real
  // decode bug.
  run_ok("CREATE TABLE arr_flt (id INTEGER, a FLOAT[4]);");
  run_ok(
    "INSERT INTO arr_flt VALUES "
    "(1, [1.5, 2.25, -3.75, 0.125]), (2, [4.5, -5.0, 6.625, 7.0]);");
  run_ok("CHECKPOINT;");
  compare_gpu_vs_cpu("SELECT id, a FROM arr_flt;");

  // Many-row FLOAT vectors crossing batch boundaries, the closest shape to the
  // embedding columns vector search will scan. Values stay exact multiples of
  // 1/8 within float precision across the whole range.
  run_ok("CREATE TABLE arr_flt_big (id INTEGER, a FLOAT[3]);");
  run_ok(
    "INSERT INTO arr_flt_big SELECT i, [i * 0.5, i * 0.25, i * -0.125]::FLOAT[3] "
    "FROM range(5000) t(i);");
  run_ok("CHECKPOINT;");
  compare_gpu_vs_cpu("SELECT id, a FROM arr_flt_big;");
}

TEST_CASE_METHOD(ArrayFixture,
                 "gpu_execution array - single-element arrays (array_size == 1)",
                 "[integration][gpu_execution][array][scan]")
{
  // array_size == 1 is the degenerate stride: the child cursor advances by
  // exactly row_count and each array spans a single child element. A stride
  // off-by-one that wider arrays tolerate (because the slipped element still
  // lands within the same row) misaligns every row here.
  run_ok("CREATE TABLE arr_one (id INTEGER, a INTEGER[1]);");
  // Mix a whole-array NULL with a single-element NULL: at size 1 these stay
  // distinct (array absent vs array present holding one NULL element), so both
  // validity levels are exercised at the minimum stride.
  run_ok("INSERT INTO arr_one VALUES (1, [10]), (2, NULL), (3, [30]), (4, [NULL]);");
  run_ok("CHECKPOINT;");
  compare_gpu_vs_cpu("SELECT id, a FROM arr_one;");

  // Many single-element arrays crossing batch boundaries exercise the cumulative
  // offset bookkeeping at the minimum stride.
  run_ok("CREATE TABLE arr_one_big (id INTEGER, a INTEGER[1]);");
  run_ok("INSERT INTO arr_one_big SELECT i, [i] FROM range(5000) t(i);");
  run_ok("CHECKPOINT;");
  compare_gpu_vs_cpu("SELECT id, a FROM arr_one_big;");
}

//===----------------------------------------------------------------------===//
// Multiple ARRAY columns / rowid alongside ARRAY
//===----------------------------------------------------------------------===//

TEST_CASE_METHOD(ArrayFixture,
                 "gpu_execution array - multiple ARRAY columns in one table",
                 "[integration][gpu_execution][array][scan]")
{
  // Two ARRAY columns of different element types and sizes in one table. Each
  // array column must maintain its own child-element cursor; a shared/global
  // cursor would cross-contaminate b's offsets with a's element count.
  run_ok("CREATE TABLE arr_multi (id INTEGER, a INTEGER[3], b BIGINT[2]);");
  run_ok(
    "INSERT INTO arr_multi VALUES "
    "(1, [1, 2, 3], [100, 200]), (2, [4, 5, 6], [300, 400]), (3, [7, 8, 9], [500, 600]);");
  run_ok("CHECKPOINT;");
  compare_gpu_vs_cpu("SELECT id, a, b FROM arr_multi;");
  // Project the two array columns in the opposite order and without the scalar,
  // so column-position bookkeeping can't lean on a fixed projection layout.
  compare_gpu_vs_cpu("SELECT b, a FROM arr_multi;");

  // Independent nulls across batches: a and b carry whole-array NULLs on
  // different rows (i % 5 vs i % 3), so per-column validity and child cursors
  // must stay independent as the scan crosses vector boundaries.
  run_ok("CREATE TABLE arr_multi_big (id INTEGER, a INTEGER[2], b DOUBLE[3]);");
  run_ok(
    "INSERT INTO arr_multi_big SELECT i, "
    "CASE WHEN i % 5 = 0 THEN NULL ELSE [i, i + 1] END, "
    "CASE WHEN i % 3 = 0 THEN NULL ELSE [i * 0.5, i * 0.25, i * 0.125] END "
    "FROM range(5000) t(i);");
  run_ok("CHECKPOINT;");
  compare_gpu_vs_cpu("SELECT id, a, b FROM arr_multi_big;");
}

TEST_CASE_METHOD(ArrayFixture,
                 "gpu_execution array - rowid projected alongside an ARRAY column",
                 "[integration][gpu_execution][array][scan]")
{
  // rowid is a synthesized column with no on-disk segments. Projecting it next
  // to an ARRAY column checks that the array's child cursor is driven by row
  // counts, not by a sibling's segment layout, and isn't thrown off by the
  // zero-segment rowid neighbor.
  run_ok("CREATE TABLE arr_rowid (a INTEGER[3]);");
  run_ok("INSERT INTO arr_rowid SELECT [i, i + 1, i + 2] FROM range(5000) t(i);");
  run_ok("CHECKPOINT;");
  compare_gpu_vs_cpu("SELECT rowid, a FROM arr_rowid;");
}

//===----------------------------------------------------------------------===//
// List-level validity (NULL arrays)
//===----------------------------------------------------------------------===//

TEST_CASE_METHOD(ArrayFixture,
                 "gpu_execution array - NULL arrays carry list-level validity",
                 "[integration][gpu_execution][array][nulls]")
{
  run_ok("CREATE TABLE arr_null (id INTEGER, a INTEGER[3]);");
  run_ok("INSERT INTO arr_null VALUES (1, [1, 2, 3]), (2, NULL), (3, [7, 8, 9]), (4, NULL);");
  run_ok("CHECKPOINT;");
  compare_gpu_vs_cpu("SELECT id, a FROM arr_null;");
}

TEST_CASE_METHOD(ArrayFixture,
                 "gpu_execution array - NULL elements carry child-level validity",
                 "[integration][gpu_execution][array][nulls]")
{
  run_ok("CREATE TABLE arr_elem_null (id INTEGER, a INTEGER[3]);");
  // Element-level nulls (child validity, path [col,1,0]) are distinct from
  // whole-array nulls (array-level validity, path [col,0]): the array is
  // present but individual elements are NULL. Row 4 mixes in a whole-array
  // NULL so both validity levels decode together in one column.
  run_ok(
    "INSERT INTO arr_elem_null VALUES "
    "(1, [1, NULL, 3]), (2, [NULL, NULL, NULL]), (3, [4, 5, 6]), (4, NULL);");
  run_ok("CHECKPOINT;");
  compare_gpu_vs_cpu("SELECT id, a FROM arr_elem_null;");
}

TEST_CASE_METHOD(ArrayFixture,
                 "gpu_execution array - NULL arrays and elements span multiple scan batches",
                 "[integration][gpu_execution][array][nulls][scan]")
{
  run_ok("CREATE TABLE arr_null_big (id INTEGER, a INTEGER[3]);");
  // The smaller null tests above use a handful of rows, so array-level validity,
  // child-level validity, and cumulative stride-offset bookkeeping are only ever
  // exercised within a single batch. Here 5000 rows > STANDARD_VECTOR_SIZE (2048)
  // forces the scan to emit multiple batches, while whole-array NULLs (i % 10 == 0)
  // and element-level NULLs (i % 7 == 0) recur in every batch, so all three
  // interact as the scan crosses vector boundaries.
  run_ok(
    "INSERT INTO arr_null_big SELECT i, "
    "CASE WHEN i % 10 = 0 THEN NULL "
    "WHEN i % 7 = 0 THEN [NULL, i + 1, NULL] "
    "ELSE [i, i + 1, i + 2] END "
    "FROM range(5000) t(i);");
  run_ok("CHECKPOINT;");
  compare_gpu_vs_cpu("SELECT id, a FROM arr_null_big;");
}

TEST_CASE_METHOD(ArrayFixture,
                 "gpu_execution array - NULL arrays survive an ORDER BY",
                 "[integration][gpu_execution][array][nulls]")
{
  // Regression: once a gather (ORDER BY) sorts a LIST column holding NULL arrays,
  // cudf compacts the NULL rows' children out, so the values child is no longer
  // fixed-stride. copy_array reads the LIST offsets to place each row's children
  // at their array_size slot; a fixed-stride read would shift elements across
  // rows. Mix whole-array NULLs with child-element NULLs so both validity levels
  // ride through the sort. Every other null-array test omits ORDER BY.
  run_ok("CREATE TABLE arr_null_sort (id INTEGER, a INTEGER[3]);");
  run_ok(
    "INSERT INTO arr_null_sort VALUES "
    "(1, [1, 2, 3]), (2, NULL), (3, [NULL, 5, NULL]), (4, [7, NULL, 9]), (5, NULL);");
  run_ok("CHECKPOINT;");
  // DESC so the sort's gather actually permutes the null rows out of input order.
  compare_gpu_vs_cpu_ordered("SELECT id, a FROM arr_null_sort ORDER BY id DESC;");
}

TEST_CASE_METHOD(ArrayFixture,
                 "gpu_execution array - NULL arrays survive an ORDER BY across scan batches",
                 "[integration][gpu_execution][array][nulls]")
{
  // Multi-batch regression for the compacted-child read-back. After ORDER BY
  // compacts NULL arrays' children out, a later 2048-row window with no NULL
  // arrays takes copy_array's fast path, but its element mask no longer starts
  // at row_offset*array_size, it starts at the compacted base (and unaligned).
  // Cluster whole-array NULLs early (ids 1000-1099) so the first window is
  // compacted (slow path) and later windows are dense (fast path reading from the
  // shifted base); element-level NULLs everywhere keep the child mask exercised.
  // 5000 rows > STANDARD_VECTOR_SIZE (2048) forces multiple scan batches.
  run_ok("CREATE TABLE arr_null_sort_big (id INTEGER, a INTEGER[3]);");
  run_ok(
    "INSERT INTO arr_null_sort_big SELECT i, "
    "CASE WHEN i BETWEEN 1000 AND 1099 THEN NULL "
    "WHEN i % 7 = 0 THEN [NULL, i + 1, NULL] "
    "ELSE [i, i + 1, i + 2] END "
    "FROM range(5000) t(i);");
  run_ok("CHECKPOINT;");
  compare_gpu_vs_cpu_ordered("SELECT id, a FROM arr_null_sort_big ORDER BY id;");
}

//===----------------------------------------------------------------------===//
// CONSTANT-compressed child data
//===----------------------------------------------------------------------===//

TEST_CASE_METHOD(ArrayFixture,
                 "gpu_execution array - CONSTANT-compressed child data",
                 "[integration][gpu_execution][array][codec]")
{
  run_ok("CREATE TABLE arr_const (id INTEGER, a INTEGER[3]);");
  // Every array element is the same value, so at CHECKPOINT DuckDB
  // CONSTANT-compresses the child data segment (path [col,1]), storing one
  // value in statistics instead of a data block. This is the only case that
  // drives the decoder's extract_constant_bytes() / constant_stats_for() branch
  // in stage_one_array_column; varied data lands UNCOMPRESSED and skips it.
  // 3000 rows (> STANDARD_VECTOR_SIZE) forces a persistent on-disk segment.
  run_ok("INSERT INTO arr_const SELECT i, [7, 7, 7] FROM range(3000) t(i);");
  run_ok("CHECKPOINT;");

  // Guard the test's own premise: confirm a segment of column `a` really is
  // CONSTANT-compressed. Without this, a future change in DuckDB's codec
  // selection could make this test silently pass without ever entering the
  // CONSTANT-child branch it exists to cover.
  auto storage = con->Query(
    "SELECT COUNT(*) FROM pragma_storage_info('arr_const') "
    "WHERE column_name = 'a' AND compression = 'Constant';");
  REQUIRE(storage);
  REQUIRE_FALSE(storage->HasError());
  REQUIRE(storage->GetValue(0, 0).GetValue<int64_t>() > 0);

  compare_gpu_vs_cpu("SELECT id, a FROM arr_const;");
}

TEST_CASE_METHOD(ArrayFixture,
                 "gpu_execution array - compressed child data codecs (RLE/BitPacking/ALP)",
                 "[integration][gpu_execution][array][codec]")
{
  // CONSTANT child data is covered above; UNCOMPRESSED is the default everywhere
  // else. This drives the array child onto the real compressed codecs the GPU
  // decode kernel supports, exercising stage_one_array_column's element-unit
  // segment_start bookkeeping against a non-trivial segment layout. Only the
  // kernel-decodable fixed-width codecs are exercised (RLE, BitPacking for ints;
  // ALP for floats), DICTIONARY is intentionally NOT here: the walker's coarse
  // viability gate accepts it but there is no fixed-width dictionary decode
  // kernel, so forcing it would route to the GPU and throw mid-scan. force_compression
  // makes the codec deterministic; pragma_storage_info guards the premise so a
  // future analyzer change can't silently skip the branch. The reset runs before
  // each guard, so even a failed REQUIRE leaves force_compression restored for
  // later tests on the shared connection.
  auto require_child_codec = [&](const std::string& table, const std::string& codec) {
    auto storage = con->Query("SELECT COUNT(*) FROM pragma_storage_info('" + table +
                              "') WHERE column_name = 'a' AND compression = '" + codec + "';");
    REQUIRE(storage);
    REQUIRE_FALSE(storage->HasError());
    UNSCOPED_INFO("expected child codec " << codec << " on table " << table);
    REQUIRE(storage->GetValue(0, 0).GetValue<int64_t>() > 0);
  };

  // RLE: long runs of equal child elements (300 consecutive equal values per
  // distinct key) but not constant within the segment.
  run_ok("SET force_compression = 'rle';");
  run_ok("CREATE TABLE arr_rle (id INTEGER, a INTEGER[3]);");
  run_ok("INSERT INTO arr_rle SELECT i, [i // 100, i // 100, i // 100] FROM range(5000) t(i);");
  run_ok("CHECKPOINT;");
  run_ok("RESET force_compression;");
  require_child_codec("arr_rle", "RLE");
  compare_gpu_vs_cpu("SELECT id, a FROM arr_rle;");

  // BitPacking: small-magnitude varied child values (< 1000, ~10 bits) that the
  // bit-packer can narrow below the full 32-bit width.
  run_ok("SET force_compression = 'bitpacking';");
  run_ok("CREATE TABLE arr_bp (id INTEGER, a INTEGER[3]);");
  run_ok(
    "INSERT INTO arr_bp SELECT i, [i % 1000, (i * 2) % 1000, (i * 3) % 1000] "
    "FROM range(5000) t(i);");
  run_ok("CHECKPOINT;");
  run_ok("RESET force_compression;");
  require_child_codec("arr_bp", "BitPacking");
  compare_gpu_vs_cpu("SELECT id, a FROM arr_bp;");

  // ALP: DuckDB's default float codec, so a FLOAT array child lands on it
  // naturally, no force needed. This is the codec a real embedding column hits,
  // and the case that motivated enabling ALP/ALPRD in is_supported_fixed_width_codec.
  run_ok("CREATE TABLE arr_alp (id INTEGER, a FLOAT[3]);");
  run_ok(
    "INSERT INTO arr_alp SELECT i, [i * 0.5, i * 0.25, i * -0.125]::FLOAT[3] "
    "FROM range(5000) t(i);");
  run_ok("CHECKPOINT;");
  // Accept ALP *or* ALPRD: both are DuckDB float codecs routed through the same
  // newly-enabled fixed-width gate and both decode on the GPU, so the premise is
  // "a real float codec, not UNCOMPRESSED/CONSTANT", pinning the analyzer's
  // exact choice would make this flake in CI if DuckDB picks the other one.
  auto alp_seg = con->Query(
    "SELECT COUNT(*) FROM pragma_storage_info('arr_alp') "
    "WHERE column_name = 'a' AND compression IN ('ALP', 'ALPRD');");
  REQUIRE(alp_seg);
  REQUIRE_FALSE(alp_seg->HasError());
  REQUIRE(alp_seg->GetValue(0, 0).GetValue<int64_t>() > 0);
  compare_gpu_vs_cpu("SELECT id, a FROM arr_alp;");
}

//===----------------------------------------------------------------------===//
// Array column flowing through a filter on a scalar key
//===----------------------------------------------------------------------===//

TEST_CASE_METHOD(ArrayFixture,
                 "gpu_execution array - array column survives a filter",
                 "[integration][gpu_execution][array][filter]")
{
  run_ok("CREATE TABLE arr_filter (id INTEGER, a INTEGER[2]);");
  run_ok("INSERT INTO arr_filter VALUES (1, [1, 1]), (2, [2, 2]), (3, [3, 3]), (4, [4, 4]);");
  run_ok("CHECKPOINT;");
  compare_gpu_vs_cpu("SELECT id, a FROM arr_filter WHERE id >= 3;");
}

TEST_CASE_METHOD(ArrayFixture,
                 "gpu_execution array - array projected while a filter prunes a row group",
                 "[integration][gpu_execution][array][filter]")
{
  run_ok("CREATE TABLE arr_prune (id INTEGER, a INTEGER[3]);");
  run_ok("INSERT INTO arr_prune SELECT i, [i, i + 1, i + 2] FROM range(200000) t(i);");
  run_ok("CHECKPOINT;");
  compare_gpu_vs_cpu("SELECT id, a FROM arr_prune WHERE id >= 150000;");
}

TEST_CASE_METHOD(ArrayFixture,
                 "gpu_execution array - fully pruned ARRAY scan through ORDER BY LIMIT",
                 "[integration][gpu_execution][array][filter]")
{
  // id >= 1000000 is FILTER_ALWAYS_FALSE for every row group, so the scan serves one
  // schema-correct 0-row split whose ARRAY column is a cuDF LIST. TOP-N then rebuilds an empty
  // output table from that view (compute_top_n_table -> duckdb::make_empty_like), which must
  // reproduce the LIST child hierarchy; the fixture's no-fallback delta turns a silent CPU
  // fallback into a failure. statistics_propagation must be off so the provably-false filter is
  // not folded into an EMPTY_RESULT before the scan exists (the standing Sirius mask's
  // late_materialization entry already keeps the ARRAY column inside TOP-N rather than joined
  // back after an id-only sort).
  run_ok("CREATE TABLE arr_prune (id INTEGER, a INTEGER[3]);");
  run_ok("INSERT INTO arr_prune SELECT i, [i, i + 1, i + 2] FROM range(3000) t(i);");
  run_ok("CHECKPOINT;");
  disabled_optimizers_guard const guard(*con, "statistics_propagation");
  // Non-empty control first: proves the non-empty ARRAY top-n path, so a failure below isolates
  // to emptiness rather than to ARRAY-through-TOP-N support.
  compare_gpu_vs_cpu_ordered("SELECT id, a FROM arr_prune ORDER BY id LIMIT 5;");
  compare_gpu_vs_cpu_ordered(
    "SELECT id, a FROM arr_prune WHERE id >= 1000000 ORDER BY id LIMIT 5;");
}

TEST_CASE_METHOD(ArrayFixture,
                 "gpu_execution array - join against a fully pruned ARRAY side",
                 "[integration][gpu_execution][array][filter][join]")
{
  // The fully pruned subquery serves one schema-correct 0-row LIST batch that must flow through
  // the hash join's empty-side handling on the GPU (the outer join NULL-pads each surviving row);
  // the no-fallback delta fails if any step of that pipeline falls back to CPU. The join's
  // orphan/absent-side synthesis (sirius::make_empty_table at the hash join's orphan branch) is
  // NOT reachable from SQL -- the partition slicer deposits a zero-row batch in every partition,
  // so a side that yields no row still deposits batches (see
  // test_compressed_materialization_partition.cpp); that helper's ARRAY handling is pinned by the
  // make_empty_table unit tests in test_cudf_utils.cpp instead. statistics_propagation must be
  // off so the provably-false filter is not folded into an EMPTY_RESULT before the scan exists.
  run_ok("CREATE TABLE arr_prune (id INTEGER, a INTEGER[3]);");
  run_ok("INSERT INTO arr_prune SELECT i, [i, i + 1, i + 2] FROM range(3000) t(i);");
  run_ok("CHECKPOINT;");
  run_ok("CREATE TABLE dim (id INTEGER);");
  run_ok("INSERT INTO dim SELECT i FROM range(100) t(i);");
  run_ok("CHECKPOINT;");
  disabled_optimizers_guard const guard(*con, "statistics_propagation");
  compare_gpu_vs_cpu(
    "SELECT d.id, t.a FROM dim d LEFT JOIN "
    "(SELECT * FROM arr_prune WHERE id >= 1000000) t USING (id);");
}

//===----------------------------------------------------------------------===//
// Batching: exceed a single DuckDB vector to span multiple scan batches
//===----------------------------------------------------------------------===//

TEST_CASE_METHOD(ArrayFixture,
                 "gpu_execution array - many rows span multiple scan batches",
                 "[integration][gpu_execution][array][scan]")
{
  run_ok("CREATE TABLE arr_big_n (id INTEGER, a INTEGER[3]);");
  // 5000 rows > STANDARD_VECTOR_SIZE (2048), forcing the scan to produce
  // multiple batches and exercising cumulative offset bookkeeping.
  run_ok("INSERT INTO arr_big_n SELECT i, [i, i + 1, i + 2] FROM range(5000) t(i);");
  run_ok("CHECKPOINT;");
  compare_gpu_vs_cpu("SELECT id, a FROM arr_big_n;");
}
