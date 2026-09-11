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

// `CALL pin_table('x.hpln', format => 'simpatico', tier => 'host', name => 't')`, end to end:
// write a file with COPY, pin it, query it through read_simpatico().
//
// The thing worth testing here is not that the rows are right -- a query that ignored the pinned
// entry entirely and re-read the file would return exactly the same rows. It is that the query is
// SERVED from the entry, that the entry holds every chunk of the file exactly once, and that the
// file's own zone maps became the entry's bounds so a pinned scan still prunes. Each of those is
// asserted against something that would differ if it were false: the file is rewritten with
// different values after the pin, so a cache miss returns the NEW values; per-chunk counts come
// from a GROUP BY over the chunk-encoding key; and pruning is probed on the live entry with
// build_cached_scan_plan, because a query that prunes nothing still returns the right answer.

#include "compression/simpatico_file_ingest.hpp"
#include "sirius_context.hpp"

#include <catch.hpp>
#include <duckdb.hpp>
#include <duckdb/planner/filter/constant_filter.hpp>
#include <duckdb/planner/table_filter.hpp>
#include <scan_manager/sirius_scan_manager.hpp>
#include <unistd.h>
#include <utils/gpu_execution_fixture.hpp>

#include <cstdint>
#include <filesystem>
#include <string>

namespace {

namespace fs = std::filesystem;

/// 6000 rows at 2048 rows per chunk is three chunks, the last one short -- so a walk that assumed
/// equal chunks, or dropped the tail, is visible in a per-chunk count.
constexpr int kRows       = 6000;
constexpr int kChunkRows  = 2048;
constexpr int kGroupRows  = 1024;
constexpr int kChunkCount = 3;

sirius::scan_manager::pinned_entry const* find_entry(duckdb::SiriusContext& ctx,
                                                     std::string_view entry_name)
{
  sirius::scan_manager::pinned_entry const* found = nullptr;
  ctx.get_scan_manager().visit_pinned_entries(
    [&found, entry_name](std::string_view name, auto const& e) {
      if (name == entry_name) {
        found = &e;
        return true;
      }
      return false;
    });
  return found;
}

/// Chunks the entry's own bounds prove empty for `k >= lo`, asked of the live entry.
///
/// Result equality cannot answer this: a pinned scan that prunes nothing returns the same rows.
std::size_t probe_pruned_count(sirius::scan_manager::pinned_entry const& entry, std::int32_t lo)
{
  duckdb::TableFilterSet filters;
  filters.filters[0] = duckdb::make_uniq<duckdb::ConstantFilter>(
    duckdb::ExpressionType::COMPARE_GREATERTHANOREQUALTO, duckdb::Value::INTEGER(lo));
  duckdb::vector<duckdb::ColumnIndex> query_columns;
  query_columns.emplace_back(duckdb::ColumnIndex(0));
  return sirius::scan_manager::build_cached_scan_plan(entry, &filters, &query_columns).pruned;
}

class PinSimpaticoFixture : public sirius::test::GpuExecutionFixture {
 public:
  PinSimpaticoFixture()
  {
    dir = fs::temp_directory_path() /
          ("sirius_pin_hpln_" + std::to_string(::getpid()) + "_" + std::to_string(::rand()));
    fs::create_directories(dir);
    run_ok("SET gpu_execution = true;");
    // A GPU failure over a .hpln would otherwise be replayed on the CPU and surface as the table
    // function's "no CPU reader" error, hiding the real cause.
    run_ok("SET enable_duckdb_fallback = false;");
  }

  ~PinSimpaticoFixture()
  {
    if (con) { con->Query("CALL unpin_table('" + entry_name() + "');"); }
    fs::remove_all(dir);
  }

  [[nodiscard]] std::string entry_name() const { return "hpln_pin"; }
  [[nodiscard]] std::string path(std::string const& name) const { return (dir / name).string(); }

  /// Write a .hpln at @p file whose `v` column is `k * @p multiplier`, so two files of the same
  /// shape differ in every value of one column.
  void write_file(std::string const& file, int multiplier, int group_rows = kGroupRows)
  {
    query("COPY (SELECT i::INTEGER AS k, (i * " + std::to_string(multiplier) +
          ")::INTEGER AS v FROM range(" + std::to_string(kRows) + ") t(i) ORDER BY k) TO '" + file +
          "' (FORMAT simpatico, chunk_rows " + std::to_string(kChunkRows) + ", group_rows " +
          std::to_string(group_rows) + ");");
  }

  duckdb::unique_ptr<duckdb::MaterializedQueryResult> query(std::string const& sql)
  {
    auto result = con->Query(sql);
    REQUIRE(result);
    if (result->HasError()) { UNSCOPED_INFO("query error: " << result->GetError()); }
    REQUIRE_FALSE(result->HasError());
    return duckdb::unique_ptr<duckdb::MaterializedQueryResult>(
      static_cast<duckdb::MaterializedQueryResult*>(result.release()));
  }

  std::string query_error(std::string const& sql)
  {
    auto result = con->Query(sql);
    REQUIRE(result);
    REQUIRE(result->HasError());
    return result->GetError();
  }

  void pin(std::string const& file)
  {
    query("CALL pin_table('" + file + "', format => 'simpatico', tier => 'host', name => '" +
          entry_name() + "');");
  }

  [[nodiscard]] sirius::scan_manager::pinned_entry const& entry()
  {
    auto ctx = con->context->registered_state->Get<duckdb::SiriusContext>("sirius_state");
    REQUIRE(ctx);
    auto const* e = find_entry(*ctx, entry_name());
    REQUIRE(e != nullptr);
    return *e;
  }

  [[nodiscard]] std::int64_t sum_v(std::string const& file)
  {
    auto r = query("SELECT sum(v) FROM read_simpatico('" + file + "');");
    return r->GetValue(0, 0).GetValue<std::int64_t>();
  }

  fs::path dir;
};

/// sum(k * multiplier) over k in [0, kRows).
std::int64_t expected_sum(int multiplier)
{
  std::int64_t total = 0;
  for (int i = 0; i < kRows; ++i) {
    total += static_cast<std::int64_t>(i) * multiplier;
  }
  return total;
}

}  // namespace

TEST_CASE_METHOD(PinSimpaticoFixture,
                 "pin_table ingests a .hpln whole, and a read_simpatico query serves from it",
                 "[integration][pin_table][pin_simpatico]")
{
  auto const file = path("pinned.hpln");
  write_file(file, /*multiplier=*/2);
  pin(file);

  // Every chunk of the file, exactly once: a dropped chunk or a duplicated one changes the row
  // count and the entry's chunk count, and neither is visible in a SELECT that happens to sum to
  // the same thing.
  auto const& e = entry();
  REQUIRE(e.tier == cucascade::memory::Tier::HOST);
  REQUIRE(e.host_chunks.size() == kChunkCount);
  REQUIRE(e.num_rows == static_cast<std::size_t>(kRows));
  REQUIRE(e.cache_info.column_names() == std::vector<std::string>{"k", "v"});

  // The file's bounds reached the entry -- one arena cell per (column, chunk) at the file's own
  // group resolution, plus the chunk-level sidecar reduced from it. Without the coarse one
  // nothing prunes at all, so both are load-bearing.
  REQUIRE(e.group_bounds.chunk_count() == kChunkCount);
  REQUIRE(e.group_bounds.group_rows() == kGroupRows);
  REQUIRE(e.zone_maps.has_stats());
  REQUIRE(probe_pruned_count(e, 2 * kChunkRows) == kChunkCount - 1);

  REQUIRE(sum_v(file) == expected_sum(2));

  // The proof that the query came from the ENTRY and not from the file: rewrite the file in place
  // with different values. The bind still reads the (new) file, so a scan that re-read it would
  // return the new sum; a scan served from the pinned chunks returns the old one.
  write_file(file, /*multiplier=*/5);
  REQUIRE(sum_v(file) == expected_sum(2));

  // ... and the control on that control: with the entry gone, the same query over the same path
  // returns the NEW values, so the assertion above was not passing because the rewrite silently
  // failed.
  query("CALL unpin_table('" + entry_name() + "');");
  REQUIRE(sum_v(file) == expected_sum(5));
}

TEST_CASE_METHOD(PinSimpaticoFixture,
                 "a pinned .hpln serves NULLs, and does not prune a group holding one",
                 "[integration][pin_table][pin_simpatico]")
{
  auto const file = path("nullable.hpln");
  // Nulls SCATTERED inside their groups, not filling them. A group that is entirely null has no
  // bounds at all, and an absent cell never prunes -- so it would pass this test whatever the
  // reader believed about nulls. A group that holds both values and nulls has real bounds, which
  // is what makes `column_has_no_nulls` the thing deciding whether `v IS NULL` can prune it.
  query(
    "COPY (SELECT i::INTEGER AS k,"
    " CASE WHEN i BETWEEN 3072 AND 4095 AND i % 3 = 0 THEN NULL ELSE i * 2 END::INTEGER AS v"
    " FROM range(" +
    std::to_string(kRows) + ") t(i) ORDER BY k) TO '" + file + "' (FORMAT simpatico, chunk_rows " +
    std::to_string(kChunkRows) + ", group_rows " + std::to_string(kGroupRows) + ");");
  pin(file);

  auto const& e = entry();
  REQUIRE(e.num_rows == static_cast<std::size_t>(kRows));
  REQUIRE(e.host_chunks.size() == kChunkCount);

  std::int64_t null_rows    = 0;
  std::int64_t null_key_sum = 0;
  for (int i = 3072; i <= 4095; ++i) {
    if (i % 3 != 0) { continue; }
    ++null_rows;
    null_key_sum += i;
  }
  REQUIRE(null_rows > 0);

  // Every row is served, NULLs included: count(*) sees them, count(v) does not.
  auto all = query("SELECT count(*), count(v) FROM read_simpatico('" + file + "');");
  REQUIRE(all->GetValue(0, 0).GetValue<std::int64_t>() == kRows);
  REQUIRE(all->GetValue(1, 0).GetValue<std::int64_t>() == kRows - null_rows);

  // The load-bearing one. `v IS NULL` prunes a group exactly when the column is known to have no
  // nulls, so a reader that lost that fact prunes EVERY group and this returns nothing. Summing a
  // never-null key over the survivors pins which rows came back, not merely how many.
  auto where =
    query("SELECT count(*), sum(k) FROM read_simpatico('" + file + "') WHERE v IS NULL;");
  REQUIRE(where->GetValue(0, 0).GetValue<std::int64_t>() == null_rows);
  REQUIRE(where->GetValue(1, 0).GetValue<std::int64_t>() == null_key_sum);

  // The complement, over the same groups: a NULL satisfies neither side of a range split, so the
  // two halves account for every non-null row and no more.
  auto lo = query("SELECT count(*) FROM read_simpatico('" + file + "') WHERE v < 6144;");
  auto hi = query("SELECT count(*) FROM read_simpatico('" + file + "') WHERE v >= 6144;");
  REQUIRE(lo->GetValue(0, 0).GetValue<std::int64_t>() +
            hi->GetValue(0, 0).GetValue<std::int64_t>() ==
          kRows - null_rows);
}

TEST_CASE_METHOD(PinSimpaticoFixture,
                 "a pinned .hpln serves every chunk's rows exactly once",
                 "[integration][pin_table][pin_simpatico]")
{
  auto const file = path("chunks.hpln");
  write_file(file, /*multiplier=*/2);
  pin(file);
  // The file is rewritten with different values FIRST, so every value below can only have come
  // from the pinned chunks.
  write_file(file, /*multiplier=*/5);

  auto rows = query("SELECT k // " + std::to_string(kChunkRows) +
                    " AS chunk, count(*), min(k), max(k), sum(v) FROM read_simpatico('" + file +
                    "') GROUP BY 1 ORDER BY 1;");
  REQUIRE(rows->RowCount() == kChunkCount);
  for (int c = 0; c < kChunkCount; ++c) {
    auto const row      = static_cast<duckdb::idx_t>(c);
    auto const first    = c * kChunkRows;
    auto const last     = std::min(kRows, (c + 1) * kChunkRows) - 1;
    std::int64_t sum_of = 0;
    for (int i = first; i <= last; ++i) {
      sum_of += static_cast<std::int64_t>(i) * 2;
    }
    REQUIRE(rows->GetValue(0, row).GetValue<std::int32_t>() == c);
    REQUIRE(rows->GetValue(1, row).GetValue<std::int64_t>() == last - first + 1);
    REQUIRE(rows->GetValue(2, row).GetValue<std::int32_t>() == first);
    REQUIRE(rows->GetValue(3, row).GetValue<std::int32_t>() == last);
    REQUIRE(rows->GetValue(4, row).GetValue<std::int64_t>() == sum_of);
  }
}

TEST_CASE_METHOD(PinSimpaticoFixture,
                 "a pinned .hpln answers filtered queries exactly, pruning or not",
                 "[integration][pin_table][pin_simpatico]")
{
  auto const file = path("filtered.hpln");
  write_file(file, /*multiplier=*/2);
  pin(file);
  write_file(file, /*multiplier=*/5);

  // Selective: two of three chunks are provably empty, and the survivor is narrowed further by
  // the per-group bounds -- neither of which may cost a matching row.
  auto sel = query("SELECT count(*), sum(v) FROM read_simpatico('" + file + "') WHERE k >= 5000;");
  std::int64_t expected_count = 0, expected_v = 0;
  for (int i = 5000; i < kRows; ++i) {
    ++expected_count;
    expected_v += static_cast<std::int64_t>(i) * 2;
  }
  REQUIRE(sel->GetValue(0, 0).GetValue<std::int64_t>() == expected_count);
  REQUIRE(sel->GetValue(1, 0).GetValue<std::int64_t>() == expected_v);

  // Selects nothing: every chunk prunes, and the sentinel chunk keeps the pipeline alive rather
  // than leaving a zero-batch scan to hang.
  auto none = query("SELECT count(*) FROM read_simpatico('" + file + "') WHERE k >= 1000000;");
  REQUIRE(none->GetValue(0, 0).GetValue<std::int64_t>() == 0);

  auto rows = query("SELECT k FROM read_simpatico('" + file + "') WHERE k < 0;");
  REQUIRE(rows->RowCount() == 0);
}

TEST_CASE_METHOD(PinSimpaticoFixture,
                 "a .hpln written without zone maps pins and serves, but prunes nothing",
                 "[integration][pin_table][pin_simpatico]")
{
  // The negative control for the bounds path: same data, same pin, only the file's `zone_maps`
  // segment missing. Everything must still be correct -- absent statistics mean "serve unpruned",
  // never a wrong answer -- and the pruning probe must report exactly zero, which is what makes
  // the positive case above evidence rather than coincidence.
  auto const file = path("no_bounds.hpln");
  write_file(file, /*multiplier=*/2, /*group_rows=*/0);
  REQUIRE(sirius::read_hpln_schema(file).group_bounds.empty());

  pin(file);
  auto const& e = entry();
  REQUIRE(e.host_chunks.size() == kChunkCount);
  REQUIRE(e.group_bounds.empty());
  REQUIRE_FALSE(e.zone_maps.has_stats());
  REQUIRE(probe_pruned_count(e, 2 * kChunkRows) == 0);

  write_file(file, /*multiplier=*/5);
  REQUIRE(sum_v(file) == expected_sum(2));
}

TEST_CASE_METHOD(PinSimpaticoFixture,
                 "pin_table refuses the .hpln pins it cannot serve correctly",
                 "[integration][pin_table][pin_simpatico]")
{
  auto const file = path("refused.hpln");
  write_file(file, /*multiplier=*/2);

  // A compressed chunk holds every column of the file, so a subset pin would store all of them
  // and describe only some -- and the serve-time projection indexes the chunk by the ENTRY's
  // column position, which would then name the wrong column.
  auto const cols_error =
    query_error("CALL pin_table('" + file + "', format => 'simpatico', tier => 'host', name => '" +
                entry_name() + "', cols => ['k']);");
  REQUIRE(cols_error.find("cols") != std::string::npos);

  // The file IS the host representation; there is no GPU-tier form of it to pin without decoding.
  auto const tier_error =
    query_error("CALL pin_table('" + file + "', format => 'simpatico', tier => 'gpu', name => '" +
                entry_name() + "');");
  REQUIRE(tier_error.find("host") != std::string::npos);

  // The extension is inferred, so the common spelling needs no format argument at all.
  query("CALL pin_table('" + file + "', tier => 'host', name => '" + entry_name() + "');");
  REQUIRE(entry().host_chunks.size() == kChunkCount);
}
