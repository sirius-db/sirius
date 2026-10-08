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

// Byte-range splits end to end: Substrait plans with ranged LocalFiles items run through the
// public sirius::ffi Context against a multi-row-group parquet file. Which plans may carry ranges
// is unit-tested in test/cpp/planner/test_substrait_scan_ranges.cpp; row-group ownership in
// test/cpp/scan/test_parquet_byte_range.cpp.

#include "from_substrait.hpp"
#include "sirius/ffi.hpp"
#include "utils/parquet_fixture_utils.hpp"

#include <catch.hpp>
#include <duckdb.hpp>
#include <duckdb/common/arrow/arrow.hpp>
#include <substrait/plan.pb.h>

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <numeric>
#include <source_location>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

constexpr std::int64_t kRows = 30000;

fs::path isolated_memory_config_path()
{
  std::source_location loc = std::source_location::current();
  return fs::path(loc.file_name()).parent_path().parent_path() / "scan" / "memory.yaml";
}

struct item {
  std::string path;
  std::uint64_t start  = 0;
  std::uint64_t length = 0;
};

void fill_read(substrait::Rel& rel, std::vector<item> const& items)
{
  auto* files = rel.mutable_read()->mutable_local_files();
  for (auto const& it : items) {
    auto* file = files->add_items();
    file->set_uri_file(it.path);
    file->mutable_parquet();
    file->set_start(it.start);
    file->set_length(it.length);
  }
}

std::string bytes(substrait::Plan const& plan)
{
  std::string out;
  REQUIRE(plan.SerializeToString(&out));
  return out;
}

std::string read_plan(std::vector<item> const& items)
{
  substrait::Plan plan;
  auto* root = plan.add_relations()->mutable_root();
  root->add_names("a");
  fill_read(*root->mutable_input(), items);
  return bytes(plan);
}

/// `a` = 0..kRows-1 in row groups of a few thousand rows, so byte ranges own different groups.
void write_multi_row_group_parquet(std::string const& path)
{
  sirius::test::scoped_sirius_disable disable;
  duckdb::DuckDB db(nullptr);
  duckdb::Connection con(db);
  auto copied =
    con.Query("COPY (SELECT range::BIGINT AS a FROM range(" + std::to_string(kRows) + ")) TO " +
              sirius::test::sql_literal(path) + " (FORMAT PARQUET, ROW_GROUP_SIZE 4096)");
  REQUIRE(copied);
  REQUIRE_FALSE(copied->HasError());
}

std::vector<std::int64_t> collect_sorted(ArrowArrayStream& stream)
{
  ArrowSchema schema{};
  REQUIRE(stream.get_schema(&stream, &schema) == 0);
  if (schema.release) { schema.release(&schema); }
  std::vector<std::int64_t> out;
  for (;;) {
    ArrowArray batch{};
    REQUIRE(stream.get_next(&stream, &batch) == 0);
    if (batch.release == nullptr) { break; }
    auto const* col  = batch.n_children >= 1 ? batch.children[0] : &batch;
    auto const* data = static_cast<std::int64_t const*>(col->buffers[1]);
    for (std::int64_t i = 0; i < col->length; ++i) {
      out.push_back(data[i + col->offset]);
    }
    batch.release(&batch);
  }
  if (stream.release) { stream.release(&stream); }
  std::sort(out.begin(), out.end());
  return out;
}

struct fixture {
  sirius::test::scratch_dir scratch{"ffi_byte_ranges"};
  std::string path   = scratch.file("many.parquet");
  std::string other  = scratch.file("other.parquet");
  std::uint64_t size = 0;
  std::unique_ptr<sirius::ffi::Context> ctx;

  fixture()
  {
    write_multi_row_group_parquet(path);
    write_multi_row_group_parquet(other);
    size = fs::file_size(path);
    ctx  = sirius::ffi::make_context_from_config(isolated_memory_config_path().string());
  }

  std::vector<std::int64_t> run(std::string const& plan)
  {
    ArrowArrayStream stream{};
    ctx->execute_substrait(plan, reinterpret_cast<std::uintptr_t>(&stream));
    return collect_sorted(stream);
  }

  static std::vector<std::int64_t> all_rows()
  {
    std::vector<std::int64_t> rows(kRows);
    std::iota(rows.begin(), rows.end(), 0);
    return rows;
  }
};

}  // namespace

TEST_CASE("FFI byte ranges: splits of a file read every row exactly once",
          "[isolated_context][sirius_ffi][parquet_byte_range]")
{
  fixture f;
  auto const third  = f.size / 3;
  auto const first  = f.run(read_plan({{f.path, 0, third}}));
  auto const middle = f.run(read_plan({{f.path, third, third}}));
  auto const last   = f.run(read_plan({{f.path, 2 * third, f.size - 2 * third}}));
  REQUIRE_FALSE(first.empty());
  REQUIRE_FALSE(last.empty());

  std::vector<std::int64_t> all;
  for (auto const* part : {&first, &middle, &last}) {
    all.insert(all.end(), part->begin(), part->end());
  }
  std::sort(all.begin(), all.end());
  REQUIRE(all == fixture::all_rows());

  // One instance owning two non-adjacent splits reads exactly those two.
  auto outer = first;
  outer.insert(outer.end(), last.begin(), last.end());
  std::sort(outer.begin(), outer.end());
  REQUIRE(f.run(read_plan({{f.path, 0, third}, {f.path, 2 * third, f.size - 2 * third}})) == outer);
}

TEST_CASE("FFI byte ranges: a scan mixing a whole file and a ranged file",
          "[isolated_context][sirius_ffi][parquet_byte_range]")
{
  fixture f;
  auto const half = f.size / 2;
  auto expected   = fixture::all_rows();
  auto const left = f.run(read_plan({{f.path, 0, half}}));
  expected.insert(expected.end(), left.begin(), left.end());
  std::sort(expected.begin(), expected.end());
  REQUIRE(f.run(read_plan({{f.other}, {f.path, 0, half}})) == expected);
}

TEST_CASE("FFI byte ranges: a ranged file read twice in one plan is refused",
          "[isolated_context][sirius_ffi][parquet_byte_range]")
{
  fixture f;
  auto const half = f.size / 2;

  // An unreferenced relation holds the other split. Before: that read was never bound, so the
  // root's read claimed both splits and read rows that belong to the other one, with no error.
  substrait::Plan plan;
  fill_read(*plan.add_relations()->mutable_rel(), {{f.path, half, f.size - half}});
  auto* root = plan.add_relations()->mutable_root();
  root->add_names("a");
  fill_read(*root->mutable_input(), {{f.path, 0, half}});
  REQUIRE_THROWS_WITH(f.run(bytes(plan)), Catch::Matchers::ContainsSubstring("reads it 2 times"));
}

TEST_CASE("FFI byte ranges: malformed ranges fail loudly instead of reading too little",
          "[isolated_context][sirius_ffi][parquet_byte_range]")
{
  fixture f;
  // Before: (0,0) was dropped, (10,5) owns no row group, and the read returned no rows.
  REQUIRE_THROWS_WITH(f.run(read_plan({{f.path, 0, 0}, {f.path, 10, 5}})),
                      Catch::Matchers::ContainsSubstring("both whole and by byte range"));
  // Before: start + length wrapped to 3 and the read returned no rows.
  REQUIRE_THROWS_WITH(f.run(read_plan({{f.path, 4, std::numeric_limits<std::uint64_t>::max()}})),
                      Catch::Matchers::ContainsSubstring("overflows"));
  // Before: a split sized for another file was read as if it fit.
  REQUIRE_THROWS_WITH(f.run(read_plan({{f.path, 0, f.size + 100}})),
                      Catch::Matchers::ContainsSubstring("ends past the file"));
}

TEST_CASE("FFI byte ranges: a plan without ranges is not refused for its rel types",
          "[isolated_context][sirius_ffi][parquet_byte_range]")
{
  fixture f;
  // Before: range extraction threw on the ReferenceRel even though nothing was ranged.
  substrait::Plan plan;
  fill_read(*plan.add_relations()->mutable_rel(), {{f.path}});
  auto* root = plan.add_relations()->mutable_root();
  root->add_names("a");
  root->mutable_input()->mutable_reference()->set_subtree_ordinal(0);
  REQUIRE(f.run(bytes(plan)) == fixture::all_rows());
}

// Known issue, not fixed here: DuckDB's Substrait consumer lists a file once per LocalFiles item
// and parquet_scan estimates each entry at the whole file's rows, so a scan owning half a file
// through two ranges is estimated at twice the file. The estimate feeds join build-side choice.
// [!shouldfail] keeps CI green and flags the day it passes.
TEST_CASE("FFI byte ranges: a ranged scan is not estimated above the rows it can own",
          "[sirius_ffi][parquet_byte_range][!shouldfail]")
{
  sirius::test::scratch_dir scratch("ffi_byte_ranges_estimate");
  auto const path = scratch.file("many.parquet");
  write_multi_row_group_parquet(path);
  auto const size = fs::file_size(path);

  sirius::test::scoped_sirius_disable disable;
  duckdb::DuckDB db(nullptr);
  duckdb::Connection con(db);
  REQUIRE_FALSE(con.Query("SET parquet_metadata_cache = true")->HasError());
  REQUIRE_FALSE(con.Query("SELECT count(*) FROM " + sirius::test::sql_literal(path))->HasError());

  duckdb::SubstraitToDuckDB transformer(
    con.context, read_plan({{path, 0, size / 4}, {path, size / 2, size / 4}}), /*json=*/false);
  auto relation = transformer.TransformPlan();
  auto explained =
    relation->Explain(duckdb::ExplainType::EXPLAIN_STANDARD, duckdb::ExplainFormat::JSON);
  REQUIRE_FALSE(explained->HasError());
  auto const text = explained->ToString();
  INFO(text);
  auto const key = std::string("\"Estimated Cardinality\": \"");
  auto const at  = text.find(key);
  REQUIRE(at != std::string::npos);
  // Half the file's bytes are owned, so no more than the file's rows can come back.
  REQUIRE(std::stoll(text.substr(at + key.size())) <= kRows);
}
