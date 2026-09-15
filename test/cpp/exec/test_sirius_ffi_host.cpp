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

// Public Host methods only. Builds Substrait in the test because the FFI has no SQL helper.
// Covers a result fragment, a relay_from chain, the one-window-at-a-time rule, and drop after
// build(). Spec errors and failed-build rollback live in test_streaming_fragment.cpp and
// test_sirius_ffi_fragment.cpp.

#include "sirius/exception.hpp"
#include "sirius_ffi.hpp"
#include "utils/parquet_fixture_utils.hpp"

#include <catch.hpp>
#include <duckdb.hpp>
#include <duckdb/common/arrow/arrow.hpp>
#include <substrait/plan.pb.h>

#include <cstdint>
#include <filesystem>
#include <memory>
#include <source_location>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

fs::path isolated_memory_config_path()
{
  std::source_location loc = std::source_location::current();
  return fs::path(loc.file_name()).parent_path().parent_path() / "scan" / "memory.yaml";
}

std::string serialize_plan(substrait::Plan const& plan)
{
  std::string bytes;
  REQUIRE(plan.SerializeToString(&bytes));
  return bytes;
}

std::string local_files_plan(std::string const& path)
{
  substrait::Plan plan;
  auto* root = plan.add_relations()->mutable_root();
  root->add_names("a");
  auto* item = root->mutable_input()->mutable_read()->mutable_local_files()->add_items();
  item->set_uri_file(path);
  item->mutable_parquet();
  return serialize_plan(plan);
}

std::string stream_read_plan(std::uint64_t stream_id)
{
  substrait::Plan plan;
  auto* root = plan.add_relations()->mutable_root();
  root->add_names("a");
  auto* read = root->mutable_input()->mutable_read();
  read->mutable_named_table()->add_names(*sirius::ffi::stream_view_name(stream_id));
  auto* schema = read->mutable_base_schema();
  schema->add_names("a");
  auto* st = schema->mutable_struct_();
  st->set_nullability(::substrait::Type_Nullability_NULLABILITY_REQUIRED);
  st->add_types()->mutable_i64()->set_nullability(
    ::substrait::Type_Nullability_NULLABILITY_NULLABLE);
  return serialize_plan(plan);
}

void write_ids_parquet(std::string const& path)
{
  sirius::test::scoped_sirius_disable disable;
  duckdb::DuckDB db(nullptr);
  duckdb::Connection con(db);
  auto copied = con.Query(
    "COPY (SELECT * FROM (VALUES (1::BIGINT), (2::BIGINT), (3::BIGINT), "
    "(4::BIGINT), (5::BIGINT)) t(a)) TO " +
    sirius::test::sql_literal(path) + " (FORMAT PARQUET)");
  REQUIRE(copied);
  REQUIRE_FALSE(copied->HasError());
}

void release_if(ArrowArrayStream& stream)
{
  if (stream.release) { stream.release(&stream); }
}

const ArrowArray* first_column(ArrowArray const& batch)
{
  if (batch.n_children >= 1 && batch.children != nullptr && batch.children[0] != nullptr) {
    return batch.children[0];
  }
  return &batch;
}

std::vector<std::int64_t> collect_i64_column(ArrowArrayStream& stream)
{
  ArrowSchema schema{};
  REQUIRE(stream.get_schema != nullptr);
  REQUIRE(stream.get_schema(&stream, &schema) == 0);
  if (schema.release) { schema.release(&schema); }

  std::vector<std::int64_t> out;
  for (;;) {
    ArrowArray batch{};
    REQUIRE(stream.get_next(&stream, &batch) == 0);
    if (batch.release == nullptr) { break; }
    auto const* col = first_column(batch);
    REQUIRE(col->n_buffers >= 2);
    REQUIRE(col->buffers[1] != nullptr);
    auto const* data = static_cast<std::int64_t const*>(col->buffers[1]);
    for (std::int64_t i = 0; i < col->length; ++i) {
      out.push_back(data[i + col->offset]);
    }
    batch.release(&batch);
  }
  release_if(stream);
  return out;
}

std::vector<std::int64_t> result_i64s(sirius::ffi::Fragment& fragment)
{
  ArrowArrayStream stream{};
  fragment.result_to_arrow(reinterpret_cast<std::uintptr_t>(&stream));
  return collect_i64_column(stream);
}

}  // namespace

TEST_CASE("FFI leaf result_to_arrow returns parquet rows", "[isolated_context][sirius_ffi]")
{
  sirius::test::scratch_dir scratch("ffi_host_leaf");
  auto const path = scratch.file("ids.parquet");
  write_ids_parquet(path);

  auto ctx    = sirius::ffi::make_context_from_config(isolated_memory_config_path().string());
  auto result = sirius::ffi::make_fragment(*ctx);
  result->build(local_files_plan(path));
  result->run();
  REQUIRE(result_i64s(*result) == std::vector<std::int64_t>{1, 2, 3, 4, 5});
}

TEST_CASE("FFI relay_from chain matches a single-fragment parquet scan",
          "[isolated_context][sirius_ffi]")
{
  sirius::test::scratch_dir scratch("ffi_host_relay");
  auto const path = scratch.file("ids.parquet");
  write_ids_parquet(path);

  auto ctx    = sirius::ffi::make_context_from_config(isolated_memory_config_path().string());
  auto sender = sirius::ffi::make_fragment(*ctx);
  sender->declare_output(0);
  sender->build(local_files_plan(path));
  sender->run();

  auto receiver = sirius::ffi::make_fragment(*ctx);
  receiver->declare_input_column(0, "a", "BIGINT");
  receiver->build(stream_read_plan(0));
  REQUIRE(receiver->relay_from(*sender, 0, 0, 0) > 0);
  receiver->run();
  REQUIRE(result_i64s(*receiver) == std::vector<std::int64_t>{1, 2, 3, 4, 5});
}

TEST_CASE("FFI only one fragment may sit between build() and run()",
          "[isolated_context][sirius_ffi]")
{
  sirius::test::scratch_dir scratch("ffi_host_nested_window");
  auto const path = scratch.file("ids.parquet");
  write_ids_parquet(path);

  auto ctx    = sirius::ffi::make_context_from_config(isolated_memory_config_path().string());
  auto sender = sirius::ffi::make_fragment(*ctx);
  sender->declare_output(0);
  sender->build(local_files_plan(path));

  auto receiver = sirius::ffi::make_fragment(*ctx);
  receiver->declare_input_column(0, "a", "BIGINT");
  REQUIRE_THROWS(receiver->build(stream_read_plan(0)));
}

TEST_CASE("FFI drop after build releases the query window", "[isolated_context][sirius_ffi]")
{
  sirius::test::scratch_dir scratch("ffi_host_drop");
  auto const path = scratch.file("ids.parquet");
  write_ids_parquet(path);
  auto const plan = local_files_plan(path);

  auto ctx = sirius::ffi::make_context_from_config(isolated_memory_config_path().string());
  {
    auto abandoned = sirius::ffi::make_fragment(*ctx);
    abandoned->declare_output(0);
    abandoned->build(plan);
  }

  auto next = sirius::ffi::make_fragment(*ctx);
  next->declare_output(0);
  next->build(plan);
  next->run();
  REQUIRE(next->output_batch_count(0) > 0);
}
