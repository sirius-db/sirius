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

// Public sirius::ffi Context and Fragment methods only.
// Builds Substrait in the test because the FFI has no SQL helper.
// Covers a result fragment, a relay_from chain, several fragments built before any runs (on one
// and on two threads), build() and run()/drop on different threads, drop after build(), a build()
// that fails after setup, and execute_substrait. Spec errors and failed-build rollback live in
// test_streaming_fragment.cpp and test_sirius_ffi_fragment.cpp.

#include "sirius/exception.hpp"
#include "sirius/ffi.hpp"
#include "utils/parquet_fixture_utils.hpp"

#include <catch.hpp>
#include <duckdb.hpp>
#include <duckdb/common/arrow/arrow.hpp>
#include <substrait/plan.pb.h>

#include <cstdint>
#include <filesystem>
#include <future>
#include <memory>
#include <source_location>
#include <string>
#include <thread>
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
  sirius::test::scratch_dir scratch("ffi_embedder_leaf");
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
  sirius::test::scratch_dir scratch("ffi_embedder_relay");
  auto const path = scratch.file("ids.parquet");
  write_ids_parquet(path);

  auto ctx    = sirius::ffi::make_context_from_config(isolated_memory_config_path().string());
  auto sender = sirius::ffi::make_fragment(*ctx);
  sender->declare_output(0);
  sender->build(local_files_plan(path));
  sender->run();
  REQUIRE(sender->output_row_count(0) == 5);

  auto receiver = sirius::ffi::make_fragment(*ctx);
  receiver->declare_input_column(0, "a", "BIGINT");
  receiver->declare_input_cardinality(0, sender->output_row_count(0));
  receiver->build(stream_read_plan(0));
  REQUIRE(receiver->relay_from(*sender, 0, 0, 0) > 0);
  receiver->run();
  REQUIRE(result_i64s(*receiver) == std::vector<std::int64_t>{1, 2, 3, 4, 5});
}

TEST_CASE("FFI several fragments may be built before any of them runs",
          "[isolated_context][sirius_ffi]")
{
  sirius::test::scratch_dir scratch("ffi_embedder_build_before_run");
  auto const path = scratch.file("ids.parquet");
  write_ids_parquet(path);

  auto ctx    = sirius::ffi::make_context_from_config(isolated_memory_config_path().string());
  auto sender = sirius::ffi::make_fragment(*ctx);
  sender->declare_output(0);
  sender->build(local_files_plan(path));

  // build() holds no query window, so the receiver builds while the sender is still unrun.
  auto receiver = sirius::ffi::make_fragment(*ctx);
  receiver->declare_input_column(0, "a", "BIGINT");
  receiver->build(stream_read_plan(0));

  sender->run();
  REQUIRE(receiver->relay_from(*sender, 0, 0, 0) > 0);
  receiver->run();
  REQUIRE(result_i64s(*receiver) == std::vector<std::int64_t>{1, 2, 3, 4, 5});
}

TEST_CASE("FFI build() on another thread proceeds while a fragment is built but not run",
          "[isolated_context][sirius_ffi]")
{
  sirius::test::scratch_dir scratch("ffi_embedder_cross_thread_build");
  auto const path = scratch.file("ids.parquet");
  write_ids_parquet(path);

  auto ctx    = sirius::ffi::make_context_from_config(isolated_memory_config_path().string());
  auto sender = sirius::ffi::make_fragment(*ctx);
  sender->declare_output(0);
  sender->build(local_files_plan(path));

  std::unique_ptr<sirius::ffi::Fragment> receiver;
  std::thread([&] {
    receiver = sirius::ffi::make_fragment(*ctx);
    receiver->declare_input_column(0, "a", "BIGINT");
    receiver->build(stream_read_plan(0));
  }).join();

  sender->run();
  REQUIRE(receiver->relay_from(*sender, 0, 0, 0) > 0);
  receiver->run();
  REQUIRE(result_i64s(*receiver) == std::vector<std::int64_t>{1, 2, 3, 4, 5});
}

TEST_CASE("FFI build() and run() may happen on different threads", "[isolated_context][sirius_ffi]")
{
  sirius::test::scratch_dir scratch("ffi_embedder_run_other_thread");
  auto const path = scratch.file("ids.parquet");
  write_ids_parquet(path);
  auto const plan = local_files_plan(path);

  auto ctx = sirius::ffi::make_context_from_config(isolated_memory_config_path().string());
  auto ran = sirius::ffi::make_fragment(*ctx);
  ran->build(plan);
  std::thread([&] { ran->run(); }).join();
  REQUIRE(result_i64s(*ran) == std::vector<std::int64_t>{1, 2, 3, 4, 5});

  auto dropped = sirius::ffi::make_fragment(*ctx);
  dropped->build(plan);
  std::thread([&] { dropped.reset(); }).join();

  auto next = sirius::ffi::make_fragment(*ctx);
  next->build(plan);
  next->run();
  REQUIRE(result_i64s(*next) == std::vector<std::int64_t>{1, 2, 3, 4, 5});
}

TEST_CASE("FFI drop after build leaves the Context usable", "[isolated_context][sirius_ffi]")
{
  sirius::test::scratch_dir scratch("ffi_embedder_drop");
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

TEST_CASE("FFI a build() that fails after setup rolls back and leaves the Context usable",
          "[isolated_context][sirius_ffi]")
{
  sirius::test::scratch_dir scratch("ffi_embedder_failed_lowering");
  auto const path = scratch.file("ids.parquet");
  write_ids_parquet(path);

  auto ctx = sirius::ffi::make_context_from_config(isolated_memory_config_path().string());

  SECTION("malformed Substrait bytes")
  {
    auto failed = sirius::ffi::make_fragment(*ctx);
    failed->declare_input_column(0, "a", "BIGINT");
    REQUIRE_THROWS(failed->build("not a substrait plan"));
  }

  SECTION("a plan that does not read the declared stream")
  {
    auto failed = sirius::ffi::make_fragment(*ctx);
    failed->declare_input_column(0, "a", "BIGINT");
    REQUIRE_THROWS_WITH(failed->build(local_files_plan(path)),
                        Catch::Matchers::ContainsSubstring("the plan does not read it"));
  }

  auto next = sirius::ffi::make_fragment(*ctx);
  next->build(local_files_plan(path));
  next->run();
  REQUIRE(result_i64s(*next) == std::vector<std::int64_t>{1, 2, 3, 4, 5});
}

TEST_CASE("FFI execute_substrait returns parquet rows", "[isolated_context][sirius_ffi]")
{
  sirius::test::scratch_dir scratch("ffi_embedder_execute_substrait");
  auto const path = scratch.file("ids.parquet");
  write_ids_parquet(path);

  auto ctx = sirius::ffi::make_context_from_config(isolated_memory_config_path().string());
  ArrowArrayStream stream{};
  ctx->execute_substrait(local_files_plan(path), reinterpret_cast<std::uintptr_t>(&stream));
  REQUIRE(collect_i64_column(stream) == std::vector<std::int64_t>{1, 2, 3, 4, 5});
}

TEST_CASE("FFI a hash key on a single output is rejected at build()",
          "[isolated_context][sirius_ffi]")
{
  sirius::test::scratch_dir scratch("ffi_embedder_hash_one_output");
  auto const path = scratch.file("ids.parquet");
  write_ids_parquet(path);
  auto const plan = local_files_plan(path);

  auto ctx    = sirius::ffi::make_context_from_config(isolated_memory_config_path().string());
  auto routed = sirius::ffi::make_fragment(*ctx);
  routed->declare_output(0);
  routed->declare_output_hash_key(0);
  REQUIRE_THROWS_WITH(routed->build(plan), Catch::Matchers::ContainsSubstring("at least two"));

  auto next = sirius::ffi::make_fragment(*ctx);
  next->build(plan);
  next->run();
  REQUIRE(result_i64s(*next) == std::vector<std::int64_t>{1, 2, 3, 4, 5});
}

TEST_CASE("FFI concurrent run() and execute_substrait on one Context wait for each other",
          "[isolated_context][sirius_ffi]")
{
  sirius::test::scratch_dir scratch("ffi_embedder_concurrent_calls");
  auto const path = scratch.file("ids.parquet");
  write_ids_parquet(path);
  auto const plan = local_files_plan(path);

  // Every Fragment shares the Context's one DuckDB connection, so overlapping transactions
  // would fail with "cannot start a transaction within a transaction" if not serialized.
  auto ctx    = sirius::ffi::make_context_from_config(isolated_memory_config_path().string());
  auto first  = sirius::ffi::make_fragment(*ctx);
  auto second = sirius::ffi::make_fragment(*ctx);
  first->build(plan);
  second->build(plan);

  ArrowArrayStream direct{};
  std::future<void> runs[] = {
    std::async(std::launch::async, [&] { first->run(); }),
    std::async(std::launch::async, [&] { second->run(); }),
    std::async(std::launch::async,
               [&] { ctx->execute_substrait(plan, reinterpret_cast<std::uintptr_t>(&direct)); }),
  };
  for (auto& run : runs) {
    run.get();  // rethrows that call's exception on the test thread
  }

  auto const expected = std::vector<std::int64_t>{1, 2, 3, 4, 5};
  REQUIRE(result_i64s(*first) == expected);
  REQUIRE(result_i64s(*second) == expected);
  REQUIRE(collect_i64_column(direct) == expected);
}
