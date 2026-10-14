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
// that fails after setup, execute_substrait, make_context_with_gpus / active_gpu_ids, and the
// single-live-Context guard, the JSON plan entry points (execute_substrait_json, build_json),
// push_arrow (including fan-out to a shared relation), any_value, and TPC-H Q1-Q22 over a
// Substrait corpus. Spec errors and failed-build rollback live in test_streaming_fragment.cpp
// and test_sirius_ffi_fragment.cpp.

#include "sirius/exception.hpp"
#include "sirius/ffi.hpp"
#include "utils/parquet_fixture_utils.hpp"
#include "utils/sirius_test_env.hpp"
#include "yyjson.hpp"

#include <catch.hpp>
#include <duckdb.hpp>
#include <duckdb/common/arrow/arrow.hpp>
#include <duckdb/common/arrow/result_arrow_wrapper.hpp>
#include <duckdb/common/types/date.hpp>
#include <duckdb/common/types/decimal.hpp>
#include <google/protobuf/util/json_util.h>  // DuckDB-vendored: duckdb::google::protobuf
#include <substrait/plan.pb.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <future>
#include <memory>
#include <optional>
#include <source_location>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
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

std::vector<std::int64_t> execute_i64s(sirius::ffi::Context& ctx, std::string const& plan)
{
  ArrowArrayStream stream{};
  ctx.execute_substrait(plan, reinterpret_cast<std::uintptr_t>(&stream));
  return collect_i64_column(stream);
}

// Small per-GPU and host budgets for make_context_with_gpus. The num_gpus key checks that the
// GPU list replaces it rather than conflicting with it.
std::string write_gpu_list_config(sirius::test::scratch_dir& scratch)
{
  auto const path = scratch.file("gpus.yaml");
  std::ofstream(path) << "sirius:\n"
                         "  topology:\n"
                         "    num_gpus: 1\n"
                         "  memory:\n"
                         "    gpu:\n"
                         "      usage_limit_bytes: 2147483648\n"
                         "    host:\n"
                         "      capacity_bytes: 4294967296\n";
  return path;
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
  REQUIRE_THROWS(sender->output_row_count(0));
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

TEST_CASE("FFI make_context_with_gpus on one GPU reports it and executes",
          "[isolated_context][sirius_ffi]")
{
  sirius::test::scratch_dir scratch("ffi_embedder_gpus_one");
  auto const path = scratch.file("ids.parquet");
  write_ids_parquet(path);

  auto ctx = sirius::ffi::make_context_with_gpus({0}, write_gpu_list_config(scratch));
  REQUIRE(*ctx->active_gpu_ids() == std::vector<int>{0});
  REQUIRE(execute_i64s(*ctx, local_files_plan(path)) == std::vector<std::int64_t>{1, 2, 3, 4, 5});
}

TEST_CASE("FFI make_context_with_gpus on two GPUs reports both and executes",
          "[isolated_context][sirius_ffi][multi_gpu]")
{
  if (!sirius::test::has_gpus(2)) { SKIP("needs 2 GPUs"); }
  sirius::test::scratch_dir scratch("ffi_embedder_gpus_two");
  auto const path = scratch.file("ids.parquet");
  write_ids_parquet(path);

  auto ctx = sirius::ffi::make_context_with_gpus({1, 0}, write_gpu_list_config(scratch));
  REQUIRE(*ctx->active_gpu_ids() == std::vector<int>{0, 1});
  REQUIRE(execute_i64s(*ctx, local_files_plan(path)) == std::vector<std::int64_t>{1, 2, 3, 4, 5});
}

TEST_CASE("FFI a Context may be recreated on a different GPU set",
          "[isolated_context][sirius_ffi][multi_gpu]")
{
  if (!sirius::test::has_gpus(2)) { SKIP("needs 2 GPUs"); }
  sirius::test::scratch_dir scratch("ffi_embedder_gpus_recreate");
  auto const path   = scratch.file("ids.parquet");
  auto const config = write_gpu_list_config(scratch);
  write_ids_parquet(path);

  auto ctx = sirius::ffi::make_context_with_gpus({0, 1}, config);
  REQUIRE(*ctx->active_gpu_ids() == std::vector<int>{0, 1});
  ctx.reset();

  ctx = sirius::ffi::make_context_with_gpus({0}, config);
  REQUIRE(*ctx->active_gpu_ids() == std::vector<int>{0});
  REQUIRE(execute_i64s(*ctx, local_files_plan(path)) == std::vector<std::int64_t>{1, 2, 3, 4, 5});
}

TEST_CASE("FFI a second live Context is rejected by every constructor",
          "[isolated_context][sirius_ffi]")
{
  sirius::test::scratch_dir scratch("ffi_embedder_second_context");
  auto const path   = scratch.file("ids.parquet");
  auto const config = write_gpu_list_config(scratch);
  write_ids_parquet(path);

  auto ctx = sirius::ffi::make_context_with_gpus({0}, config);
  REQUIRE_THROWS_AS(sirius::ffi::make_context(), std::logic_error);
  REQUIRE_THROWS_AS(sirius::ffi::make_context_from_config(isolated_memory_config_path().string()),
                    std::logic_error);
  REQUIRE_THROWS_AS(sirius::ffi::make_context_with_gpus({0}, config), std::logic_error);
  REQUIRE_THROWS_WITH(sirius::ffi::make_context_with_gpus({0}, config),
                      Catch::Matchers::ContainsSubstring("already exists"));

  REQUIRE(*ctx->active_gpu_ids() == std::vector<int>{0});
  REQUIRE(execute_i64s(*ctx, local_files_plan(path)) == std::vector<std::int64_t>{1, 2, 3, 4, 5});
}

TEST_CASE("FFI make_context_with_gpus rejects an invalid GPU list",
          "[isolated_context][sirius_ffi]")
{
  sirius::test::scratch_dir scratch("ffi_embedder_gpus_invalid");
  auto const config = write_gpu_list_config(scratch);

  REQUIRE_THROWS_AS(sirius::ffi::make_context_with_gpus({}, config), std::invalid_argument);
  REQUIRE_THROWS_AS(sirius::ffi::make_context_with_gpus({-1}, config), std::invalid_argument);
  REQUIRE_THROWS_AS(sirius::ffi::make_context_with_gpus({0, 0}, config), std::invalid_argument);
  REQUIRE_THROWS_WITH(sirius::ffi::make_context_with_gpus({99}, config),
                      Catch::Matchers::ContainsSubstring("99"));
  REQUIRE_THROWS_AS(sirius::ffi::make_context_with_gpus({99}, config), std::invalid_argument);

  // No failed attempt left a Context alive.
  auto ctx = sirius::ffi::make_context_with_gpus({0}, config);
  REQUIRE(*ctx->active_gpu_ids() == std::vector<int>{0});
}

TEST_CASE("FFI make_context_with_gpus names the offending id", "[isolated_context][sirius_ffi]")
{
  sirius::test::scratch_dir scratch("ffi_embedder_gpus_messages");
  auto const config = write_gpu_list_config(scratch);

  REQUIRE_THROWS_WITH(sirius::ffi::make_context_with_gpus({-1}, config),
                      Catch::Matchers::ContainsSubstring("-1"));
  REQUIRE_THROWS_WITH(
    sirius::ffi::make_context_with_gpus({0, 0}, config),
    Catch::Matchers::ContainsSubstring("duplicate") && Catch::Matchers::ContainsSubstring("0"));
  REQUIRE_THROWS_WITH(sirius::ffi::make_context_with_gpus({}, config),
                      Catch::Matchers::ContainsSubstring("empty"));
}

TEST_CASE("FFI make_context_with_gpus config failures release the live-Context claim",
          "[isolated_context][sirius_ffi]")
{
  sirius::test::scratch_dir scratch("ffi_embedder_gpus_bad_config");
  auto const config = write_gpu_list_config(scratch);

  REQUIRE_THROWS_AS(sirius::ffi::make_context_with_gpus({0}, scratch.file("missing.yaml")),
                    std::runtime_error);

  auto const no_root = scratch.file("no_root.yaml");
  std::ofstream(no_root) << "not_sirius:\n  x: 1\n";
  REQUIRE_THROWS_AS(sirius::ffi::make_context_with_gpus({0}, no_root), std::runtime_error);

  auto ctx = sirius::ffi::make_context_with_gpus({0}, config);
  REQUIRE(*ctx->active_gpu_ids() == std::vector<int>{0});
}

TEST_CASE("FFI make_context_with_gpus overrides the config's gpu_ids and fills a null topology",
          "[isolated_context][sirius_ffi][multi_gpu]")
{
  if (!sirius::test::has_gpus(2)) { SKIP("needs 2 GPUs"); }
  sirius::test::scratch_dir scratch("ffi_embedder_gpus_override");
  auto const path = scratch.file("ids.parquet");
  write_ids_parquet(path);

  auto const with_ids = scratch.file("with_ids.yaml");
  std::ofstream(with_ids) << "sirius:\n"
                             "  topology:\n"
                             "    gpu_ids: [1]\n"
                             "    gpus_per_query: 1\n"
                             "  memory:\n"
                             "    gpu:\n"
                             "      usage_limit_bytes: 2147483648\n"
                             "    host:\n"
                             "      capacity_bytes: 4294967296\n";
  auto ctx = sirius::ffi::make_context_with_gpus({0}, with_ids);
  REQUIRE(*ctx->active_gpu_ids() == std::vector<int>{0});
  REQUIRE(execute_i64s(*ctx, local_files_plan(path)) == std::vector<std::int64_t>{1, 2, 3, 4, 5});
  ctx.reset();

  auto const null_topology = scratch.file("null_topology.yaml");
  std::ofstream(null_topology) << "sirius:\n"
                                  "  topology:\n"
                                  "  memory:\n"
                                  "    gpu:\n"
                                  "      usage_limit_bytes: 2147483648\n"
                                  "    host:\n"
                                  "      capacity_bytes: 4294967296\n";
  ctx = sirius::ffi::make_context_with_gpus({1}, null_topology);
  REQUIRE(*ctx->active_gpu_ids() == std::vector<int>{1});
  REQUIRE(execute_i64s(*ctx, local_files_plan(path)) == std::vector<std::int64_t>{1, 2, 3, 4, 5});
}

TEST_CASE("FFI destroying a config-path Context lets make_context_with_gpus succeed",
          "[isolated_context][sirius_ffi]")
{
  sirius::test::scratch_dir scratch("ffi_embedder_gpus_after_config_ctx");
  auto const config = write_gpu_list_config(scratch);

  auto first = sirius::ffi::make_context_from_config(isolated_memory_config_path().string());
  REQUIRE_FALSE(first->active_gpu_ids()->empty());
  REQUIRE_THROWS_AS(sirius::ffi::make_context_with_gpus({0}, config), std::logic_error);
  first.reset();

  auto second = sirius::ffi::make_context_with_gpus({0}, config);
  REQUIRE(*second->active_gpu_ids() == std::vector<int>{0});
}

// ---------------------------------------------------------------------------
// JSON plans and push_arrow
// ---------------------------------------------------------------------------

namespace {

namespace pb_util = duckdb::google::protobuf::util;

std::string to_json(substrait::Plan const& plan, bool snake_case = false)
{
  pb_util::JsonPrintOptions options;
  options.preserve_proto_field_names = snake_case;
  std::string json;
  REQUIRE(pb_util::MessageToJsonString(plan, &json, options).ok());
  return json;
}

substrait::Plan parse_plan(std::string const& bytes)
{
  substrait::Plan plan;
  REQUIRE(plan.ParseFromString(bytes));
  return plan;
}

// A plan that reads the single column `a` of input stream `stream_id` with Substrait type `set`.
template <typename SetType>
std::string stream_read_json(std::uint64_t stream_id, SetType&& set)
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
  set(*st->add_types());
  return to_json(plan);
}

constexpr auto kNullable = ::substrait::Type_Nullability_NULLABILITY_NULLABLE;

std::string i64_read_json(std::uint64_t stream_id)
{
  return stream_read_json(stream_id,
                          [](substrait::Type& t) { t.mutable_i64()->set_nullability(kNullable); });
}

// --- Arrow inputs, produced by a plain DuckDB through its Arrow export -------------------------

// A CPU-only DuckDB whose query results become Arrow streams. Lives as long as the streams.
struct arrow_source {
  std::unique_ptr<duckdb::DuckDB> db;
  std::unique_ptr<duckdb::Connection> con;

  arrow_source()
  {
    sirius::test::scoped_sirius_disable disable;
    db  = std::make_unique<duckdb::DuckDB>(nullptr);
    con = std::make_unique<duckdb::Connection>(*db);
  }

  void set(std::string const& sql)
  {
    auto res = con->Query(sql);
    REQUIRE_FALSE(res->HasError());
  }

  // One stream with every row of `sql` (one batch for these small queries).
  ArrowArrayStream stream(std::string const& sql)
  {
    sirius::test::scoped_sirius_disable disable;
    auto result = con->Query(sql);
    INFO(sql);
    REQUIRE_FALSE(result->HasError());
    // The stream's release callback deletes the wrapper.
    auto* wrapper = new duckdb::ResultArrowArrayStreamWrapper(std::move(result), 1u << 20);
    return wrapper->stream;
  }
};

// An ArrowArrayStream over the batches of several streams in order (all share the first's
// schema). Options: end right after the schema (zero-row stream), or fail the first get_next.
struct concat_state {
  std::vector<ArrowArrayStream> parts;
  std::size_t current{0};
  bool schema_only{false};
  bool fail_get_next{false};
  std::string error;
};

int concat_get_schema(ArrowArrayStream* self, ArrowSchema* out)
{
  auto* state = static_cast<concat_state*>(self->private_data);
  return state->parts.front().get_schema(&state->parts.front(), out);
}

int concat_get_next(ArrowArrayStream* self, ArrowArray* out)
{
  auto* state = static_cast<concat_state*>(self->private_data);
  if (state->fail_get_next) {
    state->error = "injected get_next failure";
    return EIO;
  }
  out->release = nullptr;
  while (!state->schema_only && state->current < state->parts.size()) {
    auto& part = state->parts[state->current];
    if (int rc = part.get_next(&part, out); rc != 0) { return rc; }
    if (out->release != nullptr) { return 0; }
    ++state->current;
  }
  return 0;
}

const char* concat_last_error(ArrowArrayStream* self)
{
  return static_cast<concat_state*>(self->private_data)->error.c_str();
}

void concat_release(ArrowArrayStream* self)
{
  auto* state = static_cast<concat_state*>(self->private_data);
  for (auto& part : state->parts) {
    release_if(part);
  }
  delete state;
  self->release = nullptr;
}

ArrowArrayStream concat_streams(std::vector<ArrowArrayStream> parts,
                                bool schema_only   = false,
                                bool fail_get_next = false)
{
  auto* state          = new concat_state{std::move(parts), 0, schema_only, fail_get_next, {}};
  ArrowArrayStream out = {};
  out.get_schema       = concat_get_schema;
  out.get_next         = concat_get_next;
  out.get_last_error   = concat_last_error;
  out.release          = concat_release;
  out.private_data     = state;
  return out;
}

std::string arrow_format_of(ArrowArrayStream& stream)
{
  ArrowSchema schema{};
  REQUIRE(stream.get_schema(&stream, &schema) == 0);
  REQUIRE(schema.n_children >= 1);
  std::string format = schema.children[0]->format;
  schema.release(&schema);
  return format;
}

// --- Arrow outputs, rendered as text -----------------------------------------------------------

using cell  = std::optional<std::string>;
using table = std::vector<std::vector<cell>>;

cell render_cell(ArrowSchema const& schema, ArrowArray const& array, std::int64_t row)
{
  auto const i = row + array.offset;
  if (array.null_count != 0 && array.buffers[0] != nullptr) {
    auto const* validity = static_cast<std::uint8_t const*>(array.buffers[0]);
    if (((validity[i / 8] >> (i % 8)) & 1) == 0) { return std::nullopt; }
  }
  std::string const format = schema.format;
  auto const* values       = array.buffers[1];
  if (format == "l") { return std::to_string(static_cast<std::int64_t const*>(values)[i]); }
  if (format == "i" || format == "tdD") {
    return std::to_string(static_cast<std::int32_t const*>(values)[i]);
  }
  if (format == "g") {
    char buffer[64];
    std::snprintf(buffer, sizeof(buffer), "%.17g", static_cast<double const*>(values)[i]);
    return std::string(buffer);
  }
  if (format.rfind("d:", 0) == 0) {
    int precision = 0, scale = 0;
    REQUIRE(std::sscanf(format.c_str(), "d:%d,%d", &precision, &scale) == 2);
    std::uint64_t words[2];  // little-endian decimal128: low word first
    std::memcpy(words, static_cast<char const*>(values) + 16 * i, sizeof(words));
    duckdb::hugeint_t const unscaled(static_cast<std::int64_t>(words[1]), words[0]);
    return duckdb::Decimal::ToString(
      unscaled, static_cast<std::uint8_t>(precision), static_cast<std::uint8_t>(scale));
  }
  if (format == "u") {
    auto const* offsets = static_cast<std::int32_t const*>(values);
    auto const* chars   = static_cast<char const*>(array.buffers[2]);
    return std::string(chars + offsets[i], chars + offsets[i + 1]);
  }
  if (format == "U") {
    auto const* offsets = static_cast<std::int64_t const*>(values);
    auto const* chars   = static_cast<char const*>(array.buffers[2]);
    return std::string(chars + offsets[i], chars + offsets[i + 1]);
  }
  FAIL("unexpected Arrow result format " << format);
  return std::nullopt;
}

table collect_table(ArrowArrayStream& stream)
{
  ArrowSchema schema{};
  REQUIRE(stream.get_schema(&stream, &schema) == 0);
  table rows;
  for (;;) {
    ArrowArray batch{};
    REQUIRE(stream.get_next(&stream, &batch) == 0);
    if (batch.release == nullptr) { break; }
    for (std::int64_t r = 0; r < batch.length; ++r) {
      std::vector<cell> row;
      for (std::int64_t c = 0; c < schema.n_children; ++c) {
        row.push_back(render_cell(*schema.children[c], *batch.children[c], r + batch.offset));
      }
      rows.push_back(std::move(row));
    }
    batch.release(&batch);
  }
  schema.release(&schema);
  release_if(stream);
  return rows;
}

table result_table(sirius::ffi::Fragment& fragment)
{
  ArrowArrayStream stream{};
  fragment.result_to_arrow(reinterpret_cast<std::uintptr_t>(&stream));
  return collect_table(stream);
}

// The first column of `rows`, sorted (nulls first): GPU batches may finish in any order.
std::vector<cell> sorted_column(table const& rows)
{
  std::vector<cell> column;
  for (auto const& row : rows) {
    column.push_back(row.at(0));
  }
  std::sort(column.begin(), column.end());
  return column;
}

std::size_t push(sirius::ffi::Fragment& fragment,
                 std::uint64_t stream_id,
                 ArrowArrayStream& stream,
                 std::uint32_t sender_id = 0)
{
  return fragment.push_arrow(stream_id, reinterpret_cast<std::uintptr_t>(&stream), sender_id);
}

struct round_trip_case {
  std::string declared;  // DuckDB type name for declare_input_column
  std::function<void(substrait::Type&)> substrait_type;
  std::string batch_1;  // two SQL queries producing column `a`; together they include a NULL
  std::string batch_2;
  std::vector<cell> expected;  // sorted, nulls first
  std::string source_setting;  // optional SET on the Arrow source (string layout)
  std::string arrow_format;    // expected Arrow format of the input
};

// Declare `a` of the case's type, build_json a read of it, push the two batches, run, check.
void run_round_trip(sirius::ffi::Context& ctx, round_trip_case const& c)
{
  INFO("declared " << c.declared << ", Arrow " << c.arrow_format);
  arrow_source source;
  if (!c.source_setting.empty()) { source.set(c.source_setting); }
  auto first = source.stream(c.batch_1);
  REQUIRE(arrow_format_of(first) == c.arrow_format);
  auto input = concat_streams({first, source.stream(c.batch_2)});

  auto fragment = sirius::ffi::make_fragment(ctx);
  fragment->declare_input_column(0, "a", c.declared);
  fragment->build_json(stream_read_json(0, c.substrait_type));
  REQUIRE(push(*fragment, 0, input) == 2);
  REQUIRE(input.release == nullptr);
  fragment->run();
  REQUIRE(sorted_column(result_table(*fragment)) == c.expected);
}

}  // namespace

TEST_CASE("FFI execute_substrait_json returns parquet rows for camelCase and snake_case plans",
          "[isolated_context][sirius_ffi]")
{
  sirius::test::scratch_dir scratch("ffi_embedder_execute_json");
  auto const path = scratch.file("ids.parquet");
  write_ids_parquet(path);
  auto const plan = parse_plan(local_files_plan(path));

  auto ctx = sirius::ffi::make_context_from_config(isolated_memory_config_path().string());
  auto const expected = std::vector<std::int64_t>{1, 2, 3, 4, 5};
  for (bool snake_case : {false, true}) {
    auto const json = to_json(plan, snake_case);
    REQUIRE(json.find(snake_case ? "local_files" : "localFiles") != std::string::npos);
    ArrowArrayStream stream{};
    ctx->execute_substrait_json(json, reinterpret_cast<std::uintptr_t>(&stream));
    REQUIRE(collect_i64_column(stream) == expected);
  }
}

TEST_CASE("FFI execute_substrait_json rejects malformed JSON and the Context stays usable",
          "[isolated_context][sirius_ffi]")
{
  sirius::test::scratch_dir scratch("ffi_embedder_execute_json_bad");
  auto const path = scratch.file("ids.parquet");
  write_ids_parquet(path);

  auto ctx = sirius::ffi::make_context_from_config(isolated_memory_config_path().string());
  ArrowArrayStream stream{};
  auto const out = reinterpret_cast<std::uintptr_t>(&stream);
  REQUIRE_THROWS_AS(ctx->execute_substrait_json("{not json", out), std::invalid_argument);
  REQUIRE_THROWS_WITH(ctx->execute_substrait_json("{\"relations\": 7}", out),
                      Catch::Matchers::ContainsSubstring("Substrait Plan"));
  // Binary plan bytes are not JSON either.
  REQUIRE_THROWS_AS(ctx->execute_substrait_json(local_files_plan(path), out),
                    std::invalid_argument);
  REQUIRE(stream.release == nullptr);

  ctx->execute_substrait_json(to_json(parse_plan(local_files_plan(path))), out);
  REQUIRE(collect_i64_column(stream) == std::vector<std::int64_t>{1, 2, 3, 4, 5});
}

TEST_CASE("FFI build_json reads a declared stream; a malformed plan leaves the Fragment unbuilt",
          "[isolated_context][sirius_ffi]")
{
  sirius::test::scratch_dir scratch("ffi_embedder_build_json");
  auto const path = scratch.file("ids.parquet");
  write_ids_parquet(path);

  auto ctx    = sirius::ffi::make_context_from_config(isolated_memory_config_path().string());
  auto sender = sirius::ffi::make_fragment(*ctx);
  sender->declare_output(0);
  sender->build_json(to_json(parse_plan(local_files_plan(path))));
  sender->run();

  auto receiver = sirius::ffi::make_fragment(*ctx);
  receiver->declare_input_column(0, "a", "BIGINT");
  REQUIRE_THROWS_AS(receiver->build_json("{\"relations\": [}"), std::invalid_argument);
  // Still unbuilt: declarations are accepted, and relay_from requires a build.
  REQUIRE_THROWS(receiver->relay_from(*sender, 0, 0, 0));
  REQUIRE_NOTHROW(receiver->declare_input_sender(0, 0));

  receiver->build_json(i64_read_json(0));
  REQUIRE(receiver->relay_from(*sender, 0, 0, 0) > 0);
  receiver->run();
  REQUIRE(result_i64s(*receiver) == std::vector<std::int64_t>{1, 2, 3, 4, 5});
}

TEST_CASE("FFI push_arrow round-trips each supported type with nulls",
          "[isolated_context][sirius_ffi]")
{
  auto ctx = sirius::ffi::make_context_from_config(isolated_memory_config_path().string());

  auto const decimal_15_2 = [](substrait::Type& t) {
    auto* d = t.mutable_decimal();
    d->set_precision(15);
    d->set_scale(2);
    d->set_nullability(kNullable);
  };
  auto const varchar   = [](substrait::Type& t) { t.mutable_string()->set_nullability(kNullable); };
  auto const strings_1 = "SELECT * FROM (VALUES ('hello'::VARCHAR), (NULL), ('')) t(a)";
  auto const strings_2 =
    "SELECT * FROM (VALUES ('a string longer than twelve bytes'::VARCHAR), ('ünïcode')) t(a)";
  std::vector<cell> const strings_expected = {
    std::nullopt, "", "a string longer than twelve bytes", "hello", "ünïcode"};

  std::vector<round_trip_case> const cases = {
    {"DECIMAL(15,2)",
     decimal_15_2,
     "SELECT * FROM (VALUES (1.25::DECIMAL(15,2)), (NULL)) t(a)",
     "SELECT * FROM (VALUES (-3.50::DECIMAL(15,2)), (9999999999999.99::DECIMAL(15,2))) t(a)",
     {std::nullopt, "-3.50", "1.25", "9999999999999.99"},
     "",
     "d:15,2,128"},
    // Rescale (scale 4 -> 2, rounding half away from zero like DuckDB CAST) and narrow
    // (precision 38 -> 15).
    {"DECIMAL(15,2)",
     decimal_15_2,
     "SELECT * FROM (VALUES (1.2350::DECIMAL(38,4)), (NULL), (12.3456::DECIMAL(38,4))) t(a)",
     "SELECT * FROM (VALUES (-1.2350::DECIMAL(38,4))) t(a)",
     {std::nullopt, "-1.24", "1.24", "12.35"},
     "",
     "d:38,4,128"},
    {"DATE",
     [](substrait::Type& t) { t.mutable_date()->set_nullability(kNullable); },
     "SELECT * FROM (VALUES (DATE '1970-01-02'), (NULL)) t(a)",
     "SELECT * FROM (VALUES (DATE '2000-01-01'), (DATE '1969-12-31')) t(a)",
     {std::nullopt, "-1", "1", "10957"},
     "",
     "tdD"},
    {"VARCHAR", varchar, strings_1, strings_2, strings_expected, "", "u"},
    {"VARCHAR",
     varchar,
     strings_1,
     strings_2,
     strings_expected,
     "SET arrow_large_buffer_size = true",
     "U"},
    {"BIGINT",
     [](substrait::Type& t) { t.mutable_i64()->set_nullability(kNullable); },
     "SELECT * FROM (VALUES (1::BIGINT), (NULL)) t(a)",
     "SELECT * FROM (VALUES (-4::BIGINT), (9007199254740993::BIGINT)) t(a)",
     {std::nullopt, "-4", "1", "9007199254740993"},
     "",
     "l"},
    {"INTEGER",
     [](substrait::Type& t) { t.mutable_i32()->set_nullability(kNullable); },
     "SELECT * FROM (VALUES (7::INTEGER), (NULL)) t(a)",
     "SELECT * FROM (VALUES (-8::INTEGER)) t(a)",
     {std::nullopt, "-8", "7"},
     "",
     "i"},
    {"DOUBLE",
     [](substrait::Type& t) { t.mutable_fp64()->set_nullability(kNullable); },
     "SELECT * FROM (VALUES (0.5::DOUBLE), (NULL)) t(a)",
     "SELECT * FROM (VALUES (-2.25::DOUBLE)) t(a)",
     {std::nullopt, "-2.25", "0.5"},
     "",
     "g"},
  };
  for (auto const& c : cases) {
    run_round_trip(*ctx, c);
  }
}

TEST_CASE("FFI push_arrow accepts string_view input", "[isolated_context][sirius_ffi]")
{
  auto ctx = sirius::ffi::make_context_from_config(isolated_memory_config_path().string());
  run_round_trip(*ctx,
                 {"VARCHAR",
                  [](substrait::Type& t) { t.mutable_string()->set_nullability(kNullable); },
                  "SELECT * FROM (VALUES ('short'::VARCHAR), (NULL)) t(a)",
                  "SELECT * FROM (VALUES ('a string longer than twelve bytes'::VARCHAR)) t(a)",
                  {std::nullopt, "a string longer than twelve bytes", "short"},
                  "SET arrow_output_version = '1.4'; SET produce_arrow_string_view = true",
                  "vu"});
}

TEST_CASE("FFI push_arrow accepts a zero-row stream", "[isolated_context][sirius_ffi]")
{
  auto ctx = sirius::ffi::make_context_from_config(isolated_memory_config_path().string());
  arrow_source source;
  auto input = concat_streams({source.stream("SELECT 1::BIGINT AS a")}, /*schema_only=*/true);

  auto fragment = sirius::ffi::make_fragment(*ctx);
  fragment->declare_input_column(0, "a", "BIGINT");
  fragment->build_json(i64_read_json(0));
  REQUIRE(push(*fragment, 0, input) == 0);
  REQUIRE(input.release == nullptr);
  fragment->run();
  REQUIRE(result_table(*fragment).empty());
}

TEST_CASE("FFI push_arrow takes ownership of the stream on success and on failure",
          "[isolated_context][sirius_ffi]")
{
  auto ctx = sirius::ffi::make_context_from_config(isolated_memory_config_path().string());
  arrow_source source;

  auto fragment = sirius::ffi::make_fragment(*ctx);
  fragment->declare_input_column(0, "a", "BIGINT");
  fragment->declare_input_sender(0, 0);
  fragment->declare_input_sender(0, 1);
  fragment->build_json(i64_read_json(0));

  // Failure before any data moves: wrong type.
  auto wrong = source.stream("SELECT 'x'::VARCHAR AS a");
  REQUIRE_THROWS_AS(push(*fragment, 0, wrong), std::invalid_argument);
  REQUIRE(wrong.release == nullptr);

  // Failure while reading: get_next fails.
  auto failing = concat_streams({source.stream("SELECT 1::BIGINT AS a")}, false, true);
  REQUIRE_THROWS_WITH(push(*fragment, 0, failing),
                      Catch::Matchers::ContainsSubstring("injected get_next failure"));
  REQUIRE(failing.release == nullptr);
  auto failing_again = concat_streams({source.stream("SELECT 1::BIGINT AS a")}, false, true);
  REQUIRE_THROWS_AS(push(*fragment, 0, failing_again), std::runtime_error);
  REQUIRE(failing_again.release == nullptr);

  // Success.
  auto good = source.stream("SELECT 5::BIGINT AS a");
  REQUIRE(push(*fragment, 0, good, 0) == 1);
  REQUIRE(good.release == nullptr);
  auto more = source.stream("SELECT 6::BIGINT AS a");
  REQUIRE(push(*fragment, 0, more, 1) == 1);
  REQUIRE(more.release == nullptr);

  fragment->run();
  REQUIRE(sorted_column(result_table(*fragment)) == std::vector<cell>{"5", "6"});
}

TEST_CASE("FFI push_arrow rejects bad calls with typed errors", "[isolated_context][sirius_ffi]")
{
  auto ctx = sirius::ffi::make_context_from_config(isolated_memory_config_path().string());
  arrow_source source;
  auto const one_bigint = "SELECT 1::BIGINT AS a";

  auto fragment = sirius::ffi::make_fragment(*ctx);
  fragment->declare_input_column(0, "a", "BIGINT");

  SECTION("before build")
  {
    auto s = source.stream(one_bigint);
    REQUIRE_THROWS_AS(push(*fragment, 0, s), std::logic_error);
    REQUIRE(s.release == nullptr);
    return;
  }

  fragment->build_json(i64_read_json(0));

  SECTION("undeclared stream")
  {
    auto s = source.stream(one_bigint);
    REQUIRE_THROWS_AS(push(*fragment, 7, s), std::invalid_argument);
    auto t = source.stream(one_bigint);
    REQUIRE_THROWS_WITH(push(*fragment, 7, t), Catch::Matchers::ContainsSubstring("7"));
  }

  SECTION("sender outside the declared set")
  {
    auto s = source.stream(one_bigint);
    REQUIRE_THROWS_AS(push(*fragment, 0, s, 3), std::invalid_argument);
    auto t = source.stream(one_bigint);
    REQUIRE_THROWS_WITH(push(*fragment, 0, t, 3), Catch::Matchers::ContainsSubstring("sender 3"));
  }

  SECTION("column-count mismatch")
  {
    auto s = source.stream("SELECT 1::BIGINT AS a, 2::BIGINT AS b");
    REQUIRE_THROWS_AS(push(*fragment, 0, s), std::invalid_argument);
    auto t = source.stream("SELECT 1::BIGINT AS a, 2::BIGINT AS b");
    REQUIRE_THROWS_WITH(
      push(*fragment, 0, t),
      Catch::Matchers::ContainsSubstring("1 columns") && Catch::Matchers::ContainsSubstring("2"));
  }

  SECTION("unconvertible type")
  {
    auto s = source.stream("SELECT 'x'::VARCHAR AS a");
    REQUIRE_THROWS_AS(push(*fragment, 0, s), std::invalid_argument);
    auto t = source.stream("SELECT 'x'::VARCHAR AS a");
    REQUIRE_THROWS_WITH(push(*fragment, 0, t),
                        Catch::Matchers::ContainsSubstring("stream 0") &&
                          Catch::Matchers::ContainsSubstring("column 0") &&
                          Catch::Matchers::ContainsSubstring("utf8") &&
                          Catch::Matchers::ContainsSubstring("BIGINT"));
  }

  SECTION("second push on a closed sender")
  {
    auto s = source.stream(one_bigint);
    REQUIRE(push(*fragment, 0, s) == 1);
    auto t = source.stream(one_bigint);
    REQUIRE_THROWS_AS(push(*fragment, 0, t), std::runtime_error);
    REQUIRE(t.release == nullptr);
    fragment->run();
    REQUIRE(result_i64s(*fragment) == std::vector<std::int64_t>{1});
    return;
  }

  SECTION("after run")
  {
    auto s = source.stream(one_bigint);
    REQUIRE(push(*fragment, 0, s) == 1);
    fragment->run();
    auto t = source.stream(one_bigint);
    REQUIRE_THROWS_AS(push(*fragment, 0, t), std::logic_error);
    REQUIRE(t.release == nullptr);
    return;
  }

  // Every rejected call above ran before any data moved: the fragment still accepts input.
  auto s = source.stream("SELECT 2::BIGINT AS a");
  REQUIRE(push(*fragment, 0, s) == 1);
  fragment->run();
  REQUIRE(result_i64s(*fragment) == std::vector<std::int64_t>{2});
}

TEST_CASE("FFI push_arrow rejects a decimal that overflows the declared precision",
          "[isolated_context][sirius_ffi]")
{
  auto ctx = sirius::ffi::make_context_from_config(isolated_memory_config_path().string());
  arrow_source source;
  auto fragment = sirius::ffi::make_fragment(*ctx);
  fragment->declare_input_column(0, "a", "DECIMAL(15,2)");
  fragment->build_json(stream_read_json(0, [](substrait::Type& t) {
    auto* d = t.mutable_decimal();
    d->set_precision(15);
    d->set_scale(2);
    d->set_nullability(kNullable);
  }));
  auto s = source.stream("SELECT 10000000000000::DECIMAL(38,0) AS a");
  REQUIRE_THROWS_AS(push(*fragment, 0, s), std::invalid_argument);
  REQUIRE(s.release == nullptr);
}

TEST_CASE("FFI push_arrow rejects a decimal upscale that overflows before it wraps",
          "[isolated_context][sirius_ffi]")
{
  auto ctx = sirius::ffi::make_context_from_config(isolated_memory_config_path().string());
  arrow_source source;
  auto fragment = sirius::ffi::make_fragment(*ctx);
  fragment->declare_input_column(0, "a", "DECIMAL(38,4)");
  fragment->build_json(stream_read_json(0, [](substrait::Type& t) {
    auto* d = t.mutable_decimal();
    d->set_precision(38);
    d->set_scale(4);
    d->set_nullability(kNullable);
  }));
  // 10^37 * 10^4 = 10^41 is past the int128 range: the check must run before the rescale wraps.
  auto const ten_to_37 = "1" + std::string(37, '0');
  auto s               = source.stream("SELECT '" + ten_to_37 + "'::DECIMAL(38,0) AS a");
  REQUIRE(arrow_format_of(s) == "d:38,0,128");
  REQUIRE_THROWS_AS(push(*fragment, 0, s), std::invalid_argument);
  REQUIRE(s.release == nullptr);
}

// ---------------------------------------------------------------------------
// TPC-H from a Substrait corpus (Q1/Q6 on one GPU, all 22 on two) through execute_substrait_json
// ---------------------------------------------------------------------------

namespace {

// Directory from environment variable `name`; empty when unset.
fs::path env_dir(char const* name)
{
  char const* value = std::getenv(name);
  return value ? fs::path(value) : fs::path();
}

std::string read_file(fs::path const& path)
{
  std::ifstream in(path);
  std::stringstream text;
  text << in.rdbuf();
  return text.str();
}

// Point every named-table read at <input_dir>/<table>.parquet as a localFiles read.
void named_tables_to_local_files(substrait::Rel& rel, fs::path const& input_dir)
{
  switch (rel.rel_type_case()) {
    case substrait::Rel::kRead: {
      auto* read = rel.mutable_read();
      if (!read->has_named_table()) { return; }
      auto const table = read->named_table().names(read->named_table().names_size() - 1);
      read->clear_named_table();
      auto* item = read->mutable_local_files()->add_items();
      item->set_uri_file("file://" + (input_dir / (table + ".parquet")).string());
      item->mutable_parquet();
      return;
    }
    case substrait::Rel::kFilter:
      return named_tables_to_local_files(*rel.mutable_filter()->mutable_input(), input_dir);
    case substrait::Rel::kProject:
      return named_tables_to_local_files(*rel.mutable_project()->mutable_input(), input_dir);
    case substrait::Rel::kAggregate:
      return named_tables_to_local_files(*rel.mutable_aggregate()->mutable_input(), input_dir);
    case substrait::Rel::kSort:
      return named_tables_to_local_files(*rel.mutable_sort()->mutable_input(), input_dir);
    case substrait::Rel::kFetch:
      return named_tables_to_local_files(*rel.mutable_fetch()->mutable_input(), input_dir);
    case substrait::Rel::kJoin:
      named_tables_to_local_files(*rel.mutable_join()->mutable_left(), input_dir);
      return named_tables_to_local_files(*rel.mutable_join()->mutable_right(), input_dir);
    case substrait::Rel::kCross:
      named_tables_to_local_files(*rel.mutable_cross()->mutable_left(), input_dir);
      return named_tables_to_local_files(*rel.mutable_cross()->mutable_right(), input_dir);
    case substrait::Rel::kReference: return;  // a shared relation, rewritten where it is defined
    default: FAIL("unexpected relation in corpus plan: " << rel.rel_type_case());
  }
}

std::string local_files_corpus_plan(fs::path const& plan_path, fs::path const& input_dir)
{
  substrait::Plan plan;
  REQUIRE(pb_util::JsonStringToMessage(read_file(plan_path), &plan).ok());
  for (auto& relation : *plan.mutable_relations()) {
    if (relation.has_root()) {
      named_tables_to_local_files(*relation.mutable_root()->mutable_input(), input_dir);
    } else if (relation.has_rel()) {
      named_tables_to_local_files(*relation.mutable_rel(), input_dir);
    }
  }
  return to_json(plan);
}

// All rows of `rows`, sorted (nulls first): GPU batches may finish in any order.
table sorted_rows(table rows)
{
  std::sort(rows.begin(), rows.end());
  return rows;
}

// True when the plan's root output passes through a sort (an ORDER BY), looking below
// projections, fetches and filters.
bool corpus_plan_is_ordered(fs::path const& plan_path)
{
  substrait::Plan plan;
  REQUIRE(pb_util::JsonStringToMessage(read_file(plan_path), &plan).ok());
  REQUIRE(plan.relations_size() > 0);
  auto const* rel = &plan.relations(plan.relations_size() - 1).root().input();
  for (;;) {
    switch (rel->rel_type_case()) {
      case substrait::Rel::kSort: return true;
      case substrait::Rel::kProject: rel = &rel->project().input(); break;
      case substrait::Rel::kFetch: rel = &rel->fetch().input(); break;
      case substrait::Rel::kFilter: rel = &rel->filter().input(); break;
      default: return false;
    }
  }
}

// The rows of an expected-results CSV (comma-separated, no header, '"' quoting, as DuckDB's
// COPY writes them) as text. An unquoted empty field is NULL; a quoted one ("") is ''.
table read_expected_csv(arrow_source& source, fs::path const& path)
{
  sirius::test::scoped_sirius_disable disable;
  auto result =
    source.con->Query("SELECT * FROM read_csv(" + sirius::test::sql_literal(path.string()) +
                      ", header = false, delim = ',', quote = '\"', escape = '\"', "
                      "all_varchar = true, allow_quoted_nulls = false)");
  INFO(path.string());
  REQUIRE_FALSE(result->HasError());
  table rows;
  for (duckdb::idx_t r = 0; r < result->RowCount(); ++r) {
    std::vector<cell> row;
    for (duckdb::idx_t c = 0; c < result->ColumnCount(); ++c) {
      auto const value = result->GetValue(c, r);
      row.push_back(value.IsNull() ? cell{} : cell{value.ToString()});
    }
    rows.push_back(std::move(row));
  }
  return rows;
}

// The physical_types of query `name` in the corpus's result-types.json.
std::vector<std::string> corpus_physical_types(fs::path const& result_types,
                                               std::string const& name)
{
  namespace yy       = duckdb_yyjson;
  auto const text    = read_file(result_types);
  auto const release = [](yy::yyjson_doc* doc) { yy::yyjson_doc_free(doc); };
  std::unique_ptr<yy::yyjson_doc, decltype(release)> doc(
    yy::yyjson_read(text.data(), text.size(), 0), release);
  INFO(result_types.string() << ", query " << name);
  REQUIRE(doc != nullptr);
  auto* queries = yy::yyjson_obj_get(yy::yyjson_doc_get_root(doc.get()), "queries");
  auto* types   = yy::yyjson_obj_get(yy::yyjson_obj_get(queries, name.c_str()), "physical_types");
  REQUIRE(yy::yyjson_is_arr(types));
  std::vector<std::string> out;
  for (std::size_t i = 0; i < yy::yyjson_arr_size(types); ++i) {
    auto* type = yy::yyjson_arr_get(types, i);
    REQUIRE(yy::yyjson_is_str(type));
    out.emplace_back(yy::yyjson_get_str(type));
  }
  return out;
}

// Whether Arrow format `format` has the layout of result-types.json physical type `type`.
bool arrow_matches_physical_type(std::string const& format, std::string const& type)
{
  if (type == "string") { return format == "u" || format == "U" || format == "vu"; }
  if (type == "i64") { return format == "l"; }
  if (type == "i32") { return format == "i"; }
  if (type == "fp64") { return format == "g"; }
  if (type == "date") { return format == "tdD"; }
  int precision = 0, scale = 0;
  if (std::sscanf(type.c_str(), "decimal(%d,%d)", &precision, &scale) == 2) {
    auto const d128 = "d:" + std::to_string(precision) + "," + std::to_string(scale);
    return format == d128 || format == d128 + ",128";
  }
  return false;
}

std::string days_to_iso_date(std::int64_t days)
{
  return duckdb::Date::ToString(duckdb::date_t(static_cast<std::int32_t>(days)));
}

// collect_table with the column Arrow formats, DATE cells rendered as YYYY-MM-DD.
table collect_tpch_table(ArrowArrayStream& stream, std::vector<std::string>& formats)
{
  ArrowSchema schema{};
  REQUIRE(stream.get_schema(&stream, &schema) == 0);
  formats.clear();
  for (std::int64_t c = 0; c < schema.n_children; ++c) {
    formats.emplace_back(schema.children[c]->format);
  }
  schema.release(&schema);
  auto rows = collect_table(stream);
  for (auto& row : rows) {
    for (std::size_t c = 0; c < row.size() && c < formats.size(); ++c) {
      if (formats[c] == "tdD" && row[c]) { row[c] = days_to_iso_date(std::stoll(*row[c])); }
    }
  }
  return rows;
}

double parse_double(std::string const& text)
{
  char* end          = nullptr;
  double const value = std::strtod(text.c_str(), &end);
  INFO("not a number: '" << text << "'");
  REQUIRE((!text.empty() && end == text.c_str() + text.size()));
  return value;
}

// Floating-point columns (Arrow `formats` g / f) match within a 1e-6 relative tolerance; every
// other cell, integers and decimals included, must match exactly, NULL only NULL.
void require_matches(table const& actual,
                     table const& expected,
                     std::vector<std::string> const& formats)
{
  REQUIRE(actual.size() == expected.size());
  for (std::size_t r = 0; r < expected.size(); ++r) {
    REQUIRE(actual[r].size() == formats.size());
    REQUIRE(expected[r].size() == formats.size());
    for (std::size_t c = 0; c < formats.size(); ++c) {
      INFO("row " << r << " column " << c << " (Arrow " << formats[c] << "): got "
                  << actual[r][c].value_or("NULL") << ", expected "
                  << expected[r][c].value_or("NULL"));
      bool const floating = formats[c] == "g" || formats[c] == "f";
      if (!floating || !actual[r][c] || !expected[r][c]) {
        REQUIRE(actual[r][c] == expected[r][c]);
        continue;
      }
      auto const got  = parse_double(*actual[r][c]);
      auto const want = parse_double(*expected[r][c]);
      REQUIRE(std::abs(got - want) <= 1e-6 * std::max(1.0, std::abs(want)));
    }
  }
}

// GPU and host budgets for TPC-H SF1 on each GPU of the Context.
std::string write_tpch_config(sirius::test::scratch_dir& scratch,
                              std::uint64_t gpu_bytes,
                              std::uint64_t host_bytes)
{
  auto const path = scratch.file("tpch.yaml");
  std::ofstream(path) << "sirius:\n  memory:\n    gpu:\n      usage_limit_bytes: " << gpu_bytes
                      << "\n    host:\n      capacity_bytes: " << host_bytes << "\n";
  return path;
}

}  // namespace

namespace {

// Runs `queries` (all 22 when empty, or SIRIUS_TPCH_QUERIES=q15,q21 while debugging) from the
// corpus on `gpus` and checks rows and Arrow types against the expected results. The corpus
// directory (SIRIUS_TPCH_SUBSTRAIT_DIR) is the --out directory of tools/substrait_gen's
// sirius_substrait_gen, holding qNN.substrait.json and result-types.json; the data directory
// (SIRIUS_TPCH_DATA_DIR) holds input/ (TPC-H SF1 Parquet) and expected_results/ (qNN.csv).
// The test skips when either variable is unset; once both are set, every file it needs must
// exist.
void run_tpch_corpus(std::vector<int> const& gpus,
                     std::vector<std::string> queries,
                     std::uint64_t gpu_bytes,
                     std::uint64_t host_bytes)
{
  auto const corpus = env_dir("SIRIUS_TPCH_SUBSTRAIT_DIR");
  auto const data   = env_dir("SIRIUS_TPCH_DATA_DIR");
  if (corpus.empty() || data.empty()) {
    SKIP("set SIRIUS_TPCH_SUBSTRAIT_DIR and SIRIUS_TPCH_DATA_DIR to a TPC-H SF1 corpus and data");
  }
  auto const result_types = corpus / "result-types.json";

  if (queries.empty()) {
    if (char const* only = std::getenv("SIRIUS_TPCH_QUERIES")) {
      std::stringstream names(only);
      for (std::string q; std::getline(names, q, ',');) {
        queries.push_back(q);
      }
    } else {
      for (int n = 1; n <= 22; ++n) {
        char name[8];
        std::snprintf(name, sizeof(name), "q%02d", n);
        queries.emplace_back(name);
      }
    }
  }
  // A missing file is an incomplete corpus or a misspelled query name (q1 for q01): fail.
  auto const require_file = [](fs::path const& path) {
    INFO("missing " << path.string());
    REQUIRE(fs::is_regular_file(path));
  };
  require_file(result_types);
  require_file(data / "input" / "lineitem.parquet");
  for (auto const& q : queries) {
    require_file(corpus / (q + ".substrait.json"));
    require_file(data / "expected_results" / (q + ".csv"));
  }

  arrow_source csv_reader;
  sirius::test::scratch_dir scratch("ffi_embedder_tpch");
  auto ctx =
    sirius::ffi::make_context_with_gpus(gpus, write_tpch_config(scratch, gpu_bytes, host_bytes));
  for (auto const& q : queries) {
    INFO(q << " on " << gpus.size() << " GPU(s)");
    auto const plan_path = corpus / (q + ".substrait.json");
    auto const plan      = local_files_corpus_plan(plan_path, data / "input");
    ArrowArrayStream stream{};
    auto const start = std::chrono::steady_clock::now();
    ctx->execute_substrait_json(plan, reinterpret_cast<std::uintptr_t>(&stream));
    std::vector<std::string> formats;
    auto actual       = collect_tpch_table(stream, formats);
    auto const millis = std::chrono::duration_cast<std::chrono::milliseconds>(
                          std::chrono::steady_clock::now() - start)
                          .count();

    auto const types = corpus_physical_types(result_types, q);
    REQUIRE(formats.size() == types.size());
    for (std::size_t c = 0; c < types.size(); ++c) {
      INFO("column " << c << ": Arrow " << formats[c] << ", expected " << types[c]);
      REQUIRE(arrow_matches_physical_type(formats[c], types[c]));
    }

    auto expected = read_expected_csv(csv_reader, data / "expected_results" / (q + ".csv"));
    if (!corpus_plan_is_ordered(plan_path)) {
      actual   = sorted_rows(std::move(actual));
      expected = sorted_rows(std::move(expected));
    }
    require_matches(actual, expected, formats);
    std::printf("[tpch] %s on %zu GPU(s): PASS, %zu rows, %lld ms\n",
                q.c_str(),
                gpus.size(),
                actual.size(),
                static_cast<long long>(millis));
  }
}

}  // namespace

// Not [multi_gpu], so scripts/run_unit_tests.py runs it in a shard: two per GPU, each sized for
// a fifth of a 15 GiB card (integration-shard.yaml). SF1 Q1/Q6 fit in that 3 GiB.
TEST_CASE("FFI execute_substrait_json runs TPC-H Q1 and Q6 from a corpus on one GPU",
          "[isolated_context][sirius_ffi][tpch]")
{
  run_tpch_corpus({0}, {"q01", "q06"}, 3ull << 30, 4ull << 30);
}

TEST_CASE("FFI execute_substrait_json runs all 22 TPC-H queries from a corpus on two GPUs",
          "[isolated_context][sirius_ffi][tpch][multi_gpu]")
{
  if (!sirius::test::has_gpus(2)) { SKIP("needs 2 GPUs"); }
  run_tpch_corpus({0, 1}, {}, 16ull << 30, 8ull << 30);
}

// ---------------------------------------------------------------------------
// push_arrow: several columns, aggregates over pushed data, declared-type edges
// ---------------------------------------------------------------------------

namespace {

struct stream_column {
  std::string name;
  std::function<void(substrait::Type&)> type;
};

// A read of input stream `stream_id` with `columns`, in declaration order.
substrait::Rel stream_read_rel(std::uint64_t stream_id, std::vector<stream_column> const& columns)
{
  substrait::Rel rel;
  auto* read = rel.mutable_read();
  read->mutable_named_table()->add_names(*sirius::ffi::stream_view_name(stream_id));
  auto* schema = read->mutable_base_schema();
  auto* st     = schema->mutable_struct_();
  st->set_nullability(::substrait::Type_Nullability_NULLABILITY_REQUIRED);
  for (auto const& column : columns) {
    schema->add_names(column.name);
    column.type(*st->add_types());
  }
  return rel;
}

std::string multi_column_read_json(std::uint64_t stream_id,
                                   std::vector<stream_column> const& columns)
{
  substrait::Plan plan;
  auto* root = plan.add_relations()->mutable_root();
  for (auto const& column : columns) {
    root->add_names(column.name);
  }
  *root->mutable_input() = stream_read_rel(stream_id, columns);
  return to_json(plan);
}

struct measure {
  std::string function;               // extension function name, e.g. "sum" or "count"
  std::optional<std::int32_t> field;  // argument column; none for count(*)
};

// SELECT <measures> FROM <read of stream_id> (no grouping).
std::string aggregate_json(std::uint64_t stream_id,
                           std::vector<stream_column> const& columns,
                           std::vector<measure> const& measures)
{
  substrait::Plan plan;
  auto* root            = plan.add_relations()->mutable_root();
  auto* agg             = root->mutable_input()->mutable_aggregate();
  *agg->mutable_input() = stream_read_rel(stream_id, columns);
  agg->add_groupings();
  std::uint32_t anchor = 0;
  for (auto const& m : measures) {
    ++anchor;
    auto* fn = plan.add_extensions()->mutable_extension_function();
    fn->set_function_anchor(anchor);
    fn->set_name(m.function);
    root->add_names(m.function + std::to_string(anchor));
    auto* call = agg->add_measures()->mutable_measure();
    call->set_function_reference(anchor);
    if (m.field) {
      auto* ref = call->add_arguments()->mutable_value()->mutable_selection();
      ref->mutable_direct_reference()->mutable_struct_field()->set_field(*m.field);
      ref->mutable_root_reference();
    }
  }
  return to_json(plan);
}

auto const kBigint = [](substrait::Type& t) { t.mutable_i64()->set_nullability(kNullable); };
auto const kString = [](substrait::Type& t) { t.mutable_string()->set_nullability(kNullable); };

std::function<void(substrait::Type&)> decimal_type(int precision, int scale)
{
  return [precision, scale](substrait::Type& t) {
    auto* d = t.mutable_decimal();
    d->set_precision(precision);
    d->set_scale(scale);
    d->set_nullability(kNullable);
  };
}

}  // namespace

TEST_CASE("FFI push_arrow matches several columns by position, not by name",
          "[isolated_context][sirius_ffi]")
{
  auto ctx = sirius::ffi::make_context_from_config(isolated_memory_config_path().string());
  arrow_source source;
  std::vector<stream_column> const columns = {
    {"a", kBigint}, {"b", kString}, {"c", decimal_type(15, 2)}};

  auto fragment = sirius::ffi::make_fragment(*ctx);
  fragment->declare_input_column(0, "a", "BIGINT");
  fragment->declare_input_column(0, "b", "VARCHAR");
  fragment->declare_input_column(0, "c", "DECIMAL(15,2)");
  fragment->build_json(multi_column_read_json(0, columns));

  // Arrow names differ from the declared ones; only position and type matter.
  auto input = concat_streams(
    {source.stream("SELECT * FROM (VALUES (1::BIGINT, 'one', 1.10::DECIMAL(15,2)), "
                   "(2::BIGINT, NULL, NULL)) t(x, y, z)"),
     source.stream("SELECT * FROM (VALUES (NULL::BIGINT, 'three', -3.30::DECIMAL(15,2))) "
                   "t(other, names, here)")});
  REQUIRE(push(*fragment, 0, input) == 2);
  fragment->run();

  table const expected = {
    {std::nullopt, "three", "-3.30"}, {"1", "one", "1.10"}, {"2", std::nullopt, std::nullopt}};
  REQUIRE(sorted_rows(result_table(*fragment)) == expected);
}

TEST_CASE("FFI push_arrow rejects columns in the wrong order and names the column",
          "[isolated_context][sirius_ffi]")
{
  auto ctx = sirius::ffi::make_context_from_config(isolated_memory_config_path().string());
  arrow_source source;
  auto fragment = sirius::ffi::make_fragment(*ctx);
  fragment->declare_input_column(0, "a", "BIGINT");
  fragment->declare_input_column(0, "b", "VARCHAR");
  fragment->build_json(multi_column_read_json(0, {{"a", kBigint}, {"b", kString}}));

  // Column 0 is fine; column 1 is int64 where VARCHAR was declared.
  auto s = source.stream("SELECT 1::BIGINT AS a, 2::BIGINT AS b");
  REQUIRE_THROWS_WITH(push(*fragment, 0, s),
                      Catch::Matchers::ContainsSubstring("column 1") &&
                        Catch::Matchers::ContainsSubstring("int64") &&
                        Catch::Matchers::ContainsSubstring("VARCHAR"));
  REQUIRE(s.release == nullptr);
  // Swapped order: column 0 is the offender.
  auto t = source.stream("SELECT 'x'::VARCHAR AS a, 1::BIGINT AS b");
  REQUIRE_THROWS_AS(push(*fragment, 0, t), std::invalid_argument);

  // Nothing moved: a correct stream is still accepted.
  auto good = source.stream("SELECT 7::BIGINT AS a, 'ok'::VARCHAR AS b");
  REQUIRE(push(*fragment, 0, good) == 1);
  fragment->run();
  REQUIRE(result_table(*fragment) == table{{"7", "ok"}});
}

TEST_CASE("FFI push_arrow feeds an aggregate across batches with nulls",
          "[isolated_context][sirius_ffi]")
{
  auto ctx = sirius::ffi::make_context_from_config(isolated_memory_config_path().string());
  arrow_source source;
  std::vector<stream_column> const columns = {{"price", decimal_type(15, 2)}, {"id", kBigint}};

  auto fragment = sirius::ffi::make_fragment(*ctx);
  fragment->declare_input_column(0, "price", "DECIMAL(15,2)");
  fragment->declare_input_column(0, "id", "BIGINT");
  fragment->build_json(aggregate_json(
    0, columns, {{"sum", 0}, {"count", 0}, {"count", std::nullopt}, {"min", 1}, {"max", 1}}));

  // 100000 + 100000 rows; every 10th price is NULL.
  auto const batch = [](int from, int to) {
    return "SELECT CASE WHEN i % 10 = 0 THEN NULL ELSE (i / 100)::DECIMAL(15,2) END AS price, "
           "i::BIGINT AS id FROM range(" +
           std::to_string(from) + ", " + std::to_string(to) + ") r(i)";
  };
  auto input =
    concat_streams({source.stream(batch(1, 100001)), source.stream(batch(100001, 200001))});
  REQUIRE(push(*fragment, 0, input) == 2);
  fragment->run();

  // Reference: the same aggregate on the CPU.
  auto const reference = [&] {
    sirius::test::scoped_sirius_disable disable;
    auto r = source.con->Query(
      "SELECT sum(price)::VARCHAR, count(price), count(*), min(id), max(id) FROM (" +
      batch(1, 200001) + ")");
    REQUIRE_FALSE(r->HasError());
    return std::vector<std::string>{r->GetValue(0, 0).ToString(),
                                    r->GetValue(1, 0).ToString(),
                                    r->GetValue(2, 0).ToString(),
                                    r->GetValue(3, 0).ToString(),
                                    r->GetValue(4, 0).ToString()};
  }();
  auto const actual = result_table(*fragment);
  REQUIRE(actual.size() == 1);
  REQUIRE(actual[0].size() == 5);
  for (std::size_t c = 0; c < 5; ++c) {
    INFO("measure " << c);
    REQUIRE(actual[0][c] == cell{reference[c]});
  }
  REQUIRE(reference[2] == "200000");
}

TEST_CASE("FFI push_arrow of a zero-row stream feeds an aggregate of nothing",
          "[isolated_context][sirius_ffi]")
{
  auto ctx = sirius::ffi::make_context_from_config(isolated_memory_config_path().string());
  arrow_source source;
  auto input = concat_streams({source.stream("SELECT 1.00::DECIMAL(15,2) AS a")},
                              /*schema_only=*/true);

  auto fragment = sirius::ffi::make_fragment(*ctx);
  fragment->declare_input_column(0, "a", "DECIMAL(15,2)");
  fragment->build_json(
    aggregate_json(0, {{"a", decimal_type(15, 2)}}, {{"count", std::nullopt}, {"sum", 0}}));
  REQUIRE(push(*fragment, 0, input) == 0);
  fragment->run();
  REQUIRE(result_table(*fragment) == table{{"0", std::nullopt}});
}

TEST_CASE("FFI push_arrow converts decimals to other declared precisions and scales",
          "[isolated_context][sirius_ffi]")
{
  auto ctx = sirius::ffi::make_context_from_config(isolated_memory_config_path().string());
  std::vector<round_trip_case> const cases = {
    // Upscale: scale 0 -> 2.
    {"DECIMAL(15,2)",
     decimal_type(15, 2),
     "SELECT * FROM (VALUES (12::DECIMAL(15,0)), (NULL)) t(a)",
     "SELECT * FROM (VALUES (-3::DECIMAL(15,0))) t(a)",
     {std::nullopt, "-3.00", "12.00"},
     "",
     "d:15,0,128"},
    // A 128-bit carrier, values beyond int64.
    {"DECIMAL(38,4)",
     decimal_type(38, 4),
     "SELECT * FROM (VALUES (12345678901234567890.1234::DECIMAL(38,4)), (NULL)) t(a)",
     "SELECT * FROM (VALUES (-0.0001::DECIMAL(38,4))) t(a)",
     {std::nullopt, "-0.0001", "12345678901234567890.1234"},
     "",
     "d:38,4,128"},
    // A 32-bit carrier, narrowed from a wide source whose values fit.
    {"DECIMAL(9,2)",
     decimal_type(9, 2),
     "SELECT * FROM (VALUES (1234567.89::DECIMAL(38,2)), (NULL)) t(a)",
     "SELECT * FROM (VALUES (-1234567.89::DECIMAL(38,2))) t(a)",
     {std::nullopt, "-1234567.89", "1234567.89"},
     "",
     "d:38,2,128"},
  };
  for (auto const& c : cases) {
    run_round_trip(*ctx, c);
  }
}

TEST_CASE("FFI push_arrow decimal overflow checks only non-null values at the boundary",
          "[isolated_context][sirius_ffi]")
{
  auto ctx = sirius::ffi::make_context_from_config(isolated_memory_config_path().string());
  // The largest DECIMAL(15,2) value, from a source whose precision cannot bound it statically.
  run_round_trip(*ctx,
                 {"DECIMAL(15,2)",
                  decimal_type(15, 2),
                  "SELECT * FROM (VALUES (9999999999999.99::DECIMAL(38,2)), (NULL)) t(a)",
                  "SELECT * FROM (VALUES (-9999999999999.99::DECIMAL(38,2))) t(a)",
                  {std::nullopt, "-9999999999999.99", "9999999999999.99"},
                  "",
                  "d:38,2,128"});
  // An all-null wide column has nothing to overflow.
  run_round_trip(*ctx,
                 {"DECIMAL(15,2)",
                  decimal_type(15, 2),
                  "SELECT NULL::DECIMAL(38,2) AS a",
                  "SELECT NULL::DECIMAL(38,2) AS a",
                  {std::nullopt, std::nullopt},
                  "",
                  "d:38,2,128"});

  // One past the limit, negative side, in the second batch.
  arrow_source source;
  auto fragment = sirius::ffi::make_fragment(*ctx);
  fragment->declare_input_column(0, "a", "DECIMAL(15,2)");
  fragment->build_json(stream_read_json(0, decimal_type(15, 2)));
  auto s = concat_streams({source.stream("SELECT 1.00::DECIMAL(38,2) AS a"),
                           source.stream("SELECT -10000000000000.00::DECIMAL(38,2) AS a")});
  REQUIRE_THROWS_WITH(push(*fragment, 0, s),
                      Catch::Matchers::ContainsSubstring("overflow") &&
                        Catch::Matchers::ContainsSubstring("column 0"));
  REQUIRE(s.release == nullptr);

  // The first batch already went in, so the input is incomplete: the Fragment cannot finish.
  REQUIRE_THROWS(fragment->close_input(0, 0));
  REQUIRE_THROWS(fragment->run());
}

TEST_CASE("FFI push_arrow rejects declared types outside the supported set",
          "[isolated_context][sirius_ffi]")
{
  auto ctx = sirius::ffi::make_context_from_config(isolated_memory_config_path().string());
  arrow_source source;

  struct unsupported {
    std::string declared;
    std::function<void(substrait::Type&)> substrait_type;
    std::string sql;
  };
  std::vector<unsupported> const cases = {
    {"SMALLINT",
     [](substrait::Type& t) { t.mutable_i16()->set_nullability(kNullable); },
     "SELECT 1::SMALLINT AS a"},
    {"REAL",
     [](substrait::Type& t) { t.mutable_fp32()->set_nullability(kNullable); },
     "SELECT 1::REAL AS a"},
    // DuckDB stores DECIMAL(p<=4) as INT16.
    {"DECIMAL(4,2)", decimal_type(4, 2), "SELECT 1.25::DECIMAL(4,2) AS a"},
    // A matching-looking source of the wrong width is still not converted.
    {"INTEGER",
     [](substrait::Type& t) { t.mutable_i32()->set_nullability(kNullable); },
     "SELECT 1::BIGINT AS a"},
    {"DATE",
     [](substrait::Type& t) { t.mutable_date()->set_nullability(kNullable); },
     "SELECT TIMESTAMP '2000-01-01 00:00:00' AS a"},
  };
  for (auto const& c : cases) {
    INFO("declared " << c.declared << " from " << c.sql);
    auto fragment = sirius::ffi::make_fragment(*ctx);
    fragment->declare_input_column(0, "a", c.declared);
    fragment->build_json(stream_read_json(0, c.substrait_type));
    auto s = source.stream(c.sql);
    REQUIRE_THROWS_WITH(push(*fragment, 0, s),
                        Catch::Matchers::ContainsSubstring("stream 0") &&
                          Catch::Matchers::ContainsSubstring("column 0") &&
                          Catch::Matchers::ContainsSubstring("declared"));
    REQUIRE(s.release == nullptr);
    auto again = source.stream(c.sql);
    REQUIRE_THROWS_AS(push(*fragment, 0, again), std::invalid_argument);
  }
}

TEST_CASE("FFI push_arrow rejects a null or already-released stream",
          "[isolated_context][sirius_ffi]")
{
  auto ctx      = sirius::ffi::make_context_from_config(isolated_memory_config_path().string());
  auto fragment = sirius::ffi::make_fragment(*ctx);
  fragment->declare_input_column(0, "a", "BIGINT");
  fragment->build_json(i64_read_json(0));

  REQUIRE_THROWS_AS(fragment->push_arrow(0, 0, 0), std::invalid_argument);
  ArrowArrayStream released{};
  REQUIRE_THROWS_AS(push(*fragment, 0, released), std::invalid_argument);

  // Neither consumed the sender.
  arrow_source source;
  auto good = source.stream("SELECT 3::BIGINT AS a");
  REQUIRE(push(*fragment, 0, good) == 1);
  fragment->run();
  REQUIRE(result_i64s(*fragment) == std::vector<std::int64_t>{3});
}

TEST_CASE("FFI push_arrow works on a two-GPU Context", "[isolated_context][sirius_ffi][multi_gpu]")
{
  if (!sirius::test::has_gpus(2)) { SKIP("needs two GPUs"); }
  sirius::test::scratch_dir scratch("ffi_embedder_push_two_gpus");
  auto ctx = sirius::ffi::make_context_with_gpus({0, 1}, write_gpu_list_config(scratch));
  arrow_source source;

  auto fragment = sirius::ffi::make_fragment(*ctx);
  fragment->declare_input_column(0, "a", "BIGINT");
  fragment->declare_input_sender(0, 0);
  fragment->declare_input_sender(0, 1);
  fragment->build_json(aggregate_json(0, {{"a", kBigint}}, {{"count", std::nullopt}, {"min", 0}}));
  auto first  = source.stream("SELECT i::BIGINT AS a FROM range(10, 1010) r(i)");
  auto second = source.stream("SELECT i::BIGINT AS a FROM range(5, 15) r(i)");
  REQUIRE(push(*fragment, 0, first, 0) == 1);
  REQUIRE(push(*fragment, 0, second, 1) == 1);
  fragment->run();
  REQUIRE(result_table(*fragment) == table{{"1010", "5"}});
}

TEST_CASE("FFI push_arrow closes an input whose two senders both end empty",
          "[isolated_context][sirius_ffi]")
{
  auto ctx = sirius::ffi::make_context_from_config(isolated_memory_config_path().string());
  arrow_source source;
  auto empty = concat_streams({source.stream("SELECT 1::BIGINT AS a")}, /*schema_only=*/true);

  auto fragment = sirius::ffi::make_fragment(*ctx);
  fragment->declare_input_column(0, "a", "BIGINT");
  fragment->declare_input_sender(0, 0);
  fragment->declare_input_sender(0, 1);
  fragment->build_json(aggregate_json(0, {{"a", kBigint}}, {{"count", std::nullopt}, {"max", 0}}));
  REQUIRE(push(*fragment, 0, empty, 0) == 0);
  fragment->close_input(0, 1);  // a remote sender that sent no batches
  fragment->run();
  REQUIRE(result_table(*fragment) == table{{"0", std::nullopt}});
}

namespace {

// Wraps `inner` and turns its k-th batch into a slice of it: the parent struct gets
// slices[k] = {offset, length} while its children keep their full, longer length.
struct slice_state {
  ArrowArrayStream inner;
  std::vector<std::pair<std::int64_t, std::int64_t>> slices;
  std::size_t next{0};
  std::string error;
};

int slice_get_schema(ArrowArrayStream* self, ArrowSchema* out)
{
  auto* state = static_cast<slice_state*>(self->private_data);
  return state->inner.get_schema(&state->inner, out);
}

int slice_get_next(ArrowArrayStream* self, ArrowArray* out)
{
  auto* state = static_cast<slice_state*>(self->private_data);
  if (int rc = state->inner.get_next(&state->inner, out); rc != 0) {
    state->error = state->inner.get_last_error(&state->inner);
    return rc;
  }
  if (out->release == nullptr) { return 0; }
  if (state->next == state->slices.size() ||
      state->slices[state->next].first + state->slices[state->next].second > out->length) {
    out->release(out);
    state->error = "slice_get_next: no slice for this batch";
    return EINVAL;
  }
  auto const [offset, length] = state->slices[state->next++];
  out->offset += offset;
  out->length = length;
  return 0;
}

const char* slice_last_error(ArrowArrayStream* self)
{
  return static_cast<slice_state*>(self->private_data)->error.c_str();
}

void slice_release(ArrowArrayStream* self)
{
  auto* state = static_cast<slice_state*>(self->private_data);
  release_if(state->inner);
  delete state;
  self->release = nullptr;
}

ArrowArrayStream sliced_stream(ArrowArrayStream inner,
                               std::vector<std::pair<std::int64_t, std::int64_t>> slices)
{
  ArrowArrayStream out = {};
  out.get_schema       = slice_get_schema;
  out.get_next         = slice_get_next;
  out.get_last_error   = slice_last_error;
  out.release          = slice_release;
  out.private_data     = new slice_state{inner, std::move(slices), 0, {}};
  return out;
}

}  // namespace

TEST_CASE("FFI push_arrow reads only the rows of a sliced batch", "[isolated_context][sirius_ffi]")
{
  auto ctx = sirius::ffi::make_context_from_config(isolated_memory_config_path().string());
  arrow_source source;
  std::vector<stream_column> const columns = {{"a", kBigint}, {"b", kString}};

  auto fragment = sirius::ffi::make_fragment(*ctx);
  fragment->declare_input_column(0, "a", "BIGINT");
  fragment->declare_input_column(0, "b", "VARCHAR");
  fragment->build_json(multi_column_read_json(0, columns));

  // Batch 1 keeps rows 2..4 of 0..5, batch 2 rows 1..2 of 10..13; NULLs fall on both sides of
  // each slice edge, so validity bits are read at the parent's offset too.
  auto batch_1 = source.stream(
    "SELECT * FROM (VALUES (0::BIGINT, 'zero'), (1, NULL), (2, 'two'), (NULL, 'three'), "
    "(4, NULL), (5, 'five')) t(a, b)");
  auto batch_2 = source.stream(
    "SELECT * FROM (VALUES (10::BIGINT, NULL), (11, 'a string longer than twelve bytes'), "
    "(NULL, 'twelve'), (13, 'thirteen')) t(a, b)");
  auto input = sliced_stream(concat_streams({batch_1, batch_2}), {{2, 3}, {1, 2}});
  REQUIRE(push(*fragment, 0, input) == 2);
  REQUIRE(input.release == nullptr);
  fragment->run();

  table const expected = {{std::nullopt, "three"},
                          {std::nullopt, "twelve"},
                          {"11", "a string longer than twelve bytes"},
                          {"2", "two"},
                          {"4", std::nullopt}};
  REQUIRE(sorted_rows(result_table(*fragment)) == expected);
}

namespace {

// relations[0] reads input stream `stream_id` (one BIGINT column `a`); the root is the inner
// self-join ReferenceRel 0 ⋈ ReferenceRel 0 ON l.a = r.a. One declared stream and one read rel,
// but two plan leaves once the shared relation is inlined, so the stream fans out to both.
std::string shared_read_self_join_json(std::uint64_t stream_id)
{
  substrait::Plan plan;
  auto* equal = plan.add_extensions()->mutable_extension_function();
  equal->set_function_anchor(1);
  equal->set_name("equal:i64_i64");
  *plan.add_relations()->mutable_rel() = stream_read_rel(stream_id, {{"a", kBigint}});
  auto* root                           = plan.add_relations()->mutable_root();
  root->add_names("l");
  root->add_names("r");
  auto* join = root->mutable_input()->mutable_join();
  join->set_type(substrait::JoinRel_JoinType_JOIN_TYPE_INNER);
  join->mutable_left()->mutable_reference()->set_subtree_ordinal(0);
  join->mutable_right()->mutable_reference()->set_subtree_ordinal(0);
  auto* call = join->mutable_expression()->mutable_scalar_function();
  call->set_function_reference(1);
  call->mutable_output_type()->mutable_bool_()->set_nullability(kNullable);
  for (std::int32_t field : {0, 1}) {
    auto* ref = call->add_arguments()->mutable_value()->mutable_selection();
    ref->mutable_direct_reference()->mutable_struct_field()->set_field(field);
    ref->mutable_root_reference();
  }
  return to_json(plan);
}

// Each of [0, n) must come back exactly once, joined with itself: only rows that reached both
// leaves can match.
void require_self_join_of_range(table const& rows, int n)
{
  REQUIRE(rows.size() == static_cast<std::size_t>(n));
  for (auto const& row : rows) {
    REQUIRE(row.size() == 2);
    REQUIRE(row[0] == row[1]);
  }
  std::vector<cell> expected;
  for (int i = 0; i < n; ++i) {
    expected.emplace_back(std::to_string(i));
  }
  std::sort(expected.begin(), expected.end());
  REQUIRE(sorted_column(rows) == expected);
}

}  // namespace

TEST_CASE("FFI push_arrow fans one stream out to a shared relation joined with itself",
          "[isolated_context][sirius_ffi]")
{
  auto ctx = sirius::ffi::make_context_from_config(isolated_memory_config_path().string());
  arrow_source source;

  auto fragment = sirius::ffi::make_fragment(*ctx);
  fragment->declare_input_column(0, "a", "BIGINT");
  fragment->build_json(shared_read_self_join_json(0));
  // Two batches: every batch, and the end of the stream, must reach both leaves.
  auto input =
    concat_streams({source.stream("SELECT i::BIGINT AS a FROM range(0, 6000) r(i)"),
                    source.stream("SELECT i::BIGINT AS a FROM range(6000, 10000) r(i)")});
  REQUIRE(push(*fragment, 0, input) == 2);
  fragment->run();
  require_self_join_of_range(result_table(*fragment), 10000);
}

TEST_CASE("FFI push_arrow fans one stream out to two leaves on a two-GPU Context",
          "[isolated_context][sirius_ffi][multi_gpu]")
{
  if (!sirius::test::has_gpus(2)) { SKIP("needs two GPUs"); }
  sirius::test::scratch_dir scratch("ffi_embedder_fan_out_two_gpus");
  auto ctx = sirius::ffi::make_context_with_gpus({0, 1}, write_gpu_list_config(scratch));
  arrow_source source;

  auto fragment = sirius::ffi::make_fragment(*ctx);
  fragment->declare_input_column(0, "a", "BIGINT");
  fragment->declare_input_sender(0, 0);
  fragment->declare_input_sender(0, 1);
  fragment->build_json(shared_read_self_join_json(0));
  // Two senders: the input ends for both leaves only after both senders closed it.
  auto first  = source.stream("SELECT i::BIGINT AS a FROM range(0, 7000) r(i)");
  auto second = source.stream("SELECT i::BIGINT AS a FROM range(7000, 10000) r(i)");
  REQUIRE(push(*fragment, 0, first, 0) == 1);
  REQUIRE(push(*fragment, 0, second, 1) == 1);
  fragment->run();
  require_self_join_of_range(result_table(*fragment), 10000);
}

TEST_CASE("FFI build_json of a plan that does not read the declared stream rolls back",
          "[isolated_context][sirius_ffi]")
{
  sirius::test::scratch_dir scratch("ffi_embedder_build_json_rollback");
  auto const path = scratch.file("ids.parquet");
  write_ids_parquet(path);

  auto ctx      = sirius::ffi::make_context_from_config(isolated_memory_config_path().string());
  auto fragment = sirius::ffi::make_fragment(*ctx);
  fragment->declare_input_column(0, "a", "BIGINT");
  // Valid JSON, valid plan, but it reads a parquet file instead of the declared stream.
  REQUIRE_THROWS(fragment->build_json(to_json(parse_plan(local_files_plan(path)))));
  // An empty document is not a plan either.
  REQUIRE_THROWS_AS(fragment->build_json(""), std::invalid_argument);

  fragment->build_json(i64_read_json(0));
  arrow_source source;
  auto s = source.stream("SELECT 11::BIGINT AS a");
  REQUIRE(push(*fragment, 0, s) == 1);
  fragment->run();
  REQUIRE(result_i64s(*fragment) == std::vector<std::int64_t>{11});

  // The Context runs JSON plans afterwards too.
  ArrowArrayStream out{};
  ctx->execute_substrait_json(to_json(parse_plan(local_files_plan(path))),
                              reinterpret_cast<std::uintptr_t>(&out));
  REQUIRE(collect_i64_column(out) == std::vector<std::int64_t>{1, 2, 3, 4, 5});
}

TEST_CASE("FFI execute_substrait_json runs an aggregate plan and rejects an empty document",
          "[isolated_context][sirius_ffi]")
{
  sirius::test::scratch_dir scratch("ffi_embedder_execute_json_agg");
  auto const path = scratch.file("ids.parquet");
  write_ids_parquet(path);

  auto ctx = sirius::ffi::make_context_from_config(isolated_memory_config_path().string());
  ArrowArrayStream out{};
  auto const addr = reinterpret_cast<std::uintptr_t>(&out);
  REQUIRE_THROWS_AS(ctx->execute_substrait_json("", addr), std::invalid_argument);
  REQUIRE(out.release == nullptr);

  // count(*), min(a), max(a) over the parquet fixture.
  substrait::Plan plan  = parse_plan(local_files_plan(path));
  auto* root            = plan.mutable_relations(0)->mutable_root();
  substrait::Rel read   = root->input();
  auto* agg             = root->mutable_input()->mutable_aggregate();
  *agg->mutable_input() = read;
  agg->add_groupings();
  root->clear_names();
  std::vector<measure> const measures = {{"count", std::nullopt}, {"min", 0}, {"max", 0}};
  std::uint32_t anchor                = 0;
  for (auto const& m : measures) {
    ++anchor;
    auto* fn = plan.add_extensions()->mutable_extension_function();
    fn->set_function_anchor(anchor);
    fn->set_name(m.function);
    root->add_names(m.function);
    auto* call = agg->add_measures()->mutable_measure();
    call->set_function_reference(anchor);
    if (m.field) {
      auto* ref = call->add_arguments()->mutable_value()->mutable_selection();
      ref->mutable_direct_reference()->mutable_struct_field()->set_field(*m.field);
      ref->mutable_root_reference();
    }
  }
  ctx->execute_substrait_json(to_json(plan, /*snake_case=*/true), addr);
  REQUIRE(collect_table(out) == table{{"5", "1", "5"}});
}

// ---------------------------------------------------------------------------
// any_value (DuckDB adds it for columns functionally dependent on the group key, e.g. TPC-H
// Q11, Q15 and Q22)
// ---------------------------------------------------------------------------

namespace {

constexpr int kAnyValueRows = 40000;

// Columns: g BIGINT (i % 4), s VARCHAR, d DECIMAL(15,2), f DOUBLE, plus u_s, u_d, u_f for the
// ungrouped cases. Small row groups so the scan yields several batches to merge.
//   g = 0: one value per column, NULL on every third row (the functional-dependency shape).
//   g = 1: every s, d, f is NULL.
//   g = 2: seven distinct values per column, no NULLs.
//   g = 3: NULL in the first half of the file, then one value.
//   u_s / u_f: NULL except in the last 5 rows; u_d: NULL everywhere.
void write_any_value_parquet(std::string const& path)
{
  sirius::test::scoped_sirius_disable disable;
  duckdb::DuckDB db(nullptr);
  duckdb::Connection con(db);
  auto const n      = std::to_string(kAnyValueRows);
  auto const copied = con.Query(
    "COPY (SELECT i % 4 AS g, "
    "CASE WHEN i % 4 = 0 AND i % 3 <> 0 THEN 'only' "
    "     WHEN i % 4 = 2 THEN 'v' || (i % 7) "
    "     WHEN i % 4 = 3 AND i >= " +
    n +
    " / 2 THEN 'late' END AS s, "
    "CASE WHEN i % 4 = 0 AND i % 3 <> 0 THEN 12.34::DECIMAL(15,2) "
    "     WHEN i % 4 = 2 THEN (i % 7)::DECIMAL(15,2) "
    "     WHEN i % 4 = 3 AND i >= " +
    n +
    " / 2 THEN -7.25::DECIMAL(15,2) END AS d, "
    "CASE WHEN i % 4 = 0 AND i % 3 <> 0 THEN 1.5::DOUBLE "
    "     WHEN i % 4 = 2 THEN (i % 7)::DOUBLE "
    "     WHEN i % 4 = 3 AND i >= " +
    n +
    " / 2 THEN 0.25::DOUBLE END AS f, "
    "CASE WHEN i >= " +
    n +
    " - 5 THEN 'tail' END AS u_s, "
    "NULL::DECIMAL(15,2) AS u_d, "
    "CASE WHEN i >= " +
    n +
    " - 5 THEN 2.5::DOUBLE END AS u_f "
    "FROM range(" +
    n + ") t(i) ORDER BY i) TO " + sirius::test::sql_literal(path) +
    " (FORMAT PARQUET, ROW_GROUP_SIZE 2048)");
  REQUIRE(copied);
  REQUIRE_FALSE(copied->HasError());
}

// SELECT [g,] any_value(<fields>) FROM <path> [GROUP BY g], as a Substrait JSON plan.
std::string any_value_json(std::string const& path, bool grouped, std::vector<int> const& fields)
{
  substrait::Plan plan;
  auto* fn = plan.add_extensions()->mutable_extension_function();
  fn->set_function_anchor(1);
  fn->set_name("any_value");

  auto* root = plan.add_relations()->mutable_root();
  auto* agg  = root->mutable_input()->mutable_aggregate();
  auto* item = agg->mutable_input()->mutable_read()->mutable_local_files()->add_items();
  item->set_uri_file(path);
  item->mutable_parquet();

  auto field_ref = [](substrait::Expression& e, int field) {
    auto* ref = e.mutable_selection();
    ref->mutable_direct_reference()->mutable_struct_field()->set_field(field);
    ref->mutable_root_reference();
  };
  auto* grouping = agg->add_groupings();
  if (grouped) {
    field_ref(*agg->add_grouping_expressions(), 0);
    grouping->add_expression_references(0);
    root->add_names("g");
  }
  for (auto const field : fields) {
    auto* call = agg->add_measures()->mutable_measure();
    call->set_function_reference(1);
    field_ref(*call->add_arguments()->mutable_value(), field);
    root->add_names("any_value_" + std::to_string(field));
  }
  return to_json(plan);
}

table execute_table(sirius::ffi::Context& ctx, std::string const& json)
{
  ArrowArrayStream stream{};
  ctx.execute_substrait_json(json, reinterpret_cast<std::uintptr_t>(&stream));
  return collect_table(stream);
}

// Grouped and ungrouped any_value over VARCHAR, DECIMAL and DOUBLE with NULLs.
void check_any_value(sirius::ffi::Context& ctx, std::string const& path)
{
  {
    INFO("grouped");
    auto const rows = sorted_rows(execute_table(ctx, any_value_json(path, true, {1, 2, 3})));
    REQUIRE(rows.size() == 4);
    REQUIRE(rows[0] == std::vector<cell>{"0", "only", "12.34", "1.5"});
    REQUIRE(rows[1] == std::vector<cell>{"1", std::nullopt, std::nullopt, std::nullopt});
    // Any non-NULL value of group 2 is correct.
    REQUIRE(rows[2][0] == cell{"2"});
    REQUIRE(rows[2][1].has_value());
    REQUIRE(rows[2][1]->size() == 2);
    REQUIRE(rows[2][1]->front() == 'v');
    REQUIRE((*rows[2][1])[1] >= '0');
    REQUIRE((*rows[2][1])[1] <= '6');
    REQUIRE(rows[2][2].has_value());
    REQUIRE(rows[2][2]->size() == 4);
    REQUIRE(rows[2][2]->substr(1) == ".00");
    REQUIRE(rows[2][3].has_value());
    REQUIRE(rows[2][3]->size() == 1);
    REQUIRE((*rows[2][3])[0] >= '0');
    REQUIRE((*rows[2][3])[0] <= '6');
    REQUIRE(rows[3] == std::vector<cell>{"3", "late", "-7.25", "0.25"});
  }
  {
    INFO("ungrouped");
    auto const rows = execute_table(ctx, any_value_json(path, false, {4, 5, 6, 1}));
    REQUIRE(rows.size() == 1);
    // u_s and u_f skip the leading NULLs, u_d is all NULL; s is any non-NULL s.
    REQUIRE(rows[0][0] == cell{"tail"});
    REQUIRE(rows[0][1] == std::nullopt);
    REQUIRE(rows[0][2] == cell{"2.5"});
    REQUIRE(rows[0][3].has_value());
  }
}

}  // namespace

TEST_CASE("FFI any_value runs on the GPU grouped and ungrouped over VARCHAR, DECIMAL and DOUBLE",
          "[isolated_context][sirius_ffi]")
{
  sirius::test::scratch_dir scratch("ffi_embedder_any_value");
  auto const path = scratch.file("any_value.parquet");
  write_any_value_parquet(path);
  auto ctx = sirius::ffi::make_context_from_config(isolated_memory_config_path().string());
  check_any_value(*ctx, path);
}

TEST_CASE("FFI any_value merges across two GPUs", "[isolated_context][sirius_ffi][multi_gpu]")
{
  if (!sirius::test::has_gpus(2)) { SKIP("needs 2 GPUs"); }
  sirius::test::scratch_dir scratch("ffi_embedder_any_value_two");
  auto const path = scratch.file("any_value.parquet");
  write_any_value_parquet(path);
  auto ctx = sirius::ffi::make_context_with_gpus({0, 1}, write_gpu_list_config(scratch));
  check_any_value(*ctx, path);
}

namespace {

// Columns: g BIGINT (i % 3), n INTEGER, dt DATE, d DECIMAL(15,2), w DECIMAL(38,4). Each value
// column is functionally dependent on g (one value per group) and NULL on every other row;
// group 0 leads with NULL, group 2 is all NULL.
void write_any_value_types_parquet(std::string const& path)
{
  sirius::test::scoped_sirius_disable disable;
  duckdb::DuckDB db(nullptr);
  duckdb::Connection con(db);
  auto const copied = con.Query(
    "COPY (SELECT i % 3 AS g, "
    "CASE WHEN i % 3 < 2 AND i % 2 = 1 THEN (7 - 10 * (i % 3))::INTEGER END AS n, "
    "CASE WHEN i % 3 < 2 AND i % 2 = 1 THEN DATE '1995-03-15' + (i % 3)::INTEGER END AS dt, "
    "CASE WHEN i % 3 < 2 AND i % 2 = 1 THEN (12.34 - 20 * (i % 3))::DECIMAL(15,2) END AS d, "
    "CASE WHEN i % 3 < 2 AND i % 2 = 1 "
    "     THEN '123456789012345678901234567890.1234'::DECIMAL(38,4) * (1 - 2 * (i % 3)) END AS w "
    "FROM range(12000) t(i) ORDER BY i) TO " +
    sirius::test::sql_literal(path) + " (FORMAT PARQUET, ROW_GROUP_SIZE 2048)");
  REQUIRE(copied);
  REQUIRE_FALSE(copied->HasError());
}

void check_any_value_types(sirius::ffi::Context& ctx, std::string const& path)
{
  {
    INFO("grouped");
    ArrowArrayStream stream{};
    ctx.execute_substrait_json(any_value_json(path, true, {1, 2, 3, 4}),
                               reinterpret_cast<std::uintptr_t>(&stream));
    std::vector<std::string> formats;
    auto const rows = sorted_rows(collect_tpch_table(stream, formats));
    // any_value keeps its input's type, as DuckDB binds it.
    REQUIRE(formats.size() == 5);
    REQUIRE(formats[0] == "l");
    REQUIRE(formats[1] == "i");
    REQUIRE(formats[2] == "tdD");
    REQUIRE(arrow_matches_physical_type(formats[3], "decimal(15,2)"));
    REQUIRE(arrow_matches_physical_type(formats[4], "decimal(38,4)"));
    REQUIRE(rows.size() == 3);
    REQUIRE(rows[0] == std::vector<cell>{
                         "0", "7", "1995-03-15", "12.34", "123456789012345678901234567890.1234"});
    REQUIRE(rows[1] == std::vector<cell>{
                         "1", "-3", "1995-03-16", "-7.66", "-123456789012345678901234567890.1234"});
    REQUIRE(rows[2] ==
            std::vector<cell>{"2", std::nullopt, std::nullopt, std::nullopt, std::nullopt});
  }
  {
    INFO("ungrouped");
    ArrowArrayStream stream{};
    ctx.execute_substrait_json(any_value_json(path, false, {1, 2, 3, 4}),
                               reinterpret_cast<std::uintptr_t>(&stream));
    std::vector<std::string> formats;
    auto const rows = collect_tpch_table(stream, formats);
    REQUIRE(formats.size() == 4);
    REQUIRE(formats[0] == "i");
    REQUIRE(formats[1] == "tdD");
    REQUIRE(arrow_matches_physical_type(formats[2], "decimal(15,2)"));
    REQUIRE(arrow_matches_physical_type(formats[3], "decimal(38,4)"));
    REQUIRE(rows.size() == 1);
    // Two distinct non-NULL values per column across the file; either is correct, NULL is not.
    auto const is_one_of = [](cell const& c, std::string const& a, std::string const& b) {
      return c.has_value() && (*c == a || *c == b);
    };
    REQUIRE(is_one_of(rows[0][0], "7", "-3"));
    REQUIRE(is_one_of(rows[0][1], "1995-03-15", "1995-03-16"));
    REQUIRE(is_one_of(rows[0][2], "12.34", "-7.66"));
    REQUIRE(is_one_of(
      rows[0][3], "123456789012345678901234567890.1234", "-123456789012345678901234567890.1234"));
  }
}

}  // namespace

TEST_CASE("FFI any_value keeps its input type for INTEGER, DATE, DECIMAL(15,2) and DECIMAL(38,4)",
          "[isolated_context][sirius_ffi]")
{
  sirius::test::scratch_dir scratch("ffi_embedder_any_value_types");
  auto const path = scratch.file("any_value_types.parquet");
  write_any_value_types_parquet(path);
  auto ctx = sirius::ffi::make_context_from_config(isolated_memory_config_path().string());
  check_any_value_types(*ctx, path);
}
