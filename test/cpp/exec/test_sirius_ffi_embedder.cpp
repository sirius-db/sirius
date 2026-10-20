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

// Public sirius::ffi Context, Fragment and ExchangeInbox methods only.
// Builds Substrait in the test because the FFI has no SQL helper.
// Covers a result fragment, a relay_from chain, several fragments built before any runs (on one
// and on two threads), build() and run()/drop on different threads, drop after build(), a build()
// that fails after setup, execute_substrait, the ExchangeInbox handle, pin_table/unpin_table,
// and byte-range splits of a pinned file. Spec errors and failed-build rollback live in
// test_streaming_fragment.cpp and test_sirius_ffi_fragment.cpp.

#include "sirius/exception.hpp"
#include "sirius/ffi.hpp"
#include "utils/parquet_fixture_utils.hpp"

#include <cuda_runtime_api.h>

#include <catch.hpp>
#include <duckdb.hpp>
#include <duckdb/common/arrow/arrow.hpp>
#include <substrait/plan.pb.h>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <future>
#include <limits>
#include <memory>
#include <source_location>
#include <string>
#include <thread>
#include <tuple>
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

// One LocalFiles item per `{start, length}` byte range of `path`.
std::string local_files_ranged_plan(
  std::string const& path, std::vector<std::pair<std::uint64_t, std::uint64_t>> const& ranges)
{
  substrait::Plan plan;
  auto* root = plan.add_relations()->mutable_root();
  root->add_names("a");
  auto* files = root->mutable_input()->mutable_read()->mutable_local_files();
  for (auto const& [start, length] : ranges) {
    auto* item = files->add_items();
    item->set_uri_file(path);
    item->set_start(start);
    item->set_length(length);
    item->mutable_parquet();
  }
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

// Writes `values` (a SQL VALUES list of BIGINTs) as a one-column parquet file with column a.
void write_i64_parquet(std::string const& path, std::string const& values)
{
  sirius::test::scoped_sirius_disable disable;
  duckdb::DuckDB db(nullptr);
  duckdb::Connection con(db);
  auto copied = con.Query("COPY (SELECT * FROM (VALUES " + values + ") t(a)) TO " +
                          sirius::test::sql_literal(path) + " (FORMAT PARQUET)");
  REQUIRE(copied);
  REQUIRE_FALSE(copied->HasError());
}

void write_ids_parquet(std::string const& path)
{
  write_i64_parquet(path, "(1::BIGINT), (2::BIGINT), (3::BIGINT), (4::BIGINT), (5::BIGINT)");
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
  auto values = collect_i64_column(stream);
  std::sort(values.begin(), values.end());
  return values;
}

/// sirius_decimal_divide(a, divisor, scale) over the parquet file's BIGINT column `a`, the way the
/// StarRocks CN's translator spells it: a Substrait extension function named
/// `sirius_decimal_divide:<signature>`, an i64 divisor and an i32 scale literal.
std::string decimal_divide_plan(std::string const& path, std::int64_t divisor, std::int32_t scale)
{
  substrait::Plan plan;
  auto* function = plan.add_extensions()->mutable_extension_function();
  function->set_function_anchor(1);
  function->set_name("sirius_decimal_divide:i64_i64_i32");

  auto* root = plan.add_relations()->mutable_root();
  root->add_names("q");
  auto* project = root->mutable_input()->mutable_project();
  project->mutable_common()->mutable_emit()->add_output_mapping(1);
  auto* item = project->mutable_input()->mutable_read()->mutable_local_files()->add_items();
  item->set_uri_file(path);
  item->mutable_parquet();
  auto* call = project->add_expressions()->mutable_scalar_function();
  call->set_function_reference(1);
  call->add_arguments()
    ->mutable_value()
    ->mutable_selection()
    ->mutable_direct_reference()
    ->mutable_struct_field()
    ->set_field(0);
  call->add_arguments()->mutable_value()->mutable_literal()->set_i64(divisor);
  call->add_arguments()->mutable_value()->mutable_literal()->set_i32(scale);
  return serialize_plan(plan);
}

/// The unscaled values of a DECIMAL128 result column.
std::vector<__int128_t> collect_i128_column(ArrowArrayStream& stream)
{
  ArrowSchema schema{};
  REQUIRE(stream.get_schema(&stream, &schema) == 0);
  auto const* column_schema = schema.n_children >= 1 ? schema.children[0] : &schema;
  CHECK(std::string(column_schema->format) == "d:38,8,128");
  if (schema.release) { schema.release(&schema); }

  std::vector<__int128_t> out;
  for (;;) {
    ArrowArray batch{};
    REQUIRE(stream.get_next(&stream, &batch) == 0);
    if (batch.release == nullptr) { break; }
    auto const* col  = first_column(batch);
    auto const* data = static_cast<__int128_t const*>(col->buffers[1]);
    for (std::int64_t i = 0; i < col->length; ++i) {
      out.push_back(data[i + col->offset]);
    }
    batch.release(&batch);
  }
  release_if(stream);
  return out;
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

TEST_CASE("FFI execute_substrait resolves sirius_decimal_divide", "[isolated_context][sirius_ffi]")
{
  sirius::test::scratch_dir scratch("ffi_embedder_decimal_divide");
  auto const path = scratch.file("ids.parquet");
  write_ids_parquet(path);

  // The Context's embedded DuckDB does not load the Sirius extension; it registers the function.
  auto ctx = sirius::ffi::make_context_from_config(isolated_memory_config_path().string());
  ArrowArrayStream stream{};
  ctx->execute_substrait(decimal_divide_plan(path, 7, 8),
                         reinterpret_cast<std::uintptr_t>(&stream));
  // 1/7 .. 5/7 at scale 8, rounded half away from zero.
  REQUIRE(collect_i128_column(stream) ==
          std::vector<__int128_t>{14285714, 28571429, 42857143, 57142857, 71428571});

  // 1/1 at scale 38 is 10^38 units: more than 38 digits. The fragment has no CPU fallback, so the
  // GPU's error reaches the caller.
  ArrowArrayStream overflow{};
  REQUIRE_THROWS_WITH(ctx->execute_substrait(decimal_divide_plan(path, 1, 38),
                                             reinterpret_cast<std::uintptr_t>(&overflow)),
                      Catch::Matchers::ContainsSubstring("more than 38 digits"));
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

TEST_CASE("FFI ExchangeInbox refuses unknown tokens and every call after its Context",
          "[isolated_context][sirius_ffi]")
{
  auto ctx      = sirius::ffi::make_context_from_config(isolated_memory_config_path().string());
  auto exchange = ctx->exchange_inbox();
  REQUIRE(exchange != nullptr);
  REQUIRE(exchange->outstanding() == 0);
  exchange->release(7);

  auto receiver = sirius::ffi::make_fragment(*ctx);
  receiver->declare_input_column(0, "a", "BIGINT");
  receiver->declare_input_sender(0, 0);
  receiver->build(stream_read_plan(0));
  REQUIRE_THROWS_WITH(receiver->push_received(0, 7),
                      Catch::Matchers::ContainsSubstring("holds no received batch"));

  receiver.reset();
  ctx.reset();
  REQUIRE_THROWS(exchange->release(1));
  REQUIRE_THROWS(exchange->outstanding());
}

TEST_CASE("FFI copies a key column out of parked and received batches without taking them",
          "[isolated_context][sirius_ffi]")
{
  sirius::test::scratch_dir scratch("ffi_embedder_key_copy");
  auto const path = scratch.file("ids.parquet");
  write_ids_parquet(path);
  constexpr std::uint64_t frame = 1ull << 20;
  auto ctx                      = sirius::ffi::make_context_with_ring(
    isolated_memory_config_path().string(), frame, 4 * frame, 4 * frame, 0.5, 64 * frame, 1);
  auto exchange = ctx->exchange_inbox();
  auto ring     = ctx->ring_exchange();
  REQUIRE(exchange != nullptr);
  REQUIRE(ring != nullptr);
  auto const scan = [&] {
    auto fragment = sirius::ffi::make_fragment(*ctx);
    fragment->declare_output(0);
    fragment->build(local_files_plan(path));
    fragment->run();
    return fragment;
  };

  // A parked sender's output.
  auto parked            = scan();
  std::uint64_t rows     = 0;
  std::uint64_t distinct = 0;
  std::int64_t min       = std::numeric_limits<std::int64_t>::max();
  std::int64_t max       = std::numeric_limits<std::int64_t>::min();
  parked->output_key_stats(0, 0, rows, distinct, min, max);
  CHECK(std::tuple{rows, distinct, min, max} ==
        std::tuple{std::uint64_t{5}, std::uint64_t{5}, std::int64_t{1}, std::int64_t{5}});
  CHECK_THROWS_WITH(parked->output_key_stats(0, 1, rows, distinct, min, max),
                    Catch::Matchers::ContainsSubstring("out of range"));

  // A batch received from a remote sender and landed, as the CN does on arrival: here the
  // context's ring ships to itself, one frame at a time.
  auto sent            = scan();
  std::uint64_t remote = 0;
  {
    std::uint64_t token = 0, sent_rows = 0, stream_bytes = 0;
    auto const header = sent->export_ring(0, token, sent_rows, stream_bytes);
    REQUIRE(header);
    std::uint8_t tier    = 0;
    std::uint32_t frames = 0;
    auto const landing   = ring->begin(
      reinterpret_cast<std::uintptr_t>(header->data()), header->size(), tier, sent_rows, frames);
    auto const wait = [&](std::uint64_t ticket) {
      while (!ring->done(ticket)) {}
    };
    for (std::uint32_t k = 0; k < frames; ++k) {
      auto const send = ring->lease_send();
      auto const recv = ring->lease_recv();
      REQUIRE(send >= 0);
      REQUIRE(recv >= 0);
      wait(ring->stage(token, k, static_cast<std::uint32_t>(send)));
      auto const len = std::min(frame, stream_bytes - std::uint64_t{k} * frame);
      REQUIRE(cudaMemcpy(reinterpret_cast<void*>(ring->recv_base() +
                                                 static_cast<std::uint64_t>(recv) * frame),
                         reinterpret_cast<void const*>(ring->send_base() +
                                                       static_cast<std::uint64_t>(send) * frame),
                         len,
                         cudaMemcpyDeviceToDevice) == cudaSuccess);
      REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
      ring->release_send(static_cast<std::uint32_t>(send));
      wait(ring->copy_out(landing, k, static_cast<std::uint32_t>(recv), len));
      ring->release_recv(static_cast<std::uint32_t>(recv));
    }
    ring->release_job(token);
    remote = ring->seal(landing);
  }
  rows     = 0;
  distinct = 0;
  min      = std::numeric_limits<std::int64_t>::max();
  max      = std::numeric_limits<std::int64_t>::min();
  exchange->key_stats(remote, 0, rows, distinct, min, max);
  CHECK(std::tuple{rows, distinct, min, max} ==
        std::tuple{std::uint64_t{5}, std::uint64_t{5}, std::int64_t{1}, std::int64_t{5}});

  // A receiver reads both copies; the sources keep their batches.
  auto receiver = sirius::ffi::make_fragment(*ctx);
  receiver->declare_input_column(0, "a", "BIGINT");
  receiver->build(stream_read_plan(0));
  CHECK(receiver->copy_output_column(*parked, 0, 0, 0) == parked->output_batch_count(0));
  receiver->copy_received_column(0, remote, 0);
  receiver->close_input(0, 0);
  receiver->run();
  auto copied = result_i64s(*receiver);
  std::sort(copied.begin(), copied.end());
  CHECK(copied == std::vector<std::int64_t>{1, 1, 2, 2, 3, 3, 4, 4, 5, 5});
  CHECK(parked->output_row_count(0) == 5);
  rows = 0;
  exchange->key_stats(remote, 0, rows, distinct, min, max);
  CHECK(rows == 5);
  CHECK_THROWS_WITH(exchange->key_stats(remote + 1, 0, rows, distinct, min, max),
                    Catch::Matchers::ContainsSubstring("holds no received batch"));
  exchange->release(remote);
  CHECK(exchange->outstanding() == 0);

  // Repeated keys count once: a build side that repeats a few keys is a small filter.
  auto const repeated_path = scratch.file("repeated.parquet");
  {
    sirius::test::scoped_sirius_disable disable;
    duckdb::DuckDB db(nullptr);
    duckdb::Connection con(db);
    auto copied = con.Query("COPY (SELECT (range % 3 + 8)::BIGINT AS a FROM range(1000)) TO " +
                            sirius::test::sql_literal(repeated_path) + " (FORMAT PARQUET)");
    REQUIRE(copied);
    REQUIRE_FALSE(copied->HasError());
  }
  auto repeated = sirius::ffi::make_fragment(*ctx);
  repeated->declare_output(0);
  repeated->build(local_files_plan(repeated_path));
  repeated->run();
  rows     = 0;
  distinct = 0;
  min      = std::numeric_limits<std::int64_t>::max();
  max      = std::numeric_limits<std::int64_t>::min();
  repeated->output_key_stats(0, 0, rows, distinct, min, max);
  CHECK(rows == 1000);
  CHECK(min == 8);
  CHECK(max == 10);
  // Counted per batch, so at least the 3 distinct keys and at most 3 per batch.
  CHECK(distinct >= 3);
  CHECK(distinct <= 3 * repeated->output_batch_count(0));
}

TEST_CASE("FFI interrupt() between runs does nothing, and an Interrupter may outlive its Context",
          "[isolated_context][sirius_ffi]")
{
  sirius::test::scratch_dir scratch("ffi_embedder_interrupt_idle");
  auto const path = scratch.file("ids.parquet");
  write_ids_parquet(path);
  auto const plan = local_files_plan(path);

  auto ctx         = sirius::ffi::make_context_from_config(isolated_memory_config_path().string());
  auto interrupter = ctx->interrupter();
  std::thread([&] {
    interrupter->interrupt();
    ctx->interrupt();
  }).join();

  auto fragment = sirius::ffi::make_fragment(*ctx);
  fragment->build(plan);
  fragment->run();
  REQUIRE(result_i64s(*fragment) == std::vector<std::int64_t>{1, 2, 3, 4, 5});

  // A Fragment must not outlive its Context; only the Interrupter may.
  fragment.reset();
  ctx.reset();
  interrupter->interrupt();
}

TEST_CASE("FFI interrupt() from another thread cancels a run, and the next run works",
          "[isolated_context][sirius_ffi]")
{
  sirius::test::scratch_dir scratch("ffi_embedder_interrupt_run");
  auto const path = scratch.file("ids.parquet");
  write_ids_parquet(path);
  auto const plan = local_files_plan(path);

  auto ctx         = sirius::ffi::make_context_from_config(isolated_memory_config_path().string());
  auto interrupter = ctx->interrupter();
  auto cancelled   = sirius::ffi::make_fragment(*ctx);
  cancelled->build(plan);

  // Interrupt continuously, so the run is cancelled wherever it is. Interrupts keep arriving
  // during the rollback after it, which must still complete for the next run to begin.
  std::atomic<bool> stop{false};
  std::thread interrupting([&] {
    while (!stop.load()) {
      interrupter->interrupt();
      std::this_thread::yield();
    }
  });
  std::exception_ptr run_error;
  try {
    cancelled->run();
  } catch (...) {
    run_error = std::current_exception();
  }
  stop = true;
  interrupting.join();
  REQUIRE(run_error);
  REQUIRE_THROWS_WITH(std::rethrow_exception(run_error),
                      Catch::Matchers::ContainsSubstring("Interrupted"));

  auto next = sirius::ffi::make_fragment(*ctx);
  next->build(plan);
  next->run();
  REQUIRE(result_i64s(*next) == std::vector<std::int64_t>{1, 2, 3, 4, 5});

  ArrowArrayStream stream{};
  ctx->execute_substrait(plan, reinterpret_cast<std::uintptr_t>(&stream));
  REQUIRE(collect_i64_column(stream) == std::vector<std::int64_t>{1, 2, 3, 4, 5});
}

TEST_CASE("FFI pin_table serves scans of the pinned files, and unpin_table releases the pin",
          "[isolated_context][sirius_ffi]")
{
  sirius::test::scratch_dir scratch("ffi_embedder_pin_table");
  auto const low  = scratch.file("ids_low.parquet");
  auto const high = scratch.file("ids_high.parquet");
  write_ids_parquet(low);
  write_i64_parquet(high, "(6::BIGINT), (7::BIGINT), (8::BIGINT)");
  auto const glob = (fs::path(low).parent_path() / "ids_*.parquet").string();

  auto ctx = sirius::ffi::make_context_from_config(isolated_memory_config_path().string());
  std::vector<std::int64_t> const low_rows{1, 2, 3, 4, 5};
  std::vector<std::int64_t> const high_rows{6, 7, 8};
  REQUIRE(execute_i64s(*ctx, local_files_plan(low)) == low_rows);

  // An empty format infers parquet from the glob's suffix; empty columns pin every column.
  REQUIRE(*ctx->pin_table(glob, "gpu", "ids", "", "", "") == "pinned 'ids' tier=gpu");

  // Each scan reads one of the two pinned files, so the pin may serve it only from that file's
  // chunks: the rows are that file's, never the other's.
  REQUIRE(execute_i64s(*ctx, local_files_plan(low)) == low_rows);
  REQUIRE(execute_i64s(*ctx, local_files_plan(high)) == high_rows);
  auto fragment = sirius::ffi::make_fragment(*ctx);
  fragment->build(local_files_plan(high));
  fragment->run();
  auto fragment_rows = result_i64s(*fragment);
  std::sort(fragment_rows.begin(), fragment_rows.end());
  REQUIRE(fragment_rows == high_rows);
  fragment.reset();

  REQUIRE(*ctx->unpin_table("ids") == "unpinned 'ids'");
  // Like CALL unpin_table, removing a name that isn't pinned does nothing.
  REQUIRE(*ctx->unpin_table("ids") == "unpinned 'ids'");
  REQUIRE_NOTHROW(ctx->unpin_table("never_pinned"));
  REQUIRE(execute_i64s(*ctx, local_files_plan(low)) == low_rows);

  // The same name can be pinned again after an unpin, here on the host tier.
  REQUIRE(*ctx->pin_table(glob, "host", "ids", "a", "parquet", "") == "pinned 'ids' tier=host");
  REQUIRE(execute_i64s(*ctx, local_files_plan(high)) == high_rows);
  REQUIRE(*ctx->unpin_table("ids") == "unpinned 'ids'");
}

TEST_CASE("FFI byte-range splits of a pinned file read from disk and return each row once",
          "[isolated_context][sirius_ffi][byte_range]")
{
  sirius::test::scratch_dir scratch("ffi_embedder_pin_byte_ranges");
  auto const path = scratch.file("ids.parquet");
  {
    sirius::test::scoped_sirius_disable disable;
    duckdb::DuckDB db(nullptr);
    duckdb::Connection con(db);
    // Several row groups, so each half of the file owns some of them.
    auto copied =
      con.Query("COPY (SELECT range::BIGINT AS a FROM range(30000)) TO " +
                sirius::test::sql_literal(path) + " (FORMAT PARQUET, ROW_GROUP_SIZE 5000)");
    REQUIRE(copied);
    REQUIRE_FALSE(copied->HasError());
  }
  auto const size = static_cast<std::uint64_t>(fs::file_size(path));
  auto const half = size / 2;

  auto ctx         = sirius::ffi::make_context_from_config(isolated_memory_config_path().string());
  auto const whole = execute_i64s(*ctx, local_files_plan(path));
  REQUIRE(whole.size() == 30000);
  REQUIRE(*ctx->pin_table(path, "gpu", "ids", "", "", "") == "pinned 'ids' tier=gpu");
  REQUIRE(execute_i64s(*ctx, local_files_plan(path)) == whole);

  // Each split is its own scan, as on two compute nodes. Served from the pin, each would return
  // all 30000 rows.
  auto const left  = execute_i64s(*ctx, local_files_ranged_plan(path, {{0, half}}));
  auto const right = execute_i64s(*ctx, local_files_ranged_plan(path, {{half, size - half}}));
  REQUIRE_FALSE(left.empty());
  REQUIRE_FALSE(right.empty());
  REQUIRE(left.size() < whole.size());
  auto both = left;
  both.insert(both.end(), right.begin(), right.end());
  std::sort(both.begin(), both.end());
  REQUIRE(both == whole);

  REQUIRE(execute_i64s(*ctx, local_files_ranged_plan(path, {{0, half}, {half, size - half}})) ==
          whole);
  REQUIRE(*ctx->unpin_table("ids") == "unpinned 'ids'");
}

TEST_CASE("FFI pin_table refuses bad arguments and leaves the Context usable",
          "[isolated_context][sirius_ffi]")
{
  sirius::test::scratch_dir scratch("ffi_embedder_pin_table_refusals");
  auto const path = scratch.file("ids.parquet");
  write_ids_parquet(path);

  auto ctx = sirius::ffi::make_context_from_config(isolated_memory_config_path().string());
  REQUIRE_THROWS_WITH(ctx->pin_table(path, "disk", "ids", "", "", ""),
                      Catch::Matchers::ContainsSubstring("tier='disk'"));
  REQUIRE_THROWS_WITH(ctx->pin_table(path, "", "ids", "", "", ""),
                      Catch::Matchers::ContainsSubstring("'tier'"));
  auto const missing = (fs::path(path).parent_path() / "none_*.parquet").string();
  REQUIRE_THROWS(ctx->pin_table(missing, "gpu", "ids", "", "", ""));

  REQUIRE(execute_i64s(*ctx, local_files_plan(path)) == std::vector<std::int64_t>{1, 2, 3, 4, 5});
}
