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

// FRAG-2 through the FFI Arrow hop: pull_arrow / push_arrow / close_input must
// deliver the same rows as native relay_from. Substrait is built here because
// the public FFI surface has no SQL→Substrait helper.

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

// local_files parquet read — the shape DuckDB's Substrait reader resolves to parquet_scan.
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

// Named-table read of sirius_stream_<id> with schema (a BIGINT).
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

std::unique_ptr<sirius::ffi::Fragment> run_parquet_sender(sirius::ffi::Context& ctx,
                                                          std::string const& plan)
{
  auto sender = sirius::ffi::make_fragment(ctx);
  sender->declare_output(0);
  sender->build(plan);
  sender->run();
  return sender;
}

std::unique_ptr<sirius::ffi::Fragment> make_gather_receiver(sirius::ffi::Context& ctx,
                                                            std::string const& plan)
{
  auto receiver = sirius::ffi::make_fragment(ctx);
  receiver->declare_input_column(0, "a", "BIGINT");
  receiver->build(plan);
  return receiver;
}

}  // namespace

TEST_CASE("FFI Arrow hop matches native relay_from for a parquet scan",
          "[isolated_context][sirius_ffi]")
{
  sirius::test::scratch_dir scratch("ffi_arrow_hop");
  auto const path = scratch.file("ids.parquet");
  write_ids_parquet(path);

  auto ctx = sirius::ffi::make_context_from_config(isolated_memory_config_path().string());
  auto const sender_plan   = local_files_plan(path);
  auto const receiver_plan = stream_read_plan(0);
  auto const expected      = std::vector<std::int64_t>{1, 2, 3, 4, 5};

  std::vector<std::int64_t> native;
  {
    auto sender   = run_parquet_sender(*ctx, sender_plan);
    auto receiver = make_gather_receiver(*ctx, receiver_plan);
    auto moved    = receiver->relay_from(*sender, 0, 0, 0);
    REQUIRE(moved > 0);
    receiver->run();
    native = result_i64s(*receiver);
  }
  REQUIRE(native == expected);

  std::vector<std::int64_t> via_arrow;
  {
    auto sender   = run_parquet_sender(*ctx, sender_plan);
    auto receiver = make_gather_receiver(*ctx, receiver_plan);

    std::size_t moved = 0;
    for (;;) {
      ArrowArrayStream hop{};
      if (!sender->pull_arrow(0, reinterpret_cast<std::uintptr_t>(&hop))) { break; }
      receiver->push_arrow(0, reinterpret_cast<std::uintptr_t>(&hop));
      release_if(hop);
      ++moved;
    }
    REQUIRE(moved > 0);
    REQUIRE(sender->drained(0));
    {
      ArrowArrayStream extra{};
      REQUIRE_FALSE(sender->pull_arrow(0, reinterpret_cast<std::uintptr_t>(&extra)));
    }
    receiver->close_input(0, 0);
    receiver->run();
    via_arrow = result_i64s(*receiver);
  }

  REQUIRE(via_arrow == native);
}

TEST_CASE("Fragment::pull_arrow requires run() like relay_from", "[isolated_context][sirius_ffi]")
{
  sirius::test::scratch_dir scratch("ffi_arrow_before_run");
  auto const path = scratch.file("ids.parquet");
  write_ids_parquet(path);

  auto ctx    = sirius::ffi::make_context_from_config(isolated_memory_config_path().string());
  auto sender = sirius::ffi::make_fragment(*ctx);
  sender->declare_output(0);
  sender->build(local_files_plan(path));

  ArrowArrayStream hop{};
  REQUIRE_THROWS(sender->pull_arrow(0, reinterpret_cast<std::uintptr_t>(&hop)));
  REQUIRE_FALSE(sender->drained(0));
}
