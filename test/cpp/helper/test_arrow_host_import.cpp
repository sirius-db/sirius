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

// sirius::import_arrow_host_table on Arrow batches DuckDB produces, plus hand-edited shapes for
// the refusals DuckDB does not emit by default.

#include "../operator/operator_test_utils.hpp"
#include "cudf/cudf_utils.hpp"
#include "helper/arrow_host_import.hpp"
#include "helper/type_conversions.hpp"
#include "sirius/exception.hpp"
#include "utils/arrow_batch_utils.hpp"

#include <cudf/utilities/default_stream.hpp>

#include <rmm/cuda_stream.hpp>
#include <rmm/error.hpp>
#include <rmm/mr/cuda_async_memory_resource.hpp>
#include <rmm/mr/per_device_resource.hpp>

#include <cuda_runtime_api.h>

#include <catch.hpp>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <limits>
#include <memory>
#include <memory_resource>
#include <string>
#include <utility>
#include <vector>

using Catch::Matchers::ContainsSubstring;
using sirius::test::arrow_batch_from_sql;
using sirius::test::operator_utils::copy_column_to_host;

namespace {

std::vector<sirius::logical_type> declared(const duckdb::vector<duckdb::LogicalType>& types)
{
  auto converted = sirius::from_duckdb_vec(types);
  return {converted.begin(), converted.end()};
}

std::unique_ptr<cudf::table> import(
  sirius::test::arrow_batch& batch,
  const std::vector<std::string>& names,
  const std::vector<sirius::logical_type>& types,
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref())
{
  const auto plan = sirius::inspect_arrow_host_batch(
    &batch.array, &batch.schema, "test batch", names, types, *std::pmr::new_delete_resource());
  return sirius::import_arrow_host_table(plan, cudf::get_default_stream(), mr);
}

// resource_ref may copy the facade, so observed counters share one owner.
class import_test_resource {
 public:
  struct state {
    std::atomic<long> attempts{0};
    std::atomic<std::size_t> live_bytes{0};
    std::atomic<bool> failed{false};
    long fail_at{-1};
  };
  explicit import_test_resource(
    long fail_at                            = -1,
    rmm::device_async_resource_ref upstream = cudf::get_current_device_resource_ref())
    : counts(std::make_shared<state>()), upstream_(upstream)
  {
    counts->fail_at = fail_at;
  }
  void* allocate(::cuda::stream_ref stream,
                 std::size_t bytes,
                 std::size_t alignment = alignof(std::max_align_t))
  {
    attempt();
    auto* pointer = upstream_.allocate(stream, bytes, alignment);
    counts->live_bytes += bytes;
    return pointer;
  }
  void deallocate(::cuda::stream_ref stream,
                  void* pointer,
                  std::size_t bytes,
                  std::size_t alignment = alignof(std::max_align_t)) noexcept
  {
    upstream_.deallocate(stream, pointer, bytes, alignment);
    counts->live_bytes -= bytes;
  }
  void* allocate_sync(std::size_t bytes, std::size_t alignment = alignof(std::max_align_t))
  {
    attempt();
    auto* pointer = upstream_.allocate_sync(bytes, alignment);
    counts->live_bytes += bytes;
    return pointer;
  }
  void deallocate_sync(void* pointer,
                       std::size_t bytes,
                       std::size_t alignment = alignof(std::max_align_t)) noexcept
  {
    upstream_.deallocate_sync(pointer, bytes, alignment);
    counts->live_bytes -= bytes;
  }
  bool operator==(const import_test_resource& other) const noexcept
  {
    return counts == other.counts;
  }
  friend void get_property(const import_test_resource&, ::cuda::mr::device_accessible) noexcept {}
  std::shared_ptr<state> counts;

 private:
  void attempt()
  {
    const auto attempt = counts->attempts++;
    if (counts->fail_at == -2 || attempt == counts->fail_at) {
      counts->failed = true;
      throw rmm::out_of_memory("injected Arrow import allocation failure");
    }
  }
  rmm::device_async_resource_ref upstream_;
};

class bounded_host_resource : public std::pmr::memory_resource {
 public:
  explicit bounded_host_resource(std::size_t limit) : limit_(limit) {}
  std::size_t used{0};
  std::size_t peak{0};

 private:
  void* do_allocate(std::size_t bytes, std::size_t alignment) override
  {
    if (bytes > limit_ - used) { throw std::bad_alloc(); }
    auto* pointer = std::pmr::new_delete_resource()->allocate(bytes, alignment);
    used += bytes;
    peak = std::max(peak, used);
    return pointer;
  }
  void do_deallocate(void* pointer, std::size_t bytes, std::size_t alignment) override
  {
    std::pmr::new_delete_resource()->deallocate(pointer, bytes, alignment);
    used -= bytes;
  }
  bool do_is_equal(const std::pmr::memory_resource& other) const noexcept override
  {
    return this == &other;
  }
  std::size_t limit_;
};

}  // namespace

TEST_CASE("arrow_host_import: DuckDB's scalar types import at the declared cudf types",
          "[arrow_host_import]")
{
  auto batch = arrow_batch_from_sql(
    "SELECT CASE WHEN i = 3 THEN NULL ELSE i END::BIGINT AS a, i::INTEGER AS b, "
    "(i * 1.25)::DECIMAL(15, 2) AS d, DATE '2024-01-01' + i::INTEGER AS e, 'x' || i AS f, "
    "i % 2 = 0 AS g FROM range(5) t(i)");
  const auto types = declared({duckdb::LogicalType::BIGINT,
                               duckdb::LogicalType::INTEGER,
                               duckdb::LogicalType::DECIMAL(15, 2),
                               duckdb::LogicalType::DATE,
                               duckdb::LogicalType::VARCHAR,
                               duckdb::LogicalType::BOOLEAN});
  auto table       = import(*batch, {"a", "b", "d", "e", "f", "g"}, types);
  cudf::get_default_stream().synchronize();

  REQUIRE(table->num_rows() == 5);
  for (cudf::size_type i = 0; i < table->num_columns(); ++i) {
    REQUIRE(table->get_column(i).type() == sirius::get_cudf_type(types[i]));
  }
  REQUIRE(table->get_column(0).null_count() == 1);
  // Decimal128 narrowed to DECIMAL64 by the declared precision; unscaled values are i * 125.
  REQUIRE(copy_column_to_host<std::int64_t>(table->get_column(2).view()) ==
          std::vector<std::int64_t>{0, 125, 250, 375, 500});
  REQUIRE(copy_column_to_host<std::string>(table->get_column(4).view()) ==
          std::vector<std::string>{"x0", "x1", "x2", "x3", "x4"});
  REQUIRE(copy_column_to_host<bool>(table->get_column(5).view()) ==
          std::vector<bool>{true, false, true, false, true});
}

TEST_CASE("arrow_host_import: a window on the struct selects its rows from every child",
          "[arrow_host_import]")
{
  auto batch = arrow_batch_from_sql("SELECT i::BIGINT AS a FROM range(10) t(i)");
  // Arrow C++ StructArray::Slice shape: the struct carries the window, the children do not.
  batch->array.offset = 2;
  batch->array.length = 5;
  auto table          = import(*batch, {"a"}, declared({duckdb::LogicalType::BIGINT}));
  cudf::get_default_stream().synchronize();
  REQUIRE(copy_column_to_host<std::int64_t>(table->get_column(0).view()) ==
          std::vector<std::int64_t>{2, 3, 4, 5, 6});
}

TEST_CASE("arrow_host_import: a window counts only the nulls inside it", "[arrow_host_import]")
{
  // Nulls at rows 0 and 3; the window [2, 7) holds only row 3's.
  auto batch = arrow_batch_from_sql(
    "SELECT CASE WHEN i IN (0, 3) THEN NULL ELSE i END::BIGINT AS a FROM range(10) t(i)");
  batch->array.offset = 2;
  batch->array.length = 5;
  auto table          = import(*batch, {"a"}, declared({duckdb::LogicalType::BIGINT}));
  REQUIRE(table->get_column(0).size() == 5);
  REQUIRE(table->get_column(0).null_count() == 1);
}

TEST_CASE("arrow_host_import: refuses mismatched and unsupported batches before any copy",
          "[arrow_host_import]")
{
  const auto bigint = declared({duckdb::LogicalType::BIGINT});
  auto batch        = arrow_batch_from_sql("SELECT i::BIGINT AS a FROM range(3) t(i)");
  auto& column      = *batch->schema.children[0];

  SECTION("type mismatch")
  {
    REQUIRE_THROWS_WITH(import(*batch, {"a"}, declared({duckdb::LogicalType::INTEGER})),
                        ContainsSubstring("column 0 (a) is declared INTEGER"));
  }
  SECTION("column count")
  {
    REQUIRE_THROWS_WITH(
      import(
        *batch, {"a", "b"}, declared({duckdb::LogicalType::BIGINT, duckdb::LogicalType::BIGINT})),
      ContainsSubstring("the stream declares 2"));
  }
  SECTION("declared 128-bit integer or nested")
  {
    for (const auto& type :
         std::vector<duckdb::LogicalType>{duckdb::LogicalType::HUGEINT,
                                          duckdb::LogicalType::UHUGEINT,
                                          duckdb::LogicalType::LIST(duckdb::LogicalType::BIGINT)}) {
      REQUIRE_THROWS_WITH(import(*batch, {"a"}, declared({type})),
                          ContainsSubstring("which cannot be imported"));
    }
  }
  SECTION("released structs")
  {
    auto release         = batch->array.release;
    batch->array.release = nullptr;
    REQUIRE_THROWS_WITH(import(*batch, {"a"}, bigint), ContainsSubstring("already released"));
    batch->array.release = release;
  }
  SECTION("a released or missing column")
  {
    auto& child   = *batch->array.children[0];
    auto release  = child.release;
    child.release = nullptr;
    REQUIRE_THROWS_WITH(import(*batch, {"a"}, bigint), ContainsSubstring("missing or released"));
    child.release         = release;
    auto* children        = batch->array.children;
    batch->array.children = nullptr;
    REQUIRE_THROWS_WITH(import(*batch, {"a"}, bigint), ContainsSubstring("missing or released"));
    batch->array.children = children;
  }
  SECTION("a format cudf refuses")
  {
    column.format = "tts";  // time32[s]: cudf has no time type
    REQUIRE_THROWS_AS(import(*batch, {"a"}, bigint), sirius::invalid_input_exception);
    REQUIRE_THROWS_WITH(import(*batch, {"a"}, bigint),
                        ContainsSubstring("unsupported Arrow format"));
  }
  SECTION("more rows than cudf holds")
  {
    batch->array.length = std::int64_t{std::numeric_limits<cudf::size_type>::max()} + 1;
    REQUIRE_THROWS_WITH(import(*batch, {"a"}, bigint), ContainsSubstring("longer than cudf's"));
    batch->array.length = 3;
  }
  SECTION("a window past a child")
  {
    batch->array.offset = 1;
    REQUIRE_THROWS_WITH(import(*batch, {"a"}, bigint), ContainsSubstring("spans rows [1, 4)"));
    batch->array.offset = 0;
  }
  SECTION("struct-level nulls")
  {
    std::uint8_t validity   = 0b101;  // row 1 is null
    const void* buffers[]   = {&validity};
    auto** saved            = batch->array.buffers;
    auto saved_nulls        = batch->array.null_count;
    batch->array.buffers    = buffers;
    batch->array.null_count = 1;
    REQUIRE_THROWS_WITH(import(*batch, {"a"}, bigint), ContainsSubstring("has null rows"));
    batch->array.buffers    = saved;
    batch->array.null_count = saved_nulls;
  }
  SECTION("shapes refused by name")
  {
    const char* saved_format = column.format;
    for (auto [format, reason] : {std::pair{"U", "64-bit offsets"},
                                  std::pair{"Z", "64-bit offsets"},
                                  std::pair{"+L", "64-bit offsets"},
                                  std::pair{"tsu:UTC", "timezone-aware"},
                                  std::pair{"d:40,2,256", "decimal256"}}) {
      column.format = format;
      CAPTURE(format);
      REQUIRE_THROWS_WITH(import(*batch, {"a"}, bigint), ContainsSubstring(reason));
    }
    column.format = saved_format;

    ArrowSchema dictionary{};
    column.dictionary = &dictionary;
    REQUIRE_THROWS_WITH(import(*batch, {"a"}, bigint), ContainsSubstring("dictionary-encoded"));
    column.dictionary = nullptr;
  }
}

TEST_CASE("arrow_host_import: a bad final column does no device allocation", "[arrow_host_import]")
{
  auto batch       = arrow_batch_from_sql("SELECT 42::BIGINT AS a, 7::INTEGER AS b");
  const auto types = declared({duckdb::LogicalType::BIGINT, duckdb::LogicalType::INTEGER});
  auto& child      = *batch->array.children[1];
  auto& schema     = *batch->schema.children[1];
  SECTION("unknown format") { schema.format = "tts"; }
  SECTION("negative child offset") { child.offset = -1; }
  SECTION("combined window overflow") { child.offset = std::numeric_limits<std::int64_t>::max(); }
  SECTION("missing buffers") { child.buffers = nullptr; }
  SECTION("wrong buffer count") { child.n_buffers = 1; }
  SECTION("null count without bitmap") { child.null_count = 1; }
  import_test_resource observed;
  REQUIRE_THROWS_AS(import(*batch, {"a", "b"}, types, rmm::device_async_resource_ref{observed}),
                    sirius::invalid_input_exception);
  REQUIRE(observed.counts->attempts == 0);
  REQUIRE(observed.counts->live_bytes == 0);
}

TEST_CASE("arrow_host_import: checked plans keep combined slices and actual nulls",
          "[arrow_host_import]")
{
  auto batch = arrow_batch_from_sql(
    "SELECT CASE WHEN i IN (0, 3, 8) THEN NULL ELSE i END::BIGINT AS a FROM range(12) t(i)");
  auto& child         = *batch->array.children[0];
  child.offset        = 1;
  child.length        = 11;
  child.null_count    = 2;
  batch->array.offset = 1;
  batch->array.length = 7;
  const auto plan     = sirius::inspect_arrow_host_batch(&batch->array,
                                                     &batch->schema,
                                                     "slice",
                                                         {"a"},
                                                     declared({duckdb::LogicalType::BIGINT}),
                                                     *std::pmr::new_delete_resource());
  REQUIRE(plan.num_rows() == 7);
  REQUIRE(plan.estimated_device_bytes() >= 7 * sizeof(std::int64_t));
  rmm::cuda_stream stream;
  auto table =
    sirius::import_arrow_host_table(plan, stream.view(), cudf::get_current_device_resource_ref());
  REQUIRE(cudaStreamQuery(stream.value()) == cudaSuccess);
  REQUIRE(table->num_rows() == 7);
  REQUIRE(table->get_column(0).null_count() == 2);
  const auto values = copy_column_to_host<std::int64_t>(table->get_column(0).view());
  REQUIRE(values[0] == 2);
  REQUIRE(values[2] == 4);
  REQUIRE(values[5] == 7);
}

TEST_CASE("arrow_host_import: string windows are checked before copy", "[arrow_host_import]")
{
  auto batch             = arrow_batch_from_sql("SELECT 'value' || i AS a FROM range(4) t(i)");
  const auto strings     = declared({duckdb::LogicalType::VARCHAR});
  auto& child            = *batch->array.children[0];
  const void* buffers[3] = {child.buffers[0], child.buffers[1], child.buffers[2]};
  auto* saved            = child.buffers;
  SECTION("negative offsets")
  {
    const std::int32_t offsets[] = {-1, 0, 1, 2, 3};
    buffers[1]                   = offsets;
    child.buffers                = buffers;
    REQUIRE_THROWS_WITH(import(*batch, {"a"}, strings),
                        ContainsSubstring("negative string offsets"));
    child.buffers = saved;
  }
  SECTION("decreasing offsets")
  {
    const std::int32_t offsets[] = {0, 2, 1, 3, 4};
    buffers[1]                   = offsets;
    child.buffers                = buffers;
    REQUIRE_THROWS_WITH(import(*batch, {"a"}, strings),
                        ContainsSubstring("decreasing string offsets"));
    child.buffers = saved;
  }
  SECTION("missing data")
  {
    buffers[2]    = nullptr;
    child.buffers = buffers;
    REQUIRE_THROWS_WITH(import(*batch, {"a"}, strings), ContainsSubstring("no string data buffer"));
    child.buffers = saved;
  }
  SECTION("valid string slice")
  {
    batch->array.offset = 1;
    batch->array.length = 2;
    auto table          = import(*batch, {"a"}, strings);
    REQUIRE(copy_column_to_host<std::string>(table->get_column(0).view()) ==
            std::vector<std::string>{"value1", "value2"});
  }
}

TEST_CASE("arrow_host_import: every observed allocation refusal drains and keeps roots reusable",
          "[arrow_host_import]")
{
  auto batch = arrow_batch_from_sql(
    "SELECT i::BIGINT AS a, 'value' || i AS b, (i * 1.25)::DECIMAL(15,2) AS d FROM range(32) t(i)");
  const auto plan =
    sirius::inspect_arrow_host_batch(&batch->array,
                                     &batch->schema,
                                     "allocation",
                                     {"a", "b", "d"},
                                     declared({duckdb::LogicalType::BIGINT,
                                               duckdb::LogicalType::VARCHAR,
                                               duckdb::LogicalType::DECIMAL(15, 2)}),
                                     *std::pmr::new_delete_resource());
  rmm::cuda_stream stream;
  import_test_resource observed;
  {
    auto table = sirius::import_arrow_host_table(
      plan, stream.view(), rmm::device_async_resource_ref{observed});
    REQUIRE(table->num_rows() == 32);
  }
  stream.synchronize();
  REQUIRE(observed.counts->live_bytes == 0);
  const auto attempts = observed.counts->attempts.load();
  REQUIRE(attempts > 2);
  for (long failure = 0; failure < attempts; ++failure) {
    CAPTURE(failure);
    import_test_resource failing(failure);
    REQUIRE_THROWS_AS(
      sirius::import_arrow_host_table(plan, stream.view(), rmm::device_async_resource_ref{failing}),
      rmm::out_of_memory);
    REQUIRE(failing.counts->failed);
    REQUIRE(cudaStreamQuery(stream.value()) == cudaSuccess);
    REQUIRE(failing.counts->live_bytes == 0);
    REQUIRE(batch->array.release != nullptr);
    REQUIRE(batch->schema.release != nullptr);
    auto retry =
      sirius::import_arrow_host_table(plan, stream.view(), cudf::get_current_device_resource_ref());
    REQUIRE(retry->num_rows() == 32);
  }
}

TEST_CASE("arrow_host_import: bit-offset boundary slices normalize before GPU copy",
          "[arrow_host_import]")
{
  const int rows = GENERATE(1, 7, 8, 31, 32, 33, 511, 512, 513);
  auto batch     = arrow_batch_from_sql(
    "SELECT i % 2 = 0 AS b, CASE WHEN i % 7 = 0 THEN NULL ELSE i END::INTEGER AS v "
        "FROM range(" +
    std::to_string(rows + 1) + ") t(i)");
  batch->array.offset = 1;
  batch->array.length = rows;
  auto table          = import(
    *batch, {"b", "v"}, declared({duckdb::LogicalType::BOOLEAN, duckdb::LogicalType::INTEGER}));
  const auto booleans  = copy_column_to_host<bool>(table->get_column(0).view());
  const auto integers  = copy_column_to_host<std::int32_t>(table->get_column(1).view());
  const auto& nullable = table->get_column(1);
  REQUIRE(nullable.null_count() == rows / 7);
  std::vector<cudf::bitmask_type> mask((rows + 31) / 32);
  if (rows >= 7) {
    REQUIRE(cudaMemcpy(mask.data(),
                       nullable.view().null_mask(),
                       mask.size() * sizeof(cudf::bitmask_type),
                       cudaMemcpyDeviceToHost) == cudaSuccess);
  }
  for (int row = 0; row < rows; ++row) {
    REQUIRE(booleans[row] == ((row + 1) % 2 == 0));
    const bool valid = (row + 1) % 7 != 0;
    if (rows >= 7) {
      REQUIRE(((mask[row / 32] >> (row % 32)) & 1) == static_cast<unsigned>(valid));
    }
    if (valid) { REQUIRE(integers[row] == row + 1); }
  }
}

TEST_CASE("arrow_host_import: host staging limits unwind before device work and survive plan moves",
          "[arrow_host_import]")
{
  auto batch = arrow_batch_from_sql(
    "SELECT i % 2 = 0 AS b, 'value' || i AS s, (i * 1.25)::DECIMAL(15,2) AS d FROM range(32) t(i)");
  const auto types = declared({duckdb::LogicalType::BOOLEAN,
                               duckdb::LogicalType::VARCHAR,
                               duckdb::LogicalType::DECIMAL(15, 2)});
  bounded_host_resource host(std::numeric_limits<std::size_t>::max());
  import_test_resource observed;
  {
    auto plan = sirius::inspect_arrow_host_batch(
      &batch->array, &batch->schema, "host", {"b", "s", "d"}, types, host);
    REQUIRE(plan.host_staging_bytes() > 0);
    REQUIRE(host.used >= plan.host_staging_bytes());
    auto moved = std::move(plan);
    REQUIRE_THROWS_WITH(
      sirius::import_arrow_host_table(
        plan, cudf::get_default_stream(), rmm::device_async_resource_ref{observed}),
      ContainsSubstring("moved-from"));
    REQUIRE(observed.counts->attempts == 0);
    auto table = sirius::import_arrow_host_table(
      moved, cudf::get_default_stream(), rmm::device_async_resource_ref{observed});
    REQUIRE(table->num_rows() == 32);
  }
  REQUIRE(host.used == 0);
  REQUIRE(observed.counts->live_bytes == 0);
  const auto attempts_before = observed.counts->attempts.load();
  for (const auto capacity : {std::size_t{0}, host.peak - 1}) {
    bounded_host_resource limited(capacity);
    REQUIRE_THROWS_AS(sirius::inspect_arrow_host_batch(
                        &batch->array, &batch->schema, "host", {"b", "s", "d"}, types, limited),
                      std::bad_alloc);
    REQUIRE(limited.used == 0);
    REQUIRE(observed.counts->attempts == attempts_before);
  }
}

namespace {

// This is only a negative test. Production imports never replace a process-global resource.
class current_resource_guard {
 public:
  explicit current_resource_guard(const import_test_resource& replacement)
    : previous_(cudf::set_current_device_resource(
        ::cuda::mr::any_resource<::cuda::mr::device_accessible>{replacement}))
  {
  }
  ~current_resource_guard() { (void)cudf::set_current_device_resource(std::move(previous_)); }

 private:
  ::cuda::mr::any_resource<::cuda::mr::device_accessible> previous_;
};

}  // namespace

TEST_CASE("arrow_host_import: a rejecting current resource cannot intercept supplied allocations",
          "[arrow_host_import]")
{
  auto batch = arrow_batch_from_sql(
    "SELECT CASE WHEN i % 5 = 0 THEN NULL ELSE i END::BIGINT AS a, "
    "(i * 1.25)::DECIMAL(15,2) AS d, 'value' || i AS s, i % 2 = 0 AS b FROM range(513) t(i)");
  const auto plan = sirius::inspect_arrow_host_batch(&batch->array,
                                                     &batch->schema,
                                                     "explicit resource",
                                                     {"a", "d", "s", "b"},
                                                     declared({duckdb::LogicalType::BIGINT,
                                                               duckdb::LogicalType::DECIMAL(15, 2),
                                                               duckdb::LogicalType::VARCHAR,
                                                               duckdb::LogicalType::BOOLEAN}),
                                                     *std::pmr::new_delete_resource());
  rmm::mr::cuda_async_memory_resource upstream;
  import_test_resource supplied(-1, rmm::device_async_resource_ref{upstream});
  import_test_resource rejecting(-2, rmm::device_async_resource_ref{upstream});
  rmm::cuda_stream stream;
  {
    current_resource_guard guard(rejecting);
    auto table = sirius::import_arrow_host_table(
      plan, stream.view(), rmm::device_async_resource_ref{supplied});
    REQUIRE(table->num_rows() == 513);
    REQUIRE(rejecting.counts->attempts == 0);
  }
  stream.synchronize();
  REQUIRE(supplied.counts->attempts > 0);
  REQUIRE(supplied.counts->live_bytes == 0);
}

TEST_CASE("arrow_host_import: a decimal must carry the declared scale and fit its precision",
          "[arrow_host_import]")
{
  auto batch = arrow_batch_from_sql("SELECT 1.25::DECIMAL(18, 2) AS d");
  REQUIRE_THROWS_WITH(import(*batch, {"d"}, declared({duckdb::LogicalType::DECIMAL(15, 2)})),
                      ContainsSubstring("carries precision 18"));
  REQUIRE_THROWS_WITH(import(*batch, {"d"}, declared({duckdb::LogicalType::DECIMAL(18, 3)})),
                      ContainsSubstring("is declared DECIMAL"));
  auto table = import(*batch, {"d"}, declared({duckdb::LogicalType::DECIMAL(18, 2)}));
  cudf::get_default_stream().synchronize();
  REQUIRE(copy_column_to_host<std::int64_t>(table->get_column(0).view()) ==
          std::vector<std::int64_t>{125});
}
