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

#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>

#include <rmm/cuda_stream.hpp>
#include <rmm/error.hpp>
#include <rmm/mr/cuda_async_memory_resource.hpp>
#include <rmm/mr/per_device_resource.hpp>

#include <cuda_runtime_api.h>

#include <catch.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <future>
#include <limits>
#include <memory>
#include <memory_resource>
#include <string>
#include <thread>
#include <utility>
#include <vector>

using Catch::Matchers::ContainsSubstring;
using sirius::test::arrow_batch_from_sql;

namespace {

template <typename T>
class scoped_restore {
 public:
  explicit scoped_restore(T& target) : target_(target), saved_(target) {}
  ~scoped_restore() { target_ = saved_; }
  scoped_restore(const scoped_restore&)            = delete;
  scoped_restore& operator=(const scoped_restore&) = delete;

 private:
  T& target_;
  T saved_;
};

template <typename T>
std::vector<T> copy_column_to_host(const cudf::column_view& column)
{
  if constexpr (!std::is_same_v<T, std::string>) {
    return sirius::test::operator_utils::copy_column_to_host<T>(column, cudf::get_default_stream());
  } else {
    REQUIRE(column.offset() == 0);
    if (column.size() == 0) { return {}; }
    cudf::strings_column_view strings(column);
    const auto offsets = sirius::test::operator_utils::copy_column_to_host<std::int32_t>(
      strings.offsets(), cudf::get_default_stream());
    REQUIRE(offsets.size() == static_cast<std::size_t>(column.size()) + 1);
    REQUIRE(offsets.front() == 0);
    REQUIRE(offsets.back() >= 0);
    std::vector<char> chars(static_cast<std::size_t>(offsets.back()));
    if (!chars.empty()) {
      REQUIRE(cudaMemcpy(chars.data(),
                         strings.chars_begin(cudf::get_default_stream()),
                         chars.size(),
                         cudaMemcpyDeviceToHost) == cudaSuccess);
    }
    std::vector<std::string> result;
    result.reserve(column.size());
    for (cudf::size_type row = 0; row < column.size(); ++row) {
      REQUIRE(offsets[row] >= 0);
      REQUIRE(offsets[row] <= offsets[row + 1]);
      REQUIRE(static_cast<std::size_t>(offsets[row + 1]) <= chars.size());
      if (offsets[row] == offsets[row + 1]) {
        result.emplace_back();
      } else {
        result.emplace_back(chars.data() + offsets[row], offsets[row + 1] - offsets[row]);
      }
    }
    return result;
  }
}

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
    std::atomic<int> callback_status{cudaSuccess};
    cudaStream_t expected_stream{nullptr};
    std::atomic<bool> wrong_stream{false};
    bool delay_deallocation{false};
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
    if (counts->expected_stream && stream.get() != counts->expected_stream) {
      counts->wrong_stream = true;
    }
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
    if (counts->expected_stream && stream.get() != counts->expected_stream) {
      counts->wrong_stream = true;
    }
    if (counts->delay_deallocation) {
      counts->callback_status = cudaLaunchHostFunc(
        stream.get(),
        [](void*) { std::this_thread::sleep_for(std::chrono::milliseconds(150)); },
        nullptr);
    }
    upstream_.deallocate(stream, pointer, bytes, alignment);
    counts->live_bytes -= bytes;
  }
  void* allocate_sync(std::size_t bytes, std::size_t alignment = alignof(std::max_align_t))
  {
    if (counts->expected_stream) { counts->wrong_stream = true; }
    attempt();
    auto* pointer = upstream_.allocate_sync(bytes, alignment);
    counts->live_bytes += bytes;
    return pointer;
  }
  void deallocate_sync(void* pointer,
                       std::size_t bytes,
                       std::size_t alignment = alignof(std::max_align_t)) noexcept
  {
    if (counts->expected_stream) { counts->wrong_stream = true; }
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

#if defined(__linux__)
struct import_copy_observation {
  cudaStream_t stream;
  int calls{0};
  std::size_t bytes{0};
  bool wrong_stream{false};
};
thread_local import_copy_observation* active_import_copies = nullptr;

struct import_drain_fault {
  cudaStream_t stream;
  const import_test_resource::state& resource;
  const import_copy_observation& copies;
  cudaError_t results[2];
  int calls{0};
  std::size_t live_bytes[2]{};
  int copy_calls[2]{};
  std::size_t copy_bytes[2]{};
  cudaError_t actual_status[2]{};
};
thread_local import_drain_fault* active_import_drain_fault = nullptr;
#endif

}  // namespace

#if defined(__linux__)
extern "C" cudaError_t __real_cudaMemcpyAsync(void* destination,
                                              const void* source,
                                              std::size_t bytes,
                                              cudaMemcpyKind kind,
                                              cudaStream_t stream);
extern "C" cudaError_t __wrap_cudaMemcpyAsync(void* destination,
                                              const void* source,
                                              std::size_t bytes,
                                              cudaMemcpyKind kind,
                                              cudaStream_t stream)
{
  auto* copies = active_import_copies;
  if (copies != nullptr && kind == cudaMemcpyHostToDevice) {
    ++copies->calls;
    copies->bytes += bytes;
    copies->wrong_stream |= stream != copies->stream;
  }
  return __real_cudaMemcpyAsync(destination, source, bytes, kind, stream);
}

extern "C" cudaError_t __real_cudaStreamSynchronize(cudaStream_t stream);
extern "C" cudaError_t __wrap_cudaStreamSynchronize(cudaStream_t stream)
{
  auto* fault = active_import_drain_fault;
  if (fault == nullptr || fault->stream != stream || fault->calls >= 2) {
    return __real_cudaStreamSynchronize(stream);
  }
  const auto call            = fault->calls++;
  fault->live_bytes[call]    = fault->resource.live_bytes.load();
  fault->copy_calls[call]    = fault->copies.calls;
  fault->copy_bytes[call]    = fault->copies.bytes;
  const auto actual          = __real_cudaStreamSynchronize(stream);
  fault->actual_status[call] = actual;
  // Drain real work before reporting a synthetic failure, keeping this process usable.
  return actual == cudaSuccess ? fault->results[call] : actual;
}
#endif

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
  scoped_restore root_guard(batch->array);
  scoped_restore child_guard(*batch->array.children[0]);
  scoped_restore schema_guard(column);

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
    batch->array.release = nullptr;
    REQUIRE_THROWS_WITH(import(*batch, {"a"}, bigint), ContainsSubstring("already released"));
  }
  SECTION("a released column")
  {
    batch->array.children[0]->release = nullptr;
    REQUIRE_THROWS_WITH(import(*batch, {"a"}, bigint), ContainsSubstring("missing or released"));
  }
  SECTION("a missing column")
  {
    batch->array.children = nullptr;
    REQUIRE_THROWS_WITH(import(*batch, {"a"}, bigint), ContainsSubstring("missing or released"));
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
  }
  SECTION("a window past a child")
  {
    batch->array.offset = 1;
    REQUIRE_THROWS_WITH(import(*batch, {"a"}, bigint), ContainsSubstring("spans rows [1, 4)"));
  }
  SECTION("struct-level nulls")
  {
    std::uint8_t validity   = 0b101;  // row 1 is null
    const void* buffers[]   = {&validity};
    batch->array.buffers    = buffers;
    batch->array.null_count = GENERATE(-1, 0, 1);
    REQUIRE_THROWS_WITH(import(*batch, {"a"}, bigint), ContainsSubstring("has null rows"));
  }
  SECTION("struct null count contradicts an all-valid bitmap")
  {
    std::uint8_t validity   = 0b111;
    const void* buffers[]   = {&validity};
    batch->array.buffers    = buffers;
    batch->array.null_count = 1;
    import_test_resource observed;
    REQUIRE_THROWS_WITH(import(*batch, {"a"}, bigint, rmm::device_async_resource_ref{observed}),
                        ContainsSubstring("has null rows"));
    REQUIRE(observed.counts->attempts == 0);
  }
  SECTION("shapes refused by name")
  {
    for (auto [format, reason] : {std::pair{"U", "64-bit offsets"},
                                  std::pair{"Z", "64-bit offsets"},
                                  std::pair{"+L", "64-bit offsets"},
                                  std::pair{"tsu:UTC", "timezone-aware"},
                                  std::pair{"d:40,2,256", "decimal256"}}) {
      scoped_restore iteration_guard(column);
      column.format = format;
      CAPTURE(format);
      REQUIRE_THROWS_WITH(import(*batch, {"a"}, bigint), ContainsSubstring(reason));
    }

    ArrowSchema dictionary{};
    column.dictionary = &dictionary;
    REQUIRE_THROWS_WITH(import(*batch, {"a"}, bigint), ContainsSubstring("dictionary-encoded"));
  }
}

TEST_CASE("arrow_host_import: a bad final column does no device allocation", "[arrow_host_import]")
{
  auto batch       = arrow_batch_from_sql("SELECT 42::BIGINT AS a, 7::INTEGER AS b");
  const auto types = declared({duckdb::LogicalType::BIGINT, duckdb::LogicalType::INTEGER});
  auto& child      = *batch->array.children[1];
  auto& schema     = *batch->schema.children[1];
  scoped_restore child_guard(child);
  scoped_restore schema_guard(schema);
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

TEST_CASE("arrow_host_import: impossible sliced null totals unwind before device work",
          "[arrow_host_import]")
{
  const auto child_offset   = GENERATE(0, 3);
  const bool below_selected = GENERATE(true, false);
  CAPTURE(child_offset, below_selected);
  auto batch   = arrow_batch_from_sql("SELECT i % 2 = 0 AS b, i::INTEGER AS v FROM range(8) t(i)");
  auto& child  = *batch->array.children[1];
  auto& schema = *batch->schema.children[1];
  scoped_restore child_guard(child);
  scoped_restore schema_guard(schema);
  const std::uint8_t validity = (below_selected ? 0b1001 : 0b1111) << child_offset;
  const void* buffers[]       = {&validity, child.buffers[1]};
  child.buffers               = buffers;
  child.offset                = child_offset;
  child.length                = 4;
  child.null_count            = below_selected ? 1 : 4;
  schema.flags |= ARROW_FLAG_NULLABLE;
  batch->array.offset = 1;
  batch->array.length = 2;
  bounded_host_resource host(std::numeric_limits<std::size_t>::max());
  import_test_resource observed(-2);
  const auto attempt = [&] {
    auto plan = sirius::inspect_arrow_host_batch(
      &batch->array,
      &batch->schema,
      "slice",
      {"b", "v"},
      declared({duckdb::LogicalType::BOOLEAN, duckdb::LogicalType::INTEGER}),
      host);
    return sirius::import_arrow_host_table(
      plan, cudf::get_default_stream(), rmm::device_async_resource_ref{observed});
  };
  REQUIRE_THROWS_WITH(attempt(), ContainsSubstring("inconsistent with its null metadata"));
  REQUIRE(host.peak > 0);
  REQUIRE(host.used == 0);
  REQUIRE(observed.counts->attempts == 0);
  REQUIRE(observed.counts->live_bytes == 0);
  REQUIRE(batch->array.release != nullptr);
  REQUIRE(batch->schema.release != nullptr);
}

TEST_CASE("arrow_host_import: sliced null totals allow outside distributions and unknown counts",
          "[arrow_host_import]")
{
  const auto child_offset          = GENERATE(0, 3);
  const unsigned selected_validity = GENERATE(0u, 1u, 3u);
  const auto outside_nulls         = GENERATE(0, 1, 2);
  const bool unknown               = GENERATE(false, true);
  CAPTURE(child_offset, selected_validity, outside_nulls, unknown);
  auto batch   = arrow_batch_from_sql("SELECT i::INTEGER AS v FROM range(8) t(i)");
  auto& child  = *batch->array.children[0];
  auto& schema = *batch->schema.children[0];
  scoped_restore child_guard(child);
  scoped_restore schema_guard(schema);
  const auto selected_nulls = 2 - (selected_validity & 1) - ((selected_validity >> 1) & 1);
  const std::uint8_t validity =
    ((selected_validity << 1) | (outside_nulls == 0 ? 1u : 0u) | (outside_nulls < 2 ? 8u : 0u))
    << child_offset;
  const void* buffers[] = {&validity, child.buffers[1]};
  child.buffers         = buffers;
  child.offset          = child_offset;
  child.length          = 4;
  child.null_count      = unknown ? -1 : static_cast<std::int64_t>(selected_nulls + outside_nulls);
  schema.flags |= ARROW_FLAG_NULLABLE;
  batch->array.offset = 1;
  batch->array.length = 2;
  auto table          = import(*batch, {"v"}, declared({duckdb::LogicalType::INTEGER}));
  const auto& column  = table->get_column(0);
  REQUIRE(table->num_rows() == 2);
  REQUIRE(column.type().id() == cudf::type_id::INT32);
  REQUIRE(column.null_count() == selected_nulls);
  cudf::bitmask_type mask = 0;
  if (selected_nulls != 0) {
    REQUIRE(cudaMemcpy(&mask, column.view().null_mask(), sizeof(mask), cudaMemcpyDeviceToHost) ==
            cudaSuccess);
    REQUIRE((mask & 3u) == selected_validity);
  }
  const auto values = copy_column_to_host<std::int32_t>(column.view());
  for (int row = 0; row < 2; ++row) {
    if ((selected_validity >> row) & 1) { REQUIRE(values[row] == child_offset + 1 + row); }
  }
}

TEST_CASE("arrow_host_import: string windows are checked before copy", "[arrow_host_import]")
{
  auto batch         = arrow_batch_from_sql("SELECT 'value' || i AS a FROM range(4) t(i)");
  const auto strings = declared({duckdb::LogicalType::VARCHAR});
  auto& child        = *batch->array.children[0];
  scoped_restore child_guard(child);
  const void* buffers[3] = {child.buffers[0], child.buffers[1], child.buffers[2]};
  SECTION("negative offsets")
  {
    const std::int32_t offsets[] = {-1, 0, 1, 2, 3};
    buffers[1]                   = offsets;
    child.buffers                = buffers;
    REQUIRE_THROWS_WITH(import(*batch, {"a"}, strings),
                        ContainsSubstring("negative string offsets"));
  }
  SECTION("decreasing offsets")
  {
    const std::int32_t offsets[] = {0, 2, 1, 3, 4};
    buffers[1]                   = offsets;
    child.buffers                = buffers;
    REQUIRE_THROWS_WITH(import(*batch, {"a"}, strings),
                        ContainsSubstring("decreasing string offsets"));
  }
  SECTION("missing data")
  {
    buffers[2]    = nullptr;
    child.buffers = buffers;
    REQUIRE_THROWS_WITH(import(*batch, {"a"}, strings), ContainsSubstring("no string data buffer"));
  }
  SECTION("valid string slice")
  {
    batch->array.offset = 1;
    batch->array.length = 2;
    auto table          = import(*batch, {"a"}, strings);
    REQUIRE(copy_column_to_host<std::string>(table->get_column(0).view()) ==
            std::vector<std::string>{"value1", "value2"});
  }
  SECTION("unaligned offsets")
  {
    std::vector<std::uint8_t> storage(5 * sizeof(std::int32_t) + 1);
    std::memcpy(storage.data() + 1, child.buffers[1], 5 * sizeof(std::int32_t));
    buffers[1]    = storage.data() + 1;
    child.buffers = buffers;
    auto table    = import(*batch, {"a"}, strings);
    REQUIRE(copy_column_to_host<std::string>(table->get_column(0).view()) ==
            std::vector<std::string>{"value0", "value1", "value2", "value3"});
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
  const int rows         = GENERATE(1, 7, 8, 31, 32, 33, 511, 512, 513);
  const int child_offset = GENERATE(0, 1, 7);
  CAPTURE(rows, child_offset);
  auto batch = arrow_batch_from_sql(
    "SELECT CASE WHEN i % 5 = 0 THEN NULL ELSE i % 3 = 0 END AS b, "
    "CASE WHEN i % 7 = 0 THEN NULL ELSE i END::INTEGER AS v "
    "FROM range(" +
    std::to_string(rows + child_offset + 1) + ") t(i)");
  auto& boolean_child  = *batch->array.children[0];
  boolean_child.offset = child_offset;
  boolean_child.length -= child_offset;
  boolean_child.null_count = -1;
  batch->array.offset      = 1;
  batch->array.length      = rows;
  const auto plan          = sirius::inspect_arrow_host_batch(
    &batch->array,
    &batch->schema,
    "bit offsets",
    {"b", "v"},
    declared({duckdb::LogicalType::BOOLEAN, duckdb::LogicalType::INTEGER}),
    *std::pmr::new_delete_resource());
  rmm::cuda_stream stream;
  import_test_resource resource;
  resource.counts->expected_stream = stream.value();
  std::unique_ptr<cudf::table> table;
  {
#if defined(__linux__)
    import_copy_observation copies{stream.value()};
    scoped_restore restore(active_import_copies);
    active_import_copies = &copies;
#endif
    table = sirius::import_arrow_host_table(
      plan, stream.view(), rmm::device_async_resource_ref{resource});
#if defined(__linux__)
    REQUIRE(copies.calls >= 2);
    REQUIRE_FALSE(copies.wrong_stream);
#endif
  }
  REQUIRE(resource.counts->attempts > 0);
  const auto booleans       = copy_column_to_host<bool>(table->get_column(0).view());
  const auto integers       = copy_column_to_host<std::int32_t>(table->get_column(1).view());
  const auto& nullable      = table->get_column(1);
  const auto& nullable_bool = table->get_column(0);
  REQUIRE(nullable.null_count() == rows / 7);
  std::vector<cudf::bitmask_type> mask((rows + 31) / 32);
  std::vector<cudf::bitmask_type> bool_mask((rows + 31) / 32);
  if (rows >= 7) {
    REQUIRE(cudaMemcpy(mask.data(),
                       nullable.view().null_mask(),
                       mask.size() * sizeof(cudf::bitmask_type),
                       cudaMemcpyDeviceToHost) == cudaSuccess);
  }
  const int bool_nulls = (child_offset + rows) / 5 - child_offset / 5;
  REQUIRE(nullable_bool.null_count() == bool_nulls);
  if (bool_nulls != 0) {
    REQUIRE(cudaMemcpy(bool_mask.data(),
                       nullable_bool.view().null_mask(),
                       bool_mask.size() * sizeof(cudf::bitmask_type),
                       cudaMemcpyDeviceToHost) == cudaSuccess);
  }
  for (int row = 0; row < rows; ++row) {
    const auto source_row = child_offset + row + 1;
    const bool bool_valid = source_row % 5 != 0;
    if (bool_nulls != 0) {
      REQUIRE(((bool_mask[row / 32] >> (row % 32)) & 1) == static_cast<unsigned>(bool_valid));
    }
    if (bool_valid) { REQUIRE(booleans[row] == (source_row % 3 == 0)); }
    const bool valid = (row + 1) % 7 != 0;
    if (rows >= 7) {
      REQUIRE(((mask[row / 32] >> (row % 32)) & 1) == static_cast<unsigned>(valid));
    }
    if (valid) { REQUIRE(integers[row] == row + 1); }
  }
  table.reset();
  stream.synchronize();
  REQUIRE(resource.counts->live_bytes == 0);
  REQUIRE_FALSE(resource.counts->wrong_stream);
}

#if defined(__linux__)
TEST_CASE("arrow_host_import: actual import classifies failed drains and preserves causes",
          "[arrow_host_import]")
{
  // Allocation failure + successful/failed drain, then failed success-path sync + failed drain.
  const int scenario = GENERATE(0, 1, 2);
  CAPTURE(scenario);
  auto batch = arrow_batch_from_sql("SELECT i::INTEGER AS a, i::INTEGER AS b FROM range(32) t(i)");
  bounded_host_resource host(std::numeric_limits<std::size_t>::max());
  const auto plan = sirius::inspect_arrow_host_batch(
    &batch->array,
    &batch->schema,
    "drain",
    {"a", "b"},
    declared({duckdb::LogicalType::INTEGER, duckdb::LogicalType::INTEGER}),
    host);
  rmm::cuda_stream stream;
  import_test_resource resource(scenario == 2 ? -1 : 1);
  resource.counts->expected_stream = stream.value();
  import_copy_observation copies{stream.value()};
  import_drain_fault fault{stream.value(),
                           *resource.counts,
                           copies,
                           {scenario == 0 ? cudaSuccess : cudaErrorUnknown, cudaErrorInvalidValue}};
  std::exception_ptr failure;
  {
    scoped_restore restore(active_import_drain_fault);
    scoped_restore restore_copies(active_import_copies);
    active_import_drain_fault = &fault;
    active_import_copies      = &copies;
    try {
      auto table = sirius::import_arrow_host_table(
        plan, stream.view(), rmm::device_async_resource_ref{resource});
    } catch (...) {
      failure = std::current_exception();
    }
  }
  REQUIRE(failure != nullptr);
  REQUIRE(fault.calls == (scenario == 2 ? 2 : 1));
  REQUIRE(resource.counts->attempts == 2);
  REQUIRE_FALSE(copies.wrong_stream);
  REQUIRE(fault.copy_calls[0] == (scenario == 2 ? 2 : 1));
  REQUIRE(fault.copy_bytes[0] == (scenario == 2 ? 256 : 128));
  REQUIRE(fault.live_bytes[fault.calls - 1] == 0);
  for (int call = 0; call < fault.calls; ++call) {
    REQUIRE(fault.actual_status[call] == cudaSuccess);
  }
  if (scenario != 0) {
    try {
      std::rethrow_exception(failure);
      FAIL("missing terminal drain error");
    } catch (const sirius::arrow_host_import_drain_error& error) {
      REQUIRE(error.cuda_status() == (scenario == 2 ? cudaErrorInvalidValue : cudaErrorUnknown));
      failure = error.original_failure();
      REQUIRE(failure != nullptr);
    }
  }
  if (scenario == 2) {
    REQUIRE(fault.live_bytes[0] == 256);
    REQUIRE(fault.copy_calls[1] == 2);
    REQUIRE(fault.copy_bytes[1] == 256);
    try {
      std::rethrow_exception(failure);
      FAIL("missing original success-sync error");
    } catch (const cudf::cuda_error& error) {
      REQUIRE(error.error_code() == cudaErrorUnknown);
      REQUIRE_THAT(error.what(), ContainsSubstring("cudaErrorUnknown"));
    }
  } else {
    REQUIRE(resource.counts->failed);
    REQUIRE_THROWS_MATCHES(std::rethrow_exception(failure),
                           rmm::out_of_memory,
                           ContainsSubstring("injected Arrow import allocation failure"));
  }
  REQUIRE(resource.counts->live_bytes == 0);
  REQUIRE_FALSE(resource.counts->wrong_stream);
  REQUIRE(batch->array.release != nullptr);
  REQUIRE(batch->schema.release != nullptr);
  REQUIRE(host.used > 0);
  REQUIRE(cudaStreamQuery(stream.value()) == cudaSuccess);
}
#endif

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

TEST_CASE("arrow_host_import: enabled scalar boundaries and typed empties", "[arrow_host_import]")
{
  const auto check = []<typename T>(const std::string& sql_type,
                                    duckdb::LogicalType type,
                                    const std::string& low_text,
                                    const std::string& high_text,
                                    T low,
                                    T high) {
    CAPTURE(sql_type);
    auto batch = arrow_batch_from_sql("SELECT x::" + sql_type + " AS a FROM (VALUES ('" + low_text +
                                      "'), (NULL), ('" + high_text + "')) t(x)");
    auto table = import(*batch, {"a"}, declared({type}));
    REQUIRE(table->get_column(0).type() == sirius::get_cudf_type(declared({type})[0]));
    REQUIRE(table->get_column(0).null_count() == 1);
    const auto values = copy_column_to_host<T>(table->get_column(0).view());
    REQUIRE(values.size() == 3);
    REQUIRE(values[0] == low);
    REQUIRE(values[2] == high);
    std::uint32_t mask = 0;
    REQUIRE(cudaMemcpy(&mask,
                       table->get_column(0).view().null_mask(),
                       sizeof(mask),
                       cudaMemcpyDeviceToHost) == cudaSuccess);
    REQUIRE((mask & 7) == 5);
    batch->array.length = 0;
    auto empty          = import(*batch, {"a"}, declared({type}));
    REQUIRE(empty->num_columns() == 1);
    REQUIRE(empty->num_rows() == 0);
    REQUIRE(empty->get_column(0).type() == table->get_column(0).type());
    REQUIRE(empty->get_column(0).null_count() == 0);
  };
  check("BOOLEAN", duckdb::LogicalType::BOOLEAN, "false", "true", false, true);
  check(
    "TINYINT", duckdb::LogicalType::TINYINT, "-128", "127", std::int8_t{-128}, std::int8_t{127});
  check("UTINYINT", duckdb::LogicalType::UTINYINT, "0", "255", std::uint8_t{0}, std::uint8_t{255});
  check("SMALLINT",
        duckdb::LogicalType::SMALLINT,
        "-32768",
        "32767",
        std::int16_t{-32768},
        std::int16_t{32767});
  check("USMALLINT",
        duckdb::LogicalType::USMALLINT,
        "0",
        "65535",
        std::uint16_t{0},
        std::uint16_t{65535});
  check("INTEGER",
        duckdb::LogicalType::INTEGER,
        "-2147483648",
        "2147483647",
        std::numeric_limits<std::int32_t>::min(),
        std::numeric_limits<std::int32_t>::max());
  check("UINTEGER",
        duckdb::LogicalType::UINTEGER,
        "0",
        "4294967295",
        std::uint32_t{0},
        std::numeric_limits<std::uint32_t>::max());
  check("BIGINT",
        duckdb::LogicalType::BIGINT,
        "-9223372036854775808",
        "9223372036854775807",
        std::numeric_limits<std::int64_t>::min(),
        std::numeric_limits<std::int64_t>::max());
  check("UBIGINT",
        duckdb::LogicalType::UBIGINT,
        "0",
        "18446744073709551615",
        std::uint64_t{0},
        std::numeric_limits<std::uint64_t>::max());
  check("FLOAT",
        duckdb::LogicalType::FLOAT,
        "-inf",
        "inf",
        -std::numeric_limits<float>::infinity(),
        std::numeric_limits<float>::infinity());
  check("DOUBLE",
        duckdb::LogicalType::DOUBLE,
        "-inf",
        "inf",
        -std::numeric_limits<double>::infinity(),
        std::numeric_limits<double>::infinity());
  check("VARCHAR", duckdb::LogicalType::VARCHAR, "", "hello", std::string{}, std::string{"hello"});
  check("DATE",
        duckdb::LogicalType::DATE,
        "1969-12-31",
        "1970-01-02",
        std::int32_t{-1},
        std::int32_t{1});
  check("TIMESTAMP_S",
        duckdb::LogicalType::TIMESTAMP_S,
        "1969-12-31 23:59:59",
        "1970-01-01 00:00:01",
        std::int64_t{-1},
        std::int64_t{1});
  check("TIMESTAMP_MS",
        duckdb::LogicalType::TIMESTAMP_MS,
        "1969-12-31 23:59:59.999",
        "1970-01-01 00:00:00.001",
        std::int64_t{-1},
        std::int64_t{1});
  check("TIMESTAMP",
        duckdb::LogicalType::TIMESTAMP,
        "1969-12-31 23:59:59.999999",
        "1970-01-01 00:00:00.000001",
        std::int64_t{-1},
        std::int64_t{1});
  check("TIMESTAMP_NS",
        duckdb::LogicalType::TIMESTAMP_NS,
        "1969-12-31 23:59:59.999999999",
        "1970-01-01 00:00:00.000000001",
        std::int64_t{-1},
        std::int64_t{1});
  for (const auto precision : {5, 9, 10, 18, 19, 38}) {
    const std::string digits(precision - 2, '9');
    const auto high = digits + ".99";
    __int128_t max  = 0;
    for (int i = 0; i < precision; ++i) {
      max = max * 10 + 9;
    }
    const auto type     = duckdb::LogicalType::DECIMAL(precision, 2);
    const auto sql_type = "DECIMAL(" + std::to_string(precision) + ",2)";
    if (precision <= 9) {
      check(sql_type,
            type,
            "-" + high,
            high,
            static_cast<std::int32_t>(-max),
            static_cast<std::int32_t>(max));
    } else if (precision <= 18) {
      check(sql_type,
            type,
            "-" + high,
            high,
            static_cast<std::int64_t>(-max),
            static_cast<std::int64_t>(max));
    } else {
      check(sql_type, type, "-" + high, high, -max, max);
    }
  }
  auto nan = arrow_batch_from_sql("SELECT 'nan'::FLOAT AS a, 'nan'::DOUBLE AS b");
  auto table =
    import(*nan, {"a", "b"}, declared({duckdb::LogicalType::FLOAT, duckdb::LogicalType::DOUBLE}));
  REQUIRE(std::isnan(copy_column_to_host<float>(table->get_column(0).view())[0]));
  REQUIRE(std::isnan(copy_column_to_host<double>(table->get_column(1).view())[0]));
}

TEST_CASE("arrow_host_import: failure drains asynchronous output deallocation",
          "[arrow_host_import]")
{
  auto batch      = arrow_batch_from_sql("SELECT 42::BIGINT AS a, 43::BIGINT AS b");
  const auto plan = sirius::inspect_arrow_host_batch(
    &batch->array,
    &batch->schema,
    "drain",
    {"a", "b"},
    declared({duckdb::LogicalType::BIGINT, duckdb::LogicalType::BIGINT}),
    *std::pmr::new_delete_resource());
  rmm::mr::cuda_async_memory_resource upstream;
  import_test_resource failing(1, rmm::device_async_resource_ref{upstream});
  failing.counts->delay_deallocation = true;
  rmm::cuda_stream stream;
  REQUIRE_THROWS_AS(
    sirius::import_arrow_host_table(plan, stream.view(), rmm::device_async_resource_ref{failing}),
    rmm::out_of_memory);
  REQUIRE(failing.counts->failed);
  REQUIRE(failing.counts->callback_status == cudaSuccess);
  REQUIRE(failing.counts->live_bytes == 0);
  REQUIRE(cudaStreamQuery(stream.value()) == cudaSuccess);
}

TEST_CASE("arrow_host_import: cardinality and all-null batches are explicit", "[arrow_host_import]")
{
  auto batch = arrow_batch_from_sql("SELECT NULL::INTEGER AS a, NULL::VARCHAR AS b FROM range(9)");
  const auto types = declared({duckdb::LogicalType::INTEGER, duckdb::LogicalType::VARCHAR});
  auto table       = import(*batch, {"a", "b"}, types);
  REQUIRE(table->num_rows() == 9);
  REQUIRE(table->get_column(0).null_count() == 9);
  REQUIRE(table->get_column(1).null_count() == 9);
  scoped_restore root_guard(batch->array);
  scoped_restore schema_guard(batch->schema);
  batch->array.n_children  = 0;
  batch->schema.n_children = 0;
  import_test_resource observed;
  REQUIRE_THROWS_WITH(import(*batch, {}, {}, rmm::device_async_resource_ref{observed}),
                      ContainsSubstring("physical row-count carrier"));
  REQUIRE(observed.counts->attempts == 0);
  batch->array.length = 0;
  auto empty          = import(*batch, {}, {}, rmm::device_async_resource_ref{observed});
  REQUIRE(empty->num_columns() == 0);
  REQUIRE(empty->num_rows() == 0);
  REQUIRE(observed.counts->attempts == 0);
}

TEST_CASE("arrow_host_import: null metadata is checked against selected bits",
          "[arrow_host_import]")
{
  auto batch = arrow_batch_from_sql(
    "SELECT CASE WHEN i = 1 THEN NULL ELSE i END::INTEGER AS a FROM range(3) t(i)");
  auto& child  = *batch->array.children[0];
  auto& schema = *batch->schema.children[0];
  scoped_restore child_guard(child);
  scoped_restore schema_guard(schema);
  const auto types = declared({duckdb::LogicalType::INTEGER});
  SECTION("unknown null count is derived")
  {
    child.null_count = -1;
    auto table       = import(*batch, {"a"}, types);
    REQUIRE(table->get_column(0).null_count() == 1);
  }
  SECTION("advertised zero contradicts bitmap")
  {
    child.null_count = 0;
    REQUIRE_THROWS_WITH(import(*batch, {"a"}, types), ContainsSubstring("inconsistent"));
  }
  SECTION("advertised positive count contradicts bitmap")
  {
    child.null_count = 2;
    REQUIRE_THROWS_WITH(import(*batch, {"a"}, types), ContainsSubstring("inconsistent"));
  }
  SECTION("nonnullable schema contradicts bitmap")
  {
    schema.flags &= ~ARROW_FLAG_NULLABLE;
    REQUIRE_THROWS_WITH(import(*batch, {"a"}, types), ContainsSubstring("inconsistent"));
  }
}

TEST_CASE("arrow_host_import: immutable plans import on independent worker streams",
          "[arrow_host_import]")
{
  auto batch = arrow_batch_from_sql("SELECT i::INTEGER AS a, i % 2 = 0 AS b FROM range(513) t(i)");
  const auto plan = sirius::inspect_arrow_host_batch(
    &batch->array,
    &batch->schema,
    "workers",
    {"a", "b"},
    declared({duckdb::LogicalType::INTEGER, duckdb::LogicalType::BOOLEAN}),
    *std::pmr::new_delete_resource());
  int device = -1;
  REQUIRE(cudaGetDevice(&device) == cudaSuccess);
  rmm::mr::cuda_async_memory_resource upstream;
  const auto run = [&]() {
    if (cudaSetDevice(device) != cudaSuccess) { throw std::runtime_error("select worker GPU"); }
    rmm::cuda_stream stream;
    auto table = sirius::import_arrow_host_table(
      plan, stream.view(), rmm::device_async_resource_ref{upstream});
    return std::pair{copy_column_to_host<std::int32_t>(table->get_column(0).view()),
                     copy_column_to_host<bool>(table->get_column(1).view())};
  };
  auto first       = std::async(std::launch::async, run);
  auto second      = std::async(std::launch::async, run);
  const auto left  = first.get();
  const auto right = second.get();
  REQUIRE(left == right);
  REQUIRE(left.first.size() == 513);
  for (std::size_t row = 0; row < left.first.size(); ++row) {
    REQUIRE(left.first[row] == row);
    REQUIRE(left.second[row] == (row % 2 == 0));
  }
}

TEST_CASE("arrow_host_import: decimal payload overflow is refused before device allocation",
          "[arrow_host_import]")
{
  for (const auto precision : {5, 9, 10, 18, 19, 38}) {
    CAPTURE(precision);
    auto batch =
      arrow_batch_from_sql("SELECT 0::DECIMAL(" + std::to_string(precision) + ",2) AS d");
    auto& child = *batch->array.children[0];
    scoped_restore guard(child);
    const auto* validity = child.buffers[0];
    __int128_t limit     = 1;
    for (int digit = 0; digit < precision; ++digit) {
      limit *= 10;
    }
    import_test_resource observed;
    for (auto value : {limit, -limit}) {
      const void* buffers[] = {validity, &value};
      child.buffers         = buffers;
      REQUIRE_THROWS_WITH(import(*batch,
                                 {"d"},
                                 declared({duckdb::LogicalType::DECIMAL(precision, 2)}),
                                 rmm::device_async_resource_ref{observed}),
                          ContainsSubstring("outside its declared precision"));
      REQUIRE(observed.counts->attempts == 0);
    }
  }
}

TEST_CASE("arrow_host_import: terminal drain errors retain structured causes",
          "[arrow_host_import]")
{
  const auto original = std::make_exception_ptr(std::runtime_error("original copy failure"));
  static_assert(noexcept(sirius::arrow_host_import_drain_error(cudaErrorUnknown, original)));
  const sirius::arrow_host_import_drain_error error(cudaErrorUnknown, original);
  REQUIRE(error.cuda_status() == cudaErrorUnknown);
  REQUIRE(error.original_failure() == original);
  REQUIRE(std::string_view(error.what()) == "Arrow import CUDA stream drain failed");
  REQUIRE_THROWS_WITH(std::rethrow_exception(error.original_failure()), "original copy failure");
}
