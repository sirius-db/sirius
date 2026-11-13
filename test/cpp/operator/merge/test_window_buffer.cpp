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

#include "catch.hpp"
#include "op/merge/detail/window_buffer.hpp"
#include "operator/operator_test_utils.hpp"

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>

#include <rmm/cuda_stream.hpp>
#include <rmm/device_buffer.hpp>

#include <cuda_runtime_api.h>

#include <cucascade/cudf/host_data_representation.hpp>
#include <cucascade/data/data_batch.hpp>
#include <cucascade/data/disk_data_representation.hpp>
#include <cucascade/data/row_range_read.hpp>
#include <cucascade/memory/reservation_manager_configurator.hpp>
#include <data/data_batch_utils.hpp>
#include <data/sirius_converter_registry.hpp>
#include <data/sirius_row_range_reader_registry.hpp>
#include <memory/sirius_memory_reservation_manager.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using sirius::op::merge::detail::window_buffer;
using sirius::test::operator_utils::tiered_memory_fixture;

constexpr std::array<std::size_t, 3> fetch_cols{0, 1, 2};

/// Host copy of rows of an (INT64, FLOAT64 nullable, STRING nullable) table.
struct host_rows {
  std::vector<int64_t> ints;
  std::vector<std::optional<double>> doubles;
  std::vector<std::optional<std::string>> strings;

  [[nodiscard]] std::size_t size() const { return ints.size(); }

  [[nodiscard]] host_rows slice(std::size_t begin, std::size_t end) const
  {
    using diff = std::ptrdiff_t;
    return {{ints.begin() + static_cast<diff>(begin), ints.begin() + static_cast<diff>(end)},
            {doubles.begin() + static_cast<diff>(begin), doubles.begin() + static_cast<diff>(end)},
            {strings.begin() + static_cast<diff>(begin), strings.begin() + static_cast<diff>(end)}};
  }

  bool operator==(const host_rows&) const = default;
};

/// Random rows; strings are 0..max_string_len bytes, about one value in seven is null.
host_rows make_rows(std::size_t num_rows, std::size_t max_string_len, std::mt19937& rng)
{
  std::uniform_int_distribution<int64_t> int_dist(-1'000'000, 1'000'000);
  std::uniform_int_distribution<std::size_t> len_dist(0, max_string_len);
  std::uniform_int_distribution<int> null_dist(0, 6);
  std::uniform_int_distribution<int> char_dist('a', 'z');
  host_rows rows;
  for (std::size_t i = 0; i < num_rows; ++i) {
    rows.ints.push_back(int_dist(rng));
    if (null_dist(rng) == 0) {
      rows.doubles.emplace_back(std::nullopt);
    } else {
      rows.doubles.emplace_back(static_cast<double>(int_dist(rng)) / 8.0);
    }
    if (null_dist(rng) == 0) {
      rows.strings.emplace_back(std::nullopt);
    } else {
      std::string s(len_dist(rng), ' ');
      for (auto& c : s) {
        c = static_cast<char>(char_dist(rng));
      }
      rows.strings.emplace_back(std::move(s));
    }
  }
  return rows;
}

template <typename T>
std::vector<T> device_to_host(const void* device_ptr, std::size_t count)
{
  std::vector<T> host(count);
  if (count > 0) {
    REQUIRE(cudaMemcpy(host.data(), device_ptr, count * sizeof(T), cudaMemcpyDeviceToHost) ==
            cudaSuccess);
  }
  return host;
}

template <typename Optional>
void set_validity(cudf::column& col, const std::vector<Optional>& values)
{
  std::vector<cudf::bitmask_type> words(
    cudf::bitmask_allocation_size_bytes(static_cast<cudf::size_type>(values.size())) /
      sizeof(cudf::bitmask_type),
    0);
  cudf::size_type nulls = 0;
  for (std::size_t i = 0; i < values.size(); ++i) {
    if (values[i].has_value()) {
      words[i / 32] |= cudf::bitmask_type{1} << (i % 32);
    } else {
      ++nulls;
    }
  }
  rmm::device_buffer mask(words.size() * sizeof(cudf::bitmask_type), cudf::get_default_stream());
  if (!words.empty()) {
    REQUIRE(cudaMemcpy(mask.data(), words.data(), mask.size(), cudaMemcpyHostToDevice) ==
            cudaSuccess);
  }
  col.set_null_mask(std::move(mask), nulls);
}

/// A GPU-tier batch in @p gpu_space holding @p rows.
std::shared_ptr<cucascade::data_batch> make_gpu_batch(cucascade::memory::memory_space& gpu_space,
                                                      const host_rows& rows,
                                                      ::cuda::stream_ref stream)
{
  using namespace sirius::test::operator_utils;
  auto const mr = get_resource_ref(gpu_space);
  auto const n  = static_cast<cudf::size_type>(rows.size());

  auto int_col = cudf::make_numeric_column(
    cudf::data_type{cudf::type_id::INT64}, n, cudf::mask_state::UNALLOCATED, stream, mr);
  auto dbl_col = cudf::make_numeric_column(
    cudf::data_type{cudf::type_id::FLOAT64}, n, cudf::mask_state::UNALLOCATED, stream, mr);
  std::vector<double> doubles(rows.size());
  std::vector<std::string> strings(rows.size());
  for (std::size_t i = 0; i < rows.size(); ++i) {
    doubles[i] = rows.doubles[i].value_or(-1.0);
    strings[i] = rows.strings[i].value_or("null-row");
  }
  if (n > 0) {
    REQUIRE(cudaMemcpy(int_col->mutable_view().data<int64_t>(),
                       rows.ints.data(),
                       rows.size() * sizeof(int64_t),
                       cudaMemcpyHostToDevice) == cudaSuccess);
    REQUIRE(cudaMemcpy(dbl_col->mutable_view().data<double>(),
                       doubles.data(),
                       rows.size() * sizeof(double),
                       cudaMemcpyHostToDevice) == cudaSuccess);
  }
  auto str_col = make_string_column(strings, stream, mr);
  set_validity(*dbl_col, rows.doubles);
  set_validity(*str_col, rows.strings);

  std::vector<std::unique_ptr<cudf::column>> cols;
  cols.push_back(std::move(int_col));
  cols.push_back(std::move(dbl_col));
  cols.push_back(std::move(str_col));
  stream.sync();
  return sirius::make_data_batch(std::make_unique<cudf::table>(std::move(cols)),
                                 gpu_space,
                                 stream,
                                 sirius::telemetry::batch_telemetry_info{});
}

/// A HOST-tier batch holding @p rows. The host converter needs a non-default @p stream.
std::shared_ptr<cucascade::data_batch> make_host_batch(tiered_memory_fixture& fixture,
                                                       const host_rows& rows,
                                                       ::cuda::stream_ref stream)
{
  auto batch = make_gpu_batch(*fixture.gpu0, rows, stream);
  {
    auto mut = batch->to_mutable();
    mut.convert_to<cucascade::host_data_representation>(
      sirius::converter_registry::get(), fixture.host0, stream);
  }
  stream.sync();
  return batch;
}

std::vector<bool> validity_to_host(const cudf::column_view& col)
{
  auto const n = static_cast<std::size_t>(col.size());
  std::vector<bool> valid(n, true);
  if (!col.nullable()) { return valid; }
  REQUIRE(col.offset() == 0);
  auto const words = device_to_host<cudf::bitmask_type>(
    col.null_mask(), static_cast<std::size_t>(cudf::num_bitmask_words(col.size())));
  cudf::size_type nulls = 0;
  for (std::size_t i = 0; i < n; ++i) {
    valid[i] = ((words[i / 32] >> (i % 32)) & 1U) != 0;
    if (!valid[i]) { ++nulls; }
  }
  REQUIRE(col.null_count() == nulls);
  return valid;
}

std::vector<int64_t> int_column_to_host(const cudf::column_view& col)
{
  REQUIRE(col.type().id() == cudf::type_id::INT64);
  REQUIRE(col.null_count() == 0);
  return device_to_host<int64_t>(col.head(), static_cast<std::size_t>(col.size()));
}

std::vector<std::optional<double>> double_column_to_host(const cudf::column_view& col)
{
  REQUIRE(col.type().id() == cudf::type_id::FLOAT64);
  auto const n       = static_cast<std::size_t>(col.size());
  auto const doubles = device_to_host<double>(col.head(), n);
  auto const valid   = validity_to_host(col);
  std::vector<std::optional<double>> out;
  for (std::size_t i = 0; i < n; ++i) {
    out.emplace_back(valid[i] ? std::optional<double>{doubles[i]} : std::nullopt);
  }
  return out;
}

std::vector<std::optional<std::string>> string_column_to_host(const cudf::column_view& col)
{
  REQUIRE(col.type().id() == cudf::type_id::STRING);
  auto const n = static_cast<std::size_t>(col.size());
  std::vector<std::optional<std::string>> out;
  if (n == 0) { return out; }
  auto const valid        = validity_to_host(col);
  auto const offsets_view = cudf::strings_column_view(col).offsets();
  REQUIRE(static_cast<std::size_t>(offsets_view.size()) == n + 1);
  std::vector<int64_t> offsets(n + 1);
  if (offsets_view.type().id() == cudf::type_id::INT32) {
    auto const narrow = device_to_host<int32_t>(offsets_view.head(), n + 1);
    std::copy(narrow.begin(), narrow.end(), offsets.begin());
  } else {
    REQUIRE(offsets_view.type().id() == cudf::type_id::INT64);
    offsets = device_to_host<int64_t>(offsets_view.head(), n + 1);
  }
  REQUIRE(offsets.front() == 0);
  auto const chars = device_to_host<char>(col.head(), static_cast<std::size_t>(offsets.back()));
  for (std::size_t i = 0; i < n; ++i) {
    REQUIRE(offsets[i] <= offsets[i + 1]);
    if (valid[i]) {
      out.emplace_back(std::string(chars.data() + offsets[i],
                                   static_cast<std::size_t>(offsets[i + 1] - offsets[i])));
    } else {
      out.emplace_back(std::nullopt);
    }
  }
  return out;
}

/// The window's rows copied to the host.
host_rows window_to_host(const window_buffer& window, const rmm::cuda_stream& stream)
{
  stream.synchronize();
  auto const view = window.view();
  REQUIRE(view.num_columns() == 3);
  REQUIRE(static_cast<std::size_t>(view.num_rows()) == window.size());
  return {int_column_to_host(view.column(0)),
          double_column_to_host(view.column(1)),
          string_column_to_host(view.column(2))};
}

/// Fixture with the memory spaces, the reader registry, and a stream the window works on.
struct window_buffer_fixture {
  tiered_memory_fixture memory;
  rmm::cuda_stream stream;

  window_buffer_fixture()
  {
    REQUIRE(memory.setup());
    // Left initialized: a live SiriusContext serves its readers from this registry.
    if (!sirius::row_range_reader_registry::is_initialized()) {
      sirius::row_range_reader_registry::initialize();
    }
  }

  [[nodiscard]] rmm::cuda_stream_view stream_view() const
  {
    return rmm::cuda_stream_view{stream.value()};
  }

  [[nodiscard]] window_buffer make_window(std::size_t window_rows, std::size_t est_string_bytes)
  {
    return window_buffer({cudf::data_type{cudf::type_id::INT64},
                          cudf::data_type{cudf::type_id::FLOAT64},
                          cudf::data_type{cudf::type_id::STRING}},
                         window_rows,
                         est_string_bytes,
                         stream_view(),
                         sirius::test::operator_utils::get_resource_ref(*memory.gpu0));
  }

  cucascade::row_range_result refill(window_buffer& window,
                                     const std::shared_ptr<cucascade::data_batch>& batch,
                                     std::size_t row_begin)
  {
    auto ro = batch->to_read_only();
    return window.refill(sirius::row_range_reader_registry::get(),
                         ro,
                         fetch_cols,
                         row_begin,
                         nullptr,
                         nullptr,
                         stream_view());
  }
};

/// Chars bytes of rows [begin, end) of @p rows, counting null rows as stored in the source.
std::size_t string_bytes(const host_rows& rows, std::size_t begin, std::size_t end)
{
  std::size_t bytes = 0;
  for (std::size_t i = begin; i < end; ++i) {
    bytes += rows.strings[i].value_or("null-row").size();
  }
  return bytes;
}

}  // anonymous namespace

TEST_CASE("window_buffer rejects nested types", "[window_buffer]")
{
  window_buffer_fixture fixture;
  REQUIRE_THROWS_AS(
    window_buffer({cudf::data_type{cudf::type_id::INT32}, cudf::data_type{cudf::type_id::LIST}},
                  16,
                  8,
                  fixture.stream_view(),
                  sirius::test::operator_utils::get_resource_ref(*fixture.memory.gpu0)),
    std::invalid_argument);
}

TEST_CASE("window_buffer starts empty with both slots allocated", "[window_buffer]")
{
  window_buffer_fixture fixture;
  auto window = fixture.make_window(100, 16);

  REQUIRE(window.size() == 0);
  REQUIRE(window.capacity_rows() == 100);
  auto const mask_bytes = cudf::bitmask_allocation_size_bytes(100);
  auto const per_slot = (100 * sizeof(int64_t) + mask_bytes) + (100 * sizeof(double) + mask_bytes) +
                        (100 * 16 * 5 / 4 + mask_bytes + 101 * sizeof(int32_t));
  REQUIRE(window.device_bytes() == 2 * per_slot);

  auto const view = window.view();
  REQUIRE(view.num_columns() == 3);
  REQUIRE(view.num_rows() == 0);
  window.consume(0, fixture.stream_view());
  REQUIRE(window.size() == 0);
  REQUIRE_THROWS_AS(window.consume(1, fixture.stream_view()), std::out_of_range);
}

TEST_CASE("window_buffer refills append rows of host batches", "[window_buffer]")
{
  window_buffer_fixture fixture;
  std::mt19937 rng(7);
  constexpr std::size_t batch_rows = 100;
  auto const rows                  = make_rows(3 * batch_rows, 12, rng);
  std::vector<std::shared_ptr<cucascade::data_batch>> batches;
  for (std::size_t b = 0; b < 3; ++b) {
    batches.push_back(make_host_batch(
      fixture.memory, rows.slice(b * batch_rows, (b + 1) * batch_rows), fixture.stream));
  }

  auto window = fixture.make_window(256, 12);

  // Appends at rows 100 and 200: neither is a multiple of the 32-bit mask word.
  auto result = fixture.refill(window, batches[0], 0);
  REQUIRE(result.rows_written == batch_rows);
  REQUIRE(result.source_exhausted);
  REQUIRE(result.data_bytes_written.size() == 3);
  REQUIRE(result.data_bytes_written[0] == batch_rows * sizeof(int64_t));
  REQUIRE(result.data_bytes_written[2] == string_bytes(rows, 0, batch_rows));
  REQUIRE(window_to_host(window, fixture.stream) == rows.slice(0, 100));

  result = fixture.refill(window, batches[1], 0);
  REQUIRE(result.rows_written == batch_rows);
  REQUIRE(result.source_exhausted);
  REQUIRE(window_to_host(window, fixture.stream) == rows.slice(0, 200));

  result = fixture.refill(window, batches[2], 0);
  REQUIRE(result.rows_written == 56);
  REQUIRE_FALSE(result.source_exhausted);
  REQUIRE(window.size() == 256);
  REQUIRE(window_to_host(window, fixture.stream) == rows.slice(0, 256));

  // A full window reads nothing and does not grow.
  auto const bytes_before = window.device_bytes();
  result                  = fixture.refill(window, batches[2], 56);
  REQUIRE(result.rows_written == 0);
  REQUIRE_FALSE(result.source_exhausted);
  REQUIRE(window.device_bytes() == bytes_before);
  REQUIRE(window_to_host(window, fixture.stream) == rows.slice(0, 256));

  // An exhausted source reads nothing.
  window.consume(256, fixture.stream_view());
  result = fixture.refill(window, batches[2], batch_rows);
  REQUIRE(result.rows_written == 0);
  REQUIRE(result.source_exhausted);
  REQUIRE(window.size() == 0);
}

TEST_CASE("window_buffer consume carries the tail and refill appends after it", "[window_buffer]")
{
  window_buffer_fixture fixture;
  std::mt19937 rng(11);
  constexpr std::size_t window_rows = 200;
  auto const rows                   = make_rows(1000, 12, rng);
  auto const batch                  = make_host_batch(fixture.memory, rows, fixture.stream);

  for (std::size_t consumed : {std::size_t{0},
                               std::size_t{1},
                               std::size_t{31},
                               std::size_t{32},
                               std::size_t{33},
                               window_rows}) {
    CAPTURE(consumed);
    auto window = fixture.make_window(window_rows, 12);
    REQUIRE(fixture.refill(window, batch, 0).rows_written == window_rows);

    window.consume(consumed, fixture.stream_view());
    REQUIRE(window.size() == window_rows - consumed);
    REQUIRE(window_to_host(window, fixture.stream) == rows.slice(consumed, window_rows));

    auto const result = fixture.refill(window, batch, window_rows);
    REQUIRE(result.rows_written == consumed);
    REQUIRE(result.data_bytes_written[2] ==
            string_bytes(rows, window_rows, window_rows + consumed));
    REQUIRE(window.size() == window_rows);
    REQUIRE(window_to_host(window, fixture.stream) == rows.slice(consumed, consumed + window_rows));

    // Two consumes in a row, with no refill between them.
    window.consume(33, fixture.stream_view());
    window.consume(31, fixture.stream_view());
    REQUIRE(window_to_host(window, fixture.stream) ==
            rows.slice(consumed + 64, consumed + window_rows));
    REQUIRE(fixture.refill(window, batch, consumed + window_rows).rows_written == 64);
    REQUIRE(window_to_host(window, fixture.stream) ==
            rows.slice(consumed + 64, consumed + 64 + window_rows));
  }
}

TEST_CASE("window_buffer grows the chars slots for a row that does not fit", "[window_buffer]")
{
  window_buffer_fixture fixture;
  std::mt19937 rng(13);
  auto rows        = make_rows(8, 4, rng);
  rows.strings[0]  = std::string(4, 'a');
  rows.strings[1]  = std::string(4, 'b');
  rows.strings[2]  = std::string(4, 'c');
  rows.strings[3]  = std::string(100, 'x');
  rows.strings[5]  = std::string(1000, 'y');
  auto const batch = make_host_batch(fixture.memory, rows, fixture.stream);

  // 8 rows of an estimated 4 bytes: the chars slot is 64 bytes.
  auto window             = fixture.make_window(8, 4);
  auto const bytes_before = window.device_bytes();

  // Rows 0..2 fit; row 3 does not, so the read stops before it without growing.
  auto result = fixture.refill(window, batch, 0);
  REQUIRE(result.rows_written == 3);
  REQUIRE_FALSE(result.source_exhausted);
  REQUIRE(window.device_bytes() == bytes_before);

  // Row 3 alone does not fit behind the carried tail: both chars slots double, the tail survives.
  window.consume(2, fixture.stream_view());
  result = fixture.refill(window, batch, 3);
  REQUIRE(result.rows_written == 2);
  REQUIRE(window.device_bytes() == bytes_before + 2 * 64);
  REQUIRE(window_to_host(window, fixture.stream) == rows.slice(2, 5));

  // Row 5 does not fit even after doubling once more.
  window.consume(3, fixture.stream_view());
  REQUIRE_THROWS_AS(fixture.refill(window, batch, 5), std::runtime_error);
  REQUIRE(window.size() == 0);
  REQUIRE(window.device_bytes() == bytes_before + 2 * 64 + 2 * 128);

  // The window stays usable: the slots have grown, so the retry fits the row.
  REQUIRE_THROWS_AS(fixture.refill(window, batch, 5), std::runtime_error);
  result = fixture.refill(window, batch, 5);
  REQUIRE(result.rows_written == 3);
  REQUIRE(result.source_exhausted);
  REQUIRE(window_to_host(window, fixture.stream) == rows.slice(5, 8));
}

TEST_CASE("window_buffer matches a host model over 1000 consume/refill cycles", "[window_buffer]")
{
  window_buffer_fixture fixture;
  std::mt19937 rng(17);
  constexpr std::size_t batch_rows  = 20'000;
  constexpr std::size_t num_batches = 8;
  constexpr std::size_t window_rows = 300;
  auto const rows                   = make_rows(batch_rows * num_batches, 20, rng);
  std::vector<std::shared_ptr<cucascade::data_batch>> batches;
  for (std::size_t b = 0; b < num_batches; ++b) {
    batches.push_back(make_host_batch(
      fixture.memory, rows.slice(b * batch_rows, (b + 1) * batch_rows), fixture.stream));
  }

  // An estimate below the 10-byte average, so the chars slot (not the row count) often ends a
  // refill.
  auto window = fixture.make_window(window_rows, 8);
  std::size_t window_begin{0};  // Source row of the window's first row.
  std::size_t read_end{0};      // Source row after the window's last row.
  std::size_t chars_limited{0};
  for (int cycle = 0; cycle < 1000; ++cycle) {
    CAPTURE(cycle);
    while (window.size() < window_rows && read_end < rows.size()) {
      auto const result =
        fixture.refill(window, batches[read_end / batch_rows], read_end % batch_rows);
      read_end += result.rows_written;
      REQUIRE(result.source_exhausted == (read_end % batch_rows == 0));
      if (!result.source_exhausted) {
        if (window.size() < window_rows) { ++chars_limited; }
        break;
      }
    }
    REQUIRE(window.size() == read_end - window_begin);
    REQUIRE(window_to_host(window, fixture.stream) == rows.slice(window_begin, read_end));

    std::uniform_int_distribution<std::size_t> consume_dist(0, window.size());
    auto const consumed = consume_dist(rng);
    window.consume(consumed, fixture.stream_view());
    window_begin += consumed;
    if (cycle % 16 == 0) {
      REQUIRE(window_to_host(window, fixture.stream) == rows.slice(window_begin, read_end));
    }
  }
  REQUIRE(window_begin > 0);
  REQUIRE(chars_limited > 0);
}

TEST_CASE("window_buffer reads a GPU-tier batch the same way", "[window_buffer]")
{
  window_buffer_fixture fixture;
  std::mt19937 rng(19);
  auto const rows  = make_rows(150, 10, rng);
  auto const batch = make_gpu_batch(*fixture.memory.gpu0, rows, fixture.stream);

  auto window = fixture.make_window(64, 10);
  REQUIRE(fixture.refill(window, batch, 0).rows_written == 64);
  window.consume(33, fixture.stream_view());
  REQUIRE(fixture.refill(window, batch, 64).rows_written == 33);
  REQUIRE(window_to_host(window, fixture.stream) == rows.slice(33, 97));
}

TEST_CASE("window_buffer reads a DISK-tier batch the same way", "[window_buffer]")
{
  // The shared fixture has no disk space: build a manager with GPU, HOST and DISK tiers.
  sirius::converter_registry::reset_for_testing();
  std::string tmp_template =
    (std::filesystem::temp_directory_path() / "sirius_window_buffer_XXXXXX").string();
  REQUIRE(mkdtemp(tmp_template.data()) != nullptr);
  std::filesystem::path const tmp_dir{tmp_template};

  {
    cucascade::memory::reservation_manager_configurator builder;
    builder.set_number_of_gpus(1)
      .set_gpu_usage_limit(std::size_t{256} << 20)
      .set_reservation_fraction_per_gpu(0.75)
      .set_per_numa_region_capacity(std::size_t{1} << 30)
      .use_gpu_id_as_host_id()
      .set_reservation_fraction_per_numa_region(0.75)
      .set_disk_mounting_point(0, std::size_t{1} << 30, tmp_dir.string());
    auto manager =
      std::make_unique<sirius::memory::sirius_memory_reservation_manager>(builder.build());
    sirius::converter_registry::initialize();
    if (!sirius::row_range_reader_registry::is_initialized()) {
      sirius::row_range_reader_registry::initialize();
    }
    auto* gpu_space  = manager->get_memory_space(cucascade::memory::Tier::GPU, 0);
    auto* disk_space = manager->get_memory_space(cucascade::memory::Tier::DISK, 0);
    REQUIRE(gpu_space != nullptr);
    REQUIRE(disk_space != nullptr);

    rmm::cuda_stream stream;
    rmm::cuda_stream_view const sv{stream.value()};
    std::mt19937 rng(23);
    auto const rows  = make_rows(500, 16, rng);
    auto const batch = make_gpu_batch(*gpu_space, rows, stream);
    {
      auto mut = batch->to_mutable();
      mut.convert_to<cucascade::disk_data_representation>(
        sirius::converter_registry::get(), disk_space, stream);
    }
    stream.synchronize();
    REQUIRE(batch->to_read_only().get_memory_space()->get_tier() == cucascade::memory::Tier::DISK);

    {
      window_buffer window({cudf::data_type{cudf::type_id::INT64},
                            cudf::data_type{cudf::type_id::FLOAT64},
                            cudf::data_type{cudf::type_id::STRING}},
                           100,
                           16,
                           sv,
                           sirius::test::operator_utils::get_resource_ref(*gpu_space));
      auto refill = [&](std::size_t row_begin) {
        auto ro = batch->to_read_only();
        return window.refill(sirius::row_range_reader_registry::get(),
                             ro,
                             fetch_cols,
                             row_begin,
                             nullptr,
                             nullptr,
                             sv);
      };
      std::size_t window_begin{0};
      std::size_t read_end{0};
      for (std::size_t consumed :
           {std::size_t{0}, std::size_t{1}, std::size_t{31}, std::size_t{33}, std::size_t{64}}) {
        CAPTURE(consumed);
        window.consume(consumed, sv);
        window_begin += consumed;
        while (window.size() < window.capacity_rows() && read_end < rows.size()) {
          auto const result = refill(read_end);
          if (result.rows_written == 0) { break; }
          read_end += result.rows_written;
        }
        REQUIRE(window_to_host(window, stream) == rows.slice(window_begin, read_end));
      }
      REQUIRE(read_end > 200);
    }
    manager->shutdown();
  }
  sirius::converter_registry::shutdown();
  std::filesystem::remove_all(tmp_dir);
}

TEST_CASE("window_buffer propagates row_range_unsupported_error from the registry",
          "[window_buffer]")
{
  window_buffer_fixture fixture;
  std::mt19937 rng(29);
  auto const rows  = make_rows(40, 8, rng);
  auto const batch = make_host_batch(fixture.memory, rows, fixture.stream);
  auto window      = fixture.make_window(32, 8);

  // A registry with no readers cannot serve a HOST batch.
  cucascade::row_range_reader_registry const empty;
  {
    auto ro = batch->to_read_only();
    REQUIRE_THROWS_AS(
      window.refill(empty, ro, fetch_cols, 0, nullptr, nullptr, fixture.stream_view()),
      cucascade::row_range_unsupported_error);
  }
  REQUIRE(window.size() == 0);

  // The window is still usable with the real registry.
  REQUIRE(fixture.refill(window, batch, 0).rows_written == 32);
  REQUIRE(window_to_host(window, fixture.stream) == rows.slice(0, 32));
}

TEST_CASE("window_buffer refill rejects a fetch_cols size that does not match the columns",
          "[window_buffer]")
{
  window_buffer_fixture fixture;
  std::mt19937 rng(31);
  auto const batch = make_host_batch(fixture.memory, make_rows(10, 4, rng), fixture.stream);
  auto window      = fixture.make_window(16, 4);
  std::array<std::size_t, 2> const two_cols{0, 1};
  auto ro = batch->to_read_only();
  REQUIRE_THROWS_AS(window.refill(sirius::row_range_reader_registry::get(),
                                  ro,
                                  two_cols,
                                  0,
                                  nullptr,
                                  nullptr,
                                  fixture.stream_view()),
                    std::invalid_argument);
  REQUIRE(window.size() == 0);
}

TEST_CASE("window_buffer fetches a reordered subset of the source columns", "[window_buffer]")
{
  window_buffer_fixture fixture;
  std::mt19937 rng(37);
  auto const rows  = make_rows(120, 9, rng);
  auto const batch = make_host_batch(fixture.memory, rows, fixture.stream);

  window_buffer window(
    {cudf::data_type{cudf::type_id::STRING}, cudf::data_type{cudf::type_id::INT64}},
    50,
    9,
    fixture.stream_view(),
    sirius::test::operator_utils::get_resource_ref(*fixture.memory.gpu0));
  std::array<std::size_t, 2> const cols{2, 0};
  auto refill = [&](std::size_t row_begin) {
    auto ro = batch->to_read_only();
    return window.refill(sirius::row_range_reader_registry::get(),
                         ro,
                         cols,
                         row_begin,
                         nullptr,
                         nullptr,
                         fixture.stream_view());
  };
  auto check = [&](std::size_t begin, std::size_t end) {
    fixture.stream.synchronize();
    auto const view = window.view();
    REQUIRE(view.num_columns() == 2);
    auto const expected = rows.slice(begin, end);
    REQUIRE(string_column_to_host(view.column(0)) == expected.strings);
    REQUIRE(int_column_to_host(view.column(1)) == expected.ints);
  };

  std::size_t read_end = 0;
  while (window.size() < 50) {
    auto const r = refill(read_end);
    REQUIRE(r.rows_written > 0);
    read_end += r.rows_written;
  }
  check(0, read_end);
  window.consume(17, fixture.stream_view());
  check(17, read_end);
  while (window.size() < 50) {
    auto const r = refill(read_end);
    REQUIRE(r.rows_written > 0);
    read_end += r.rows_written;
  }
  check(17, read_end);
}

TEST_CASE("window_buffer without STRING columns carries unaligned null masks", "[window_buffer]")
{
  window_buffer_fixture fixture;
  std::mt19937 rng(41);
  auto const rows  = make_rows(1000, 1, rng);
  auto const batch = make_host_batch(fixture.memory, rows, fixture.stream);

  constexpr std::size_t window_rows = 97;
  window_buffer window(
    {cudf::data_type{cudf::type_id::FLOAT64}, cudf::data_type{cudf::type_id::INT64}},
    window_rows,
    0,
    fixture.stream_view(),
    sirius::test::operator_utils::get_resource_ref(*fixture.memory.gpu0));
  auto const mask_bytes = cudf::bitmask_allocation_size_bytes(window_rows);
  REQUIRE(window.device_bytes() == 2 * (window_rows * sizeof(double) + mask_bytes +
                                        window_rows * sizeof(int64_t) + mask_bytes));
  std::array<std::size_t, 2> const cols{1, 0};

  std::size_t window_begin = 0;
  std::size_t read_end     = 0;
  for (std::size_t consumed : {0, 1, 7, 31, 32, 33, 63, 64, 65, 96, 97, 5, 40}) {
    CAPTURE(consumed);
    consumed = std::min(consumed, window.size());
    window.consume(consumed, fixture.stream_view());
    window_begin += consumed;
    auto ro           = batch->to_read_only();
    auto const result = window.refill(sirius::row_range_reader_registry::get(),
                                      ro,
                                      cols,
                                      read_end,
                                      nullptr,
                                      nullptr,
                                      fixture.stream_view());
    read_end += result.rows_written;
    REQUIRE(window.size() == std::min(window_rows, rows.size() - window_begin));
    fixture.stream.synchronize();
    auto const view     = window.view();
    auto const expected = rows.slice(window_begin, read_end);
    REQUIRE(double_column_to_host(view.column(0)) == expected.doubles);
    REQUIRE(int_column_to_host(view.column(1)) == expected.ints);
  }
}

TEST_CASE("window_buffer orders work across a change of stream", "[window_buffer]")
{
  window_buffer_fixture fixture;
  rmm::cuda_stream other;
  rmm::cuda_stream_view const other_view{other.value()};
  std::mt19937 rng(43);
  auto const rows  = make_rows(2000, 24, rng);
  auto const batch = make_host_batch(fixture.memory, rows, fixture.stream);
  auto window      = fixture.make_window(500, 24);

  std::size_t window_begin = 0;
  std::size_t read_end     = 0;
  for (int i = 0; i < 20; ++i) {
    CAPTURE(i);
    // Alternate streams between every call with no host synchronization in between.
    auto const s = (i % 2 == 0) ? fixture.stream_view() : other_view;
    while (window.size() < window.capacity_rows() && read_end < rows.size()) {
      auto ro           = batch->to_read_only();
      auto const result = window.refill(
        sirius::row_range_reader_registry::get(), ro, fetch_cols, read_end, nullptr, nullptr, s);
      if (result.rows_written == 0) { break; }
      read_end += result.rows_written;
    }
    auto const consumed =
      std::min<std::size_t>(window.size(), 37 + 13 * static_cast<std::size_t>(i));
    window.consume(consumed, (i % 2 == 0) ? other_view : fixture.stream_view());
    window_begin += consumed;
  }
  other.synchronize();
  REQUIRE(window_to_host(window, fixture.stream) == rows.slice(window_begin, read_end));
}
