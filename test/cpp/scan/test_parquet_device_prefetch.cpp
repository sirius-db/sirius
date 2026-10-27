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

// The Parquet consumer's device route: decoding from column chunks a device-tier readahead already
// landed on the GPU, the checks that reject chunks that do not match what the reader expects, and
// the fall back to the demand (general) read.

#include "catch.hpp"
#include "exec/semi_future.hpp"
#include "io/io_context.hpp"
#include "io/kvikio/kvikio_context.hpp"
#include "io/types.hpp"
#include "op/scan/parquet_gpu_ingestible.hpp"
#include "op/scan/parquet_materialize.hpp"
#include "op/scan/prefetched_ranges.hpp"
#include "utils/dynamic_filter_test_utils.hpp"
#include "utils/gpu_execution_fixture.hpp"
#include "utils/parquet_fixture_utils.hpp"
#include "utils/test_validation_utility.hpp"

#include <cudf/ast/expressions.hpp>
#include <cudf/io/datasource.hpp>
#include <cudf/io/experimental/hybrid_scan.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/io/text/byte_range_info.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/cuda_stream.hpp>
#include <rmm/device_buffer.hpp>

#include <cuda_runtime_api.h>

#include <duckdb.hpp>

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <memory>
#include <numeric>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

namespace fs   = std::filesystem;
namespace scan = sirius::op::scan;

using byte_range = cudf::io::text::byte_range_info;
using scan::prefetched_chunk_spans;
using scan::prefetched_ranges;
using release_reason = prefetched_ranges::release_reason;

constexpr cudf::size_type k_rows       = 10240;
constexpr cudf::size_type k_row_groups = 5;

/// `x` BIGINT (offset + i), `y` INTEGER, `z` VARCHAR; k_row_groups row groups.
fs::path write_rows(duckdb::Connection& con,
                    fs::path const& dir,
                    std::string const& table,
                    std::int64_t offset)
{
  auto run = [&](std::string const& sql) {
    auto result = con.Query(sql);
    REQUIRE(result);
    INFO((result->HasError() ? result->GetError() : ""));
    REQUIRE_FALSE(result->HasError());
  };
  run("DROP TABLE IF EXISTS " + table);
  run("CREATE TABLE " + table + " AS SELECT (i + " + std::to_string(offset) +
      ")::BIGINT AS x, (i * 7)::INTEGER AS y, ('v' || i::VARCHAR) AS z FROM range(" +
      std::to_string(k_rows) + ") t(i)");
  auto path = dir / (table + ".parquet");
  run("COPY " + table + " TO '" + path.string() + "' (FORMAT PARQUET, ROW_GROUP_SIZE " +
      std::to_string(k_rows / k_row_groups) + ")");
  return path;
}

/// The parsed footer of the parquet file at @p path.
cudf::io::parquet::FileMetaData read_metadata(fs::path const& path)
{
  auto source          = cudf::io::datasource::create(path.string());
  auto constexpr tail  = sizeof(cudf::io::parquet::file_ender_s);
  auto const file_size = source->size();
  REQUIRE(file_size >= tail);
  auto const ender          = source->host_read(file_size - tail, tail);
  std::uint32_t footer_size = 0;
  std::memcpy(&footer_size, ender->data(), sizeof(footer_size));
  auto const footer = source->host_read(file_size - tail - footer_size, footer_size);
  auto const opts   = cudf::io::parquet_reader_options::builder().build();
  cudf::io::parquet::experimental::hybrid_scan_reader reader{
    cudf::host_span<std::uint8_t const>(footer->data(), footer->size()), opts};
  return reader.parquet_metadata();
}

/// A source with a null datasource: the device route never reads through it (only error messages
/// look at its path), so these tests need no IO backend.
scan::parquet_source source_for(fs::path const& path)
{
  auto metadata = std::make_shared<cudf::io::parquet::FileMetaData const>(read_metadata(path));
  std::vector<cudf::size_type> groups(metadata->row_groups.size());
  std::iota(groups.begin(), groups.end(), 0);
  REQUIRE(groups.size() == k_row_groups);
  return {nullptr, std::move(metadata), std::move(groups)};
}

/// One file's column chunks copied to the device, the shape a device-tier prefetch lands.
struct staged_chunks {
  std::vector<rmm::device_buffer> buffers;
  std::vector<cudf::device_span<std::uint8_t const>> spans;
};

staged_chunks stage_ranges(fs::path const& path,
                           std::span<byte_range const> ranges,
                           ::cuda::stream_ref stream)
{
  auto source = cudf::io::datasource::create(path.string());
  staged_chunks staged;
  for (auto const& range : ranges) {
    auto const host = source->host_read(static_cast<std::size_t>(range.offset()),
                                        static_cast<std::size_t>(range.size()));
    staged.buffers.emplace_back(host->data(), host->size(), stream);
    // The host copy dies with this iteration.
    REQUIRE(cudaStreamSynchronize(stream.get()) == cudaSuccess);
  }
  for (auto const& buffer : staged.buffers) {
    staged.spans.emplace_back(static_cast<std::uint8_t const*>(buffer.data()), buffer.size());
  }
  return staged;
}

struct staged_source {
  std::vector<byte_range> ranges;
  staged_chunks chunks;
};

staged_source stage(scan::parquet_source const& source,
                    fs::path const& path,
                    cudf::io::parquet_reader_options const& options,
                    ::cuda::stream_ref stream)
{
  staged_source out;
  out.ranges = scan::column_chunk_ranges(*source.metadata, options, source.row_group_indices);
  out.chunks = stage_ranges(path, out.ranges, stream);
  return out;
}

bool tables_equal(cudf::table_view lhs, cudf::table_view rhs, ::cuda::stream_ref stream)
{
  return sirius::test::expect_tables_equivalent_impl(lhs, rhs, stream);
}

/// Two same-schema parquet files with different values, written by a CPU DuckDB of their own.
struct parquet_files {
  sirius::test::scratch_dir scratch{"parquet_device_prefetch"};
  fs::path first;
  fs::path second;

  parquet_files()
  {
    duckdb::DuckDB db(nullptr);
    duckdb::Connection con(db);
    first  = write_rows(con, scratch.path(), "device_prefetch_first", 0);
    second = write_rows(con, scratch.path(), "device_prefetch_second", 1'000'000);
  }
};

}  // namespace

TEST_CASE("materialize_parquet decodes prefetched device spans like read_parquet",
          "[scan][parquet][device_prefetch]")
{
  parquet_files const files;
  auto const stream  = cudf::get_default_stream();
  auto const mr      = cudf::get_current_device_resource_ref();
  auto const options = cudf::io::parquet_reader_options::builder().build();

  SECTION("one source")
  {
    std::vector<scan::parquet_source> const sources{source_for(files.first)};
    auto const staged = stage(sources.front(), files.first, options, stream);
    std::vector<std::vector<byte_range>> const ranges{staged.ranges};
    std::vector<prefetched_chunk_spans> prefetched;
    prefetched.emplace_back(staged.chunks.spans);
    REQUIRE(scan::can_materialize_from_prefetched(sources, ranges, prefetched));

    auto const from_device =
      scan::materialize_parquet(sources, options, ranges, prefetched, stream, mr);
    REQUIRE(from_device);
    CHECK(from_device->num_rows() == k_rows);
    auto const expected =
      cudf::io::read_parquet(
        cudf::io::parquet_reader_options::builder(cudf::io::source_info{files.first.string()})
          .build(),
        stream)
        .tbl;
    CHECK(tables_equal(from_device->view(), expected->view(), stream));
  }

  SECTION("two sources, decoded in source order")
  {
    std::vector<scan::parquet_source> const sources{source_for(files.first),
                                                    source_for(files.second)};
    auto const first  = stage(sources[0], files.first, options, stream);
    auto const second = stage(sources[1], files.second, options, stream);
    std::vector<std::vector<byte_range>> const ranges{first.ranges, second.ranges};
    std::vector<prefetched_chunk_spans> prefetched;
    prefetched.emplace_back(first.chunks.spans);
    prefetched.emplace_back(second.chunks.spans);

    auto const from_device =
      scan::materialize_parquet(sources, options, ranges, prefetched, stream, mr);
    REQUIRE(from_device);
    CHECK(from_device->num_rows() == 2 * k_rows);
    auto const expected =
      cudf::io::read_parquet(
        cudf::io::parquet_reader_options::builder(cudf::io::source_info{std::vector<std::string>{
                                                    files.first.string(), files.second.string()}})
          .build(),
        stream)
        .tbl;
    CHECK(tables_equal(from_device->view(), expected->view(), stream));
  }
}

TEST_CASE("prefetched spans that do not match the reader are rejected before decoding",
          "[scan][parquet][device_prefetch]")
{
  parquet_files const files;
  auto const stream  = cudf::get_default_stream();
  auto const mr      = cudf::get_current_device_resource_ref();
  auto const options = cudf::io::parquet_reader_options::builder().build();

  std::vector<scan::parquet_source> const sources{source_for(files.first)};
  auto const staged = stage(sources.front(), files.first, options, stream);
  std::vector<std::vector<byte_range>> const ranges{staged.ranges};
  auto const& spans = staged.chunks.spans;
  REQUIRE(spans.size() > 1);

  SECTION("a span dropped")
  {
    std::vector<prefetched_chunk_spans> dropped;
    dropped.emplace_back(std::span(spans).first(spans.size() - 1));
    CHECK_FALSE(scan::can_materialize_from_prefetched(sources, ranges, dropped));
    CHECK_THROWS_AS(scan::materialize_parquet(sources, options, ranges, dropped, stream, mr),
                    std::invalid_argument);
    // Without the split's ranges only the reader's own enumeration can tell.
    CHECK_THROWS_AS(scan::materialize_parquet(sources, options, {}, dropped, stream, mr),
                    scan::prefetched_mismatch);
  }

  SECTION("a span of the wrong size with the right count")
  {
    auto resized = spans;
    resized[0]   = cudf::device_span<std::uint8_t const>(resized[0].data(), resized[0].size() - 1);
    std::vector<prefetched_chunk_spans> wrong;
    wrong.emplace_back(resized);
    CHECK_FALSE(scan::can_materialize_from_prefetched(sources, ranges, wrong));
    CHECK_THROWS_AS(scan::materialize_parquet(sources, options, {}, wrong, stream, mr),
                    scan::prefetched_mismatch);
  }

  SECTION("the claimed file offsets differ from the reader's")
  {
    // Same sizes, so the shape check passes: only the exact offset check can refuse it.
    auto shifted  = ranges;
    shifted[0][0] = byte_range(shifted[0][0].offset() + 1, shifted[0][0].size());
    std::vector<prefetched_chunk_spans> prefetched;
    prefetched.emplace_back(spans);
    CHECK(scan::can_materialize_from_prefetched(sources, shifted, prefetched));
    CHECK_THROWS_AS(scan::materialize_parquet(sources, options, shifted, prefetched, stream, mr),
                    scan::prefetched_mismatch);
  }

  SECTION("an empty entry")
  {
    std::vector<prefetched_chunk_spans> empty(1);
    CHECK_FALSE(scan::can_materialize_from_prefetched(sources, {}, empty));
    CHECK_THROWS_AS(scan::materialize_parquet(sources, options, {}, empty, stream, mr),
                    std::invalid_argument);
  }

  SECTION("chunks staged for another projection")
  {
    // Staged for `y, z` alone; decoded with an `x < 3000` filter that needs `x` as well.
    cudf::numeric_scalar<std::int64_t> limit{3000};
    cudf::ast::literal value{limit};
    cudf::ast::column_name_reference column{"x"};
    cudf::ast::operation filter{cudf::ast::ast_operator::LESS, column, value};
    auto staging_options = cudf::io::parquet_reader_options::builder().build();
    staging_options.set_column_names({"y", "z"});
    auto decode_options = cudf::io::parquet_reader_options::builder(
                            cudf::io::source_info{std::vector<std::string>{files.first.string()}})
                            .filter(filter)
                            .build();
    decode_options.set_column_names({"y", "z"});

    auto const other = stage(sources.front(), files.first, staging_options, stream);
    std::vector<std::vector<byte_range>> const other_ranges{other.ranges};
    std::vector<prefetched_chunk_spans> prefetched;
    prefetched.emplace_back(other.chunks.spans);
    REQUIRE(scan::can_materialize_from_prefetched(sources, other_ranges, prefetched));
    CHECK_THROWS_AS(
      scan::materialize_parquet(sources, decode_options, other_ranges, prefetched, stream, mr),
      scan::prefetched_mismatch);
  }
}

namespace {

/// A datasource with no io context or object: never read, only present.
std::shared_ptr<sirius::io::sirius_datasource> placeholder_datasource()
{
  return std::make_shared<sirius::io::sirius_datasource>(nullptr, nullptr);
}

/// Slices a, b, c: a and c have a datasource, row groups and ranges; b (when @p b_empty) has no
/// row groups and no ranges, as an all-pruned file in a bundle would.
std::unique_ptr<scan::parquet_split_info> three_slice_split(bool b_empty)
{
  auto split =
    std::make_unique<scan::parquet_split_info>(std::vector<scan::scan_info::fadvise_entry>{});
  split->rg_slices.emplace_back(
    nullptr, "a", std::vector<cudf::size_type>{0}, 0, 0, 0, placeholder_datasource(), 0);
  split->rg_slices.emplace_back(
    nullptr,
    "b",
    b_empty ? std::vector<cudf::size_type>{} : std::vector<cudf::size_type>{0},
    0,
    0,
    0,
    placeholder_datasource(),
    1);
  split->rg_slices.emplace_back(
    nullptr, "c", std::vector<cudf::size_type>{0, 1}, 0, 0, 0, placeholder_datasource(), 2);
  split->set_column_chunk_byte_ranges(
    {{byte_range(0, 16), byte_range(16, 32)},
     b_empty ? std::vector<byte_range>{} : std::vector<byte_range>{byte_range(4, 4)},
     {byte_range(100, 8)}});
  return split;
}

prefetched_ranges::device_ranges fake_entry(std::vector<std::size_t> const& sizes)
{
  prefetched_ranges::device_ranges entry;
  for (auto const size : sizes) {
    entry.buffers.emplace_back(size, cudf::get_default_stream());
  }
  for (auto const& buffer : entry.buffers) {
    entry.spans.emplace_back(static_cast<std::uint8_t const*>(buffer.data()), buffer.size());
  }
  return entry;
}

}  // namespace

TEST_CASE("prefetched_spans_per_slice mirrors the split's fadvise entries",
          "[scan][parquet][device_prefetch]")
{
  auto const split = three_slice_split(/*b_empty=*/true);
  std::vector<prefetched_ranges::device_ranges> entries;
  entries.push_back(fake_entry({16, 32}));
  entries.push_back(fake_entry({8}));

  auto const mapped = split->prefetched_spans_per_slice(entries);
  REQUIRE(mapped.has_value());
  REQUIRE(mapped->size() == 3);
  REQUIRE((*mapped)[0].size() == 2);
  CHECK((*mapped)[0][0].data() == entries[0].spans[0].data());
  CHECK((*mapped)[0][1].data() == entries[0].spans[1].data());
  CHECK((*mapped)[1].empty());
  REQUIRE((*mapped)[2].size() == 1);
  CHECK((*mapped)[2][0].data() == entries[1].spans[0].data());

  SECTION("a span of the wrong size")
  {
    std::vector<prefetched_ranges::device_ranges> wrong;
    wrong.push_back(fake_entry({16, 31}));
    wrong.push_back(fake_entry({8}));
    CHECK_FALSE(split->prefetched_spans_per_slice(wrong).has_value());
  }
  SECTION("a chunk missing from an entry")
  {
    std::vector<prefetched_ranges::device_ranges> wrong;
    wrong.push_back(fake_entry({16}));
    wrong.push_back(fake_entry({8}));
    CHECK_FALSE(split->prefetched_spans_per_slice(wrong).has_value());
  }
  SECTION("an extra or a missing entry")
  {
    auto extra = std::vector<prefetched_ranges::device_ranges>{};
    extra.push_back(fake_entry({16, 32}));
    extra.push_back(fake_entry({8}));
    extra.push_back(fake_entry({8}));
    CHECK_FALSE(split->prefetched_spans_per_slice(extra).has_value());

    auto missing = std::vector<prefetched_ranges::device_ranges>{};
    missing.push_back(fake_entry({16, 32}));
    CHECK_FALSE(split->prefetched_spans_per_slice(missing).has_value());
  }
}

TEST_CASE("supports_device_prefetch requires every slice to be decodable from device spans",
          "[scan][parquet][device_prefetch]")
{
  CHECK(three_slice_split(/*b_empty=*/false)->supports_device_prefetch());
  CHECK_FALSE(three_slice_split(/*b_empty=*/true)->supports_device_prefetch());

  auto split = three_slice_split(/*b_empty=*/false);
  SECTION("the ingestible opted out")
  {
    split->device_prefetch_allowed = false;
    CHECK_FALSE(split->supports_device_prefetch());
  }
  SECTION("no slices")
  {
    split->rg_slices.clear();
    split->set_column_chunk_byte_ranges({});
    CHECK_FALSE(split->supports_device_prefetch());
  }
  SECTION("a slice without a datasource")
  {
    split->rg_slices[2].datasource.reset();
    CHECK_FALSE(split->supports_device_prefetch());
  }
  SECTION("a slice without ranges")
  {
    split->set_column_chunk_byte_ranges(
      {{byte_range(0, 16), byte_range(16, 32)}, {byte_range(4, 4)}, {}});
    CHECK_FALSE(split->supports_device_prefetch());
  }
}

namespace {

/// A local-file backend (kvikIO underneath) that counts every read the engine asks of it, so a test
/// can tell whether a decode read the file or consumed device copies that were already there.
/// Only reads reach the counters: opening a file and sizing it do not.
class counting_ioctx final : public sirius::io::ioctx {
 public:
  counting_ioctx() = default;
  ~counting_ioctx() override { pre_destroy(); }

  /// Backend reads (vectored/device/async and synchronous host reads) since construction.
  [[nodiscard]] std::size_t reads() const noexcept { return _reads.load(); }
  /// Bytes those reads asked for.
  [[nodiscard]] std::size_t bytes() const noexcept { return _bytes.load(); }

  [[nodiscard]] sirius::io::io_context_type type() const noexcept override
  {
    return _inner->type();
  }
  void shutdown() noexcept override {}
  [[nodiscard]] bool supports(std::string_view path) const noexcept override
  {
    return _inner->supports(path);
  }
  [[nodiscard]] bool supports_device_read() const noexcept override { return true; }
  [[nodiscard]] bool supports_host_to_device_read() const noexcept override { return false; }
  [[nodiscard]] bool supports_vector_host_read() const noexcept override { return false; }
  [[nodiscard]] bool supports_device_range_read() const noexcept override { return false; }

  [[nodiscard]] std::vector<cudf::io::text::byte_range_info> align_and_coalesce(
    std::span<cudf::io::text::byte_range_info const> ranges,
    std::optional<std::size_t> alignment) const noexcept override
  {
    return _inner->align_and_coalesce(ranges, alignment);
  }

  std::size_t host_read_io(sirius::io::io_object const& obj,
                           std::size_t offset,
                           std::size_t size,
                           std::uint8_t* dst) override
  {
    _reads.fetch_add(1);
    _bytes.fetch_add(size);
    return _inner->host_read_io(obj, offset, size, dst);
  }

  sirius::exec::semi_future<std::size_t> mixed_readv_async_io(
    sirius::io::io_object const& obj,
    std::vector<sirius::io::prepared_io_slice>&& slices) noexcept override
  {
    _reads.fetch_add(1);
    std::size_t requested = 0;
    for (auto const& slice : slices) {
      requested += slice.size();
    }
    _bytes.fetch_add(requested);
    return _inner->host_device_readv_async_io(obj, std::move(slices));
  }

 protected:
  std::shared_ptr<sirius::io::io_object> create_io_object(std::string path) override
  {
    // Borrow the object the real backend opens, keeping its datasource alive alongside.
    std::shared_ptr<sirius::io::sirius_datasource> opened =
      _inner->open_datasource(std::move(path));
    return std::shared_ptr<sirius::io::io_object>(
      opened, const_cast<sirius::io::io_object*>(&opened->get_io_object()));
  }

 private:
  std::shared_ptr<sirius::io::kvikio_context> _inner =
    std::make_shared<sirius::io::kvikio_context>();
  std::atomic<std::size_t> _reads{0};
  std::atomic<std::size_t> _bytes{0};
};

}  // namespace

namespace {

/// What the hooks saw, shared with them so they outlive the prefetch.
struct hook_counts {
  std::atomic<int> claims{0};
  std::array<std::atomic<int>, 4> releases{};

  [[nodiscard]] int of(release_reason reason) const
  {
    return releases[static_cast<std::size_t>(reason)].load();
  }
};

/// One Parquet split built by the real split provider and coalescer, and the ingestible that
/// decodes it (the demand path), so a landed prefetch is consumed by the real consumer.
struct consumer_fixture : sirius::test::GpuExecutionFixture {
  sirius::test::scratch_dir scratch{"parquet_device_prefetch_consumer"};
  fs::path file;
  std::shared_ptr<scan::parquet_gpu_ingestible> ingestible;
  std::unique_ptr<scan::scan_info> owned_split;
  scan::parquet_split_info* split                = nullptr;
  cucascade::memory::memory_space* space         = nullptr;
  std::shared_ptr<scan::device_stream_pool> pool = scan::device_stream_pool::create(0, 1);
  std::shared_ptr<hook_counts> counts            = std::make_shared<hook_counts>();
  std::shared_ptr<counting_ioctx> backend        = std::make_shared<counting_ioctx>();
  rmm::cuda_stream stream;

  consumer_fixture()
  {
    {
      sirius::test::scoped_setting cpu(*con, "gpu_execution", false);
      file = write_rows(*con, scratch.path(), "device_prefetch_consumer", 0);
    }
    auto info                    = std::make_unique<scan::parquet_ingestible_table_info>();
    info->contract_id            = 53;
    info->resolved_file_paths    = {file.string()};
    info->names                  = {"x", "y", "z"};
    info->returned_types         = {sirius::logical_type::make(sirius::type_id::BIGINT),
                                    sirius::logical_type::make(sirius::type_id::INTEGER),
                                    sirius::logical_type::make(sirius::type_id::VARCHAR)};
    info->column_ids             = {duckdb::ColumnIndex(0), duckdb::ColumnIndex(2)};
    info->projected_output_count = 2;
    info->approximate_batch_size = std::size_t{1} << 30;
    ingestible                   = scan::make_ingestible(std::move(info));

    auto prepare = ingestible->next_split_provider(
      [ioctx = backend](std::string_view) -> std::shared_ptr<sirius::io::ioctx> { return ioctx; });
    auto coalescer = ingestible->create_batch_coalescer();
    auto batches   = coalescer->push(prepare({}));
    for (auto& batch : coalescer->flush()) {
      batches.push_back(std::move(batch));
    }
    REQUIRE(batches.size() == 1);
    owned_split = std::move(batches.front());
    split       = dynamic_cast<scan::parquet_split_info*>(owned_split.get());
    REQUIRE(split != nullptr);
    REQUIRE(split->supports_device_prefetch());

    auto state = sirius::test::get_registered_sirius_context(*con);
    space      = state->get_memory_manager().get_memory_space(cucascade::memory::Tier::GPU, 0);
    REQUIRE(space != nullptr);
  }

  scan::filtered_table materialize()
  {
    return ingestible->materialize_metadata_to_table(*split, *space, stream, false, nullptr);
  }

  /// The split's hint ranges copied to the device on a pool stream, one entry per fadvise hint.
  std::shared_ptr<prefetched_ranges> landed(int device_id, bool short_first_span = false)
  {
    auto lease = pool->acquire();
    std::vector<prefetched_ranges::device_ranges> entries;
    std::size_t total = 0;
    auto const ranges = split->column_chunk_byte_ranges();
    for (std::size_t i = 0; i < split->rg_slices.size(); ++i) {
      if (!split->rg_slices[i].datasource || ranges[i].empty()) { continue; }
      auto staged = stage_ranges(split->rg_slices[i].file_path, ranges[i], lease.view);
      for (auto const& range : ranges[i]) {
        total += static_cast<std::size_t>(range.size());
      }
      entries.push_back({std::move(staged.buffers), std::move(staged.spans)});
    }
    REQUIRE(entries.size() == split->fadvise_hints().size());
    if (short_first_span) {
      auto& first = entries.front().spans.front();
      first       = cudf::device_span<std::uint8_t const>(first.data(), first.size() - 1);
    }
    return std::make_shared<prefetched_ranges>(
      device_id,
      std::move(lease),
      std::move(entries),
      total,
      total,
      [c = counts]() noexcept { c->claims.fetch_add(1); },
      [c = counts](prefetched_ranges::release_event event) noexcept {
        c->releases[static_cast<std::size_t>(event.reason)].fetch_add(1);
      });
  }

  void install(std::shared_ptr<prefetched_ranges> prefetch)
  {
    REQUIRE(split->set_prefetched_ranges(std::move(prefetch)));
  }
};

}  // namespace

TEST_CASE_METHOD(consumer_fixture,
                 "the parquet consumer decodes a landed prefetch without reading the file again",
                 "[integration][scan][parquet][device_prefetch]")
{
  // The demand path reads the column chunks from the backend: the counter sees it.
  auto const reads_before_demand = backend->reads();
  auto const baseline            = materialize();
  REQUIRE(baseline.table.num_rows() == k_rows);
  auto const demand_reads = backend->reads() - reads_before_demand;
  auto const demand_bytes = backend->bytes();
  REQUIRE(demand_reads > 0);

  auto prefetch = landed(0);
  REQUIRE(prefetch->set_ready());
  std::weak_ptr<prefetched_ranges> const watch = prefetch;
  install(std::move(prefetch));

  // Consuming the landed copy issues no read at all -- in particular no demand read of the payload
  // the prefetch already carries.
  auto const reads_before_consume = backend->reads();
  auto const decoded              = materialize();
  CHECK(backend->reads() == reads_before_consume);
  CHECK(backend->bytes() == demand_bytes);
  CHECK(counts->of(release_reason::consumed) == 1);
  CHECK(counts->claims.load() == 1);
  CHECK(watch.expired());
  CHECK(split->prefetched() == nullptr);
  CHECK(decoded.table.num_rows() == k_rows);
  CHECK(tables_equal(decoded.table.view(), baseline.table.view(), stream));
}

TEST_CASE_METHOD(consumer_fixture,
                 "a prefetch that cannot be decoded falls back to the demand read",
                 "[integration][scan][parquet][device_prefetch]")
{
  auto const baseline = materialize();

  SECTION("failed and abandoned prefetches are read for the demand path")
  {
    auto failed = landed(0);
    failed->set_failed(std::make_exception_ptr(std::runtime_error("injected read failure")));
    install(std::move(failed));
    auto const reads_before_failure = backend->reads();
    auto const after_failure        = materialize();
    CHECK(backend->reads() > reads_before_failure);  // the demand path really read the file
    CHECK(tables_equal(after_failure.table.view(), baseline.table.view(), stream));

    auto abandoned = landed(0);
    abandoned->abandon();
    install(std::move(abandoned));
    auto const after_abandon = materialize();
    CHECK(tables_equal(after_abandon.table.view(), baseline.table.view(), stream));
    CHECK(counts->of(release_reason::failed) == 2);
    CHECK(counts->of(release_reason::consumed) == 0);
  }

  SECTION("a prefetch on another GPU is discarded without waiting")
  {
    // An unusable device id: the prefetch is flagged wrong-device before any wait (it is still
    // pending, so a wait would take the full bound) and never touches a buffer.
    install(std::make_shared<prefetched_ranges>(
      63,
      pool->acquire(),
      std::vector<prefetched_ranges::device_ranges>{},
      0,
      0,
      [c = counts]() noexcept { c->claims.fetch_add(1); },
      [c = counts](prefetched_ranges::release_event event) noexcept {
        c->releases[static_cast<std::size_t>(event.reason)].fetch_add(1);
      }));
    auto const decoded = materialize();
    CHECK(counts->of(release_reason::wrong_device) == 1);
    CHECK(tables_equal(decoded.table.view(), baseline.table.view(), stream));
  }

  SECTION("spans that do not line up with the split are released unconsumed")
  {
    auto prefetch = landed(0, /*short_first_span=*/true);
    REQUIRE(prefetch->set_ready());
    install(std::move(prefetch));
    auto const decoded = materialize();
    CHECK(counts->of(release_reason::unconsumed) == 1);
    CHECK(counts->of(release_reason::consumed) == 0);
    CHECK(tables_equal(decoded.table.view(), baseline.table.view(), stream));
  }

  SECTION("chunks the reader does not expect are taken, rejected, and read through the demand path")
  {
    // The split's own ranges and the landed spans agree with each other, so the mapping and the
    // shape check pass, but not with the reader's enumeration: only the exact check catches it.
    std::vector<std::vector<byte_range>> ranges(split->column_chunk_byte_ranges().begin(),
                                                split->column_chunk_byte_ranges().end());
    auto const first       = ranges.front().front();
    ranges.front().front() = byte_range(first.offset(), first.size() - 1);
    split->set_column_chunk_byte_ranges(std::move(ranges));
    auto prefetch = landed(0);
    REQUIRE(prefetch->set_ready());
    install(std::move(prefetch));

    auto const reads_before_mismatch = backend->reads();
    auto const decoded               = materialize();
    CHECK(counts->of(release_reason::consumed) == 1);
    CHECK(backend->reads() > reads_before_mismatch);
    CHECK(tables_equal(decoded.table.view(), baseline.table.view(), stream));
  }

  SECTION("a split that opted out of the device route")
  {
    split->device_prefetch_allowed = false;
    auto prefetch                  = landed(0);
    REQUIRE(prefetch->set_ready());
    install(std::move(prefetch));
    auto const decoded = materialize();
    CHECK(counts->of(release_reason::unconsumed) == 1);
    CHECK(decoded.table.num_rows() == k_rows);
  }
}
