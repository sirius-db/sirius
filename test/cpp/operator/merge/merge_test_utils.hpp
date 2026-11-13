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

#pragma once

#include "catch.hpp"
#include "data/convertible_data_batch.hpp"
#include "data/data_batch_utils.hpp"
#include "data/sirius_converter_registry.hpp"
#include "data/sirius_row_range_reader_registry.hpp"
#include "memory/sirius_memory_reservation_manager.hpp"
#include "op/merge/bounded_kway_merge.hpp"
#include "op/merge/windowed_merge.hpp"
#include "operator/operator_test_utils.hpp"
#include "utils/test_validation_utility.hpp"

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/copying.hpp>
#include <cudf/filling.hpp>
#include <cudf/merge.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/sorting.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <rmm/cuda_stream.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/resource_ref.hpp>

#include <cuda_runtime.h>

#include <api/compressed_table_io.hpp>
#include <compression/compressed_disk_representation.hpp>
#include <compression/compressed_payload_sources.hpp>
#include <compression/compressed_representation.hpp>
#include <compression/device_compressed_blob.hpp>
#include <compression/spill_compression.hpp>
#include <cucascade/cudf/host_data_representation.hpp>
#include <cucascade/data/data_batch.hpp>
#include <cucascade/data/data_repository.hpp>
#include <cucascade/data/disk_data_representation.hpp>
#include <cucascade/memory/memory_space.hpp>
#include <cucascade/memory/reservation_manager_configurator.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

namespace sirius::test::merge_utils {

// ---------------------------------------------------------------------------
// Reference bounded merge (the oracle)
// ---------------------------------------------------------------------------

/// @p table with the exact null count of every column: the windows of a windowed merge report an
/// upper bound, which libcudf algorithms do not accept.
inline cudf::table_view with_exact_null_counts(const cudf::table_view& table,
                                               rmm::cuda_stream_view stream)
{
  std::vector<cudf::column_view> columns;
  columns.reserve(static_cast<std::size_t>(table.num_columns()));
  for (const auto& col : table) {
    const auto nulls =
      col.nullable() && col.size() > 0
        ? cudf::null_count(col.null_mask(), col.offset(), col.offset() + col.size(), stream)
        : 0;
    columns.emplace_back(col.type(),
                         col.size(),
                         col.head(),
                         nulls > 0 ? col.null_mask() : nullptr,
                         nulls,
                         col.offset(),
                         std::vector<cudf::column_view>(col.child_begin(), col.child_end()));
  }
  return cudf::table_view{columns};
}

/**
 * @brief Reference implementation of op::merge::bounded_kway_merge, usable as a bounded_merge_fn.
 *
 * Merges ALL window rows with cudf::merge (ties broken by window index, then row, through two
 * extra key columns), then applies the safe-target rule on the host and slices the first
 * `target` rows. Slow and memory-hungry on purpose: it is the oracle the real kernel and the
 * windowed merger are checked against.
 */
inline op::merge::bounded_merge_result reference_bounded_merge(
  std::span<const cudf::table_view> windows,
  const op::merge::merge_key_spec& keys,
  std::span<const std::uint8_t> truncated,
  cudf::size_type max_rows,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  const auto k = windows.size();
  if (truncated.size() != k) {
    throw std::invalid_argument("reference_bounded_merge: one truncated flag per window");
  }
  op::merge::bounded_merge_result result;
  result.consumed.assign(k, 0);
  if (k == 0) {
    result.output = std::make_unique<cudf::table>();
    return result;
  }
  const auto num_cols = windows.front().num_columns();

  // Tag every row with (window, row) so the merged order is the stable one.
  std::vector<std::unique_ptr<cudf::column>> tags;
  std::vector<cudf::table_view> tagged;
  std::int64_t total_rows = 0;
  bool empty_truncated    = false;
  for (std::size_t i = 0; i < k; ++i) {
    const auto rows = windows[i].num_rows();
    if (rows == 0) {
      empty_truncated = empty_truncated || truncated[i] != 0;
      continue;
    }
    total_rows += rows;
    cudf::numeric_scalar<std::int32_t> window_idx(static_cast<std::int32_t>(i), true, stream);
    cudf::numeric_scalar<std::int32_t> zero(0, true, stream);
    tags.push_back(cudf::make_column_from_scalar(window_idx, rows, stream));
    tags.push_back(cudf::sequence(rows, zero, stream));
    const auto window = with_exact_null_counts(windows[i], stream);
    std::vector<cudf::column_view> cols(window.begin(), window.end());
    cols.push_back(tags[tags.size() - 2]->view());
    cols.push_back(tags[tags.size() - 1]->view());
    tagged.emplace_back(cols);
  }

  auto key_cols        = keys.key_cols;
  auto column_order    = keys.column_order;
  auto null_precedence = keys.null_precedence;
  if (null_precedence.empty()) {
    null_precedence.assign(key_cols.size(), cudf::null_order::BEFORE);
  }
  for (cudf::size_type tag = 0; tag < 2; ++tag) {
    key_cols.push_back(num_cols + tag);
    column_order.push_back(cudf::order::ASCENDING);
    null_precedence.push_back(cudf::null_order::BEFORE);
  }

  std::unique_ptr<cudf::table> merged;
  std::vector<std::int32_t> source;  // window of each merged row
  if (!tagged.empty()) {
    merged = cudf::merge(tagged, key_cols, column_order, null_precedence, stream);
    stream.synchronize();  // the host copy below runs on the default stream
    source = operator_utils::copy_column_to_host<std::int32_t>(merged->view().column(num_cols));
  }

  // target = min(max_rows, r_safe, Σ len); r_safe = min over truncated windows of the merged
  // position of the window's last row, plus one.
  auto target = std::min<std::int64_t>(max_rows, total_rows);
  if (empty_truncated) { target = 0; }
  std::vector<std::int64_t> last_position(k, -1);
  for (std::size_t pos = 0; pos < source.size(); ++pos) {
    last_position[static_cast<std::size_t>(source[pos])] = static_cast<std::int64_t>(pos);
  }
  for (std::size_t i = 0; i < k; ++i) {
    if (truncated[i] != 0 && last_position[i] >= 0) {
      target = std::min(target, last_position[i] + 1);
    }
  }

  for (std::int64_t pos = 0; pos < target; ++pos) {
    ++result.consumed[static_cast<std::size_t>(source[static_cast<std::size_t>(pos)])];
  }

  std::vector<cudf::size_type> payload(static_cast<std::size_t>(num_cols));
  for (cudf::size_type c = 0; c < num_cols; ++c) {
    payload[static_cast<std::size_t>(c)] = c;
  }
  if (merged) {
    const auto head =
      cudf::slice(merged->view().select(payload), {0, static_cast<cudf::size_type>(target)}, stream)
        .front();
    result.output = std::make_unique<cudf::table>(head, stream, mr);
  } else {
    result.output = cudf::empty_like(windows.front());
  }
  return result;
}

// ---------------------------------------------------------------------------
// Column and table builders
// ---------------------------------------------------------------------------

/// A fixed-width column from host values; @p valids[i] == false makes row i NULL. With @p valids
/// the column has a null mask even when no row is NULL. The mask is built on the host and
/// uploaded at once, so this is cheap for millions of rows.
template <typename T>
inline std::unique_ptr<cudf::column> make_column(const std::vector<T>& values,
                                                 cudf::type_id type_id,
                                                 const std::vector<bool>& valids = {})
{
  if (!valids.empty() && valids.size() != values.size()) {
    throw std::invalid_argument("make_column: one validity flag per value");
  }
  auto stream     = cudf::get_default_stream();
  const auto size = static_cast<cudf::size_type>(values.size());
  auto col        = cudf::make_fixed_width_column(
    cudf::data_type{type_id}, size, cudf::mask_state::UNALLOCATED, stream);
  if (size > 0) {
    cudaMemcpyAsync(col->mutable_view().data<T>(),
                    values.data(),
                    sizeof(T) * values.size(),
                    cudaMemcpyHostToDevice,
                    stream.get());
  }
  if (!valids.empty()) {
    std::vector<cudf::bitmask_type> words(static_cast<std::size_t>(cudf::num_bitmask_words(size)),
                                          0);
    constexpr std::size_t word_bits = sizeof(cudf::bitmask_type) * 8;
    cudf::size_type nulls           = 0;
    for (std::size_t i = 0; i < valids.size(); ++i) {
      if (valids[i]) {
        words[i / word_bits] |= cudf::bitmask_type{1} << (i % word_bits);
      } else {
        ++nulls;
      }
    }
    auto mask = cudf::create_null_mask(size, cudf::mask_state::UNINITIALIZED, stream);
    if (!words.empty()) {
      cudaMemcpyAsync(mask.data(),
                      words.data(),
                      words.size() * sizeof(cudf::bitmask_type),
                      cudaMemcpyHostToDevice,
                      stream.get());
    }
    col->set_null_mask(std::move(mask), nulls);
  }
  // The host vectors may go away as soon as this returns.
  stream.sync();
  return col;
}

/// A STRING column from host values (no nulls).
inline std::unique_ptr<cudf::column> make_string_column(const std::vector<std::string>& values)
{
  return operator_utils::make_string_column(
    values, cudf::get_default_stream(), cudf::get_current_device_resource_ref());
}

template <typename... Columns>
inline std::unique_ptr<cudf::table> make_table(Columns&&... columns)
{
  std::vector<std::unique_ptr<cudf::column>> cols;
  (cols.push_back(std::forward<Columns>(columns)), ...);
  return std::make_unique<cudf::table>(std::move(cols));
}

/// @p table stably sorted by @p keys.
inline std::unique_ptr<cudf::table> stable_sort(const cudf::table_view& table,
                                                const op::merge::merge_key_spec& keys)
{
  return cudf::stable_sort_by_key(
    table, table.select(keys.key_cols), keys.column_order, keys.null_precedence);
}

/// The oracle of a whole merge: the runs concatenated in run order, then stably sorted, which is
/// the order a k-way merge with ties broken by run index produces.
inline std::unique_ptr<cudf::table> expected_full_sort(std::span<const cudf::table_view> runs,
                                                       const op::merge::merge_key_spec& keys)
{
  std::vector<cudf::table_view> views(runs.begin(), runs.end());
  auto all = cudf::concatenate(views);
  return stable_sort(all->view(), keys);
}

/// True when both tables have the same types, row count, row validity and values. Valid values
/// are compared by their bits (NaN equals NaN, -0 differs from +0); NULL equals NULL.
inline bool tables_equal(const cudf::table_view& lhs, const cudf::table_view& rhs)
{
  return ::sirius::test::expect_tables_equivalent_impl(lhs, rhs);
}

// ---------------------------------------------------------------------------
// Runs as repository batches
// ---------------------------------------------------------------------------

/// One sorted run: its batches (which stay in the repository) and its merge_run description.
struct test_run {
  std::vector<std::shared_ptr<cucascade::data_batch>> batches;
  op::merge::merge_run run;
  std::unique_ptr<cudf::table> rows;  ///< all rows of the run, in run order (for the oracle)
};

/**
 * @brief Split an already sorted table into GPU batches of @p batch_rows rows and add them to
 * @p repository.
 *
 * @param batch_rows Rows of each batch, in run order; must sum to sorted.num_rows(). A zero entry
 *        makes an empty batch.
 */
inline test_run make_gpu_run(cucascade::shared_data_repository& repository,
                             cucascade::memory::memory_space& space,
                             const cudf::table_view& sorted,
                             const std::vector<std::size_t>& batch_rows)
{
  auto stream = operator_utils::default_stream();
  auto mr     = operator_utils::get_resource_ref(space);
  test_run out;
  out.rows          = std::make_unique<cudf::table>(sorted);
  std::size_t begin = 0;
  for (const auto rows : batch_rows) {
    const auto piece =
      cudf::slice(sorted,
                  {static_cast<cudf::size_type>(begin), static_cast<cudf::size_type>(begin + rows)})
        .front();
    auto batch = ::sirius::make_data_batch(std::make_unique<cudf::table>(piece, stream, mr),
                                           space,
                                           stream,
                                           ::sirius::telemetry::batch_telemetry_info{});
    repository.add_data_batch(batch);
    out.run.batch_ids.push_back(batch->get_batch_id());
    out.run.batch_rows.push_back(rows);
    out.batches.push_back(std::move(batch));
    begin += rows;
  }
  if (begin != static_cast<std::size_t>(sorted.num_rows())) {
    throw std::invalid_argument("make_gpu_run: batch_rows must sum to the run's rows");
  }
  return out;
}

/// Sort @p unsorted by @p keys and make it a GPU run of @p num_batches near-equal batches.
inline test_run make_sorted_gpu_run(cucascade::shared_data_repository& repository,
                                    cucascade::memory::memory_space& space,
                                    const cudf::table_view& unsorted,
                                    const op::merge::merge_key_spec& keys,
                                    std::size_t num_batches = 1)
{
  auto sorted      = stable_sort(unsorted, keys);
  const auto total = static_cast<std::size_t>(sorted->num_rows());
  std::vector<std::size_t> batch_rows(num_batches, total / num_batches);
  batch_rows.back() += total % num_batches;
  return make_gpu_run(repository, space, sorted->view(), batch_rows);
}

/// One cursor per run, at the start of the run.
inline std::vector<op::merge::merge_input_cursor> make_cursors(const std::vector<test_run>& runs)
{
  std::vector<op::merge::merge_input_cursor> cursors;
  cursors.reserve(runs.size());
  for (const auto& run : runs) {
    cursors.push_back(op::merge::merge_input_cursor{.run = run.run});
  }
  return cursors;
}

/// The resolved batches of each run, as windowed_merger takes them.
inline std::vector<std::vector<std::shared_ptr<cucascade::data_batch>>> run_batches(
  const std::vector<test_run>& runs)
{
  std::vector<std::vector<std::shared_ptr<cucascade::data_batch>>> batches;
  batches.reserve(runs.size());
  for (const auto& run : runs) {
    batches.push_back(run.batches);
  }
  return batches;
}

/// The oracle for @p runs: expected_full_sort of their rows.
inline std::unique_ptr<cudf::table> expected_full_sort(const std::vector<test_run>& runs,
                                                       const op::merge::merge_key_spec& keys)
{
  std::vector<cudf::table_view> views;
  views.reserve(runs.size());
  for (const auto& run : runs) {
    views.push_back(run.rows->view());
  }
  return expected_full_sort(std::span<const cudf::table_view>(views), keys);
}

inline bool all_exhausted(std::span<const op::merge::merge_input_cursor> cursors)
{
  return std::all_of(
    cursors.begin(), cursors.end(), [](const auto& cursor) { return cursor.exhausted(); });
}

/// What collect_full_sort gathered: the table of every run() call and their concatenation.
struct collected_sort {
  std::vector<std::unique_ptr<cudf::table>> chunks;
  std::unique_ptr<cudf::table> table;
};

/**
 * @brief Call merger.run() until every cursor is exhausted and concatenate the outputs.
 *
 * @param cursors The cursors the merger was built over.
 * @param max_calls Fails (std::runtime_error) if the merge is not finished after this many calls.
 */
inline collected_sort collect_full_sort(op::merge::windowed_merger& merger,
                                        std::span<const op::merge::merge_input_cursor> cursors,
                                        rmm::cuda_stream_view stream,
                                        rmm::device_async_resource_ref mr,
                                        op::merge::merge_round_stats* stats = nullptr,
                                        std::size_t max_calls               = 100000)
{
  collected_sort out;
  std::vector<cudf::table_view> views;
  do {
    if (out.chunks.size() == max_calls) {
      throw std::runtime_error("collect_full_sort: the merge did not finish");
    }
    out.chunks.push_back(merger.run(stream, mr, stats));
    views.push_back(out.chunks.back()->view());
  } while (!all_exhausted(cursors));
  out.table = cudf::concatenate(views, stream, mr);
  return out;
}

// ---------------------------------------------------------------------------
// Memory: a GPU space, a host space and a disk space; spill compression
// ---------------------------------------------------------------------------

/// Declare it before anything that holds batches of its spaces: the destructor shuts the manager
/// and the converter registry down.
struct tiers_fixture {
  std::filesystem::path tmp_dir;
  std::unique_ptr<sirius::memory::sirius_memory_reservation_manager> manager;
  cucascade::memory::memory_space* gpu  = nullptr;
  cucascade::memory::memory_space* host = nullptr;
  cucascade::memory::memory_space* disk = nullptr;
  /// The spill converters reject the default stream, so conversions and merges run on this one.
  rmm::cuda_stream stream;

  tiers_fixture()
  {
    sirius::converter_registry::reset_for_testing();
    std::string tmp_template =
      (std::filesystem::temp_directory_path() / "sirius_merge_tiers_XXXXXX").string();
    REQUIRE(mkdtemp(tmp_template.data()) != nullptr);
    tmp_dir = tmp_template;

    cucascade::memory::reservation_manager_configurator builder;
    builder.set_number_of_gpus(1)
      .set_gpu_usage_limit(512 * (std::size_t{1} << 20))
      .set_reservation_fraction_per_gpu(0.75)
      .set_per_numa_region_capacity(std::size_t{1} << 30)
      .use_gpu_id_as_host_id()
      .set_reservation_fraction_per_numa_region(0.75)
      .track_reservation_per_stream(false)
      .set_disk_mounting_point(0, std::size_t{1} << 30, tmp_dir.string());
    manager = std::make_unique<sirius::memory::sirius_memory_reservation_manager>(builder.build());
    sirius::converter_registry::initialize();
    if (!sirius::row_range_reader_registry::is_initialized()) {
      sirius::row_range_reader_registry::initialize();
    }
    gpu  = manager->get_memory_space(cucascade::memory::Tier::GPU, 0);
    host = manager->get_memory_space(cucascade::memory::Tier::HOST, 0);
    disk = manager->get_memory_space(cucascade::memory::Tier::DISK, 0);
    REQUIRE(gpu != nullptr);
    REQUIRE(host != nullptr);
    REQUIRE(disk != nullptr);
  }

  ~tiers_fixture()
  {
    manager->shutdown();
    sirius::converter_registry::shutdown();
    std::error_code ignored;
    std::filesystem::remove_all(tmp_dir, ignored);
  }

  tiers_fixture(const tiers_fixture&)            = delete;
  tiers_fixture& operator=(const tiers_fixture&) = delete;

  [[nodiscard]] ::cuda::stream_ref stream_ref() const { return ::cuda::stream_ref{stream.value()}; }
  [[nodiscard]] rmm::cuda_stream_view stream_view() const
  {
    return rmm::cuda_stream_view{stream.value()};
  }
  [[nodiscard]] rmm::device_async_resource_ref mr() const
  {
    return operator_utils::get_resource_ref(*gpu);
  }
};

/// Switches spill compression on for one test and off afterwards. Every batch is a candidate
/// (`spill_min_batch_bytes` lowered to 0).
struct scoped_spill_compression {
  /// @param max_compressed_fraction Largest compressed size, as a fraction of the batch, that
  ///        is kept; the default keeps every encode, also of columns that do not compress.
  explicit scoped_spill_compression(double max_compressed_fraction = 2.0)
  {
    sirius::compression::spill_compression_settings settings;
    settings.enabled                 = true;
    settings.min_batch_bytes         = 0;
    settings.max_compressed_fraction = max_compressed_fraction;
    sirius::compression::set_spill_compression_settings(settings);
  }
  ~scoped_spill_compression() { sirius::compression::set_spill_compression_settings({}); }
};

// ---------------------------------------------------------------------------
// Batches in other tiers and representations
// ---------------------------------------------------------------------------

/// True when a CUDA device is visible.
inline bool has_gpu()
{
  int count = 0;
  return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

/// True when @p batch holds a @p Representation right now.
template <typename Representation>
inline bool holds(cucascade::data_batch& batch)
{
  auto ro = batch.to_read_only();
  return dynamic_cast<Representation const*>(ro.get_data()) != nullptr;
}

/// What a batch built on the GPU is turned into: left as is, moved uncompressed, or spilled
/// compressed (and, for compressed_device, rebuilt on the device).
enum class placement { gpu, host, disk, compressed_device, compressed_host, compressed_disk };

inline bool is_compressed(placement where)
{
  return where == placement::compressed_device || where == placement::compressed_host ||
         where == placement::compressed_disk;
}

/**
 * @brief Spills the GPU batch @p batch compressed, as a downgrade does, and leaves it in the
 * compressed placement @p where.
 *
 * Spill compression must be on (scoped_spill_compression), or the spill is not compressed and the
 * test fails.
 */
inline void compress_batch(tiers_fixture& fx,
                           const std::shared_ptr<cucascade::data_batch>& batch,
                           placement where)
{
  REQUIRE(is_compressed(where));
  // The batch may have been built on the default stream; the converters run on the fixture's.
  cudf::get_default_stream().sync();
  sirius::convertible_data_batch convertible(batch);
  REQUIRE(convertible.convert({fx.host}, fx.stream_ref(), *fx.manager, true).has_value());
  fx.stream.synchronize();
  REQUIRE(holds<sirius::compressed_host_representation>(*batch));
  if (where == placement::compressed_host) { return; }

  if (where == placement::compressed_disk) {
    REQUIRE(convertible.convert({fx.disk}, fx.stream_ref(), *fx.manager, true).has_value());
    fx.stream.synchronize();
    REQUIRE(holds<sirius::compressed_disk_representation>(*batch));
    return;
  }

  // The device tier: the compressed table rebuilt on the device from the pinned payload.
  auto mut           = batch->to_mutable();
  auto const& host   = mut.get_data()->cast<sirius::compressed_host_representation>();
  auto const& blocks = host.payload();
  auto blob          = std::make_shared<sirius::compressed_device_blob>();
  std::string error;
  simpatico::payload_fetch_fn fetch =
    [&blocks](std::uint64_t offset, std::size_t size, void* dst, ::cuda::stream_ref stream) {
      sirius::copy_pinned_blocks_to_device(blocks, offset, dst, size, stream);
    };
  blob->table = simpatico::read_compressed_table_from_memory(
    host.header(), fetch, fx.stream_ref(), fx.mr(), &error);
  REQUIRE(error.empty());
  fx.stream.synchronize();
  auto device = std::make_unique<sirius::compressed_device_representation>(
    *fx.gpu,
    std::move(blob),
    host.column_names(),
    host.get_size_in_bytes(),
    host.get_uncompressed_data_size_in_bytes(),
    host.num_rows());
  mut.set_data(std::move(device));
}

/// Turns the GPU batch @p batch into @p where. A compressed placement needs spill compression on,
/// as compress_batch does.
inline void place_batch(tiers_fixture& fx,
                        const std::shared_ptr<cucascade::data_batch>& batch,
                        placement where)
{
  if (where == placement::gpu) { return; }
  if (is_compressed(where)) {
    compress_batch(fx, batch, where);
    return;
  }
  cudf::get_default_stream().sync();
  auto mut = batch->to_mutable();
  mut.convert_to<cucascade::host_data_representation>(
    sirius::converter_registry::get(), fx.host, fx.stream_ref());
  fx.stream.synchronize();
  if (where == placement::disk) {
    mut.convert_to<cucascade::disk_data_representation>(
      sirius::converter_registry::get(), fx.disk, fx.stream_ref());
    fx.stream.synchronize();
  }
}

}  // namespace sirius::test::merge_utils
