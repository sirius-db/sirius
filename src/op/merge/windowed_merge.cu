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

#include "op/merge/detail/run_fallback.hpp"
#include "op/merge/detail/window_buffer.hpp"
#include "op/merge/windowed_merge.hpp"
#include "sirius/exception.hpp"

#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/copying.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <cucascade/cudf/gpu_data_representation.hpp>
#include <cucascade/memory/common.hpp>

#include <algorithm>
#include <limits>
#include <optional>
#include <stdexcept>
#include <typeindex>
#include <typeinfo>
#include <utility>

namespace sirius::op::merge {

namespace {

/// Attempts to obtain a window of one batch before giving up: a downgrade can win the race against
/// the whole-batch fallback, and then the batch is converted again.
constexpr int kMaxWindowAttempts = 3;

/// Device bytes of an owned column, including its null mask and children.
std::size_t column_bytes(const cudf::column_view& col, rmm::cuda_stream_view stream)
{
  std::size_t bytes = col.nullable() ? cudf::bitmask_allocation_size_bytes(col.size()) : 0;
  if (cudf::is_fixed_width(col.type())) {
    return bytes + static_cast<std::size_t>(col.size()) * cudf::size_of(col.type());
  }
  if (col.type().id() == cudf::type_id::STRING) {
    if (col.size() == 0) { return bytes; }
    return bytes + static_cast<std::size_t>(cudf::strings_column_view(col).chars_size(stream)) +
           (static_cast<std::size_t>(col.size()) + 1) * sizeof(cudf::size_type);
  }
  for (cudf::size_type i = 0; i < col.num_children(); ++i) {
    bytes += column_bytes(col.child(i), stream);
  }
  return bytes;
}

std::size_t table_bytes(const cudf::table_view& table, rmm::cuda_stream_view stream)
{
  std::size_t bytes = 0;
  for (const auto& col : table) {
    bytes += column_bytes(col, stream);
  }
  return bytes;
}

/// The window state of one run during a run() call.
struct run_window {
  merge_input_cursor cursor;

  // GPU tier: the read lock of the current batch, kept until the batch is finished or run() ends.
  std::optional<cucascade::read_only_data_batch> gpu_lock;
  cudf::table_view gpu_view;  // the fetched columns of the locked batch

  // Other tiers: the buffered rows are rows [cursor.row, cursor.row + buffer->size()) of the
  // current batch.
  std::unique_ptr<detail::window_buffer> buffer;
  std::unique_ptr<cucascade::row_range_reader_state> reader_state;
  std::optional<std::type_index> reader_type;

  cudf::table_view window;  // this round's window
};

/// The windows of one run() call. Releases every GPU read lock when it goes out of scope, also on
/// an exception.
struct round_windows {
  std::vector<run_window> runs;
  rmm::cuda_stream_view stream;

  round_windows(std::size_t num_runs, rmm::cuda_stream_view stream_)
    : runs(num_runs), stream(stream_)
  {
  }
  round_windows(const round_windows&)            = delete;
  round_windows& operator=(const round_windows&) = delete;

  ~round_windows()
  {
    for (auto& run : runs) {
      release_gpu(run);
    }
  }

  /// Drops the GPU read lock of @p run, ordering later writers after the work enqueued on stream.
  void release_gpu(run_window& run) noexcept
  {
    if (!run.gpu_lock) { return; }
    try {
      run.gpu_lock->record_reader_event(stream);
    } catch (...) {
      // record_reader_event synchronizes the stream before it throws, so no read is left untracked.
    }
    run.gpu_lock.reset();
    run.gpu_view = cudf::table_view{};
  }
};

}  // namespace

struct windowed_merger::impl {
  std::vector<std::vector<std::shared_ptr<cucascade::data_batch>>> runs;
  std::span<merge_input_cursor> cursors;
  std::vector<std::size_t> fetch_cols;
  std::vector<cudf::size_type> select_cols;  // fetch_cols as cudf indices
  std::vector<cudf::data_type> fetch_types;
  merge_key_spec keys;
  window_config cfg;
  const cucascade::row_range_reader_registry* readers;
  cucascade::memory::memory_space* gpu_space;
  bounded_merge_fn merge;
  bool has_strings{false};

  [[nodiscard]] bool gpu_resident(const cucascade::read_only_data_batch& lock) const
  {
    return lock.get_current_tier() == cucascade::memory::Tier::GPU &&
           lock.get_memory_space() == gpu_space;
  }

  /// Keeps @p lock for the rest of run() and windows the batch through a zero-copy slice.
  void use_gpu_batch(run_window& run, std::size_t run_idx, cucascade::read_only_data_batch&& lock)
  {
    const auto batch_rows = run.cursor.run.batch_rows[run.cursor.batch];
    auto view =
      lock.get_data()->cast<cucascade::gpu_table_representation>().get_table_view().select(
        select_cols);
    if (static_cast<std::size_t>(view.num_rows()) != batch_rows) {
      throw sirius::internal_exception(
        "windowed merge: batch {} of run {} has {} rows, its cursor expects {}",
        run.cursor.batch,
        run_idx,
        view.num_rows(),
        batch_rows);
    }
    run.gpu_view = view;
    run.gpu_lock.emplace(std::move(lock));
  }

  void slice_gpu_window(run_window& run, rmm::cuda_stream_view stream) const
  {
    const auto batch_rows = run.cursor.run.batch_rows[run.cursor.batch];
    const auto begin      = run.cursor.row;
    const auto end        = std::min(begin + cfg.window_rows, batch_rows);
    run.window =
      cudf::slice(run.gpu_view,
                  {static_cast<cudf::size_type>(begin), static_cast<cudf::size_type>(end)},
                  stream)
        .front();
  }

  /// Sets run.window to at most window_rows rows of the run's current batch, from the cursor on.
  void fill_window(run_window& run,
                   std::size_t run_idx,
                   rmm::cuda_stream_view stream,
                   rmm::device_async_resource_ref mr,
                   merge_round_stats& stats)
  {
    if (run.gpu_lock) {
      slice_gpu_window(run, stream);
      return;
    }
    const auto& batch     = runs[run_idx][run.cursor.batch];
    const auto batch_rows = run.cursor.run.batch_rows[run.cursor.batch];
    const auto wanted     = std::min(cfg.window_rows, batch_rows - run.cursor.row);
    if (run.buffer && run.buffer->size() >= wanted) {
      run.window = run.buffer->view();
      return;
    }

    for (int attempt = 0; attempt < kMaxWindowAttempts; ++attempt) {
      {
        // Per-refill read lock: whatever tier this lock observes is the tier the window reads.
        auto lock        = batch->to_read_only();
        const auto* data = lock.get_data();
        if (data == nullptr) {
          throw sirius::internal_exception(
            "windowed merge: batch {} of run {} has no data", run.cursor.batch, run_idx);
        }
        if (gpu_resident(lock)) {
          // The slice restarts at the cursor, so rows buffered while the batch was on another tier
          // are dropped.
          if (run.buffer && run.buffer->size() > 0) {
            run.buffer->consume(run.buffer->size(), stream);
          }
          run.reader_state.reset();
          run.reader_type.reset();
          use_gpu_batch(run, run_idx, std::move(lock));
          slice_gpu_window(run, stream);
          return;
        }
        if (readers->has_reader(*data)) {
          const std::type_index type{typeid(*data)};
          if (run.reader_type != type) {
            run.reader_state.reset();
            run.reader_type = type;
          }
          if (!run.buffer) {
            const auto est_string_bytes_per_row =
              has_strings ? (data->get_uncompressed_data_size_in_bytes() / batch_rows) + 1 : 0;
            run.buffer = std::make_unique<detail::window_buffer>(
              fetch_types, cfg.window_rows, est_string_bytes_per_row, stream, mr);
          }
          try {
            const auto read = run.buffer->refill(*readers,
                                                 lock,
                                                 fetch_cols,
                                                 run.cursor.row + run.buffer->size(),
                                                 &run.reader_state,
                                                 nullptr,
                                                 stream);
            stats.refill_rows += read.rows_written;
            for (const auto bytes : read.data_bytes_written) {
              stats.refill_bytes += bytes;
            }
            if (run.buffer->size() == 0) {
              throw sirius::internal_exception(
                "windowed merge: refill of run {} read no rows at row {} of batch {} ({} rows)",
                run_idx,
                run.cursor.row,
                run.cursor.batch,
                batch_rows);
            }
            run.window = run.buffer->view();
            return;
          } catch (const cucascade::row_range_unsupported_error&) {
            // Falls through to the whole-batch conversion below.
          }
        }
      }
      // No reader can serve this batch. The read lock is released: the conversion locks the batch
      // exclusively.
      if (detail::convert_run_batch_to_gpu(*batch, *gpu_space, stream)) {
        ++stats.fallback_conversions;
      }
    }
    throw sirius::internal_exception(
      "windowed merge: batch {} of run {} could not be read or kept on the GPU after {} attempts",
      run.cursor.batch,
      run_idx,
      kMaxWindowAttempts);
  }

  /// Moves @p run past the first @p consumed rows of its window.
  void advance(run_window& run,
               round_windows& windows,
               std::size_t consumed,
               rmm::cuda_stream_view stream) const
  {
    if (consumed == 0) { return; }
    const auto batch_before = run.cursor.batch;
    run.cursor.advance(consumed);
    if (!run.gpu_lock) { run.buffer->consume(consumed, stream); }
    if (run.cursor.batch == batch_before) { return; }
    // The batch is finished. A window never spans batches, so a buffer is empty here and the next
    // batch refills into it.
    windows.release_gpu(run);
    run.reader_state.reset();
    run.reader_type.reset();
  }

  [[nodiscard]] std::unique_ptr<cudf::table> empty_output() const
  {
    std::vector<std::unique_ptr<cudf::column>> columns;
    columns.reserve(fetch_types.size());
    for (const auto& type : fetch_types) {
      columns.push_back(cudf::make_empty_column(type));
    }
    return std::make_unique<cudf::table>(std::move(columns));
  }

  std::unique_ptr<cudf::table> run(rmm::cuda_stream_view stream,
                                   rmm::device_async_resource_ref mr,
                                   merge_round_stats* stats)
  {
    // Everything below works on copies; the caller's cursors and stats change only on return.
    round_windows windows(runs.size(), stream);
    for (std::size_t i = 0; i < runs.size(); ++i) {
      windows.runs[i].cursor = cursors[i];
      windows.runs[i].cursor.advance(0);  // normalize: never rest at the end of a batch
    }

    merge_round_stats local;
    std::vector<std::unique_ptr<cudf::table>> outputs;
    std::size_t output_bytes = 0;

    std::vector<std::size_t> active;
    std::vector<cudf::table_view> round_views;
    std::vector<std::uint8_t> truncated;

    while (true) {
      active.clear();
      round_views.clear();
      truncated.clear();
      // Every window is topped up before each merge, so no truncated window is ever empty.
      for (std::size_t i = 0; i < windows.runs.size(); ++i) {
        auto& run = windows.runs[i];
        if (run.cursor.exhausted()) { continue; }
        fill_window(run, i, stream, mr, local);
        active.push_back(i);
        round_views.push_back(run.window);
        truncated.push_back(
          static_cast<std::size_t>(run.window.num_rows()) < run.cursor.remaining() ? 1 : 0);
      }
      if (active.empty()) { break; }

      auto merged = merge(
        round_views, keys, truncated, static_cast<cudf::size_type>(cfg.window_rows), stream, mr);
      ++local.rounds;
      if (!merged.output || merged.consumed.size() != active.size()) {
        throw sirius::internal_exception(
          "windowed merge: the bounded merge returned {} consumed counts for {} windows",
          merged.consumed.size(),
          active.size());
      }
      const auto out_rows       = static_cast<std::size_t>(merged.output->num_rows());
      std::size_t consumed_rows = 0;
      for (std::size_t w = 0; w < active.size(); ++w) {
        const auto consumed = merged.consumed[w];
        if (consumed < 0 || consumed > round_views[w].num_rows()) {
          throw sirius::internal_exception(
            "windowed merge: the bounded merge consumed {} rows of a {}-row window",
            consumed,
            round_views[w].num_rows());
        }
        consumed_rows += static_cast<std::size_t>(consumed);
      }
      if (consumed_rows != out_rows) {
        throw sirius::internal_exception(
          "windowed merge: the bounded merge emitted {} rows but consumed {}",
          out_rows,
          consumed_rows);
      }
      if (out_rows == 0) {
        throw sirius::internal_exception(
          "windowed merge: a round over {} refilled windows made no progress", active.size());
      }

      // The output owns its rows, so the windows may be released or overwritten from here on.
      round_views.clear();
      for (std::size_t w = 0; w < active.size(); ++w) {
        advance(
          windows.runs[active[w]], windows, static_cast<std::size_t>(merged.consumed[w]), stream);
      }

      local.rows_out += out_rows;
      output_bytes += table_bytes(merged.output->view(), stream);
      outputs.push_back(std::move(merged.output));
      if (output_bytes >= cfg.output_target_bytes) { break; }
    }

    std::unique_ptr<cudf::table> result;
    if (outputs.empty()) {
      result = empty_output();
    } else if (outputs.size() == 1) {
      result = std::move(outputs.front());
    } else {
      std::vector<cudf::table_view> views;
      views.reserve(outputs.size());
      for (const auto& output : outputs) {
        views.push_back(output->view());
      }
      result = cudf::concatenate(views, stream, mr);
    }

    // Commit: nothing below throws.
    for (std::size_t i = 0; i < runs.size(); ++i) {
      cursors[i].batch = windows.runs[i].cursor.batch;
      cursors[i].row   = windows.runs[i].cursor.row;
    }
    if (stats != nullptr) {
      stats->rounds += local.rounds;
      stats->rows_out += local.rows_out;
      stats->refill_rows += local.refill_rows;
      stats->refill_bytes += local.refill_bytes;
      stats->fallback_conversions += local.fallback_conversions;
    }
    return result;
  }
};

windowed_merger::windowed_merger(
  std::vector<std::vector<std::shared_ptr<cucascade::data_batch>>> runs,
  std::span<merge_input_cursor> cursors,
  std::vector<std::size_t> fetch_cols,
  std::vector<cudf::data_type> fetch_types,
  merge_key_spec keys,
  window_config cfg,
  const cucascade::row_range_reader_registry& readers,
  cucascade::memory::memory_space& gpu_space,
  bounded_merge_fn merge)
{
  if (runs.size() != cursors.size()) {
    throw std::invalid_argument("windowed_merger: runs and cursors differ in size");
  }
  for (std::size_t i = 0; i < runs.size(); ++i) {
    const auto& run = cursors[i].run;
    if (run.batch_ids.size() != run.batch_rows.size() || runs[i].size() != run.batch_ids.size()) {
      throw std::invalid_argument("windowed_merger: a run's batches do not match its cursor");
    }
    for (std::size_t b = 0; b < runs[i].size(); ++b) {
      if (!runs[i][b] || runs[i][b]->get_batch_id() != run.batch_ids[b]) {
        throw std::invalid_argument("windowed_merger: a run's batches do not match its cursor");
      }
    }
  }
  if (fetch_cols.size() != fetch_types.size()) {
    throw std::invalid_argument("windowed_merger: fetch_cols and fetch_types differ in size");
  }
  if (cfg.window_rows == 0 ||
      cfg.window_rows > static_cast<std::size_t>(std::numeric_limits<cudf::size_type>::max())) {
    throw std::invalid_argument("windowed_merger: window_rows must be in [1, 2^31)");
  }
  if (!merge) { throw std::invalid_argument("windowed_merger: no merge function"); }

  _impl              = std::make_unique<impl>();
  _impl->select_cols = std::vector<cudf::size_type>(fetch_cols.begin(), fetch_cols.end());
  _impl->has_strings = std::any_of(fetch_types.begin(), fetch_types.end(), [](const auto& type) {
    return type.id() == cudf::type_id::STRING;
  });
  _impl->runs        = std::move(runs);
  _impl->cursors     = cursors;
  _impl->fetch_cols  = std::move(fetch_cols);
  _impl->fetch_types = std::move(fetch_types);
  _impl->keys        = std::move(keys);
  _impl->cfg         = cfg;
  _impl->readers     = &readers;
  _impl->gpu_space   = &gpu_space;
  _impl->merge       = std::move(merge);
}

windowed_merger::windowed_merger(windowed_merger&&) noexcept            = default;
windowed_merger& windowed_merger::operator=(windowed_merger&&) noexcept = default;
windowed_merger::~windowed_merger()                                     = default;

std::unique_ptr<cudf::table> windowed_merger::run(rmm::cuda_stream_view stream,
                                                  rmm::device_async_resource_ref mr,
                                                  merge_round_stats* stats)
{
  return _impl->run(stream, mr, stats);
}

}  // namespace sirius::op::merge
