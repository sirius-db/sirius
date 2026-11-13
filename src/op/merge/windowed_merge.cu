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
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <cucascade/cudf/gpu_data_representation.hpp>
#include <cucascade/cudf/host_data_representation.hpp>
#include <cucascade/cudf/host_table.hpp>
#include <cucascade/data/disk_data_representation.hpp>
#include <cucascade/memory/column_metadata.hpp>
#include <cucascade/memory/common.hpp>
#include <cucascade/memory/disk_table.hpp>
#include <cucascade/memory/memory_reservation.hpp>

#include <algorithm>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <typeindex>
#include <typeinfo>
#include <utility>

namespace sirius::op::merge {

namespace {

/// Attempts to obtain a window of one batch before giving up: a downgrade can win the race against
/// the whole-batch fallback, and then the batch is converted again.
constexpr int kMaxWindowAttempts = 3;

/**
 * @brief Rows [begin, begin + rows) of @p col as a view, without touching the device.
 *
 * Unlike cudf::slice this does not count the nulls of the range (a device read per nullable
 * column): the null count of the view is an UPPER BOUND, exact only when it is 0. The windows
 * only reach the bounded merge, which reads the masks and never the counts.
 */
cudf::column_view slice_column(const cudf::column_view& col,
                               cudf::size_type begin,
                               cudf::size_type rows)
{
  const auto null_count = col.nullable() ? std::min(col.null_count(), rows) : 0;
  return cudf::column_view(col.type(),
                           rows,
                           col.head(),
                           col.null_mask(),
                           null_count,
                           col.offset() + begin,
                           std::vector<cudf::column_view>(col.child_begin(), col.child_end()));
}

cudf::table_view slice_rows(const cudf::table_view& table, std::size_t begin, std::size_t end)
{
  std::vector<cudf::column_view> columns;
  columns.reserve(static_cast<std::size_t>(table.num_columns()));
  for (const auto& col : table) {
    columns.push_back(slice_column(
      col, static_cast<cudf::size_type>(begin), static_cast<cudf::size_type>(end - begin)));
  }
  return cudf::table_view{columns};
}

/// The column metadata of a host or disk table; nullptr for any other representation.
const std::vector<cucascade::memory::column_metadata>* column_metadata_of(
  const cucascade::idata_representation& data)
{
  if (const auto* host = dynamic_cast<const cucascade::host_data_representation*>(&data)) {
    return host->get_host_table() ? &host->get_host_table()->columns : nullptr;
  }
  if (const auto* disk = dynamic_cast<const cucascade::disk_data_representation*>(&data)) {
    return &disk->get_disk_table().columns;
  }
  return nullptr;
}

/**
 * @brief The window buffer of one run on a non-GPU tier. Lives as long as the merger, so a later
 * run() continues with the rows an earlier one read but did not emit.
 *
 * The unread rows are rows [start, buffer->size()) of the buffer; row `start` is source row `row`
 * of batch `batch` of the run. Consuming only moves `start`: the tail is copied to the front of
 * the buffer (window_buffer::consume) only right before a refill.
 */
struct buffered_run {
  std::unique_ptr<detail::window_buffer> buffer;
  cudf::table_view view;  // buffer->view(), taken after the last refill
  std::size_t start{0};
  std::size_t batch{0};
  std::size_t row{0};
  // Unread rows below which the buffer is compacted and refilled; set by every refill.
  std::size_t low_water{1};
  bool refill_blocked{false};  // the next row does not fit behind the tail: consume the tail first
  std::unique_ptr<cucascade::row_range_reader_state> reader_state;
  std::optional<std::type_index> reader_type;

  [[nodiscard]] std::size_t available() const noexcept
  {
    return buffer ? buffer->size() - start : 0;
  }

  /// Forgets the buffered rows and the reader state; keeps the allocation.
  void clear_rows(rmm::cuda_stream_view stream)
  {
    if (buffer && buffer->size() > 0) { buffer->consume(buffer->size(), stream); }
    start          = 0;
    low_water      = 1;
    refill_blocked = false;
    view           = cudf::table_view{};
    reader_state.reset();
    reader_type.reset();
  }

  /// Forgets the buffered rows and frees the buffer.
  void release() { *this = buffered_run{}; }
};

/// The window state of one run during a run() call.
struct run_window {
  merge_input_cursor cursor;

  // GPU tier: the read lock of the current batch, kept until the batch is finished or run() ends.
  std::optional<cucascade::read_only_data_batch> gpu_lock;
  cudf::table_view gpu_view;  // the fetched columns of the locked batch

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
  std::vector<std::size_t> string_cols;  // positions in fetch order of the STRING columns
  std::size_t fixed_row_bytes{0};        // bytes per row of the fixed-width fetched columns
  std::vector<buffered_run> buffered;    // one per run; kept across run() calls

  /// Estimated chars bytes per row of each fetched column of @p data (0 for the columns that are
  /// not STRING): each STRING column's own chars from the column metadata of host and disk
  /// tables, else an even share of the row width of the whole batch. window_buffer grows its
  /// chars when a column needs more.
  [[nodiscard]] std::vector<std::size_t> estimate_chars_bytes_per_row(
    const cucascade::idata_representation& data, std::size_t batch_rows) const
  {
    std::vector<std::size_t> estimates(fetch_cols.size(), 0);
    if (string_cols.empty() || batch_rows == 0) { return estimates; }
    const auto* columns  = column_metadata_of(data);
    const auto row_bytes = data.get_uncompressed_data_size_in_bytes() / batch_rows;
    const auto shared =
      (row_bytes > fixed_row_bytes ? row_bytes - fixed_row_bytes : 0) / string_cols.size();
    for (const auto col : string_cols) {
      const auto source = fetch_cols[col];
      const auto known =
        columns != nullptr && source < columns->size() && (*columns)[source].has_data;
      estimates[col] = (known ? (*columns)[source].data_size / batch_rows : shared) + 1;
    }
    return estimates;
  }

  /// The fetched columns of @p data that are known to hold no NULL, from the column metadata of
  /// host and disk tables. window_buffer exposes those without a mask.
  [[nodiscard]] std::vector<bool> null_free_columns(
    const cucascade::idata_representation& data) const
  {
    std::vector<bool> null_free(fetch_cols.size(), false);
    const auto* columns = column_metadata_of(data);
    if (columns == nullptr) { return null_free; }
    for (std::size_t i = 0; i < fetch_cols.size(); ++i) {
      if (fetch_cols[i] >= columns->size()) { continue; }
      const auto& column = (*columns)[fetch_cols[i]];
      null_free[i]       = !column.has_null_mask || column.null_count == 0;
    }
    return null_free;
  }

  /// Row capacity of a window buffer of @p cursor's run: a window never spans batches, so no more
  /// than the largest batch the run still has to read.
  [[nodiscard]] std::size_t buffer_rows(const merge_input_cursor& cursor) const
  {
    const auto& rows = cursor.run.batch_rows;
    const auto largest =
      *std::max_element(rows.begin() + static_cast<std::ptrdiff_t>(cursor.batch), rows.end());
    return std::min(cfg.window_rows, largest);
  }

  /// True when @p lock holds an uncompressed table in the task's GPU space, which is windowed by
  /// zero-copy slices. Any other GPU-tier batch (another space, a compressed device
  /// representation) goes through its reader or the whole-batch conversion.
  [[nodiscard]] bool gpu_resident(const cucascade::read_only_data_batch& lock) const
  {
    return lock.get_current_tier() == cucascade::memory::Tier::GPU &&
           lock.get_memory_space() == gpu_space &&
           dynamic_cast<const cucascade::gpu_table_representation*>(lock.get_data()) != nullptr;
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

  void slice_gpu_window(run_window& run) const
  {
    const auto batch_rows = run.cursor.run.batch_rows[run.cursor.batch];
    const auto begin      = run.cursor.row;
    run.window = slice_rows(run.gpu_view, begin, std::min(begin + cfg.window_rows, batch_rows));
  }

  /// Sets run.window to at most window_rows rows of the run's current batch, from the cursor on.
  void fill_window(run_window& run,
                   std::size_t run_idx,
                   rmm::cuda_stream_view stream,
                   rmm::device_async_resource_ref mr,
                   cucascade::memory::reservation* scratch,
                   merge_round_stats& stats)
  {
    if (run.gpu_lock) {
      slice_gpu_window(run);
      return;
    }
    auto& held            = buffered[run_idx];
    const auto& batch     = runs[run_idx][run.cursor.batch];
    const auto batch_rows = run.cursor.run.batch_rows[run.cursor.batch];
    const auto wanted     = std::min(cfg.window_rows, batch_rows - run.cursor.row);
    // Enough rows are buffered: no copy and no read. A window below `wanted` is truncated.
    if (held.available() >= std::min(wanted, held.low_water) ||
        (held.refill_blocked && held.available() > 0)) {
      run.window = slice_rows(held.view, held.start, held.buffer->size());
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
          // are dropped, and the buffer with them: the run reads slices from here on.
          held.release();
          use_gpu_batch(run, run_idx, std::move(lock));
          slice_gpu_window(run);
          return;
        }
        if (readers->has_reader(*data)) {
          const std::type_index type{typeid(*data)};
          if (held.reader_type != type) {
            held.reader_state.reset();
            held.reader_type = type;
          }
          if (!held.buffer) {
            held.buffer = std::make_unique<detail::window_buffer>(
              fetch_types,
              buffer_rows(run.cursor),
              estimate_chars_bytes_per_row(*data, batch_rows),
              stream,
              mr);
          }
          // Compact: carry the unread tail (less than half of the rows held) to the front, once.
          held.view = cudf::table_view{};
          held.buffer->consume(held.start, stream);
          stats.compacted_rows += held.buffer->size();
          held.start = 0;
          if (held.buffer->size() == 0) { held.buffer->set_null_free(null_free_columns(*data)); }
          const auto carried   = held.buffer->size();
          const auto requested = held.buffer->capacity_rows() - carried;
          try {
            // window_buffer grows its STRING slots once per call and then reports a row that still
            // does not fit. With a tail in the buffer the tail is the window (truncated) and the
            // refill waits until it is consumed; an empty buffer is grown until the row fits.
            std::optional<cucascade::row_range_result> read;
            while (!read) {
              const auto bytes_before   = held.buffer->device_bytes();
              const auto growths_before = held.buffer->chars_growths();
              try {
                read = held.buffer->refill(*readers,
                                           lock,
                                           fetch_cols,
                                           run.cursor.row + carried,
                                           &held.reader_state,
                                           scratch,
                                           stream);
                stats.window_growths += held.buffer->chars_growths() - growths_before;
              } catch (const detail::window_row_too_large_error&) {
                stats.window_growths += held.buffer->chars_growths() - growths_before;
                if (carried > 0) {
                  held.refill_blocked = true;
                  ++stats.refill_truncated;
                  break;
                }
                if (held.buffer->device_bytes() <= bytes_before) { throw; }
              }
            }
            if (read) {
              held.refill_blocked = false;
              stats.refill_rows += read->rows_written;
              for (const auto bytes : read->data_bytes_written) {
                stats.refill_bytes += bytes;
              }
              // Fewer rows than there was room for, with rows left in the batch: a STRING slot
              // ran out of chars.
              if (read->rows_written < requested && !read->source_exhausted) {
                ++stats.refill_truncated;
              }
              // The next refill waits until fewer than half of the rows now held are left and
              // every carried row is consumed: no row is carried twice, also when the refill
              // stopped short of the window.
              held.low_water =
                std::max<std::size_t>(1, std::min(held.buffer->size() / 2, read->rows_written + 1));
            }
            if (held.buffer->size() == 0) {
              throw sirius::internal_exception(
                "windowed merge: refill of run {} read no rows at row {} of batch {} ({} rows)",
                run_idx,
                run.cursor.row,
                run.cursor.batch,
                batch_rows);
            }
            held.view  = held.buffer->view();
            run.window = held.view;
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
               std::size_t run_idx,
               round_windows& windows,
               std::size_t consumed,
               rmm::cuda_stream_view stream)
  {
    if (consumed == 0) { return; }
    const auto batch_before = run.cursor.batch;
    run.cursor.advance(consumed);
    auto& held = buffered[run_idx];
    if (!run.gpu_lock) { held.start += consumed; }  // no copy: the next window starts further in
    if (run.cursor.batch == batch_before) { return; }
    // The batch is finished. A window never spans batches, so every buffered row was consumed and
    // the next batch refills into the empty buffer. A finished run gives its buffer back.
    windows.release_gpu(run);
    if (run.cursor.exhausted()) {
      held.release();
    } else {
      held.clear_rows(stream);
    }
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
    // The readers take their device memory from the resource of a scratch reservation: the
    // temporaries of a refill and, for a compressed batch, the decode state kept in the run's
    // reader state. An empty reservation of the task's GPU space names that space's allocator,
    // which charges what is allocated on `stream` to the task, like the windows. It reserves no
    // bytes and no reader draws on it; it lives for this call only, so that no reservation of
    // the merger is left open between two tasks.
    const auto scratch = gpu_space->make_reservation_upto(0);

    // Everything below works on copies; the caller's cursors and stats change only on return.
    round_windows windows(runs.size(), stream);
    for (std::size_t i = 0; i < runs.size(); ++i) {
      auto& cursor = windows.runs[i].cursor;
      cursor       = cursors[i];
      cursor.advance(0);  // normalize: never rest at the end of a batch
      // Rows buffered by an earlier run() are reused only if the cursor is still where it left it.
      auto& held = buffered[i];
      if (!held.buffer) { continue; }
      if (cursor.exhausted()) {
        held.release();
        continue;
      }
      // An earlier run() may have used another stream: this one's reads of the buffer wait for it.
      held.buffer->use_stream(stream);
      if (held.available() > 0 && (held.batch != cursor.batch || held.row != cursor.row)) {
        held.clear_rows(stream);
      }
    }
    // The buffers move ahead of the caller's cursors during the rounds. Unless run() commits, they
    // are dropped (which also frees their memory for a retry) and the replay reads them again.
    struct buffer_rollback {
      std::vector<buffered_run>& buffered;
      bool committed{false};
      ~buffer_rollback()
      {
        if (committed) { return; }
        for (auto& held : buffered) {
          held.release();
        }
      }
    } rollback{buffered};

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
        fill_window(run, i, stream, mr, scratch.get(), local);
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
        advance(windows.runs[active[w]],
                active[w],
                windows,
                static_cast<std::size_t>(merged.consumed[w]),
                stream);
      }

      local.rows_out += out_rows;
      // The sizes of the buffers the round's output owns: known on the host, no device read.
      output_bytes += merged.output->alloc_size();
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
      cursors[i].batch  = windows.runs[i].cursor.batch;
      cursors[i].row    = windows.runs[i].cursor.row;
      buffered[i].batch = cursors[i].batch;
      buffered[i].row   = cursors[i].row;
    }
    rollback.committed = true;
    if (stats != nullptr) {
      stats->rounds += local.rounds;
      stats->rows_out += local.rows_out;
      stats->refill_rows += local.refill_rows;
      stats->refill_bytes += local.refill_bytes;
      stats->refill_truncated += local.refill_truncated;
      stats->window_growths += local.window_growths;
      stats->compacted_rows += local.compacted_rows;
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
  for (std::size_t i = 0; i < fetch_types.size(); ++i) {
    if (fetch_types[i].id() == cudf::type_id::STRING) {
      _impl->string_cols.push_back(i);
    } else if (cudf::is_fixed_width(fetch_types[i])) {
      _impl->fixed_row_bytes += cudf::size_of(fetch_types[i]);
    }
  }
  _impl->buffered.resize(runs.size());
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

void windowed_merger::rebase_consumed()
{
  auto& state = *_impl;
  for (std::size_t i = 0; i < state.runs.size(); ++i) {
    auto& cursor = state.cursors[i];
    cursor.advance(0);  // normalize: a batch whose last row was read counts as read to the end
    const auto done = cursor.batch;
    if (done == 0) { continue; }
    const auto count = static_cast<std::ptrdiff_t>(done);
    state.runs[i].erase(state.runs[i].begin(), state.runs[i].begin() + count);
    cursor.run.batch_ids.erase(cursor.run.batch_ids.begin(), cursor.run.batch_ids.begin() + count);
    cursor.run.batch_rows.erase(cursor.run.batch_rows.begin(),
                                cursor.run.batch_rows.begin() + count);
    cursor.batch = 0;
    // The buffered rows belong to a batch that stays (a finished batch leaves no rows behind).
    // Rows buffered for a position the caller moved the cursor away from keep failing the
    // position check of the next run().
    auto& held = state.buffered[i];
    held.batch = held.batch >= done ? held.batch - done : std::numeric_limits<std::size_t>::max();
  }
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
