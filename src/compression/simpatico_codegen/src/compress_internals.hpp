// SPDX-License-Identifier: Apache-2.0
// Internal helpers for the Simpatico codegen public API.
// Not part of the installed public headers — do not include from outside src/.
#pragma once

#include "api/simpatico_codegen.hpp"
#include "codegen/plan/plan_interpreter.hpp"
#include "codegen/util/stream_pool.hpp"

#include <cudf/column/column.hpp>
#include <cudf/utilities/traits.hpp>

#include <rmm/device_buffer.hpp>

#include <atomic>
#include <mutex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

namespace simpatico::detail {

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

class plan_error : public std::runtime_error {
 public:
  explicit plan_error(std::string const& msg) : std::runtime_error(msg) {}
};

// ---------------------------------------------------------------------------
// Plan DSL splitting
// ---------------------------------------------------------------------------

inline std::string trim_plan_block(std::string s)
{
  while (!s.empty() &&
         (s.back() == '\n' || s.back() == '\r' || s.back() == ' ' || s.back() == '\t'))
    s.pop_back();
  size_t start = 0;
  while (start < s.size() && (s[start] == ' ' || s[start] == '\t'))
    ++start;
  return s.substr(start);
}

// Split a multi-column DSL string on "---" separators, skipping blank lines
// and comment lines (beginning with '#'). Each returned block is trimmed.
inline std::vector<std::string> split_plan_dsl_impl(std::string_view plan_dsl)
{
  std::vector<std::string> plans;
  std::string current;
  size_t i = 0;
  while (i < plan_dsl.size()) {
    size_t line_end = plan_dsl.find('\n', i);
    if (line_end == std::string_view::npos) line_end = plan_dsl.size();
    std::string_view line = plan_dsl.substr(i, line_end - i);
    if (!line.empty() && line.back() == '\r') line.remove_suffix(1);

    std::string_view trimmed = line;
    while (!trimmed.empty() && trimmed.front() == ' ')
      trimmed.remove_prefix(1);
    while (!trimmed.empty() && trimmed.back() == ' ')
      trimmed.remove_suffix(1);

    if (trimmed == "---") {
      auto block = trim_plan_block(current);
      if (!block.empty()) plans.push_back(std::move(block));
      current.clear();
    } else if (!trimmed.empty() && trimmed.front() != '#') {
      current.append(trimmed);
      current.push_back('\n');
    }
    i = (line_end == plan_dsl.size()) ? plan_dsl.size() : line_end + 1;
  }
  auto block = trim_plan_block(current);
  if (!block.empty()) plans.push_back(std::move(block));
  return plans;
}

// ---------------------------------------------------------------------------
// Argument validation
// ---------------------------------------------------------------------------

inline void validate_plan_count(size_t plan_count, int table_columns)
{
  if (plan_count != static_cast<size_t>(table_columns)) {
    throw plan_error("plan count (" + std::to_string(plan_count) +
                     ") does not match table.num_columns() (" + std::to_string(table_columns) +
                     ")");
  }
}

inline void validate_column_names(std::vector<std::string> const& column_names, size_t num_columns)
{
  if (!column_names.empty() && column_names.size() != num_columns) {
    throw plan_error("column_names size (" + std::to_string(column_names.size()) +
                     ") does not match num_columns (" + std::to_string(num_columns) + ")");
  }
}

// ---------------------------------------------------------------------------
// Stream pool construction
// ---------------------------------------------------------------------------

// Create a stream pool with exactly max(1, column_threads) streams.
inline simpatico::stream_pool make_internal_pool(int column_threads)
{
  simpatico::stream_pool pool;
  if (!pool.init(static_cast<size_t>(std::max(1, column_threads)))) {
    throw plan_error("failed to initialize internal stream_pool");
  }
  return pool;
}

// ---------------------------------------------------------------------------
// Parallel column workers
// ---------------------------------------------------------------------------

inline compressed_table compress_columns_parallel(cudf::table_view table,
                                                  std::vector<std::string> const& plans,
                                                  simpatico::stream_pool& pool,
                                                  rmm::device_async_resource_ref mr,
                                                  std::vector<std::string> const& column_names)
{
  compressed_table out;
  out.columns.resize(plans.size());
  std::atomic<size_t> next{0};
  std::atomic<bool> failed{false};
  std::string err_msg;
  std::mutex err_mu;

  size_t const n_workers = pool.streams.size();
  std::vector<std::thread> workers;
  workers.reserve(n_workers);
  for (size_t w = 0; w < n_workers; ++w) {
    workers.emplace_back([&, w]() {
      while (true) {
        size_t i = next.fetch_add(1, std::memory_order_relaxed);
        if (i >= plans.size()) break;
        if (failed.load(std::memory_order_relaxed)) continue;

        rmm::cuda_stream_view stream{pool.streams[w % pool.streams.size()]};
        std::string err;
        auto compound = compress_column(
          table.column(static_cast<cudf::size_type>(i)), plans[i], stream, mr, &err);
        if (!compound) {
          std::lock_guard<std::mutex> lock(err_mu);
          if (!failed.exchange(true)) err_msg = err;
          continue;
        }
        compressed_column col;
        col.dtype    = table.column(static_cast<cudf::size_type>(i)).type();
        col.num_rows = table.num_rows();
        col.compound = std::move(compound);
        if (!column_names.empty()) col.name = column_names[i];
        out.columns[i] = std::move(col);
      }
    });
  }
  for (auto& t : workers)
    t.join();
  pool.sync_all();
  if (failed.load()) throw plan_error(err_msg.empty() ? "compress failed" : err_msg);
  return out;
}

// Restore a decoded column's logical type when it differs from the stored column
// dtype only in interpretation of identical bits (same physical width) — e.g. the
// INT64 storage a codec produced for a DECIMAL64 column back to DECIMAL64 with its
// scale. The codecs run on the underlying integer storage of fixed-point columns,
// so the bytes are already correct; this only re-tags the column. A no-op when the
// types already match.
inline std::unique_ptr<cudf::column> apply_stored_dtype(std::unique_ptr<cudf::column> col,
                                                        cudf::data_type stored)
{
  if (!col || col->type() == stored) return col;
  if (!cudf::is_fixed_width(col->type()) || !cudf::is_fixed_width(stored) ||
      cudf::size_of(col->type()) != cudf::size_of(stored)) {
    return col;
  }
  auto const n  = col->size();
  auto const nc = col->null_count();
  auto contents = col->release();
  rmm::device_buffer null_mask =
    contents.null_mask ? std::move(*contents.null_mask) : rmm::device_buffer{};
  return std::make_unique<cudf::column>(
    stored, n, std::move(*contents.data), std::move(null_mask), nc, std::move(contents.children));
}

inline std::unique_ptr<cudf::table> decompress_columns_parallel(compressed_table const& table,
                                                                simpatico::stream_pool& pool,
                                                                rmm::device_async_resource_ref mr)
{
  std::vector<std::unique_ptr<cudf::column>> cols(table.num_columns());
  std::atomic<size_t> next{0};
  std::atomic<bool> failed{false};
  std::string err_msg;
  std::mutex err_mu;

  size_t const n_workers = pool.streams.size();
  std::vector<std::thread> workers;
  workers.reserve(n_workers);
  for (size_t w = 0; w < n_workers; ++w) {
    workers.emplace_back([&, w]() {
      while (true) {
        size_t i = next.fetch_add(1, std::memory_order_relaxed);
        if (i >= table.num_columns()) break;
        if (failed.load(std::memory_order_relaxed)) continue;

        rmm::cuda_stream_view stream{pool.streams[w % pool.streams.size()]};
        std::string err;
        auto col = decompress_column(*table.columns[i].compound, stream, mr, &err);
        if (!col) {
          std::lock_guard<std::mutex> lock(err_mu);
          if (!failed.exchange(true)) err_msg = err;
          continue;
        }
        cols[i] = apply_stored_dtype(std::move(col), table.columns[i].dtype);
      }
    });
  }
  for (auto& t : workers)
    t.join();
  pool.sync_all();
  if (failed.load()) throw plan_error(err_msg.empty() ? "decompress failed" : err_msg);
  return std::make_unique<cudf::table>(std::move(cols));
}

}  // namespace simpatico::detail
