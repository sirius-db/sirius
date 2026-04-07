/*
 * Copyright 2025, Sirius Contributors.
 * Licensed under the Apache License, Version 2.0
 */

#include "debug_utils.hpp"

#include "data/data_batch_utils.hpp"
#include "log/logging.hpp"

#include <cudf/copying.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/reduction.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/bit.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <cucascade/data/data_batch.hpp>

#include <spdlog/fmt/fmt.h>

#include <cuda_runtime.h>

#include <algorithm>

namespace sirius {

// ---------------------------------------------------------------------------
// host_column_nulls
// ---------------------------------------------------------------------------

bool host_column_nulls::is_null(int row) const
{
  if (!has_nulls) { return false; }
  return !cudf::bit_is_set(mask.data(), row);
}

// ---------------------------------------------------------------------------
// copy_null_mask_to_host
// ---------------------------------------------------------------------------

host_column_nulls copy_null_mask_to_host(cudf::column_view const& col,
                                         rmm::cuda_stream_view stream)
{
  host_column_nulls result;
  result.has_nulls = col.has_nulls();
  if (!result.has_nulls) { return result; }

  auto const word_count =
    cudf::bitmask_allocation_size_bytes(col.size()) / sizeof(cudf::bitmask_type);
  result.mask.resize(word_count);
  cudaMemcpyAsync(result.mask.data(),
                  col.null_mask(),
                  word_count * sizeof(cudf::bitmask_type),
                  cudaMemcpyDeviceToHost,
                  stream.value());
  stream.synchronize();
  return result;
}

// ---------------------------------------------------------------------------
// Tier guard helper (internal)
// ---------------------------------------------------------------------------

namespace {

bool is_gpu_tier(cucascade::data_batch const& batch, const char* func_name)
{
  auto* data = batch.get_data();
  if (data == nullptr) {
    SIRIUS_LOG_WARN("[SIRIUS_DIAG] {}: batch has no data", func_name);
    return false;
  }
  if (data->get_current_tier() != cucascade::memory::Tier::GPU) {
    SIRIUS_LOG_WARN("[SIRIUS_DIAG] {}: batch not in GPU tier (tier={}), skipping",
                    func_name,
                    static_cast<int>(data->get_current_tier()));
    return false;
  }
  return true;
}

// STATS-02: Classify types eligible for statistics computation.
// BOOL8 is explicitly excluded even though cudf::is_numeric(BOOL8) returns true.
bool is_stats_numeric(cudf::type_id id)
{
  switch (id) {
    case cudf::type_id::INT8:
    case cudf::type_id::INT16:
    case cudf::type_id::INT32:
    case cudf::type_id::INT64:
    case cudf::type_id::UINT8:
    case cudf::type_id::UINT16:
    case cudf::type_id::UINT32:
    case cudf::type_id::UINT64:
    case cudf::type_id::FLOAT32:
    case cudf::type_id::FLOAT64: return true;
    default: return false;
  }
}

// Determine widened output type for SUM to prevent overflow (Pitfall 3, Pitfall 6).
cudf::data_type sum_output_type(cudf::type_id id)
{
  switch (id) {
    case cudf::type_id::INT8:
    case cudf::type_id::INT16:
    case cudf::type_id::INT32: return cudf::data_type{cudf::type_id::INT64};
    case cudf::type_id::UINT8:
    case cudf::type_id::UINT16:
    case cudf::type_id::UINT32: return cudf::data_type{cudf::type_id::UINT64};
    case cudf::type_id::FLOAT32: return cudf::data_type{cudf::type_id::FLOAT64};
    default: return cudf::data_type{id};  // INT64, UINT64, FLOAT64 stay as-is
  }
}

// Extract a cudf scalar value as a formatted string, dispatching on the data type.
std::string scalar_to_string(cudf::scalar const& s,
                             cudf::data_type dt,
                             rmm::cuda_stream_view stream)
{
  if (!s.is_valid(stream)) { return "NULL"; }  // D-10

  switch (dt.id()) {
    case cudf::type_id::INT8: {
      auto val = static_cast<cudf::numeric_scalar<int8_t> const&>(s).value(stream);
      return fmt::format("{}", static_cast<int>(val));
    }
    case cudf::type_id::INT16: {
      auto val = static_cast<cudf::numeric_scalar<int16_t> const&>(s).value(stream);
      return fmt::format("{}", val);
    }
    case cudf::type_id::INT32: {
      auto val = static_cast<cudf::numeric_scalar<int32_t> const&>(s).value(stream);
      return fmt::format("{}", val);
    }
    case cudf::type_id::INT64: {
      auto val = static_cast<cudf::numeric_scalar<int64_t> const&>(s).value(stream);
      return fmt::format("{}", val);
    }
    case cudf::type_id::UINT8: {
      auto val = static_cast<cudf::numeric_scalar<uint8_t> const&>(s).value(stream);
      return fmt::format("{}", val);
    }
    case cudf::type_id::UINT16: {
      auto val = static_cast<cudf::numeric_scalar<uint16_t> const&>(s).value(stream);
      return fmt::format("{}", val);
    }
    case cudf::type_id::UINT32: {
      auto val = static_cast<cudf::numeric_scalar<uint32_t> const&>(s).value(stream);
      return fmt::format("{}", val);
    }
    case cudf::type_id::UINT64: {
      auto val = static_cast<cudf::numeric_scalar<uint64_t> const&>(s).value(stream);
      return fmt::format("{}", val);
    }
    case cudf::type_id::FLOAT32: {
      auto val = static_cast<cudf::numeric_scalar<float> const&>(s).value(stream);
      return fmt::format("{:g}", val);  // D-04
    }
    case cudf::type_id::FLOAT64: {
      auto val = static_cast<cudf::numeric_scalar<double> const&>(s).value(stream);
      return fmt::format("{:g}", val);  // D-04
    }
    default: return "?";
  }
}

}  // namespace

// ---------------------------------------------------------------------------
// debug_schema
// ---------------------------------------------------------------------------

void debug_schema(cucascade::data_batch const& batch,
                  rmm::cuda_stream_view stream,
                  std::vector<std::string> const& col_names)
{
  try {
    if (!is_gpu_tier(batch, "debug_schema")) { return; }

    cudf::table_view tv = get_cudf_table_view(batch);
    stream.synchronize();

    std::string output;
    output += fmt::format(
      "[SIRIUS_DIAG] schema: batch_id={} rows={} cols={}\n",
      batch.get_batch_id(),
      tv.num_rows(),
      tv.num_columns());
    output += fmt::format(
      "[SIRIUS_DIAG]   {:<6s} {:<20s} {:<15s} {:>8s} {:>8s}\n",
      "idx", "name", "type", "nulls", "null%");
    output += fmt::format(
      "[SIRIUS_DIAG]   {:-<6s} {:-<20s} {:-<15s} {:->8s} {:->8s}\n",
      "", "", "", "", "");

    for (cudf::size_type c = 0; c < tv.num_columns(); ++c) {
      auto const& col = tv.column(c);
      std::string name =
        (static_cast<std::size_t>(c) < col_names.size())
          ? col_names[static_cast<std::size_t>(c)]
          : fmt::format("col[{}]", c);
      auto nc  = col.null_count();
      if (nc < 0) { nc = 0; }
      double pct = (col.size() > 0) ? 100.0 * nc / col.size() : 0.0;
      output += fmt::format(
        "[SIRIUS_DIAG]   {:<6d} {:<20s} {:<15s} {:>8d} {:>7.1f}%\n",
        static_cast<int>(c),
        name,
        cudf::type_to_name(col.type()),
        static_cast<int>(nc),
        pct);
    }

    SIRIUS_LOG_DEBUG("{}", output);

  } catch (std::exception const& e) {
    SIRIUS_LOG_WARN("[SIRIUS_DIAG] debug_schema failed: {}", e.what());
  } catch (...) {
    SIRIUS_LOG_WARN("[SIRIUS_DIAG] debug_schema failed: unknown error");
  }
}

// ---------------------------------------------------------------------------
// debug_nulls
// ---------------------------------------------------------------------------

void debug_nulls(cucascade::data_batch const& batch,
                 rmm::cuda_stream_view stream,
                 std::vector<std::string> const& col_names)
{
  try {
    if (!is_gpu_tier(batch, "debug_nulls")) { return; }

    cudf::table_view tv = get_cudf_table_view(batch);
    stream.synchronize();

    std::string output;
    output += fmt::format(
      "[SIRIUS_DIAG] nulls: batch_id={} rows={} cols={}\n",
      batch.get_batch_id(),
      tv.num_rows(),
      tv.num_columns());
    output += fmt::format(
      "[SIRIUS_DIAG]   {:<6s} {:<20s} {:>8s} {:>8s}\n",
      "idx", "name", "nulls", "null%");
    output += fmt::format(
      "[SIRIUS_DIAG]   {:-<6s} {:-<20s} {:->8s} {:->8s}\n",
      "", "", "", "");

    for (cudf::size_type c = 0; c < tv.num_columns(); ++c) {
      auto const& col = tv.column(c);
      std::string name =
        (static_cast<std::size_t>(c) < col_names.size())
          ? col_names[static_cast<std::size_t>(c)]
          : fmt::format("col[{}]", c);
      auto nc  = col.null_count();
      if (nc < 0) { nc = 0; }
      double pct = (col.size() > 0) ? 100.0 * nc / col.size() : 0.0;
      output += fmt::format(
        "[SIRIUS_DIAG]   {:<6d} {:<20s} {:>8d} {:>7.1f}%\n",
        static_cast<int>(c),
        name,
        static_cast<int>(nc),
        pct);
    }

    SIRIUS_LOG_DEBUG("{}", output);

  } catch (std::exception const& e) {
    SIRIUS_LOG_WARN("[SIRIUS_DIAG] debug_nulls failed: {}", e.what());
  } catch (...) {
    SIRIUS_LOG_WARN("[SIRIUS_DIAG] debug_nulls failed: unknown error");
  }
}

// ---------------------------------------------------------------------------
// debug_head
// ---------------------------------------------------------------------------

void debug_head(cucascade::data_batch const& batch,
                cudf::size_type n,
                rmm::cuda_stream_view stream,
                DebugFormat format,
                std::vector<std::string> const& col_names)
{
  try {
    if (!is_gpu_tier(batch, "debug_head")) { return; }
    cudf::table_view tv = get_cudf_table_view(batch);
    stream.synchronize();

    auto num_cols = tv.num_columns();

    // D-13: Empty batch handling
    if (tv.num_rows() == 0) {
      std::string output;
      output += fmt::format("[SIRIUS_DIAG] head: batch_id={} rows=0 cols={}\n",
                            batch.get_batch_id(), num_cols);
      output += "[SIRIUS_DIAG]   (empty batch)\n";
      SIRIUS_LOG_DEBUG("{}", output);
      return;
    }

    // D-12: Clamp N to actual row count
    auto keep = std::min(n, tv.num_rows());

    // HEAD-03: cudf::slice for zero-copy row selection
    cudf::table_view sliced_tv = tv;
    if (keep < tv.num_rows()) {
      auto slices = cudf::slice(tv, {0, keep}, stream);
      sliced_tv   = slices.front();
    }

    auto num_rows = sliced_tv.num_rows();

    // Build column names
    std::vector<std::string> names(num_cols);
    for (cudf::size_type c = 0; c < num_cols; ++c) {
      names[c] = (static_cast<std::size_t>(c) < col_names.size())
                   ? col_names[static_cast<std::size_t>(c)]
                   : fmt::format("col[{}]", c);
    }

    // Extract string representations for each cell: cells[col][row]
    std::vector<std::vector<std::string>> cells(num_cols);
    for (cudf::size_type c = 0; c < num_cols; ++c) {
      auto const& col = sliced_tv.column(c);
      auto nulls      = copy_null_mask_to_host(col, stream);
      cells[c].resize(num_rows);

      auto tid = col.type().id();

      // Helper lambda: copy typed data from GPU, format each value.
      // col.data<T>() is offset-adjusted in cuDF 26.02 -- do NOT add col.offset().
      // But null_mask() is NOT offset-adjusted, so use col.offset() + r for null checks.
      auto extract_numeric = [&]<typename T>() {
        std::vector<T> host_vals(num_rows);
        cudaMemcpyAsync(host_vals.data(),
                        col.data<T>(),
                        sizeof(T) * num_rows,
                        cudaMemcpyDeviceToHost,
                        stream.value());
        stream.synchronize();
        for (cudf::size_type r = 0; r < num_rows; ++r) {
          if (nulls.is_null(col.offset() + r)) {
            cells[c][r] = "NULL";  // D-06
          } else {
            if constexpr (std::is_floating_point_v<T>) {
              cells[c][r] = fmt::format("{:g}", host_vals[r]);  // D-04
            } else if constexpr (std::is_same_v<T, int8_t>) {
              cells[c][r] = fmt::format("{}", static_cast<int>(host_vals[r]));
            } else {
              cells[c][r] = fmt::format("{}", host_vals[r]);
            }
          }
        }
      };

      // BOOL8 special handling (stored as int8_t, display as true/false per D-05)
      auto extract_bool = [&]() {
        std::vector<int8_t> host_vals(num_rows);
        cudaMemcpyAsync(host_vals.data(),
                        col.data<int8_t>(),
                        sizeof(int8_t) * num_rows,
                        cudaMemcpyDeviceToHost,
                        stream.value());
        stream.synchronize();
        for (cudf::size_type r = 0; r < num_rows; ++r) {
          if (nulls.is_null(col.offset() + r)) {
            cells[c][r] = "NULL";
          } else {
            cells[c][r] = host_vals[r] ? "true" : "false";  // D-05
          }
        }
      };

      switch (tid) {
        case cudf::type_id::INT8: extract_numeric.template operator()<int8_t>(); break;
        case cudf::type_id::INT16: extract_numeric.template operator()<int16_t>(); break;
        case cudf::type_id::INT32: extract_numeric.template operator()<int32_t>(); break;
        case cudf::type_id::INT64: extract_numeric.template operator()<int64_t>(); break;
        case cudf::type_id::UINT8: extract_numeric.template operator()<uint8_t>(); break;
        case cudf::type_id::UINT16: extract_numeric.template operator()<uint16_t>(); break;
        case cudf::type_id::UINT32: extract_numeric.template operator()<uint32_t>(); break;
        case cudf::type_id::UINT64: extract_numeric.template operator()<uint64_t>(); break;
        case cudf::type_id::FLOAT32: extract_numeric.template operator()<float>(); break;
        case cudf::type_id::FLOAT64: extract_numeric.template operator()<double>(); break;
        case cudf::type_id::BOOL8: extract_bool(); break;
        default:
          // Unsupported types (STRING, DECIMAL, TIMESTAMP, DATE) -- Phase 3
          for (cudf::size_type r = 0; r < num_rows; ++r) {
            cells[c][r] = "(unsupported)";
          }
          break;
      }
    }

    // Build output string
    std::string output;
    output += fmt::format("[SIRIUS_DIAG] head: batch_id={} rows={} cols={} showing={}\n",
                          batch.get_batch_id(), tv.num_rows(), num_cols, num_rows);

    if (format == DebugFormat::CSV) {
      // CSV format (HEAD-02)
      output += "[SIRIUS_DIAG]   ";
      for (cudf::size_type c = 0; c < num_cols; ++c) {
        if (c > 0) { output += ","; }
        output += names[c];
      }
      output += "\n";
      for (cudf::size_type r = 0; r < num_rows; ++r) {
        output += "[SIRIUS_DIAG]   ";
        for (cudf::size_type c = 0; c < num_cols; ++c) {
          if (c > 0) { output += ","; }
          output += cells[c][r];
        }
        output += "\n";
      }
    } else {
      // ALIGNED format (HEAD-01, D-03: dynamic column widths)
      std::vector<std::size_t> widths(num_cols);
      for (cudf::size_type c = 0; c < num_cols; ++c) {
        widths[c] = names[c].size();
        for (cudf::size_type r = 0; r < num_rows; ++r) {
          widths[c] = std::max(widths[c], cells[c][r].size());
        }
        widths[c] += 2;  // padding
      }

      // Header row
      output += "[SIRIUS_DIAG]   ";
      for (cudf::size_type c = 0; c < num_cols; ++c) {
        output += fmt::format("{:<{}s}", names[c], widths[c]);
      }
      output += "\n";

      // Separator row
      output += "[SIRIUS_DIAG]   ";
      for (cudf::size_type c = 0; c < num_cols; ++c) {
        output += std::string(widths[c], '-');
      }
      output += "\n";

      // Data rows
      for (cudf::size_type r = 0; r < num_rows; ++r) {
        output += "[SIRIUS_DIAG]   ";
        for (cudf::size_type c = 0; c < num_cols; ++c) {
          output += fmt::format("{:<{}s}", cells[c][r], widths[c]);
        }
        output += "\n";
      }
    }

    SIRIUS_LOG_DEBUG("{}", output);

  } catch (std::exception const& e) {
    SIRIUS_LOG_WARN("[SIRIUS_DIAG] debug_head failed: {}", e.what());
  } catch (...) {
    SIRIUS_LOG_WARN("[SIRIUS_DIAG] debug_head failed: unknown error");
  }
}

// ---------------------------------------------------------------------------
// debug_stats
// ---------------------------------------------------------------------------

void debug_stats(cucascade::data_batch const& batch,
                 rmm::cuda_stream_view stream,
                 std::vector<std::string> const& col_names)
{
  try {
    if (!is_gpu_tier(batch, "debug_stats")) { return; }
    cudf::table_view tv = get_cudf_table_view(batch);
    stream.synchronize();

    auto num_cols = tv.num_columns();

    std::string output;
    output += fmt::format("[SIRIUS_DIAG] stats: batch_id={} rows={} cols={}\n",
                          batch.get_batch_id(), tv.num_rows(), num_cols);

    // D-13: Empty batch
    if (tv.num_rows() == 0) {
      output += "[SIRIUS_DIAG]   (empty batch)\n";
      SIRIUS_LOG_DEBUG("{}", output);
      return;
    }

    // D-07: Summary table format consistent with debug_schema
    output += fmt::format(
      "[SIRIUS_DIAG]   {:<6s} {:<20s} {:<15s} {:>15s} {:>15s} {:>15s}\n",
      "idx", "name", "type", "min", "max", "sum");
    output += fmt::format(
      "[SIRIUS_DIAG]   {:-<6s} {:-<20s} {:-<15s} {:->15s} {:->15s} {:->15s}\n",
      "", "", "", "", "", "");

    for (cudf::size_type c = 0; c < num_cols; ++c) {
      auto const& col = tv.column(c);
      std::string name =
        (static_cast<std::size_t>(c) < col_names.size())
          ? col_names[static_cast<std::size_t>(c)]
          : fmt::format("col[{}]", c);
      auto type_name = cudf::type_to_name(col.type());

      if (!is_stats_numeric(col.type().id())) {
        // D-08: Non-numeric columns skipped
        output += fmt::format(
          "[SIRIUS_DIAG]   {:<6d} {:<20s} {:<15s} {:>15s} {:>15s} {:>15s}\n",
          static_cast<int>(c), name, type_name,
          "(non-numeric, skipped)", "", "");
        continue;
      }

      // STATS-03: Use cudf::minmax for combined min+max (1 kernel launch)
      auto [min_scalar, max_scalar] = cudf::minmax(col, stream);

      // Use cudf::reduce for SUM with widened output type (Pitfall 3)
      auto sum_agg  = cudf::make_sum_aggregation<cudf::reduce_aggregation>();
      auto sum_type = sum_output_type(col.type().id());
      auto sum_scalar = cudf::reduce(col, *sum_agg, sum_type, stream);

      // D-10: All-NULL columns show NULL
      std::string min_str = scalar_to_string(*min_scalar, col.type(), stream);
      std::string max_str = scalar_to_string(*max_scalar, col.type(), stream);
      std::string sum_str = scalar_to_string(*sum_scalar, sum_type, stream);

      output += fmt::format(
        "[SIRIUS_DIAG]   {:<6d} {:<20s} {:<15s} {:>15s} {:>15s} {:>15s}\n",
        static_cast<int>(c), name, type_name, min_str, max_str, sum_str);
    }

    SIRIUS_LOG_DEBUG("{}", output);

  } catch (std::exception const& e) {
    SIRIUS_LOG_WARN("[SIRIUS_DIAG] debug_stats failed: {}", e.what());
  } catch (...) {
    SIRIUS_LOG_WARN("[SIRIUS_DIAG] debug_stats failed: unknown error");
  }
}

}  // namespace sirius
