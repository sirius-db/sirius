/*
 * Copyright 2025, Sirius Contributors.
 * Licensed under the Apache License, Version 2.0
 */

#include "debug_utils.hpp"

#include "data/data_batch_utils.hpp"
#include "log/logging.hpp"

#include <cudf/copying.hpp>
#include <cudf/fixed_point/fixed_point.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/reduction.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/bit.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <cucascade/data/data_batch.hpp>

#include <spdlog/fmt/fmt.h>

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdlib>
#include <type_traits>

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

// ---------------------------------------------------------------------------
// Decimal formatting helpers (D-04, D-05)
// ---------------------------------------------------------------------------

// Format DECIMAL32/DECIMAL64 raw integer with fixed-point decimal placement.
// Uses unsigned magnitude to handle INT_MIN / INT64_MIN correctly (T-03-03).
template <typename T>
std::string format_decimal_value(T raw, int abs_scale)
{
  if (abs_scale == 0) { return fmt::format("{}", raw); }
  bool negative = raw < 0;
  // Cast to unsigned BEFORE negation to avoid UB on MIN values (T-03-03)
  using U     = std::make_unsigned_t<T>;
  U magnitude = negative ? static_cast<U>(0) - static_cast<U>(raw) : static_cast<U>(raw);
  std::string digits = fmt::format("{}", magnitude);
  // Pad with leading zeros if digit count <= abs_scale (Pitfall 4: 5 with scale=2 -> "0.05")
  while (digits.size() <= static_cast<std::size_t>(abs_scale)) {
    digits.insert(digits.begin(), '0');
  }
  auto decimal_pos = digits.size() - static_cast<std::size_t>(abs_scale);
  digits.insert(decimal_pos, ".");
  return (negative ? "-" : "") + digits;
}

// Format DECIMAL128 (__int128_t) raw integer with fixed-point decimal placement.
// Manual int128-to-string because fmt has no __int128_t formatter (Pitfall 2).
std::string format_decimal128_value(__int128_t raw, int abs_scale)
{
  bool negative = raw < 0;
  // Use unsigned __int128 for magnitude to handle MIN correctly
  using U128  = unsigned __int128;
  U128 magnitude = negative ? static_cast<U128>(0) - static_cast<U128>(raw) : static_cast<U128>(raw);

  // Build digits in reverse by repeated division by 10
  std::string digits;
  if (magnitude == 0) {
    digits = "0";
  } else {
    while (magnitude > 0) {
      digits.push_back(static_cast<char>('0' + static_cast<int>(magnitude % 10)));
      magnitude /= 10;
    }
    std::reverse(digits.begin(), digits.end());
  }

  if (abs_scale == 0) { return (negative ? "-" : "") + digits; }

  // Pad with leading zeros if digit count <= abs_scale
  while (digits.size() <= static_cast<std::size_t>(abs_scale)) {
    digits.insert(digits.begin(), '0');
  }
  auto decimal_pos = digits.size() - static_cast<std::size_t>(abs_scale);
  digits.insert(decimal_pos, ".");
  return (negative ? "-" : "") + digits;
}

// ---------------------------------------------------------------------------
// Timestamp/Date formatting helpers (D-06, D-07, D-08, D-09)
// ---------------------------------------------------------------------------

// Howard Hinnant's civil_from_days algorithm (public domain).
// Converts days since Unix epoch (1970-01-01) to calendar date.
// Handles negative days (pre-1970) correctly via floor division (Pitfall 5).
struct civil_date {
  int year;
  unsigned month;
  unsigned day;
};

civil_date civil_from_days(int32_t days_since_epoch)
{
  // Shift epoch from 1970-01-01 to 0000-03-01 for simpler leap year math
  int z   = days_since_epoch + 719468;
  int era = (z >= 0 ? z : z - 146096) / 146097;
  auto doe = static_cast<unsigned>(z - era * 146097);                   // day of era [0, 146096]
  auto yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;    // year of era [0, 399]
  int y    = static_cast<int>(yoe) + era * 400;
  auto doy = doe - (365 * yoe + yoe / 4 - yoe / 100);                  // day of year [0, 365]
  auto mp  = (5 * doy + 2) / 153;                                      // month [0, 11]
  auto d   = doy - (153 * mp + 2) / 5 + 1;                             // day [1, 31]
  auto m   = mp + (mp < 10 ? 3 : static_cast<unsigned>(-9));            // month [1, 12]
  y += (m <= 2);
  return {y, m, d};
}

// Format TIMESTAMP_SECONDS (int64_t seconds since epoch)
std::string format_timestamp_s(int64_t raw_s)
{
  // Floor division for days (Pitfall 5: negative timestamps)
  int32_t days =
    static_cast<int32_t>((raw_s >= 0) ? raw_s / 86400 : (raw_s - 86399) / 86400);
  int seconds_in_day = static_cast<int>(raw_s - static_cast<int64_t>(days) * 86400);
  auto [y, m, d]     = civil_from_days(days);
  int hh             = seconds_in_day / 3600;
  int mm             = (seconds_in_day % 3600) / 60;
  int ss             = seconds_in_day % 60;
  return fmt::format("{:04d}-{:02d}-{:02d} {:02d}:{:02d}:{:02d}", y, m, d, hh, mm, ss);
}

// Format TIMESTAMP_MILLISECONDS (int64_t milliseconds since epoch)
std::string format_timestamp_ms(int64_t raw_ms)
{
  int64_t total_seconds =
    (raw_ms >= 0) ? raw_ms / 1'000 : (raw_ms - 999) / 1'000;
  int frac_ms = static_cast<int>(raw_ms - total_seconds * 1'000);

  int32_t days =
    static_cast<int32_t>((total_seconds >= 0) ? total_seconds / 86400
                                              : (total_seconds - 86399) / 86400);
  int seconds_in_day = static_cast<int>(total_seconds - static_cast<int64_t>(days) * 86400);
  auto [y, m, d]     = civil_from_days(days);
  int hh             = seconds_in_day / 3600;
  int mm_t           = (seconds_in_day % 3600) / 60;
  int ss             = seconds_in_day % 60;

  std::string result =
    fmt::format("{:04d}-{:02d}-{:02d} {:02d}:{:02d}:{:02d}", y, m, d, hh, mm_t, ss);
  if (frac_ms != 0) {
    std::string frac = fmt::format(".{:03d}", frac_ms);
    // Trim trailing zeros from fractional part (D-08)
    while (frac.back() == '0') { frac.pop_back(); }
    result += frac;
  }
  return result;
}

// Format TIMESTAMP_MICROSECONDS (int64_t microseconds since epoch)
std::string format_timestamp_us(int64_t raw_us)
{
  int64_t total_seconds =
    (raw_us >= 0) ? raw_us / 1'000'000 : (raw_us - 999'999) / 1'000'000;
  int frac_us = static_cast<int>(raw_us - total_seconds * 1'000'000);

  int32_t days =
    static_cast<int32_t>((total_seconds >= 0) ? total_seconds / 86400
                                              : (total_seconds - 86399) / 86400);
  int seconds_in_day = static_cast<int>(total_seconds - static_cast<int64_t>(days) * 86400);
  auto [y, m, d]     = civil_from_days(days);
  int hh             = seconds_in_day / 3600;
  int mm             = (seconds_in_day % 3600) / 60;
  int ss             = seconds_in_day % 60;

  std::string result =
    fmt::format("{:04d}-{:02d}-{:02d} {:02d}:{:02d}:{:02d}", y, m, d, hh, mm, ss);
  if (frac_us != 0) {
    std::string frac = fmt::format(".{:06d}", frac_us);
    // Trim trailing zeros from fractional part (D-08)
    while (frac.back() == '0') { frac.pop_back(); }
    result += frac;
  }
  return result;
}

// Format TIMESTAMP_NANOSECONDS (int64_t nanoseconds since epoch)
std::string format_timestamp_ns(int64_t raw_ns)
{
  int64_t total_seconds =
    (raw_ns >= 0) ? raw_ns / 1'000'000'000 : (raw_ns - 999'999'999) / 1'000'000'000;
  int frac_ns = static_cast<int>(raw_ns - total_seconds * 1'000'000'000);

  int32_t days =
    static_cast<int32_t>((total_seconds >= 0) ? total_seconds / 86400
                                              : (total_seconds - 86399) / 86400);
  int seconds_in_day = static_cast<int>(total_seconds - static_cast<int64_t>(days) * 86400);
  auto [y, m, d]     = civil_from_days(days);
  int hh             = seconds_in_day / 3600;
  int mm             = (seconds_in_day % 3600) / 60;
  int ss             = seconds_in_day % 60;

  std::string result =
    fmt::format("{:04d}-{:02d}-{:02d} {:02d}:{:02d}:{:02d}", y, m, d, hh, mm, ss);
  if (frac_ns != 0) {
    std::string frac = fmt::format(".{:09d}", frac_ns);
    // Trim trailing zeros from fractional part (D-08)
    while (frac.back() == '0') { frac.pop_back(); }
    result += frac;
  }
  return result;
}

// Format TIMESTAMP_DAYS (int32_t days since epoch) as YYYY-MM-DD only (D-07)
std::string format_date_days(int32_t raw_days)
{
  auto [y, m, d] = civil_from_days(raw_days);
  return fmt::format("{:04d}-{:02d}-{:02d}", y, m, d);
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
                std::vector<std::string> const& col_names,
                cudf::size_type max_string_len)
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

        // ----- STRING (D-01, D-02, T-03-01) -----
        case cudf::type_id::STRING: {
          cudf::strings_column_view scv(col);
          // Copy offsets for the sliced range (col.offset() adjusts for sliced views)
          auto const num_offsets = num_rows + 1;
          std::vector<int32_t> host_offsets(num_offsets);
          cudaMemcpyAsync(host_offsets.data(),
                          scv.offsets().data<int32_t>() + col.offset(),
                          num_offsets * sizeof(int32_t),
                          cudaMemcpyDeviceToHost,
                          stream.value());
          stream.synchronize();
          // Copy only the chars bytes we need (offset[0] to offset[N])
          auto const chars_start = host_offsets[0];
          auto const chars_end   = host_offsets[num_rows];
          auto const chars_bytes = chars_end - chars_start;
          std::vector<char> host_chars(chars_bytes);
          if (chars_bytes > 0) {
            cudaMemcpyAsync(host_chars.data(),
                            scv.chars_begin(stream) + chars_start,
                            chars_bytes,
                            cudaMemcpyDeviceToHost,
                            stream.value());
            stream.synchronize();
          }
          for (cudf::size_type r = 0; r < num_rows; ++r) {
            if (nulls.is_null(col.offset() + r)) {
              cells[c][r] = "NULL";
            } else {
              auto const start = host_offsets[r] - chars_start;
              auto const end   = host_offsets[r + 1] - chars_start;
              std::string val(host_chars.data() + start, end - start);
              if (max_string_len > 0 &&
                  val.size() > static_cast<std::size_t>(max_string_len)) {
                val.resize(max_string_len);
                val += "...";
              }
              cells[c][r] = std::move(val);
            }
          }
          break;
        }

        // ----- DECIMAL32/64/128 (D-04, D-05) -----
        case cudf::type_id::DECIMAL32: {
          int32_t scale     = col.type().scale();
          int abs_scale     = std::abs(scale);
          std::vector<int32_t> host_vals(num_rows);
          cudaMemcpyAsync(host_vals.data(),
                          col.data<int32_t>(),
                          sizeof(int32_t) * num_rows,
                          cudaMemcpyDeviceToHost,
                          stream.value());
          stream.synchronize();
          for (cudf::size_type r = 0; r < num_rows; ++r) {
            if (nulls.is_null(col.offset() + r)) {
              cells[c][r] = "NULL";
            } else {
              cells[c][r] = format_decimal_value(host_vals[r], abs_scale);
            }
          }
          break;
        }
        case cudf::type_id::DECIMAL64: {
          int32_t scale     = col.type().scale();
          int abs_scale     = std::abs(scale);
          std::vector<int64_t> host_vals(num_rows);
          cudaMemcpyAsync(host_vals.data(),
                          col.data<int64_t>(),
                          sizeof(int64_t) * num_rows,
                          cudaMemcpyDeviceToHost,
                          stream.value());
          stream.synchronize();
          for (cudf::size_type r = 0; r < num_rows; ++r) {
            if (nulls.is_null(col.offset() + r)) {
              cells[c][r] = "NULL";
            } else {
              cells[c][r] = format_decimal_value(host_vals[r], abs_scale);
            }
          }
          break;
        }
        case cudf::type_id::DECIMAL128: {
          int32_t scale     = col.type().scale();
          int abs_scale     = std::abs(scale);
          std::vector<__int128_t> host_vals(num_rows);
          cudaMemcpyAsync(host_vals.data(),
                          col.data<__int128_t>(),
                          sizeof(__int128_t) * num_rows,
                          cudaMemcpyDeviceToHost,
                          stream.value());
          stream.synchronize();
          for (cudf::size_type r = 0; r < num_rows; ++r) {
            if (nulls.is_null(col.offset() + r)) {
              cells[c][r] = "NULL";
            } else {
              cells[c][r] = format_decimal128_value(host_vals[r], abs_scale);
            }
          }
          break;
        }

        // ----- TIMESTAMP types (D-06, D-08, D-09) -----
        case cudf::type_id::TIMESTAMP_SECONDS: {
          std::vector<int64_t> host_vals(num_rows);
          cudaMemcpyAsync(host_vals.data(),
                          col.data<int64_t>(),
                          sizeof(int64_t) * num_rows,
                          cudaMemcpyDeviceToHost,
                          stream.value());
          stream.synchronize();
          for (cudf::size_type r = 0; r < num_rows; ++r) {
            if (nulls.is_null(col.offset() + r)) {
              cells[c][r] = "NULL";
            } else {
              cells[c][r] = format_timestamp_s(host_vals[r]);
            }
          }
          break;
        }
        case cudf::type_id::TIMESTAMP_MILLISECONDS: {
          std::vector<int64_t> host_vals(num_rows);
          cudaMemcpyAsync(host_vals.data(),
                          col.data<int64_t>(),
                          sizeof(int64_t) * num_rows,
                          cudaMemcpyDeviceToHost,
                          stream.value());
          stream.synchronize();
          for (cudf::size_type r = 0; r < num_rows; ++r) {
            if (nulls.is_null(col.offset() + r)) {
              cells[c][r] = "NULL";
            } else {
              cells[c][r] = format_timestamp_ms(host_vals[r]);
            }
          }
          break;
        }
        case cudf::type_id::TIMESTAMP_MICROSECONDS: {
          std::vector<int64_t> host_vals(num_rows);
          cudaMemcpyAsync(host_vals.data(),
                          col.data<int64_t>(),
                          sizeof(int64_t) * num_rows,
                          cudaMemcpyDeviceToHost,
                          stream.value());
          stream.synchronize();
          for (cudf::size_type r = 0; r < num_rows; ++r) {
            if (nulls.is_null(col.offset() + r)) {
              cells[c][r] = "NULL";
            } else {
              cells[c][r] = format_timestamp_us(host_vals[r]);
            }
          }
          break;
        }
        case cudf::type_id::TIMESTAMP_NANOSECONDS: {
          std::vector<int64_t> host_vals(num_rows);
          cudaMemcpyAsync(host_vals.data(),
                          col.data<int64_t>(),
                          sizeof(int64_t) * num_rows,
                          cudaMemcpyDeviceToHost,
                          stream.value());
          stream.synchronize();
          for (cudf::size_type r = 0; r < num_rows; ++r) {
            if (nulls.is_null(col.offset() + r)) {
              cells[c][r] = "NULL";
            } else {
              cells[c][r] = format_timestamp_ns(host_vals[r]);
            }
          }
          break;
        }

        // ----- DATE (TIMESTAMP_DAYS) (D-07) -----
        case cudf::type_id::TIMESTAMP_DAYS: {
          std::vector<int32_t> host_vals(num_rows);
          cudaMemcpyAsync(host_vals.data(),
                          col.data<int32_t>(),
                          sizeof(int32_t) * num_rows,
                          cudaMemcpyDeviceToHost,
                          stream.value());
          stream.synchronize();
          for (cudf::size_type r = 0; r < num_rows; ++r) {
            if (nulls.is_null(col.offset() + r)) {
              cells[c][r] = "NULL";
            } else {
              cells[c][r] = format_date_days(host_vals[r]);
            }
          }
          break;
        }

        default:
          // Truly unknown types
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
