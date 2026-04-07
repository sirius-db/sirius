/*
 * Copyright 2025, Sirius Contributors.
 * Licensed under the Apache License, Version 2.0
 */

#include "debug_utils.hpp"

#include "data/data_batch_utils.hpp"
#include "log/logging.hpp"

#include <cudf/null_mask.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/bit.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <cucascade/data/data_batch.hpp>

#include <spdlog/fmt/fmt.h>

#include <cuda_runtime.h>

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

}  // namespace sirius
