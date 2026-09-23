/*
 * Copyright 2026, Sirius Contributors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 */

#include "op/scan/parquet_batch_layout.hpp"

#include "op/scan/parquet_gpu_ingestible.hpp"
#include "sirius/exception.hpp"

#include <limits>
#include <optional>
#include <string>
#include <vector>

namespace sirius::op::scan {
namespace {

int64_t checked_add(int64_t lhs, int64_t rhs, std::string const& context)
{
  if (rhs < 0) {
    throw sirius::internal_exception("[parquet_batch_layout] " + context +
                                     " has a negative row count");
  }
  if (lhs > std::numeric_limits<int64_t>::max() - rhs) {
    throw sirius::internal_exception("[parquet_batch_layout] " + context +
                                     " overflows a 64-bit row offset");
  }
  return lhs + rhs;
}

}  // namespace

std::vector<batch_row_run> build_batch_layout(parquet_split_info const& split)
{
  std::vector<batch_row_run> runs;
  int64_t batch_row_offset = 0;

  for (auto const& slice : split.rg_slices) {
    if (slice.file_index == invalid_parquet_file_index) {
      throw sirius::internal_exception("[parquet_batch_layout] row-group slice for '" +
                                       slice.file_path + "' has no bound file index");
    }
    if (!slice.file_metadata) {
      throw sirius::internal_exception("[parquet_batch_layout] row-group slice for '" +
                                       slice.file_path + "' has no footer metadata");
    }

    auto const& row_groups = slice.file_metadata->row_groups;
    std::vector<int64_t> first_row_of(row_groups.size() + 1, 0);
    for (std::size_t i = 0; i < row_groups.size(); ++i) {
      first_row_of[i + 1] = checked_add(first_row_of[i],
                                        static_cast<int64_t>(row_groups[i].num_rows),
                                        "footer for '" + slice.file_path + "'");
    }

    std::optional<std::size_t> previous_index;
    for (auto const rg_index : slice.row_group_indices) {
      if (rg_index < 0 || static_cast<std::size_t>(rg_index) >= row_groups.size()) {
        throw sirius::internal_exception("[parquet_batch_layout] row group index " +
                                         std::to_string(rg_index) + " for '" + slice.file_path +
                                         "' is outside its footer's row-group list");
      }
      auto const idx = static_cast<std::size_t>(rg_index);
      if (previous_index && idx <= *previous_index) {
        throw sirius::internal_exception("[parquet_batch_layout] selected row groups for '" +
                                         slice.file_path + "' must be strictly increasing");
      }
      previous_index               = idx;
      auto const num_rows          = static_cast<int64_t>(row_groups[idx].num_rows);
      auto const next_batch_offset = checked_add(
        batch_row_offset, num_rows, "decoded batch containing '" + slice.file_path + "'");
      runs.push_back(batch_row_run{
        slice.file_path, first_row_of[idx], batch_row_offset, num_rows, slice.file_index});
      batch_row_offset = next_batch_offset;
    }
  }

  return runs;
}

}  // namespace sirius::op::scan
