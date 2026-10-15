/*
 * Copyright 2026, Sirius Contributors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace sirius::op::scan {

class parquet_split_info;

inline constexpr std::size_t invalid_parquet_file_index = std::numeric_limits<std::size_t>::max();

/// Decoded rows and their source-file offsets.
struct batch_row_run {
  std::string data_file_path;
  int64_t file_row_offset{0};
  int64_t batch_row_offset{0};
  int64_t num_rows{0};
  std::size_t file_index{invalid_parquet_file_index};
};

/// File offsets include pruned row groups; batch offsets do not.
[[nodiscard]] std::vector<batch_row_run> build_batch_layout(parquet_split_info const& split);

}  // namespace sirius::op::scan
