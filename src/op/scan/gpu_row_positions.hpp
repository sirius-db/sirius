/* Copyright 2026, Sirius Contributors. SPDX-License-Identifier: Apache-2.0 */
#pragma once

#include <cudf/column/column.hpp>

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace sirius::op::scan {
// Reader-local source dictionary. Paths and selected input intervals are per source,
// never per output row. The row column remains cuDF's original UINT64 file position.
struct gpu_row_positions {
  struct source {
    std::string file;
    uint64_t file_rows = 0;
    std::vector<std::pair<uint64_t, uint64_t>> ranges;  // [first, end), selected input
  };
  std::vector<source> sources;
  std::unique_ptr<cudf::column> rows;
  // Absent when source context alone identifies every row (one source).
  std::unique_ptr<cudf::column> source_indices;
};
}  // namespace sirius::op::scan
