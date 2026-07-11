/*
 * Copyright 2026, Sirius Contributors.
 * Licensed under the Apache License, Version 2.0 (see LICENSE).
 */
//! @file cuCascade column_metadata — ROCm stub.
//! Used by host_table_allocation to describe each column's layout within
//! the pinned host buffer. Sirius populates these fields in cpu_source_task.cpp.

#pragma once
#include <cstdint>
#include <string>
#include <vector>

namespace cucascade::memory {

struct column_metadata {
  std::string name;
  std::int32_t type_id{0};
  std::int32_t size{0};
  bool nullable{false};

  // Per-column layout within the host allocation (populated by Sirius).
  std::int32_t num_rows{0};           // cudf::size_type is int32_t
  std::int32_t scale{0};
  bool has_data{false};
  std::size_t data_offset{0};
  std::size_t data_size{0};
  std::int32_t null_count{0};
  bool has_null_mask{false};
  std::size_t null_mask_offset{0};
  std::size_t null_mask_size{0};

  std::vector<column_metadata> children;
};

}  // namespace cucascade::memory
