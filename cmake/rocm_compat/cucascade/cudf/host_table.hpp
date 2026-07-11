/*
 * Copyright 2026, Sirius Contributors.
 * Licensed under the Apache License, Version 2.0 (see LICENSE).
 */
//! @file cuCascade host_table_allocation — ROCm stub.
//! Holds a pinned host buffer (multiple_blocks_allocation) + per-column
//! metadata describing the layout within that buffer.

#pragma once
#include "cucascade/memory/column_metadata.hpp"
#include "cucascade/memory/fixed_size_host_memory_resource.hpp"
#include "cucascade/memory/common.hpp"
#include <cstddef>
#include <memory>
#include <vector>

namespace cucascade::memory {

class host_table_allocation {
 public:
  /// Typedef for the shared_ptr type — Sirius uses this as
  /// `host_table_allocation::buffers_ptr` in host_table_chunk_reader.hpp.
  using buffers_ptr = std::shared_ptr<fixed_size_host_memory_resource::multiple_blocks_allocation>;

  virtual ~host_table_allocation() = default;

  /// The pinned host buffer allocation (may be null for empty tables).
  buffers_ptr allocation;

  /// Per-column layout metadata. Sirius reads .num_rows, .data_offset, etc.
  std::vector<column_metadata> columns;

  /// Total bytes of the allocation (sum of all column data + masks).
  std::size_t total_bytes{0};

  // --- Legacy accessors (kept for compatibility) ---
  virtual std::size_t num_columns() const { return columns.size(); }
  virtual std::size_t column_size(std::size_t i) const {
    return i < columns.size() ? static_cast<std::size_t>(columns[i].num_rows) : 0;
  }
  virtual std::vector<column_metadata> const& get_schema() const { return columns; }
  virtual void slice(std::span<std::size_t> /*row_offsets*/) {}

  /// Factory: creates a host_table_allocation from a buffer + column metadata.
  /// Sirius calls this as host_table_allocation::create(allocation, columns, offset).
  static std::unique_ptr<host_table_allocation> create(
    std::shared_ptr<fixed_size_host_memory_resource::multiple_blocks_allocation> alloc,
    std::vector<column_metadata> cols,
    std::size_t offset) {
    auto hta = std::make_unique<host_table_allocation>();
    hta->allocation = std::move(alloc);
    hta->columns = std::move(cols);
    hta->total_bytes = offset;
    return hta;
  }

 protected:
  host_table_allocation() = default;
};

struct host_table_packed_allocation {
  std::size_t total_bytes{0};
};

}  // namespace cucascade::memory
