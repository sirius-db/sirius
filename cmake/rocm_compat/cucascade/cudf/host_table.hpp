/*
 * Copyright 2026, Sirius Contributors.
 * Licensed under the Apache License, Version 2.0 (see LICENSE).
 */
//! @file cuCascade host_table_allocation — ROCm stub.

#pragma once
#include "cucascade/memory/column_metadata.hpp"
#include <cstddef>
#include <vector>

namespace cucascade::memory {

class host_table_allocation {
 public:
  virtual ~host_table_allocation() = default;
  virtual std::size_t num_columns() const { return 0; }
  virtual std::size_t column_size(std::size_t /*i*/) const { return 0; }
  virtual std::vector<column_metadata> const& get_schema() const { return schema_; }
  virtual void slice(std::span<std::size_t> /*row_offsets*/) {}
 protected:
  std::vector<column_metadata> schema_;
};

struct host_table_packed_allocation {
  std::size_t total_bytes{0};
};

}  // namespace cucascade::memory
