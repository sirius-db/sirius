/*
 * Copyright 2026, Sirius Contributors.
 * Licensed under the Apache License, Version 2.0 (see LICENSE).
 */
//! @file cuCascade host_data_representation — ROCm stub.

#pragma once
#include "cucascade/data/common.hpp"
#include "cucascade/cudf/host_table.hpp"
#include "cucascade/memory/common.hpp"
#include <rmm/cuda_stream.hpp>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <utility>

namespace cucascade::memory { class memory_space; }

namespace cucascade {

class host_data_representation : public idata_representation {
 public:
  host_data_representation(std::unique_ptr<memory::host_table_allocation> table,
                           memory::memory_space* space)
    : idata_representation(*space), table_(std::move(table)) {}

  std::unique_ptr<memory::host_table_allocation> const& get_host_table() const { return table_; }

  std::size_t num_columns() const { return table_ ? table_->num_columns() : 0; }
  std::size_t column_size(std::size_t i) const { return table_ ? table_->column_size(i) : 0; }
  void slice(std::span<std::size_t> offsets) { if (table_) table_->slice(offsets); }

  std::unique_ptr<idata_representation> clone(rmm::cuda_stream_view) override {
    throw std::runtime_error("cuCascade stub: host_data clone");
  }

  std::size_t get_size_in_bytes() const override { return 0; }
  std::size_t get_uncompressed_data_size_in_bytes() const override { return 0; }

 private:
  std::unique_ptr<memory::host_table_allocation> table_;
};

class host_data_packed_representation : public idata_representation {
 public:
  host_data_packed_representation(std::unique_ptr<memory::host_table_packed_allocation> /*alloc*/,
                                  memory::memory_space& space)
    : idata_representation(space) {}

  std::unique_ptr<idata_representation> clone(rmm::cuda_stream_view) override {
    throw std::runtime_error("cuCascade stub: host_packed clone");
  }
  std::size_t get_size_in_bytes() const override { return 0; }
  std::size_t get_uncompressed_data_size_in_bytes() const override { return 0; }
};

}  // namespace cucascade
