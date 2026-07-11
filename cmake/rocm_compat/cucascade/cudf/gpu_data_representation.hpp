/*
 * Copyright 2026, Sirius Contributors.
 * Licensed under the Apache License, Version 2.0 (see LICENSE).
 */
//! @file cuCascade gpu_data_representation — ROCm stub.

#pragma once
#include "cucascade/data/common.hpp"
#include "cucascade/memory/common.hpp"
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <rmm/cuda_stream.hpp>
#include <memory>
#include <stdexcept>
#include <utility>

namespace cucascade::memory { class memory_space; }

namespace cucascade {

/// GPU representation holding a cudf::table. Sirius calls get_table_view().
class gpu_table_representation : public idata_representation {
 public:
  gpu_table_representation(std::unique_ptr<cudf::table> table,
                           memory::memory_space& space,
                           rmm::cuda_stream_view /*writer_stream*/)
    : idata_representation(space), table_(std::move(table)) {}

  template <typename Owner>
  gpu_table_representation(cudf::table_view /*view*/, Owner&& /*owner*/,
                           std::size_t /*alloc_size*/, memory::memory_space& space,
                           rmm::cuda_stream_view /*stream*/)
    : idata_representation(space) {}

  cudf::table_view get_table_view() const {
    if (!table_) throw std::runtime_error("cuCascade stub: no table");
    return table_->view();
  }

  std::unique_ptr<cudf::table> release_table(rmm::cuda_stream_view) {
    return std::move(table_);
  }

  std::unique_ptr<idata_representation> clone(rmm::cuda_stream_view) override {
    throw std::runtime_error("cuCascade stub: gpu_table clone");
  }

  std::size_t get_size_in_bytes() const override { return 0; }
  std::size_t get_uncompressed_data_size_in_bytes() const override { return 0; }

 private:
  std::unique_ptr<cudf::table> table_;
};

}  // namespace cucascade
