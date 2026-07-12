/*
 * Copyright 2026, Sirius Contributors.
 * Licensed under the Apache License, Version 2.0 (see LICENSE).
 */
//! @file cuCascade disk_data_representation — ROCm stub.

#pragma once
#include "cucascade/data/common.hpp"
#include "cucascade/memory/common.hpp"
#include "cucascade/memory/disk_table.hpp"
#include <rmm/cuda_stream.hpp>
#include <memory>
#include <stdexcept>

namespace cucascade {

class disk_data_representation : public idata_representation {
 public:
  disk_data_representation(std::unique_ptr<memory::disk_table_allocation> alloc,
                           memory::memory_space& space)
    : idata_representation(space), alloc_(std::move(alloc)) {}

  memory::disk_table_allocation const& get_disk_table() const {
    if (!alloc_) throw std::runtime_error("cuCascade stub: disk_data get_disk_table — no allocation");
    return *alloc_;
  }

  std::unique_ptr<idata_representation> clone(rmm::cuda_stream_view) override {
    throw std::runtime_error("cuCascade stub: disk_data clone");
  }
  std::size_t get_size_in_bytes() const override { return 0; }
  std::size_t get_uncompressed_data_size_in_bytes() const override { return 0; }

 private:
  std::unique_ptr<memory::disk_table_allocation> alloc_;
};

}  // namespace cucascade
