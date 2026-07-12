/*
 * Copyright 2026, Sirius Contributors.
 * Licensed under the Apache License, Version 2.0 (see LICENSE).
 */
//! @file cuCascade disk_table — ROCm stub.

#pragma once
#include <cstddef>
#include <string>

namespace cucascade::memory {

struct disk_table_allocation {
  std::string file_path;
  std::size_t size_bytes{0};
};

inline std::string generate_disk_file_path(std::string const& base, std::size_t idx) {
  return base + "/cucascade_disk_" + std::to_string(idx) + ".bin";
}

}  // namespace cucascade::memory
