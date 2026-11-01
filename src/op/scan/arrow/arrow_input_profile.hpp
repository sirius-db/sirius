/* Copyright 2026, Sirius Contributors. SPDX-License-Identifier: Apache-2.0 */
#pragma once

#include <cudf/types.hpp>

#include <cuda_runtime_api.h>

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace sirius::op::scan {
// Planning-owned storage schema. It is never learned from the first input root.
struct arrow_bound_schema {
  struct column {
    std::string name;
    std::string format;
    cudf::data_type type;
    bool nullable = false;
  };
  std::vector<column> columns;
};
enum class arrow_source_memory { device, pinned_host, pageable_host };
struct arrow_allocation {
  uint64_t id      = 0;
  void const* base = nullptr;
  uint64_t bytes   = 0;
};
struct arrow_buffer_use {
  uint64_t allocation = 0;
  uint64_t offset     = 0;
  uint64_t bytes      = 0;
};
struct arrow_string_window {
  int64_t first = 0, last = 0;
};
struct arrow_source_profile {
  arrow_source_memory memory = arrow_source_memory::pageable_host;
  int device                 = -1;
  // Either an explicit synchronous-completion declaration or an owned event.
  bool completed          = false;
  cudaEvent_t ready_event = nullptr;
  // Retains every allocation, readiness event, allocator and plugin release chain.
  std::shared_ptr<void> leases;
  // Full retained host control backing: C Data trees, names/metadata and callback
  // state, including spare capacity. Payload parent allocations are separate.
  uint64_t control_bytes = 0;
  std::vector<arrow_allocation> allocations;
  std::optional<arrow_buffer_use> root_validity;
  // One entry per C Data child buffer, including absent validity/empty payloads.
  std::vector<std::vector<std::optional<arrow_buffer_use>>> buffers;
  // Declared selected character windows, verified against the actual offsets
  // before conversion. Required for nonempty string columns, on every placement.
  std::vector<std::optional<arrow_string_window>> strings;
};
struct arrow_input_estimate {
  uint64_t retained = 0, staging = 0, destination = 0, control = 0;
  uint64_t total() const;
};
}  // namespace sirius::op::scan
