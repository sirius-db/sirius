/* Copyright 2026, Sirius Contributors. SPDX-License-Identifier: Apache-2.0 */
#pragma once
#include "exec/cuda_event_completion_poll.hpp"
#include "op/scan/materialization_contract.hpp"

#include <duckdb/common/arrow/arrow.hpp>

namespace sirius::op::scan {
// Ownership of the entire schema/array tree is transferred here. The producer must
// detach any statement-affine release callbacks before publication. Capacity retains
// the producer's input/staging reservation until the last asynchronous input read.
class owned_arrow_batch {
 public:
  owned_arrow_batch(ArrowSchema& schema, ArrowArray& array, std::shared_ptr<void> capacity);
  ~owned_arrow_batch();
  owned_arrow_batch(owned_arrow_batch const&)            = delete;
  owned_arrow_batch& operator=(owned_arrow_batch const&) = delete;

 private:
  friend materialized_batch import_arrow_host(std::shared_ptr<owned_arrow_batch>,
                                              materialization_program const&,
                                              exec::retire_lane&,
                                              rmm::device_async_resource_ref);
  ArrowSchema schema{};
  ArrowArray array{};
  std::shared_ptr<void> capacity;
};

std::vector<cudf::data_type> validate_arrow_host_flat(ArrowSchema const&, ArrowArray const&);
// The caller owns/drains the completion lane and its stream for the entire attempt.
// Output device reservations are the same task reservation used for Parquet decode.
materialized_batch import_arrow_host(std::shared_ptr<owned_arrow_batch>,
                                     materialization_program const&,
                                     exec::retire_lane&,
                                     rmm::device_async_resource_ref);
}  // namespace sirius::op::scan
