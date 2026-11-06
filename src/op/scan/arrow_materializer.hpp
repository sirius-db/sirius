/* Copyright 2026, Sirius Contributors. SPDX-License-Identifier: Apache-2.0 */
#pragma once
#include "exec/cuda_event_completion_poll.hpp"
#include "op/scan/materialization_contract.hpp"

#include <duckdb/common/arrow/arrow.hpp>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory_resource>
#include <optional>
#include <span>
#include <unordered_map>

namespace sirius::op::scan {
// Trusted readable lengths, keyed by the exact metadata buffer in the schema tree.
using arrow_metadata_lengths = std::unordered_map<char const*, std::size_t>;

// Ownership of the entire schema/array tree is transferred here. The producer must
// detach any statement-affine release callbacks before publication. Capacity retains
// the producer's input/staging reservation until the last asynchronous input read.
class owned_arrow_batch {
 public:
  // Every non-null metadata pointer requires its own trusted byte length.
  // Arrow C Data itself carries no metadata lengths.
  owned_arrow_batch(ArrowSchema& schema,
                    ArrowArray& array,
                    std::shared_ptr<void> capacity,
                    std::pmr::memory_resource& staging_resource,
                    arrow_metadata_lengths metadata_lengths = {});
  ~owned_arrow_batch();
  owned_arrow_batch(owned_arrow_batch const&)            = delete;
  owned_arrow_batch& operator=(owned_arrow_batch const&) = delete;
  bool has_capacity() const noexcept { return static_cast<bool>(capacity); }
  std::int64_t row_count() const noexcept { return array.length; }
  std::vector<cudf::data_type> input_types() const;
  void validate_for(materialization_program const&) const;
  std::size_t estimated_device_bytes() const;

 private:
  friend materialized_batch import_arrow_host(std::shared_ptr<owned_arrow_batch>,
                                              materialization_program const&,
                                              exec::retire_lane&,
                                              rmm::device_async_resource_ref);
  ArrowSchema schema{};
  ArrowArray array{};
  std::shared_ptr<void> capacity;
  // The producer's capacity owner must keep this resource alive through retirement.
  // It must support allocation here and deallocation on the lane's host drainer.
  std::pmr::memory_resource& staging_resource;
  arrow_metadata_lengths metadata_lengths;
};

std::vector<cudf::data_type> validate_arrow_host_flat(
  ArrowSchema const&, ArrowArray const&, arrow_metadata_lengths const& metadata_lengths = {});
// Conservative PMR storage for a validated flat input, including allocation alignment.
std::size_t arrow_import_staging_bytes(ArrowArray const&, std::span<cudf::data_type const>);
// The caller owns/drains the completion lane and its stream for the entire attempt.
// Output device reservations are the same task reservation used for Parquet decode.
// Producer buffers must be readable to their declared Arrow extents. All host
// normalization uses the input's staging resource and stays retained on the lane.
// Neither return nor an exception establishes full-attempt completion. Keep the
// output MR/backing reservation alive through output destruction and a subsequent
// lane.quiesce(), even if lane.idle(): commit recovery can precede queued output
// frees during unwinding. drain()/input retirement alone is not that barrier and
// does not authorize replay. Failed quiescence retains unconfirmed input ownership.
materialized_batch import_arrow_host(std::shared_ptr<owned_arrow_batch>,
                                     materialization_program const&,
                                     exec::retire_lane&,
                                     rmm::device_async_resource_ref);

struct arrow_stream_batch {
  std::shared_ptr<owned_arrow_batch> input;
  std::uint64_t sequence = 0;
  std::int64_t rows      = 0;
  std::size_t bytes      = 0;
};

class arrow_batch_source {
 public:
  virtual ~arrow_batch_source()                    = default;
  virtual std::optional<arrow_stream_batch> next() = 0;
  virtual void cancel() noexcept                   = 0;
};

struct arrow_stream_limits {
  std::int64_t max_batch_rows = std::numeric_limits<std::int64_t>::max();
  std::size_t max_batch_bytes = std::numeric_limits<std::size_t>::max();
};

struct arrow_stream_result {
  std::size_t batches = 0;
  std::int64_t rows   = 0;
};

using arrow_batch_sink = std::function<void(materialized_batch&&)>;

// Sequentially verifies and materializes a bounded source of owned Arrow batches.
// The source owns cursor progress and backpressure; this function never creates an
// unbounded queue and treats std::nullopt as EOS (a zero-row batch remains data).
arrow_stream_result materialize_arrow_stream(arrow_batch_source&,
                                             materialization_program const&,
                                             exec::retire_lane&,
                                             rmm::device_async_resource_ref,
                                             arrow_batch_sink,
                                             arrow_stream_limits = {});
}  // namespace sirius::op::scan
