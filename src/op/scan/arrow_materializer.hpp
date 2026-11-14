/* Copyright 2026, Sirius Contributors. SPDX-License-Identifier: Apache-2.0 */
#pragma once
#include "exec/cuda_event_completion_poll.hpp"
#include "op/scan/arrow/arrow_input_profile.hpp"
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

// Owns the C Data tree and capacity through readiness and the final async read.
// Release callbacks must be thread-neutral. See scan.md#arrow-input-contract.
class owned_arrow_batch {
 public:
  owned_arrow_batch(ArrowSchema& schema,
                    ArrowArray& array,
                    std::shared_ptr<void> capacity,
                    std::pmr::memory_resource& staging_resource,
                    arrow_metadata_lengths metadata_lengths            = {},
                    std::shared_ptr<arrow_source_profile const> source = {});
  ~owned_arrow_batch();
  owned_arrow_batch(owned_arrow_batch const&)            = delete;
  owned_arrow_batch& operator=(owned_arrow_batch const&) = delete;
  bool has_capacity() const noexcept { return static_cast<bool>(capacity); }
  std::int64_t row_count() const noexcept { return array.length; }
  std::vector<cudf::data_type> input_types() const;
  void validate_for(materialization_program const&) const;
  std::size_t estimated_device_bytes() const;
  arrow_input_estimate estimate() const;
  arrow_source_profile const* source_profile() const noexcept { return source.get(); }
  // Called before publication; keeps admitted staging and the engine claim alive.
  void bind_staging(std::shared_ptr<void>, std::pmr::memory_resource&);
  void bind_credit(std::shared_ptr<void> credit);

 private:
  friend materialized_batch import_arrow_host(std::shared_ptr<owned_arrow_batch>,
                                              materialization_program const&,
                                              exec::retire_lane&,
                                              rmm::device_async_resource_ref);
  ArrowSchema schema{};
  ArrowArray array{};
  // Destroyed last, after root callbacks, leases, staging and backing owners.
  std::shared_ptr<void> credit;
  std::shared_ptr<void> staging_owner;
  std::shared_ptr<void> capacity;
  // Capacity keeps the PMR alive, including deallocation on the host drainer.
  std::pmr::memory_resource* staging_resource;
  arrow_metadata_lengths metadata_lengths;
  std::shared_ptr<arrow_source_profile const> source;
};

// Metadata and declared extents only: never dereferences device/pending payloads.
arrow_input_estimate validate_arrow_source(ArrowSchema const&,
                                           ArrowArray const&,
                                           arrow_source_profile const&,
                                           arrow_bound_schema const&,
                                           arrow_metadata_lengths const& = {});
void validate_arrow_binding(arrow_bound_schema const&, materialization_program const&);

std::vector<cudf::data_type> validate_arrow_host_flat(
  ArrowSchema const&, ArrowArray const&, arrow_metadata_lengths const& metadata_lengths = {});
// Conservative PMR storage for a validated flat input, including allocation alignment.
std::size_t arrow_import_staging_bytes(ArrowArray const&, std::span<cudf::data_type const>);
// The caller keeps the lane, stream and output MR/backing alive through output
// destruction and a subsequent lane.quiesce(), even if idle(): commit recovery can
// precede output frees during unwinding. drain() alone does not authorize replay;
// failed quiescence retains unconfirmed ownership.
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

// Sequential bounded verification; only nullopt means EOS, not a zero-row batch.
arrow_stream_result materialize_arrow_stream(arrow_batch_source&,
                                             materialization_program const&,
                                             exec::retire_lane&,
                                             rmm::device_async_resource_ref,
                                             arrow_batch_sink,
                                             arrow_stream_limits = {});
}  // namespace sirius::op::scan
