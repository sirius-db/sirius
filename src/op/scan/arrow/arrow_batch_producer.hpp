/*
 * Copyright 2026, Sirius Contributors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once
#include "op/scan/arrow_materializer.hpp"
#include "scan_manager/preparation_job.hpp"
#include "scan_manager/preparation_ledger.hpp"

#include <atomic>
#include <variant>

namespace sirius::op::scan {
// Shared by producers whose retained inputs must not occupy two temporary permits.
struct arrow_input_gate {
  std::atomic<bool> active{false};
  scan_manager::preparation_wakeup wake;
};
struct admitted_arrow_input {
  scan_manager::unit_key key;
  scan_manager::charging_allocator allocator;
  std::shared_ptr<void> lifetime;
};
struct produced_arrow_input {
  std::shared_ptr<owned_arrow_batch> batch;
  std::shared_ptr<prepared_visibility const> provenance;
};
struct arrow_end_of_stream {};
using arrow_production = std::variant<produced_arrow_input, arrow_end_of_stream>;
enum class arrow_readiness { pending, ready };
struct arrow_producer_limits {
  int64_t max_rows;
  scan_manager::unit_envelope envelope;
};
// Trusted factories bind the program, original read identity and provider validator.
// A producer must allocate returned data through the admitted capability, detach
// thread-affine release callbacks, and retain the capability's lifetime in its root.
class arrow_batch_producer {
 public:
  virtual ~arrow_batch_producer()                                                      = default;
  virtual scan_manager::preparation_execution_domain execution_domain() const noexcept = 0;
  virtual arrow_producer_limits limits() const                                         = 0;
  virtual void preflight()                                                             = 0;
  // Cached host bookkeeping only: neither observation nor wake binding enters a reader.
  virtual arrow_readiness readiness() const noexcept                                        = 0;
  virtual void bind_wakeup(scan_manager::preparation_wakeup)                                = 0;
  virtual arrow_production next(admitted_arrow_input, scan_manager::preparation_stop_token) = 0;
  // Runs on the execution domain (control thread for metadata_worker).
  virtual void close() = 0;
  // Thread-neutral interrupt hook; must wake external offers without closing an affine reader.
  virtual void cancel() noexcept = 0;
};

// Accepted roots retain temporary permits while downstream consumers run.
scan_manager::scan_envelope arrow_preparation_envelope(arrow_batch_producer const&);

// Views and their complete backing must remain immutable and readable until the
// owner is released. retained_bytes includes parent extents and control storage,
// not just the selected slice. Final release may occur on any retaining thread;
// owners and callbacks must be thread-neutral. There is no affine reclamation handshake.
// A false offer result does not join an in-progress copy: retained views and backing
// must remain readable and immutable until their last retained use ends.
struct offered_arrow_input {
  ArrowSchema const* schema;
  ArrowArray const* array;
  std::shared_ptr<void> owner;
  uint64_t retained_bytes;
  arrow_metadata_lengths metadata_lengths;
};
struct arrow_push_limits {
  int64_t max_rows;
  uint64_t max_offered_bytes;
  uint64_t max_copy_bytes;
  uint64_t staging_bytes;
};
// One slot across all submitting threads. Only an external thread may wait in
// offer(), holding no engine lock/lease needed by consumers to return capacity.
class arrow_push_queue final : public arrow_batch_producer {
 public:
  arrow_push_queue(materialization_program,
                   arrow_push_limits,
                   std::function<void(std::exception_ptr)> report_failure);
  ~arrow_push_queue() override;
  bool offer(offered_arrow_input const&);
  void finish();
  void fail(std::exception_ptr);
  scan_manager::preparation_execution_domain execution_domain() const noexcept override;
  arrow_producer_limits limits() const override;
  void preflight() override;
  arrow_readiness readiness() const noexcept override;
  void bind_wakeup(scan_manager::preparation_wakeup) override;
  arrow_production next(admitted_arrow_input, scan_manager::preparation_stop_token) override;
  void close() override;
  void cancel() noexcept override;

 private:
  struct impl;
  std::unique_ptr<impl> impl_;
};
}  // namespace sirius::op::scan
