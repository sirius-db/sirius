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
// Serializes reader entry. Bind wake before claims or final-owner releases can run.
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
  virtual arrow_readiness readiness() const noexcept         = 0;
  virtual void bind_wakeup(scan_manager::preparation_wakeup) = 0;
  // Token-driven stops use system_error(operation_canceled); the coordinator decides
  // whether the attempt ended normally or failed.
  virtual arrow_production next(admitted_arrow_input, scan_manager::preparation_stop_token) = 0;
  // Runs on the execution domain (control thread for metadata_worker).
  virtual void close() = 0;
  // Thread-neutral interrupt hook; must wake external offers without closing an affine reader.
  virtual void cancel() noexcept = 0;
  // Qualified queues bound retained roots themselves; affine legacy readers
  // continue holding the shared serial gate through final input retirement.
  virtual bool releases_gate_after_next() const noexcept { return false; }
  virtual uint32_t minimum_permits() const noexcept { return 1; }
  // Bind before producers acquire roots; staging is charged in provider blocks.
  virtual void bind_allocation_granularity(uint64_t) {}
  virtual void check_status() const {}
};

// Accepted roots retain temporary permits while downstream consumers run.
scan_manager::scan_envelope arrow_preparation_envelope(arrow_batch_producer const&);

// Legacy immutable views: retained_bytes includes full parents and control storage.
// Owners/callbacks must be thread-neutral. A false offer does not join an active copy;
// backing stays readable until its final retained use.
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
// Waiters must hold no engine resource needed by consumers to return capacity.
class arrow_push_queue final : public arrow_batch_producer {
  struct impl;

 public:
  arrow_push_queue(materialization_program,
                   arrow_push_limits,
                   std::function<void(std::exception_ptr)> report_failure);
  // Fixed Q=2 plus one offered slot; bytes use the existing scan_task_batch_size.
  arrow_push_queue(materialization_program,
                   uint64_t scan_task_batch_size,
                   std::function<void(std::exception_ptr)> report_failure);
  enum class offer_outcome { accepted, capacity_wait, terminal };
  enum class capacity_result { available, closed, failed };
  class offered_slot {
   public:
    offered_slot() = default;
    offered_slot(offered_slot&&) noexcept;
    offered_slot& operator=(offered_slot&&) noexcept;
    offered_slot(offered_slot const&) = delete;
    ~offered_slot();
    bool held() const noexcept { return bool(state_); }
    capacity_result status() const noexcept { return status_; }
    // Call only after acquisition, before producing another root. Ownership is
    // retained by this slot on capacity_wait/terminal and cleared only on accept.
    void set_input(ArrowSchema&,
                   ArrowArray&,
                   std::shared_ptr<void> owner,
                   arrow_source_profile,
                   arrow_metadata_lengths = {});

   private:
    friend class arrow_push_queue;
    void reset() noexcept;
    std::shared_ptr<impl> state_;
    std::shared_ptr<void> credit_;
    std::shared_ptr<owned_arrow_batch> input_;
    capacity_result status_ = capacity_result::closed;
  };
  struct observation {
    uint32_t offered = 0, accepted = 0;
    uint64_t unique_retained_bytes = 0, reserved_bytes = 0;
    uint64_t control_reservation_bytes = 0;
  };
  offered_slot acquire_offered_slot();
  offer_outcome offer(offered_slot&);
  capacity_result wait_for_capacity_or_terminal(offered_slot const&);
  observation observe() const;
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
  bool releases_gate_after_next() const noexcept override;
  uint32_t minimum_permits() const noexcept override;
  void bind_allocation_granularity(uint64_t) override;
  void check_status() const override;

 private:
  std::shared_ptr<impl> impl_;
};
}  // namespace sirius::op::scan
