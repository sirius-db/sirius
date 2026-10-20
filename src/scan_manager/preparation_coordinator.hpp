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
#include "exec/invocable.hpp"
#include "exec/scoped_dispatcher.hpp"
#include "op/scan/batch_coalescer.hpp"
#include "op/scan/sirius_gpu_scan_operator_data.hpp"
#include "pipeline/completion_handler.hpp"
#include "scan_manager/native_reader_executor.hpp"
#include "scan_manager/preparation_claim.hpp"
#include "scan_manager/preparation_job.hpp"
#include "scan_manager/preparation_options.hpp"
#include "scan_manager/preparation_stop_token.hpp"
#include "scan_manager/split_connector.hpp"

#include <functional>
#include <stop_token>
#include <thread>
#include <type_traits>
#include <variant>
#include <vector>
namespace sirius::scan_manager {
class preparation_coordinator {
 public:
  using job          = preparation_job;
  using claim_result = preparation_claim<job>;
  struct publication {
    std::unique_ptr<op::scan::scan_operator_input> input;
    size_t bytes             = 0;
    size_t preparation_bytes = 0;
  };
  using execution_domain = preparation_execution_domain;
  struct source {
    std::shared_ptr<op::scan::batch_coalescer> coalescer;
    // Always called by C. Must only claim independent control state, never enter a provider.
    exec::invocable<claim_result()> try_claim;
    // Return an input with deferred prefetch; construction runs outside the publication gate.
    std::function<publication(std::unique_ptr<op::scan::scan_info>)> construct;
    // The gate exposes only this concrete queue operation, never arbitrary source hooks.
    std::shared_ptr<split_connector> destination;
    std::function<void()> close;
    std::function<void(preparation_wakeup)> bind_wakeup;
    // query_owner: work and close run on the registering/arming thread. Returned scan_info
    // must own detached storage; close must release all thread-affine session resources.
    // native_reader: initialize lazily on the leased executor. Work and close share its
    // thread. Only detached, independently releasable storage may cross to materialization.
    execution_domain execution = execution_domain::metadata_worker;
  };
  enum class lifecycle { constructed, armed, running, quiescent };
  struct statistics {
    size_t runs = 0, jobs_peak = 0, results_peak = 0, output_peak = 0, wakes = 0,
           partial_emissions = 0, size_emissions = 0, normal_emissions = 0;
    std::thread::id owner, runner, publisher;
    std::chrono::microseconds max_residence{0}, max_deadline_lateness{0};

    lifecycle phase = lifecycle::constructed;
  };
  preparation_coordinator(pipeline::completion_handler&,
                          exec::scoped_dispatcher&,
                          preparation_options,
                          native_reader_executor* native = nullptr);
  ~preparation_coordinator();
  preparation_coordinator(preparation_coordinator const&)            = delete;
  preparation_coordinator& operator=(preparation_coordinator const&) = delete;
  void add_source(source);
  // Lanes have independent coalescers and credits, but one scan output. Close it after
  // every lane settles its tail; close_output runs outside the gate, before reader close.
  void add_sources(std::vector<source>, std::function<void()> close_output = {});
  std::shared_ptr<void> external_use();
  std::shared_ptr<preparation_gate> publication_gate() const;
  // Called on the coordinator: must be nonblocking and safe across threads.
  void set_interrupt_check(std::function<bool()>);
  void arm();
  // Arm creates the owned thread; start opens its gate after consumers are ready.
  void start();
  // Run at most one provider operation or close on Q. Never holds the gate during the call.
  // The owner must keep pumping during execution and retirement; external threads only stop/wait.
  // If idle, optionally wait for a progress notification before returning false.
  // The generation check also catches notifications racing the empty mailbox check.
  bool pump_provider(std::chrono::milliseconds idle_wait = std::chrono::milliseconds{0});
  void request_stop(stop_reason) noexcept;
  void drain();
  std::shared_ptr<preparation_unit> admit_unit(unit_key, required_input_set);
  void retire_unit(unit_key);
  std::optional<unit_record> unit_state_snapshot(unit_key) const;
  statistics snapshot() const;
  void before_thread_for_testing(std::function<void()>);
  void before_submission_for_testing(std::function<void()>);
  void before_publication_for_testing(std::function<void()>);
  void after_publication_for_testing(
    std::function<void(size_t, std::shared_ptr<op::scan::scan_info> const&, size_t)>);
  void clock_for_testing(std::function<op::scan::batch_coalescer::clock::time_point()>);
  void wake_for_testing();

 private:
  struct state;
  std::shared_ptr<state> state_;
  // Serializes handle creation and joining. The runner never takes this mutex.
  std::mutex execution_mutex_;
  std::thread runner_;
};
}  // namespace sirius::scan_manager
