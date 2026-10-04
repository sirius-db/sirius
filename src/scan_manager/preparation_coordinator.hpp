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
#include "exec/scoped_dispatcher.hpp"
#include "op/scan/batch_coalescer.hpp"
#include "op/sirius_physical_operator.hpp"
#include "pipeline/completion_handler.hpp"
#include "scan_manager/preparation_options.hpp"

#include <functional>
#include <thread>
namespace sirius::scan_manager {
enum class stop_reason : uint8_t { user_cancel, failure_induced, normal_eos };
class preparation_coordinator {
 public:
  struct job {
    std::function<std::unique_ptr<op::scan::scan_info>()> work;
    std::shared_ptr<preparation_unit> unit;
  };
  struct publication {
    std::unique_ptr<op::operator_data> input;
    size_t bytes = 0;
  };
  struct source {
    std::shared_ptr<op::scan::batch_coalescer> coalescer;
    std::function<std::optional<job>()> claim;
    std::function<publication(std::unique_ptr<op::scan::scan_info>)> construct;
    std::function<void(publication)> publish;
    std::function<void()> close;
    std::function<void(std::function<void()>)> bind_consumption;
  };
  enum class lifecycle { constructed, armed, running, quiescent };
  struct statistics {
    size_t runs = 0, jobs_peak = 0, results_peak = 0, output_peak = 0, wakes = 0,
           partial_emissions = 0;
    std::thread::id owner, runner, publisher;
    std::chrono::microseconds max_residence{0}, max_deadline_lateness{0};
    lifecycle phase = lifecycle::constructed;
  };
  preparation_coordinator(pipeline::completion_handler&,
                          exec::scoped_dispatcher&,
                          preparation_options);
  ~preparation_coordinator();
  preparation_coordinator(preparation_coordinator const&)            = delete;
  preparation_coordinator& operator=(preparation_coordinator const&) = delete;
  void add_source(source);
  std::shared_ptr<void> external_use();
  std::shared_ptr<preparation_gate> publication_gate() const;
  void arm();
  void run_on_query_thread();
  void request_stop(stop_reason) noexcept;
  void drain();
  std::shared_ptr<preparation_unit> admit_unit(unit_key, required_input_set);
  void retire_unit(unit_key);
  std::optional<unit_record> unit_state_snapshot(unit_key) const;
  statistics snapshot() const;
  void before_submission_for_testing(std::function<void()>);
  void clock_for_testing(std::function<op::scan::batch_coalescer::clock::time_point()>);
  void wake_for_testing();

 private:
  struct state;
  std::shared_ptr<state> state_;
};
}  // namespace sirius::scan_manager
