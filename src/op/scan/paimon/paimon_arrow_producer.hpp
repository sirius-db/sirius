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
#include "op/scan/arrow/arrow_input_copy.hpp"
#include "op/scan/paimon/paimon_batch_cursor.hpp"
namespace sirius::op::scan {
// The factory owns separately admitted native planning and read resources. It opens
// on the native lane and must return fields in the bound stable-ID order with plain
// Arrow layouts and no extension semantics. The producer charges detached data and staging.
// Bind with arrow_import_binding::emit_empty_source = false for native EOF.
class paimon_arrow_producer final : public arrow_batch_producer {
 public:
  using read_factory = std::function<std::unique_ptr<paimon_batch_cursor>(
    int64_t max_rows, scan_manager::preparation_stop_token)>;
  paimon_arrow_producer(materialization_program, arrow_copy_limits, read_factory);
  ~paimon_arrow_producer() override;
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
