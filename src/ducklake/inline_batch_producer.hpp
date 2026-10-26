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
#include "ducklake/unit_descriptor.hpp"
#include "op/scan/arrow/arrow_import_ingestible.hpp"
namespace sirius::ducklake {
class binding_handle;
struct unit_scan_options;
// Reads on the original query owner and detaches each admitted Arrow input.
class inline_batch_producer final : public op::scan::arrow_batch_producer {
 public:
  ~inline_batch_producer() override;
  static constexpr size_t max_batch_rows = 2048, max_columns = 32;
  static constexpr size_t max_batch_cells   = 8192;
  static constexpr size_t max_string_bytes  = 1024 * 1024;
  static constexpr size_t max_staging_bytes = 64 * 1024;
  static size_t batch_rows(size_t columns);
  static uint64_t batch_capacity(size_t columns, size_t string_columns = 0);
  static scan_manager::unit_envelope envelope(size_t columns, size_t string_columns = 0);
  scan_manager::preparation_execution_domain execution_domain() const noexcept override;
  op::scan::arrow_producer_limits limits() const override;
  void preflight() override;
  op::scan::arrow_readiness readiness() const noexcept override;
  void bind_wakeup(scan_manager::preparation_wakeup) override;
  op::scan::arrow_production next(op::scan::admitted_arrow_input,
                                  scan_manager::preparation_stop_token) override;
  void close() override;
  void cancel() noexcept override;

 private:
  friend class binding_handle;
  inline_batch_producer(std::shared_ptr<binding_handle>,
                        unit_descriptor,
                        uint64_t,
                        uint64_t,
                        std::shared_ptr<scan_manager::preparation_admission>,
                        bool,
                        std::vector<size_t>,
                        unit_scan_options);
  op::scan::arrow_import_binding take_binding();
  struct impl;
  std::unique_ptr<impl> impl_;
};
}  // namespace sirius::ducklake
