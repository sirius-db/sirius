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
#include "scan_manager/preparation_stop_token.hpp"

#include <cstdint>
#include <exception>
#include <functional>
#include <memory>
struct ArrowArray;
struct ArrowSchema;
namespace paimon {
class BatchReader;
class Status;
}  // namespace paimon
namespace sirius::op::scan {
namespace detail {
[[noreturn]] void throw_paimon_status(paimon::Status const&);
}
// Owned by one leased native worker. Calls, including close/destruction, are serialized
// on that worker; cancellation is requested through the read-only stop token.
// Planning and read allocations require separate admission BEFORE entering this cursor.
class paimon_batch_cursor {
 public:
  using consumer = std::function<void(ArrowSchema const&, ArrowArray const&)>;
  paimon_batch_cursor(std::unique_ptr<paimon::BatchReader>, std::shared_ptr<void> native_owners);
  ~paimon_batch_cursor();
  paimon_batch_cursor(paimon_batch_cursor const&)            = delete;
  paimon_batch_cursor& operator=(paimon_batch_cursor const&) = delete;
  // The callback must synchronously create detached, admitted output and must not
  // publish or retain SDK pointers. Roots are released before this function returns.
  // true is a data batch (including zero rows); false is clean EOF only.
  // Stops cannot interrupt an entered native call; the caller drains it before close.
  bool consume(scan_manager::preparation_stop_token, int64_t max_rows, consumer const&);
  // Explicit close reports failures. The destructor is only best-effort cleanup.
  void close();

 private:
  enum class phase { active, eof, closed };
  // Reader and all root callbacks are destroyed before pool/module owners.
  std::shared_ptr<void> native_owners_;
  std::unique_ptr<paimon::BatchReader> reader_;
  std::exception_ptr failure_;
  phase phase_ = phase::active;
};
}  // namespace sirius::op::scan
