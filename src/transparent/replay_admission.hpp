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

#include <cstdint>
#include <exception>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>

namespace duckdb {
class ClientContext;
}
namespace sirius::op::scan {
struct test_injections;
}
namespace sirius::transparent {
std::shared_ptr<op::scan::test_injections const> latch_replay_injections(duckdb::ClientContext&);
enum class late_failure_condition : uint8_t {
  not_read_only,
  transaction_invalid,
  cancelled,
  non_rollbackable,
  result_emitted,
  policy_refused_in_replay
};
struct replay_admission {
  bool admitted;
  std::optional<late_failure_condition> refused_by;
};
replay_admission admit_cpu_replay(duckdb::ClientContext&,
                                  bool statement_read_only,
                                  std::optional<uint64_t> captured_transaction_id,
                                  bool non_rollbackable_state,
                                  bool read_only_refuses);
enum class late_failure_cause : uint8_t {
  physical_input,
  certificate,
  checkpoint_revalidation,
  reader_io,
  oom_exhausted,
  retry_exhausted,
  gpu_error,
  other
};
// One bounded, best-effort decision record, including policy/cleanup early exits.
struct late_failure_trace {
  ~late_failure_trace() noexcept;
  std::optional<late_failure_cause> cause;
  bool replay = false;
  std::optional<late_failure_condition> refused_by;
};
class classified_execution_error : public std::runtime_error {
 public:
  classified_execution_error(late_failure_cause kind, std::string message)
    : std::runtime_error(std::move(message)), cause(kind)
  {
  }
  late_failure_cause cause;
};
struct failure_cause {
  late_failure_cause cause = late_failure_cause::other;
  std::string detail;
};
failure_cause classify_failure(std::exception_ptr error,
                               late_failure_cause fallback = late_failure_cause::other);
}  // namespace sirius::transparent
