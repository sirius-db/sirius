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

#include "transparent/replay_admission.hpp"

#include "io/io_errors.hpp"
#include "log/logging.hpp"
#include "op/scan/table_scan/scan_contract.hpp"

#include <duckdb/common/exception.hpp>
#include <duckdb/main/client_context.hpp>
#include <duckdb/transaction/meta_transaction.hpp>
#include <duckdb/transaction/transaction_context.hpp>

#include <cstdlib>

namespace sirius::transparent {
late_failure_trace::~late_failure_trace() noexcept
{
  if (!cause) return;
  try {
    SIRIUS_LOG_INFO("late failure cause={} replay={} refused_by={}",
                    static_cast<unsigned>(*cause),
                    replay,
                    refused_by ? static_cast<int>(*refused_by) : -1);
  } catch (...) {
  }
}

std::shared_ptr<op::scan::test_injections const> latch_replay_injections(
  duckdb::ClientContext& context)
{
  auto const* enabled = std::getenv("SIRIUS_ENABLE_TEST_OPTIONS");
  if (!enabled || std::string_view(enabled) != "1") return {};
  auto result = std::make_shared<op::scan::test_injections>();
  duckdb::Value value;
  if (context.TryGetCurrentSetting("sirius_test_inject_gpu_task_oom", value) && !value.IsNull())
    result->gpu_task_oom = value.GetValue<uint64_t>();
  if (context.TryGetCurrentSetting("sirius_test_inject_gpu_task_launch_error", value) &&
      !value.IsNull())
    result->gpu_task_launch_error = value.GetValue<uint64_t>();
  if (context.TryGetCurrentSetting("sirius_test_gpu_task_retry_limit", value) && !value.IsNull())
    result->gpu_task_retry_limit = value.GetValue<uint64_t>();
  if (context.TryGetCurrentSetting("sirius_test_gpu_task_retry_backoff_ms", value) &&
      !value.IsNull())
    result->gpu_task_retry_backoff_ms = value.GetValue<uint64_t>();
  if (context.TryGetCurrentSetting("sirius_test_override_read_only", value) && !value.IsNull())
    result->override_read_only = value.GetValue<bool>();
  if (context.TryGetCurrentSetting("sirius_test_inject_transaction_mismatch", value) &&
      !value.IsNull())
    result->transaction_mismatch = value.GetValue<bool>();
  if (context.TryGetCurrentSetting("sirius_test_interrupt_before_replay", value) && !value.IsNull())
    result->interrupt_before_replay = value.GetValue<bool>();
  if (context.TryGetCurrentSetting("sirius_test_inject_non_rollbackable_state", value) &&
      !value.IsNull())
    result->non_rollbackable_state = value.GetValue<bool>();
  if (context.TryGetCurrentSetting("sirius_test_hold_footer_index", value) && !value.IsNull())
    result->hold_footer_index = value.GetValue<uint64_t>();
  if (context.TryGetCurrentSetting("sirius_test_hold_published_batch", value) && !value.IsNull())
    result->hold_published_batch = value.GetValue<bool>();
  return result;
}

replay_admission admit_cpu_replay(duckdb::ClientContext& context,
                                  bool statement_read_only,
                                  std::optional<uint64_t> captured_transaction_id,
                                  bool non_rollbackable_state,
                                  bool read_only_refuses)
{
  if (read_only_refuses && !statement_read_only) {
    return {false, late_failure_condition::not_read_only};
  }
  if (captured_transaction_id &&
      (!context.transaction.HasActiveTransaction() ||
       context.transaction.ActiveTransaction().global_transaction_id != *captured_transaction_id ||
       context.transaction.ActiveTransaction().transaction_validity.IsInvalidated())) {
    return {false, late_failure_condition::transaction_invalid};
  }
  if (context.interrupted.load()) { return {false, late_failure_condition::cancelled}; }
  if (non_rollbackable_state) { return {false, late_failure_condition::non_rollbackable}; }
  return {true, std::nullopt};
}

failure_cause classify_failure(std::exception_ptr error, late_failure_cause fallback)
{
  if (!error) return {fallback, {}};
  try {
    std::rethrow_exception(error);
  } catch (classified_execution_error const& e) {
    return {e.cause, e.what()};
  } catch (op::scan::unsupported_physical_input const& e) {
    return {late_failure_cause::physical_input, e.what()};
  } catch (op::scan::certificate_incomplete const& e) {
    return {late_failure_cause::certificate, e.what()};
  } catch (duckdb::IOException const& e) {
    return {late_failure_cause::reader_io, e.what()};
  } catch (io::credential_error const& e) {
    return {late_failure_cause::reader_io, e.what()};
  } catch (std::exception const& e) {
    return {fallback, e.what()};
  } catch (...) {
    return {fallback, "unknown exception"};
  }
}
}  // namespace sirius::transparent
