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

#include "exec/bounded_thread_pool.hpp"
#include "exec/query_lifecycle_registry.hpp"

#include <exception>
#include <type_traits>
#include <utility>

namespace sirius::exec {

/// Extend an existing work claim through invocation AND capture destruction. A callback may
/// destroy its task early to signal completion while still using its plan in the epilogue.
/// Member order keeps that plan alive until all captures have retired, including on discard.
/// Construct the callable before transferring the claim; neither transfer can throw.
template <typename F>
class leased_callable {
 public:
  static_assert(std::is_nothrow_move_constructible_v<F>);

  leased_callable(query_lifecycle_registry::work_lease&& work, F&& fn) noexcept
    : work_(std::move(work)), fn_(std::move(fn))
  {
  }
  leased_callable(leased_callable&&) noexcept        = default;
  leased_callable(const leased_callable&)            = delete;
  leased_callable& operator=(const leased_callable&) = delete;
  // Assignment could release the old lease before destroying its old callable.
  leased_callable& operator=(leased_callable&&) = delete;

  void operator()() { fn_(); }

 private:
  query_lifecycle_registry::work_lease work_;
  F fn_;
};

/// Submit work with a queue-disposal tag; the lease owns its query resource lifetime.
/// Report submission failure while captures are still alive: their destructors may signal success.
/// Constructing the caller's callable argument must not throw after taking ownership of a task.
template <typename ReportError, typename F>
void dispatch_with_work(bounded_thread_pool& pool,
                        bounded_thread_pool::slot&& slot,
                        query_id_t query,
                        query_lifecycle_registry::work_lease&& work,
                        ReportError&& report_error,
                        F&& fn)
{
  static_assert(std::is_nothrow_invocable_v<ReportError, std::exception_ptr>);
  auto owned = leased_callable{std::move(work), std::forward<F>(fn)};
  invocable<void()> queued;
  try {
    // Allocation precedes the nonthrowing callable move. Either owned or queued retains all
    // captures until the catch below; pool.dispatch also preserves queued on insertion failure.
    queued = std::move(owned);
    pool.dispatch(std::move(slot), std::move(queued), query);
  } catch (...) {
    report_error(std::current_exception());
    throw;
  }
}

}  // namespace sirius::exec
