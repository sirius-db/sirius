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

/// Submit query work without depending on the pool's per-query lifetime accounting.
/// The caller's claim remains intact if constructing the callable argument throws.
template <typename F>
void dispatch_with_work(bounded_thread_pool& pool,
                        bounded_thread_pool::slot&& slot,
                        query_lifecycle_registry::work_lease&& work,
                        F&& fn)
{
  pool.dispatch(std::move(slot), leased_callable{std::move(work), std::forward<F>(fn)});
}

}  // namespace sirius::exec
