/*
 * Copyright 2025, Sirius Contributors.
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

#include "scan_manager/split_provider.hpp"

#include "exec/completion_controller.hpp"
#include "exec/scoped_dispatcher.hpp"
#include "exec/try.hpp"
#include "op/scan/gpu_ingestible.hpp"

#include <cassert>
#include <exception>
#include <utility>

namespace sirius::scan_manager {

split_provider::split_provider(op::scan::gpu_ingestible& ingestible, io::sirius_ioctx& io_ctx)
  : _ingestible(&ingestible), _io_ctx(io_ctx.shared_from_this())
{
}

bool split_provider::has_more_splits() const { return !_ingestible->has_processed_all_metadata(); }

std::function<std::unique_ptr<op::scan::scan_info>()> split_provider::next_split_provider()
{
  return _ingestible->next_split_provider(_io_ctx);
}

template <typename Scheduler>
void split_provider::run(Scheduler& scheduler, const push_callback_t& on_split)
{
  using split_type = typename value_type::value_type;
  assert(on_split);
  exec::completion_controller ctrl;
  _completion_token =
    ctrl.on_completion([on_split] { on_split(exec::make_empty_try<split_type>()); });
  while (has_more_splits()) {
    auto work = next_split_provider();
    if (!work) { continue; }
    scheduler.enqueue([this, on_split, work = std::move(work), token = ctrl.acquire()]() mutable {
      try {
        auto split = work();
        on_split(std::move(split));
      } catch (...) {
        on_split(std::current_exception());
      }
    });
  }
}

// run() is a template parameterised on the scheduler, so its definition lives
// here rather than in the header. Explicitly instantiate it for the only
// scheduler the codebase drives it with (sirius_scan_manager's
// scoped_dispatcher); add a line here if a new scheduler type ever calls run().
template void split_provider::run<exec::scoped_dispatcher>(exec::scoped_dispatcher&,
                                                           const push_callback_t&);

}  // namespace sirius::scan_manager
