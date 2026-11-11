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

#include <cucascade/exec/cuda_event_completion_poll.hpp>

namespace sirius::exec {

// Every public namespace-scope name of the cuCascade header; its `detail` namespace is not
// re-exported.
using cucascade::exec::cacheline_v;
using cucascade::exec::completion_slot;
using cucascade::exec::cuda_event_completion_poll;
using cucascade::exec::no_pending_v;
using cucascade::exec::retire_fn;
using cucascade::exec::retire_lane;
using cucascade::exec::submission;

}  // namespace sirius::exec
