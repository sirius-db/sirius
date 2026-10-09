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

#pragma once

#include "exec/invocable.hpp"
#include "exec/try.hpp"

#include <cucascade/exec/semi_future.hpp>

namespace sirius::exec {

using cucascade::exec::broken_promise_error;
using cucascade::exec::executor_concept;
using cucascade::exec::executor_func;
using cucascade::exec::future;
using cucascade::exec::future_already_retrieved_error;
using cucascade::exec::future_invalid_error;
using cucascade::exec::inline_executor;
using cucascade::exec::make_semi_future;
using cucascade::exec::make_semi_future_with;
using cucascade::exec::promise;
using cucascade::exec::promise_already_satisfied_error;
using cucascade::exec::semi_future;
using cucascade::exec::tag;
using cucascade::exec::tag_t;

}  // namespace sirius::exec
