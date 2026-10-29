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

#include <cucascade/exec/invocable.hpp>

#ifndef CUCASCADE_USE_ABSEIL_INVOCABLE
// Enabled by set(CUCASCADE_USE_ABSEIL_INVOCABLE ON) in cmake/sirius-dependencies.cmake.
#error "sirius::exec::invocable requires cucascade::exec::invocable to be absl::AnyInvocable"
#endif

namespace sirius::exec {

/// Move-only type-erased callable; the same absl::AnyInvocable cuCascade's exec/io layers use.
template <typename Signature>
using invocable = cucascade::exec::invocable<Signature>;

}  // namespace sirius::exec
