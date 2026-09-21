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

#include <absl/functional/any_invocable.h>

namespace sirius::exec {

/// Move-only type-erased callable. Unlike @c std::function it accepts move-only
/// targets, and unlike @c std::move_only_function (C++23) it is available today.
/// Supports @c noexcept- and ref-qualified signatures, both of which the
/// pipeline and future primitives rely on.
template <typename Signature>
using invocable = absl::AnyInvocable<Signature>;

}  // namespace sirius::exec
