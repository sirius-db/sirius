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
#include <variant>

namespace sirius::scan_manager {
struct preparation_pending {};
struct preparation_exhausted {};
// Pending changes no cursor or resource ownership; exhausted is final for this attempt.
template <typename Job>
using preparation_claim = std::variant<preparation_exhausted, preparation_pending, Job>;
}  // namespace sirius::scan_manager
