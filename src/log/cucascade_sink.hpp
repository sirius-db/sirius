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

#include "log/level.hpp"

namespace sirius::log {

/// Installs, once, a cuCascade log sink that forwards every cuCascade record to
/// whichever Sirius sink get_sink() returns at the time of the record.
///
/// Idempotent and thread-safe; set_sink calls it on every swap, so cuCascade's
/// records always follow the active Sirius sink. It leaves cuCascade's level
/// threshold untouched; set_cucascade_log_level mirrors the Sirius level into it.
void attach_cucascade_logging();

/// Sets cuCascade's level threshold to `lvl`, so cuCascade skips formatting any
/// record the Sirius sink would drop. Thread-safe.
void set_cucascade_log_level(level lvl);

}  // namespace sirius::log
