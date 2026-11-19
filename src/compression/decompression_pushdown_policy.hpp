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

#include <codegen/selection/decompression_pushdown_policy.hpp>
#include <log/logging.hpp>

/// Routes one decision-trace line to INFO when the query requests diagnostics, DEBUG
/// otherwise. A macro (not a function) so the level dispatch keeps the
/// call-site file/line and the lazy formatting of the underlying macros.
#define SIRIUS_DECOMPRESSION_PUSHDOWN_DIAG(diagnostics, ...) \
  do {                                                       \
    if (diagnostics) {                                       \
      SIRIUS_LOG_INFO(__VA_ARGS__);                          \
    } else {                                                 \
      SIRIUS_LOG_DEBUG(__VA_ARGS__);                         \
    }                                                        \
  } while (0)
