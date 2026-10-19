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

#include <cudf/column/column_view.hpp>

#include <cuda/stream>

#include <optional>

namespace sirius::op {

/**
 * @brief Folds an optional BOOL8 mask into a nonnullable conjunction and counts survivors
 *
 * With `initialize`, the prior conjunction is all true. Otherwise `running` contains the preceding
 * conjunction. Null incoming values reject rows; offsets are honored. A null `count` skips
 * counting; otherwise the caller zeros `count` on `stream`. The caller retains every buffer until
 * completion. No host readback is performed.
 */
void fold_dynamic_filter_mask(cudf::mutable_column_view running,
                              std::optional<cudf::column_view> incoming,
                              cudf::size_type* count,
                              bool initialize,
                              ::cuda::stream_ref stream);

}  // namespace sirius::op
