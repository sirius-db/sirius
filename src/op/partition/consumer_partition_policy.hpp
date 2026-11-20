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

namespace sirius::op {
class sirius_physical_partition;

/// Join-specific wiring is supplied by the planner, not recovered from pipeline dependencies.
class sirius_physical_hash_join;
class sirius_physical_nested_loop_join;
class sirius_physical_dense_count_join;
class sirius_physical_grouped_aggregate_merge;
class sirius_physical_concat;

void bind_partition_inputs(sirius_physical_hash_join& join,
                           sirius_physical_partition& build,
                           sirius_physical_partition& probe,
                           sirius_physical_concat* build_concat,
                           bool probe_drives = false);
void bind_partition_inputs(sirius_physical_nested_loop_join& join,
                           sirius_physical_partition& build,
                           sirius_physical_partition& probe);
void bind_partition_inputs(sirius_physical_dense_count_join& join,
                           sirius_physical_partition& preserved,
                           sirius_physical_partition& counted);
void bind_partition_input(sirius_physical_grouped_aggregate_merge& merge,
                          sirius_physical_partition& input);

}  // namespace sirius::op
