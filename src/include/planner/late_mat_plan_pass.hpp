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

// Plan-time column-lifetime pass for late materialization
// (SIRIUS_EXP_LATE_MAT_V2). Runs once per query over the finished physical
// operator tree (after set_parent_ops) and stamps each GPU_SCAN with a
// planned_deferral annotation: for every scan output column, the first
// ancestor operator that reads its CONTENT (vs merely moving it
// positionally). Fail-closed by construction — any operator shape the
// analysis does not model consumes everything at that point, which can only
// shorten lifetimes, never fabricate them. Decisions (widths, thresholds,
// arbitration, runtime guards) stay in the lowering backend.

namespace sirius::op {
class sirius_physical_operator;
}  // namespace sirius::op

namespace sirius::planner {

/// Analyze @p root and stamp planned_deferral annotations on its GPU_SCAN
/// operators. No-op unless the v2 sub-gate is on.
void run_late_mat_plan_pass(op::sirius_physical_operator& root);

}  // namespace sirius::planner
