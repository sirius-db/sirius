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

#include <rmm/resource_ref.hpp>

namespace sirius {
namespace memory {

/**
 * @brief Route cuDF's current-device-resource allocations through the cucascade adaptor.
 *
 * Workaround for a duplicate RMM per-device-resource registry. The Sirius extension is built with
 * hidden visibility, so its copy of the function-local static
 * `rmm::mr::detail::get_ref_map()::device_id_to_resource` is private to the extension, while
 * `libcudf.so` / `librmm.so` carry their own (GNU_UNIQUE) copy.
 * `cudf::set_current_device_resource_ref` called from the extension therefore writes the
 * extension's copy, but cuDF-internal allocations that fall back to
 * `rmm::mr::get_current_device_resource_ref()` — notably the cuco hash table built by
 * `cudf::groupby::detail::hash::compute_groupby` and the analogous hash-join/distinct paths — read
 * libcudf's copy, which is never set and defaults to raw `cudaMalloc`, bypassing the reservation
 * system. See rapidsai/rmm#826.
 *
 * This installs @p resource as the current device resource in *libcudf's* registry for @p device_id
 * by resolving libcudf's exported map symbol at runtime. Call it ONCE per GPU device during engine
 * initialization (not per operator): the install is process-wide, so every subsequent cuDF operator
 * (group-by, hash join, distinct, ...) routes its internal allocations through the adaptor.
 *
 * @warning DISABLED by default; opt in with `SIRIUS_ENABLE_LIBCUDF_RESOURCE_BRIDGE=1`. Routing
 * cuDF's internal (cuco / thrust) temporaries through the adaptor's stream-ordered async pool has
 * been observed to CORRUPT results at scale (TPC-H SF1000 produced invalid-unicode string columns;
 * SF100 and the bridge-disabled path are clean). cuDF's internal temporaries previously got
 * synchronous cudaMalloc; the async pool exposes a stream-ordering / reuse hazard with their
 * high-churn alloc/free pattern. The proper linkage fix (rmm#826) routes the SAME allocations
 * through the adaptor and would hit the SAME corruption — so that stream-ordering issue must be
 * resolved (in cuCascade / the cuDF integration) before tracking cuDF internals is safe by either
 * approach. Kept for investigation only.
 *
 * It is a stopgap; if libcudf has not yet lazily constructed its registry, this triggers
 * construction with a tiny cuDF op before writing, and is safe to retry: it is idempotent per device
 * and returns false (a no-op) when disabled or if the registry could not be resolved/constructed.
 *
 * @param device_id  CUDA device id whose current resource should be set.
 * @param resource   The cucascade adaptor (typically `memory_space.get_default_allocator()`).
 * @return true once the adaptor has been installed for @p device_id.
 */
bool ensure_libcudf_current_device_resource(int device_id, rmm::device_async_resource_ref resource);

}  // namespace memory
}  // namespace sirius
