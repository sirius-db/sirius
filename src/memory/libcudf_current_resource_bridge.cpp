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

#include "memory/libcudf_current_resource_bridge.hpp"

#include "log/logging.hpp"

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <rmm/cuda_device.hpp>

#include <cuda/memory_resource>

#include <dlfcn.h>

#include <cstdlib>
#include <exception>
#include <map>
#include <mutex>
#include <set>
#include <vector>

namespace sirius {
namespace memory {
namespace {

// Mangled names of the GNU_UNIQUE statics that hold RMM's per-device
// current-device-resource registry. These are exported (as `u`) by libcudf.so /
// librmm.so — the single instance cuDF actually reads. The Sirius extension's
// own copies are hidden/local (DuckDB builds the extension with
// -fvisibility=hidden and the conda toolchain adds -fvisibility-inlines-hidden),
// so we resolve libcudf's copies directly via the global dynamic scope.
constexpr char const* kMapSym   = "_ZZN3rmm2mr6detail11get_ref_mapEvE21device_id_to_resource";
constexpr char const* kGuardSym = "_ZGVZN3rmm2mr6detail11get_ref_mapEvE21device_id_to_resource";
constexpr char const* kLockSym  = "_ZZN3rmm2mr6detail12ref_map_lockEvE12ref_map_lock";

// Must match the static in rmm::mr::detail::get_ref_map() exactly
// (rmm/mr/per_device_resource.hpp).
using resource_map_t =
  std::map<rmm::cuda_device_id::value_type, cuda::mr::any_resource<cuda::mr::device_accessible>>;

std::mutex g_mutex;
std::set<int> g_installed_devices;

// Itanium ABI: byte 0 of the guard is non-zero once the function-local static
// has been constructed. Reading/writing the map before then (it is
// zero-initialized BSS) would be undefined behavior.
bool libcudf_map_constructed(void* guard)
{
  return guard != nullptr && *reinterpret_cast<volatile char const*>(guard) != 0;
}

// Force libcudf to construct its current-device-resource registry by running a
// tiny cuco-backed op (distinct) whose hash table is allocated from
// rmm::mr::get_current_device_resource_ref() *inside libcudf* — the same path
// that otherwise bypasses the adaptor. Best-effort; the guard is re-checked by
// the caller. The current CUDA device must already be set by the caller.
void trigger_libcudf_registry_construction(rmm::device_async_resource_ref mr)
{
  try {
    auto const stream = cudf::get_default_stream();
    auto col          = cudf::make_numeric_column(
      cudf::data_type{cudf::type_id::INT32}, 2, cudf::mask_state::UNALLOCATED, stream, mr);
    (void)cudf::distinct(cudf::table_view{{col->view()}},
                         std::vector<cudf::size_type>{0},
                         cudf::duplicate_keep_option::KEEP_ANY,
                         cudf::null_equality::EQUAL,
                         cudf::nan_equality::ALL_EQUAL,
                         stream,
                         mr);
    stream.synchronize();
  } catch (std::exception const& e) {
    SIRIUS_LOG_DEBUG("[libcudf_resource_bridge] registry-construction trigger threw: {}", e.what());
  } catch (...) {
    SIRIUS_LOG_DEBUG("[libcudf_resource_bridge] registry-construction trigger threw (unknown)");
  }
}

}  // namespace

bool ensure_libcudf_current_device_resource(int device_id, rmm::device_async_resource_ref resource)
{
  // OPT-IN / default-OFF. Routing cuDF's internal (cuco hash-table / thrust scratch) allocations
  // through the cucascade adaptor's stream-ordered async pool has been observed to CORRUPT results
  // at scale (e.g. TPC-H SF1000 returns invalid-unicode string columns), almost certainly a
  // stream-ordering / memory-reuse issue with cuDF's high-churn internal temporaries — which
  // previously got synchronous cudaMalloc. It is therefore disabled unless explicitly enabled for
  // investigation. NOTE: the proper linkage fix (rmm#826) would route the same allocations through
  // the adaptor and hit the SAME corruption, so the underlying stream-ordering issue must be fixed
  // before either approach is safe. See src/include/memory/libcudf_current_resource_bridge.hpp.
  static bool const enabled = [] {
    char const* e = std::getenv("SIRIUS_ENABLE_LIBCUDF_RESOURCE_BRIDGE");
    return e != nullptr && e[0] != '\0' && e[0] != '0';
  }();
  if (!enabled) { return false; }

  std::lock_guard<std::mutex> const lk(g_mutex);
  if (g_installed_devices.count(device_id) != 0) { return true; }

  rmm::cuda_set_device_raii const dev{rmm::cuda_device_id{device_id}};

  void* guard = dlsym(RTLD_DEFAULT, kGuardSym);
  void* map   = dlsym(RTLD_DEFAULT, kMapSym);
  void* lockp = dlsym(RTLD_DEFAULT, kLockSym);
  if (guard == nullptr || map == nullptr) {
    SIRIUS_LOG_WARN(
      "[libcudf_resource_bridge] could not resolve libcudf RMM registry symbols "
      "(guard={}, map={}); cuDF current-resource allocations will NOT be reservation-tracked",
      guard != nullptr,
      map != nullptr);
    return false;
  }

  if (!libcudf_map_constructed(guard)) { trigger_libcudf_registry_construction(resource); }
  if (!libcudf_map_constructed(guard)) {
    SIRIUS_LOG_WARN(
      "[libcudf_resource_bridge] libcudf current-device-resource registry not constructed after "
      "trigger for device {}; deferring (cuDF internal allocations remain untracked)",
      device_id);
    return false;
  }

  auto& libcudf_map = *reinterpret_cast<resource_map_t*>(map);
  {
    // Hold libcudf's own registry mutex to avoid racing concurrent RMM get/set.
    if (lockp != nullptr) {
      auto& rmm_lock = *reinterpret_cast<std::mutex*>(lockp);
      std::lock_guard<std::mutex> const rmm_lk(rmm_lock);
      libcudf_map[device_id] = cuda::mr::any_resource<cuda::mr::device_accessible>{resource};
    } else {
      libcudf_map[device_id] = cuda::mr::any_resource<cuda::mr::device_accessible>{resource};
    }
  }

  g_installed_devices.insert(device_id);
  SIRIUS_LOG_INFO(
    "[libcudf_resource_bridge] routed cuDF current-device-resource for device {} through the "
    "cucascade adaptor (workaround for duplicate RMM per-device-resource map; rapidsai/rmm#826)",
    device_id);
  return true;
}

}  // namespace memory
}  // namespace sirius
