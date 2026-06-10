/*
 * Copyright 2025, Sirius Contributors.
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

// Phase 22.1 D-08 / D-09: shared per-GPU sirius_ioctx test helper.
//
// Lifted from test/cpp/scan/test_parquet_scan_task.cpp's local definition so
// that multiple scan-manager test TUs (test_parquet_scan_task.cpp +
// test_parquet_split_provider.cpp + future fixtures) can construct real
// per-GPU sirius_ioctx instances without code duplication.
//
// Phase 22.1 D-08 deletes the unit-test fallback in
// parquet_split_provider::run_batch (the legacy
// `else { datasource = cudf::io::datasource::create(file_path); }` branch
// that bypassed the sirius IO framework). Tests that drive
// parquet_split_provider in isolation must now seed _gpu_ioctxs via this
// helper so the production `sirius_ioctx::make_datasource(io_object)` path
// is the only path exercised in both test and production.

// sirius IO framework
#include <io/types.hpp>
#include <io/uring/uring_ioctx.hpp>

// rmm
#include <rmm/cuda_device.hpp>

// cuda
#include <cuda_runtime.h>

// standard library
#include <algorithm>
#include <cstddef>
#include <memory>
#include <unordered_map>

namespace sirius::scan_test_utils {

/// Construct per-GPU @c sirius_ioctx instances for tests that directly seed
/// scan-manager objects (parquet_scan_task_global_state, parquet_split_provider)
/// with the IO framework (Phase 19 IO-15 / Phase 22.1 D-08).
///
/// Mirrors the per-GPU init pattern used in
/// @c SiriusContext::initialize() (src/sirius_context.cpp:283) — each
/// @c uring_ioctx is constructed under @c rmm::cuda_set_device_raii so its
/// pinned bounce slots bind to the right CUDA context (P11 lock).
///
/// IO-05 / Plan 05-04 adds a mandatory-ioctx throw on empty gpu_ioctxs in
/// parquet_scan_task_global_state. Phase 22.1 D-08 / D-09 likewise makes
/// parquet_split_provider::run_batch throw on empty gpu_ioctxs. Tests
/// constructing either object directly must seed the map via this helper.
///
/// @param n_gpus  Desired GPU count. Clamped down to actual @c cudaGetDeviceCount
///                so the helper is safe on 1-GPU hosts (defaults to all visible
///                devices, capped at @p n_gpus).
inline std::unordered_map<int, std::shared_ptr<sirius::io::sirius_ioctx>> make_test_gpu_ioctxs(
  int n_gpus = 2)
{
  int device_count = 0;
  cudaGetDeviceCount(&device_count);
  if (device_count < 1) { device_count = 1; }
  int const effective_n = std::min(n_gpus, device_count);

  std::unordered_map<int, std::shared_ptr<sirius::io::sirius_ioctx>> ioctxs;
  ioctxs.reserve(effective_n);
  for (int dev = 0; dev < effective_n; ++dev) {
    rmm::cuda_set_device_raii device_guard{rmm::cuda_device_id{dev}};
    // Defaults match src/include/io/uring/uring_ioctx.hpp:85-88 (host_ring_depth=16,
    // ring_entries=64, n_reactors=4, bounce_slot_size=CHUNK_SIZE=1MiB).
    ioctxs[dev] = std::make_shared<sirius::io::uring_ioctx>(
      /*host_ring_depth=*/static_cast<unsigned>(16),
      /*ring_entries=*/static_cast<unsigned>(64),
      /*n_reactors=*/static_cast<size_t>(4),
      /*bounce_slot_size=*/sirius::io::CHUNK_SIZE);
  }
  return ioctxs;
}

inline std::shared_ptr<sirius::io::sirius_ioctx> make_test_ioctx()
{
  return std::make_shared<sirius::io::uring_ioctx>(
    /*host_ring_depth=*/static_cast<unsigned>(16),
    /*ring_entries=*/static_cast<unsigned>(64),
    /*n_reactors=*/static_cast<size_t>(4),
    /*bounce_slot_size=*/sirius::io::CHUNK_SIZE);
}

}  // namespace sirius::scan_test_utils
