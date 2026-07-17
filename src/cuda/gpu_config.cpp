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

//! @file gpu_config.cpp
//! @brief Implementation of GPU capability detection and adaptive tuning.

#include "cuda/gpu_config.hpp"

#include <cstdlib>
#include <algorithm>
#include <cstdio>
#include <mutex>

// On ROCm, cuda* calls are shimmed to hip* via cuda2rocm.
// On NVIDIA, these are native CUDA Runtime API calls.
#include <cuda_runtime.h>

namespace sirius::cuda {

namespace {

/// Parse an environment variable as a size_t. Returns 0 if unset/invalid.
std::size_t env_size(char const* name) {
  char const* val = std::getenv(name);
  if (!val || !*val) return 0;
  // Support suffixes: K, M, G (e.g. "6G" = 6GB)
  char* end = nullptr;
  double n = std::strtod(val, &end);
  if (end == val) return 0;
  if (end && *end) {
    switch (*end) {
      case 'k': case 'K': n *= 1024; break;
      case 'm': case 'M': n *= 1024 * 1024; break;
      case 'g': case 'G': n *= 1024ULL * 1024 * 1024; break;
      default: break;
    }
  }
  return static_cast<std::size_t>(n);
}

/// Parse an environment variable as int. Returns 0 if unset/invalid.
int env_int(char const* name) {
  char const* val = std::getenv(name);
  if (!val || !*val) return 0;
  return std::atoi(val);
}

/// Parse an environment variable as bool (1/0, true/false).
bool env_bool(char const* name, bool default_val) {
  char const* val = std::getenv(name);
  if (!val || !*val) return default_val;
  return val[0] == '1' || val[0] == 't' || val[0] == 'T';
}

}  // namespace

// NOTE: gpu_config::instance() and the gpu_config singleton are useful
// infrastructure for runtime hardware detection and adaptive tuning, but they
// are NOT yet wired into the live execution paths. The engine still derives
// tuning parameters through the legacy hardcoded/env-var code paths (e.g.
// get_target_ctas in cuda/scan/strings/common.cuh, the partition sizing in
// sirius_physical_partition). Do not assume that constructing or consulting
// gpu_config has any effect on query execution today. This is kept as
// forward-looking plumbing; when it is wired in, the legacy paths should be
// migrated to read from this singleton instead.
gpu_config const& gpu_config::instance() {
  static gpu_config inst;
  return inst;
}

gpu_config::gpu_config() {
  detect_hardware();
  compute_tuning();
  apply_env_overrides();

  // Log the configuration once
  std::fprintf(stderr,
    "[sirius] GPU config: %s (%zu MB VRAM, %d CUs, %zu MB L2, wave=%d)\n"
    "[sirius] Tuning: batch=%zu MB, rmm_pool=%zu MB, scan_parallelism=%d, "
    "spilling=%s, target_ctas=%u\n",
    hw_.device_name.c_str(),
    hw_.total_vram / (1024 * 1024),
    hw_.multiprocessor_count,
    hw_.l2_cache_size / (1024 * 1024),
    hw_.warp_size,
    tune_.max_batch_bytes / (1024 * 1024),
    tune_.rmm_pool_size / (1024 * 1024),
    tune_.scan_parallelism,
    tune_.enable_spilling ? "on" : "off",
    tune_.target_ctas);

  initialized_ = true;
}

void gpu_config::detect_hardware() {
  int device = 0;
  cudaGetDevice(&device);
  hw_.device_id = device;

  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, device);

  hw_.total_vram = prop.totalGlobalMem;
  hw_.l2_cache_size = static_cast<std::size_t>(prop.l2CacheSize);
  hw_.multiprocessor_count = prop.multiProcessorCount;
  hw_.max_threads_per_sm = prop.maxThreadsPerMultiProcessor;
  hw_.max_shared_mem_per_block = prop.sharedMemPerBlock;
  hw_.warp_size = prop.warpSize > 0 ? prop.warpSize : 64;  // AMD default = 64
  hw_.memory_clock_khz = prop.memoryClockRate;
  hw_.bus_width_bits = prop.memoryBusWidth;
  hw_.major = prop.major;
  hw_.minor = prop.minor;
  hw_.device_name = prop.name;

  // Estimate memory bandwidth: clock * bus_width * 2 (DDR) / 8 (bits→bytes)
  if (hw_.memory_clock_khz > 0 && hw_.bus_width_bits > 0) {
    hw_.memory_bandwidth_gbs =
      (static_cast<std::size_t>(hw_.memory_clock_khz) * 1000ULL  // Hz
       * static_cast<std::size_t>(hw_.bus_width_bits) * 2ULL      // DDR
       / 8ULL)                                                    // bits→bytes
      / (1024ULL * 1024 * 1024);                                  // →GB/s
  }
}

void gpu_config::compute_tuning() {
  std::size_t const MB = 1024 * 1024;
  std::size_t const GB = 1024ULL * 1024 * 1024;

  // Detect multi-GPU configuration for pool sizing.
  // On multi-GPU systems, each GPU's pool must leave room for peer-access
  // overhead and cross-GPU transfer buffers.
  int device_count = 0;
  cudaGetDeviceCount(&device_count);
  if (device_count <= 0) device_count = 1;

  // RMM pool: scale down on multi-GPU to leave room for peer access.
  // Single GPU: 75%. 2 GPUs: 70%. 4+: 60%.
  double pool_fraction;
  if (device_count <= 1) {
    pool_fraction = 0.75;
  } else if (device_count <= 2) {
    pool_fraction = 0.70;
  } else if (device_count <= 4) {
    pool_fraction = 0.65;
  } else {
    pool_fraction = 0.60;
  }
  tune_.rmm_pool_size = static_cast<std::size_t>(hw_.total_vram * pool_fraction);

  // Max batch size for Parquet reads: 25% of VRAM, capped at 512MB minimum
  tune_.max_batch_bytes = std::max(hw_.total_vram / 4, 512 * MB);

  // Target CTAs: 2 waves of full-occupancy blocks
  // (mirrors the existing get_target_ctas logic)
  int constexpr STRINGS_BLOCK_DIM = 256;
  int occupancy_blocks = hw_.max_threads_per_sm / STRINGS_BLOCK_DIM;
  tune_.target_ctas = static_cast<uint32_t>(
    hw_.multiprocessor_count * std::max(occupancy_blocks, 1) * 2);

  // Spilling: enabled when VRAM < 64GB
  tune_.enable_spilling = hw_.total_vram < 64ULL * GB;

  // Scan parallelism: scale with VRAM
  if (hw_.total_vram >= 64ULL * GB) {
    tune_.scan_parallelism = 16;
  } else if (hw_.total_vram >= 16ULL * GB) {
    tune_.scan_parallelism = 8;
  } else {
    tune_.scan_parallelism = 4;
  }

  // Hash join partitions: start at 1, auto-scaled per-query via
  // hash_join_partitions(build_side_bytes)
  tune_.hash_join_min_partitions = 1;
  tune_.hash_join_max_partitions = 32;

  // Groupby initial hash table size: ~10% of VRAM worth of slots
  // Each slot is ~16 bytes (key + value pointer)
  tune_.groupby_initial_size = std::max(
    (hw_.total_vram / 10) / 16,    // 10% of VRAM, 16 bytes/slot
    1024ULL * 1024                 // minimum 1M slots
  );
}

void gpu_config::apply_env_overrides() {
  // SIRIUS_MAX_BATCH_BYTES (supports K/M/G suffixes)
  if (auto v = env_size("SIRIUS_MAX_BATCH_BYTES")) {
    tune_.max_batch_bytes = v;
  }

  // SIRIUS_RMM_POOL_SIZE
  if (auto v = env_size("SIRIUS_RMM_POOL_SIZE")) {
    tune_.rmm_pool_size = v;
  }

  // SIRIUS_HASH_JOIN_MIN_PARTITIONS
  if (auto v = env_int("SIRIUS_HASH_JOIN_MIN_PARTITIONS")) {
    tune_.hash_join_min_partitions = v;
  }

  // SIRIUS_HASH_JOIN_MAX_PARTITIONS
  if (auto v = env_int("SIRIUS_HASH_JOIN_MAX_PARTITIONS")) {
    tune_.hash_join_max_partitions = v;
  }

  // SIRIUS_TARGET_CTAS
  if (auto v = env_int("SIRIUS_TARGET_CTAS")) {
    tune_.target_ctas = static_cast<uint32_t>(v);
  }

  // SIRIUS_GROUPBY_INITIAL_SIZE
  if (auto v = env_size("SIRIUS_GROUPBY_INITIAL_SIZE")) {
    tune_.groupby_initial_size = v;
  }

  // SIRIUS_ENABLE_SPILLING
  tune_.enable_spilling = env_bool("SIRIUS_ENABLE_SPILLING", tune_.enable_spilling);

  // SIRIUS_SCAN_PARALLELISM
  if (auto v = env_int("SIRIUS_SCAN_PARALLELISM")) {
    tune_.scan_parallelism = v;
  }

  // SIRIUS_GPU_VRAM_LIMIT — artificially cap reported VRAM (for testing)
  static bool already_capped = false;
  if (!already_capped && (auto v = env_size("SIRIUS_GPU_VRAM_LIMIT"))) {
    already_capped = true;
    hw_.total_vram = std::min(hw_.total_vram, v);
    // Save the env-overridden tuning values that were already applied by the
    // preceding apply_env_overrides() calls. compute_tuning() below recomputes
    // EVERY tuning field from hardware defaults, which silently discards every
    // SIRIUS_* env override set above (the historical bug). Snapshotting and
    // restoring them keeps the overrides intact while still letting the capped
    // VRAM influence derived hardware-dependent fields.
    gpu_tuning_params saved = tune_;
    // Recompute derived params with the capped VRAM
    compute_tuning();
    // Restore the env-driven overrides that compute_tuning() just clobbered.
    tune_.max_batch_bytes        = saved.max_batch_bytes;
    tune_.rmm_pool_size          = saved.rmm_pool_size;
    tune_.hash_join_min_partitions = saved.hash_join_min_partitions;
    tune_.hash_join_max_partitions = saved.hash_join_max_partitions;
    tune_.target_ctas            = saved.target_ctas;
    tune_.groupby_initial_size   = saved.groupby_initial_size;
    tune_.enable_spilling        = saved.enable_spilling;
    tune_.scan_parallelism       = saved.scan_parallelism;
    // Don't call apply_env_overrides() again — env overrides were already
    // applied before this branch and have just been restored above. Since
    // already_capped is now true, we won't re-enter this branch either.
    return;  // avoid infinite recursion
  }
}

std::size_t gpu_config::available_vram() const noexcept {
  std::size_t free_bytes = 0, total_bytes = 0;
  cudaMemGetInfo(&free_bytes, &total_bytes);
  return free_bytes;
}

int gpu_config::hash_join_partitions(std::size_t build_side_bytes) const noexcept {
  // If spilling is disabled or the build fits comfortably, use 1 partition
  // (no spilling overhead).
  std::size_t avail = available_vram();
  if (!tune_.enable_spilling || build_side_bytes <= avail / 3) {
    return tune_.hash_join_min_partitions;
  }

  // Build doesn't fit in 1/3 of available VRAM — partition.
  // Each partition should fit in ~30% of available VRAM.
  std::size_t per_partition_budget = std::max(avail * 3 / 10, std::size_t{128 * 1024 * 1024});
  int partitions = static_cast<int>(
    (build_side_bytes + per_partition_budget - 1) / per_partition_budget);
  partitions = std::max(partitions, tune_.hash_join_min_partitions);
  partitions = std::min(partitions, tune_.hash_join_max_partitions);
  return partitions;
}

}  // namespace sirius::cuda
