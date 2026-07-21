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

//! @file gpu_config.hpp
//! @brief Runtime GPU capability detection and adaptive tuning for AMD ROCm.
//!
//! The engine queries GPU hardware capabilities at startup and derives tuning
//! parameters that adapt to any AMD GPU — from MI300 (192GB VRAM) to consumer
//! RDNA3 cards (8GB VRAM). All parameters can be overridden via environment
//! variables for manual tuning.
//!
//! Usage:
//!   auto const& cfg = sirius::cuda::gpu_config::instance();
//!   size_t batch = cfg.max_batch_bytes();
//!   int partitions = cfg.hash_join_partitions(build_side_bytes);

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace sirius::cuda {

/// GPU hardware capabilities queried once at startup via HIP API.
struct gpu_hardware_info {
  int device_id{-1};
  std::size_t total_vram{0};           ///< Total GPU memory in bytes
  std::size_t l2_cache_size{0};        ///< L2 cache in bytes
  int multiprocessor_count{0};         ///< Compute units (CUs)
  int max_threads_per_sm{0};           ///< Max threads per compute unit
  int max_shared_mem_per_block{0};     ///< Shared memory per block (bytes)
  int warp_size{64};                   ///< Wavefront size (64 on AMD)
  int memory_clock_khz{0};             ///< Memory clock rate (kHz)
  int bus_width_bits{0};               ///< Memory bus width (bits)
  std::size_t memory_bandwidth_gbs{0}; ///< Estimated bandwidth (GB/s)
  std::string device_name;             ///< GPU name (e.g. "AMD Instinct MI300")
  int major{0};                        ///< Compute capability major
  int minor{0};                        ///< Compute capability minor
};

/// Adaptive tuning parameters derived from hardware info.
/// All values can be overridden via environment variables.
struct gpu_tuning_params {
  /// Maximum bytes for a single Parquet read batch.
  /// Default: 25% of VRAM. Override: SIRIUS_MAX_BATCH_BYTES
  std::size_t max_batch_bytes{0};

  /// RMM memory pool size. Default: 75% of VRAM.
  /// Override: SIRIUS_RMM_POOL_SIZE (bytes)
  std::size_t rmm_pool_size{0};

  /// Minimum hash join partitions. Default: 1 (no partitioning on large VRAM).
  /// Automatically increased when build side exceeds 30% of VRAM.
  /// Override: SIRIUS_HASH_JOIN_MIN_PARTITIONS
  int hash_join_min_partitions{1};

  /// Maximum hash join partitions (safety cap).
  /// Override: SIRIUS_HASH_JOIN_MAX_PARTITIONS
  int hash_join_max_partitions{32};

  /// Target CTA count for kernel launch sizing.
  /// Default: 2x SM count × occupancy. Override: SIRIUS_TARGET_CTAS
  uint32_t target_ctas{0};

  /// Groupby hash table initial size (number of slots).
  /// Default: estimated based on VRAM. Override: SIRIUS_GROUPBY_INITIAL_SIZE
  std::size_t groupby_initial_size{0};

  /// Whether to enable spill-to-host on OOM.
  /// Default: true on GPUs with <64GB VRAM, false on larger.
  /// Override: SIRIUS_ENABLE_SPILLING=1/0
  bool enable_spilling{true};

  /// Concurrent scan parallelism (number of parallel Parquet reads).
  /// Default: 4 on <16GB, 8 on <64GB, 16 on larger.
  /// Override: SIRIUS_SCAN_PARALLELISM
  int scan_parallelism{4};

  /// VRAM threshold below which spilling is enabled (bytes).
  std::size_t spilling_threshold{64ULL * 1024 * 1024 * 1024};  // 64GB
};

/// Singleton that detects GPU capabilities at startup and provides
/// adaptive tuning parameters. Thread-safe after initialization.
class gpu_config {
 public:
  /// Get the global config instance. First call initializes it.
  static gpu_config const& instance();

  // --- Hardware info ---
  gpu_hardware_info const& hardware() const noexcept { return hw_; }

  // --- Tuning parameters ---
  gpu_tuning_params const& tuning() const noexcept { return tune_; }

  // --- Convenience accessors ---
  std::size_t total_vram() const noexcept { return hw_.total_vram; }
  std::size_t available_vram() const noexcept;

  /// Maximum Parquet read batch size (bytes). Considers available VRAM.
  std::size_t max_batch_bytes() const noexcept { return tune_.max_batch_bytes; }

  /// RMM pool size (bytes).
  std::size_t rmm_pool_size() const noexcept { return tune_.rmm_pool_size; }

  /// Hash join partition count for a given build-side size.
  /// Returns 1 if the build fits in VRAM; more if it needs spilling.
  int hash_join_partitions(std::size_t build_side_bytes) const noexcept;

  /// Target CTA count for kernel launches.
  uint32_t target_ctas() const noexcept { return tune_.target_ctas; }

  /// Whether spill-to-host is enabled.
  bool enable_spilling() const noexcept { return tune_.enable_spilling; }

  /// Scan parallelism level.
  int scan_parallelism() const noexcept { return tune_.scan_parallelism; }

  /// Groupby hash table initial size.
  std::size_t groupby_initial_size() const noexcept { return tune_.groupby_initial_size; }

 private:
  gpu_config();

  void detect_hardware();
  void compute_tuning();
  void apply_env_overrides();

  gpu_hardware_info hw_;
  gpu_tuning_params tune_;
  bool initialized_{false};
};

}  // namespace sirius::cuda
