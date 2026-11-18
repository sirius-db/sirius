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

#include <cucascade/memory/common.hpp>

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <string>

namespace sirius::spill {

/// Why a batch that qualified for spill compression was stored uncompressed.
enum class encode_outcome : std::uint8_t {
  no_headroom,   ///< the encoder's GPU headroom could not be reserved
  small_saving,  ///< the compressed form saved too little, or no column compresses
  failed,        ///< the encode threw (out of device memory, in practice)
};

/// A copy of the counters of io_stats.
struct io_totals {
  // Batches moved down a tier. "stored" is what the target tier holds (the file size for the
  // disk tier), "source" what the batch occupied in the tier it left.
  std::uint64_t to_host_batches{0};
  std::uint64_t to_host_source_bytes{0};
  std::uint64_t to_host_stored_bytes{0};
  std::uint64_t to_disk_batches{0};
  std::uint64_t to_disk_source_bytes{0};
  std::uint64_t to_disk_stored_bytes{0};
  // GPU batches stored compressed: their device size and their compressed size.
  std::uint64_t compressed_batches{0};
  std::uint64_t compressed_input_bytes{0};
  std::uint64_t compressed_output_bytes{0};
  std::uint64_t compress_micros{0};  ///< wall time of those conversions, encode and store
  // Batches that qualified for compression and were stored uncompressed.
  std::uint64_t declined_no_headroom{0};
  std::uint64_t declined_small_saving{0};
  std::uint64_t encode_failed{0};
  std::uint64_t uncompressed_fallback_bytes{0};
  std::uint64_t fallback_micros{0};  ///< wall time of the attempts that were given up
  // Encodes that ran under a reservation: what they reserved, the peak they allocated (scratch
  // and compressed output), and how many of them allocated more than they had reserved.
  std::uint64_t encodes{0};
  std::uint64_t encode_reserved_bytes{0};
  std::uint64_t encode_peak_bytes{0};
  std::uint64_t encodes_over_reservation{0};
  // Encodes that waited for another encode to give headroom back: how many, how many of them
  // got their reservation, and how long they waited.
  std::uint64_t encode_waits{0};
  std::uint64_t encode_waits_granted{0};
  std::uint64_t encode_wait_micros{0};
  // Whole batches converted back to the GPU tier: the bytes read from the tier they were in
  // and the device bytes they decoded to.
  std::uint64_t from_host_batches{0};
  std::uint64_t from_host_stored_bytes{0};
  std::uint64_t from_host_device_bytes{0};
  std::uint64_t from_host_micros{0};
  std::uint64_t from_disk_batches{0};
  std::uint64_t from_disk_stored_bytes{0};
  std::uint64_t from_disk_device_bytes{0};
  std::uint64_t from_disk_micros{0};
  // The part of the two above that was stored compressed and decoded on the way.
  std::uint64_t from_host_compressed_batches{0};
  std::uint64_t from_host_compressed_stored_bytes{0};
  std::uint64_t from_host_compressed_micros{0};
  std::uint64_t from_disk_compressed_batches{0};
  std::uint64_t from_disk_compressed_stored_bytes{0};
  std::uint64_t from_disk_compressed_micros{0};
  // Compressed payloads a row-range reader copied whole to the device.
  std::uint64_t payload_host_loads{0};
  std::uint64_t payload_host_bytes{0};
  std::uint64_t payload_disk_loads{0};
  std::uint64_t payload_disk_bytes{0};
};

/**
 * @brief Process-wide counters of what spilling stores and reads back.
 *
 * The downgrade executor's request lines count the bytes a downgrade frees in the tier it
 * relieves, which for a compressed batch is not what reaches the host pool or the disk, and
 * nothing else counts a batch that is converted back to the GPU. These counters record both
 * sides in stored bytes, so the traffic of a query can be read from one log line
 * (SiriusContext logs and resets them at the end of every query).
 *
 * Row ranges the windowed merge reads in place are not counted here: the merge reports them
 * in its own statistics (refill_bytes_*).
 *
 * Thread-safe; every counter is a relaxed atomic, so a snapshot taken while batches move is
 * not a consistent cut.
 */
class io_stats {
 public:
  [[nodiscard]] static io_stats& get() noexcept;

  /// A batch moved to @p target (HOST or DISK): it occupied @p source_bytes where it was and
  /// occupies @p stored_bytes there. @p compressed_from_gpu marks a GPU batch stored compressed,
  /// which took @p micros to encode and store.
  void record_store(cucascade::memory::Tier target,
                    std::size_t source_bytes,
                    std::size_t stored_bytes,
                    bool compressed_from_gpu,
                    std::uint64_t micros = 0) noexcept;

  /// A batch of @p bytes qualified for compression and is stored uncompressed, after an attempt
  /// that took @p micros.
  void record_uncompressed_fallback(encode_outcome outcome,
                                    std::size_t bytes,
                                    std::uint64_t micros = 0) noexcept;

  /// An encode that reserved @p reserved_bytes of GPU memory allocated @p peak_bytes at most.
  void record_encode(std::size_t reserved_bytes, std::size_t peak_bytes) noexcept;

  /// An encode waited @p micros for headroom and got it, or not.
  void record_encode_wait(std::uint64_t micros, bool granted) noexcept;

  /// A whole batch held in @p source (HOST or DISK) as @p stored_bytes was converted to a GPU
  /// table of @p device_bytes, in @p micros. @p compressed marks a batch that was decoded.
  void record_reload(cucascade::memory::Tier source,
                     std::size_t stored_bytes,
                     std::size_t device_bytes,
                     std::uint64_t micros = 0,
                     bool compressed      = false) noexcept;

  /// A row-range reader copied a compressed payload of @p bytes from @p source to the device.
  void record_payload_load(cucascade::memory::Tier source, std::size_t bytes) noexcept;

  [[nodiscard]] io_totals snapshot() const noexcept;

  /// The counters as they were, set back to zero.
  io_totals take() noexcept;

 private:
  io_stats() = default;

  struct counters {
    std::atomic<std::uint64_t> to_host_batches{0};
    std::atomic<std::uint64_t> to_host_source_bytes{0};
    std::atomic<std::uint64_t> to_host_stored_bytes{0};
    std::atomic<std::uint64_t> to_disk_batches{0};
    std::atomic<std::uint64_t> to_disk_source_bytes{0};
    std::atomic<std::uint64_t> to_disk_stored_bytes{0};
    std::atomic<std::uint64_t> compressed_batches{0};
    std::atomic<std::uint64_t> compressed_input_bytes{0};
    std::atomic<std::uint64_t> compressed_output_bytes{0};
    std::atomic<std::uint64_t> compress_micros{0};
    std::atomic<std::uint64_t> declined_no_headroom{0};
    std::atomic<std::uint64_t> declined_small_saving{0};
    std::atomic<std::uint64_t> encode_failed{0};
    std::atomic<std::uint64_t> uncompressed_fallback_bytes{0};
    std::atomic<std::uint64_t> fallback_micros{0};
    std::atomic<std::uint64_t> encodes{0};
    std::atomic<std::uint64_t> encode_reserved_bytes{0};
    std::atomic<std::uint64_t> encode_peak_bytes{0};
    std::atomic<std::uint64_t> encodes_over_reservation{0};
    std::atomic<std::uint64_t> encode_waits{0};
    std::atomic<std::uint64_t> encode_waits_granted{0};
    std::atomic<std::uint64_t> encode_wait_micros{0};
    std::atomic<std::uint64_t> from_host_batches{0};
    std::atomic<std::uint64_t> from_host_stored_bytes{0};
    std::atomic<std::uint64_t> from_host_device_bytes{0};
    std::atomic<std::uint64_t> from_host_micros{0};
    std::atomic<std::uint64_t> from_disk_batches{0};
    std::atomic<std::uint64_t> from_disk_stored_bytes{0};
    std::atomic<std::uint64_t> from_disk_device_bytes{0};
    std::atomic<std::uint64_t> from_disk_micros{0};
    std::atomic<std::uint64_t> from_host_compressed_batches{0};
    std::atomic<std::uint64_t> from_host_compressed_stored_bytes{0};
    std::atomic<std::uint64_t> from_host_compressed_micros{0};
    std::atomic<std::uint64_t> from_disk_compressed_batches{0};
    std::atomic<std::uint64_t> from_disk_compressed_stored_bytes{0};
    std::atomic<std::uint64_t> from_disk_compressed_micros{0};
    std::atomic<std::uint64_t> payload_host_loads{0};
    std::atomic<std::uint64_t> payload_host_bytes{0};
    std::atomic<std::uint64_t> payload_disk_loads{0};
    std::atomic<std::uint64_t> payload_disk_bytes{0};
  };

  template <bool Reset>
  io_totals read() noexcept;

  counters _c;
};

/// Microseconds from @p start to now, for the timed records of io_stats.
[[nodiscard]] inline std::uint64_t micros_since(
  std::chrono::steady_clock::time_point start) noexcept
{
  return static_cast<std::uint64_t>(
    std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - start)
      .count());
}

/// One line with every counter of @p totals, as "key=value" pairs.
[[nodiscard]] std::string format_io_totals(const io_totals& totals);

}  // namespace sirius::spill
