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

#include "downgrade/spill_io_stats.hpp"

#include <cudf/column/column_view.hpp>

#include <cuda/stream>

#include <cucascade/data/representation_converter.hpp>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

/**
 * Spill compression: a GPU batch the downgrade executor moves to the host or disk
 * tier is stored Simpatico-compressed instead of raw, and decoded on its way back
 * to the GPU.
 *
 * Every column gets a plan chosen from its type and a prefix sample
 * (default_spill_plan). The compressed form is kept only when it saves enough
 * (spill_max_compressed_fraction); otherwise, and on any failure, the batch spills
 * uncompressed exactly as it would with the feature off. See
 * convertible_data_batch::convert for the hook.
 */
namespace sirius {
struct compression_config;
}  // namespace sirius

namespace sirius::compression {

struct spill_compression_settings {
  /// Master switch (compression.enable_spill_compression).
  bool enabled{false};
  /// Batches smaller than this spill uncompressed: the encode's fixed costs are not
  /// worth paying for them (compression.spill_min_batch_bytes).
  std::size_t min_batch_bytes{64ULL * 1024 * 1024};
  /// Keep the compressed form only when header + payload is at most this fraction of
  /// the batch's device size (compression.spill_max_compressed_fraction).
  double max_compressed_fraction{0.75};
  /// Encode only when this fraction of the batch's size could be reserved in its GPU
  /// memory space -- the headroom the encoder needs. Not grantable means the batch
  /// spills uncompressed. 0 skips the check (compression.spill_encode_reserve_fraction).
  double encode_reserve_fraction{0.5};
  /// Largest compressed payload a row-range reader copies whole to the device before it
  /// decodes row ranges of it; 0 disables the copy (compression.spill_read_cache_max_bytes).
  std::size_t read_cache_max_bytes{64ULL * 1024 * 1024};
  /// Largest decoded size of a compressed batch the windowed merge decodes whole when it has at
  /// most two windows of rows; 0 always reads by row range
  /// (compression.spill_whole_decode_max_bytes).
  std::size_t whole_decode_max_bytes{256ULL * 1024 * 1024};
  /// Longest an encode whose headroom is not grantable waits for another encode on the same
  /// GPU to give its headroom back before the batch spills uncompressed, in milliseconds;
  /// 0 never waits (compression.spill_encode_wait_ms).
  std::uint64_t encode_wait_ms{1000};
  /// Batches of the same column types that one set of sampled plans encodes before the
  /// columns are sampled again, the sampled batch included. Sampling reads a prefix of every
  /// column back to the host, which costs a spilled batch more than the launches that encode
  /// it. The plans of a batch are dropped early when a later batch compresses clearly worse
  /// with them, and a batch that would be declined with them is encoded again with its own.
  /// 1 samples every batch (compression.spill_plan_reuse_batches).
  std::uint64_t plan_reuse_batches{8};
};

/// Install the process-wide settings. The converters run on downgrade threads with no
/// client context, so they read these rather than a session setting.
void set_spill_compression_settings(spill_compression_settings const& settings);

[[nodiscard]] spill_compression_settings get_spill_compression_settings();

/// Install the settings @p config describes (the compression section of the Sirius
/// configuration).
void apply_spill_compression_config(compression_config const& config);

/// Whether a GPU batch of @p bytes should be offered to the spill compressor.
[[nodiscard]] bool spill_compression_applies(std::size_t bytes);

/**
 * @brief Automatic per-column plan shared by spilling and pinning, chosen from type and sample.
 *
 * Fixed-width columns (including decimals and chrono types) up to 8 GiB are bitpacked;
 * larger ones exceed bitpack's encode scratch and are stored as is. Integer-valued
 * columns up to 64 bits (integers, decimals, chrono types) instead use delta -> bitpack
 * or delta -> rle -> bitpack when a size estimate on a prefix sample says that saves at
 * least 10%. A string
 * column of at least 4096 rows is dictionary-encoded, with bitpacked indices, when a
 * sample of it has few distinct values; otherwise it is split into ans-coded chars and
 * delta-bitpacked offsets. Shorter string columns and every other type are stored as is.
 *
 * Always samples @p column. The spill converters call it through a per-schema memory of the
 * last sampled plans instead (spill_compression_settings::plan_reuse_batches).
 */
[[nodiscard]] std::string default_spill_plan(cudf::column_view const& column,
                                             ::cuda::stream_ref stream);

namespace detail {

/// Mark the start and the end of a spill encode on GPU @p device that holds a headroom
/// reservation. An encode whose headroom is not grantable waits for the end of one (see
/// spill_compression_settings::encode_wait_ms). Called by the encoder; exposed for tests.
void note_encode_started(int device);
void note_encode_finished(int device) noexcept;

/// Blocks until an encode on GPU @p device ends. Returns false, without waiting, when none is
/// in flight there, and false at @p deadline.
bool wait_for_encode_end(int device, std::chrono::steady_clock::time_point deadline);

}  // namespace detail

/// Thrown by a spill converter that chose not to compress (too small a saving, no
/// encode headroom). The caller spills the batch uncompressed instead.
class spill_compression_declined : public std::runtime_error {
 public:
  explicit spill_compression_declined(
    std::string const& what, spill::encode_outcome outcome = spill::encode_outcome::small_saving)
    : std::runtime_error(what), _outcome(outcome)
  {
  }

  /// Why the batch was not compressed, for the spill counters.
  [[nodiscard]] spill::encode_outcome outcome() const noexcept { return _outcome; }

 private:
  spill::encode_outcome _outcome;
};

}  // namespace sirius::compression

namespace sirius {

/**
 * @brief Register the spill converters: gpu_table to compressed_host and to
 * compressed_disk, and compressed_host to compressed_disk.
 *
 * The decode side (compressed_host and compressed_disk to gpu_table) is registered
 * by register_compression_converters().
 */
void register_spill_compression_converters(cucascade::representation_converter_registry& registry);

}  // namespace sirius
