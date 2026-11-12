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

#include "query_id.hpp"

#include <cudf/column/column_view.hpp>

#include <cuda/stream>

#include <cucascade/data/representation_converter.hpp>

#include <cstddef>
#include <optional>
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
  /// memory space -- the headroom the encoder needs beyond the batch. Not grantable
  /// means the batch spills uncompressed. 0 skips the check
  /// (compression.spill_encode_reserve_fraction).
  double encode_reserve_fraction{1.5};
  /// Encode the strings of a batch with a known source (spill_source) with an OnPair
  /// dictionary trained per column in the background; until one is ready, and for
  /// one-byte strings, the column takes its default plan
  /// (compression.spill_onpair_strings).
  bool onpair_strings{true};
  /// Free each column of a spilled batch as soon as it is encoded, so the encode adds one
  /// column's working set to the device rather than the batch's compressed form. The batch
  /// then has no uncompressed form to fall back to: it is kept compressed whatever its
  /// ratio, and a column whose encode fails is stored as is. Applies under per-thread
  /// allocation tracking to batches that own their table
  /// (compression.spill_release_columns_early).
  bool release_columns_early{false};
};

/// The query-graph edge a spilled batch came from: the repository holding it, within
/// its query. Keys the per-column state the spill path keeps across batches.
struct spill_source {
  query_id_t query{};
  void const* repository{nullptr};
};

/// The source of the batch the calling thread is spilling, if any.
[[nodiscard]] std::optional<spill_source> current_spill_source() noexcept;

/// Makes @p source the calling thread's current_spill_source() for its lifetime. The
/// converter signature has no room for it, and a convert runs on the calling thread.
class scoped_spill_source {
 public:
  explicit scoped_spill_source(std::optional<spill_source> source) noexcept;
  ~scoped_spill_source();
  scoped_spill_source(scoped_spill_source const&)            = delete;
  scoped_spill_source& operator=(scoped_spill_source const&) = delete;

 private:
  std::optional<spill_source> _previous;
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
 * @brief The plan a spilled column is compressed with, chosen from its type and a sample.
 *
 * Fixed-width columns (including decimals and chrono types) up to 8 GiB are bitpacked;
 * larger ones exceed bitpack's encode scratch and are stored as is. Integer-valued
 * columns up to 64 bits (integers, decimals, chrono types) instead use delta -> bitpack
 * or delta -> rle -> bitpack when a size estimate on a prefix sample says that saves at
 * least 10%. A string
 * column of at least 4096 rows is dictionary-encoded, with bitpacked indices, when a
 * sample of it has few distinct values and its values average more than one byte;
 * otherwise it is split into ans-coded chars and delta-bitpacked offsets. Shorter string columns
 * and every other type are stored as is. A batch with a spill_source may instead encode
 * a string column with OnPair; see spill_compression_settings::onpair_strings.
 */
[[nodiscard]] std::string default_spill_plan(cudf::column_view const& column,
                                             ::cuda::stream_ref stream);

/// Thrown by a spill converter that chose not to compress (too small a saving, no
/// encode headroom). The caller spills the batch uncompressed instead.
class spill_compression_declined : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

/// Thrown by a spill converter that failed after it took the batch's columns
/// (release_columns_early): the batch cannot be spilled uncompressed instead.
class spill_source_released : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
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
