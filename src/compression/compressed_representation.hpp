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

#include <rmm/cuda_stream_view.hpp>

#include <cucascade/data/common.hpp>
#include <cucascade/memory/memory_space.hpp>

#include <cstddef>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace sirius {

/**
 * @brief HOST-tier idata_representation backed by a Simpatico-compressed .hpln file.
 *
 * A chunk compressed with simpatico::compress_with_plan is serialised to a file
 * (typically on /dev/shm for near-RAM bandwidth) via write_compressed_table.
 * This representation stores the path, schema metadata, and an optional column
 * projection; the actual bytes live on the filesystem until the last shared owner
 * is destroyed, at which point the file is unlinked.
 *
 * Multiple compressed_host_representation objects may share the same underlying
 * file (e.g. after select_columns() or clone()). The file is unlinked when the
 * last owner's destructor fires.
 *
 * The converter compressed_host_representation → gpu_table_representation reads
 * the file with read_compressed_table, projects to the selected columns (if any),
 * then decompresses to a cudf::table. It is registered by
 * register_compression_converters().
 */
class compressed_host_representation : public cucascade::idata_representation {
 public:
  /**
   * @brief Construct a compressed_host_representation that owns the backing file.
   *
   * @param memory_space        Host memory space this data is logically associated with.
   * @param path                Absolute path to the serialised .hpln file.
   * @param column_names        All column names stored in the file, in column order.
   * @param compressed_bytes    File size (compressed footprint in bytes).
   * @param uncompressed_bytes  Logical uncompressed size (sum of raw column bytes).
   * @param num_rows            Row count.
   */
  compressed_host_representation(cucascade::memory::memory_space& memory_space,
                                 std::string path,
                                 std::vector<std::string> column_names,
                                 std::size_t compressed_bytes,
                                 std::size_t uncompressed_bytes,
                                 std::int64_t num_rows);

  ~compressed_host_representation() override;

  // Non-copyable (shared ownership uses shared_ptr)
  compressed_host_representation(const compressed_host_representation&)            = delete;
  compressed_host_representation& operator=(const compressed_host_representation&) = delete;
  compressed_host_representation(compressed_host_representation&&)                 = delete;
  compressed_host_representation& operator=(compressed_host_representation&&)      = delete;

  // ── idata_representation interface ──────────────────────────────────────────

  /// Returns the on-disk (compressed) file size.
  [[nodiscard]] std::size_t get_size_in_bytes() const override { return _compressed_bytes; }

  /// Returns the logical (uncompressed) data size.
  [[nodiscard]] std::size_t get_uncompressed_data_size_in_bytes() const override
  {
    return _uncompressed_bytes;
  }

  /// Clone shares the same backing file (increments shared ownership).
  [[nodiscard]] std::unique_ptr<cucascade::idata_representation> clone(
    rmm::cuda_stream_view stream) override;

  // ── Projection ──────────────────────────────────────────────────────────────

  /**
   * @brief Return a projection that exposes only the requested column indices.
   *
   * The returned representation shares the same backing file.  The converter
   * will read all columns but decompress only the selected subset.
   *
   * @param indices  Indices into column_names() to expose (must be valid).
   */
  [[nodiscard]] std::unique_ptr<compressed_host_representation> select_columns(
    std::span<const std::size_t> indices) const;

  // ── Accessors ───────────────────────────────────────────────────────────────

  [[nodiscard]] const std::string& path() const noexcept { return *_path; }
  [[nodiscard]] const std::vector<std::string>& column_names() const noexcept
  {
    return _column_names;
  }
  [[nodiscard]] std::int64_t num_rows() const noexcept { return _num_rows; }

  /// Column indices to project during decompression (nullopt = all columns).
  [[nodiscard]] const std::optional<std::vector<std::size_t>>& selected_indices() const noexcept
  {
    return _selected_indices;
  }

 private:
  /// Construct a projection sharing the same backing file.
  compressed_host_representation(cucascade::memory::memory_space& memory_space,
                                 std::shared_ptr<std::string> shared_path,
                                 std::shared_ptr<bool> owns_file,
                                 std::vector<std::string> column_names,
                                 std::size_t compressed_bytes,
                                 std::size_t uncompressed_bytes,
                                 std::int64_t num_rows,
                                 std::optional<std::vector<std::size_t>> selected_indices);

  std::shared_ptr<std::string> _path;
  /// Tracks whether the file should be unlinked on destruction.
  /// Shared among all representations that alias the same file; only the
  /// last one resets the flag to false before unlinking.
  std::shared_ptr<bool> _owns_file;
  std::vector<std::string> _column_names;
  std::size_t _compressed_bytes;
  std::size_t _uncompressed_bytes;
  std::int64_t _num_rows;
  std::optional<std::vector<std::size_t>> _selected_indices;
};

}  // namespace sirius
