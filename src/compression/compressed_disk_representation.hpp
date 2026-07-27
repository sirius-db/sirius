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

#include <cucascade/data/common.hpp>
#include <cucascade/memory/memory_space.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace sirius {

/**
 * @brief DISK-tier idata_representation backed by a Simpatico .hpln file.
 *
 * Owns a path to a compressed `.hpln` file written by either:
 *  - `simpatico::write_compressed_table()` (for GPU→DISK direct spill), or
 *  - a raw flush of a `pinned_compressed_blob` (for HOST→DISK cascade).
 *
 * File ownership is shared: the file is unlinked when the last owner of
 * `_owns_file` is destroyed (RAII via `shared_ptr<bool>`). Multiple
 * representations may share the same file path after `clone()`.
 *
 * The `compressed_disk_representation → gpu_table_representation` converter
 * calls `simpatico::read_compressed_table(path, stream, mr)` then
 * `simpatico::decompress()`, optionally projecting to `_selected_indices`.
 */
class compressed_disk_representation : public cucascade::idata_representation {
 public:
  /**
   * @param memory_space        DISK memory space this data is associated with.
   * @param path                Absolute path to the .hpln file on disk.
   * @param compressed_bytes    File size / compressed footprint in bytes.
   * @param uncompressed_bytes  Original device footprint (cudf alloc_size).
   * @param num_rows            Row count.
   * @param column_names        Column names in schema order.
   */
  compressed_disk_representation(cucascade::memory::memory_space& memory_space,
                                 std::string path,
                                 std::size_t compressed_bytes,
                                 std::size_t uncompressed_bytes,
                                 std::int64_t num_rows,
                                 std::vector<std::string> column_names);

  ~compressed_disk_representation() override;

  compressed_disk_representation(const compressed_disk_representation&)            = delete;
  compressed_disk_representation& operator=(const compressed_disk_representation&) = delete;
  compressed_disk_representation(compressed_disk_representation&&)                 = delete;
  compressed_disk_representation& operator=(compressed_disk_representation&&)      = delete;

  // ── idata_representation interface ──────────────────────────────────────────

  [[nodiscard]] std::size_t get_size_in_bytes() const override { return _compressed_bytes; }

  [[nodiscard]] std::size_t get_uncompressed_data_size_in_bytes() const override
  {
    return _uncompressed_bytes;
  }

  /// Clone shares the same backing file (increments shared ownership; unlink deferred).
  [[nodiscard]] std::unique_ptr<cucascade::idata_representation> clone(
    rmm::cuda_stream_view stream) override;

  // ── Projection ──────────────────────────────────────────────────────────────

  [[nodiscard]] std::unique_ptr<compressed_disk_representation> select_columns(
    const std::vector<std::size_t>& indices) const;

  // ── Accessors ───────────────────────────────────────────────────────────────

  [[nodiscard]] const std::string& path() const noexcept { return *_path; }

  [[nodiscard]] const std::vector<std::string>& column_names() const noexcept
  {
    return _column_names;
  }

  [[nodiscard]] std::int64_t num_rows() const noexcept { return _num_rows; }

  [[nodiscard]] const std::optional<std::vector<std::size_t>>& selected_indices() const noexcept
  {
    return _selected_indices;
  }

 private:
  // Private constructor for clone/select_columns (shares path + owns_file).
  compressed_disk_representation(cucascade::memory::memory_space& memory_space,
                                 std::shared_ptr<std::string> path,
                                 std::shared_ptr<bool> owns_file,
                                 std::size_t compressed_bytes,
                                 std::size_t uncompressed_bytes,
                                 std::int64_t num_rows,
                                 std::vector<std::string> column_names,
                                 std::optional<std::vector<std::size_t>> selected_indices);

  // Shared ownership: the file is unlinked when _owns_file.use_count() drops to 1
  // (i.e., this is the last owner).
  std::shared_ptr<std::string> _path;
  std::shared_ptr<bool> _owns_file;

  std::size_t _compressed_bytes;
  std::size_t _uncompressed_bytes;
  std::int64_t _num_rows;
  std::vector<std::string> _column_names;
  std::optional<std::vector<std::size_t>> _selected_indices;
};

}  // namespace sirius
