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
 * @brief DISK-tier idata_representation backed by Simpatico .hpln file(s).
 *
 * Comes in two forms, mirroring @ref compressed_host_representation:
 *
 *  - **whole-table**: one `.hpln` holding every column, written by either
 *    `simpatico::write_compressed_table()` (GPU→DISK direct spill) or a raw
 *    flush of a `pinned_compressed_blob` (HOST→DISK cascade). `path()`.
 *  - **per-column**: one complete 1-column `.hpln` per column, in schema order,
 *    produced by flushing a per-column `compressed_host_representation` whose
 *    bytes live in `column_blobs()`. `column_paths()`; `is_per_column()`.
 *
 * File ownership is shared: every file listed in `_files` is unlinked when the
 * last owner of `_owns_file` is destroyed (RAII via `shared_ptr<bool>`).
 * Multiple representations may share the same files after `clone()`.
 *
 * The `compressed_disk_representation → gpu_table_representation` converter
 * calls `simpatico::read_compressed_table(path, stream, mr)` then
 * `simpatico::decompress()`, optionally projecting to `_selected_indices`; for
 * the per-column form it reads and decodes one file per selected column and
 * assembles the batch from the results.
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

  /**
   * @brief Construct the per-column form: one 1-column .hpln per column.
   *
   * @param column_paths       One path per column, in schema order; must be the
   *                           same length as @p column_names.
   * @param max_artifact_bytes Size of the largest single artifact — the decode
   *                           transient, since the files are read one at a time.
   */
  compressed_disk_representation(cucascade::memory::memory_space& memory_space,
                                 std::vector<std::string> column_paths,
                                 std::size_t compressed_bytes,
                                 std::size_t max_artifact_bytes,
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

  /// The whole-table artifact. Only valid when !is_per_column().
  [[nodiscard]] const std::string& path() const noexcept { return _files->front(); }

  /// True when the batch is stored as one .hpln per column; see the class docs.
  [[nodiscard]] bool is_per_column() const noexcept { return _per_column; }

  /// The per-column artifacts, in schema order. Empty unless is_per_column().
  [[nodiscard]] const std::vector<std::string>& column_paths() const noexcept
  {
    static const std::vector<std::string> kNone;
    return _per_column ? *_files : kNone;
  }

  /// Compressed bytes that decoding this batch stages on the device at once: the
  /// whole file for the whole-table form, the largest single artifact for the
  /// per-column form, which reads one file at a time. Feeds
  /// estimated_materialization_bytes().
  [[nodiscard]] std::size_t decode_transient_bytes() const noexcept
  {
    return _per_column ? _max_artifact_bytes : _compressed_bytes;
  }

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
  // Private constructor for clone/select_columns (shares files + owns_file).
  compressed_disk_representation(cucascade::memory::memory_space& memory_space,
                                 std::shared_ptr<std::vector<std::string>> files,
                                 bool per_column,
                                 std::shared_ptr<bool> owns_file,
                                 std::size_t compressed_bytes,
                                 std::size_t max_artifact_bytes,
                                 std::size_t uncompressed_bytes,
                                 std::int64_t num_rows,
                                 std::vector<std::string> column_names,
                                 std::optional<std::vector<std::size_t>> selected_indices);

  // Shared ownership: the files are unlinked when _owns_file.use_count() drops to
  // 1 (i.e., this is the last owner). One entry for the whole-table form, one per
  // column for the per-column form.
  std::shared_ptr<std::vector<std::string>> _files;
  bool _per_column = false;
  std::shared_ptr<bool> _owns_file;

  std::size_t _compressed_bytes;
  /// Largest single artifact; equals _compressed_bytes for the whole-table form.
  std::size_t _max_artifact_bytes;
  std::size_t _uncompressed_bytes;
  std::int64_t _num_rows;
  std::vector<std::string> _column_names;
  std::optional<std::vector<std::size_t>> _selected_indices;
};

}  // namespace sirius
