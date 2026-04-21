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

#include <cucascade/data/disk_io_backend.hpp>

#include <cudf/io/datasource.hpp>
#include <rmm/cuda_stream_view.hpp>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <future>
#include <memory>

namespace sirius::io {

/// @brief cudf::io::datasource adapter that delegates byte-range reads to a
/// cucascade::idisk_io_backend.
///
/// Reports supports_device_read() == false so cuDF host-stages reads and
/// issues cuda_memcpy_async on the caller's explicit stream through pinned
/// memory. No GDS, no cuFile, no kvikio in Sirius's code path.
///
/// Ownership: the backend is a shared_ptr (typically cached per-GPU on
/// SiriusContext) and must have been constructed under the CUDA context
/// appropriate for the consuming GPU. The adapter itself is single-file:
/// one instance per (backend, path). Callers (parquet_scan_task, iceberg
/// scan) are responsible for backend selection by preferred_device_id.
///
/// Rationale notes (locked by Phase 5 CONTEXT.md + REQUIREMENTS IO-01/02):
///   - supports_device_read() == false is NOT an accidental default — it
///     is load-bearing for multi-GPU safety (IO-02). Do not flip to true.
///   - device_read / device_read_async are intentionally NOT overridden.
///     cuDF reads the flag and never calls them, so the base-class
///     CUDF_FAIL never triggers (see Phase 5 research Open Q5).
///   - Copy/move deleted: the class owns shared_ptr<idisk_io_backend>
///     tied to a specific CUDA context — movability would invite context
///     mistakes across GPU scheduling boundaries.
class cucascade_datasource : public cudf::io::datasource {
 public:
  /// @param backend Per-GPU cucascade::idisk_io_backend (must not be null).
  /// @param path Local filesystem path to the parquet file. Remote URIs
  ///        (s3://, http://, hdfs://) are rejected by the constructor —
  ///        out of scope per PROJECT.md.
  /// @param file_size File size in bytes; typically from
  ///        std::filesystem::file_size(path). Cached for size() queries so
  ///        cuDF can compute footer offsets without re-statting.
  cucascade_datasource(std::shared_ptr<cucascade::idisk_io_backend> backend,
                       std::filesystem::path path,
                       std::size_t file_size);

  ~cucascade_datasource() override;

  cucascade_datasource(cucascade_datasource const&)            = delete;
  cucascade_datasource& operator=(cucascade_datasource const&) = delete;
  cucascade_datasource(cucascade_datasource&&)                 = delete;
  cucascade_datasource& operator=(cucascade_datasource&&)      = delete;

  // ---- Host reads (mandatory cuDF overrides) ----
  // Returned buffer is pinned host memory (allocated from cucascade's
  // fixed_size_host_memory_resource) so cuDF's cuda_memcpy_async stays
  // truly asynchronous (IO-03).
  [[nodiscard]] std::unique_ptr<cudf::io::datasource::buffer> host_read(size_t offset,
                                                                        size_t size) override;

  size_t host_read(size_t offset, size_t size, uint8_t* dst) override;

  // ---- Host reads (async — we override to enqueue on std::launch::async
  //     worker so concurrent reads queue into the backend concurrently) ----
  std::future<std::unique_ptr<cudf::io::datasource::buffer>> host_read_async(size_t offset,
                                                                             size_t size) override;

  std::future<size_t> host_read_async(size_t offset, size_t size, uint8_t* dst) override;

  // ---- Device reads disabled — cuDF takes the host + cuda_memcpy_async
  //     path, which is what gives us multi-GPU safety (IO-02). ----
  [[nodiscard]] bool supports_device_read() const override { return false; }
  [[nodiscard]] bool is_device_read_preferred(size_t /*size*/) const override { return false; }

  // ---- Metadata ----
  [[nodiscard]] size_t size() const override { return _file_size; }

 private:
  std::shared_ptr<cucascade::idisk_io_backend> _backend;
  std::filesystem::path _path;
  std::size_t _file_size;
};

}  // namespace sirius::io
