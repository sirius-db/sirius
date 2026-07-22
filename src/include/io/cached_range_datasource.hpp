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

#include <cudf/io/datasource.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <io/sirius_datasource.hpp>
#include <op/scan/cached_ranges.hpp>

#include <future>
#include <memory>

namespace sirius::io {

// ---------------------------------------------------------------------------
// cached_range_datasource
// ---------------------------------------------------------------------------

/**
 * @brief A @c cudf::io::datasource that serves reads from pinned host buffers.
 *
 * Backs the parquet pin tier: at pin time a table's column-chunk byte ranges are
 * read once from disk into pinned host memory and packed into a
 * @c op::scan::cache_ranges. At query time this datasource fronts the decode:
 * every @c host_read whose @c [offset, size) is fully covered by the cached
 * ranges is answered by @c memcpy from pinned memory; any read that falls
 * outside the cache (the parquet footer, page index, or a column that was not
 * pinned) is delegated to the @p fallback @c sirius_datasource, which reads it
 * from the file.
 *
 * Device reads are disabled (@c supports_device_read returns false) so cuDF
 * always takes the host path — the cached read is a pinned-memory copy and cuDF
 * performs the host-to-device transfer itself.
 */
class cached_range_datasource : public cudf::io::datasource {
 public:
  cached_range_datasource(std::shared_ptr<op::scan::cache_ranges> ranges,
                          std::shared_ptr<sirius_datasource> fallback);

  ~cached_range_datasource() override = default;

  cached_range_datasource(cached_range_datasource const&)            = delete;
  cached_range_datasource& operator=(cached_range_datasource const&) = delete;

  [[nodiscard]] size_t size() const override;

  // The pinned buffers are CUDA-pinned host memory, so device reads are served
  // by an H2D copy — matching sirius_datasource, whose device-read preference the
  // cuDF GPU parquet decoder relies on (the host-only path crashes it).
  [[nodiscard]] bool supports_device_read() const override { return true; }

  [[nodiscard]] bool is_device_read_preferred(size_t) const override { return true; }

  size_t host_read(size_t offset, size_t size, uint8_t* dst) override;

  std::unique_ptr<datasource::buffer> host_read(size_t offset, size_t size) override;

  std::unique_ptr<datasource::buffer> device_read(size_t offset,
                                                  size_t size,
                                                  rmm::cuda_stream_view stream) override;

  size_t device_read(size_t offset,
                     size_t size,
                     uint8_t* dst,
                     rmm::cuda_stream_view stream) override;

  std::future<size_t> device_read_async(size_t offset,
                                        size_t size,
                                        uint8_t* dst,
                                        rmm::cuda_stream_view stream) override;

 private:
  std::shared_ptr<op::scan::cache_ranges> _ranges;
  std::shared_ptr<sirius_datasource> _fallback;
};

}  // namespace sirius::io
