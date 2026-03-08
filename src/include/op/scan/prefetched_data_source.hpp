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

#include "op/scan/cached_ranges.hpp"

#include <cudf/io/datasource.hpp>

#include <future>
#include <memory>

namespace sirius::op::scan {

class prefetched_data_source : public cudf::io::datasource {
 public:
  // If fallback_source is provided, any offset that is not covered by cache_ranges will be
  // fetched from fallback_source instead of throwing.
  explicit prefetched_data_source(std::unique_ptr<cache_ranges> ranges,
                                  std::shared_ptr<cudf::io::datasource> fallback_source = nullptr);

  ~prefetched_data_source() override;

  [[nodiscard]] std::unique_ptr<cudf::io::datasource::buffer> host_read(size_t offset,
                                                                        size_t size) override;

  size_t host_read(size_t offset, size_t size, uint8_t* dst) override;

  [[nodiscard]] bool supports_device_read() const override { return true; }

  [[nodiscard]] bool is_device_read_preferred(size_t size) const override { return true; }

  [[nodiscard]] std::unique_ptr<cudf::io::datasource::buffer> device_read(
    size_t offset, size_t size, rmm::cuda_stream_view stream) override;

  size_t device_read(size_t offset,
                     size_t size,
                     uint8_t* dst,
                     rmm::cuda_stream_view stream) override;

  std::future<size_t> device_read_async(size_t offset,
                                        size_t size,
                                        uint8_t* dst,
                                        rmm::cuda_stream_view stream) override;

  [[nodiscard]] size_t size() const override;

 private:
  struct copy_result {
    size_t bytes;
    cudaStream_t stream_used;
  };

  copy_result enqueue_device_copies(size_t offset,
                                    size_t size,
                                    uint8_t* dst,
                                    rmm::cuda_stream_view stream);

  std::unique_ptr<cache_ranges> ranges_;
  std::shared_ptr<cudf::io::datasource> fallback_;
  std::atomic<size_t> total_bytes_read_from_cache_{0};
  std::atomic<size_t> total_bytes_read_from_fallback_{0};
};

}  // namespace sirius::op::scan
