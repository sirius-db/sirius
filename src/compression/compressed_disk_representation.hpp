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

#include "simpatico_compressed_representation.hpp"

#include <cuda/stream>

#include <cucascade/memory/memory_space.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

namespace sirius {

/**
 * @brief DISK-tier idata_representation backed by a Simpatico-compressed .hpln file.
 *
 * The file is a complete .hpln image: the header produced by
 * simpatico::build_compressed_table_header, followed by the payload at
 * header_bytes(). Clones share the file, which is deleted when the last
 * representation referring to it is destroyed.
 *
 * Produced by the spill converters (gpu_table and compressed_host to this) and
 * decoded by the compressed_disk to gpu_table converter; see
 * register_compression_converters().
 */
class compressed_disk_representation : public simpatico_compressed_representation {
 public:
  /**
   * @param memory_space        Disk memory space the file lives in.
   * @param path                The .hpln file. Ownership passes to this representation.
   * @param header_bytes        Size of the file's header; the payload starts here.
   * @param compressed_bytes    Size of the file in bytes.
   * @param uncompressed_bytes  Device footprint of the table the file decodes to.
   * @param num_rows            Row count.
   */
  compressed_disk_representation(cucascade::memory::memory_space& memory_space,
                                 std::string path,
                                 std::size_t header_bytes,
                                 std::size_t compressed_bytes,
                                 std::size_t uncompressed_bytes,
                                 std::int64_t num_rows);

  ~compressed_disk_representation() override = default;

  compressed_disk_representation(const compressed_disk_representation&)            = delete;
  compressed_disk_representation& operator=(const compressed_disk_representation&) = delete;
  compressed_disk_representation(compressed_disk_representation&&)                 = delete;
  compressed_disk_representation& operator=(compressed_disk_representation&&)      = delete;

  [[nodiscard]] std::size_t get_size_in_bytes() const override { return _compressed_bytes; }

  [[nodiscard]] std::size_t get_uncompressed_data_size_in_bytes() const override
  {
    return _uncompressed_bytes;
  }

  /// Shares the file rather than copying it: the bytes on disk are immutable.
  [[nodiscard]] std::unique_ptr<cucascade::idata_representation> clone(
    ::cuda::stream_ref stream) const override;

  [[nodiscard]] std::string const& path() const noexcept { return _file->path; }

  [[nodiscard]] std::size_t header_bytes() const noexcept { return _header_bytes; }

  [[nodiscard]] std::int64_t num_rows() const noexcept override { return _num_rows; }

 private:
  /// Deletes the file when the last owner goes away.
  struct owned_file {
    std::string path;
    explicit owned_file(std::string p) : path(std::move(p)) {}
    ~owned_file() noexcept;
    owned_file(const owned_file&)            = delete;
    owned_file& operator=(const owned_file&) = delete;
  };

  compressed_disk_representation(cucascade::memory::memory_space& memory_space,
                                 std::shared_ptr<owned_file> file,
                                 std::size_t header_bytes,
                                 std::size_t compressed_bytes,
                                 std::size_t uncompressed_bytes,
                                 std::int64_t num_rows);

  std::shared_ptr<owned_file> _file;
  std::size_t _header_bytes;
  std::size_t _compressed_bytes;
  std::size_t _uncompressed_bytes;
  std::int64_t _num_rows;
};

}  // namespace sirius
