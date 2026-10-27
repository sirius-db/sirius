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

#include "compressed_disk_representation.hpp"

#include <log/logging.hpp>

#include <filesystem>
#include <system_error>
#include <utility>

namespace sirius {

compressed_disk_representation::owned_file::~owned_file() noexcept
{
  std::error_code ec;
  std::filesystem::remove(path, ec);
  if (ec) {
    SIRIUS_LOG_WARN(
      "[compressed_disk_representation] could not delete '{}': {}", path, ec.message());
  }
}

compressed_disk_representation::compressed_disk_representation(
  cucascade::memory::memory_space& memory_space,
  std::string path,
  std::size_t header_bytes,
  std::size_t compressed_bytes,
  std::size_t uncompressed_bytes,
  std::int64_t num_rows)
  : compressed_disk_representation(memory_space,
                                   std::make_shared<owned_file>(std::move(path)),
                                   header_bytes,
                                   compressed_bytes,
                                   uncompressed_bytes,
                                   num_rows)
{
}

compressed_disk_representation::compressed_disk_representation(
  cucascade::memory::memory_space& memory_space,
  std::shared_ptr<owned_file> file,
  std::size_t header_bytes,
  std::size_t compressed_bytes,
  std::size_t uncompressed_bytes,
  std::int64_t num_rows)
  : simpatico_compressed_representation(memory_space),
    _file(std::move(file)),
    _header_bytes(header_bytes),
    _compressed_bytes(compressed_bytes),
    _uncompressed_bytes(uncompressed_bytes),
    _num_rows(num_rows)
{
}

std::unique_ptr<cucascade::idata_representation> compressed_disk_representation::clone(
  [[maybe_unused]] ::cuda::stream_ref stream) const
{
  return std::unique_ptr<cucascade::idata_representation>(new compressed_disk_representation(
    get_memory_space(), _file, _header_bytes, _compressed_bytes, _uncompressed_bytes, _num_rows));
}

}  // namespace sirius
