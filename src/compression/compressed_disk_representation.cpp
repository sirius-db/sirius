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

#include "log/logging.hpp"

#include <filesystem>
#include <memory>

namespace sirius {

compressed_disk_representation::compressed_disk_representation(
  cucascade::memory::memory_space& memory_space,
  std::string path,
  std::size_t compressed_bytes,
  std::size_t uncompressed_bytes,
  std::int64_t num_rows,
  std::vector<std::string> column_names)
  : cucascade::idata_representation(memory_space),
    _path(std::make_shared<std::string>(std::move(path))),
    _owns_file(std::make_shared<bool>(true)),
    _compressed_bytes(compressed_bytes),
    _uncompressed_bytes(uncompressed_bytes),
    _num_rows(num_rows),
    _column_names(std::move(column_names))
{
}

compressed_disk_representation::compressed_disk_representation(
  cucascade::memory::memory_space& memory_space,
  std::shared_ptr<std::string> path,
  std::shared_ptr<bool> owns_file,
  std::size_t compressed_bytes,
  std::size_t uncompressed_bytes,
  std::int64_t num_rows,
  std::vector<std::string> column_names,
  std::optional<std::vector<std::size_t>> selected_indices)
  : cucascade::idata_representation(memory_space),
    _path(std::move(path)),
    _owns_file(std::move(owns_file)),
    _compressed_bytes(compressed_bytes),
    _uncompressed_bytes(uncompressed_bytes),
    _num_rows(num_rows),
    _column_names(std::move(column_names)),
    _selected_indices(std::move(selected_indices))
{
}

compressed_disk_representation::~compressed_disk_representation()
{
  // Unlink the file only when this is the last owner.
  if (_owns_file && _owns_file.use_count() == 1 && _path && !_path->empty()) {
    std::error_code ec;
    std::filesystem::remove(*_path, ec);
    if (ec) {
      SIRIUS_LOG_WARN(
        "compressed_disk_representation: failed to remove {}: {}", *_path, ec.message());
    }
  }
}

std::unique_ptr<cucascade::idata_representation> compressed_disk_representation::clone(
  rmm::cuda_stream_view /*stream*/)
{
  auto& ms = const_cast<cucascade::memory::memory_space&>(get_memory_space());
  return std::make_unique<compressed_disk_representation>(ms,
                                                          _path,
                                                          _owns_file,
                                                          _compressed_bytes,
                                                          _uncompressed_bytes,
                                                          _num_rows,
                                                          _column_names,
                                                          _selected_indices);
}

std::unique_ptr<compressed_disk_representation> compressed_disk_representation::select_columns(
  const std::vector<std::size_t>& indices) const
{
  auto& ms = const_cast<cucascade::memory::memory_space&>(get_memory_space());
  return std::make_unique<compressed_disk_representation>(ms,
                                                          _path,
                                                          _owns_file,
                                                          _compressed_bytes,
                                                          _uncompressed_bytes,
                                                          _num_rows,
                                                          _column_names,
                                                          indices);
}

}  // namespace sirius
