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

#include "compressed_representation.hpp"

#include <log/logging.hpp>

#include <cstdio>
#include <stdexcept>
#include <utility>

namespace sirius {

// ── Owning constructor ───────────────────────────────────────────────────────

compressed_host_representation::compressed_host_representation(
  cucascade::memory::memory_space& memory_space,
  std::string path,
  std::vector<std::string> column_names,
  std::size_t compressed_bytes,
  std::size_t uncompressed_bytes,
  std::int64_t num_rows)
  : cucascade::idata_representation(memory_space),
    _path(std::make_shared<std::string>(std::move(path))),
    _owns_file(std::make_shared<bool>(true)),
    _column_names(std::move(column_names)),
    _compressed_bytes(compressed_bytes),
    _uncompressed_bytes(uncompressed_bytes),
    _num_rows(num_rows)
{
}

// ── Sharing constructor (private) ────────────────────────────────────────────

compressed_host_representation::compressed_host_representation(
  cucascade::memory::memory_space& memory_space,
  std::shared_ptr<std::string> shared_path,
  std::shared_ptr<bool> owns_file,
  std::vector<std::string> column_names,
  std::size_t compressed_bytes,
  std::size_t uncompressed_bytes,
  std::int64_t num_rows,
  std::optional<std::vector<std::size_t>> selected_indices)
  : cucascade::idata_representation(memory_space),
    _path(std::move(shared_path)),
    _owns_file(std::move(owns_file)),
    _column_names(std::move(column_names)),
    _compressed_bytes(compressed_bytes),
    _uncompressed_bytes(uncompressed_bytes),
    _num_rows(num_rows),
    _selected_indices(std::move(selected_indices))
{
}

// ── Destructor ───────────────────────────────────────────────────────────────

compressed_host_representation::~compressed_host_representation()
{
  // Unlink the file when this is the last owner.
  if (_owns_file && _owns_file.use_count() == 1 && *_owns_file && _path) {
    if (std::remove(_path->c_str()) != 0) {
      SIRIUS_LOG_WARN("[compressed_host_representation] failed to unlink temp file '{}'", *_path);
    }
  }
}

// ── idata_representation interface ───────────────────────────────────────────

std::unique_ptr<cucascade::idata_representation> compressed_host_representation::clone(
  rmm::cuda_stream_view /*stream*/)
{
  // Share the same backing file — no byte copy needed.
  return std::unique_ptr<compressed_host_representation>(
    new compressed_host_representation(get_memory_space(),
                                       _path,
                                       _owns_file,
                                       _column_names,
                                       _compressed_bytes,
                                       _uncompressed_bytes,
                                       _num_rows,
                                       _selected_indices));
}

// ── Projection ───────────────────────────────────────────────────────────────

std::unique_ptr<compressed_host_representation> compressed_host_representation::select_columns(
  std::span<const std::size_t> indices) const
{
  // Build absolute indices into _column_names, respecting any existing projection.
  std::vector<std::size_t> absolute;
  absolute.reserve(indices.size());
  for (auto idx : indices) {
    if (_selected_indices.has_value()) {
      if (idx >= _selected_indices->size()) {
        throw std::out_of_range(
          "[compressed_host_representation::select_columns] index out of range");
      }
      absolute.push_back((*_selected_indices)[idx]);
    } else {
      if (idx >= _column_names.size()) {
        throw std::out_of_range(
          "[compressed_host_representation::select_columns] index out of range");
      }
      absolute.push_back(idx);
    }
  }

  return std::unique_ptr<compressed_host_representation>(
    new compressed_host_representation(get_memory_space(),
                                       _path,
                                       _owns_file,
                                       _column_names,
                                       _compressed_bytes,
                                       _uncompressed_bytes,
                                       _num_rows,
                                       std::move(absolute)));
}

}  // namespace sirius
