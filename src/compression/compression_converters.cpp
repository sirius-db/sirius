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

#include "compression_converters.hpp"

#include "compressed_representation.hpp"

#include <rmm/mr/per_device_resource.hpp>

#include <api/compressed_table_io.hpp>
#include <api/simpatico_codegen.hpp>
#include <cucascade/data/gpu_data_representation.hpp>
#include <cucascade/data/representation_converter.hpp>
#include <cucascade/memory/memory_space.hpp>
#include <log/logging.hpp>

#include <stdexcept>
#include <utility>

namespace sirius {

namespace {

std::unique_ptr<cucascade::idata_representation> decompress_to_gpu(
  cucascade::idata_representation& source,
  const cucascade::memory::memory_space* target_memory_space,
  rmm::cuda_stream_view stream)
{
  auto& rep = source.cast<compressed_host_representation>();

  std::string read_error;
  simpatico::compressed_table ct = simpatico::read_compressed_table(
    rep.path(), stream, rmm::mr::get_current_device_resource_ref(), &read_error);
  if (!read_error.empty()) {
    throw std::runtime_error("[compression_converters] read_compressed_table failed for '" +
                             rep.path() + "': " + read_error);
  }

  // Project to the selected columns before decompressing to avoid
  // inflating memory with unrequested columns.
  simpatico::compressed_table subset;
  if (rep.selected_indices().has_value()) {
    const auto& indices = *rep.selected_indices();
    subset.columns.reserve(indices.size());
    for (auto idx : indices) {
      if (idx >= ct.columns.size()) {
        throw std::out_of_range(
          "[compression_converters] selected column index out of range during decompress");
      }
      subset.columns.push_back(std::move(ct.columns[idx]));
    }
  } else {
    subset = std::move(ct);
  }

  auto decompressed =
    simpatico::decompress(subset, stream, rmm::mr::get_current_device_resource_ref());

  const cucascade::memory::memory_space* space =
    (target_memory_space != nullptr) ? target_memory_space : &source.get_memory_space();

  SIRIUS_LOG_DEBUG("[compression_converters] decompressed '{}' cols={} rows={} → GPU device={}",
                   rep.path(),
                   decompressed->num_columns(),
                   decompressed->num_rows(),
                   space->get_device_id());

  return std::make_unique<cucascade::gpu_table_representation>(
    std::move(decompressed), *const_cast<cucascade::memory::memory_space*>(space), stream);
}

}  // namespace

void register_compression_converters(cucascade::representation_converter_registry& registry)
{
  // compressed_host_representation → GPU (decompression path used by prepare_for_processing)
  if (!registry
         .has_converter<compressed_host_representation, cucascade::gpu_table_representation>()) {
    registry
      .register_converter<compressed_host_representation, cucascade::gpu_table_representation>(
        decompress_to_gpu);
  }
}

}  // namespace sirius
