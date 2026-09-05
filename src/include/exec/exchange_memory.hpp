/*
 * Copyright 2026, Sirius Contributors.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
 * Unless required by applicable law or agreed to in writing, software distributed under the
 * License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND.
 */
#pragma once

#include "data/data_batch_utils.hpp"
#include "sirius/exception.hpp"

#include <cucascade/cudf/host_data_representation.hpp>
#include <cucascade/data/disk_data_representation.hpp>

#include <cstdlib>
#include <string_view>
#include <vector>

namespace sirius::exec {

inline bool optimized_exchange_enabled()
{
  const auto* value = std::getenv("SIRIUS_EXCHANGE_OPTIMIZED");
  return value != nullptr && std::string_view(value) == "1";
}

/// Schema and cardinality are properties of a batch, independent of its current tier.
struct exchange_batch_metadata {
  std::uint64_t rows{0};
  std::vector<cudf::data_type> types;
};

inline exchange_batch_metadata describe_exchange_batch(const cucascade::read_only_data_batch& batch)
{
  exchange_batch_metadata result;
  if (batch.get_current_tier() == cucascade::memory::Tier::GPU) {
    const auto view = sirius::get_cudf_table_view(batch);
    result.rows     = view.num_rows();
    for (const auto& column : view) {
      result.types.push_back(column.type());
    }
    return result;
  }
  const std::vector<cucascade::memory::column_metadata>* columns = nullptr;
  if (auto* host = dynamic_cast<const cucascade::host_data_representation*>(batch.get_data())) {
    columns = &host->get_host_table()->columns;
  } else if (auto* disk =
               dynamic_cast<const cucascade::disk_data_representation*>(batch.get_data())) {
    columns = &disk->get_disk_table().columns;
  } else {
    throw sirius::invalid_input_exception(
      "exchange: unsupported batch representation for metadata");
  }
  if (!columns->empty()) { result.rows = columns->front().num_rows; }
  for (const auto& column : *columns) {
    auto id = static_cast<cudf::type_id>(column.type_id);
    if (id == cudf::type_id::DECIMAL32 || id == cudf::type_id::DECIMAL64 ||
        id == cudf::type_id::DECIMAL128) {
      result.types.emplace_back(id, numeric::scale_type{column.scale});
    } else {
      result.types.emplace_back(id);
    }
  }
  return result;
}

}  // namespace sirius::exec
