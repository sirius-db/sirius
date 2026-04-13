/*
 * Copyright 2025, Sirius Contributors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); you may not
 * use this file except in compliance with the License.
 */
#pragma once

#include <cudf/column/column.hpp>
#include <cudf/types.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <cstdint>
#include <memory>
#include <optional>
#include <string>

namespace sirius::microbench {

/// Keys `int32(i % num_groups)` — TPC-H–like bounded key cardinality (e.g. NDV).
[[nodiscard]] std::unique_ptr<cudf::column> make_modulo_int32_keys(cudf::size_type num_rows,
                                                                   std::int32_t num_groups,
                                                                   rmm::cuda_stream_view stream);

/// Payload column of INT64 ones (for sum aggregate microbench).
[[nodiscard]] std::unique_ptr<cudf::column> make_int64_ones(cudf::size_type num_rows,
                                                            rmm::cuda_stream_view stream);

/// Pseudorandom sparse BOOL8 mask; approximately `permille_true / 1000` fraction true.
[[nodiscard]] std::unique_ptr<cudf::column> make_sparse_bool_mask(cudf::size_type num_rows,
                                                                  int permille_true,
                                                                  rmm::cuda_stream_view stream);

/// Read up to `max_rows` from a single Parquet file column (first row group only if small).
/// Returns nullopt if the file cannot be read or the column is missing.
[[nodiscard]] std::optional<std::unique_ptr<cudf::column>> try_read_parquet_column(
  std::string const& parquet_file,
  std::string const& column_name,
  cudf::size_type max_rows,
  rmm::cuda_stream_view stream);

}  // namespace sirius::microbench
