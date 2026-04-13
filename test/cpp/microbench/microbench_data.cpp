/*
 * Copyright 2025, Sirius Contributors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); you may not
 * use this file except in compliance with the License.
 */

#include "microbench_data.hpp"

#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/types.hpp>

#include <rmm/mr/per_device_resource.hpp>

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <vector>

namespace sirius::microbench {

namespace {

void throw_if_cuda_error(char const* ctx)
{
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string(ctx) + ": " + cudaGetErrorString(err));
  }
}

}  // namespace

std::unique_ptr<cudf::column> make_modulo_int32_keys(cudf::size_type num_rows,
                                                     std::int32_t num_groups,
                                                     rmm::cuda_stream_view stream)
{
  if (num_rows <= 0 || num_groups <= 0) {
    throw std::invalid_argument("make_modulo_int32_keys: num_rows and num_groups must be positive");
  }
  std::vector<std::int32_t> host(static_cast<std::size_t>(num_rows));
  for (cudf::size_type i = 0; i < num_rows; ++i) {
    host[static_cast<std::size_t>(i)] =
      static_cast<std::int32_t>(static_cast<std::int64_t>(i) % num_groups);
  }
  auto col          = cudf::make_numeric_column(cudf::data_type{cudf::type_id::INT32},
                                       num_rows,
                                       cudf::mask_state::UNALLOCATED,
                                       stream,
                                       rmm::mr::get_current_device_resource());
  auto mutable_view = col->mutable_view();
  cudaMemcpyAsync(mutable_view.data<std::int32_t>(),
                  host.data(),
                  host.size() * sizeof(std::int32_t),
                  cudaMemcpyHostToDevice,
                  stream.value());
  stream.synchronize();
  throw_if_cuda_error("make_modulo_int32_keys HtoD");
  return col;
}

std::unique_ptr<cudf::column> make_int64_ones(cudf::size_type num_rows,
                                              rmm::cuda_stream_view stream)
{
  if (num_rows <= 0) { throw std::invalid_argument("make_int64_ones: num_rows must be positive"); }
  std::vector<std::int64_t> host(static_cast<std::size_t>(num_rows), 1);
  auto col          = cudf::make_numeric_column(cudf::data_type{cudf::type_id::INT64},
                                       num_rows,
                                       cudf::mask_state::UNALLOCATED,
                                       stream,
                                       rmm::mr::get_current_device_resource());
  auto mutable_view = col->mutable_view();
  cudaMemcpyAsync(mutable_view.data<std::int64_t>(),
                  host.data(),
                  host.size() * sizeof(std::int64_t),
                  cudaMemcpyHostToDevice,
                  stream.value());
  stream.synchronize();
  throw_if_cuda_error("make_int64_ones HtoD");
  return col;
}

std::unique_ptr<cudf::column> make_sparse_bool_mask(cudf::size_type num_rows,
                                                    int permille_true,
                                                    rmm::cuda_stream_view stream)
{
  if (num_rows <= 0 || permille_true < 0 || permille_true > 1000) {
    throw std::invalid_argument("make_sparse_bool_mask: invalid args");
  }
  // BOOL8 columns use one byte per element (0/1), same as elsewhere in Sirius.
  std::vector<std::uint8_t> host(static_cast<std::size_t>(num_rows));
  for (cudf::size_type i = 0; i < num_rows; ++i) {
    std::uint64_t const h             = static_cast<std::uint64_t>(i) * 7919ULL;
    bool const keep                   = static_cast<int>(h % 1000) < permille_true;
    host[static_cast<std::size_t>(i)] = keep ? std::uint8_t{1} : std::uint8_t{0};
  }
  auto col          = cudf::make_numeric_column(cudf::data_type{cudf::type_id::BOOL8},
                                       num_rows,
                                       cudf::mask_state::UNALLOCATED,
                                       stream,
                                       rmm::mr::get_current_device_resource());
  auto mutable_view = col->mutable_view();
  cudaMemcpyAsync(mutable_view.data<std::uint8_t>(),
                  host.data(),
                  host.size() * sizeof(std::uint8_t),
                  cudaMemcpyHostToDevice,
                  stream.value());
  stream.synchronize();
  throw_if_cuda_error("make_sparse_bool_mask HtoD");
  return col;
}

std::optional<std::unique_ptr<cudf::column>> try_read_parquet_column(
  std::string const& parquet_file,
  std::string const& column_name,
  cudf::size_type max_rows,
  rmm::cuda_stream_view stream)
{
  if (max_rows <= 0) { return std::nullopt; }
  {
    std::ifstream f(parquet_file.c_str(), std::ios::binary);
    if (!f) { return std::nullopt; }
  }
  try {
    cudf::io::parquet_reader_options opts =
      cudf::io::parquet_reader_options::builder(cudf::io::source_info{parquet_file})
        .column_names({column_name})
        .build();
    cudf::io::table_with_metadata tw = cudf::io::read_parquet(opts, stream);
    stream.synchronize();
    if (!tw.tbl || tw.tbl->num_columns() < 1 || tw.tbl->num_rows() == 0) { return std::nullopt; }
    cudf::size_type const n = std::min(max_rows, tw.tbl->num_rows());
    if (n <= 0) { return std::nullopt; }
    if (tw.tbl->num_rows() == n) {
      return std::make_unique<cudf::column>(tw.tbl->view().column(0), stream);
    }
    auto sliced = cudf::slice(tw.tbl->view(), {0, n}, stream);
    if (sliced.empty()) { return std::nullopt; }
    return std::make_unique<cudf::column>(sliced.front().column(0), stream);
  } catch (...) {
    return std::nullopt;
  }
}

}  // namespace sirius::microbench
