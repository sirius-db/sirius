/*
 * Copyright 2025, Sirius Contributors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); you may not
 * use this file except in compliance with the License.
 */

#include "microbench_data.hpp"

#include <cudf/column/column_factories.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/table/table.hpp>
#include <cudf/types.hpp>

#include <rmm/mr/per_device_resource.hpp>

#include <cuda_runtime.h>

#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <vector>

#if defined(__linux__)
#include <fcntl.h>
#include <unistd.h>
#endif

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

void discard_os_page_cache_for_file(std::string const& parquet_path)
{
#if defined(__linux__)
  int const fd = ::open(parquet_path.c_str(), O_RDONLY | O_CLOEXEC);
  if (fd < 0) { return; }
  // len == 0 => from offset to EOF (Linux).
  (void)::posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
  ::close(fd);
#else
  (void)parquet_path;
#endif
}

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

std::optional<std::unique_ptr<cudf::table>> try_read_parquet_table(std::string const& parquet_file,
                                                                   rmm::cuda_stream_view stream)
{
  {
    std::ifstream f(parquet_file.c_str(), std::ios::binary);
    if (!f) { return std::nullopt; }
  }
  try {
    cudf::io::parquet_reader_options const opts =
      cudf::io::parquet_reader_options::builder(cudf::io::source_info{parquet_file}).build();
    cudf::io::table_with_metadata tw = cudf::io::read_parquet(opts, stream);
    stream.synchronize();
    if (!tw.tbl || tw.tbl->num_columns() < 1 || tw.tbl->num_rows() == 0) { return std::nullopt; }
    return std::move(tw.tbl);
  } catch (...) {
    return std::nullopt;
  }
}

}  // namespace sirius::microbench
