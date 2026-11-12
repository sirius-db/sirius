// Copyright 2026, Sirius Contributors. SPDX-License-Identifier: Apache-2.0
#include "output/parquet_writer.hpp"

#include <cudf/column/column_factories.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/table/table.hpp>
#include <cudf/unary.hpp>

#include <cuda_runtime_api.h>

#include <absl/cleanup/cleanup.h>

#include <stdexcept>

namespace sirius::output {
namespace {
cudf::data_type physical_type(logical_type const& type)
{
  using C = cudf::type_id;
  switch (type.id()) {
    case type_id::BOOLEAN: return cudf::data_type{C::BOOL8};
    case type_id::TINYINT: return cudf::data_type{C::INT8};
    case type_id::SMALLINT: return cudf::data_type{C::INT16};
    case type_id::INTEGER: return cudf::data_type{C::INT32};
    case type_id::BIGINT: return cudf::data_type{C::INT64};
    case type_id::UTINYINT: return cudf::data_type{C::UINT8};
    case type_id::USMALLINT: return cudf::data_type{C::UINT16};
    case type_id::UINTEGER: return cudf::data_type{C::UINT32};
    case type_id::UBIGINT: return cudf::data_type{C::UINT64};
    case type_id::FLOAT: return cudf::data_type{C::FLOAT32};
    case type_id::DOUBLE: return cudf::data_type{C::FLOAT64};
    case type_id::VARCHAR: return cudf::data_type{C::STRING};
    case type_id::DATE: return cudf::data_type{C::TIMESTAMP_DAYS};
    case type_id::TIMESTAMP_SEC: return cudf::data_type{C::TIMESTAMP_SECONDS};
    case type_id::TIMESTAMP_MS: return cudf::data_type{C::TIMESTAMP_MILLISECONDS};
    case type_id::TIMESTAMP: return cudf::data_type{C::TIMESTAMP_MICROSECONDS};
    case type_id::TIMESTAMP_NS: return cudf::data_type{C::TIMESTAMP_NANOSECONDS};
    case type_id::DECIMAL:
      return cudf::data_type{type.decimal_precision() <= 9    ? C::DECIMAL32
                             : type.decimal_precision() <= 18 ? C::DECIMAL64
                                                              : C::DECIMAL128,
                             -int32_t(type.decimal_scale())};
    default: throw std::invalid_argument("Unsupported Parquet physical type");
  }
}
cudf::io::compression_type cudf_codec(parquet_compression codec)
{
  using C = cudf::io::compression_type;
  switch (codec) {
    case parquet_compression::uncompressed: return C::NONE;
    case parquet_compression::snappy: return C::SNAPPY;
    case parquet_compression::gzip: return C::GZIP;
    case parquet_compression::brotli: return C::BROTLI;
    case parquet_compression::zstd: return C::ZSTD;
    case parquet_compression::lz4: return C::LZ4;
  }
  throw std::invalid_argument("Invalid Parquet codec");
}
void column_metadata(cudf::io::column_in_metadata& meta, parquet_field const& field)
{
  meta.set_name(field.name).set_nullability(true);
  if (field.decimal_precision) meta.set_decimal_precision(field.decimal_precision);
  if (field.is_map) meta.set_list_column_as_map();
  for (std::size_t i = 0; i < field.children.size(); ++i) {
    auto const child_index =
      field.type.id() == type_id::LIST || field.type.id() == type_id::ARRAY ? i + 1 : i;
    column_metadata(meta.child(child_index), field.children[i]);
  }
}
cudf::column_view parquet_view(parquet_field const& field,
                               cudf::column_view column,
                               std::vector<std::unique_ptr<cudf::column>>& owners,
                               ::cuda::stream_ref stream)
{
  // Sirius carries DECIMAL(1..4,s) as unscaled INT16. Parquet/cuDF require
  // DECIMAL32: widen the unscaled integer, then attach the scale without rescaling.
  if (field.type.is_decimal() && field.type.decimal_precision() <= 4 &&
      column.type().id() == cudf::type_id::INT16) {
    auto widened  = cudf::cast(column, cudf::data_type{cudf::type_id::INT32}, stream);
    auto nulls    = widened->null_count();
    auto contents = widened->release();
    owners.push_back(std::make_unique<cudf::column>(
      cudf::data_type{cudf::type_id::DECIMAL32, -int32_t(field.type.decimal_scale())},
      column.size(),
      std::move(*contents.data),
      std::move(*contents.null_mask),
      nulls));
    return owners.back()->view();
  }
  if (!field.children.empty()) {
    std::vector<cudf::column_view> children;
    bool const list = field.type.id() == type_id::LIST || field.type.id() == type_id::ARRAY;
    if (list) children.push_back(column.child(0));
    for (std::size_t i = 0; i < field.children.size(); ++i)
      children.push_back(
        parquet_view(field.children[i], column.child(i + (list ? 1 : 0)), owners, stream));
    return {column.type(),
            column.size(),
            column.head<void>(),
            column.null_mask(),
            column.null_count(),
            column.offset(),
            children};
  }
  return column;
}
std::unique_ptr<cudf::column> empty_column(parquet_field const& field)
{
  if (field.type.id() == type_id::STRUCT) {
    std::vector<std::unique_ptr<cudf::column>> children;
    for (auto const& child : field.children)
      children.push_back(empty_column(child));
    return cudf::make_structs_column(0, std::move(children), 0, {});
  }
  if (field.type.id() == type_id::LIST || field.type.id() == type_id::ARRAY) {
    auto offsets = cudf::make_numeric_column(
      cudf::data_type{cudf::type_id::INT32}, 1, cudf::mask_state::UNALLOCATED);
    if (cudaMemset(offsets->mutable_view().data<int32_t>(), 0, sizeof(int32_t)) != cudaSuccess)
      throw std::runtime_error("Initializing empty Parquet list offsets failed");
    return cudf::make_lists_column(
      0, std::move(offsets), empty_column(field.children.at(0)), 0, {});
  }
  return cudf::make_empty_column(physical_type(field.type));
}
}  // namespace
bool supported_parquet_compression(parquet_compression codec)
{
  return cudf::io::is_supported_write_parquet(cudf_codec(codec));
}
void write_parquet_part(parquet_export_spec const& spec,
                        cudf::table_view table,
                        std::filesystem::path const& path,
                        ::cuda::stream_ref stream)
{
  if (std::size_t(table.num_columns()) != spec.fields.size())
    throw std::runtime_error("Parquet export schema width mismatch");
  std::vector<std::unique_ptr<cudf::column>> owners;
  std::vector<cudf::column_view> views;
  absl::Cleanup drain_on_error = [stream] {
    if (cudaStreamSynchronize(stream.get()) != cudaSuccess) static_cast<void>(cudaGetLastError());
  };

  for (std::size_t i = 0; i < spec.fields.size(); ++i)
    views.push_back(parquet_view(spec.fields[i], table.column(i), owners, stream));
  table = cudf::table_view(views);
  cudf::io::table_input_metadata metadata(table);
  for (std::size_t i = 0; i < spec.fields.size(); ++i)
    column_metadata(metadata.column_metadata[i], spec.fields[i]);
  auto options =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info(path.string()), table)
      .metadata(std::move(metadata))
      .compression(cudf_codec(spec.compression))
      .build();
  cudf::io::write_parquet(options, stream);
  // A retry must not unlink a file or release payloads before outstanding work completes.
  stream.sync();
  std::move(drain_on_error).Cancel();
}
void write_empty_parquet(parquet_export_spec const& spec, std::filesystem::path const& path)
{
  std::vector<std::unique_ptr<cudf::column>> columns;
  for (auto const& field : spec.fields)
    columns.push_back(empty_column(field));
  cudf::table table(std::move(columns));
  write_parquet_part(spec, table.view(), path, cudf::get_default_stream());
}
}  // namespace sirius::output
