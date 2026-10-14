/* Copyright 2026, Sirius Contributors. SPDX-License-Identifier: Apache-2.0 */
#include "op/scan/arrow_materializer.hpp"

#include <cudf/interop.hpp>
#include <cudf/unary.hpp>

#include <bit>
#include <charconv>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string_view>

namespace sirius::op::scan {
owned_arrow_batch::owned_arrow_batch(ArrowSchema& s, ArrowArray& a, std::shared_ptr<void> permit)
  : capacity(std::move(permit))
{
  if (!s.release || !a.release || !capacity)
    throw std::invalid_argument("Arrow input requires ownership and capacity");
  schema    = s;
  array     = a;
  s.release = nullptr;
  a.release = nullptr;
}
owned_arrow_batch::~owned_arrow_batch()
{
  if (array.release) array.release(&array);
  if (schema.release) schema.release(&schema);
}

namespace {
[[noreturn]] void invalid()
{
  throw std::invalid_argument("unsupported or invalid arrow_host_flat_v1 input");
}
cudf::data_type type_for(std::string_view f)
{
  using T = cudf::type_id;
  if (f == "b") return cudf::data_type{T::BOOL8};
  if (f == "c") return cudf::data_type{T::INT8};
  if (f == "s") return cudf::data_type{T::INT16};
  if (f == "i") return cudf::data_type{T::INT32};
  if (f == "l") return cudf::data_type{T::INT64};
  if (f == "C") return cudf::data_type{T::UINT8};
  if (f == "S") return cudf::data_type{T::UINT16};
  if (f == "I") return cudf::data_type{T::UINT32};
  if (f == "L") return cudf::data_type{T::UINT64};
  if (f == "f") return cudf::data_type{T::FLOAT32};
  if (f == "g") return cudf::data_type{T::FLOAT64};
  if (f == "u") return cudf::data_type{T::STRING};
  if (f == "tdD") return cudf::data_type{T::TIMESTAMP_DAYS};
  if (f == "tsu:") return cudf::data_type{T::TIMESTAMP_MICROSECONDS};
  if (f.starts_with("d:")) {
    int precision = 0, scale = 0;
    auto p = std::from_chars(f.data() + 2, f.data() + f.size(), precision);
    if (p.ec != std::errc{} || p.ptr == f.data() + f.size() || *p.ptr != ',') invalid();
    auto s = std::from_chars(p.ptr + 1, f.data() + f.size(), scale);
    if (s.ec != std::errc{} ||
        (s.ptr != f.data() + f.size() &&
         std::string_view(s.ptr, f.data() + f.size() - s.ptr) != ",128") ||
        precision < 1 || precision > 38 || scale < 0 || scale > precision)
      invalid();
    return cudf::data_type{T::DECIMAL128, -scale};
  }
  invalid();
}
void validate_metadata(char const* metadata)
{
  if (!metadata) return;
  // Like the C Data buffers, metadata is producer-owned readable memory. Bound
  // lengths before traversing it and refuse Arrow extension semantics.
  auto integer = [&]() {
    int32_t n;
    std::memcpy(&n, metadata, sizeof(n));
    metadata += sizeof(n);
    if (n < 0 || n > (1 << 20)) invalid();
    return n;
  };
  auto entries = integer();
  if (entries > 4096) invalid();
  for (int32_t i = 0; i < entries; ++i) {
    auto key_length = integer();
    std::string_view key(metadata, key_length);
    metadata += key_length;
    auto value_length = integer();
    metadata += value_length;
    if (key == "ARROW:extension:name" || key == "ARROW:extension:metadata") invalid();
  }
}
void common(ArrowSchema const& s, ArrowArray const& a)
{
  if (!s.format || !s.release || !a.release || s.dictionary || a.dictionary || a.length < 0 ||
      a.length > std::numeric_limits<cudf::size_type>::max() || a.offset < 0 ||
      a.offset > std::numeric_limits<int64_t>::max() - a.length || a.n_buffers < 1 ||
      a.null_count < -1 || a.null_count > a.length || !a.buffers)
    invalid();
  validate_metadata(s.metadata);
  if (a.null_count > 0 && !a.buffers[0]) invalid();
}
}  // namespace

std::vector<cudf::data_type> validate_arrow_host_flat(ArrowSchema const& schema,
                                                      ArrowArray const& array)
{
  if constexpr (std::endian::native != std::endian::little) invalid();
  common(schema, array);
  if (std::string_view(schema.format) != "+s" || schema.n_children < 0 ||
      schema.n_children != array.n_children || array.n_buffers != 1 ||
      (array.n_children && (!array.children || !schema.children)))
    invalid();
  if (array.null_count > 0) invalid();
  if (array.buffers[0]) {
    auto bits = static_cast<uint8_t const*>(array.buffers[0]);
    for (int64_t i = array.offset; i < array.offset + array.length; ++i)
      if (!(bits[i / 8] & (1U << (i % 8)))) invalid();
  }
  std::vector<cudf::data_type> types;
  for (int64_t i = 0; i < array.n_children; ++i) {
    if (!array.children[i] || !schema.children[i]) invalid();
    auto const& a = *array.children[i];
    auto const& s = *schema.children[i];
    common(s, a);
    if (a.n_children || s.n_children || a.length < array.offset + array.length) invalid();
    auto type    = type_for(s.format);
    bool strings = type.id() == cudf::type_id::STRING;
    if (a.n_buffers != (strings ? 3 : 2)) invalid();
    if (a.length && !a.buffers[1]) invalid();
    if (strings && a.buffers[1]) {
      auto offsets = static_cast<int32_t const*>(a.buffers[1]);
      auto end     = a.offset + a.length;
      if (offsets[a.offset] < 0) invalid();
      for (int64_t j = a.offset; j < end; ++j)
        if (offsets[j + 1] < offsets[j]) invalid();
      if (offsets[end] && !a.buffers[2]) invalid();
    }
    types.push_back(type);
  }
  return types;
}

materialized_batch import_arrow_host(std::shared_ptr<owned_arrow_batch> input,
                                     materialization_program const& program,
                                     exec::retire_lane& lane,
                                     rmm::device_async_resource_ref mr)
{
  if (!input || !input->capacity || !program.contract || !program.generation)
    throw std::invalid_argument("Arrow import requires admitted input and program");
  auto types = validate_arrow_host_flat(input->schema, input->array);
  if (!program.input_types || types != *program.input_types)
    throw std::invalid_argument("Arrow schema does not match its bound program");
  for (std::size_t i = 0; i < types.size(); ++i) {
    if (types[i].id() != cudf::type_id::DECIMAL128) continue;
    int precision = 0;
    std::string_view format(input->schema.children[i]->format);
    (void)std::from_chars(format.data() + 2, format.data() + format.size(), precision);
    if (program.decimal_precisions.size() != types.size() ||
        program.decimal_precisions[i] != precision)
      throw std::invalid_argument("Arrow decimal precision does not match its bound program");
  }
  // The pinned importer consumes each child array independently and does not
  // apply a struct root slice. Normalize the root range into every child while
  // retaining both these headers and the original callback-owned tree.
  struct import_view {
    std::shared_ptr<owned_arrow_batch> owner;
    ArrowArray root;
    void const* root_buffers[1]{nullptr};
    std::vector<ArrowArray> children;
    std::vector<ArrowArray*> pointers;
  };
  auto view             = std::make_shared<import_view>();
  view->owner           = input;
  view->root            = input->array;
  view->root.offset     = 0;
  view->root.null_count = 0;
  view->root.buffers    = view->root_buffers;
  view->children.reserve(input->array.n_children);
  for (int64_t i = 0; i < input->array.n_children; ++i) {
    auto child = *input->array.children[i];
    if (input->array.offset || child.length != input->array.length) {
      child.offset += input->array.offset;
      child.length     = input->array.length;
      child.null_count = child.buffers[0] ? -1 : 0;
    }
    view->children.push_back(child);
  }
  for (auto& child : view->children)
    view->pointers.push_back(&child);
  view->root.children = view->pointers.data();
  auto submission     = lane.begin();
  // Register ownership before cuDF can enqueue even a partially failing import.
  submission.on_retire([owner = view](cudaError_t) noexcept {});
  ::cuda::stream_ref stream{submission.stream()};
  auto table = cudf::from_arrow(&input->schema, &view->root, stream, mr);
  if (table->num_columns() != static_cast<cudf::size_type>(types.size()))
    throw std::invalid_argument("Arrow import returned a different column layout");
  auto columns = table->release();
  for (std::size_t i = 0; i < types.size(); ++i) {
    if (columns[i]->type() == types[i]) continue;
    auto actual = columns[i]->type();
    if (types[i].id() != cudf::type_id::DECIMAL128 ||
        (actual.id() != cudf::type_id::DECIMAL32 && actual.id() != cudf::type_id::DECIMAL64) ||
        actual.scale() != types[i].scale())
      throw std::invalid_argument("Arrow import returned an unqualified physical type");
    columns[i] = cudf::cast(columns[i]->view(), types[i], stream, mr);
  }
  materialized_batch batch;
  batch.table      = owning_table_view(std::make_unique<cudf::table>(std::move(columns)));
  batch.rows       = input->array.length;
  batch.contract   = program.contract;
  batch.generation = program.generation;
  batch.logical_schema_complete = true;
  ensure_row_count_carrier(batch, stream, mr);
  validate_materialized_batch(batch, program);
  auto error = submission.commit();
  if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
  return batch;
}
}  // namespace sirius::op::scan
