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

#include "ducklake/inline_batch_producer.hpp"

#include "ducklake/binding_bridge.hpp"
#include "ducklake/field_mapping.hpp"
#include "ducklake/inline_reader.hpp"
#include "ducklake/inline_varchar.hpp"
#include "ducklake/preparation_gate.hpp"
#include "expression/ast/from_duckdb.hpp"
#include "helper/hugeint.hpp"
#include "helper/type_conversions.hpp"
#include "op/scan/arrow/arrow_staging_resource.hpp"
#include "op/scan/scan_plan.hpp"
#include "op/scan/scan_utils.hpp"
#include "op/scan/table_scan/owned_arrow_scan.hpp"

#include <duckdb/common/exception.hpp>
#include <duckdb/common/types/data_chunk.hpp>

#include <array>
#include <cstring>
#include <numeric>
#include <thread>

namespace sirius::ducklake {
namespace {
namespace scan           = op::scan;
namespace sm             = scan_manager;
constexpr size_t columns = inline_batch_producer::max_columns;
struct arrow_tree {
  ArrowSchema schema{}, children_schema[columns]{};
  ArrowArray array{}, children[columns]{};
  ArrowSchema* schemas[columns]{};
  ArrowArray* arrays[columns]{};
  void const* root_buffers[1]{nullptr};
  void const* buffers[columns][3]{};
  char formats[columns][24]{};
};
constexpr size_t tree_bytes = (sizeof(arrow_tree) + 7) / 8 * 8;
size_t column_bytes(size_t rows) { return rows * sizeof(int64_t) + rows / 8; }
static_assert(inline_batch_producer::max_batch_cells / inline_batch_producer::max_columns >= 64);
static_assert(inline_batch_producer::max_string_bytes <= INT32_MAX);
struct arrow_capacity {
  std::shared_ptr<void> claim;
  sm::charged_block backing, characters;
  op::scan::arrow_staging_resource staging;
  arrow_capacity(std::shared_ptr<void> owner,
                 sm::charged_block b,
                 sm::charged_block chars,
                 size_t payload_bytes,
                 size_t staging_bytes)
    : claim(std::move(owner)),
      backing(std::move(b)),
      characters(std::move(chars)),
      staging(backing.data() + payload_bytes, staging_bytes)
  {
  }
  std::byte* data() { return backing.data(); }
};
struct inline_binding {
  unit_descriptor unit;
};
struct inline_evidence final : scan::prepared_visibility {
  std::shared_ptr<inline_binding const> binding;
  std::weak_ptr<scan::owned_arrow_batch> batch;
  uint64_t sequence;
};
}  // namespace
inline_column_layout::inline_column_layout(duckdb::LogicalType bound_type)
  : type(std::move(bound_type))
{
  auto scalar_format = fixed_scalar_arrow_format(type);
  if (!scalar_format && !is_string())
    throw std::invalid_argument("DuckLake inline scalar type is outside its profile");
  device_type   = *scan::field_mapping_type(type);
  format        = is_string() ? "u" : *scalar_format;
  value_bytes   = is_string() ? sizeof(int32_t) : cudf::size_of(device_type);
  storage_bytes = std::max<size_t>(8, value_bytes);
  decimal_precision =
    type.id() == duckdb::LogicalTypeId::DECIMAL ? duckdb::DecimalType::GetWidth(type) : 0;
}
size_t inline_column_layout::copy_fixed_values(duckdb::Vector const& source,
                                               size_t offset,
                                               size_t count,
                                               std::byte* values,
                                               uint8_t* validity) const
{
  if (source.GetType() != type || is_string() || offset > source.size() ||
      count > source.size() - offset)
    throw std::invalid_argument("DuckLake inline vector differs from its bound layout");
  duckdb::UnifiedVectorFormat data;
  source.ToUnifiedFormat(data);
  size_t nulls = 0;
  for (size_t row = 0; row < count; ++row) {
    auto index = data.sel->get_index(offset + row);
    if (!data.validity.RowIsValid(index)) {
      ++nulls;
      continue;
    }
    auto* input = data.data + index * value_bytes;
    if (type.id() == duckdb::LogicalTypeId::BOOLEAN) {
      if (*input) reinterpret_cast<uint8_t*>(values)[row / 8] |= 1u << (row % 8);
    } else if (device_type.id() == cudf::type_id::DECIMAL128) {
      auto value =
        sirius::to_int128(duckdb::UnifiedVectorFormat::GetData<duckdb::hugeint_t>(data)[index]);
      std::memcpy(values + row * value_bytes, &value, value_bytes);
    } else {
      std::memcpy(values + row * value_bytes, input, value_bytes);
    }
    validity[row / 8] |= 1u << (row % 8);
  }
  return nulls;
}
struct inline_batch_producer::impl {
  std::shared_ptr<binding_handle> handle;
  std::shared_ptr<inline_binding const> binding;
  scan::arrow_import_binding import;
  std::unique_ptr<inline_reader> reader;  // query-owner lane only; reset by close.
  duckdb::DataChunk const* chunk = nullptr;
  size_t offset = 0, width = 0, row_limit = 0, wide_columns = 0;
  std::vector<inline_column_layout> layouts;
  std::vector<size_t> selected;
  std::vector<size_t> strings;
  std::atomic<bool> cancelled{false};
  std::thread::id owner = std::this_thread::get_id();
  impl(std::shared_ptr<binding_handle> h,
       unit_descriptor unit,
       uint64_t contract,
       uint64_t generation,
       std::shared_ptr<sm::preparation_admission> a,
       bool count_only,
       std::vector<size_t> required_columns,
       unit_scan_options options)
    : handle(std::move(h)),
      binding(std::make_shared<inline_binding const>(inline_binding{std::move(unit)}))
  {
    import.admission      = std::move(a);
    auto const& admission = import.admission;
    auto& program         = import.program;
    auto& outputs         = import.output_columns;
    auto& predicate       = import.predicate;
    auto const& u         = binding->unit;
    if (!contract || !generation || !u.read_view || u.kind == unit_kind::parquet ||
        !assess_local_scope(u).reasons.empty() || u.inline_delete_count ||
        u.transaction_local_delete || !admission || !admission->decision.deferred)
      throw std::invalid_argument("DuckLake inline unit has unqualified dependencies/admission");
    auto const& types = u.read_view->bound_types;
    if (count_only && (!required_columns.empty() || options.filters || options.output_columns))
      throw std::invalid_argument(
        "DuckLake count-only cannot request user columns or a filter projection");
    if (!count_only) {
      selected = required_field_indices(types.size(), std::move(required_columns));
      if (selected.empty() || selected.size() > max_columns ||
          std::any_of(selected.begin(), selected.end(), [&](auto index) {
            return !fixed_scalar_arrow_format(types[index]) &&
                   types[index] != duckdb::LogicalType::VARCHAR;
          }))
        throw std::invalid_argument(
          "DuckLake initial inline profile requires 1..32 selected fixed scalar/VARCHAR columns");
    }
    width     = count_only ? 0 : selected.size();
    row_limit = batch_rows(width);
    for (auto index : selected) {
      auto const& column = layouts.emplace_back(types[index]);
      wide_columns += column.storage_bytes > 8;
      if (column.is_string()) strings.push_back(layouts.size() - 1);
    }
    import.read_identity  = binding;
    import.display_name   = "DuckLake inline unit";
    import.profiles       = options.profiles ? std::move(options.profiles)
                                             : std::make_shared<scan::physical_profile_table>();
    import.owns_admission = options.owns_admission;
    import.gate           = std::move(options.inline_gate);
    // The composite scan emits one empty carrier after all units finish.
    import.emit_empty_source = false;
    if (options.identities)
      import.next_unit_id = [ids = std::move(options.identities)] { return ids->next(); };
    if (!count_only)
      for (auto index : selected)
        import.column_names.push_back(u.read_view->bound_names.at(index));
    validate_scan_options(options, width);
    outputs.resize(width);
    std::iota(outputs.begin(), outputs.end(), 0);
    if (options.output_columns) outputs = std::move(*options.output_columns);
    program.contract   = contract;
    program.generation = generation;
    program.input_types.emplace();
    duckdb::vector<duckdb::ColumnIndex> column_ids;
    duckdb::vector<sirius::logical_type> returned_types;
    std::vector<std::optional<size_t>> positions;
    for (size_t i = 0; i < width; ++i) {
      program.input_types->push_back(layouts[i].device_type);
      program.decimal_precisions.push_back(layouts[i].decimal_precision);
      column_ids.emplace_back(i);
      returned_types.push_back(sirius::from_duckdb(types[selected[i]]));
      positions.emplace_back(i);
    }
    for (auto i : outputs)
      program.output_types.push_back(program.input_types->at(i));
    if (options.filters) {
      auto expression = op::convert_table_filters_to_expression(
        *options.filters, column_ids, returned_types, positions);
      if (expression) {
        predicate = sirius::ast::from_duckdb(*expression);
        if (!predicate)
          throw std::invalid_argument("DuckLake inline predicate is not supported on GPU");
      }
    }
    program.predicates         = predicate ? 1 : 0;
    import.validate_provenance = [bound = binding,
                                  contract](scan::owned_arrow_scan_info const& input) {
      if (input.rootless())
        throw scan::certificate_incomplete(contract, {}, "DuckLake inline requires an owned batch");
      auto deps  = input.dependencies();
      auto proof = deps.size() == 1
                     ? std::dynamic_pointer_cast<inline_evidence const>(deps[0].visibility)
                     : nullptr;
      if (!proof || proof->binding != bound || proof->batch.lock() != input.batch() ||
          proof->sequence != input.sequence())
        throw scan::certificate_incomplete(
          contract, {}, "DuckLake inline batch provenance mismatch");
    };
  }
  void close()
  {
    if (std::this_thread::get_id() != owner)
      throw std::runtime_error("DuckLake inline close requires query owner");
    chunk = nullptr;
    reader.reset();
  }
  scan::arrow_production produce(scan::admitted_arrow_input input, sm::preparation_stop_token stop)
  {
    handle->validate_statement();
    if (cancelled || stop.stop_requested()) throw duckdb::InterruptException();
    if (!reader) reader = handle->open_inline_unit(binding->unit.original_file_index, selected);
    while (!chunk || offset == chunk->size()) {
      if (cancelled || stop.stop_requested()) throw duckdb::InterruptException();
      chunk  = reader->next();
      offset = 0;
      if (!chunk) return scan::arrow_end_of_stream{};
    }
    if (width && chunk->ColumnCount() != width)
      throw std::invalid_argument("DuckLake inline chunk differs from its binding");
    for (size_t col = 0; col < width; ++col)
      if (chunk->data[col].GetType() != layouts[col].type)
        throw std::invalid_argument("DuckLake inline vector differs from its bound layout");
    size_t count = std::min(row_limit, static_cast<size_t>(chunk->size()) - offset);
    std::array<size_t, columns> string_sizes{};
    if (!strings.empty()) {
      size_t accepted = 0, used = 0;
      for (; accepted < count; ++accepted) {
        size_t row_bytes = 0;
        std::array<size_t, columns> lengths{};
        bool fits = true;
        for (auto col : strings) {
          auto value  = borrow_inline_varchar(chunk->data[col], offset + accepted);
          auto length = value ? value->size() : 0;
          if (length > max_string_bytes - used - row_bytes) {
            fits = false;
            break;
          }
          row_bytes += length;
          lengths[col] = length;
        }
        if (!fits) {
          if (!accepted)
            throw sm::preparation_resource_error(
              "DuckLake inline row exceeds the 1 MiB VARCHAR payload budget", true);
          break;
        }
        used += row_bytes;
        for (auto col : strings)
          string_sizes[col] += lengths[col];
      }
      count = accepted;
    }
    // Temporary charged backing intentionally holds its permit through the
    // last GPU input read. Pending then needs neither a cursor advance nor a
    // second retained-payload budget or custom wakeup scheduler.
    auto& allocator = input.allocator;
    auto base_bytes = batch_capacity(width, 0, wide_columns);
    auto allocation = allocator.allocate(base_bytes);
    // Arrow buffers are independently owned. Character storage need not be
    // adjacent to the schema, fixed-width values or import staging arena.
    auto char_bytes = std::accumulate(string_sizes.begin(), string_sizes.end(), size_t{0});
    auto characters =
      strings.empty() ? sm::charged_block{} : allocator.allocate(std::max<size_t>(1, char_bytes));
    auto staging_bytes      = width ? max_staging_bytes : 0;
    auto capacity           = std::make_shared<arrow_capacity>(std::move(input.lifetime),
                                                     std::move(allocation),
                                                     std::move(characters),
                                                     base_bytes - staging_bytes,
                                                     staging_bytes);
    auto* tree              = new (capacity->data()) arrow_tree{};
    tree->schema.format     = "+s";
    tree->schema.n_children = width;
    tree->schema.children   = tree->schemas;
    tree->schema.release    = [](ArrowSchema* p) { p->release = nullptr; };
    tree->array.length      = count;
    tree->array.n_buffers   = 1;
    tree->array.buffers     = tree->root_buffers;
    tree->array.n_children  = width;
    tree->array.children    = tree->arrays;
    tree->array.release     = [](ArrowArray* p) { p->release = nullptr; };
    auto* payload           = capacity->data() + tree_bytes;
    auto* chars             = capacity->characters.data();
    for (size_t col = 0; col < width; ++col) {
      auto& schema       = tree->children_schema[col];
      auto& array        = tree->children[col];
      auto const& layout = layouts[col];
      bool is_string     = layout.is_string();
      auto const& format = layout.format;
      if (format.size() >= sizeof(tree->formats[col]))
        throw std::logic_error("DuckLake Arrow format exceeds owned schema storage");
      std::memcpy(tree->formats[col], format.c_str(), format.size() + 1);
      schema.format      = tree->formats[col];
      schema.flags       = ARROW_FLAG_NULLABLE;
      schema.release     = [](ArrowSchema* p) { p->release = nullptr; };
      auto column_stride = row_limit * layout.storage_bytes + row_limit / 8;
      auto* values       = payload;
      payload += column_stride;
      auto* valid = reinterpret_cast<uint8_t*>(values + row_limit * layout.storage_bytes);
      std::memset(values, 0, column_stride);
      auto* offsets      = reinterpret_cast<int32_t*>(values);
      size_t char_offset = 0;
      if (!is_string)
        array.null_count = layout.copy_fixed_values(chunk->data[col], offset, count, values, valid);
      else
        for (size_t row = 0; row < count; ++row) {
          auto value = borrow_inline_varchar(chunk->data[col], offset + row);
          if (value) {
            if (value->size() > string_sizes[col] - char_offset)
              throw std::logic_error("DuckLake borrowed VARCHAR changed during detach");
            if (!value->empty()) std::memcpy(chars + char_offset, value->data(), value->size());
            char_offset += value->size();
            valid[row / 8] |= 1u << (row % 8);
          } else
            ++array.null_count;
          offsets[row + 1] = static_cast<int32_t>(char_offset);
        }
      if (is_string) {
        if (char_offset != string_sizes[col])
          throw std::logic_error("DuckLake borrowed VARCHAR changed during detach");
        tree->buffers[col][2] = chars;
        chars += char_offset;
      }
      tree->buffers[col][0] = valid;
      tree->buffers[col][1] = values;
      array.length          = count;
      array.n_buffers       = is_string ? 3 : 2;
      array.buffers         = tree->buffers[col];
      array.release         = [](ArrowArray* p) { p->release = nullptr; };
      tree->schemas[col]    = &schema;
      tree->arrays[col]     = &array;
    }
    offset += count;
    auto batch = std::make_shared<scan::owned_arrow_batch>(
      tree->schema, tree->array, capacity, capacity->staging);
    auto proof      = std::make_shared<inline_evidence>();
    proof->binding  = binding;
    proof->batch    = batch;
    proof->sequence = input.key.unit_id;
    return scan::produced_arrow_input{std::move(batch), std::move(proof)};
  }
};
size_t inline_batch_producer::batch_rows(size_t columns)
{
  if (columns > max_columns) throw std::invalid_argument("DuckLake inline column limit");
  // Keep the widest batch's cell and normalization bounds. Whole validity words
  // also preserve alignment of each following column's fixed values/offsets.
  return columns ? std::min(max_batch_rows, (max_batch_cells / columns / 64) * 64) : max_batch_rows;
}
uint64_t inline_batch_producer::batch_capacity(size_t n, size_t string_columns, size_t wide_columns)
{
  if (n > max_columns || string_columns > n || wide_columns > n - string_columns)
    throw std::invalid_argument("DuckLake inline column limit");
  return tree_bytes + n * column_bytes(batch_rows(n)) + wide_columns * batch_rows(n) * 8 +
         (string_columns ? max_string_bytes : 0) + (n ? max_staging_bytes : 0);
}
sm::unit_envelope inline_batch_producer::envelope(size_t columns,
                                                  size_t string_columns,
                                                  size_t wide_columns)
{
  (void)batch_capacity(columns, string_columns, wide_columns);
  sm::unit_envelope result{{batch_capacity(columns, 0, wide_columns)}, 0};
  if (string_columns) result.temporary.push_back(max_string_bytes);
  return result;
}
inline_batch_producer::inline_batch_producer(std::shared_ptr<binding_handle> h,
                                             unit_descriptor unit,
                                             uint64_t contract,
                                             uint64_t generation,
                                             std::shared_ptr<sm::preparation_admission> admission,
                                             bool count_only,
                                             std::vector<size_t> required_columns,
                                             unit_scan_options options)
  : impl_(std::make_unique<impl>(std::move(h),
                                 std::move(unit),
                                 contract,
                                 generation,
                                 std::move(admission),
                                 count_only,
                                 std::move(required_columns),
                                 std::move(options)))
{
}
inline_batch_producer::~inline_batch_producer() = default;
scan::arrow_import_binding inline_batch_producer::take_binding()
{
  return std::move(impl_->import);
}
sm::preparation_execution_domain inline_batch_producer::execution_domain() const noexcept
{
  return sm::preparation_execution_domain::query_owner;
}
scan::arrow_producer_limits inline_batch_producer::limits() const
{
  return {static_cast<int64_t>(impl_->row_limit),
          envelope(impl_->width, impl_->strings.size(), impl_->wide_columns)};
}
void inline_batch_producer::preflight() { impl_->handle->validate_statement(); }
scan::arrow_readiness inline_batch_producer::readiness() const noexcept
{
  return scan::arrow_readiness::ready;
}
void inline_batch_producer::bind_wakeup(sm::preparation_wakeup) {}
scan::arrow_production inline_batch_producer::next(scan::admitted_arrow_input input,
                                                   sm::preparation_stop_token stop)
{
  return impl_->produce(std::move(input), stop);
}
void inline_batch_producer::close() { impl_->close(); }
void inline_batch_producer::cancel() noexcept { impl_->cancelled = true; }
}  // namespace sirius::ducklake
