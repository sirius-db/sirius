/*
 * Copyright 2025, Sirius Contributors.
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

// Regression coverage for the transaction handling in src/sirius_ffi.cpp:
//
// 1. Fragment::Impl::end_lifecycle(): a build() failure while the transaction is still open must
//    roll it back, not commit it (a7bb47e2).
// 2. lower_substrait(): Fragment::build() commits its view-creation transaction before it lowers
//    the plan, and DuckDB 1.5.5 throws "TransactionContext::ActiveTransaction called without
//    active transaction" from every catalog lookup made without one — so lowering must own a
//    transaction of its own.
//
// The public FFI surface links only DuckDB's substrait consumer (no substrait-plan-from-SQL
// helper, no raw-SQL passthrough), so a valid plan has to be built here by hand with the bundled
// protobuf, and catalog state after a failed build() cannot be inspected. The end_lifecycle()
// tests therefore use a declared column type name that TransformStringToLogicalType() can never
// resolve, which fails build() inside resolve_inputs() before `substrait_plan` is ever parsed —
// and check the one thing observable through the public API: that end_lifecycle() leaves the
// connection able to start and fail a second, independent Fragment cleanly.
//
// 3. Fragment::push_arrow(): a host Arrow record batch (Arrow C Data Interface) enters an input
//    stream, is checked against the declared schema, and comes back out of result_to_arrow() with
//    the same rows. The by-name refusals of helper/arrow_host_import.hpp are unit-tested directly.

#include "helper/arrow_c_device.hpp"  // ArrowDeviceArray (cudf::to_arrow_host result)
#include "helper/arrow_host_import.hpp"
#include "sirius_ffi.hpp"

#include <cudf/column/column_factories.hpp>
#include <cudf/interop.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/table/table.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/device_buffer.hpp>

#include <cuda_runtime_api.h>

#include <catch.hpp>
// The Arrow C Data Interface structs (ArrowSchema, ArrowArray, ArrowArrayStream): DuckDB's copy,
// the definition libsirius itself is built against. Apache Arrow's arrow/c/abi.h is not a
// dependency of this tree (absent from the vcpkg build), so it is not included here.
#include <duckdb/common/arrow/arrow.hpp>
#include <duckdb/common/exception/transaction_exception.hpp>
#include <substrait/plan.pb.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <optional>
#include <source_location>
#include <string>
#include <thread>
#include <tuple>
#include <vector>

namespace fs = std::filesystem;

namespace {

// Small 2GB-GPU/4GB-host config shared with other [isolated_context] tests.
fs::path isolated_memory_config_path()
{
  std::source_location loc = std::source_location::current();
  return fs::path(loc.file_name()).parent_path().parent_path() / "scan" / "memory.yaml";
}

void declare_unresolvable_column(sirius::ffi::Fragment& fragment, const std::string& type_name)
{
  fragment.declare_input_column(0, "a", type_name);
}

// Column kinds the hand-built plans below can declare. Substrait has no i128, so HUGEINT never
// appears in a plan; its refusal is tested against the helper directly.
enum class col_kind { I32, I64, F64, BOOL, STRING, DECIMAL_15_2, DATE };

struct plan_column {
  std::string name;
  col_kind kind;
};

void set_plan_type(substrait::Type* type, col_kind kind)
{
  constexpr auto nullable = substrait::Type::NULLABILITY_NULLABLE;
  switch (kind) {
    case col_kind::I32: type->mutable_i32()->set_nullability(nullable); break;
    case col_kind::I64: type->mutable_i64()->set_nullability(nullable); break;
    case col_kind::F64: type->mutable_fp64()->set_nullability(nullable); break;
    case col_kind::BOOL: type->mutable_bool_()->set_nullability(nullable); break;
    case col_kind::STRING: type->mutable_string()->set_nullability(nullable); break;
    case col_kind::DECIMAL_15_2: {
      auto* dec = type->mutable_decimal();
      dec->set_precision(15);
      dec->set_scale(2);
      dec->set_nullability(nullable);
      break;
    }
    case col_kind::DATE: type->mutable_date()->set_nullability(nullable); break;
  }
}

// The DuckDB type name declare_input_column() takes for each kind.
const char* duckdb_type_name(col_kind kind)
{
  switch (kind) {
    case col_kind::I32: return "INTEGER";
    case col_kind::I64: return "BIGINT";
    case col_kind::F64: return "DOUBLE";
    case col_kind::BOOL: return "BOOLEAN";
    case col_kind::STRING: return "VARCHAR";
    case col_kind::DECIMAL_15_2: return "DECIMAL(15,2)";
    case col_kind::DATE: return "DATE";
  }
  return "";
}

// A plan whose only read is the engine's stream view for input stream `stream_id`, projecting
// the given nullable columns — the shape a front end emits where it would otherwise emit a file
// scan. Mirrors what Fragment::build() creates: `CREATE VIEW sirius_stream_<id> AS
// SELECT * FROM sirius_stream_source(<id>)`.
std::string stream_read_plan(std::uint64_t stream_id, const std::vector<plan_column>& columns)
{
  substrait::Plan plan;
  auto* root = plan.add_relations()->mutable_root();

  auto* read     = root->mutable_input()->mutable_read();
  auto* schema   = read->mutable_base_schema();
  auto* row_type = schema->mutable_struct_();
  row_type->set_nullability(substrait::Type::NULLABILITY_REQUIRED);
  for (const auto& column : columns) {
    root->add_names(column.name);
    schema->add_names(column.name);
    set_plan_type(row_type->add_types(), column.kind);
  }
  read->mutable_named_table()->add_names(*sirius::ffi::stream_view_name(stream_id));

  return plan.SerializeAsString();
}

// The original single-column shape: one nullable INTEGER column `a`.
std::string stream_read_plan(std::uint64_t stream_id)
{
  return stream_read_plan(stream_id, {{"a", col_kind::I32}});
}

// Declares every column of `columns` on input stream `stream_id` of `fragment`.
void declare_columns(sirius::ffi::Fragment& fragment,
                     std::uint64_t stream_id,
                     const std::vector<plan_column>& columns)
{
  for (const auto& column : columns) {
    fragment.declare_input_column(stream_id, column.name, duckdb_type_name(column.kind));
  }
}

// ---------------------------------------------------------------------------
// cudf table construction (host vector -> device column), no cudf_test dependency.
// ---------------------------------------------------------------------------

template <typename T>
std::unique_ptr<cudf::column> fixed_width_column(cudf::data_type type, const std::vector<T>& values)
{
  const auto n = static_cast<cudf::size_type>(values.size());
  auto stream  = cudf::get_default_stream();
  std::unique_ptr<cudf::column> column;
  if (cudf::is_fixed_point(type)) {
    column = cudf::make_fixed_point_column(type, n, cudf::mask_state::UNALLOCATED, stream);
  } else if (cudf::is_timestamp(type)) {
    column = cudf::make_timestamp_column(type, n, cudf::mask_state::UNALLOCATED, stream);
  } else {
    column = cudf::make_numeric_column(type, n, cudf::mask_state::UNALLOCATED, stream);
  }
  REQUIRE(cudaMemcpyAsync(column->mutable_view().head<void>(),
                          values.data(),
                          values.size() * sizeof(T),
                          cudaMemcpyHostToDevice,
                          stream.value()) == cudaSuccess);
  stream.synchronize();
  return column;
}

std::unique_ptr<cudf::column> strings_column(const std::vector<std::string>& values)
{
  std::vector<std::int32_t> offsets{0};
  std::string chars;
  for (const auto& value : values) {
    chars += value;
    offsets.push_back(static_cast<std::int32_t>(chars.size()));
  }
  auto stream = cudf::get_default_stream();
  rmm::device_buffer chars_buffer(chars.data(), chars.size(), stream);
  auto offsets_column = fixed_width_column(cudf::data_type{cudf::type_id::INT32}, offsets);
  return cudf::make_strings_column(static_cast<cudf::size_type>(values.size()),
                                   std::move(offsets_column),
                                   std::move(chars_buffer),
                                   0,
                                   rmm::device_buffer{});
}

// The TPC-H-shaped fixture: BIGINT, DOUBLE, BOOLEAN, VARCHAR, DECIMAL(15,2), DATE. Row i carries
// id=i, x=i*0.5, flag=(i odd), name="n<i>", price=(i*100+25) scaled by 2, d=(days) 19000+i.
const std::vector<plan_column>& fixture_columns()
{
  static const std::vector<plan_column> columns{{"id", col_kind::I64},
                                                {"x", col_kind::F64},
                                                {"flag", col_kind::BOOL},
                                                {"name", col_kind::STRING},
                                                {"price", col_kind::DECIMAL_15_2},
                                                {"d", col_kind::DATE}};
  return columns;
}

struct fixture_row {
  std::int64_t id;
  double x;
  bool flag;
  std::string name;
  std::int64_t price_scaled;
  std::int32_t days;
  auto operator<=>(const fixture_row&) const = default;
};

std::vector<fixture_row> fixture_rows(std::int64_t n)
{
  std::vector<fixture_row> rows;
  for (std::int64_t i = 0; i < n; ++i) {
    rows.push_back({i,
                    static_cast<double>(i) * 0.5,
                    (i % 2) == 1,
                    "n" + std::to_string(i),
                    i * 100 + 25,
                    static_cast<std::int32_t>(19000 + i)});
  }
  return rows;
}

std::unique_ptr<cudf::table> fixture_table(const std::vector<fixture_row>& rows)
{
  std::vector<std::int64_t> ids, prices;
  std::vector<double> xs;
  std::vector<std::int8_t> flags;
  std::vector<std::string> names;
  std::vector<std::int32_t> days;
  for (const auto& row : rows) {
    ids.push_back(row.id);
    xs.push_back(row.x);
    flags.push_back(row.flag ? 1 : 0);
    names.push_back(row.name);
    prices.push_back(row.price_scaled);
    days.push_back(row.days);
  }
  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(fixed_width_column(cudf::data_type{cudf::type_id::INT64}, ids));
  columns.push_back(fixed_width_column(cudf::data_type{cudf::type_id::FLOAT64}, xs));
  columns.push_back(fixed_width_column(cudf::data_type{cudf::type_id::BOOL8}, flags));
  columns.push_back(strings_column(names));
  // cudf DECIMAL64 scale -2; to_arrow_schema() widens it to Arrow decimal128(18,2), so the hop
  // exercises the width reconciliation on the way back in.
  columns.push_back(fixed_width_column(cudf::data_type{cudf::type_id::DECIMAL64, -2}, prices));
  columns.push_back(fixed_width_column(cudf::data_type{cudf::type_id::TIMESTAMP_DAYS}, days));
  return std::make_unique<cudf::table>(std::move(columns));
}

// A host Arrow struct array + schema produced by cudf from a device table: exactly what an
// embedding host would hand to push_arrow(), except that the host built it on the CPU.
struct host_arrow_batch {
  cudf::unique_schema_t schema;
  cudf::unique_device_array_t array;

  std::uintptr_t schema_addr() const { return reinterpret_cast<std::uintptr_t>(schema.get()); }
  std::uintptr_t array_addr() const { return reinterpret_cast<std::uintptr_t>(&array->array); }
};

host_arrow_batch to_host_arrow(const cudf::table_view& table,
                               const std::vector<plan_column>& columns)
{
  std::vector<cudf::column_metadata> metadata;
  for (const auto& column : columns) {
    metadata.emplace_back(column.name);
  }
  host_arrow_batch batch{cudf::to_arrow_schema(table, metadata), cudf::to_arrow_host(table)};
  REQUIRE(batch.array->device_type == ARROW_DEVICE_CPU);
  return batch;
}

// Presents rows [offset, offset + length) of the host batch: every child gets that Arrow offset,
// the struct keeps offset 0 — the shape arrow-rs's RecordBatch::slice() hands a producer that
// splits one big batch into pieces. The buffers are untouched.
void slice_host_arrow(host_arrow_batch& host, std::int64_t offset, std::int64_t length)
{
  auto& array  = host.array->array;
  array.length = length;
  for (std::int64_t i = 0; i < array.n_children; ++i) {
    array.children[i]->offset = offset;
    array.children[i]->length = length;
  }
}

// Nullable twin of the fixture: row i is null in column c when (i + c) % 4 == 0, so every column
// carries nulls, each in different rows.
struct nullable_fixture_row {
  std::optional<std::int64_t> id;
  std::optional<double> x;
  std::optional<bool> flag;
  std::optional<std::string> name;
  std::optional<std::int64_t> price_scaled;
  std::optional<std::int32_t> days;
  auto operator<=>(const nullable_fixture_row&) const = default;
};

bool fixture_cell_is_null(std::int64_t row, int column) { return (row + column) % 4 == 0; }

std::vector<nullable_fixture_row> nullable_fixture_rows(std::int64_t n)
{
  std::vector<nullable_fixture_row> rows;
  for (const auto& r : fixture_rows(n)) {
    nullable_fixture_row row;
    if (!fixture_cell_is_null(r.id, 0)) { row.id = r.id; }
    if (!fixture_cell_is_null(r.id, 1)) { row.x = r.x; }
    if (!fixture_cell_is_null(r.id, 2)) { row.flag = r.flag; }
    if (!fixture_cell_is_null(r.id, 3)) { row.name = r.name; }
    if (!fixture_cell_is_null(r.id, 4)) { row.price_scaled = r.price_scaled; }
    if (!fixture_cell_is_null(r.id, 5)) { row.days = r.days; }
    rows.push_back(row);
  }
  return rows;
}

// The fixture table with a validity mask on every column (cudf layout, one bit per row).
std::unique_ptr<cudf::table> nullable_fixture_table(std::int64_t n)
{
  auto columns = fixture_table(fixture_rows(n))->release();
  auto stream  = cudf::get_default_stream();
  for (int c = 0; c < static_cast<int>(columns.size()); ++c) {
    const auto size = static_cast<cudf::size_type>(n);
    std::vector<cudf::bitmask_type> bits(
      cudf::bitmask_allocation_size_bytes(size) / sizeof(cudf::bitmask_type), 0);
    cudf::size_type null_count = 0;
    for (std::int64_t i = 0; i < n; ++i) {
      if (fixture_cell_is_null(i, c)) {
        ++null_count;
        continue;
      }
      bits[i / 32] |= cudf::bitmask_type{1} << (i % 32);
    }
    rmm::device_buffer mask(bits.data(), bits.size() * sizeof(cudf::bitmask_type), stream);
    stream.synchronize();
    columns[c]->set_null_mask(std::move(mask), null_count);
  }
  return std::make_unique<cudf::table>(std::move(columns));
}

// ---------------------------------------------------------------------------
// Reading the result back out of result_to_arrow()'s ArrowArrayStream.
// ---------------------------------------------------------------------------

struct arrow_stream_guard {
  ArrowArrayStream stream{};
  ~arrow_stream_guard()
  {
    if (stream.release != nullptr) { stream.release(&stream); }
  }
};

struct arrow_array_guard {
  ArrowArray array{};
  ~arrow_array_guard()
  {
    if (array.release != nullptr) { array.release(&array); }
  }
};

struct arrow_schema_guard {
  ArrowSchema schema{};
  ~arrow_schema_guard()
  {
    if (schema.release != nullptr) { schema.release(&schema); }
  }
};

template <typename T>
T fixed_width_at(const ArrowArray& column, std::int64_t row)
{
  return static_cast<const T*>(column.buffers[1])[column.offset + row];
}

bool bit_at(const ArrowArray& column, std::int64_t row)
{
  const auto bit = column.offset + row;
  return (static_cast<const std::uint8_t*>(column.buffers[1])[bit / 8] >> (bit % 8)) & 1;
}

std::string string_at(const ArrowArray& column, std::int64_t row)
{
  const auto* offsets = static_cast<const std::int32_t*>(column.buffers[1]);
  const auto* chars   = static_cast<const char*>(column.buffers[2]);
  const auto begin    = offsets[column.offset + row];
  const auto end      = offsets[column.offset + row + 1];
  return std::string(chars + begin, chars + end);
}

bool is_valid(const ArrowArray& column, std::int64_t row)
{
  if (column.buffers[0] == nullptr) { return true; }
  const auto bit = column.offset + row;
  return (static_cast<const std::uint8_t*>(column.buffers[0])[bit / 8] >> (bit % 8)) & 1;
}

// Checks the Arrow formats DuckDB produces for the fixture's declared types, so a type drift
// shows up here and not as garbage.
void require_fixture_schema(ArrowArrayStream& stream)
{
  arrow_schema_guard schema;
  REQUIRE(stream.get_schema(&stream, &schema.schema) == 0);
  REQUIRE(schema.schema.n_children == 6);
  // DuckDB spells the decimal with its explicit bit width ("d:15,2,128").
  const std::vector<std::string> expected_formats{"l", "g", "b", "u", "d:15,2,128", "tdD"};
  for (std::int64_t i = 0; i < schema.schema.n_children; ++i) {
    REQUIRE(std::string(schema.schema.children[i]->format) == expected_formats[i]);
  }
}

// Appends every row of one fixture-schema struct array. The two producers this file reads spell
// the decimal differently: DuckDB's result stream emits decimal128 (`d:15,2,128`, 16 bytes a
// row) and cudf's to_arrow_host keeps the cudf width (`d:18,2,64`, 8 bytes a row); either way
// the scaled value fits the low 64 bits for this fixture, so `decimal_words` (int64 words per
// value: 2 or 1) is all the reader needs.
void append_fixture_rows(const ArrowArray& array,
                         std::vector<fixture_row>& rows,
                         std::int64_t decimal_words)
{
  REQUIRE(array.n_children == 6);
  for (std::int64_t r = 0; r < array.length; ++r) {
    const auto& c   = array.children;
    const auto* dec = static_cast<const std::int64_t*>(c[4]->buffers[1]);
    rows.push_back({fixed_width_at<std::int64_t>(*c[0], r),
                    fixed_width_at<double>(*c[1], r),
                    bit_at(*c[2], r),
                    string_at(*c[3], r),
                    dec[(c[4]->offset + r) * decimal_words],
                    fixed_width_at<std::int32_t>(*c[5], r)});
  }
}

// int64 words per value of an Arrow decimal format: `d:p,s` and `d:p,s,128` are 128-bit, an
// explicit `,64` is 64-bit.
std::int64_t decimal_words_of(const char* format)
{
  const std::string spelled(format);
  return spelled.ends_with(",64") ? 1 : 2;
}

// Drains a result stream of the fixture schema into rows sorted by id.
std::vector<fixture_row> read_fixture_result(ArrowArrayStream& stream)
{
  require_fixture_schema(stream);

  std::vector<fixture_row> rows;
  while (true) {
    arrow_array_guard array;
    REQUIRE(stream.get_next(&stream, &array.array) == 0);
    if (array.array.release == nullptr) { break; }
    append_fixture_rows(array.array, rows, /*decimal_words=*/2);
  }
  std::sort(rows.begin(), rows.end(), [](const auto& a, const auto& b) { return a.id < b.id; });
  return rows;
}

// The nullable reader: a cell is read only where DuckDB's validity bitmap says so. Sorted on the
// whole row, since the id itself may be null.
std::vector<nullable_fixture_row> read_nullable_fixture_result(ArrowArrayStream& stream)
{
  require_fixture_schema(stream);

  std::vector<nullable_fixture_row> rows;
  while (true) {
    arrow_array_guard array;
    REQUIRE(stream.get_next(&stream, &array.array) == 0);
    if (array.array.release == nullptr) { break; }
    REQUIRE(array.array.n_children == 6);
    for (std::int64_t r = 0; r < array.array.length; ++r) {
      const auto& c = array.array.children;
      nullable_fixture_row row;
      if (is_valid(*c[0], r)) { row.id = fixed_width_at<std::int64_t>(*c[0], r); }
      if (is_valid(*c[1], r)) { row.x = fixed_width_at<double>(*c[1], r); }
      if (is_valid(*c[2], r)) { row.flag = bit_at(*c[2], r); }
      if (is_valid(*c[3], r)) { row.name = string_at(*c[3], r); }
      if (is_valid(*c[4], r)) {
        const auto* dec  = static_cast<const std::int64_t*>(c[4]->buffers[1]);
        row.price_scaled = dec[(c[4]->offset + r) * 2];
      }
      if (is_valid(*c[5], r)) { row.days = fixed_width_at<std::int32_t>(*c[5], r); }
      rows.push_back(row);
    }
  }
  std::sort(rows.begin(), rows.end());
  return rows;
}

// A result fragment over input stream 0 declared with `columns`, built and ready for a push.
std::unique_ptr<sirius::ffi::Fragment> build_result_fragment(
  sirius::ffi::Context& context, const std::vector<plan_column>& columns)
{
  auto fragment = sirius::ffi::make_fragment(context);
  declare_columns(*fragment, 0, columns);
  fragment->build(stream_read_plan(0, columns));
  return fragment;
}

// ---------------------------------------------------------------------------
// Hand-built Arrow C structs for the by-name refusals: the helper inspects the schema before it
// imports anything, so the array only has to be a well-formed empty struct.
// ---------------------------------------------------------------------------

// A live Arrow struct has a non-null `release`; the helper refuses one that was already released.
void noop_release_schema(ArrowSchema*) {}
void noop_release_array(ArrowArray*) {}

struct handmade_arrow_column {
  ArrowSchema schema{};
  ArrowSchema* schema_children[1]{};
  ArrowSchema child{};
  ArrowSchema dictionary{};
  ArrowSchema grandchild{};
  ArrowSchema* child_children[1]{};
  ArrowArray array{};
  ArrowArray* array_children[1]{};
  ArrowArray child_array{};
  const void* child_buffers[3]{};
  const void* struct_buffers[1]{};

  handmade_arrow_column(const char* child_format,
                        bool with_dictionary    = false,
                        const char* item_format = nullptr)
  {
    schema.format      = "+s";
    schema.name        = "";
    schema.n_children  = 1;
    schema_children[0] = &child;
    schema.children    = schema_children;
    schema.release     = noop_release_schema;
    array.release      = noop_release_array;

    child.format = child_format;
    child.name   = "c";
    child.flags  = ARROW_FLAG_NULLABLE;
    if (with_dictionary) {
      dictionary.format = "u";
      dictionary.name   = "";
      child.dictionary  = &dictionary;
    }
    if (item_format != nullptr) {
      grandchild.format = item_format;
      grandchild.name   = "item";
      child_children[0] = &grandchild;
      child.children    = child_children;
      child.n_children  = 1;
    }

    array.length      = 0;
    array.n_buffers   = 1;
    array.buffers     = struct_buffers;
    array.n_children  = 1;
    array_children[0] = &child_array;
    array.children    = array_children;

    child_array.length    = 0;
    child_array.n_buffers = 2;
    child_array.buffers   = child_buffers;
  }

  std::uintptr_t schema_addr() const { return reinterpret_cast<std::uintptr_t>(&schema); }
  std::uintptr_t array_addr() const { return reinterpret_cast<std::uintptr_t>(&array); }
};

std::string import_error(const handmade_arrow_column& column, sirius::logical_type declared)
{
  const std::vector<std::string> names{"c"};
  const std::vector<sirius::logical_type> types{declared};
  try {
    sirius::import_arrow_host_table(reinterpret_cast<const ArrowSchema*>(column.schema_addr()),
                                    reinterpret_cast<const ArrowArray*>(column.array_addr()),
                                    "test batch",
                                    names,
                                    types,
                                    cudf::get_default_stream(),
                                    cudf::get_current_device_resource_ref());
  } catch (const std::exception& e) {
    return e.what();
  }
  return "";
}

// Both TransactionContext::Commit() and ::Rollback() clear current_transaction before doing any
// work that can throw, so "does BeginTransaction() work afterward" cannot tell the old
// commit-on-failure bug apart from the fix. What both bugs share is that a broken/skipped
// end_lifecycle() would leave the transaction open, and the next BeginTransaction() would then
// throw TransactionException("cannot start a transaction within a transaction") — that's the
// one thing worth asserting against here.
void require_build_fails_without_transaction_exception(sirius::ffi::Fragment& fragment)
{
  bool threw_transaction_exception = false;
  bool threw_other                 = false;
  try {
    fragment.build("");
  } catch (const duckdb::TransactionException&) {
    threw_transaction_exception = true;
  } catch (...) {
    threw_other = true;
  }
  REQUIRE_FALSE(threw_transaction_exception);
  REQUIRE(threw_other);
}

}  // namespace

TEST_CASE("Fragment::build() failure during resolve_inputs() rolls back cleanly",
          "[isolated_context][sirius_ffi]")
{
  auto context = sirius::ffi::make_context_from_config(isolated_memory_config_path().string());

  auto first = sirius::ffi::make_fragment(*context);
  declare_unresolvable_column(*first, "not_a_real_type_xyz");
  REQUIRE_THROWS(first->build(""));

  auto second = sirius::ffi::make_fragment(*context);
  declare_unresolvable_column(*second, "also_not_a_real_type_xyz");
  require_build_fails_without_transaction_exception(*second);
}

TEST_CASE("Fragment destroyed between a failed build() and reuse also closes the lifecycle cleanly",
          "[isolated_context][sirius_ffi]")
{
  auto context = sirius::ffi::make_context_from_config(isolated_memory_config_path().string());

  // Exercises ~Fragment::Impl() -> end_lifecycle() (rather than the catch-block call in
  // build()): the failed fragment goes out of scope with no further use.
  {
    auto first = sirius::ffi::make_fragment(*context);
    declare_unresolvable_column(*first, "not_a_real_type_xyz");
    REQUIRE_THROWS(first->build(""));
  }

  auto second = sirius::ffi::make_fragment(*context);
  declare_unresolvable_column(*second, "also_not_a_real_type_xyz");
  require_build_fails_without_transaction_exception(*second);
}

// Exercises the success path the two cases above never reach: a well-formed plan over a declared
// input stream. Fragment::build() commits its own transaction (type resolution + CREATE VIEW)
// before opening the query lifecycle and only then calls lower_substrait(), so the Substrait
// consumer's view lookup and the planner's binding run with no ambient transaction. Without
// lower_substrait() owning one, DuckDB 1.5.5 fails this build() with
// "TransactionContext::ActiveTransaction called without active transaction".
TEST_CASE("Fragment::build() lowers its Substrait plan without an ambient transaction",
          "[isolated_context][sirius_ffi]")
{
  auto context = sirius::ffi::make_context_from_config(isolated_memory_config_path().string());

  {
    auto first = sirius::ffi::make_fragment(*context);
    first->declare_input_column(0, "a", "INTEGER");
    REQUIRE_NOTHROW(first->build(stream_read_plan(0)));
  }

  // The transaction lower_substrait() opened must be closed again on return: the next build()
  // begins its own on the same connection and would throw "cannot start a transaction within a
  // transaction" if the first one were still open.
  auto second = sirius::ffi::make_fragment(*context);
  second->declare_input_column(1, "a", "INTEGER");
  REQUIRE_NOTHROW(second->build(stream_read_plan(1)));
}

// ---------------------------------------------------------------------------
// Fragment::push_arrow
// ---------------------------------------------------------------------------

// The round trip the Doris proposal describes: build the input with cudf::to_arrow_host, push it
// as a host Arrow batch, run, and compare through result_to_arrow. Covers the TPC-H column types
// (BIGINT, DOUBLE, DECIMAL(15,2), DATE, VARCHAR) plus BOOLEAN, so the bool bitmap -> BOOL8 and
// decimal128 -> DECIMAL64 reconciliations are both exercised.
TEST_CASE("Fragment::push_arrow imports a host Arrow batch and the fragment returns its rows",
          "[isolated_context][sirius_ffi]")
{
  auto context    = sirius::ffi::make_context_from_config(isolated_memory_config_path().string());
  const auto rows = fixture_rows(7);
  auto host       = to_host_arrow(fixture_table(rows)->view(), fixture_columns());

  auto fragment = build_result_fragment(*context, fixture_columns());
  REQUIRE_NOTHROW(fragment->push_arrow(0, 0, host.array_addr(), host.schema_addr()));
  // The buffers were copied to the GPU before push_arrow returned: the caller may release the
  // Arrow structs right away, and the engine must not have kept a pointer into them.
  host.array.reset();
  host.schema.reset();

  fragment->close_input(0, 0);
  REQUIRE_NOTHROW(fragment->run());

  arrow_stream_guard out;
  fragment->result_to_arrow(reinterpret_cast<std::uintptr_t>(&out.stream));
  REQUIRE(read_fixture_result(out.stream) == rows);
}

// Two batches from two declared senders land in one stream; the result is their union.
TEST_CASE("Fragment::push_arrow accepts several batches and senders on one stream",
          "[isolated_context][sirius_ffi]")
{
  auto context = sirius::ffi::make_context_from_config(isolated_memory_config_path().string());
  auto all     = fixture_rows(10);
  const std::vector<fixture_row> first(all.begin(), all.begin() + 4);
  const std::vector<fixture_row> second(all.begin() + 4, all.end());
  const auto host_first  = to_host_arrow(fixture_table(first)->view(), fixture_columns());
  const auto host_second = to_host_arrow(fixture_table(second)->view(), fixture_columns());

  auto fragment = sirius::ffi::make_fragment(*context);
  declare_columns(*fragment, 0, fixture_columns());
  fragment->declare_input_sender(0, 1);
  fragment->declare_input_sender(0, 2);
  fragment->build(stream_read_plan(0, fixture_columns()));

  fragment->push_arrow(0, 1, host_first.array_addr(), host_first.schema_addr());
  fragment->push_arrow(0, 2, host_second.array_addr(), host_second.schema_addr());
  fragment->close_input(0, 1);
  // Sender 2 is still open: the stream has not ended, so a late batch from it is still legal,
  // and so is one that names the already-closed sender 1. sender_id is a membership check only
  // (sirius_ffi.hpp, push_arrow: the batch carries no sender identity past the call), so a push
  // from a closed sender is refused only once every sender has closed and the stream ended.
  fragment->push_arrow(0, 2, host_first.array_addr(), host_first.schema_addr());
  fragment->push_arrow(0, 1, host_second.array_addr(), host_second.schema_addr());
  fragment->close_input(0, 2);
  fragment->run();

  arrow_stream_guard out;
  fragment->result_to_arrow(reinterpret_cast<std::uintptr_t>(&out.stream));
  auto expected = all;
  expected.insert(expected.end(), all.begin(), all.end());
  std::sort(
    expected.begin(), expected.end(), [](const auto& a, const auto& b) { return a.id < b.id; });
  REQUIRE(read_fixture_result(out.stream) == expected);
}

TEST_CASE("Fragment::push_arrow refuses a batch whose schema disagrees with the declared stream",
          "[isolated_context][sirius_ffi]")
{
  auto context    = sirius::ffi::make_context_from_config(isolated_memory_config_path().string());
  const auto host = to_host_arrow(fixture_table(fixture_rows(3))->view(), fixture_columns());

  SECTION("a column type mismatch names the column and both types")
  {
    // Declares `id` as DOUBLE where the batch carries int64.
    auto columns    = fixture_columns();
    columns[0].kind = col_kind::F64;
    auto fragment   = build_result_fragment(*context, columns);
    REQUIRE_THROWS_WITH(fragment->push_arrow(0, 0, host.array_addr(), host.schema_addr()),
                        Catch::Matchers::Contains("column 0 (id)") &&
                          Catch::Matchers::Contains("declared DOUBLE") &&
                          Catch::Matchers::Contains("int64"));
  }

  SECTION("a column count mismatch is refused before any import")
  {
    const std::vector<plan_column> two{{"id", col_kind::I64}, {"x", col_kind::F64}};
    auto fragment = build_result_fragment(*context, two);
    REQUIRE_THROWS_WITH(
      fragment->push_arrow(0, 0, host.array_addr(), host.schema_addr()),
      Catch::Matchers::Contains("carries 6 columns") && Catch::Matchers::Contains("declares 2"));
  }

  SECTION("an unknown input stream is refused")
  {
    auto fragment = build_result_fragment(*context, fixture_columns());
    REQUIRE_THROWS_WITH(
      fragment->push_arrow(42, 0, host.array_addr(), host.schema_addr()),
      Catch::Matchers::Contains("42") && Catch::Matchers::Contains("never declared"));
  }

  SECTION("an undeclared sender is refused")
  {
    auto fragment = sirius::ffi::make_fragment(*context);
    declare_columns(*fragment, 0, fixture_columns());
    fragment->declare_input_sender(0, 1);
    fragment->build(stream_read_plan(0, fixture_columns()));
    REQUIRE_THROWS_WITH(fragment->push_arrow(0, 7, host.array_addr(), host.schema_addr()),
                        Catch::Matchers::Contains("sender 7"));
  }

  SECTION("null Arrow pointers are refused")
  {
    auto fragment = build_result_fragment(*context, fixture_columns());
    REQUIRE_THROWS_WITH(fragment->push_arrow(0, 0, 0, host.schema_addr()),
                        Catch::Matchers::Contains("non-null"));
    REQUIRE_THROWS_WITH(fragment->push_arrow(0, 0, host.array_addr(), 0),
                        Catch::Matchers::Contains("non-null"));
  }

  SECTION("a decimal whose scale disagrees is refused naming both scales")
  {
    // The helper narrows a decimal128 to the declared width when only the width differs; a scale
    // difference must not be mistaken for that case, so the message names the scales.
    const std::vector<plan_column> price{{"price", col_kind::DECIMAL_15_2}};
    std::vector<std::unique_ptr<cudf::column>> columns;
    columns.push_back(fixed_width_column(cudf::data_type{cudf::type_id::DECIMAL64, -3},
                                         std::vector<std::int64_t>{1234, 5678}));
    const cudf::table scale_3(std::move(columns));
    const auto host_scale_3 = to_host_arrow(scale_3.view(), price);

    auto fragment = build_result_fragment(*context, price);
    REQUIRE_THROWS_WITH(
      fragment->push_arrow(0, 0, host_scale_3.array_addr(), host_scale_3.schema_addr()),
      Catch::Matchers::Contains("column 0 (price)") &&
        Catch::Matchers::Contains("declared scale 2") &&
        Catch::Matchers::Contains("carries scale 3"));
  }
}

// A string kernel over the pushed VARCHAR column. Hash-partitioning on `name` runs
// crc32_partition_hash over the strings cudf imported, and that kernel refuses INT64 (large)
// string offsets (src/op/partition/crc32_partition_hash.cu). This is the check that keeping
// cudf's `utf8` -> 32-bit offsets, and refusing `large_utf8` by name, is what the engine needs.
TEST_CASE("Fragment::push_arrow feeds a hash partition keyed on the pushed VARCHAR column",
          "[isolated_context][sirius_ffi]")
{
  auto context    = sirius::ffi::make_context_from_config(isolated_memory_config_path().string());
  const auto rows = fixture_rows(64);
  const auto host = to_host_arrow(fixture_table(rows)->view(), fixture_columns());

  auto sender = sirius::ffi::make_fragment(*context);
  declare_columns(*sender, 0, fixture_columns());
  sender->declare_output(10);
  sender->declare_output(11);
  sender->declare_output_hash_key(3);  // `name`, the VARCHAR column
  sender->build(stream_read_plan(0, fixture_columns()));
  sender->push_arrow(0, 0, host.array_addr(), host.schema_addr());
  sender->close_input(0, 0);
  REQUIRE_NOTHROW(sender->run());
  REQUIRE(sender->output_row_count(10) + sender->output_row_count(11) == rows.size());

  // Drain both partitions through relay_from into result fragments: the union is the input.
  std::vector<fixture_row> drained;
  for (const auto output : {std::uint64_t{10}, std::uint64_t{11}}) {
    auto receiver = build_result_fragment(*context, fixture_columns());
    receiver->relay_from(*sender, output, 0, 0);
    receiver->run();
    arrow_stream_guard out;
    receiver->result_to_arrow(reinterpret_cast<std::uintptr_t>(&out.stream));
    const auto part = read_fixture_result(out.stream);
    drained.insert(drained.end(), part.begin(), part.end());
  }
  std::sort(
    drained.begin(), drained.end(), [](const auto& a, const auto& b) { return a.id < b.id; });
  REQUIRE(drained == rows);
}

// The sender-side twin: an intermediate fragment's parked output, exported batch by batch as
// host Arrow through export_arrow, carries exactly the rows the same output yields through
// relay_from — read straight off the exported struct arrays, and again after push_arrow into a
// receiver (the hop a transport makes: export_arrow on one CN, push_arrow on the other).
TEST_CASE("Fragment::export_arrow exports a parked output batch equal to the relay_from rows",
          "[isolated_context][sirius_ffi]")
{
  auto context    = sirius::ffi::make_context_from_config(isolated_memory_config_path().string());
  const auto rows = fixture_rows(64);
  const auto host = to_host_arrow(fixture_table(rows)->view(), fixture_columns());
  const auto addr = [](auto& c_struct) { return reinterpret_cast<std::uintptr_t>(&c_struct); };

  // An intermediate fragment over the fixture, its output parked on stream 10.
  const auto make_sender = [&] {
    auto sender = sirius::ffi::make_fragment(*context);
    declare_columns(*sender, 0, fixture_columns());
    sender->declare_output(10);
    sender->build(stream_read_plan(0, fixture_columns()));
    sender->push_arrow(0, 0, host.array_addr(), host.schema_addr());
    sender->close_input(0, 0);
    sender->run();
    return sender;
  };

  // Reference: the proven in-process relay into a result fragment.
  std::vector<fixture_row> relayed;
  {
    auto sender   = make_sender();
    auto receiver = build_result_fragment(*context, fixture_columns());
    REQUIRE(receiver->relay_from(*sender, 10, 0, 0) > 0);
    receiver->run();
    arrow_stream_guard out;
    receiver->result_to_arrow(reinterpret_cast<std::uintptr_t>(&out.stream));
    relayed = read_fixture_result(out.stream);
  }
  REQUIRE(relayed == rows);

  auto sender       = make_sender();
  const auto parked = sender->output_batch_count(10);
  REQUIRE(parked > 0);
  REQUIRE(sender->output_row_count(10) == rows.size());

  auto receiver = build_result_fragment(*context, fixture_columns());
  std::vector<fixture_row> exported;
  std::size_t batches      = 0;
  std::uint64_t total_rows = 0;
  while (true) {
    arrow_array_guard array;
    arrow_schema_guard schema;
    std::uint64_t n = 0;
    if (!sender->export_arrow(10, addr(array.array), addr(schema.schema), n)) { break; }
    ++batches;
    total_rows += n;
    // The caller owns live structs: a struct array over the fixture columns, in cudf's Arrow
    // spelling (the DECIMAL(15,2) column keeps its cudf width, decimal64 at cudf's widest
    // precision for it; no column names).
    REQUIRE(array.array.release != nullptr);
    REQUIRE(schema.schema.release != nullptr);
    REQUIRE(std::string(schema.schema.format) == "+s");
    REQUIRE(schema.schema.n_children == 6);
    const std::vector<std::string> expected_formats{"l", "g", "b", "u", "d:18,2,64", "tdD"};
    for (std::int64_t i = 0; i < schema.schema.n_children; ++i) {
      REQUIRE(std::string(schema.schema.children[i]->format) == expected_formats[i]);
      REQUIRE(std::string(schema.schema.children[i]->name) == "");
    }
    REQUIRE(static_cast<std::uint64_t>(array.array.length) == n);
    append_fixture_rows(array.array, exported, decimal_words_of(schema.schema.children[4]->format));
    // The same structs feed the receive side; the guards release them after the push copied.
    REQUIRE_NOTHROW(receiver->push_arrow(0, 0, addr(array.array), addr(schema.schema)));
  }
  REQUIRE(batches == parked);
  REQUIRE(total_rows == rows.size());
  REQUIRE(sender->output_batch_count(10) == 0);

  // Drained: false; `rows` is written as 0 (export_packed's drained behaviour) and the caller's
  // structs are left untouched — no live release callback.
  {
    arrow_array_guard array;
    arrow_schema_guard schema;
    std::uint64_t n = 7;
    REQUIRE_FALSE(sender->export_arrow(10, addr(array.array), addr(schema.schema), n));
    REQUIRE(n == 0);
    REQUIRE(array.array.release == nullptr);
    REQUIRE(schema.schema.release == nullptr);
  }

  std::sort(
    exported.begin(), exported.end(), [](const auto& a, const auto& b) { return a.id < b.id; });
  REQUIRE(exported == relayed);

  receiver->close_input(0, 0);
  receiver->run();
  arrow_stream_guard out;
  receiver->result_to_arrow(reinterpret_cast<std::uintptr_t>(&out.stream));
  REQUIRE(read_fixture_result(out.stream) == relayed);

  // Refusals: an output stream the fragment never declared, and null addresses.
  {
    arrow_array_guard array;
    arrow_schema_guard schema;
    std::uint64_t n = 0;
    REQUIRE_THROWS(sender->export_arrow(42, addr(array.array), addr(schema.schema), n));
    REQUIRE_THROWS_WITH(sender->export_arrow(10, 0, addr(schema.schema), n),
                        Catch::Matchers::Contains("non-null"));
  }
}

// The q15 empty-split shape on the Arrow hop: a sender whose only input batch has zero rows parks
// a zero-row batch, which export_arrow hands over as a zero-length struct array (rows 0, live
// structs — not false), and push_arrow on the receiving side imports it at length 0 into an
// empty stream: not an error, and the receiver terminates with zero rows. Both imports here
// (into the sender, and of the exported structs into the receiver) run the struct-window and
// `cudf::from_arrow` paths at length 0.
TEST_CASE("Fragment::export_arrow exports a zero-row parked batch as an empty struct array",
          "[isolated_context][sirius_ffi]")
{
  auto context    = sirius::ffi::make_context_from_config(isolated_memory_config_path().string());
  const auto host = to_host_arrow(fixture_table(fixture_rows(0))->view(), fixture_columns());
  const auto addr = [](auto& c_struct) { return reinterpret_cast<std::uintptr_t>(&c_struct); };
  REQUIRE(host.array->array.length == 0);

  auto sender = sirius::ffi::make_fragment(*context);
  declare_columns(*sender, 0, fixture_columns());
  sender->declare_output(10);
  sender->build(stream_read_plan(0, fixture_columns()));
  sender->push_arrow(0, 0, host.array_addr(), host.schema_addr());
  sender->close_input(0, 0);
  sender->run();
  REQUIRE(sender->output_row_count(10) == 0);
  REQUIRE(sender->output_batch_count(10) == 1);

  auto receiver = build_result_fragment(*context, fixture_columns());
  {
    arrow_array_guard array;
    arrow_schema_guard schema;
    std::uint64_t n = 7;
    REQUIRE(sender->export_arrow(10, addr(array.array), addr(schema.schema), n));
    REQUIRE(n == 0);
    REQUIRE(array.array.release != nullptr);
    REQUIRE(schema.schema.release != nullptr);
    REQUIRE(std::string(schema.schema.format) == "+s");
    REQUIRE(array.array.length == 0);
    REQUIRE(array.array.n_children == 6);
    REQUIRE(schema.schema.n_children == 6);
    for (std::int64_t i = 0; i < array.array.n_children; ++i) {
      REQUIRE(array.array.children[i]->length == 0);
    }
    std::vector<fixture_row> exported;
    append_fixture_rows(array.array, exported, decimal_words_of(schema.schema.children[4]->format));
    REQUIRE(exported.empty());
    REQUIRE_NOTHROW(receiver->push_arrow(0, 0, addr(array.array), addr(schema.schema)));
  }
  REQUIRE(sender->output_batch_count(10) == 0);
  {
    arrow_array_guard array;
    arrow_schema_guard schema;
    std::uint64_t n = 7;
    REQUIRE_FALSE(sender->export_arrow(10, addr(array.array), addr(schema.schema), n));
    REQUIRE(n == 0);
    REQUIRE(array.array.release == nullptr);
  }

  receiver->close_input(0, 0);
  receiver->run();
  arrow_stream_guard out;
  receiver->result_to_arrow(reinterpret_cast<std::uintptr_t>(&out.stream));
  REQUIRE(read_fixture_result(out.stream).empty());
}

// A producer that splits one big batch into pieces hands over arrays with a non-zero Arrow
// offset on every child (arrow-rs RecordBatch::slice()). cudf::from_arrow must honour the offsets
// for every fixture type, the bool bitmap and the string offsets included.
TEST_CASE("Fragment::push_arrow honours Arrow offsets on a sliced host batch",
          "[isolated_context][sirius_ffi]")
{
  auto context    = sirius::ffi::make_context_from_config(isolated_memory_config_path().string());
  const auto rows = fixture_rows(10);
  auto host       = to_host_arrow(fixture_table(rows)->view(), fixture_columns());

  auto fragment = build_result_fragment(*context, fixture_columns());
  slice_host_arrow(host, 0, 4);
  fragment->push_arrow(0, 0, host.array_addr(), host.schema_addr());
  slice_host_arrow(host, 4, 6);
  fragment->push_arrow(0, 0, host.array_addr(), host.schema_addr());
  fragment->close_input(0, 0);
  REQUIRE_NOTHROW(fragment->run());

  arrow_stream_guard out;
  fragment->result_to_arrow(reinterpret_cast<std::uintptr_t>(&out.stream));
  REQUIRE(read_fixture_result(out.stream) == rows);
}

// The other way to slice: Arrow C++'s StructArray::Slice() (or any producer that windows the
// batch rather than its columns) puts the offset and length on the struct and leaves the children
// whole; a struct shorter than its children is the same shape with offset 0. cudf::from_arrow
// imports each child by the child's own offset and length, so the helper pushes the struct's
// window into the children first; without that, every child row came back (10 for a 6-row slice).
TEST_CASE("Fragment::push_arrow honours a slice taken on the struct itself",
          "[isolated_context][sirius_ffi]")
{
  auto context    = sirius::ffi::make_context_from_config(isolated_memory_config_path().string());
  const auto rows = fixture_rows(10);
  auto host       = to_host_arrow(fixture_table(rows)->view(), fixture_columns());
  auto& top       = host.array->array;
  REQUIRE(top.length == 10);

  auto fragment = build_result_fragment(*context, fixture_columns());
  // Rows [4, 10): the offset sits on the struct; the children keep offset 0 and length 10.
  top.offset = 4;
  top.length = 6;
  fragment->push_arrow(0, 0, host.array_addr(), host.schema_addr());
  // Rows [0, 6): a struct shorter than its children.
  top.offset = 0;
  top.length = 6;
  fragment->push_arrow(0, 0, host.array_addr(), host.schema_addr());
  fragment->close_input(0, 0);
  REQUIRE_NOTHROW(fragment->run());

  std::vector<fixture_row> expected(rows.begin() + 4, rows.end());
  expected.insert(expected.end(), rows.begin(), rows.begin() + 6);
  std::sort(
    expected.begin(), expected.end(), [](const auto& a, const auto& b) { return a.id < b.id; });
  arrow_stream_guard out;
  fragment->result_to_arrow(reinterpret_cast<std::uintptr_t>(&out.stream));
  REQUIRE(read_fixture_result(out.stream) == expected);

  // A window that runs past the children is malformed Arrow: refused naming the column.
  top.offset    = 8;
  top.length    = 6;
  auto refusing = build_result_fragment(*context, fixture_columns());
  REQUIRE_THROWS_WITH(refusing->push_arrow(0, 0, host.array_addr(), host.schema_addr()),
                      Catch::Matchers::Contains("column 0 (id)") &&
                        Catch::Matchers::Contains("10 rows") &&
                        Catch::Matchers::Contains("[8, 14)"));
  top.offset = 0;
  top.length = 10;
}

// Nulls in every column: the decimal (the one column the helper rebuilds itself, decimal128 ->
// DECIMAL64 through cudf::cast), the bool bitmap and date32 included, which the Rust null test
// (Int64, Utf8) never reaches. The second push windows the struct over the nullable batch, so the
// per-child null count the helper recounts for the window is checked as well.
TEST_CASE("Fragment::push_arrow carries nulls in every fixture column, whole and struct-sliced",
          "[isolated_context][sirius_ffi]")
{
  auto context    = sirius::ffi::make_context_from_config(isolated_memory_config_path().string());
  const auto rows = nullable_fixture_rows(12);
  auto host       = to_host_arrow(nullable_fixture_table(12)->view(), fixture_columns());

  {
    auto fragment = build_result_fragment(*context, fixture_columns());
    fragment->push_arrow(0, 0, host.array_addr(), host.schema_addr());
    fragment->close_input(0, 0);
    REQUIRE_NOTHROW(fragment->run());
    auto expected = rows;
    std::sort(expected.begin(), expected.end());
    arrow_stream_guard out;
    fragment->result_to_arrow(reinterpret_cast<std::uintptr_t>(&out.stream));
    REQUIRE(read_nullable_fixture_result(out.stream) == expected);
  }
  {
    auto& top     = host.array->array;
    top.offset    = 3;
    top.length    = 5;
    auto fragment = build_result_fragment(*context, fixture_columns());
    fragment->push_arrow(0, 0, host.array_addr(), host.schema_addr());
    fragment->close_input(0, 0);
    REQUIRE_NOTHROW(fragment->run());
    std::vector<nullable_fixture_row> expected(rows.begin() + 3, rows.begin() + 8);
    std::sort(expected.begin(), expected.end());
    arrow_stream_guard out;
    fragment->result_to_arrow(reinterpret_cast<std::uintptr_t>(&out.stream));
    REQUIRE(read_nullable_fixture_result(out.stream) == expected);
    top.offset = 0;
    top.length = 12;
  }
}

TEST_CASE("Fragment::push_arrow before build() and after close_input() both throw",
          "[isolated_context][sirius_ffi]")
{
  auto context    = sirius::ffi::make_context_from_config(isolated_memory_config_path().string());
  const auto host = to_host_arrow(fixture_table(fixture_rows(3))->view(), fixture_columns());

  auto fragment = sirius::ffi::make_fragment(*context);
  declare_columns(*fragment, 0, fixture_columns());
  REQUIRE_THROWS_WITH(fragment->push_arrow(0, 0, host.array_addr(), host.schema_addr()),
                      Catch::Matchers::Contains("build() must run before push_arrow()"));

  fragment->build(stream_read_plan(0, fixture_columns()));
  fragment->push_arrow(0, 0, host.array_addr(), host.schema_addr());
  fragment->close_input(0, 0);
  // A push after EOS must refuse loudly, never vanish.
  REQUIRE_THROWS_WITH(fragment->push_arrow(0, 0, host.array_addr(), host.schema_addr()),
                      Catch::Matchers::Contains("already ended"));
  REQUIRE_NOTHROW(fragment->run());
}

// The threading contract the header states: push_arrow and close_input touch only the stream
// session and immutable post-build() state, so a producer thread other than the one that owns
// the Context may call them between build() and run(). The context thread only joins the
// producer and runs.
TEST_CASE("Fragment::push_arrow from a producer thread between build() and run()",
          "[isolated_context][sirius_ffi]")
{
  auto context    = sirius::ffi::make_context_from_config(isolated_memory_config_path().string());
  const auto rows = fixture_rows(9);
  const auto host = to_host_arrow(fixture_table(rows)->view(), fixture_columns());

  auto fragment = build_result_fragment(*context, fixture_columns());
  std::exception_ptr producer_error;
  std::thread producer([&] {
    try {
      fragment->push_arrow(0, 0, host.array_addr(), host.schema_addr());
      fragment->close_input(0, 0);
    } catch (...) {
      producer_error = std::current_exception();
    }
  });
  producer.join();
  if (producer_error) { std::rethrow_exception(producer_error); }

  REQUIRE_NOTHROW(fragment->run());
  arrow_stream_guard out;
  fragment->result_to_arrow(reinterpret_cast<std::uintptr_t>(&out.stream));
  REQUIRE(read_fixture_result(out.stream) == rows);
}

// The helper's by-name refusals: shapes cudf would import into something the engine cannot
// consume, or would import with silently changed meaning, are refused before any buffer is
// touched. Tested directly against the helper, without an engine context.
TEST_CASE("arrow_host_import refuses the shapes the engine cannot consume, by name",
          "[sirius_ffi][arrow_host_import]")
{
  using sirius::logical_type;
  using sirius::type_id;

  SECTION("timezone-aware timestamps")
  {
    handmade_arrow_column column("tsu:UTC");
    const auto what = import_error(column, logical_type::make(type_id::TIMESTAMP));
    CHECK_THAT(what, Catch::Matchers::Contains("column 0 (c)"));
    CHECK_THAT(what, Catch::Matchers::Contains("timezone"));
    CHECK_THAT(what, Catch::Matchers::Contains("UTC"));
  }

  SECTION("dictionary-encoded columns")
  {
    handmade_arrow_column column("i", /*with_dictionary=*/true);
    const auto what = import_error(column, logical_type::make(type_id::VARCHAR));
    CHECK_THAT(what, Catch::Matchers::Contains("column 0 (c)"));
    CHECK_THAT(what, Catch::Matchers::Contains("dictionary"));
  }

  SECTION("large_list")
  {
    handmade_arrow_column column("+L", false, "l");
    const auto what = import_error(column, logical_type::make(type_id::LIST));
    CHECK_THAT(what, Catch::Matchers::Contains("column 0 (c)"));
    CHECK_THAT(what, Catch::Matchers::Contains("large_list"));
  }

  SECTION("large_utf8 (64-bit string offsets)")
  {
    handmade_arrow_column column("U");
    const auto what = import_error(column, logical_type::make(type_id::VARCHAR));
    CHECK_THAT(what, Catch::Matchers::Contains("column 0 (c)"));
    CHECK_THAT(what, Catch::Matchers::Contains("large_utf8"));
  }

  SECTION("large_binary (64-bit offsets)")
  {
    handmade_arrow_column column("Z");
    const auto what = import_error(column, logical_type::make(type_id::VARCHAR));
    CHECK_THAT(what, Catch::Matchers::Contains("column 0 (c)"));
    CHECK_THAT(what, Catch::Matchers::Contains("large_binary"));
  }

  SECTION("already-released structs (release == NULL)")
  {
    handmade_arrow_column column("l");
    column.array.release = nullptr;
    CHECK_THAT(import_error(column, logical_type::make(type_id::BIGINT)),
               Catch::Matchers::Contains("already released"));
    column.array.release  = noop_release_array;
    column.schema.release = nullptr;
    CHECK_THAT(import_error(column, logical_type::make(type_id::BIGINT)),
               Catch::Matchers::Contains("already released"));
  }

  SECTION("128-bit integers declared as HUGEINT")
  {
    handmade_arrow_column column("d:38,0");
    const auto what = import_error(column, logical_type::make(type_id::HUGEINT));
    CHECK_THAT(what, Catch::Matchers::Contains("column 0 (c)"));
    CHECK_THAT(what, Catch::Matchers::Contains("HUGEINT"));
  }

  SECTION("decimal256")
  {
    handmade_arrow_column column("d:40,2,256");
    const auto what = import_error(column, logical_type::make_decimal(15, 2));
    CHECK_THAT(what, Catch::Matchers::Contains("column 0 (c)"));
    CHECK_THAT(what, Catch::Matchers::Contains("decimal256"));
  }

  SECTION("a scalar type mismatch is refused from the format alone, before any buffer is read")
  {
    // Three rows and no buffers at all: an import would read through null pointers, so this
    // message can only come from the format string.
    handmade_arrow_column column("l");
    column.array.length       = 3;
    column.child_array.length = 3;
    const auto what           = import_error(column, logical_type::make(type_id::DOUBLE));
    CHECK_THAT(what, Catch::Matchers::Contains("column 0 (c)"));
    CHECK_THAT(what, Catch::Matchers::Contains("declared DOUBLE"));
    CHECK_THAT(what, Catch::Matchers::Contains("int64"));
  }

  SECTION("a decimal scale mismatch is refused from the format alone, naming both scales")
  {
    handmade_arrow_column column("d:18,3");
    column.array.length       = 3;
    column.child_array.length = 3;
    const auto what           = import_error(column, logical_type::make_decimal(15, 2));
    CHECK_THAT(what, Catch::Matchers::Contains("column 0 (c)"));
    CHECK_THAT(what, Catch::Matchers::Contains("declared scale 2"));
    CHECK_THAT(what, Catch::Matchers::Contains("carries scale 3"));
  }

  SECTION("a non-struct top-level array")
  {
    handmade_arrow_column column("l");
    column.schema.format = "l";
    const auto what      = import_error(column, logical_type::make(type_id::BIGINT));
    CHECK_THAT(what, Catch::Matchers::Contains("struct"));
  }
}

// Bandwidth measurement (hidden; run with the [sirius_ffi_bench] tag on a free GPU): H2D bandwidth
// of push_arrow for int64/double batches in pageable host memory, against a plain cudaMemcpy of
// the same bytes; and D2H bandwidth of the result_to_arrow path (run() + drain, the four-copy
// collector) against cudf::to_arrow_host of the same table. Three sizes (128 MiB, 512 MiB, 2 GiB)
// and a zero-row run() separate the per-byte cost from the per-query fixed cost.
TEST_CASE("push_arrow / result_to_arrow bandwidth on 128 MiB, 512 MiB and 2 GiB batches",
          "[.][sirius_ffi_bench][isolated_context]")
{
  using clock        = std::chrono::steady_clock;
  const auto seconds = [](clock::time_point a, clock::time_point b) {
    return std::chrono::duration<double>(b - a).count();
  };
  const auto gbps = [](std::size_t bytes, double s) { return bytes / s / 1e9; };

  auto context = sirius::ffi::make_context();
  const std::vector<plan_column> columns{
    {"a", col_kind::I64}, {"b", col_kind::F64}, {"c", col_kind::I64}, {"d", col_kind::F64}};

  // The per-query fixed cost of run(): plan lowering, pipeline setup and scheduling over an input
  // that ended with no batches.
  {
    auto fragment = build_result_fragment(*context, columns);
    fragment->close_input(0, 0);
    const auto z0 = clock::now();
    fragment->run();
    const auto z1 = clock::now();
    std::fprintf(
      stderr, "[sirius_ffi_bench] run() with zero rows (fixed cost)  %.3f s\n", seconds(z0, z1));
  }

  for (const int shift : {22, 24, 26}) {
    const std::int64_t rows       = std::int64_t{1} << shift;  // x 4 columns x 8 B
    const std::size_t total_bytes = static_cast<std::size_t>(rows) * columns.size() * 8;

    std::vector<std::int64_t> ints(rows);
    std::vector<double> doubles(rows);
    for (std::int64_t i = 0; i < rows; ++i) {
      ints[i]    = i;
      doubles[i] = static_cast<double>(i) * 0.25;
    }
    std::vector<std::unique_ptr<cudf::column>> cols;
    cols.push_back(fixed_width_column(cudf::data_type{cudf::type_id::INT64}, ints));
    cols.push_back(fixed_width_column(cudf::data_type{cudf::type_id::FLOAT64}, doubles));
    cols.push_back(fixed_width_column(cudf::data_type{cudf::type_id::INT64}, ints));
    cols.push_back(fixed_width_column(cudf::data_type{cudf::type_id::FLOAT64}, doubles));
    auto table = std::make_unique<cudf::table>(std::move(cols));

    // D2H reference: cudf::to_arrow_host of the table.
    std::vector<cudf::column_metadata> metadata;
    for (const auto& column : columns) {
      metadata.emplace_back(column.name);
    }
    auto schema                  = cudf::to_arrow_schema(table->view(), metadata);
    const auto t0                = clock::now();
    auto host                    = cudf::to_arrow_host(table->view());
    const auto t1                = clock::now();
    const double to_arrow_host_s = seconds(t0, t1);
    table.reset();

    // H2D reference: one pageable cudaMemcpy of the same byte count.
    double memcpy_s = 0;
    {
      std::vector<std::uint8_t> pageable(total_bytes, 1);
      rmm::device_buffer device(total_bytes, cudf::get_default_stream());
      const auto m0 = clock::now();
      REQUIRE(cudaMemcpy(device.data(), pageable.data(), total_bytes, cudaMemcpyHostToDevice) ==
              cudaSuccess);
      const auto m1 = clock::now();
      memcpy_s      = seconds(m0, m1);
    }

    auto fragment = build_result_fragment(*context, columns);
    const auto p0 = clock::now();
    fragment->push_arrow(0,
                         0,
                         reinterpret_cast<std::uintptr_t>(&host->array),
                         reinterpret_cast<std::uintptr_t>(schema.get()));
    const auto p1       = clock::now();
    const double push_s = seconds(p0, p1);
    host.reset();
    fragment->close_input(0, 0);

    const auto r0 = clock::now();
    fragment->run();
    const auto r1 = clock::now();
    arrow_stream_guard out;
    fragment->result_to_arrow(reinterpret_cast<std::uintptr_t>(&out.stream));
    std::int64_t drained = 0;
    while (true) {
      arrow_array_guard array;
      REQUIRE(out.stream.get_next(&out.stream, &array.array) == 0);
      if (array.array.release == nullptr) { break; }
      drained += array.array.length;
    }
    const auto r2 = clock::now();
    REQUIRE(drained == rows);
    const double run_s   = seconds(r0, r1);
    const double drain_s = seconds(r1, r2);

    std::fprintf(stderr,
                 "[sirius_ffi_bench] bytes=%zu (%.0f MiB), rows=%lld x %zu columns\n"
                 "[sirius_ffi_bench] H2D  push_arrow           %.3f s  %.2f GB/s\n"
                 "[sirius_ffi_bench] H2D  cudaMemcpy pageable  %.3f s  %.2f GB/s\n"
                 "[sirius_ffi_bench] D2H  cudf::to_arrow_host  %.3f s  %.2f GB/s\n"
                 "[sirius_ffi_bench] D2H  run() [collector]    %.3f s  %.2f GB/s\n"
                 "[sirius_ffi_bench] D2H  result_to_arrow drain %.3f s  %.2f GB/s\n"
                 "[sirius_ffi_bench] D2H  run()+drain          %.3f s  %.2f GB/s\n",
                 total_bytes,
                 total_bytes / 1048576.0,
                 static_cast<long long>(rows),
                 columns.size(),
                 push_s,
                 gbps(total_bytes, push_s),
                 memcpy_s,
                 gbps(total_bytes, memcpy_s),
                 to_arrow_host_s,
                 gbps(total_bytes, to_arrow_host_s),
                 run_s,
                 gbps(total_bytes, run_s),
                 drain_s,
                 gbps(total_bytes, drain_s),
                 run_s + drain_s,
                 gbps(total_bytes, run_s + drain_s));
  }
}
