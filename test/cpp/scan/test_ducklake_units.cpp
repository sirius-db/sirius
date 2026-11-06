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

#include "ducklake/dv_visibility.hpp"
#include "ducklake/field_mapping.hpp"
#include "ducklake/inline_batch_producer.hpp"
#include "ducklake/inline_varchar.hpp"
#include "ducklake/scan_ingestible.hpp"
#include "ducklake/unit_descriptor.hpp"
#include "helper/hugeint.hpp"
#include "planner/connector_registry.hpp"
#include "scan_manager/preparation_test_support.hpp"
#include "utils/puffin_test_utils.hpp"

#include <catch.hpp>
#include <cucascade/io/io_errors.hpp>
#include <duckdb/common/multi_file/multi_file_function.hpp>
#include <duckdb/common/multi_file/multi_file_reader.hpp>
#include <duckdb/common/multi_file/multi_file_states.hpp>
#include <duckdb/common/types/date.hpp>
#include <duckdb/common/types/timestamp.hpp>
#include <duckdb/common/types/vector.hpp>
#include <duckdb/common/vector/flat_vector.hpp>
#include <roaring/roaring.hh>
#include <zlib.h>

#include <algorithm>
#include <array>
#include <cstring>
#include <fstream>
#include <future>

using namespace sirius::ducklake;
namespace {
bool contains(unit_gate const& gate, unit_gate_reason reason)
{
  return std::find(gate.reasons.begin(), gate.reasons.end(), reason) != gate.reasons.end();
}
unit_descriptor local()
{
  unit_descriptor unit;
  unit.input = duckdb::OpenFileInfo("/tmp/typed-data");
  return unit;
}
}  // namespace

TEST_CASE("DuckLake unit scope preserves inline visibility dependencies", "[ducklake_units]")
{
  auto unit                     = local();
  unit.inline_delete_count      = 19;
  unit.transaction_local_delete = true;
  auto gate                     = assess_local_scope(unit);
  CHECK_FALSE(gate.outside_local_scope);
  CHECK(gate.reasons.empty());
  // These remain obligations for the later visibility/materialization seam.
  CHECK(unit.inline_delete_count == 19);
  CHECK(unit.transaction_local_delete);
  for (auto kind : {unit_kind::inline_table, unit_kind::transaction_local_inline}) {
    unit.kind  = kind;
    unit.input = duckdb::OpenFileInfo("provider_inline_table");
    CHECK_FALSE(assess_local_scope(unit).outside_local_scope);
  }
}

TEST_CASE("DuckLake local scans admit valid partial snapshots", "[ducklake_units]")
{
  auto unit         = local();
  unit.snapshot_min = 0;
  auto gate         = assess_local_scope(unit);
  CHECK_FALSE(gate.outside_local_scope);
  unit.snapshot_min = 9;
  unit.snapshot_max = 8;
  gate              = assess_local_scope(unit);
  CHECK(contains(gate, unit_gate_reason::invalid_snapshot_range));
}

TEST_CASE("DuckLake data and delete transport are gated separately", "[ducklake_units]")
{
  auto unit        = local();
  unit.delete_path = "s3://bucket/deletes";
  unit.deletes     = delete_format::puffin;
  auto gate        = assess_local_scope(unit);
  CHECK(gate.outside_local_scope);
  CHECK_FALSE(contains(gate, unit_gate_reason::remote_data));
  CHECK(contains(gate, unit_gate_reason::remote_delete));
  unit.input       = duckdb::OpenFileInfo("https://example.invalid/data");
  unit.delete_path = "file:///tmp/delete";
  gate             = assess_local_scope(unit);
  CHECK(contains(gate, unit_gate_reason::remote_data));
  CHECK_FALSE(contains(gate, unit_gate_reason::remote_delete));
  unit.input = duckdb::OpenFileInfo("relative-data");
  CHECK(contains(assess_local_scope(unit), unit_gate_reason::invalid_locator));
}

TEST_CASE("DuckLake delete format never implies qualified snapshot semantics", "[ducklake_units]")
{
  auto unit        = local();
  unit.delete_path = "/tmp/looks-like-parquet.parquet";
  unit.deletes     = delete_format::puffin;
  auto gate        = assess_local_scope(unit);
  CHECK_FALSE(gate.outside_local_scope);
  CHECK(contains(gate, unit_gate_reason::dv_metadata_required));
  unit.delete_path = "/tmp/looks-like-puffin.puffin";
  unit.deletes     = delete_format::parquet;
  gate             = assess_local_scope(unit);
  CHECK(contains(gate, unit_gate_reason::parquet_delete_semantics_required));
  CHECK_FALSE(contains(gate, unit_gate_reason::dv_metadata_required));
  unit.deletes = delete_format::unknown;
  CHECK(assess_local_scope(unit).outside_local_scope);
  unit.deletes = delete_format::none;
  CHECK(contains(assess_local_scope(unit), unit_gate_reason::unknown_delete_format));
  unit.delete_path.clear();
  unit.deletes = delete_format::puffin;
  CHECK(contains(assess_local_scope(unit), unit_gate_reason::unknown_delete_format));
}

TEST_CASE("DuckLake encrypted units remain outside the qualified local scope", "[ducklake_units]")
{
  auto unit             = local();
  unit.encrypted        = true;
  unit.delete_encrypted = true;
  auto gate             = assess_local_scope(unit);
  CHECK(gate.outside_local_scope);
  CHECK(contains(gate, unit_gate_reason::encrypted_data));
  CHECK(contains(gate, unit_gate_reason::encrypted_delete));
}

TEST_CASE("DuckLake unit scope fails closed for unknown input kinds", "[ducklake_units]")
{
  auto unit = local();
  unit.kind = static_cast<unit_kind>(255);
  auto gate = assess_local_scope(unit);
  CHECK(gate.outside_local_scope);
  CHECK(contains(gate, unit_gate_reason::unknown_unit_kind));
}

namespace {
namespace pq   = cudf::io::parquet;
namespace scan = sirius::op::scan;
std::shared_ptr<pq::FileMetaData> mapping_footer()
{
  auto footer = std::make_shared<pq::FileMetaData>();
  footer->schema.resize(3);
  footer->schema[0].name         = "schema";
  footer->schema[0].num_children = 2;
  footer->schema[1].name         = "old_name";
  footer->schema[1].field_id     = 4;
  footer->schema[1].type         = pq::Type::INT32;
  footer->schema[2].name         = "dropped";
  footer->schema[2].field_id     = 9;
  footer->schema[2].type         = pq::Type::INT64;
  return footer;
}
scan::field_id_mapping compile_test(std::shared_ptr<pq::FileMetaData> footer,
                                    std::vector<scan::field_requirement> fields,
                                    bool partial_data = false)
{
  auto metadata = std::make_shared<scan::parquet_metadata>(footer,
                                                           0,
                                                           scan::parquet_encryption_evidence{},
                                                           std::vector<uint8_t>{},
                                                           std::string{},
                                                           std::vector<uint8_t>{},
                                                           "object");
  scan::materialization_program program;
  program.contract   = 71;
  program.generation = 3;
  program.input_types.emplace();
  for (auto const& field : fields)
    program.input_types->push_back(*scan::field_mapping_type(field.type));
  program.output_types = *program.input_types;
  return compile_field_mapping(metadata, "object", program, std::move(fields), partial_data);
}
}  // namespace
TEST_CASE("DuckLake stable IDs allow renamed dropped reordered and missing fields",
          "[ducklake_units]")
{
  using duckdb::LogicalType;
  using duckdb::Value;
  auto fields  = std::vector{bind_field(12, LogicalType::BIGINT, Value::BIGINT(7)),
                            bind_field(4, LogicalType::BIGINT, Value())};
  auto mapping = compile_test(mapping_footer(), fields);
  CHECK(mapping.read_names() == std::vector<std::string>{"old_name"});
  CHECK(mapping.read_indices() == std::vector<cudf::size_type>{0});
  CHECK(mapping.additional_bytes(10) >= 160);
  auto all_missing = compile_test(mapping_footer(), {fields[0]});
  CHECK(all_missing.read_names().size() == 1);  // approved physical cardinality carrier
  auto subset = compile_test(mapping_footer(), {bind_field(9, LogicalType::BIGINT, Value())});
  CHECK(subset.read_names() == std::vector<std::string>{"dropped"});
}
TEST_CASE("DuckLake mapping refuses untagged duplicate malformed and narrowing inputs",
          "[ducklake_units]")
{
  using duckdb::LogicalType;
  using duckdb::Value;
  auto required = std::vector{bind_field(4, LogicalType::BIGINT, Value())};
  auto footer   = mapping_footer();
  footer->schema[2].field_id.reset();
  CHECK_THROWS(compile_test(footer, required));
  footer                     = mapping_footer();
  footer->schema[2].field_id = 4;
  CHECK_THROWS(compile_test(footer, required));
  footer                         = mapping_footer();
  footer->schema[2].num_children = 1;
  CHECK_THROWS(compile_test(footer, required));
  footer                 = mapping_footer();
  footer->schema[1].type = pq::Type::DOUBLE;
  CHECK_THROWS(compile_test(footer, required));
  footer                 = mapping_footer();
  footer->schema[2].name = "old_name";
  CHECK_THROWS(compile_test(footer, required));
  CHECK_THROWS(compile_test(mapping_footer(), {required[0], required[0]}));
  CHECK_NOTHROW(compile_test(mapping_footer(), {bind_field(12, LogicalType::INTEGER, Value())}));
  auto unsupported                     = required;
  unsupported[0].allow_float_to_double = true;
  CHECK_THROWS(compile_test(mapping_footer(), unsupported));
}
TEST_CASE("DuckLake mapping validates internal columns even when not selected", "[ducklake_units]")
{
  using duckdb::LogicalType;
  using duckdb::Value;
  auto required   = std::vector{bind_field(4, LogicalType::BIGINT, Value())};
  auto footer     = mapping_footer();
  auto& hidden    = footer->schema[2];
  hidden.field_id = duckdb::MultiFileReader::ROW_ID_FIELD_ID;
  hidden.name     = "_ducklake_internal_row_id";
  CHECK_NOTHROW(compile_test(footer, required));
  hidden.name = "impostor";
  CHECK_THROWS(compile_test(footer, required));
  hidden.name = "_ducklake_internal_row_id";
  hidden.type = pq::Type::INT32;
  CHECK_THROWS(compile_test(footer, required));
}

TEST_CASE("DuckLake required columns preserve order and reject ambiguous selections",
          "[ducklake_units]")
{
  CHECK(required_field_indices(3, {}) == std::vector<size_t>{0, 1, 2});
  CHECK(required_field_indices(3, {2, 0}) == std::vector<size_t>{2, 0});
  CHECK_THROWS(required_field_indices(3, {3}));
  CHECK_THROWS(required_field_indices(3, {1, 1}));
}

namespace {
std::shared_ptr<pq::FileMetaData> nested_mapping_footer()
{
  auto footer = std::make_shared<pq::FileMetaData>();
  footer->schema.resize(4);
  footer->schema[0].name         = "schema";
  footer->schema[0].num_children = 2;
  footer->schema[1].name         = "opaque";
  footer->schema[1].field_id     = 20;
  footer->schema[1].num_children = 1;
  footer->schema[2].name         = "child";
  footer->schema[2].field_id     = 21;
  footer->schema[2].type         = pq::Type::INT64;
  footer->schema[3].name         = "id";
  footer->schema[3].field_id     = 4;
  footer->schema[3].type         = pq::Type::INT32;
  return footer;
}
}  // namespace
TEST_CASE("DuckLake scalar mapping skips nested roots without changing footer identity",
          "[ducklake_units]")
{
  using duckdb::LogicalType;
  using duckdb::Value;
  auto footer   = nested_mapping_footer();
  auto required = bind_field(4, LogicalType::BIGINT, Value());
  auto mapping  = compile_test(footer, {required});
  CHECK(mapping.read_indices() == std::vector<cudf::size_type>{1});
  CHECK(mapping.read_names() == std::vector<std::string>{"id"});
  CHECK(mapping.metadata()->file_metadata().get() == footer.get());
  CHECK(footer->schema.size() == 4);
  // The legacy scope remains flat-only unless an adapter opts in.
  CHECK_THROWS(
    scan::field_id_mapping::compile(mapping.metadata(), "object", mapping.program(), {required}));
  auto missing = compile_test(footer, {bind_field(99, LogicalType::BIGINT, Value::BIGINT(7))});
  CHECK(missing.read_indices() == std::vector<cudf::size_type>{1});
  // Normalized cuDF root links, not preorder array positions, define top-level ordinals.
  footer->schema[0].children_idx = {3, 1};
  footer->schema[1].children_idx = {2};
  auto reordered                 = compile_test(footer, {required});
  CHECK(reordered.read_indices() == std::vector<cudf::size_type>{0});
}
TEST_CASE("DuckLake nested IDs never become scalar inputs or missing defaults", "[ducklake_units]")
{
  using duckdb::LogicalType;
  using duckdb::Value;
  auto footer = nested_mapping_footer();
  CHECK_THROWS(compile_test(footer, {bind_field(20, LogicalType::BIGINT, Value())}));
  CHECK_THROWS(compile_test(footer, {bind_field(21, LogicalType::BIGINT, Value())}));
  footer->schema[2].field_id = 4;
  CHECK_THROWS(compile_test(footer, {bind_field(4, LogicalType::BIGINT, Value())}));
  footer = nested_mapping_footer();
  footer->schema[1].field_id.reset();
  CHECK_THROWS(compile_test(footer, {bind_field(4, LogicalType::BIGINT, Value())}));
  footer                         = nested_mapping_footer();
  footer->schema[0].children_idx = {1, 1};
  CHECK_THROWS(compile_test(footer, {bind_field(4, LogicalType::BIGINT, Value())}));
  footer = nested_mapping_footer();
  footer->schema.resize(3);
  footer->schema[0].num_children = 1;
  CHECK_THROWS(compile_test(footer, {bind_field(99, LogicalType::BIGINT, Value())}));
}

TEST_CASE("DuckLake fixed scalar defaults keep exact types and only qualified widening",
          "[ducklake_units]")
{
  using duckdb::LogicalType;
  using duckdb::Value;
  std::vector<Value> values{Value::BOOLEAN(true),
                            Value::TINYINT(-128),
                            Value::SMALLINT(-32768),
                            Value::INTEGER(-2147483647),
                            Value::BIGINT(INT64_MIN),
                            Value::UTINYINT(255),
                            Value::USMALLINT(65535),
                            Value::UINTEGER(UINT32_MAX),
                            Value::UBIGINT(UINT64_MAX),
                            Value::FLOAT(-0.0f),
                            Value::DOUBLE(-0.0)};
  for (auto const& value : values) {
    INFO(value.type().ToString());
    REQUIRE(fixed_scalar_arrow_format(value.type()));
    auto field = bind_field(4, value.type(), value);
    REQUIRE(field.missing_value);
    CHECK(field.missing_value->type() == value.type());
    CHECK(Value::NotDistinctFrom(*field.missing_value, value));
    CHECK(field.allow_float_to_double == (value.type() == LogicalType::DOUBLE));
    auto null = bind_field(4, value.type(), Value());
    REQUIRE(null.missing_value);
    CHECK(null.missing_value->IsNull());
    CHECK(null.missing_value->type() == value.type());
  }
  auto integer = bind_field(4, LogicalType::BIGINT, Value::INTEGER(-2147483647));
  REQUIRE(integer.missing_value);
  CHECK(integer.missing_value->GetValue<int64_t>() == -2147483647);
  CHECK(integer.allow_int32_to_int64);
  CHECK_FALSE(integer.allow_float_to_double);
  CHECK_THROWS(bind_field(4, LogicalType::BIGINT, Value::DOUBLE(1.5)));
  CHECK_THROWS(bind_field(4, LogicalType::BIGINT, Value("7")));
  CHECK_THROWS(bind_field(0, LogicalType::BIGINT, Value()));
  CHECK_THROWS(bind_field(duckdb::MultiFileReader::ROW_ID_FIELD_ID, LogicalType::BIGINT, Value()));
  auto widened = bind_field(4, LogicalType::DOUBLE, Value::FLOAT(1.25f));
  CHECK(widened.allow_float_to_double);
  CHECK(widened.missing_value->GetValue<double>() == 1.25);
  CHECK_THROWS(bind_field(4, LogicalType::FLOAT, Value::DOUBLE(1.25)));
  CHECK_THROWS(bind_field(4, LogicalType::UINTEGER, Value::INTEGER(-1)));
  CHECK_THROWS(bind_field(4, LogicalType::INTEGER, Value::BIGINT(7)));
  CHECK_FALSE(fixed_scalar_arrow_format(LogicalType::VARCHAR));
  CHECK(fixed_scalar_arrow_format(LogicalType::DECIMAL(10, 2)) == "d:10,2,64");
  CHECK_FALSE(fixed_scalar_arrow_format(LogicalType::TIMESTAMP_NS));
}

TEST_CASE("DuckLake VARCHAR borrows flat constant and dictionary values without copying",
          "[ducklake_units]")
{
  using duckdb::LogicalType;
  using duckdb::Value;
  duckdb::Vector flat(LogicalType::VARCHAR, 4);
  std::string embedded("a\0b", 3);
  flat.Append(Value(embedded));
  flat.Append(Value(""));
  flat.Append(Value(LogicalType::VARCHAR));
  flat.Append(Value("long string beyond the inline string buffer"));
  auto borrowed = borrow_inline_varchar(flat, 0);
  REQUIRE(borrowed);
  CHECK(*borrowed == embedded);
  CHECK(borrowed->data() == duckdb::FlatVector::GetData<duckdb::string_t>(flat)[0].GetData());
  REQUIRE(borrow_inline_varchar(flat, 1));
  CHECK(borrow_inline_varchar(flat, 1)->empty());
  CHECK_FALSE(borrow_inline_varchar(flat, 2));
  duckdb::SelectionVector selection(3);
  selection.set_index(0, 3);
  selection.set_index(1, 2);
  selection.set_index(2, 0);
  duckdb::Vector dictionary(flat, selection, 3);
  CHECK(*borrow_inline_varchar(dictionary, 0) == *borrow_inline_varchar(flat, 3));
  CHECK(borrow_inline_varchar(dictionary, 0)->data() == borrow_inline_varchar(flat, 3)->data());
  CHECK_FALSE(borrow_inline_varchar(dictionary, 1));
  CHECK(*borrow_inline_varchar(dictionary, 2) == embedded);
  duckdb::Vector constant(Value(embedded), duckdb::count_t(5));
  CHECK(*borrow_inline_varchar(constant, 4) == embedded);
  duckdb::Vector null(Value(LogicalType::VARCHAR), duckdb::count_t(5));
  CHECK_FALSE(borrow_inline_varchar(null, 4));
  CHECK_THROWS(borrow_inline_varchar(flat, 4));
  duckdb::Vector integer(Value::INTEGER(1), duckdb::count_t(1));
  CHECK_THROWS(borrow_inline_varchar(integer, 0));
}
TEST_CASE("DuckLake VARCHAR defaults preserve bytes and reject implicit conversion",
          "[ducklake_units]")
{
  using duckdb::LogicalType;
  using duckdb::Value;
  for (auto const& value :
       {Value(""), Value(std::string("a\0b", 3)), Value(LogicalType::VARCHAR)}) {
    auto field = bind_field(99, LogicalType::VARCHAR, value);
    REQUIRE(field.missing_value);
    CHECK(Value::NotDistinctFrom(*field.missing_value, value));
    auto mapping = compile_test(nested_mapping_footer(), {field});
    CHECK(mapping.read_indices() == std::vector<cudf::size_type>{1});
  }
  CHECK_THROWS(bind_field(99, LogicalType::VARCHAR, Value::INTEGER(7)));
  CHECK_THROWS(bind_field(99, LogicalType::VARCHAR, Value::BLOB("binary")));
  CHECK(inline_batch_producer::batch_capacity(2, 1) ==
        inline_batch_producer::batch_capacity(2) + inline_batch_producer::max_string_bytes);
  CHECK(inline_batch_producer::batch_capacity(2, 2) == inline_batch_producer::batch_capacity(2, 1));
  CHECK_THROWS(inline_batch_producer::batch_capacity(1, 2));
  auto envelope = inline_batch_producer::envelope(2, 1);
  REQUIRE(envelope.temporary.size() == 2);
  CHECK(envelope.temporary[0] == inline_batch_producer::batch_capacity(2));
  CHECK(envelope.temporary[1] == inline_batch_producer::max_string_bytes);
}

TEST_CASE("DuckLake temporal defaults preserve units and reject adjacent logical types",
          "[ducklake_units]")
{
  using duckdb::LogicalType;
  using duckdb::Value;
  for (auto const& value :
       {Value::DATE(duckdb::date_t(-1)), Value::TIMESTAMP(duckdb::timestamp_t(-1))}) {
    auto field = bind_field(99, value.type(), value);
    REQUIRE(field.missing_value);
    CHECK(Value::NotDistinctFrom(*field.missing_value, value));
    auto mapping = compile_test(nested_mapping_footer(), {field});
    CHECK(mapping.read_indices() == std::vector<cudf::size_type>{1});
    auto null = bind_field(99, value.type(), Value());
    CHECK(null.missing_value->type() == value.type());
    CHECK(null.missing_value->IsNull());
    CHECK_THROWS(bind_field(99, value.type(), Value::BIGINT(-1)));
    CHECK_THROWS(bind_field(99, value.type(), Value("1970-01-01")));
  }
  CHECK(*fixed_scalar_arrow_format(LogicalType::DATE) == "tdD");
  CHECK(*fixed_scalar_arrow_format(LogicalType::TIMESTAMP) == "tsu:");
  CHECK_THROWS(bind_field(99, LogicalType::TIMESTAMP, Value::DATE(duckdb::date_t(0))));
  for (auto const& type : std::vector<LogicalType>{LogicalType::TIMESTAMP_S,
                                                   LogicalType::TIMESTAMP_MS,
                                                   LogicalType::TIMESTAMP_NS,
                                                   LogicalType::TIMESTAMP_TZ,
                                                   LogicalType::TIME,
                                                   LogicalType::TIME_TZ,
                                                   LogicalType::INTERVAL}) {
    INFO(type.ToString());
    CHECK_FALSE(fixed_scalar_arrow_format(type));
    CHECK_FALSE(scan::field_mapping_type(type));
    CHECK_THROWS(bind_field(99, type, Value(type)));
  }
}
TEST_CASE("DuckLake temporal files retain a carrier for defaults and zero-output scans",
          "[ducklake_units][carrier_policy]")
{
  auto const timestamp = GENERATE(false, true);
  auto const missing   = GENERATE(false, true);
  CAPTURE(timestamp, missing);
  auto footer = mapping_footer();
  footer->schema.resize(2);
  footer->schema[0].num_children = 1;
  footer->schema[1].type         = timestamp ? pq::Type::INT64 : pq::Type::INT32;
  footer->schema[1].converted_type =
    timestamp ? pq::ConvertedType::TIMESTAMP_MICROS : pq::ConvertedType::DATE;
  std::vector<scan::field_requirement> fields;
  if (missing)
    fields.push_back(bind_field(99, duckdb::LogicalType::INTEGER, duckdb::Value::INTEGER(7)));
  auto mapping = compile_test(footer, fields);
  CHECK(mapping.read_indices() == std::vector<cudf::size_type>{0});
  CHECK(mapping.read_types() == std::vector<duckdb::LogicalType>{timestamp
                                                                   ? duckdb::LogicalType::TIMESTAMP
                                                                   : duckdb::LogicalType::DATE});
  REQUIRE(mapping.program().input_types);
  CHECK(mapping.program().input_types->size() == fields.size());
}

TEST_CASE("DuckLake temporal fields require matching physical annotations and units",
          "[ducklake_units]")
{
  using duckdb::LogicalType;
  using duckdb::Value;
  auto footer = mapping_footer();
  auto date   = bind_field(4, LogicalType::DATE, Value());
  CHECK_THROWS(compile_test(footer, {date}));
  footer->schema[1].converted_type = pq::ConvertedType::DATE;
  CHECK(compile_test(footer, {date}).read_indices() == std::vector<cudf::size_type>{0});
  auto timestamp = bind_field(4, LogicalType::TIMESTAMP, Value());
  CHECK_THROWS(compile_test(footer, {timestamp}));
  footer->schema[1].type           = pq::Type::INT64;
  footer->schema[1].converted_type = pq::ConvertedType::TIMESTAMP_MICROS;
  CHECK(compile_test(footer, {timestamp}).read_indices() == std::vector<cudf::size_type>{0});
  footer->schema[1].converted_type = pq::ConvertedType::TIMESTAMP_MILLIS;
  CHECK_THROWS(compile_test(footer, {timestamp}));
  footer->schema[1].converted_type.reset();
  CHECK_THROWS(compile_test(footer, {timestamp}));
  footer->schema[1].type = pq::Type::INT96;
  CHECK_THROWS(compile_test(footer, {timestamp}));
}

TEST_CASE("DuckLake zero-output mapping retains a physical carrier without user defaults",
          "[ducklake_units]")
{
  auto footer  = nested_mapping_footer();
  auto mapping = compile_test(footer, {});
  CHECK(mapping.read_indices() == std::vector<cudf::size_type>{1});
  REQUIRE(mapping.program().input_types);
  CHECK(mapping.program().input_types->empty());
  CHECK(mapping.program().output_types.empty());
  CHECK(mapping.additional_bytes(17) == 17);
  CHECK(mapping.metadata()->file_metadata().get() == footer.get());
  footer->schema.resize(3);
  footer->schema[0].num_children = 1;
  CHECK_THROWS(compile_test(footer, {}));
}

namespace {
namespace sm         = sirius::scan_manager;
using dv_test_source = sirius::test::memory_puffin_source;
std::vector<std::byte> dv_test_payload()
{
  std::ifstream file(
    "test/cpp/integration/data/iceberg_v3_deletion_vector/data/"
    "dv-00000-0-d7e8f9a0-0007-0007-0007-000000000002-00001.puffin",
    std::ios::binary);
  REQUIRE(file.good());
  file.seekg(4);
  std::vector<std::byte> bytes(44);
  file.read(reinterpret_cast<char*>(bytes.data()), bytes.size());
  REQUIRE(file.good());
  return bytes;
}
dv_test_source dv_test_container(std::string properties,
                                 bool corrupt_second                   = false,
                                 std::string first_properties          = {},
                                 std::vector<std::byte> first_payload  = {},
                                 std::vector<std::byte> second_payload = {})
{
  if (first_payload.empty()) first_payload = dv_test_payload();
  if (second_payload.empty()) second_payload = dv_test_payload();
  auto blobs = first_payload;
  if (corrupt_second) second_payload.back() ^= std::byte{1};
  blobs.insert(blobs.end(), second_payload.begin(), second_payload.end());
  std::string json = R"({"blobs":[)";
  for (int i = 0; i < 2; ++i) {
    if (i) json += ",";
    json +=
      R"({"type":"deletion-vector-v1","fields":[],"snapshot-id":-1,"sequence-number":-1,"offset":)" +
      std::to_string(4 + (i ? first_payload.size() : 0)) + R"(,"length":)" +
      std::to_string(i ? second_payload.size() : first_payload.size()) + R"(,"properties":{)" +
      (i ? properties : first_properties) + "}}";
  }
  json += "]}";
  return dv_test_source(sirius::test::puffin_container(blobs, json));
}
unit_descriptor dv_test_unit(dv_test_source const& source)
{
  auto unit             = local();
  unit.deletes          = delete_format::puffin;
  unit.delete_path      = "/tmp/deletes";
  unit.delete_file_size = source.bytes.size();
  return unit;
}
struct dv_test_budget {
  sm::test::test_reservation_provider provider;
  sm::preparation_ledger ledger{provider};
  sm::w_permit permit;
  explicit dv_test_budget(unit_descriptor const& unit)
  {
    provider.grant_bytes = 128 * 1024 * 1024;
    auto envelope        = parquet_unit_envelope(unit);
    std::array scans{sm::scan_envelope{{envelope}, {}, true}};
    REQUIRE(ledger.admit(scans, sm::memory_space_id(cucascade::memory::Tier::HOST, 0)).deferred);
    permit = ledger.try_acquire_unit({1, 1}, envelope.retained);
    REQUIRE(permit);
  }
};
}  // namespace
TEST_CASE("DuckLake DV visibility validates and unions every relevant blob",
          "[ducklake_units][ducklake_dv]")
{
  auto bare = GENERATE(false, true);
  auto source =
    dv_test_container(R"("cardinality":"2","referenced-data-file":"file:///tmp/typed-data")");
  if (bare) source.bytes = dv_test_payload();
  auto unit = dv_test_unit(source);
  dv_test_budget budget(unit);
  auto allocator  = budget.ledger.allocator(budget.permit);
  auto visibility = prepare_dv_visibility(
    unit, 5, "original-data-identity", source, allocator, {}, "original-delete-identity");
  CHECK(visibility->validated_blobs == (bare ? 1 : 2));
  CHECK(visibility->positions.size() == 2);
  CHECK(visibility->data_identity == "original-data-identity");
  CHECK(visibility->delete_identity == "original-delete-identity");
  CHECK(std::is_sorted(visibility->positions.begin(), visibility->positions.end()));
  REQUIRE(budget.provider.seen->allocated_bytes > 0);
  budget.ledger.retire_unit({1, 1});
  budget.permit.release();
  CHECK(budget.provider.seen->allocated_bytes > 0);
  visibility.reset();
  CHECK(budget.provider.seen->allocated_bytes == 0);
}
TEST_CASE("DuckLake DV invalid later blobs cannot publish partial visibility",
          "[ducklake_units][ducklake_dv]")
{
  auto mode = GENERATE(std::string("cardinality"),
                       std::string("negative"),
                       std::string("empty"),
                       std::string("overflow"),
                       std::string("snapshot"),
                       std::string("association"),
                       std::string("crc"),
                       std::string("domain"),
                       std::string("size"),
                       std::string("footer"),
                       std::string("io"),
                       std::string("cancel"),
                       std::string("allocation_failure"));
  INFO(mode);
  std::string properties;
  if (mode == "cardinality") properties = R"("cardinality":"3")";
  if (mode == "negative") properties = R"("cardinality":"-2")";
  if (mode == "empty") properties = R"("cardinality":"")";
  if (mode == "overflow") properties = R"("cardinality":"18446744073709551616")";
  if (mode == "snapshot") properties = R"("ducklake-snapshot-id":"")";
  if (mode == "association") properties = R"("referenced-data-file":"/tmp/another-data")";
  auto source = dv_test_container(properties, mode == "crc");
  auto unit   = dv_test_unit(source);
  if (mode == "size") ++unit.delete_file_size;
  if (mode == "footer") unit.delete_footer_size = 1;
  dv_test_budget budget(unit);
  if (mode == "allocation_failure") budget.provider.seen->fail_allocation = true;
  auto allocator = budget.ledger.allocator(budget.permit);
  std::stop_source stop;
  if (mode == "cancel") stop.request_stop();
  if (mode == "io") source.fail_read = true;
  auto prepare = [&] {
    return prepare_dv_visibility(unit,
                                 mode == "domain" ? 1 : 5,
                                 "original-data-identity",
                                 source,
                                 allocator,
                                 sm::preparation_stop_token(stop.get_token()));
  };
  if (mode == "size")
    CHECK_THROWS_AS(prepare(), cucascade::io::descriptor_mismatch_error);
  else if (mode == "io")
    CHECK_THROWS_AS(prepare(), sirius::transparent::classified_execution_error);
  else if (mode == "cancel")
    CHECK_THROWS_AS(prepare(), std::system_error);
  else if (mode == "allocation_failure")
    CHECK_THROWS_AS(prepare(), sm::preparation_resource_error);
  else
    CHECK_THROWS(prepare());
  CHECK(budget.provider.seen->allocated_bytes == 0);
  CHECK(budget.ledger.charged_bytes({1, 1}) == 0);
}
TEST_CASE("DuckLake DV bounded profile keeps adjacent dependencies unqualified",
          "[ducklake_units][ducklake_dv]")
{
  auto source = dv_test_container({});
  auto unit   = dv_test_unit(source);
  CHECK(parquet_unit_dependencies_supported(unit));
  auto mode = GENERATE(3, 5, 6);
  if (mode == 3) unit.mapping_id = 1;
  if (mode == 5) unit.delete_path = "s3://bucket/delete";
  if (mode == 6) {
    unit.snapshot_min = 1;
    unit.snapshot_max = 0;
  }
  CHECK_FALSE(parquet_unit_dependencies_supported(unit));
  CHECK_THROWS(parquet_unit_envelope(unit));
  unit                  = dv_test_unit(source);
  unit.delete_file_size = dv_max_file_bytes + 1;
  CHECK_THROWS_AS(parquet_unit_envelope(unit), sm::preparation_resource_error);
}

TEST_CASE("DuckLake visibility unions inline positions without losing file evidence",
          "[ducklake_units][ducklake_dv]")
{
  auto source                   = dv_test_container({});
  auto unit                     = dv_test_unit(source);
  unit.inline_delete_count      = 3;
  unit.transaction_local_delete = true;
  CHECK(parquet_unit_dependencies_supported(unit));
  dv_test_budget budget(unit);
  auto allocator = budget.ledger.allocator(budget.permit);
  auto base      = prepare_dv_visibility(unit, 5, "data", source, allocator, {}, "deletes");
  std::array<uint64_t, 3> positions{0, 1, 4};
  auto result = combine_inline_visibility(base, positions, 5, allocator);
  CHECK(result->inline_count == 3);
  CHECK(result->data_identity == "data");
  CHECK(result->delete_identity == "deletes");
  CHECK(result->validated_blobs == 2);
  std::vector<int64_t> expected(base->positions.begin(), base->positions.end());
  expected.insert(expected.end(), positions.begin(), positions.end());
  std::sort(expected.begin(), expected.end());
  expected.erase(std::unique(expected.begin(), expected.end()), expected.end());
  CHECK(std::vector<int64_t>(result->positions.begin(), result->positions.end()) == expected);
  base.reset();
  CHECK_FALSE(result->positions.empty());
  for (auto bad : std::vector<std::vector<uint64_t>>{{1, 1}, {4, 1}, {5}, {UINT64_MAX}})
    CHECK_THROWS(combine_inline_visibility(result, bad, 5, allocator));
  result.reset();
  CHECK(budget.provider.seen->allocated_bytes == 0);
}

TEST_CASE("Ordinary Parquet delete positions stay strict across chunks",
          "[ducklake_units][parquet_deletes]")
{
  int64_t previous = -1;
  for (auto position : {0, 3, 8})
    validate_delete_position(position, previous);
  CHECK(previous == 8);
  for (auto invalid : {int64_t{-1}, int64_t{8}, int64_t{2}}) {
    CHECK_THROWS(validate_delete_position(invalid, previous));
    CHECK(previous == 8);
  }
  validate_delete_position(INT64_MAX, previous);
  CHECK(previous == INT64_MAX);
  CHECK_THROWS(validate_delete_position(INT64_MAX, previous));
  auto source             = std::make_shared<dv_visibility>();
  source->delete_identity = "opened";
  CHECK_THROWS(bind_parquet_visibility(source, "data", 10));
  source->parquet_validated = true;
  std::array<int64_t, 2> positions{0, 9};
  source->positions = positions;
  auto bound        = bind_parquet_visibility(source, "data", 10);
  CHECK(bound->predecessor == source);
  CHECK(bound->data_identity == "data");
  CHECK(bound->parquet_validated);
  CHECK_THROWS(bind_parquet_visibility(source, "data", 9));
  CHECK_THROWS(bind_parquet_visibility(source, "", 10));
}

TEST_CASE("DuckLake inline batches use spare column capacity without expanding their byte bound",
          "[ducklake_units][ducklake_batch_capacity]")
{
  using sirius::ducklake::inline_batch_producer;
  CHECK(inline_batch_producer::batch_rows(0) == 2048);
  CHECK(inline_batch_producer::batch_rows(2) == 2048);
  CHECK(inline_batch_producer::batch_rows(5) == 1600);
  CHECK(inline_batch_producer::batch_rows(32) == 256);
  CHECK_THROWS(inline_batch_producer::batch_rows(33));
  auto ceiling = inline_batch_producer::batch_capacity(32);
  for (size_t columns = 1; columns <= 32; ++columns) {
    auto rows = inline_batch_producer::batch_rows(columns);
    CHECK(rows >= 256);
    CHECK(rows <= 2048);
    CHECK(rows % 64 == 0);
    CHECK(rows * columns <= 8192);
    CHECK(inline_batch_producer::batch_capacity(columns) <= ceiling);
  }
}

TEST_CASE("DuckLake Parquet dependency errors distinguish mappings and snapshot visibility",
          "[ducklake_units]")
{
  auto unit = local();
  CHECK_NOTHROW(validate_parquet_unit_dependencies(unit));
  unit.mapping_id = 0;
  CHECK_FALSE(parquet_unit_dependencies_supported(unit));
  CHECK_THROWS_WITH(parquet_unit_envelope(unit),
                    "DuckLake external field mappings are outside the supported scan profile");
  unit.mapping_id.reset();
  unit.snapshot_min = 0;
  CHECK(parquet_unit_dependencies_supported(unit));
  CHECK_NOTHROW(parquet_unit_envelope(unit));
  unit.snapshot_min = 9;
  unit.snapshot_max = 8;
  CHECK_THROWS_WITH(parquet_unit_envelope(unit), "DuckLake data snapshot range is invalid");
  unit.snapshot_min.reset();
  unit.snapshot_max.reset();
  unit.delete_path = "/tmp/deletes";
  for (auto format : {delete_format::parquet, delete_format::puffin}) {
    unit.deletes = format;
    CHECK(parquet_unit_dependencies_supported(unit));
    CHECK_NOTHROW(validate_parquet_unit_dependencies(unit));
  }
  unit.delete_path = "s3://bucket/deletes";
  CHECK_THROWS_WITH(validate_parquet_unit_dependencies(unit),
                    "DuckLake remote deletes are outside the local scan profile");
}

TEST_CASE("DuckLake narrow scans admit without reserving unused string capacity",
          "[ducklake_units]")
{
  auto strings = GENERATE(false, true);
  sm::test::test_reservation_provider provider;
  auto envelope        = scan_ingestible::envelope(1, strings ? 1 : 0);
  provider.grant_bytes = envelope.retained + dv_max_file_bytes;
  {
    sm::preparation_ledger ledger(provider);
    std::array scans{sm::scan_envelope{{envelope}, {}, true, 1}};
    auto admitted = ledger.admit(scans, sm::memory_space_id(cucascade::memory::Tier::HOST, 0));
    CHECK(admitted.deferred == !strings);
    if (admitted.deferred) {
      auto permit = ledger.try_acquire_unit({1, 1}, envelope.retained);
      REQUIRE(permit);
    }
  }
  CHECK(provider.outstanding() == 0);
}

TEST_CASE("DuckLake decimal mapping requires exact precision and scale",
          "[ducklake_units][decimal]")
{
  using duckdb::LogicalType;
  using duckdb::Value;
  auto precision = GENERATE(5, 9, 10, 18, 19, 38);
  auto type      = LogicalType::DECIMAL(precision, 2);
  auto value     = Value::DECIMAL(int64_t{-123}, precision, 2);
  auto field     = bind_field(4, type, value);
  REQUIRE(field.missing_value);
  CHECK(Value::NotDistinctFrom(*field.missing_value, value));
  CHECK_THROWS(bind_field(4, type, Value::DECIMAL(int64_t{123}, precision, 1)));
  auto footer              = mapping_footer();
  auto& column             = footer->schema[1];
  column.type              = precision <= 9    ? pq::Type::INT32
                             : precision <= 18 ? pq::Type::INT64
                                               : pq::Type::FIXED_LEN_BYTE_ARRAY;
  column.type_length       = precision > 18 ? 16 : 0;
  column.converted_type    = pq::ConvertedType::DECIMAL;
  column.decimal_precision = precision;
  column.decimal_scale     = 2;
  CHECK_NOTHROW(compile_test(footer, {field}));
  column.decimal_scale = 1;
  CHECK_THROWS(compile_test(footer, {field}));
  column.decimal_scale     = 2;
  column.decimal_precision = precision == 38 ? 37 : precision + 1;
  CHECK_THROWS(compile_test(footer, {field}));
  CHECK_NOTHROW(compile_test(mapping_footer(), {bind_field(99, type, value)}));
  CHECK_NOTHROW(compile_test(mapping_footer(), {bind_field(99, type, Value(type))}));
}

TEST_CASE("DuckLake inline budgets include wide decimal storage", "[ducklake_units][decimal]")
{
  for (size_t columns : {size_t{1}, size_t{3}, size_t{32}}) {
    auto rows = inline_batch_producer::batch_rows(columns);
    auto base = inline_batch_producer::batch_capacity(columns);
    auto wide = inline_batch_producer::batch_capacity(columns, 0, columns);
    CHECK(wide == base + columns * rows * 8);
    auto envelope = inline_batch_producer::envelope(columns, 0, columns);
    CHECK(envelope.temporary.front() == wide);
  }
  CHECK_THROWS(inline_batch_producer::batch_capacity(2, 1, 2));
}

TEST_CASE("DuckLake keeps the engine small decimal admission boundary", "[ducklake_units][decimal]")
{
  for (int precision = 1; precision <= 4; ++precision) {
    auto type = duckdb::LogicalType::DECIMAL(precision, precision);
    CHECK_FALSE(fixed_scalar_arrow_format(type));
    CHECK_THROWS(bind_field(4, type, duckdb::Value(type)));
  }
}

TEST_CASE("DuckLake scalar layouts preserve decimal Boolean and temporal vector encodings",
          "[ducklake_units]")
{
  using duckdb::LogicalType;
  using duckdb::Value;
  auto type     = GENERATE(LogicalType(LogicalType::BOOLEAN),
                       LogicalType(LogicalType::DATE),
                       LogicalType(LogicalType::TIMESTAMP),
                       LogicalType::DECIMAL(5, 2),
                       LogicalType::DECIMAL(9, 2),
                       LogicalType::DECIMAL(10, 2),
                       LogicalType::DECIMAL(18, 2),
                       LogicalType::DECIMAL(19, 2),
                       LogicalType::DECIMAL(38, 2));
  auto encoding = GENERATE(0, 1, 2, 3, 4);
  CAPTURE(type.ToString(), encoding);
  auto decimal   = type.id() == duckdb::LogicalTypeId::DECIMAL;
  auto precision = decimal ? duckdb::DecimalType::GetWidth(type) : 0;
  inline_column_layout layout(type);
  if (decimal) {
    CHECK(layout.decimal_precision == precision);
    CHECK(layout.storage_bytes == (precision > 18 ? 16 : 8));
  }
  duckdb::Vector flat(type, 13);
  for (int row = 0; row < 13; ++row) {
    auto value = decimal ? Value::DECIMAL(int64_t{(row - 6) * 123}, precision, 2)
                 : type == LogicalType::BOOLEAN ? Value::BOOLEAN(row % 2 == 0)
                 : type == LogicalType::DATE
                   ? Value::DATE(duckdb::date_t{row - 6})
                   : Value::TIMESTAMP(duckdb::timestamp_t{(row - 6) * 1000001LL});
    flat.Append(row % 4 == 1 ? Value(type) : value);
  }
  duckdb::SelectionVector selected(13), outer(13);
  for (size_t row = 0; row < 13; ++row) {
    selected.set_index(row, row * 5 % 13);
    outer.set_index(row, (row + 3) % 13);
  }
  duckdb::Vector dictionary(flat, selected, 13);
  duckdb::Vector nested(dictionary, outer, 13);
  duckdb::Vector constant(flat.GetValue(2), duckdb::count_t(13));
  duckdb::Vector null(Value(type), duckdb::count_t(13));
  auto const& source = encoding == 0   ? flat
                       : encoding == 1 ? dictionary
                       : encoding == 2 ? nested
                       : encoding == 3 ? constant
                                       : null;
  std::array<std::byte, 192> values{};
  std::array<uint8_t, 2> validity{};
  auto nulls            = layout.copy_fixed_values(source, 1, 12, values.data(), validity.data());
  size_t expected_nulls = 0;
  for (size_t row = 0; row < 12; ++row) {
    auto expected = source.GetValue(row + 1);
    CHECK(bool(validity[row / 8] & (1u << (row % 8))) == !expected.IsNull());
    if (expected.IsNull()) {
      ++expected_nulls;
      continue;
    }
    if (type == LogicalType::BOOLEAN) {
      auto actual = (std::to_integer<uint8_t>(values[row / 8]) >> (row % 8)) & 1;
      CHECK(bool(actual) == expected.GetValue<bool>());
    } else {
      __int128_t actual = 0;
      if (layout.value_bytes == 4) {
        int32_t value;
        std::memcpy(&value, values.data() + row * 4, 4);
        actual = value;
      } else if (layout.value_bytes == 8) {
        int64_t value;
        std::memcpy(&value, values.data() + row * 8, 8);
        actual = value;
      } else
        std::memcpy(&actual, values.data() + row * 16, 16);
      auto expected_value = decimal ? sirius::to_int128(duckdb::IntegralValue::Get(expected))
                            : type == LogicalType::DATE
                              ? duckdb::DateValue::Get(expected).days
                              : duckdb::TimestampValue::Get(expected).value;
      CHECK(actual == expected_value);
    }
  }
  CHECK(nulls == expected_nulls);
  CHECK((validity[1] & 0xf0) == 0);
  CHECK(layout.copy_fixed_values(source, 13, 0, values.data(), validity.data()) == 0);
  CHECK_THROWS(layout.copy_fixed_values(source, 12, 2, values.data(), validity.data()));
  if (decimal) {
    inline_column_layout other(LogicalType::DECIMAL(precision, 1));
    CHECK_THROWS(other.copy_fixed_values(source, 0, 1, values.data(), validity.data()));
  }
}

TEST_CASE("DuckLake snapshot DV rejects invalid future dependencies",
          "[ducklake_units][ducklake_dv][ducklake_snapshot_visibility]")
{
  auto mode  = GENERATE(std::string("crc"),
                       std::string("mixed"),
                       std::string("negative"),
                       std::string("overflow"),
                       std::string("cardinality"),
                       std::string("cancel"));
  auto early = std::string(R"("ducklake-snapshot-id":"10")");
  auto late  = std::string(R"("ducklake-snapshot-id":"20")");
  if (mode == "mixed") late.clear();
  if (mode == "negative") late = R"("ducklake-snapshot-id":"-1")";
  if (mode == "overflow") late = R"("ducklake-snapshot-id":"18446744073709551616")";
  if (mode == "cardinality") late += R"(,"cardinality":"3")";
  auto source         = dv_test_container(late, mode == "crc", early);
  auto unit           = dv_test_unit(source);
  unit.bound_snapshot = 15;
  dv_test_budget budget(unit);
  auto allocator = budget.ledger.allocator(budget.permit);
  std::stop_source stop;
  if (mode == "cancel") stop.request_stop();
  CHECK_THROWS(prepare_dv_visibility(
    unit, 5, "data", source, allocator, sm::preparation_stop_token(stop.get_token()), "delete"));
  CHECK(budget.provider.seen->allocated_bytes == 0);
  CHECK(budget.ledger.charged_bytes({1, 1}) == 0);
}

namespace {
std::vector<std::byte> snapshot_test_payload(std::initializer_list<uint32_t> positions)
{
  roaring::Roaring bitmap;
  for (auto position : positions)
    bitmap.add(position);
  std::vector<std::byte> body(16 + bitmap.getSizeInBytes(), std::byte{0});
  body[0] = std::byte{0xD1};
  body[1] = std::byte{0xD3};
  body[2] = std::byte{0x39};
  body[3] = std::byte{0x64};
  // Roaring64: one low-word bitmap, high-word key zero.
  body[4] = std::byte{1};
  bitmap.write(reinterpret_cast<char*>(body.data() + 16));
  std::vector<std::byte> payload;
  auto append_be = [&](uint32_t value) {
    for (int shift = 24; shift >= 0; shift -= 8)
      payload.push_back(std::byte((value >> shift) & 255));
  };
  append_be(body.size());
  payload.insert(payload.end(), body.begin(), body.end());
  append_be(crc32(0, reinterpret_cast<Bytef const*>(body.data()), body.size()));
  return payload;
}
}  // namespace
TEST_CASE("DuckLake snapshot DV selects earliest deletes and preserves visible inline deletes",
          "[ducklake_units][ducklake_dv][ducklake_snapshot_visibility]")
{
  auto snapshot     = GENERATE(uint64_t{9}, uint64_t{10}, uint64_t{15}, uint64_t{20});
  auto local_delete = GENERATE(false, true);
  auto reversed     = GENERATE(false, true);
  CAPTURE(snapshot, local_delete, reversed);
  auto early         = std::string(R"("ducklake-snapshot-id":"10","cardinality":"2")");
  auto late          = std::string(R"("ducklake-snapshot-id":"20","cardinality":"3")");
  auto early_payload = snapshot_test_payload({1, 2});
  auto late_payload  = snapshot_test_payload({1, 2, 5});
  auto source        = dv_test_container(reversed ? early : late,
                                  false,
                                  reversed ? late : early,
                                  reversed ? late_payload : early_payload,
                                  reversed ? early_payload : late_payload);
  auto unit                     = dv_test_unit(source);
  unit.bound_snapshot           = snapshot;
  unit.transaction_local_delete = local_delete;
  dv_test_budget budget(unit);
  auto allocator  = budget.ledger.allocator(budget.permit);
  auto visibility = prepare_dv_visibility(unit, 6, "data", source, allocator, {}, "delete");
  CHECK(visibility->validated_blobs == 2);
  std::vector<int64_t> expected = !local_delete && snapshot < 10   ? std::vector<int64_t>{}
                                  : !local_delete && snapshot < 20 ? std::vector<int64_t>{1, 2}
                                                                   : std::vector<int64_t>{1, 2, 5};
  CHECK(std::vector<int64_t>(visibility->positions.begin(), visibility->positions.end()) ==
        expected);
  // Inline deletes remain effective even before the first file-delete snapshot.
  std::array<uint64_t, 2> inlined{2, 4};
  auto combined = combine_inline_visibility(visibility, inlined, 6, allocator);
  expected      = !local_delete && snapshot < 10   ? std::vector<int64_t>{2, 4}
                  : !local_delete && snapshot < 20 ? std::vector<int64_t>{1, 2, 4}
                                                   : std::vector<int64_t>{1, 2, 4, 5};
  CHECK(std::vector<int64_t>(combined->positions.begin(), combined->positions.end()) == expected);
  CHECK(combined->validated_blobs == 2);
  combined.reset();
  visibility.reset();
  CHECK(budget.provider.seen->allocated_bytes == 0);
}

TEST_CASE("DuckLake partial Parquet deletes validate all rows before snapshot selection",
          "[ducklake_units][ducklake_snapshot_visibility]")
{
  auto snapshot = GENERATE(uint64_t{9}, uint64_t{10}, uint64_t{15}, uint64_t{20});
  auto local    = GENERATE(false, true);
  std::array<int64_t, 3> output{};
  parquet_delete_accumulator rows(output, true, snapshot, local);
  rows.append(1, 10);
  rows.append(2, 20);
  rows.append(5, 10);  // Snapshot order need not follow position order.
  CHECK(rows.validated_count() == 3);
  CHECK(rows.last_position() == 5);
  std::vector<int64_t> expected = local || snapshot >= 20 ? std::vector<int64_t>{1, 2, 5}
                                  : snapshot >= 10        ? std::vector<int64_t>{1, 5}
                                                          : std::vector<int64_t>{};
  CHECK(std::vector<int64_t>(rows.positions().begin(), rows.positions().end()) == expected);
}
TEST_CASE("DuckLake partial Parquet deletes reject invalid invisible rows across batches",
          "[ducklake_units][ducklake_snapshot_visibility]")
{
  auto mode  = GENERATE(0, 1, 2, 3, 4, 5);
  auto local = GENERATE(false, true);
  std::array<int64_t, 2> output{};
  parquet_delete_accumulator rows(output, true, 15, local);
  rows.append(2, 20);
  if (mode == 0) CHECK_THROWS(rows.append(2, 30));
  if (mode == 1) CHECK_THROWS(rows.append(1, 30));
  if (mode == 2) CHECK_THROWS(rows.append({}, 30));
  if (mode == 3) CHECK_THROWS(rows.append(3, {}));
  if (mode == 4) CHECK_THROWS(rows.append(3, -1));
  if (mode == 5) {
    rows.append(3, 30);
    CHECK_THROWS(rows.append(4, 30));
  }
}
TEST_CASE("DuckLake partial Parquet deletes retain future row-domain evidence",
          "[ducklake_units][ducklake_snapshot_visibility]")
{
  auto source                     = std::make_shared<dv_visibility>();
  source->parquet_validated       = true;
  source->delete_identity         = "delete-object";
  source->last_validated_position = 5;
  // No effective deletes at S15, but a future row 5 still needs validation.
  CHECK_THROWS(bind_parquet_visibility(source, "data-object", 5));
  auto visibility = bind_parquet_visibility(source, "data-object", 6);
  CHECK(visibility->positions.empty());
  CHECK(visibility->last_validated_position == 5);
}
TEST_CASE("DuckLake ordinary Parquet deletes do not require snapshot values",
          "[ducklake_units][ducklake_snapshot_visibility]")
{
  std::array<int64_t, 2> output{};
  parquet_delete_accumulator rows(output, false, 0, false);
  rows.append(0, {});
  rows.append(4, {});
  CHECK(rows.validated_count() == 2);
  REQUIRE(rows.positions().size() == 2);
  CHECK(rows.positions()[0] == 0);
  CHECK(rows.positions()[1] == 4);
}

TEST_CASE("DuckLake partial data adds an inclusive hidden snapshot filter",
          "[ducklake_units][ducklake_snapshot_visibility]")
{
  auto count_only = GENERATE(false, true);
  auto bounds     = GENERATE(0, 1, 2);
  auto unit       = local();
  if (bounds != 1) unit.snapshot_min = 10;
  if (bounds != 0) unit.snapshot_max = 20;
  std::vector<scan::field_requirement> fields;
  if (!count_only) fields.push_back(bind_field(4, duckdb::LogicalType::BIGINT, duckdb::Value()));
  auto original_width = fields.size();
  unit_scan_options options;
  REQUIRE(add_partial_data_filter(unit, fields, options));
  REQUIRE(options.output_columns);
  CHECK(options.output_columns->size() == original_width);
  REQUIRE(fields.size() == original_width + 1);
  CHECK(fields.back().id == duckdb::MultiFileReader::LAST_UPDATED_SEQUENCE_NUMBER_ID);
  CHECK_FALSE(fields.back().missing_value);
  duckdb::TableFilterSet expected;
  auto add = [&](uint64_t value, duckdb::ExpressionType comparison) {
    expected.PushFilter(
      duckdb::ProjectionIndex(original_width),
      duckdb::ExpressionFilter::CreateComparisonFilter(comparison, duckdb::Value::BIGINT(value)));
  };
  if (unit.snapshot_min) add(10, duckdb::ExpressionType::COMPARE_GREATERTHANOREQUALTO);
  if (unit.snapshot_max) add(20, duckdb::ExpressionType::COMPARE_LESSTHANOREQUALTO);
  CHECK(options.filters->Equals(expected));
  auto footer                = mapping_footer();
  footer->schema[2].field_id = fields.back().id;
  footer->schema[2].name     = "_ducklake_internal_snapshot_id";
  CHECK_NOTHROW(compile_test(footer, fields, true));
  CHECK_THROWS(
    compile_test(footer, fields));  // The default mapping cannot request hidden snapshot columns.
  footer->schema[2].field_id = 9;
  footer->schema[2].name     = "dropped";
  CHECK_THROWS(compile_test(footer, fields, true));  // No default for missing snapshots.
}
TEST_CASE("DuckLake partial data preserves user projection and rejects unqualified bounds",
          "[ducklake_units][ducklake_snapshot_visibility]")
{
  auto unit         = local();
  unit.snapshot_min = 10;
  unit.snapshot_max = 20;
  auto fields       = std::vector{bind_field(4, duckdb::LogicalType::BIGINT, duckdb::Value()),
                            bind_field(9, duckdb::LogicalType::BIGINT, duckdb::Value())};
  unit_scan_options options;
  options.output_columns = std::vector<size_t>{1, 0, 1};
  CHECK_FALSE(assess_local_scope(unit).outside_local_scope);
  REQUIRE(add_partial_data_filter(unit, fields, options));
  CHECK(*options.output_columns == std::vector<size_t>{1, 0, 1});
  CHECK(options.filters->HasFilter(duckdb::ProjectionIndex(2)));
  unit.snapshot_min = 21;
  CHECK(assess_local_scope(unit).outside_local_scope);
  CHECK_THROWS(add_partial_data_filter(unit, fields, options));
  unit.snapshot_min = 10;
  unit.snapshot_max = uint64_t(INT64_MAX) + 1;
  CHECK_THROWS(add_partial_data_filter(unit, fields, options));
}
TEST_CASE("DuckLake partial data requires exact snapshot identity and type",
          "[ducklake_units][ducklake_snapshot_visibility]")
{
  auto mode       = GENERATE(0, 1, 2, 3, 4);
  auto footer     = mapping_footer();
  auto& column    = footer->schema[2];
  column.name     = "_ducklake_internal_snapshot_id";
  column.field_id = duckdb::MultiFileReader::LAST_UPDATED_SEQUENCE_NUMBER_ID;
  scan::field_requirement snapshot{*column.field_id, duckdb::LogicalType::BIGINT, std::nullopt};
  if (mode == 0) column.type = pq::Type::INT32;
  if (mode == 1) column.name = "wrong_name";
  if (mode == 2) snapshot.missing_value = duckdb::Value::BIGINT(0);
  if (mode == 3) snapshot.allow_int32_to_int64 = true;
  if (mode == 4) column.num_children = 1;
  CHECK_THROWS(compile_test(footer, {snapshot}, true));
  CHECK_THROWS(compile_test(mapping_footer(), {}, true));
}
