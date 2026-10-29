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
#include "io/io_errors.hpp"
#include "planner/connector_registry.hpp"
#include "scan_manager/preparation_test_support.hpp"
#include "utils/puffin_test_utils.hpp"

#include <catch.hpp>
#include <duckdb/common/multi_file/multi_file_function.hpp>
#include <duckdb/common/multi_file/multi_file_reader.hpp>
#include <duckdb/common/multi_file/multi_file_states.hpp>
#include <duckdb/common/types/date.hpp>
#include <duckdb/common/types/timestamp.hpp>
#include <duckdb/common/types/vector.hpp>
#include <duckdb/common/vector/flat_vector.hpp>

#include <algorithm>
#include <array>
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

TEST_CASE("DuckLake local scans refuse partial snapshots", "[ducklake_units]")
{
  auto unit         = local();
  unit.snapshot_min = 0;
  auto gate         = assess_local_scope(unit);
  CHECK(gate.outside_local_scope);
  CHECK(contains(gate, unit_gate_reason::partial_data));
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
                                    std::vector<scan::field_requirement> fields)
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
  return compile_field_mapping(metadata, "object", program, std::move(fields));
}
}  // namespace
TEST_CASE("DuckLake evolution binds historical defaults without lossy conversion",
          "[ducklake_units]")
{
  using duckdb::LogicalType;
  using duckdb::Value;
  auto null = bind_field(4, LogicalType::BIGINT, Value());
  REQUIRE(null.missing_value);
  CHECK(null.missing_value->IsNull());
  CHECK(null.missing_value->type() == LogicalType::BIGINT);
  auto integer = bind_field(4, LogicalType::BIGINT, Value::INTEGER(-2147483647));
  REQUIRE(integer.missing_value);
  CHECK(integer.missing_value->GetValue<int64_t>() == -2147483647);
  CHECK(integer.allow_int32_to_int64);
  CHECK_FALSE(integer.allow_float_to_double);
  CHECK_THROWS(bind_field(4, LogicalType::BIGINT, Value::DOUBLE(1.5)));
  CHECK_THROWS(bind_field(4, LogicalType::BIGINT, Value("7")));
  CHECK_THROWS(bind_field(0, LogicalType::BIGINT, Value()));
  CHECK_THROWS(bind_field(duckdb::MultiFileReader::ROW_ID_FIELD_ID, LogicalType::BIGINT, Value()));
  auto other = bind_field(4, LogicalType::DOUBLE, Value::DOUBLE(1.5));
  CHECK(other.missing_value);
  CHECK(other.allow_float_to_double);
}
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
    auto null = bind_field(4, value.type(), Value());
    REQUIRE(null.missing_value);
    CHECK(null.missing_value->IsNull());
    CHECK(null.missing_value->type() == value.type());
  }
  auto widened = bind_field(4, LogicalType::DOUBLE, Value::FLOAT(1.25f));
  CHECK(widened.allow_float_to_double);
  CHECK(widened.missing_value->GetValue<double>() == 1.25);
  CHECK_THROWS(bind_field(4, LogicalType::FLOAT, Value::DOUBLE(1.25)));
  CHECK_THROWS(bind_field(4, LogicalType::UINTEGER, Value::INTEGER(-1)));
  CHECK_THROWS(bind_field(4, LogicalType::INTEGER, Value::BIGINT(7)));
  CHECK_FALSE(fixed_scalar_arrow_format(LogicalType::VARCHAR));
  CHECK_FALSE(fixed_scalar_arrow_format(LogicalType::DECIMAL(10, 2)));
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
  CHECK(std::string(fixed_scalar_arrow_format(LogicalType::DATE)) == "tdD");
  CHECK(std::string(fixed_scalar_arrow_format(LogicalType::TIMESTAMP)) == "tsu:");
  CHECK_THROWS(bind_field(99, LogicalType::TIMESTAMP, Value::DATE(duckdb::date_t(0))));
  for (auto const& type : std::vector<LogicalType>{LogicalType::TIMESTAMP_S,
                                                   LogicalType::TIMESTAMP_MS,
                                                   LogicalType::TIMESTAMP_NS,
                                                   LogicalType::TIMESTAMP_TZ,
                                                   LogicalType::TIME,
                                                   LogicalType::TIME_TZ,
                                                   LogicalType::INTERVAL,
                                                   LogicalType::DECIMAL(4, 2),
                                                   LogicalType::DECIMAL(18, 6),
                                                   LogicalType::DECIMAL(38, 10)}) {
    INFO(type.ToString());
    CHECK_FALSE(fixed_scalar_arrow_format(type));
    CHECK_FALSE(scan::field_mapping_type(type));
    CHECK_THROWS(bind_field(99, type, Value(type)));
  }
}
TEST_CASE("Generic temporal mapping does not widen Iceberg admission",
          "[ducklake_units][contracts]")
{
  using duckdb::LogicalType;
  duckdb::MultiFileBindData bind;
  bind.columns.emplace_back("x", LogicalType::INTEGER);
  bind.columns.back().identifier = duckdb::Value::INTEGER(1);
  CHECK(sirius::planner::iceberg_flat_mapping_supported(&bind));
  for (auto const& type : std::vector<LogicalType>{
         LogicalType::DATE, LogicalType::TIMESTAMP, LogicalType::DECIMAL(18, 2)}) {
    bind.columns.back().type = type;
    CHECK_FALSE(sirius::planner::iceberg_flat_mapping_supported(&bind));
  }
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
dv_test_source dv_test_container(std::string properties, bool corrupt_second = false)
{
  auto payload = dv_test_payload();
  auto blobs   = payload;
  if (corrupt_second) payload.back() ^= std::byte{1};
  blobs.insert(blobs.end(), payload.begin(), payload.end());
  std::string json = R"({"blobs":[)";
  for (int i = 0; i < 2; ++i) {
    if (i) json += ",";
    json +=
      R"({"type":"deletion-vector-v1","fields":[],"snapshot-id":-1,"sequence-number":-1,"offset":)" +
      std::to_string(4 + i * 44) + R"(,"length":44,"properties":{)" +
      (i ? properties : std::string{}) + "}}";
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
    CHECK_THROWS_AS(prepare(), sirius::io::descriptor_mismatch_error);
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
  if (mode == 6) unit.snapshot_min = 1;
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
  CHECK_FALSE(parquet_unit_dependencies_supported(unit));
  CHECK_THROWS_WITH(parquet_unit_envelope(unit),
                    "DuckLake per-row snapshot-filtered data is outside the local scan profile");
  unit.snapshot_min.reset();
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
