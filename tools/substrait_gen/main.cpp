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

// sirius_substrait_gen: turns TPC-H style qNN.sql files into Substrait JSON
// plans with the DuckDB + duckdb-substrait that Sirius parses plans with, and
// verifies each plan by a CPU round trip (from_substrait_json vs. the SQL).
//
//   sirius_substrait_gen --input DIR --queries DIR --out DIR [--only q01,q06]
//
// Exit codes: 0 every requested query round-tripped; 3 at least one failed
// (the others are still written); 2 usage error; 1 I/O or setup error.

#include "core_functions_extension.hpp"
#include "duckdb.hpp"
#include "duckdb/main/valid_checker.hpp"
#include "duckdb/parser/parser.hpp"
#include "duckdb/parser/statement/select_statement.hpp"
#include "parquet_extension.hpp"
#include "substrait_extension.hpp"
#include "yyjson.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <regex>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace fs = std::filesystem;
using namespace duckdb_yyjson;  // NOLINT

namespace {

constexpr int kExitOk          = 0;
constexpr int kExitIoError     = 1;
constexpr int kExitUsage       = 2;
constexpr int kExitQueryFailed = 3;

const char* const kTables[] = {
  "customer", "lineitem", "nation", "orders", "part", "partsupp", "region", "supplier"};

struct Options {
  fs::path input;
  fs::path queries;
  fs::path out;
  std::set<std::string> only;
};

struct ManifestEntry {
  std::string plan;
  std::vector<std::string> logical_types;
  std::vector<std::string> physical_types;
};

class QueryFailure : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

void print_usage(std::ostream& os)
{
  os << "Usage: sirius_substrait_gen --input DIR --queries DIR --out DIR [--only q01,q06]\n"
        "  --input    dir with <table>.parquet for the 8 TPC-H tables\n"
        "  --queries  dir with qNN.sql (q01..q22)\n"
        "  --out      output dir; writes qNN.substrait.json for each query that produced a plan,\n"
        "             and result-types.json\n"
        "  --only     restrict to these queries (default: every qNN.sql found)\n"

        "Exit: 0 all requested queries round-trip; 3 at least one failed; 1/2 on IO/usage errors\n";
}

std::string sql_literal(const std::string& s)
{
  std::string out = "'";
  for (char c : s) {
    if (c == '\'') { out += '\''; }
    out += c;
  }
  return out + "'";
}

std::string strip_sql(std::string sql)
{
  auto is_trailing = [](char c) { return c == ';' || std::isspace(static_cast<unsigned char>(c)); };
  while (!sql.empty() && is_trailing(sql.back())) {
    sql.pop_back();
  }
  size_t start = 0;
  while (start < sql.size() && std::isspace(static_cast<unsigned char>(sql[start]))) {
    ++start;
  }
  return sql.substr(start);
}

std::string read_file(const fs::path& path)
{
  std::ifstream in(path, std::ios::binary);
  if (!in) { throw std::runtime_error("cannot read " + path.string()); }
  std::ostringstream ss;
  ss << in.rdbuf();
  return ss.str();
}

void write_file(const fs::path& path, const std::string& content)
{
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  if (!out) { throw std::runtime_error("cannot write " + path.string()); }
  out << content;
  if (!out) { throw std::runtime_error("failed writing " + path.string()); }
}

std::string json_string(const std::string& s)
{
  std::string out = "\"";
  for (char c : s) {
    switch (c) {
      case '"': out += "\\\""; break;
      case '\\': out += "\\\\"; break;
      case '\n': out += "\\n"; break;
      case '\t': out += "\\t"; break;
      default: out += c;
    }
  }
  return out + "\"";
}

// Re-serializes protobuf's JSON with 2-space indentation.
std::string pretty_json(const std::string& json)
{
  yyjson_doc* doc = yyjson_read(json.data(), json.size(), 0);
  if (doc == nullptr) { throw QueryFailure("get_substrait_json returned invalid JSON"); }
  size_t len = 0;
  char* text = yyjson_write(doc, YYJSON_WRITE_PRETTY_TWO_SPACES, &len);
  yyjson_doc_free(doc);
  if (text == nullptr) { throw QueryFailure("could not pretty-print the plan JSON"); }
  std::string out(text, len);
  free(text);
  return out + "\n";
}

// The result-types.json "physical_types" name of a result column: the Arrow
// layout DuckDB exports the type as, matched by arrow_matches_physical_type in
// test/cpp/exec/test_sirius_ffi_embedder.cpp.
std::string physical_type(const duckdb::LogicalType& type)
{
  using duckdb::LogicalTypeId;
  switch (type.id()) {
    case LogicalTypeId::VARCHAR: return "string";
    case LogicalTypeId::BIGINT: return "i64";
    case LogicalTypeId::INTEGER: return "i32";
    case LogicalTypeId::SMALLINT: return "i16";
    case LogicalTypeId::TINYINT: return "i8";
    case LogicalTypeId::DOUBLE: return "fp64";
    case LogicalTypeId::FLOAT: return "fp32";
    case LogicalTypeId::DATE: return "date";
    // DuckDB exports HUGEINT to Arrow as decimal128(38,0) (16-byte int128).
    case LogicalTypeId::HUGEINT: return "decimal(38,0)";
    case LogicalTypeId::DECIMAL:
      return "decimal(" + std::to_string(duckdb::DecimalType::GetWidth(type)) + "," +
             std::to_string(duckdb::DecimalType::GetScale(type)) + ")";
    default: throw QueryFailure("unsupported result type " + type.ToString());
  }
}

bool is_float(const duckdb::LogicalType& type)
{
  return type.id() == duckdb::LogicalTypeId::FLOAT || type.id() == duckdb::LogicalTypeId::DOUBLE;
}

// One cell: FLOAT/DOUBLE columns compare as doubles within a relative
// tolerance (aggregation order changes the last bits), every other column
// exactly by its string form.
struct Cell {
  bool is_null  = false;
  bool is_float = false;
  double number = 0;
  std::string text;
};

using Row = std::vector<Cell>;

bool cell_less(const Cell& a, const Cell& b)
{
  if (a.is_null != b.is_null) { return a.is_null < b.is_null; }
  if (a.is_float) { return a.number < b.number; }
  return a.text < b.text;
}

bool cell_equal(const Cell& a, const Cell& b)
{
  if (a.is_null || b.is_null) { return a.is_null == b.is_null; }
  if (!a.is_float) { return a.text == b.text; }
  double scale = std::max({1.0, std::fabs(a.number), std::fabs(b.number)});
  return std::fabs(a.number - b.number) <= 1e-6 * scale;
}

// The result's rows in result order, or sorted when the query does not order
// them.
std::vector<Row> collect_rows(duckdb::MaterializedQueryResult& result, bool ordered)
{
  std::vector<Row> rows;
  rows.reserve(result.RowCount());
  for (auto& chunk_row : result.Collection().Rows()) {
    Row row;
    for (duckdb::idx_t c = 0; c < result.ColumnCount(); ++c) {
      duckdb::Value v = chunk_row.GetValue(c);
      Cell cell;
      if (v.IsNull()) {
        cell.is_null = true;
      } else if (is_float(result.types[c])) {
        cell.is_float = true;
        cell.number   = v.GetValue<double>();
      } else {
        cell.text = v.ToString();
      }
      row.push_back(std::move(cell));
    }
    rows.push_back(std::move(row));
  }
  if (!ordered) {
    auto row_less = [](const Row& a, const Row& b) {
      return std::lexicographical_compare(a.begin(), a.end(), b.begin(), b.end(), cell_less);
    };
    std::sort(rows.begin(), rows.end(), row_less);
  }
  return rows;
}

// Whether the query's top-level SELECT (or set operation) has an ORDER BY, so
// the plan must reproduce the row order too.
bool has_top_level_order_by(const std::string& sql)
{
  duckdb::Parser parser;
  parser.ParseQuery(sql);
  if (parser.statements.size() != 1 ||
      parser.statements[0]->type != duckdb::StatementType::SELECT_STATEMENT) {
    return false;
  }
  auto& select = parser.statements[0]->Cast<duckdb::SelectStatement>();
  for (auto& modifier : select.node->modifiers) {
    if (modifier->type == duckdb::ResultModifierType::ORDER_MODIFIER) { return true; }
  }
  return false;
}

std::unique_ptr<duckdb::MaterializedQueryResult> run(duckdb::Connection& con,
                                                     const std::string& sql,
                                                     const std::string& what)
{
  auto result = con.Query(sql);
  if (result->HasError()) { throw QueryFailure(what + ": " + result->GetError()); }
  return result;
}

std::string get_plan(duckdb::Connection& con, const std::string& sql)
{
  auto result = run(con, "CALL get_substrait_json(" + sql_literal(sql) + ")", "get_substrait_json");
  if (result->RowCount() != 1 || result->ColumnCount() != 1) {
    throw QueryFailure("get_substrait_json returned an unexpected shape");
  }
  return pretty_json(result->GetValue(0, 0).ToString());
}

void compare_results(duckdb::MaterializedQueryResult& from_plan,
                     duckdb::MaterializedQueryResult& from_sql,
                     bool ordered)
{
  if (from_plan.ColumnCount() != from_sql.ColumnCount()) {
    throw QueryFailure("column count " + std::to_string(from_plan.ColumnCount()) + " != SQL " +
                       std::to_string(from_sql.ColumnCount()));
  }
  for (duckdb::idx_t c = 0; c < from_sql.ColumnCount(); ++c) {
    if (from_plan.types[c] != from_sql.types[c]) {
      throw QueryFailure("result types differ between the plan and the SQL: column " +
                         std::to_string(c) + " (" + from_sql.names[c] + ") is " +
                         from_plan.types[c].ToString() + " vs SQL " + from_sql.types[c].ToString());
    }
  }
  if (from_plan.RowCount() != from_sql.RowCount()) {
    throw QueryFailure("row count " + std::to_string(from_plan.RowCount()) + " != SQL " +
                       std::to_string(from_sql.RowCount()));
  }
  auto a = collect_rows(from_plan, ordered);
  auto b = collect_rows(from_sql, ordered);
  for (size_t r = 0; r < a.size(); ++r) {
    for (size_t c = 0; c < a[r].size(); ++c) {
      if (!cell_equal(a[r][c], b[r][c])) {
        auto show = [](const Cell& x) {
          return x.is_null ? std::string("NULL") : x.is_float ? std::to_string(x.number) : x.text;
        };
        throw QueryFailure("value mismatch at " + std::string(ordered ? "" : "sorted ") + "row " +
                           std::to_string(r) + " column " + std::to_string(c) + ": plan " +
                           show(a[r][c]) + " vs SQL " + show(b[r][c]));
      }
    }
  }
}

// Reads the existing result-types.json so a --only run keeps the other
// queries' entries.
std::map<std::string, ManifestEntry> read_manifest(const fs::path& path)
{
  std::map<std::string, ManifestEntry> entries;
  if (!fs::exists(path)) { return entries; }
  std::string text = read_file(path);
  yyjson_doc* doc  = yyjson_read(text.data(), text.size(), 0);
  if (doc == nullptr) { throw std::runtime_error("cannot parse " + path.string()); }
  yyjson_val* queries = yyjson_obj_get(yyjson_doc_get_root(doc), "queries");
  size_t idx, max;
  yyjson_val *key, *val;
  // Missing or non-string values read as "" instead of crashing on nullptr.
  auto str = [](yyjson_val* v) {
    const char* s = yyjson_get_str(v);
    return std::string(s != nullptr ? s : "");
  };
  auto strings = [&str](yyjson_val* arr) {
    std::vector<std::string> out;
    size_t i, n;
    yyjson_val* item;
    yyjson_arr_foreach(arr, i, n, item) { out.push_back(str(item)); }
    return out;
  };
  if (yyjson_is_obj(queries)) {
    yyjson_obj_foreach(queries, idx, max, key, val)
    {
      ManifestEntry entry;
      entry.plan           = str(yyjson_obj_get(val, "plan"));
      entry.logical_types  = strings(yyjson_obj_get(val, "logical_types"));
      entry.physical_types = strings(yyjson_obj_get(val, "physical_types"));
      entries[str(key)]    = std::move(entry);
    }
  }
  yyjson_doc_free(doc);
  return entries;
}

void write_manifest(const fs::path& path, const std::map<std::string, ManifestEntry>& entries)
{
  auto list = [](const std::vector<std::string>& items) {
    std::string out = "[";
    for (size_t i = 0; i < items.size(); ++i) {
      out += (i == 0 ? "\n        " : ",\n        ") + json_string(items[i]);
    }
    return out + (items.empty() ? "]" : "\n      ]");
  };
  std::string out = "{\n  \"queries\": {";
  bool first      = true;
  for (auto& [name, e] : entries) {
    out += first ? "\n" : ",\n";
    first = false;
    out += "    " + json_string(name) + ": {\n";
    out += "      \"logical_types\": " + list(e.logical_types) + ",\n";
    out += "      \"physical_types\": " + list(e.physical_types) + ",\n";
    out += "      \"plan\": " + json_string(e.plan) + "\n    }";
  }
  out += entries.empty() ? "},\n" : "\n  },\n";
  out += "  \"version\": 1\n}\n";
  write_file(path, out);
}

// DuckDB internal errors carry a stack trace; the reason is the first line.
std::string first_line(const std::string& message) { return message.substr(0, message.find('\n')); }

// In-memory DuckDB with the 8 TPC-H tables materialized from Parquet, so plans
// read real tables (namedTable) with real statistics.
std::pair<std::unique_ptr<duckdb::DuckDB>, std::unique_ptr<duckdb::Connection>> open_database(
  const fs::path& input)
{
  auto db = std::make_unique<duckdb::DuckDB>(nullptr);
  db->LoadStaticExtension<duckdb::CoreFunctionsExtension>();
  db->LoadStaticExtension<duckdb::ParquetExtension>();
  db->LoadStaticExtension<duckdb::SubstraitExtension>();
  auto con = std::make_unique<duckdb::Connection>(*db);
  for (const char* table : kTables) {
    fs::path parquet = input / (std::string(table) + ".parquet");
    if (!fs::is_regular_file(parquet)) { throw std::runtime_error("missing " + parquet.string()); }
    auto r = con->Query("CREATE TABLE " + std::string(table) + " AS SELECT * FROM read_parquet(" +
                        sql_literal(parquet.string()) + ")");
    if (r->HasError()) {
      throw std::runtime_error("loading " + parquet.string() + ": " + r->GetError());
    }
  }
  return {std::move(db), std::move(con)};
}

int parse_args(int argc, char** argv, Options& opts)
{
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "-h" || arg == "--help") {
      print_usage(std::cout);
      std::exit(kExitOk);
    }
    if (i + 1 >= argc) {
      std::cerr << "error: " << arg << " needs a value (or is unknown)\n";
      return kExitUsage;
    }
    std::string value = argv[++i];
    if (arg == "--input") {
      opts.input = value;
    } else if (arg == "--queries") {
      opts.queries = value;
    } else if (arg == "--out") {
      opts.out = value;
    } else if (arg == "--only") {
      std::stringstream ss(value);
      std::string item;
      while (std::getline(ss, item, ',')) {
        if (!item.empty()) { opts.only.insert(item); }
      }
    } else {
      std::cerr << "error: unknown argument '" << arg << "'\n";
      return kExitUsage;
    }
  }
  if (opts.input.empty() || opts.queries.empty() || opts.out.empty()) {
    std::cerr << "error: --input, --queries and --out are required\n";
    return kExitUsage;
  }
  return kExitOk;
}

}  // namespace

int main(int argc, char** argv)
{
  Options opts;
  if (int rc = parse_args(argc, argv, opts); rc != kExitOk) {
    print_usage(std::cerr);
    return rc;
  }

  // Requested queries: every qNN.sql in --queries, or the --only subset.
  static const std::regex query_file(R"(q(\d\d)\.sql)");
  std::map<std::string, fs::path> sql_files;
  std::error_code ec;
  for (auto& entry : fs::directory_iterator(opts.queries, ec)) {
    std::string name = entry.path().filename().string();
    if (std::regex_match(name, query_file)) { sql_files[name.substr(0, 3)] = entry.path(); }
  }
  if (ec) {
    std::cerr << "error: cannot list " << opts.queries << ": " << ec.message() << "\n";
    return kExitIoError;
  }
  std::vector<std::string> requested;
  if (opts.only.empty()) {
    for (auto& [name, path] : sql_files) {
      requested.push_back(name);
    }
  } else {
    for (auto& name : opts.only) {
      if (sql_files.count(name) == 0) {
        std::cerr << "error: --only " << name << ": no " << name << ".sql in " << opts.queries
                  << "\n";
        return kExitUsage;
      }
      requested.push_back(name);
    }
  }
  if (requested.empty()) {
    std::cerr << "error: no qNN.sql files in " << opts.queries << "\n";
    return kExitIoError;
  }

  try {
    fs::create_directories(opts.out);
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    return kExitIoError;
  }
  const fs::path manifest_path = opts.out / "result-types.json";

  std::map<std::string, ManifestEntry> manifest;
  std::unique_ptr<duckdb::DuckDB> db;
  std::unique_ptr<duckdb::Connection> con;
  try {
    manifest          = read_manifest(manifest_path);
    std::tie(db, con) = open_database(opts.input);
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    return kExitIoError;
  }

  // Writes the manifest for the queries completed so far; false on error.
  auto save_manifest = [&]() {
    try {
      write_manifest(manifest_path, manifest);
      return true;
    } catch (const std::exception& e) {
      std::cerr << "error: " << e.what() << "\n";
      return false;
    }
  };

  std::vector<std::string> failed;
  for (auto& name : requested) {
    const std::string plan_name = name + ".substrait.json";
    const fs::path plan_path    = opts.out / plan_name;
    std::string plan;
    ManifestEntry entry;
    duckdb::idx_t row_count = 0;
    bool ok                 = false;
    try {
      std::string sql = strip_sql(read_file(sql_files[name]));

      plan = get_plan(*con, sql);

      auto from_plan =
        run(*con, "CALL from_substrait_json(" + sql_literal(plan) + ")", "from_substrait_json");
      auto from_sql = run(*con, sql, "SQL");
      compare_results(*from_plan, *from_sql, has_top_level_order_by(sql));

      entry.plan = plan_name;
      for (auto& type : from_plan->types) {
        entry.logical_types.push_back(type.ToString());
        entry.physical_types.push_back(physical_type(type));
      }
      row_count = from_sql->RowCount();
      ok        = true;
    } catch (const QueryFailure& e) {
      std::cout << name << ": ROUND-TRIP FAIL: " << first_line(e.what()) << std::endl;
    } catch (const std::exception& e) {
      // DuckDB throws its own exceptions from inside the substrait functions.
      duckdb::ErrorData error(e);
      std::cout << name << ": ROUND-TRIP FAIL: " << first_line(error.Message()) << std::endl;
    }
    if (ok) {
      try {
        write_file(plan_path, plan);
      } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        manifest.erase(name);
        fs::remove(plan_path, ec);
        save_manifest();
        return kExitIoError;
      }
      manifest[name] = std::move(entry);
      std::cout << name << ": ok (" << row_count << " rows)" << std::endl;
      continue;
    }
    failed.push_back(name);
    // Never leave a stale plan from an earlier run next to a failing query.
    manifest.erase(name);
    fs::remove(plan_path, ec);
    // A fatal (e.g. internal) error invalidates the whole database; only then
    // start the next query from a fresh one.
    if (duckdb::ValidChecker::IsInvalidated(*db->instance)) {
      try {
        con.reset();
        db.reset();
        std::tie(db, con) = open_database(opts.input);
      } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        save_manifest();
        return kExitIoError;
      }
    }
  }

  if (!save_manifest()) { return kExitIoError; }

  std::string failed_list;
  for (size_t i = 0; i < failed.size(); ++i) {
    failed_list += (i ? "," : "") + failed[i];
  }
  std::cout << "generated " << (requested.size() - failed.size()) << "/" << requested.size()
            << "; failed: " << (failed.empty() ? "none" : failed_list) << std::endl;
  return failed.empty() ? kExitOk : kExitQueryFailed;
}
