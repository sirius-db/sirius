#include "sirius_licenses.hpp"

#include "duckdb/common/types/value.hpp"
#include "duckdb/function/table_function.hpp"
#include "duckdb/parser/parsed_data/create_table_function_info.hpp"

#include <string>

namespace sirius_duckdb {
namespace {

/// sirius_licenses(): one row per bundled component.
struct licenses_function {
  struct state : duckdb::GlobalTableFunctionState {
    third_party_notices::row const* next = third_party_notices::rows_begin();
  };

  static duckdb::unique_ptr<duckdb::FunctionData> bind(duckdb::ClientContext&,
                                                       duckdb::TableFunctionBindInput&,
                                                       duckdb::vector<duckdb::LogicalType>& types,
                                                       duckdb::vector<duckdb::Identifier>& names)
  {
    for (auto const* name : {"component", "version", "license", "homepage", "license_text"}) {
      names.emplace_back(name);
      types.push_back(duckdb::LogicalType::VARCHAR);
    }
    return nullptr;
  }

  static duckdb::unique_ptr<duckdb::GlobalTableFunctionState> init(duckdb::ClientContext&,
                                                                   duckdb::TableFunctionInitInput&)
  {
    return duckdb::make_uniq<state>();
  }

  static duckdb::Value text_value(std::string_view text)
  {
    std::string value(text);
    if (!duckdb::Value::StringIsValid(value.data(), value.size())) {
      // License files are expected to be UTF-8; keep the ASCII text if one is not.
      for (auto& c : value) {
        if (static_cast<unsigned char>(c) >= 0x80) { c = '?'; }
      }
    }
    return duckdb::Value(std::move(value));
  }

  static duckdb::Value optional_value(char const* value)
  {
    return *value ? text_value(value) : duckdb::Value(duckdb::LogicalType::VARCHAR);
  }

  static void scan(duckdb::ClientContext&,
                   duckdb::TableFunctionInput& input,
                   duckdb::DataChunk& output)
  {
    auto& data          = input.global_state->Cast<state>();
    duckdb::idx_t count = 0;
    for (; data.next != third_party_notices::rows_end() && count < STANDARD_VECTOR_SIZE;
         ++data.next, ++count) {
      auto const& row = *data.next;
      output.data[0].Append(text_value(row.name));
      output.data[1].Append(optional_value(row.version));
      output.data[2].Append(text_value(row.license));
      output.data[3].Append(optional_value(row.homepage));
      output.data[4].Append(text_value(third_party_notices::text(row.text)));
    }
  }
};

}  // namespace

void third_party_notices::register_function(duckdb::ExtensionLoader& loader)
{
  duckdb::TableFunction function("sirius_licenses",
                                 {},
                                 licenses_function::scan,
                                 licenses_function::bind,
                                 licenses_function::init);
  duckdb::CreateTableFunctionInfo info(std::move(function));
  // A backend loaded after the loader's CPU fallback replaces the loader's table.
  info.on_conflict = duckdb::OnCreateConflict::REPLACE_ON_CONFLICT;
  duckdb::FunctionDescription description;
  description.description =
    "Lists Sirius and the third-party software bundled in this extension binary, one row per "
    "component, with its version, SPDX license expression, homepage and the full license and "
    "notice text (components sharing a text repeat it).";
  description.examples = {
    "SELECT component, version, license FROM sirius_licenses();",
    "SELECT license_text FROM sirius_licenses() WHERE component = 'DuckDB';",
    "COPY (SELECT component, version, license, homepage FROM sirius_licenses()) TO "
    "'sirius_licenses.csv';"};
  description.categories = {"metadata"};
  info.descriptions.push_back(std::move(description));
  loader.RegisterFunction(std::move(info));
}

}  // namespace sirius_duckdb
