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

#include "vss/vector_join_binding.hpp"

#include "duckdb/catalog/catalog.hpp"
#include "duckdb/catalog/catalog_entry/duck_table_entry.hpp"
#include "duckdb/common/types/value.hpp"
#include "duckdb/parser/qualified_name.hpp"
#include "scan_manager/sirius_scan_manager.hpp"
#include "sirius_context.hpp"

#include <algorithm>

namespace sirius::vss {

std::int64_t resolve_vector_join_side(duckdb::ClientContext& context,
                                      duckdb::SiriusContext& sirius_ctx,
                                      const std::string& label,
                                      const std::string& table_arg,
                                      const std::string& column_arg,
                                      const std::string& schema_name,
                                      const std::vector<std::string>& out_cols,
                                      vector_join_side& side,
                                      duckdb::vector<duckdb::LogicalType>& out_types,
                                      duckdb::vector<duckdb::string>& out_names)
{
  side.column = column_arg;

  // Resolve the table + vector column against the catalog.
  auto const qname          = duckdb::QualifiedName::Parse(table_arg);
  std::string const catalog = qname.catalog;
  std::string const schema  = !qname.schema.empty() ? qname.schema : schema_name;
  auto& entry_base          = duckdb::Catalog::GetEntry(
    context, duckdb::CatalogType::TABLE_ENTRY, catalog, schema, qname.name);
  auto& entry  = entry_base.Cast<duckdb::DuckTableEntry>();
  side.catalog = entry.ParentCatalog().GetName();
  side.schema  = entry.ParentSchema().name;
  side.table   = entry.name;  // catalog-resolved name (matches query-side derivation)

  auto const& columns     = entry.GetColumns();
  auto const schema_names = columns.GetColumnNames();
  auto const schema_types = columns.GetColumnTypes();

  auto type_of = [&](const std::string& col) -> const duckdb::LogicalType& {
    for (std::size_t i = 0; i < schema_names.size(); ++i) {
      if (schema_names[i] == col) { return schema_types[i]; }
    }
    throw duckdb::BinderException("sirius_knn_join: " + label + " column '" + col +
                                  "' not found in table '" + side.table + "'");
  };

  auto const& vec_type = type_of(side.column);
  if (vec_type.id() != duckdb::LogicalTypeId::ARRAY ||
      duckdb::ArrayType::GetChildType(vec_type).id() != duckdb::LogicalTypeId::FLOAT) {
    throw duckdb::BinderException("sirius_knn_join: " + label + " column '" + side.column +
                                  "' must be a FLOAT[N] array column");
  }
  auto const dim = static_cast<std::int64_t>(duckdb::ArrayType::GetSize(vec_type));

  // The side must be pinned; output columns must be a subset of the pin.
  const auto* pin = sirius_ctx.get_scan_manager().find_pinned_entry_for_duckdb_table(
    side.catalog, side.schema, side.table);
  if (pin == nullptr) {
    throw duckdb::BinderException("sirius_knn_join: " + label + " table '" + side.table +
                                  "' must be pinned");
  }
  auto const& pinned_names = pin->cache_info.column_names();
  auto is_pinned           = [&](const std::string& col) {
    return std::ranges::find(pinned_names.begin(), pinned_names.end(), col) != pinned_names.end();
  };

  if (out_cols.empty()) {
    // Default to all pinned columns in catalog schema order
    for (auto const& name : schema_names) {
      if (is_pinned(name)) { side.output_columns.push_back(name); }
    }
  } else {
    for (auto const& col : out_cols) {
      bool const in_catalog =
        std::ranges::find(schema_names.begin(), schema_names.end(), col) != schema_names.end();
      if (!in_catalog) {
        throw duckdb::BinderException("sirius_knn_join: " + label + " column '" + col +
                                      "' not found in table '" + side.table + "'");
      }
      if (!is_pinned(col)) {
        throw duckdb::BinderException(
          "sirius_knn_join: " + label + " output column '" + col + "' is not pinned on table '" +
          side.table + "'; pin it (pin_table cols => [...]) or omit the output_columns list");
      }
      side.output_columns.push_back(col);
    }
  }

  for (auto const& col : side.output_columns) {
    out_types.push_back(type_of(col));
    out_names.push_back(label + "_" + col);
  }
  return dim;
}

std::vector<std::string> parse_output_columns(const duckdb::Value& v, const std::string& key)
{
  std::vector<std::string> out;
  for (auto const& c : duckdb::ListValue::GetChildren(v)) {
    out.push_back(c.ToString());
  }
  if (out.empty()) {
    throw duckdb::BinderException("sirius_knn_join: " + key +
                                  " cannot be empty; omit it to default to the pinned columns");
  }
  return out;
}

}  // namespace sirius::vss
