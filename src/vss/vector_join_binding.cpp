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
#include "duckdb/common/enums/catalog_type.hpp"
#include "duckdb/common/serializer/deserializer.hpp"
#include "duckdb/common/serializer/serializer.hpp"
#include "duckdb/common/types/value.hpp"
#include "duckdb/parser/keyword_helper.hpp"
#include "duckdb/parser/parser.hpp"
#include "duckdb/parser/qualified_name.hpp"
#include "duckdb/planner/binder.hpp"
#include "duckdb/storage/data_table.hpp"
#include "scan_manager/sirius_scan_manager.hpp"
#include "sirius_context.hpp"

#include <algorithm>
#include <limits>

namespace sirius::vss {

duckdb::BoundStatement bind_view_select(duckdb::ClientContext& context,
                                        const vector_join_side& side)
{
  auto const sql = "SELECT * FROM " + duckdb::KeywordHelper::WriteOptionallyQuoted(side.catalog) +
                   "." + duckdb::KeywordHelper::WriteOptionallyQuoted(side.schema) + "." +
                   duckdb::KeywordHelper::WriteOptionallyQuoted(side.table);
  duckdb::Parser parser(context.GetParserOptions());
  parser.ParseQuery(sql);
  if (parser.statements.size() != 1) {
    throw duckdb::BinderException("sirius_knn_join: could not bind view '" + side.table + "'");
  }
  auto binder = duckdb::Binder::CreateBinder(context);
  return binder->Bind(*parser.statements[0]);
}

std::int64_t resolve_vector_join_side(duckdb::ClientContext& context,
                                      duckdb::SiriusContext& sirius_ctx,
                                      const std::string& label,
                                      const std::string& table_arg,
                                      const std::string& column_arg,
                                      const std::string& schema_name,
                                      const std::vector<std::string>& out_cols,
                                      bool require_pin,
                                      vector_join_side& side,
                                      duckdb::vector<duckdb::LogicalType>& out_types,
                                      duckdb::vector<duckdb::string>& out_names,
                                      std::uint64_t& out_num_rows)
{
  side.column = column_arg;

  // Resolve the table + vector column against the catalog.
  auto const qname          = duckdb::QualifiedName::Parse(table_arg);
  std::string const catalog = qname.catalog;
  std::string const schema  = !qname.schema.empty() ? qname.schema : schema_name;
  auto& entry_base          = duckdb::Catalog::GetEntry(
    context, duckdb::CatalogType::TABLE_ENTRY, catalog, schema, qname.name);
  // A VIEW lives in the TABLE_ENTRY namespace, so GetEntry returns one and a Cast to a table
  // entry would be undefined behaviour. A view is a named subquery: on a scanned corpus side the
  // planner binds it and streams its rows into the join, which is what a filtered or joined
  // corpus needs without a CTAS or a pin. Its columns come from binding "SELECT * FROM view".
  bool const is_view = entry_base.type == duckdb::CatalogType::VIEW_ENTRY;
  if (is_view && label == "left") {
    throw duckdb::BinderException(
      "sirius_knn_join: " + label + " '" + qname.name +
      "' is a view; a probe-side relation is passed as a subquery via sirius_knn_join_rel");
  }
  if (is_view && require_pin) {
    throw duckdb::BinderException(
      "sirius_knn_join: " + label + " '" + qname.name +
      "' is a view; a view corpus streams through a scan, so pass build_source => 'scan'");
  }
  if (!is_view && entry_base.type != duckdb::CatalogType::TABLE_ENTRY) {
    throw duckdb::BinderException(
      "sirius_knn_join: " + label + " '" + qname.name + "' is a " +
      duckdb::CatalogTypeToString(entry_base.type) +
      ", not a base table or view. Materialize it (CREATE TABLE ... AS SELECT ...) or wrap it "
      "in a view; a subquery is supported on the probe side via sirius_knn_join_rel.");
  }
  side.catalog = entry_base.ParentCatalog().GetName();
  side.schema  = entry_base.ParentSchema().name;
  side.table   = entry_base.name;  // catalog-resolved name (matches query-side derivation)
  side.is_view = is_view;

  duckdb::vector<duckdb::string> schema_names;
  duckdb::vector<duckdb::LogicalType> schema_types;
  std::uint64_t table_rows = 0;
  if (is_view) {
    auto bound   = bind_view_select(context, side);
    schema_names = std::move(bound.names);
    schema_types = std::move(bound.types);
  } else {
    auto& entry         = entry_base.Cast<duckdb::DuckTableEntry>();
    auto const& columns = entry.GetColumns();
    schema_names        = columns.GetColumnNames();
    schema_types        = columns.GetColumnTypes();
    table_rows          = static_cast<std::uint64_t>(entry.GetStorage().GetTotalRows());
  }

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

  // A side the operator reads straight out of a pin can only emit what the pin holds. A side
  // fed by the build phase is read by an ordinary scan, which serves the table pinned or not --
  // the pin is a cache in front of it, not the source -- so the catalog decides the columns and
  // an absent pin is not an error. That is what stops a column-subset pin from making the rest
  // of the table unusable in the same query.
  auto pin = sirius_ctx.get_scan_manager().find_pinned_entry_for_duckdb_table(
    side.catalog, side.schema, side.table);
  if (pin == nullptr && require_pin) {
    throw duckdb::BinderException("sirius_knn_join: " + label + " table '" + side.table +
                                  "' must be pinned");
  }
  auto const emittable = [&](const std::string& col) {
    if (pin == nullptr || !require_pin) { return true; }
    auto const& pinned_names = pin->cache_info.column_names();
    return std::ranges::find(pinned_names.begin(), pinned_names.end(), col) != pinned_names.end();
  };

  if (out_cols.empty()) {
    // Everything the table has, MINUS its FLOAT[N] columns. The relational probe side already
    // drops the one column it joins on, for a reason that covers all of them: an embedding is
    // the join's input rather than something the caller wants echoed back, and at 128-960 floats
    // it is by far the widest column -- a bare `SELECT *` that includes one prints a screenful
    // of raw coordinates per row and buries the ids and the score.
    //
    // Dropping only the joined column is not enough: joining two embedding columns of ONE table
    // (`sirius_knn_join('t','a_vec','t','b_vec')`) leaves each side echoing the OTHER side's
    // vector, so `SELECT *` is still unreadable. Keying on the type rather than on which column
    // this side happens to join on is both simpler to state and the one that actually holds.
    //
    // Nothing is withdrawn -- name a vector in left/right_output_columns to get it back.
    for (auto const& name : schema_names) {
      auto const& col_type = type_of(name);
      if (col_type.id() == duckdb::LogicalTypeId::ARRAY &&
          duckdb::ArrayType::GetChildType(col_type).id() == duckdb::LogicalTypeId::FLOAT) {
        continue;
      }
      if (emittable(name)) { side.output_columns.push_back(name); }
    }
  } else {
    for (auto const& col : out_cols) {
      bool const in_catalog =
        std::ranges::find(schema_names.begin(), schema_names.end(), col) != schema_names.end();
      if (!in_catalog) {
        throw duckdb::BinderException("sirius_knn_join: " + label + " column '" + col +
                                      "' not found in table '" + side.table + "'");
      }
      if (!emittable(col)) {
        throw duckdb::BinderException(
          "sirius_knn_join: " + label + " output column '" + col + "' is not pinned on table '" +
          side.table + "'; pin it (pin_table cols => [...]) or omit the output_columns list");
      }
      side.output_columns.push_back(col);
    }
  }

  // Row count for the cardinality estimate and the plan's k clamp. The pin's count is the
  // authority when the operator reads the pin; otherwise the table's own.
  out_num_rows =
    (pin != nullptr && require_pin) ? static_cast<std::uint64_t>(pin->num_rows) : table_rows;

  for (auto const& col : side.output_columns) {
    out_types.push_back(type_of(col));
    out_names.push_back(label + "_" + col);
  }
  return dim;
}

std::int64_t resolve_relational_probe_side(const duckdb::vector<duckdb::LogicalType>& input_types,
                                           const duckdb::vector<duckdb::string>& input_names,
                                           const std::string& column_arg,
                                           const std::vector<std::string>& out_cols,
                                           vector_join_side& side,
                                           duckdb::vector<duckdb::LogicalType>& out_types,
                                           duckdb::vector<duckdb::string>& out_names)
{
  side.column           = column_arg;
  side.from_relation    = true;
  side.relation_columns = std::vector<std::string>(input_names.begin(), input_names.end());
  // catalog/schema/table stay empty: there is no table behind this side, and every path that
  // would look one up is the pinned path, which this side never takes.

  auto index_of = [&](const std::string& col) -> std::size_t {
    for (std::size_t i = 0; i < input_names.size(); ++i) {
      if (input_names[i] == col) { return i; }
    }
    throw duckdb::BinderException("sirius_knn_join_rel: column '" + col +
                                  "' is not produced by the probe relation");
  };

  auto const& vec_type = input_types[index_of(side.column)];
  if (vec_type.id() != duckdb::LogicalTypeId::ARRAY ||
      duckdb::ArrayType::GetChildType(vec_type).id() != duckdb::LogicalTypeId::FLOAT) {
    throw duckdb::BinderException("sirius_knn_join_rel: probe column '" + side.column +
                                  "' must be a FLOAT[N] array column");
  }

  if (out_cols.empty()) {
    // Everything the relation produces, minus the vector column: it is the join's input, not
    // usually something the caller wants echoed back, and it is by far the widest column.
    for (auto const& name : input_names) {
      if (name != side.column) { side.output_columns.push_back(name); }
    }
  } else {
    side.output_columns = out_cols;
  }

  for (auto const& col : side.output_columns) {
    out_types.push_back(input_types[index_of(col)]);
    out_names.push_back("left_" + col);
  }
  return static_cast<std::int64_t>(duckdb::ArrayType::GetSize(vec_type));
}

std::uint64_t estimate_vector_join_cardinality(const vector_join_request& req,
                                               std::uint64_t left_rows,
                                               std::uint64_t right_rows)
{
  // create_plan_knn_join lowers k to the right-table row count; asking for more
  // neighbours than the corpus holds cannot produce more pairs than exist.
  auto const k = std::min(static_cast<std::uint64_t>(std::max<std::int64_t>(req.k, 0)), right_rows);

  // Global top-k finishes in a TOP_N above materialize, which cuts the whole
  // result to k rows regardless of how many left rows fed it.
  if (req.mode == vector_join_mode::global_top_k) { return k; }

  // Per-row emits exactly k rows per left row. Threshold searches to the same
  // depth and then drops pairs outside eps, so this is its ceiling, not its
  // expectation — no selectivity is knowable at bind time.
  constexpr auto max_rows = std::numeric_limits<std::uint64_t>::max();
  if (k != 0 && left_rows > max_rows / k) { return max_rows; }
  return left_rows * k;
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

namespace {

duckdb::vector<std::string> to_duckdb_strings(const std::vector<std::string>& v)
{
  return duckdb::vector<std::string>(v.begin(), v.end());
}

std::vector<std::string> from_duckdb_strings(const duckdb::vector<std::string>& v)
{
  return std::vector<std::string>(v.begin(), v.end());
}

void write_side(duckdb::Serializer& s, duckdb::field_id_t base, const vector_join_side& side)
{
  s.WriteProperty(base + 0, "catalog", side.catalog);
  s.WriteProperty(base + 1, "schema", side.schema);
  s.WriteProperty(base + 2, "table", side.table);
  s.WriteProperty(base + 3, "column", side.column);
  s.WriteProperty(base + 4, "output_columns", to_duckdb_strings(side.output_columns));
  s.WriteProperty(base + 5, "is_view", side.is_view);
  s.WriteProperty(base + 6, "from_relation", side.from_relation);
  s.WriteProperty(base + 7, "relation_columns", to_duckdb_strings(side.relation_columns));
}

vector_join_side read_side(duckdb::Deserializer& d, duckdb::field_id_t base)
{
  vector_join_side side;
  side.catalog = d.ReadProperty<std::string>(base + 0, "catalog");
  side.schema  = d.ReadProperty<std::string>(base + 1, "schema");
  side.table   = d.ReadProperty<std::string>(base + 2, "table");
  side.column  = d.ReadProperty<std::string>(base + 3, "column");
  side.output_columns =
    from_duckdb_strings(d.ReadProperty<duckdb::vector<std::string>>(base + 4, "output_columns"));
  side.is_view       = d.ReadProperty<bool>(base + 5, "is_view");
  side.from_relation = d.ReadProperty<bool>(base + 6, "from_relation");
  side.relation_columns =
    from_duckdb_strings(d.ReadProperty<duckdb::vector<std::string>>(base + 7, "relation_columns"));
  return side;
}

}  // namespace

void serialize_vector_join_bind_data(duckdb::Serializer& serializer,
                                     const duckdb::optional_ptr<duckdb::FunctionData> bind_data,
                                     const duckdb::TableFunction& /*function*/)
{
  auto const& data = bind_data->Cast<SiriusVectorJoinBindData>();
  auto const& req  = data.req;
  write_side(serializer, 100, req.left);
  write_side(serializer, 110, req.right);
  serializer.WriteProperty(120, "mode", static_cast<std::uint8_t>(req.mode));
  serializer.WriteProperty(121, "metric", req.metric);
  serializer.WriteProperty(122, "search_mode", static_cast<std::uint8_t>(req.search_mode));
  serializer.WriteProperty(123, "k", req.k);
  serializer.WriteProperty(124, "n_clusters", req.n_clusters);
  serializer.WriteProperty(125, "n_probes", req.n_probes);
  serializer.WriteProperty(126, "dim", req.dim);
  serializer.WriteProperty(127, "eps", req.eps);
  serializer.WriteProperty(128, "output_type", static_cast<std::uint8_t>(req.output_type));
  serializer.WriteProperty(129, "build_from_scan", req.build_from_scan);
  serializer.WriteProperty(130, "probe_from_scan", req.probe_from_scan);
  serializer.WriteProperty(131, "clustering", req.clustering);
  serializer.WriteProperty(132, "build_cluster_column", req.build_cluster_column);
  duckdb::vector<std::string> pred_columns;
  duckdb::vector<std::uint8_t> pred_ops;
  duckdb::vector<duckdb::Value> pred_values;
  for (auto const& p : req.right_predicates) {
    pred_columns.push_back(p.column);
    pred_ops.push_back(static_cast<std::uint8_t>(p.cmp));
    pred_values.push_back(p.value);
  }
  serializer.WriteProperty(133, "predicate_columns", pred_columns);
  serializer.WriteProperty(134, "predicate_ops", pred_ops);
  serializer.WriteProperty(135, "predicate_values", pred_values);
  serializer.WriteProperty(136, "probe_scalar", req.probe_scalar);
  serializer.WriteProperty(140, "left_rows", data.left_rows);
  serializer.WriteProperty(141, "right_rows", data.right_rows);
  serializer.WriteProperty(142, "probe_is_relation", data.probe_is_relation);
}

duckdb::unique_ptr<duckdb::FunctionData> deserialize_vector_join_bind_data(
  duckdb::Deserializer& deserializer, duckdb::TableFunction& /*function*/)
{
  auto result = duckdb::make_uniq<SiriusVectorJoinBindData>();
  auto& req   = result->req;
  req.left    = read_side(deserializer, 100);
  req.right   = read_side(deserializer, 110);
  req.mode    = static_cast<vector_join_mode>(deserializer.ReadProperty<std::uint8_t>(120, "mode"));
  req.metric  = deserializer.ReadProperty<std::string>(121, "metric");
  req.search_mode = static_cast<vector_join_search_mode>(
    deserializer.ReadProperty<std::uint8_t>(122, "search_mode"));
  req.k           = deserializer.ReadProperty<std::int64_t>(123, "k");
  req.n_clusters  = deserializer.ReadProperty<std::int64_t>(124, "n_clusters");
  req.n_probes    = deserializer.ReadProperty<std::int64_t>(125, "n_probes");
  req.dim         = deserializer.ReadProperty<std::int64_t>(126, "dim");
  req.eps         = deserializer.ReadProperty<double>(127, "eps");
  req.output_type = static_cast<vector_join_output_type>(
    deserializer.ReadProperty<std::uint8_t>(128, "output_type"));
  req.build_from_scan      = deserializer.ReadProperty<bool>(129, "build_from_scan");
  req.probe_from_scan      = deserializer.ReadProperty<bool>(130, "probe_from_scan");
  req.clustering           = deserializer.ReadProperty<std::string>(131, "clustering");
  req.build_cluster_column = deserializer.ReadProperty<std::string>(132, "build_cluster_column");
  auto const pred_columns =
    deserializer.ReadProperty<duckdb::vector<std::string>>(133, "predicate_columns");
  auto const pred_ops =
    deserializer.ReadProperty<duckdb::vector<std::uint8_t>>(134, "predicate_ops");
  auto const pred_values =
    deserializer.ReadProperty<duckdb::vector<duckdb::Value>>(135, "predicate_values");
  for (std::size_t i = 0; i < pred_columns.size(); ++i) {
    req.right_predicates.push_back(
      {pred_columns[i], static_cast<corpus_predicate::op>(pred_ops[i]), pred_values[i]});
  }
  req.probe_scalar          = deserializer.ReadProperty<bool>(136, "probe_scalar");
  result->left_rows         = deserializer.ReadProperty<std::uint64_t>(140, "left_rows");
  result->right_rows        = deserializer.ReadProperty<std::uint64_t>(141, "right_rows");
  result->probe_is_relation = deserializer.ReadProperty<bool>(142, "probe_is_relation");
  return std::move(result);
}

}  // namespace sirius::vss
