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

#include "simpatico_copy_function.hpp"

#include "cudf/cudf_utils.hpp"
#include "helper/duckdb_chunk_staging.hpp"
#include "helper/type_conversions.hpp"
#include "plan_register.hpp"
#include "simpatico_file_ingest.hpp"
#include "sirius_context.hpp"

#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <rmm/mr/per_device_resource.hpp>

#include <api/simpatico_codegen.hpp>
#include <duckdb/common/exception.hpp>
#include <duckdb/common/string_util.hpp>
#include <duckdb/function/table_function.hpp>
#include <duckdb/main/client_context.hpp>
#include <log/logging.hpp>

#include <algorithm>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace sirius::compression {

namespace {

/// The plan used when the COPY names none and no plan file covers the table.
///
/// `identity` stores the column verbatim, which is the only per-column plan that is valid for
/// every type -- and a wrong-shaped plan is a hard compression failure, not a worse ratio. A file
/// written this way is still a real .hpln (chunk directory, logical types, zone maps), so it
/// prunes and decodes exactly like a compressed one; only its size differs.
constexpr const char* kIdentityPlanBlock = "input -> identity";

std::string identity_plan_for(std::size_t num_columns)
{
  std::string dsl;
  for (std::size_t i = 0; i < num_columns; i++) {
    if (i != 0) { dsl += "\n---\n"; }
    dsl += kIdentityPlanBlock;
  }
  return dsl;
}

std::string single_string_option(const duckdb::vector<duckdb::Value>& values,
                                 const std::string& name)
{
  if (values.size() != 1 || values[0].IsNull()) {
    throw duckdb::BinderException("COPY (FORMAT simpatico): '%s' expects a single string value",
                                  name);
  }
  return values[0].ToString();
}

std::size_t single_positive_option(const duckdb::vector<duckdb::Value>& values,
                                   const std::string& name)
{
  if (values.size() != 1 || values[0].IsNull()) {
    throw duckdb::BinderException("COPY (FORMAT simpatico): '%s' expects a single integer value",
                                  name);
  }
  auto const v = values[0]
                   .DefaultCastAs(duckdb::LogicalType(duckdb::LogicalTypeId::BIGINT))
                   .GetValue<std::int64_t>();
  if (v <= 0) {
    throw duckdb::BinderException(
      "COPY (FORMAT simpatico): '%s' must be positive, got %lld", name, static_cast<long long>(v));
  }
  return static_cast<std::size_t>(v);
}

struct simpatico_copy_bind_data : public duckdb::TableFunctionData {
  std::vector<std::string> names;
  /// The types recorded in the file's `logical_types` segment. They must be the types the payload
  /// really has, because the reader refuses a file whose declared type disagrees with the cuDF
  /// type its column decodes to -- so these are also what the staged cuDF columns are built from,
  /// not an independently chosen declaration.
  duckdb::vector<duckdb::LogicalType> types;
  duckdb::vector<sirius::logical_type> staging_types;
  std::string plan_dsl;
  std::size_t chunk_rows = kDefaultHplnChunkRows;
  std::size_t group_rows = 0;
};

/// Refuse at bind anything the writer cannot represent, so a COPY fails before it has run its
/// query rather than partway through writing a file.
void validate_writable_types(const std::vector<std::string>& names,
                             const duckdb::vector<duckdb::LogicalType>& types,
                             duckdb::vector<sirius::logical_type>& out_staging_types)
{
  for (std::size_t i = 0; i < types.size(); i++) {
    auto const lt = sirius::from_duckdb(types[i]);
    if (lt.id() == sirius::type_id::HUGEINT || lt.id() == sirius::type_id::UHUGEINT) {
      // cuDF has no 128-bit integer, and the 64-bit carrier Sirius substitutes elsewhere would
      // write values that do not read back.
      throw duckdb::BinderException(
        "COPY (FORMAT simpatico): column '%s' is %s, which has no cuDF carrier that round-trips",
        names[i],
        types[i].ToString());
    }
    if (!lt.is_varchar() && !lt.is_fixed_width()) {
      throw duckdb::BinderException(
        "COPY (FORMAT simpatico): column '%s' is %s; only fixed-width and VARCHAR columns can be "
        "written",
        names[i],
        types[i].ToString());
    }
    // Throws with its own diagnosis for a type with no cuDF mapping (e.g. DECIMAL(4,s)).
    static_cast<void>(sirius::get_cudf_type(lt));
    if (!lt.is_varchar()) { static_cast<void>(lt.fixed_width_byte_size()); }
    out_staging_types.push_back(lt);
  }
}

duckdb::unique_ptr<duckdb::FunctionData> simpatico_copy_bind(
  duckdb::ClientContext& context,
  duckdb::CopyFunctionBindInput& input,
  const duckdb::vector<std::string>& names,
  const duckdb::vector<duckdb::LogicalType>& sql_types)
{
  auto result = duckdb::make_uniq<simpatico_copy_bind_data>();
  result->names.assign(names.begin(), names.end());
  result->types = sql_types;
  if (sql_types.empty()) {
    throw duckdb::BinderException("COPY (FORMAT simpatico): a .hpln needs at least one column");
  }
  validate_writable_types(result->names, result->types, result->staging_types);

  // Zone-map granularity defaults to the same setting the pin path uses, so a file and a pin of
  // the same data prune at the same resolution. The Sirius state is absent when the extension is
  // loaded but the runtime was never brought up on this connection; the settings' own defaults
  // still describe what a pin would do, so a COPY works there rather than refusing.
  auto sirius_ctx    = context.registered_state->Get<duckdb::SiriusContext>("sirius_state");
  result->group_rows = sirius_ctx
                         ? sirius_ctx->get_config().get_operator_params().pinned_zone_map_group_rows
                         : sirius::operator_params{}.pinned_zone_map_group_rows;

  std::string plan_table;
  std::string explicit_plan;
  for (auto const& option : input.info.options) {
    auto const key = duckdb::StringUtil::Lower(option.first);
    if (key == "chunk_rows") {
      result->chunk_rows = single_positive_option(option.second, "chunk_rows");
    } else if (key == "group_rows") {
      // 0 is meaningful here (write no zone-map segment), so it does not go through the
      // positive-only parser.
      if (option.second.size() != 1 || option.second[0].IsNull()) {
        throw duckdb::BinderException(
          "COPY (FORMAT simpatico): 'group_rows' expects a single integer value");
      }
      auto const v = option.second[0]
                       .DefaultCastAs(duckdb::LogicalType(duckdb::LogicalTypeId::BIGINT))
                       .GetValue<std::int64_t>();
      if (v < 0) {
        throw duckdb::BinderException("COPY (FORMAT simpatico): 'group_rows' cannot be negative");
      }
      result->group_rows = static_cast<std::size_t>(v);
    } else if (key == "plan") {
      explicit_plan = single_string_option(option.second, "plan");
    } else if (key == "plan_table") {
      plan_table = single_string_option(option.second, "plan_table");
    } else {
      throw duckdb::BinderException("COPY (FORMAT simpatico): unrecognized option '%s'",
                                    option.first);
    }
  }
  if (!explicit_plan.empty() && !plan_table.empty()) {
    throw duckdb::BinderException(
      "COPY (FORMAT simpatico): set either 'plan' or 'plan_table', not both");
  }

  if (!explicit_plan.empty()) {
    result->plan_dsl = explicit_plan;
  } else if (!plan_table.empty()) {
    // The same registry and the same plan directory the pin path resolves a table's plan from:
    // a plan good enough to pin a table with is good enough to write it with.
    std::string const plan_dir =
      sirius_ctx ? sirius_ctx->get_config().get_compression_config().input_plan_dir : std::string{};
    std::string scan_error;
    auto resolved = resolve_table_plan_from_dir(plan_dir, plan_table, &scan_error);
    if (!resolved.has_value()) {
      throw duckdb::BinderException(
        "COPY (FORMAT simpatico): no compression plan for table '%s' in the registry or in "
        "pin_table_input_compression_plan_dir ('%s')%s",
        plan_table,
        plan_dir,
        scan_error.empty() ? "" : (": " + scan_error));
    }
    result->plan_dsl = std::move(*resolved);
  } else {
    result->plan_dsl = identity_plan_for(result->types.size());
  }

  auto const blocks = simpatico::split_plan_dsl(result->plan_dsl);
  if (blocks.size() != result->types.size()) {
    throw duckdb::BinderException(
      "COPY (FORMAT simpatico): the compression plan has %llu column blocks but the query "
      "produces %llu columns",
      static_cast<unsigned long long>(blocks.size()),
      static_cast<unsigned long long>(result->types.size()));
  }

  return std::move(result);
}

struct simpatico_copy_global_state : public duckdb::GlobalFunctionData {
  std::string path;
  std::mutex lock;
  std::unique_ptr<sirius::duckdb_chunk_staging> staging;
  /// One entry per closed chunk, device-resident until finalize hands them all to the writer.
  std::vector<std::unique_ptr<cudf::table>> chunks;
};

duckdb::unique_ptr<duckdb::GlobalFunctionData> simpatico_copy_initialize_global(
  duckdb::ClientContext& context, duckdb::FunctionData& bind_data, const std::string& file_path)
{
  auto& bind = bind_data.Cast<simpatico_copy_bind_data>();
  if (file_path.find("://") != std::string::npos) {
    // The container writer writes with a local stream. Reading a .hpln goes through the
    // io_context (milestone D) but writing one does not, and a path that looks remote must not
    // quietly become a local file of that name.
    throw duckdb::NotImplementedException(
      "COPY (FORMAT simpatico): only local paths can be written, got '%s'", file_path);
  }
  auto state     = duckdb::make_uniq<simpatico_copy_global_state>();
  state->path    = file_path;
  state->staging = std::make_unique<sirius::duckdb_chunk_staging>(
    bind.staging_types,
    static_cast<cudf::size_type>(std::min<std::size_t>(bind.chunk_rows, 1U << 20U)));
  return std::move(state);
}

struct simpatico_copy_local_state : public duckdb::LocalFunctionData {};

duckdb::unique_ptr<duckdb::LocalFunctionData> simpatico_copy_initialize_local(
  duckdb::ExecutionContext&, duckdb::FunctionData&)
{
  return duckdb::make_uniq_base<duckdb::LocalFunctionData, simpatico_copy_local_state>();
}

/// Upload what is staged as one chunk and start a fresh one.
void close_chunk(simpatico_copy_global_state& state)
{
  // Nulls need no special handling here: the staging carries each column's validity to the
  // cudf::table, and compress_column strips it into the plan tree's sidecar, which the container
  // serializes per column (see push_validity). A chunk that is entirely null costs no payload
  // bytes at all.
  state.chunks.push_back(
    state.staging->build(cudf::get_default_stream(), rmm::mr::get_current_device_resource_ref()));
  state.staging->reset();
}

void simpatico_copy_sink(duckdb::ExecutionContext&,
                         duckdb::FunctionData& bind_data,
                         duckdb::GlobalFunctionData& gstate,
                         duckdb::LocalFunctionData&,
                         duckdb::DataChunk& input)
{
  auto& bind  = bind_data.Cast<simpatico_copy_bind_data>();
  auto& state = gstate.Cast<simpatico_copy_global_state>();
  std::lock_guard<std::mutex> guard(state.lock);

  duckdb::idx_t offset = 0;
  while (offset < input.size()) {
    // Take only what fits in the open chunk, so a chunk holds exactly chunk_rows rows rather than
    // however many DuckDB's vector size happens to overshoot by. Chunk boundaries decide what a
    // scan can drop, and a zone-map group that straddles one would be bounded across both.
    auto const room = bind.chunk_rows - static_cast<std::size_t>(state.staging->num_rows());
    auto const take = std::min<duckdb::idx_t>(input.size() - offset, room);
    state.staging->append(input, offset, take);
    offset += take;
    if (static_cast<std::size_t>(state.staging->num_rows()) >= bind.chunk_rows) {
      close_chunk(state);
    }
  }
}

void simpatico_copy_combine(duckdb::ExecutionContext&,
                            duckdb::FunctionData&,
                            duckdb::GlobalFunctionData&,
                            duckdb::LocalFunctionData&)
{
}

void simpatico_copy_finalize(duckdb::ClientContext&,
                             duckdb::FunctionData& bind_data,
                             duckdb::GlobalFunctionData& gstate)
{
  auto& bind  = bind_data.Cast<simpatico_copy_bind_data>();
  auto& state = gstate.Cast<simpatico_copy_global_state>();
  std::lock_guard<std::mutex> guard(state.lock);

  // A trailing partial chunk is a chunk; and a query that produced no rows still writes one empty
  // chunk, so the file carries its schema and binds like any other.
  if (state.staging->num_rows() > 0 || state.chunks.empty()) { close_chunk(state); }

  std::vector<cudf::table_view> views;
  views.reserve(state.chunks.size());
  for (auto const& chunk : state.chunks) {
    views.push_back(chunk->view());
  }

  auto const error = sirius::write_tables_to_hpln(views,
                                                  bind.types,
                                                  bind.names,
                                                  bind.plan_dsl,
                                                  bind.group_rows,
                                                  state.path,
                                                  cudf::get_default_stream(),
                                                  rmm::mr::get_current_device_resource_ref());
  // Free the device tables before reporting either way: a failed COPY must not leave a file's
  // worth of GPU memory held by a state that outlives the error.
  state.chunks.clear();
  state.staging.reset();
  if (!error.empty()) {
    throw duckdb::IOException("COPY (FORMAT simpatico) to '%s' failed: %s", state.path, error);
  }
  SIRIUS_LOG_INFO("[hpln copy] wrote '{}': {} chunk(s), group_rows={}",
                  state.path,
                  views.size(),
                  bind.group_rows);
}

}  // namespace

duckdb::CopyFunction make_simpatico_copy_function()
{
  duckdb::CopyFunction function("simpatico");
  function.copy_to_bind              = simpatico_copy_bind;
  function.copy_to_initialize_local  = simpatico_copy_initialize_local;
  function.copy_to_initialize_global = simpatico_copy_initialize_global;
  function.copy_to_sink              = simpatico_copy_sink;
  function.copy_to_combine           = simpatico_copy_combine;
  function.copy_to_finalize          = simpatico_copy_finalize;
  // Lets `COPY ... TO 'x.hpln'` pick this function without a FORMAT clause, the way .parquet does.
  function.extension = "hpln";
  // No execution_mode override: the default is a single unordered-insensitive writer, which is
  // what keeps the file's row order equal to the query's and therefore keeps a clustered SELECT
  // clustered in the file -- which is the whole reason to write one.
  return function;
}

}  // namespace sirius::compression
