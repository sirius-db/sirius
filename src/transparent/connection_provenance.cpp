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

#include "transparent/connection_provenance.hpp"

#include "sirius_context.hpp"

#include <duckdb/catalog/catalog.hpp>
#include <duckdb/catalog/catalog_entry/table_catalog_entry.hpp>
#include <duckdb/catalog/catalog_search_path.hpp>
#include <duckdb/common/constants.hpp>
#include <duckdb/common/enums/logical_operator_type.hpp>
#include <duckdb/common/enums/statement_type.hpp>
#include <duckdb/common/optional_ptr.hpp>
#include <duckdb/common/shared_ptr.hpp>
#include <duckdb/common/string_util.hpp>
#include <duckdb/common/types/value.hpp>
#include <duckdb/main/attached_database.hpp>
#include <duckdb/main/client_context.hpp>
#include <duckdb/main/client_data.hpp>
#include <duckdb/main/database_manager.hpp>
#include <duckdb/planner/logical_operator.hpp>
#include <duckdb/planner/operator/logical_get.hpp>
#include <duckdb/transaction/meta_transaction.hpp>
#include <log/logging.hpp>
#include <util/duckdb_error_message.hpp>

#include <cstdint>
#include <exception>
#include <stdexcept>
#include <string>

namespace sirius::transparent {

namespace {

/// DuckLake metadata connections set this to 0 before querying their hidden catalog.
constexpr char const* PROVIDER_FINGERPRINT_SETTING = "catalog_error_max_schemas";

/// Test-only setting, registered under SIRIUS_ENABLE_TEST_OPTIONS=1.
constexpr char const* INJECT_FAILURE_SETTING = "sirius_test_inject_provenance_failure";

bool bool_setting_is_true(duckdb::ClientContext& context, char const* name)
{
  duckdb::Value value;
  return context.TryGetCurrentSetting(name, value) && !value.IsNull() && value.GetValue<bool>();
}

bool setting_is_zero(duckdb::ClientContext& context, char const* name)
{
  duckdb::Value value;
  return context.TryGetCurrentSetting(name, value) && !value.IsNull() &&
         value.GetValue<uint64_t>() == 0;
}

/// Transaction results borrow its owning reference; temp and global results retain their own.
struct resolved_database {
  duckdb::optional_ptr<duckdb::AttachedDatabase> database;
  duckdb::shared_ptr<duckdb::AttachedDatabase> keep_alive;
};

/// Match binder lookup without adding transaction references. Transaction name keys survive
/// alias changes and take precedence over the global map, including after another connection
/// detaches.
resolved_database resolve_attached_database(duckdb::ClientContext& context, std::string const& name)
{
  if (duckdb::StringUtil::CIEquals(name, TEMP_CATALOG)) {
    auto temp = duckdb::ClientData::Get(context).temporary_objects;
    return {temp.get(), temp};
  }
  if (context.transaction.HasActiveTransaction()) {
    auto referenced = duckdb::MetaTransaction::Get(context).GetReferencedDatabase(name);
    if (referenced) { return {referenced, nullptr}; }
  }
  auto global = duckdb::DatabaseManager::Get(context).GetDatabase(name);
  return {global.get(), global};
}

bool attached_database_is_hidden(duckdb::ClientContext& context, std::string const& name)
{
  if (name.empty()) { return false; }
  auto resolved = resolve_attached_database(context, name);
  return resolved.database &&
         resolved.database->GetVisibility() == duckdb::AttachVisibility::HIDDEN;
}

bool registered_catalog_is_hidden(duckdb::ClientContext& context,
                                  std::string const& name,
                                  duckdb::StatementProperties::CatalogIdentity const& identity,
                                  bool& hidden)
{
  auto resolved = resolve_attached_database(context, name);
  if (!resolved.database || resolved.database->oid != identity.catalog_oid) { return false; }
  hidden = resolved.database->GetVisibility() == duckdb::AttachVisibility::HIDDEN;
  return true;
}

bool default_catalog_is_hidden(duckdb::ClientContext& context, std::string& catalog_name)
{
  auto const& entry = duckdb::ClientData::Get(context).catalog_search_path->GetDefault();
  catalog_name      = entry.catalog;
  return attached_database_is_hidden(context, catalog_name);
}

/// Logging failures must not escape noexcept classification.
template <typename Log>
void log_best_effort(Log&& log) noexcept
{
  try {
    log();
  } catch (...) {
  }
}

void walk_for_hidden_catalog(duckdb::LogicalOperator const& op, bool& hidden, bool& failed)
{
  if (op.type == duckdb::LogicalOperatorType::LOGICAL_GET) {
    try {
      auto table = op.Cast<duckdb::LogicalGet>().GetTable();
      if (table && table->ParentCatalog().GetAttached().GetVisibility() ==
                     duckdb::AttachVisibility::HIDDEN) {
        hidden = true;
      }
    } catch (...) {
      failed = true;
    }
  }
  for (auto const& child : op.children) {
    walk_for_hidden_catalog(*child, hidden, failed);
  }
}

}  // namespace

classification_result classify_connection_provenance(duckdb::ClientContext& context,
                                                     duckdb::SiriusConnectionState& state) noexcept
{
  if (state.provenance() == connection_provenance::provider_internal) {
    return {connection_provenance::provider_internal, decline_reason::provider_internal};
  }
  try {
    if (bool_setting_is_true(context, INJECT_FAILURE_SETTING)) {
      throw std::runtime_error("injected provenance classification failure");
    }
    std::string catalog_name;
    if (default_catalog_is_hidden(context, catalog_name) &&
        setting_is_zero(context, PROVIDER_FINGERPRINT_SETTING)) {
      state.latch_provider_internal();
      log_best_effort([&] {
        SIRIUS_LOG_INFO("connection {}: provider-internal, default catalog '{}' is hidden",
                        state.connection_id(),
                        catalog_name);
      });
      return {connection_provenance::provider_internal, decline_reason::provider_internal};
    }
    state.mark_user_provenance();
    return {connection_provenance::user, decline_reason::none};
  } catch (std::exception& e) {
    log_best_effort([&] {
      SIRIUS_LOG_DEBUG("connection {}: provenance classification failed: {}",
                       state.connection_id(),
                       sirius::sanitized_message(e));
    });
    return {state.provenance(), decline_reason::classification_failed};
  } catch (...) {
    return {state.provenance(), decline_reason::classification_failed};
  }
}

decline_reason inspect_statement_for_hidden_catalog(
  duckdb::ClientContext& context, duckdb::StatementProperties const& properties) noexcept
{
  bool hidden  = false;
  bool failed  = false;
  auto inspect = [&](std::string const& name,
                     duckdb::StatementProperties::CatalogIdentity const& identity) {
    bool entry_hidden = false;
    if (!registered_catalog_is_hidden(context, name, identity, entry_hidden)) {
      failed = true;
      return;
    }
    hidden = hidden || entry_hidden;
  };
  try {
    for (auto const& entry : properties.read_databases) {
      inspect(entry.first, entry.second);
    }
    for (auto const& entry : properties.modified_databases) {
      inspect(entry.first, entry.second.identity);
    }
  } catch (...) {
    failed = true;
  }
  if (hidden) { return decline_reason::hidden_catalog; }
  if (failed) { return decline_reason::classification_failed; }
  return decline_reason::none;
}

decline_reason inspect_plan_for_hidden_catalog(duckdb::LogicalOperator const& plan) noexcept
{
  bool hidden = false;
  bool failed = false;
  try {
    walk_for_hidden_catalog(plan, hidden, failed);
  } catch (...) {
    failed = true;
  }
  if (hidden) { return decline_reason::hidden_catalog; }
  if (failed) { return decline_reason::classification_failed; }
  return decline_reason::none;
}

decline_reason screen_planning_attempt(duckdb::ClientContext& context,
                                       duckdb::SiriusContext& sirius,
                                       duckdb::SiriusConnectionState& state,
                                       duckdb::LogicalOperator const* plan,
                                       duckdb::StatementProperties const* properties) noexcept
{
  if (state.attempt_declined()) { return state.attempt_decline_reason(); }
  auto reason = decline_reason::none;
  if (!state.classified_this_attempt()) {
    reason = classify_connection_provenance(context, state).reason;
    state.mark_classified();
  }
  if (reason == decline_reason::none && properties) {
    reason = inspect_statement_for_hidden_catalog(context, *properties);
  }
  if (reason == decline_reason::none && plan) { reason = inspect_plan_for_hidden_catalog(*plan); }
  if (reason == decline_reason::none) { return reason; }
  state.decline_attempt(reason);
  sirius.record_transparent_decline(reason);
  return reason;
}

}  // namespace sirius::transparent
