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

#include "transaction_utils.hpp"

#include <duckdb/catalog/catalog.hpp>
#include <duckdb/main/database_manager.hpp>
#include <duckdb/transaction/duck_transaction.hpp>

namespace sirius::helper {

duckdb::transaction_t get_query_start_time(duckdb::ClientContext& context)
{
  auto const& name = duckdb::DatabaseManager::Get(context).GetDefaultDatabase(context);
  auto& catalog    = duckdb::Catalog::GetCatalog(context, name);
  return duckdb::DuckTransaction::Get(context, catalog).start_time;
}

duckdb::shared_ptr<duckdb::AttachedDatabase> find_global_database(duckdb::ClientContext& context,
                                                                  duckdb::Identifier const& name)
{
  return duckdb::DatabaseManager::Get(context).GetDatabase(name);
}

}  // namespace sirius::helper
