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

#include "udf/registration.hpp"

#include <duckdb/catalog/catalog.hpp>
#include <duckdb/catalog/catalog_entry/scalar_function_catalog_entry.hpp>
#include <duckdb/catalog/catalog_entry/schema_catalog_entry.hpp>
#include <duckdb/catalog/catalog_transaction.hpp>
#include <duckdb/common/exception.hpp>
#include <duckdb/function/scalar_function.hpp>
#include <duckdb/main/extension/extension_loader.hpp>
#include <duckdb/parser/parsed_data/create_scalar_function_info.hpp>
#include <duckdb/planner/expression/bound_function_expression.hpp>

#include <utility>

namespace sirius {
namespace {

struct gpu_function_info final : duckdb::ScalarFunctionInfo {
  explicit gpu_function_info(gpu_scalar_binder binder) : bind(std::move(binder)) {}
  gpu_scalar_binder bind;
};

/// The original CPU callback does not require bind data. Adding this identity
/// makes DuckDB expression equality distinguish registrations even when their
/// names/signatures and CPU trampolines coincide (e.g. in different schemas).
struct gpu_function_bind_data final : duckdb::FunctionData {
  explicit gpu_function_bind_data(duckdb::shared_ptr<duckdb::ScalarFunctionInfo> info)
    : identity(std::move(info))
  {
  }

  duckdb::unique_ptr<duckdb::FunctionData> Copy() const override
  {
    return duckdb::make_uniq<gpu_function_bind_data>(identity);
  }

  bool Equals(duckdb::FunctionData const& other) const override
  {
    auto const* data = dynamic_cast<gpu_function_bind_data const*>(&other);
    return data && identity.get() == data->identity.get();
  }

  duckdb::shared_ptr<duckdb::ScalarFunctionInfo> identity;
};

duckdb::unique_ptr<duckdb::FunctionData> bind_cpu_identity(duckdb::BindScalarFunctionInput& input)
{
  return duckdb::make_uniq<gpu_function_bind_data>(input.GetBoundFunction().GetFunctionInfo());
}

bool supported_type(duckdb::LogicalType const& type)
{
  using id = duckdb::LogicalTypeId;
  switch (type.id()) {
    case id::BOOLEAN:
    case id::TINYINT:
    case id::SMALLINT:
    case id::INTEGER:
    case id::BIGINT:
    case id::UTINYINT:
    case id::USMALLINT:
    case id::UINTEGER:
    case id::UBIGINT:
    case id::FLOAT:
    case id::DOUBLE:
    case id::VARCHAR:
    case id::DATE:
    case id::TIMESTAMP_SEC:
    case id::TIMESTAMP_MS:
    case id::TIMESTAMP:
    case id::TIMESTAMP_NS: return true;
    default: return false;
  }
}

void validate_registration(duckdb::ScalarFunction const& function, gpu_scalar_binder const& binder)
{
  if (function.GetName().empty() || !function.HasFunctionCallback() || !binder) {
    throw duckdb::InvalidInputException(
      "GPU scalar registration requires a name and CPU/GPU callbacks");
  }
  if (!function.GetCatalogName().empty() &&
      function.GetCatalogName() != duckdb::Identifier::SystemCatalog()) {
    throw duckdb::InvalidInputException(
      "GPU scalar functions must be registered in the system catalog");
  }
  if (function.GetStability() != duckdb::FunctionStability::CONSISTENT ||
      function.RequiresOrderedExecution() || function.HasModifiedDatabasesCallback()) {
    throw duckdb::InvalidInputException(
      "GPU scalar functions must be consistent, read-only and order independent");
  }
  if (function.HasBindCallback() || function.HasResolveTypesCallback() ||
      function.HasBindExpressionCallback() || function.HasExtraFunctionInfo() ||
      function.GetSerializeCallback() || function.GetDeserializeCallback()) {
    throw duckdb::InvalidInputException(
      "GPU scalar registration does not support existing bind/resolve/serialization callbacks or "
      "metadata");
  }
  auto const& signature = function.GetSignature();
  if (signature.GetParameterCount() == 0 || !supported_type(function.GetReturnType())) {
    throw duckdb::InvalidInputException(
      "GPU scalar functions require a nonempty fixed scalar signature");
  }
  for (auto const& parameter : signature.GetParameters()) {
    if (parameter.IsVariadic() || !supported_type(parameter.GetType())) {
      throw duckdb::InvalidInputException(
        "GPU scalar functions require fixed supported argument types");
    }
  }
}

}  // namespace

void register_gpu_scalar_function(duckdb::ExtensionLoader& loader,
                                  duckdb::ScalarFunction cpu_function,
                                  gpu_scalar_binder binder)
{
  validate_registration(cpu_function, binder);
  cpu_function.SetExtraFunctionInfo(duckdb::make_shared_ptr<gpu_function_info>(std::move(binder)));
  cpu_function.SetBindCallback(bind_cpu_identity);

  // The loader's lookup helpers always inspect main, even if its registration
  // schema was overridden. Resolve the CPU definition's explicit schema here
  // so overload identity and registration always use the same catalog entry.
  auto& database         = loader.GetDatabaseInstance();
  auto& catalog          = duckdb::Catalog::GetSystemCatalog(database);
  auto transaction       = duckdb::CatalogTransaction::GetSystemTransaction(database);
  auto const schema_name = cpu_function.GetSchemaName().empty()
                             ? duckdb::Identifier::DefaultSchema()
                             : cpu_function.GetSchemaName();
  auto& schema           = catalog.GetSchema(transaction, schema_name);
  auto const entry       = schema.GetEntry(
    transaction, duckdb::CatalogType::SCALAR_FUNCTION_ENTRY, cpu_function.GetName());
  if (entry) {
    for (auto const& existing :
         entry->Cast<duckdb::ScalarFunctionCatalogEntry>().functions.functions) {
      if (existing->GetSignature().IsSameOverload(cpu_function.GetSignature())) {
        throw duckdb::InvalidInputException("GPU scalar function overload is already registered");
      }
    }
  }
  // The regular catalog alter path copies the overload set, keeping definitions
  // held by already-bound CPU/GPU plans alive. Its merge permits replacement,
  // so the exact-signature duplicate check above is deliberate.
  auto const name = cpu_function.GetName();
  duckdb::CreateScalarFunctionInfo info(std::move(cpu_function));
  info.SetQualifiedName(
    duckdb::QualifiedName(duckdb::Identifier::SystemCatalog(), schema_name, name));
  info.extension_name = loader.GetRegisteredExtensionName();
  info.on_conflict    = entry ? duckdb::OnCreateConflict::ALTER_ON_CONFLICT
                              : duckdb::OnCreateConflict::ERROR_ON_CONFLICT;
  catalog.CreateFunction(transaction, info);
}

namespace udf {

gpu_scalar_binding bind_gpu_scalar_function(duckdb::BoundFunctionExpression const& expression)
{
  auto const& function = expression.Function();
  auto const info      = function.GetFunctionInfo();
  auto const* gpu_info = dynamic_cast<gpu_function_info const*>(info.get());
  if (!gpu_info) { return {}; }

  gpu_scalar_binding result{true, nullptr};
  auto const* data = dynamic_cast<gpu_function_bind_data const*>(expression.BindInfo().get());
  if (!data || data->identity.get() != info.get() ||
      function.GetStability() != duckdb::FunctionStability::CONSISTENT ||
      function.RequiresOrderedExecution() || function.HasModifiedDatabasesCallback() ||
      function.GetBindCallback() != bind_cpu_identity) {
    return result;
  }
  result.implementation = gpu_info->bind(expression);
  if (result.implementation && !result.implementation->execute) {
    throw duckdb::InvalidInputException("GPU scalar binder returned an empty execution callback");
  }
  return result;
}

}  // namespace udf
}  // namespace sirius
