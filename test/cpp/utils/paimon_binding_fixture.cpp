// Copyright 2026, Sirius Contributors. SPDX-License-Identifier: Apache-2.0
// Adversarial descriptor ABI fixture. This is not a Paimon reader or native oracle.
#include "utils/paimon_binding_fixture.hpp"

#include "planner/paimon_read_bridge.hpp"

#include <duckdb/main/extension/extension_loader.hpp>

#include <array>
#include <atomic>
#include <cstdlib>
#include <string>

namespace {
using namespace sirius::test::paimon_fixture;

int selected_mode()
{
  auto const* value = std::getenv("SIRIUS_PAIMON_BINDING_FIXTURE_MODE");
  return value ? std::stoi(value) : valid;
}

struct bind_data final : duckdb::TableFunctionData {
  int value = selected_mode();
};
struct descriptor {
  int value;
};
std::atomic<uint64_t> released{0};

int32_t accepts(void const* data)
{
  return data && dynamic_cast<bind_data const*>(static_cast<duckdb::FunctionData const*>(data));
}
int32_t retain(void const* data, void** out, char*, uint64_t)
{
  *out             = nullptr;
  auto const value = static_cast<bind_data const*>(data)->value;
  if (value == null_output) return PAIMON_READ_OK;
  *out = new descriptor{value};
  return value == partial_error ? PAIMON_READ_INTERNAL_ERROR : PAIMON_READ_OK;
}
int32_t describe(void const* data, PaimonBoundReadViewV1* out, uint64_t)
{
  auto const value = static_cast<descriptor const*>(data)->value;
  if (value == describe_error) return PAIMON_READ_INTERNAL_ERROR;
  *out              = {};
  out->warehouse    = {"warehouse", 9};
  out->database     = {"database", 8};
  out->table        = {"table", 5};
  out->location     = {value == invalid_view ? nullptr : "/fixture", 8};
  out->branch       = {"main", 4};
  out->schema_json  = {"{}", 2};
  out->snapshot_id  = 5;
  out->has_snapshot = 1;
  return PAIMON_READ_OK;
}
void release(void* data)
{
  ++released;
  delete static_cast<descriptor*>(data);
}

auto const apis = [] {
  std::array<PaimonReadBridgeV1, mode_count> result;
  result.fill({PAIMON_READ_BRIDGE_MAGIC,
               PAIMON_READ_BRIDGE_VERSION,
               sizeof(PaimonReadBridgeV1),
               accepts,
               retain,
               describe,
               release});
  result[bad_magic].magic              = 0;
  result[bad_version].version          = 99;
  result[bad_size].struct_size         = sizeof(PaimonReadBridgeV1) - 1;
  result[missing_callback].retain      = nullptr;
  result[missing_accepts].accepts_bind = nullptr;
  result[missing_describe].describe    = nullptr;
  result[missing_release].release      = nullptr;
  result[foreign_callback].retain = reinterpret_cast<decltype(PaimonReadBridgeV1::retain)>(&malloc);
  return result;
}();

auto const v2 = [] {
  PaimonReadBridgeV2 result{};
  result.descriptor             = apis[valid];
  result.descriptor.version     = PAIMON_READ_BRIDGE_NATIVE_VERSION;
  result.descriptor.struct_size = sizeof(result);
  return result;
}();

auto const infos = [] {
  std::array<duckdb::shared_ptr<duckdb::TableFunctionInfo>, mode_count> result;
  for (std::size_t i = 0; i < result.size(); ++i)
    result[i] = duckdb::make_shared_ptr<duckdb::PaimonReadBridgeInfo>(apis[i]);
  result[valid_v2] = duckdb::make_shared_ptr<duckdb::PaimonReadBridgeInfo>(v2.descriptor);
  return result;
}();

duckdb::unique_ptr<duckdb::FunctionData> bind(duckdb::ClientContext&,
                                              duckdb::TableFunctionBindInput&,
                                              duckdb::vector<duckdb::LogicalType>& types,
                                              duckdb::vector<duckdb::Identifier>& names)
{
  types = {duckdb::LogicalType::INTEGER};
  names = {"id"};
  return duckdb::make_uniq<bind_data>();
}
duckdb::unique_ptr<duckdb::GlobalTableFunctionState> init_global(duckdb::ClientContext&,
                                                                 duckdb::TableFunctionInitInput&)
{
  return duckdb::make_uniq<duckdb::GlobalTableFunctionState>();
}
duckdb::unique_ptr<duckdb::LocalTableFunctionState> init_local(duckdb::ExecutionContext&,
                                                               duckdb::TableFunctionInitInput&,
                                                               duckdb::GlobalTableFunctionState*)
{
  return duckdb::make_uniq<duckdb::LocalTableFunctionState>();
}
void scan(duckdb::ClientContext&, duckdb::TableFunctionInput&, duckdb::DataChunk&) {}
}  // namespace

extern "C" {
DUCKDB_EXTENSION_API uint64_t sirius_paimon_fixture_released() { return released.load(); }
DUCKDB_CPP_EXTENSION_ENTRY(paimon, loader)
{
  duckdb::TableFunction function(
    "paimon_scan", {duckdb::LogicalType::VARCHAR}, scan, bind, init_global, init_local);
  function.function_info = infos.at(selected_mode());
  loader.RegisterFunction(function);
}
}
