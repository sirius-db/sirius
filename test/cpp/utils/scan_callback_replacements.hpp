// Copyright 2026, Sirius Contributors. SPDX-License-Identifier: Apache-2.0
#pragma once

#include <catch.hpp>
#include <duckdb/function/table_function.hpp>

#include <string_view>

namespace sirius::test {
inline void fake_scan(duckdb::ClientContext&, duckdb::TableFunctionInput&, duckdb::DataChunk&) {}

inline duckdb::unique_ptr<duckdb::GlobalTableFunctionState> fake_global(
  duckdb::ClientContext&, duckdb::TableFunctionInitInput&)
{
  throw duckdb::InvalidInputException("replacement global initializer");
}
inline duckdb::unique_ptr<duckdb::LocalTableFunctionState> fake_local(
  duckdb::ExecutionContext&, duckdb::TableFunctionInitInput&, duckdb::GlobalTableFunctionState*)
{
  throw duckdb::InvalidInputException("replacement local initializer");
}
inline void fake_serialize(duckdb::Serializer&,
                           duckdb::optional_ptr<duckdb::FunctionData>,
                           duckdb::BoundTableFunction const&)
{
  throw duckdb::NotImplementedException("replacement serializer");
}
inline duckdb::unique_ptr<duckdb::FunctionData> fake_deserialize(duckdb::Deserializer&,
                                                                 duckdb::BoundTableFunction&)
{
  throw duckdb::NotImplementedException("replacement deserializer");
}

inline void replace_callback(duckdb::TableFunction& function, std::string_view phase)
{
  if (phase.ends_with("init_global"))
    function.init_global = fake_global;
  else if (phase.ends_with("init_local"))
    function.init_local = fake_local;
  else if (phase.ends_with("_deserialize"))
    function.deserialize = fake_deserialize;
  else if (phase.ends_with("_serialize"))
    function.serialize = fake_serialize;
  else
    function.function = fake_scan;
}

inline void require_registered_callbacks(duckdb::BaseTableFunction const& actual,
                                         duckdb::BaseTableFunction const& expected)
{
  REQUIRE(actual.function == expected.function);
  REQUIRE(actual.init_global == expected.init_global);
  REQUIRE(actual.init_local == expected.init_local);
  REQUIRE(actual.serialize == expected.serialize);
  REQUIRE(actual.deserialize == expected.deserialize);
}

}  // namespace sirius::test
