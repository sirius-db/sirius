// Copyright 2026, Sirius Contributors. SPDX-License-Identifier: Apache-2.0
#include <duckdb/function/scalar_function.hpp>
#include <duckdb/planner/expression/bound_function_expression.hpp>
#include <sirius/context/config_builder.hpp>
#include <sirius/duckdb.hpp>
#include <sirius/ffi.hpp>
#include <sirius/udf.hpp>

// Keep symbol references without initializing a GPU during the link smoke test.
auto* volatile context_factory  = &sirius::ffi::make_context;
auto* volatile registration     = &sirius::register_duckdb_extension;
auto* volatile udf_registration = &sirius::register_gpu_scalar_function;

int main()
{
  sirius::ContextConfigBuilder builder;
  auto copy                   = builder;
  builder                     = copy;
  auto* volatile yaml_factory = &sirius::ContextConfigBuilder::from_yaml;
  auto volatile build         = &sirius::ContextConfigBuilder::build;
  return context_factory == nullptr || registration == nullptr || yaml_factory == nullptr ||
         build == nullptr || udf_registration == nullptr;
}
