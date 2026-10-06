// Copyright 2026, Sirius Contributors. SPDX-License-Identifier: Apache-2.0
#include <sirius/context/config_builder.hpp>
#include <sirius/duckdb.hpp>
#include <sirius/ffi.hpp>

// Keep symbol references without initializing a GPU during the link smoke test.
auto* volatile context_factory = &sirius::ffi::make_context;
auto* volatile registration    = &sirius::register_duckdb_extension;

int main()
{
  sirius::ContextConfigBuilder builder;
  auto copy = builder;
  copy.gpu_usage_limit_bytes(8ULL << 30).gpu_usage_limit_fraction(0.5);
  auto* volatile yaml_factory = &sirius::ContextConfigBuilder::from_yaml;
  auto volatile build         = &sirius::ContextConfigBuilder::build;
  return context_factory == nullptr || registration == nullptr || yaml_factory == nullptr ||
         build == nullptr;
}
