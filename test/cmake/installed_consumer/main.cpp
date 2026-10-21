// Copyright 2026, Sirius Contributors. SPDX-License-Identifier: Apache-2.0
#include <sirius/context/config_builder.hpp>
#include <sirius/duckdb.hpp>
#include <sirius/ffi.hpp>
#include <sirius/log/duckdb_sink.hpp>
#include <sirius/log/level.hpp>
#include <sirius/log/noop_sink.hpp>
#include <sirius/log/sink.hpp>
#include <sirius/log/spdlog_owning_sink.hpp>

// Keep symbol references without initializing a GPU during the link smoke test.
auto* volatile context_factory = &sirius::ffi::make_context;
auto* volatile registration    = &sirius::register_duckdb_extension;

int main()
{
  sirius::ContextConfigBuilder builder;
  auto copy                   = builder;
  builder                     = copy;
  auto* volatile yaml_factory = &sirius::ContextConfigBuilder::from_yaml;
  auto volatile build         = &sirius::ContextConfigBuilder::build;
  return context_factory == nullptr || registration == nullptr || yaml_factory == nullptr ||
         build == nullptr;
}

auto* volatile log_sink_reader    = &sirius::log::get_sink;
auto* volatile log_sink_writer    = &sirius::log::set_sink;
auto* volatile log_level_parser   = &sirius::log::string_to_enum;
auto* volatile file_log_factory   = &sirius::log::make_spdlog_owning_sink;
auto* volatile noop_log_factory   = &sirius::log::make_noop_sink;
auto* volatile duckdb_log_factory = &sirius::log::make_duckdb_sink;
