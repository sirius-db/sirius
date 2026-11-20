// Copyright 2026, Sirius Contributors. SPDX-License-Identifier: Apache-2.0
#include <sirius/context/config_builder.hpp>

#include <type_traits>

static_assert(std::is_nothrow_copy_constructible_v<sirius::ContextConfig>);
static_assert(std::is_nothrow_copy_constructible_v<sirius::ContextConfigBuilder>);
int main()
{
  auto config = [] {
    sirius::ContextConfigBuilder builder;
    auto copy = builder;
    return copy.build();
  }();
  if (!config) { return 1; }
  auto copy    = *config;
  copy         = *config;
  auto missing = sirius::ContextConfigBuilder::from_yaml("");
  if (missing || missing.error().code != sirius::ErrorCode::configuration_io) { return 2; }
  auto nul = sirius::ContextConfigBuilder::from_yaml(std::string("a\0b", 3));
  if (nul || nul.error().code != sirius::ErrorCode::invalid_argument) { return 3; }
  return 0;
}
