// Copyright 2026, Sirius Contributors. SPDX-License-Identifier: Apache-2.0
#include <sirius/c/context/config_builder.h>
#include <sirius/context/config_builder.hpp>

#include <atomic>
#include <cstdlib>
#include <new>

namespace {
std::atomic<bool> fail_allocation{false};
std::atomic<int> remaining_allocations{0};
}  // namespace
void* operator new(std::size_t size)
{
  if (fail_allocation.load() && remaining_allocations.fetch_sub(1) <= 0) { throw std::bad_alloc{}; }
  if (auto* result = std::malloc(size ? size : 1)) { return result; }
  throw std::bad_alloc{};
}
void operator delete(void* value) noexcept { std::free(value); }
void operator delete(void* value, std::size_t) noexcept { std::free(value); }

int main()
{
  sirius_config_builder* builder = nullptr;
  sirius_error* error            = nullptr;
  fail_allocation                = true;
  auto status                    = sirius_config_builder_create(&builder, &error);
  if (status != SIRIUS_ALLOCATION_FAILURE || builder || error) { return 1; }
  status = sirius_config_builder_create(nullptr, &error);
  if (status != SIRIUS_INVALID_ARGUMENT || error) { return 2; }
  fail_allocation = false;
  // Warm stream state before injecting failure into a loader diagnostic.
  status = sirius_config_builder_from_yaml("", 0, &builder, &error);
  if (status != SIRIUS_CONFIGURATION_IO) { return 5; }
  sirius_error_destroy(error);
  // Permit handle storage; the empty path needs no path-string allocation.
  remaining_allocations = 1;
  fail_allocation       = true;
  status                = sirius_config_builder_from_yaml("", 0, &builder, &error);
  if (status != SIRIUS_CONFIGURATION_IO || builder || error) { return 6; }
  fail_allocation = false;
  if (sirius_config_builder_create(&builder, nullptr) != SIRIUS_SUCCESS) { return 3; }
  fail_allocation = true;
  sirius_config_builder_retain(builder);
  sirius_config_builder_release(builder);
  sirius_config* config = nullptr;
  status                = sirius_config_builder_build(builder, &config, &error);
  sirius_config_builder_release(builder);
  if (status != SIRIUS_ALLOCATION_FAILURE || config || error) { return 4; }
  fail_allocation = false;
  sirius::ContextConfigBuilder cpp_builder;
  auto cpp_config = cpp_builder.build();
  if (!cpp_config) { return 7; }
  const std::filesystem::path empty_path;
  fail_allocation    = true;
  auto copy          = cpp_builder;
  auto snapshot_copy = *cpp_config;
  copy               = cpp_builder;
  snapshot_copy      = *cpp_config;
  auto failed_build  = cpp_builder.build();
  if (failed_build || failed_build.error().code != sirius::ErrorCode::allocation_failure) {
    return 8;
  }
  auto failed_load = sirius::ContextConfigBuilder::from_yaml(empty_path);
  if (failed_load || failed_load.error().code != sirius::ErrorCode::allocation_failure) {
    return 9;
  }
  return 0;
}
