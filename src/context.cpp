// Copyright 2026, Sirius Contributors. SPDX-License-Identifier: Apache-2.0
#include "context_config_internal.hpp"
#include "context_runtime.hpp"

#include <sirius/context/context.hpp>

#include <utility>

namespace sirius {

struct Context::Impl {
  explicit Impl(const parsed_sirius_config& config) : runtime(config) {}
  context_runtime runtime;
};

Context::Context(std::unique_ptr<Impl> impl) noexcept : impl_(std::move(impl)) {}
Context::~Context() noexcept = default;

std::expected<std::unique_ptr<Context>, Error> Context::create(const ContextConfig& config) noexcept
try {
  try {
    // A new-expression allocates the handle before evaluating its initializer.
    return std::unique_ptr<Context>(new Context(std::make_unique<Impl>(config.impl_->config)));
  } catch (const std::bad_alloc&) {
    throw;
  } catch (const std::exception& e) {
    return std::unexpected(Error{ErrorCode::context_initialization, e.what()});
  } catch (...) {
    return std::unexpected(
      Error{ErrorCode::context_initialization, "Unknown failure while initializing Sirius"});
  }
} catch (const std::bad_alloc&) {
  // Also covers allocation while constructing another error's diagnostic.
  return std::unexpected(Error{ErrorCode::allocation_failure, {}});
}

}  // namespace sirius
