// Copyright 2026, Sirius Contributors. SPDX-License-Identifier: Apache-2.0
/**
 * @file
 * @brief Construct and own an initialized Sirius engine instance.
 * Requires C++23 and standard-library support for std::expected.
 */
#pragma once

#include <sirius/context/config.hpp>
#include <sirius/error.hpp>
#include <sirius/export.hpp>

#include <expected>
#include <memory>

namespace sirius {

/**
 * @brief Own the resources of an initialized Sirius engine.
 *
 * Construction resolves configuration against available hardware and starts the
 * engine. Destruction releases its resources. There is no partially initialized
 * public state and no explicit initialization or shutdown operation.
 *
 * Only one active Sirius engine context is supported per process. A second
 * creation attempt fails while another context is initializing, active, or
 * shutting down. Contexts managed by other Sirius integrations share this limit.
 * Multiple configurations may exist independently of the active context. If
 * teardown fails, the runtime remains reserved until the process exits.
 * Forking a process with an active context is unsupported.
 *
 * Context has unique ownership: move its std::unique_ptr handle to transfer
 * ownership. The configuration supplied to create() need not outlive the context.
 * Do not destroy the context concurrently with its use.
 */
class SIRIUS_EXPORT Context {
 public:
  /**
   * @brief Create an initialized engine from a validated configuration.
   * @param config Requested engine settings; hardware limits are checked here.
   * @return An owned context, or an Error with code ErrorCode::context_in_use or
   *         ErrorCode::context_initialization.
   * @throws std::bad_alloc if host allocation fails. GPU allocation failures are returned as Error.
   *
   * @code{.cpp}
   * #include <sirius/context/config_builder.hpp>
   * #include <sirius/context/context.hpp>
   * #include <iostream>
   *
   * int main()
   * {
   *   auto config = sirius::ContextConfigBuilder{}.build();
   *   if (!config) {
   *     std::cerr << config.error().message << '\n';
   *     return 1;
   *   }
   *   auto context = sirius::Context::create(*config);
   *   if (!context) {
   *     std::cerr << context.error().message << '\n';
   *     return 1;
   *   }
   * }
   * @endcode
   */
  [[nodiscard]] static std::expected<std::unique_ptr<Context>, Error> create(
    const ContextConfig& config);

  /// Release the engine and its resources without throwing.
  ///
  /// @code{.cpp}
  /// #include <sirius/context/config_builder.hpp>
  /// #include <sirius/context/context.hpp>
  ///
  /// auto config = sirius::ContextConfigBuilder{}.build();
  /// if (config) {
  ///   auto context = sirius::Context::create(*config);
  ///   if (context) { context->reset(); } // Release the engine now.
  /// }
  /// @endcode
  ~Context() noexcept;

  Context(const Context&)            = delete;
  Context& operator=(const Context&) = delete;
  Context(Context&&)                 = delete;
  Context& operator=(Context&&)      = delete;

 private:
  struct Impl;
  explicit Context(std::unique_ptr<Impl> impl) noexcept;
  std::unique_ptr<Impl> impl_;
};

}  // namespace sirius
