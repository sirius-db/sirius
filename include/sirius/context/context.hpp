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
 * Only one active Sirius engine context is supported per process, including
 * contexts managed by other Sirius integrations. This restriction is not enforced:
 * callers must ensure context lifetimes do not overlap. Constructing another
 * context may succeed, but shared runtime resources can interfere with each other.
 * Multiple configurations may exist independently of the active context.
 * Destruction does not reset all process-wide settings: changing
 * `sirius.executor.downgrade.copy_chunk_bytes` after a successful creation is unsupported.
 * CUDA/NVTX initialization persists even after failed creation; NVTX injection settings
 * must remain unchanged for the process lifetime.
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
   * @return An owned context, or an Error with code ErrorCode::context_initialization
   *         or ErrorCode::allocation_failure. Allocation failures have an empty message.
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
    const ContextConfig& config) noexcept;

  /// Release the engine and its resources without throwing.
  /// An unrecoverable failure to stop workers or destroy resources terminates the process.
  ///
  /// @code{.cpp}
  /// #include <sirius/context/config_builder.hpp>
  /// #include <sirius/context/context.hpp>
  ///
  /// auto config = sirius::ContextConfigBuilder{}.build();
  /// if (config) {
  ///   auto context = sirius::Context::create(*config);
  ///   // Use the context if creation succeeded.
  /// } // The owned context is destroyed on leaving this scope.
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
