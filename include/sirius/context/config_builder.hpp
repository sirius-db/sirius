/*
 * Copyright 2026, Sirius Contributors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/**
 * @file
 * @brief Build Sirius context configurations from defaults or YAML.
 *
 * Requires C++23 and standard-library support for std::expected. Include
 * `<sirius/context/config_builder.hpp>` and link `sirius::sirius`, or
 * `sirius::sirius_static` when the static library is enabled.
 */

#pragma once

#include <sirius/context/config.hpp>
#include <sirius/error.hpp>
#include <sirius/export.hpp>

#include <expected>
#include <filesystem>
#include <memory>

namespace sirius {

/**
 * @brief Assemble a configuration using defaults or YAML.
 *
 * @par Thread safety
 * Objects may be transferred between threads, including for destruction.
 * Const operations and copying may run concurrently while the source stays alive.
 * Assignment and destruction require exclusive access to that object; separate
 * copies may be used independently on different threads.
 *
 * YAML settings take precedence over built-in defaults. Copies share immutable
 * settings and remain valid independently of the original builder.
 *
 * from_yaml() and build() validate settings without hardware access. They can be
 * used on machines without GPUs. GPU availability and capacity are checked during
 * context creation. Ordinary configuration failures are returned as Error values;
 * allocation failures may throw std::bad_alloc.
 *
 * Create a configuration from defaults:
 * @code{.cpp}
 * #include <sirius/context/config_builder.hpp>
 * #include <iostream>
 *
 * int main()
 * {
 *   auto config = sirius::ContextConfigBuilder{}.build();
 *   if (!config) {
 *     std::cerr << config.error().message << '\n';
 *     return 1;
 *   }
 * }
 * @endcode
 *
 * Load a YAML file:
 * @code{.cpp}
 * #include <sirius/context/config_builder.hpp>
 *
 * std::expected<sirius::ContextConfig, sirius::Error> make_config()
 * {
 *   auto builder = sirius::ContextConfigBuilder::from_yaml("sirius.yaml");
 *   if (!builder) { return std::unexpected(builder.error()); }
 *
 *   return builder->build();
 * }
 * @endcode
 */
class SIRIUS_EXPORT ContextConfigBuilder {
 public:
  /// Start with built-in defaults. Hardware resolution is deferred to context creation.
  /// @throws std::bad_alloc if builder storage cannot be allocated.
  ///
  /// @code{.cpp}
  /// #include <sirius/context/config_builder.hpp>
  ///
  /// sirius::ContextConfigBuilder builder;
  /// auto config = builder.build();
  /// @endcode
  ContextConfigBuilder();
  /// Share the current immutable settings with another builder.
  ///
  /// @code{.cpp}
  /// #include <sirius/context/config_builder.hpp>
  ///
  /// sirius::ContextConfigBuilder original;
  /// sirius::ContextConfigBuilder copy = original;
  /// auto config = copy.build();
  /// @endcode
  ContextConfigBuilder(const ContextConfigBuilder&) noexcept;
  /// Replace this builder's settings with an independent copy of another's.
  ///
  /// @code{.cpp}
  /// #include <sirius/context/config_builder.hpp>
  ///
  /// sirius::ContextConfigBuilder defaults;
  /// sirius::ContextConfigBuilder builder;
  /// builder = defaults; // Replace the settings with the defaults.
  /// @endcode
  ContextConfigBuilder& operator=(const ContextConfigBuilder&) noexcept;
  /// Release the builder without affecting configurations already built from it.
  ///
  /// @code{.cpp}
  /// #include <sirius/context/config_builder.hpp>
  ///
  /// auto config = [] {
  ///   sirius::ContextConfigBuilder builder;
  ///   return builder.build();
  /// }(); // The configuration result outlives the builder.
  /// @endcode
  ~ContextConfigBuilder() noexcept;

  /**
   * @brief Read and validate a YAML file, retaining its contents for later builds.
   *
   * Uses the existing Sirius YAML schema, including its defaults and byte units.
   * Unknown keys and conflicting settings are rejected. Editing or deleting the file after loading
   * has no effect on this builder. Loading does not discover hardware or resolve hardware-dependent
   * values.
   *
   * The [YAML configuration
   * reference](https://github.com/sirius-db/sirius/blob/main/docs/super-sirius/configuration.md)
   * lists supported fields, defaults, byte units, and constraints. Settings belong
   * under the top-level `sirius` key; omitted settings use built-in defaults.
   * For example, `sirius.yaml` can contain:
   * @code{.yaml}
   * sirius:
   *   topology:
   *     num_gpus: 1
   *   memory:
   *     gpu:
   *       usage_limit_bytes: 8Gi
   *   executor:
   *     pipeline:
   *       num_threads: 4
   * @endcode
   *
   * This method reads only the supplied path; the automatic config-file search
   * described in the reference does not apply here.
   *
   * @param path Configuration file to read; relative paths use the working directory.
   * @return A builder on success, otherwise an Error with code
   *         ErrorCode::configuration_io, ErrorCode::malformed_yaml,
   *         or ErrorCode::invalid_configuration.
   * @throws std::bad_alloc if allocation fails.
   *
   * @code{.cpp}
   * #include <sirius/context/config_builder.hpp>
   *
   * auto builder = sirius::ContextConfigBuilder::from_yaml("sirius.yaml");
   * if (builder) {
   *   auto config = builder->build();
   * }
   * @endcode
   */
  [[nodiscard]] static std::expected<ContextConfigBuilder, Error> from_yaml(
    const std::filesystem::path& path);

  /**
   * @brief Produce an immutable snapshot of the validated settings.
   *
   * YAML settings take precedence over defaults. Fractions remain fractions;
   * GPU selection and capacity-derived operator defaults are resolved during
   * context creation. Explicit YAML operator settings retain precedence over
   * those derived defaults.
   *
   * The source file is never reread, and hardware is never queried. This operation
   * leaves the builder unchanged.
   *
   * @return A valid ContextConfig. Settings are validated when loaded.
   * @throws std::bad_alloc if allocation fails.
   *
   * @code{.cpp}
   * #include <sirius/context/config_builder.hpp>
   * #include <iostream>
   *
   * auto config = sirius::ContextConfigBuilder{}.build();
   * if (!config) {
   *   std::cerr << config.error().message << '\n';
   * }
   * @endcode
   */
  [[nodiscard]] std::expected<ContextConfig, Error> build() const;

 private:
  struct Impl;
  explicit ContextConfigBuilder(std::shared_ptr<const Impl> impl);
  std::shared_ptr<const Impl> impl_;
};

}  // namespace sirius
