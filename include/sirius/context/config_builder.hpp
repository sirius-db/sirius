/*
 * Copyright 2025, Sirius Contributors.
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
 * @brief Build Sirius context configurations from defaults, YAML, and C++ overrides.
 *
 * Requires C++23 and standard-library support for std::expected. Include
 * `<sirius/context/config_builder.hpp>` and link the CMake target `sirius::sirius`.
 */

#pragma once

#include <sirius/context/config.hpp>
#include <sirius/error.hpp>
#include <sirius/export.hpp>

#include <cstdint>
#include <expected>
#include <filesystem>
#include <memory>

namespace sirius {

/**
 * @brief Assemble a configuration using defaults, YAML, and explicit overrides.
 *
 * Explicit setters take precedence over YAML, which takes precedence over
 * built-in defaults. Setters record candidate values; build() validates them.
 * Copies can be edited independently. Editing a builder does not change any
 * ContextConfig already produced from it. Concurrent mutation of the same
 * builder requires external synchronization.
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
 * Load a YAML file and override its GPU usage limit:
 * @code{.cpp}
 * #include <sirius/context/config_builder.hpp>
 *
 * std::expected<sirius::ContextConfig, sirius::Error> make_config()
 * {
 *   auto builder = sirius::ContextConfigBuilder::from_yaml("sirius.yaml");
 *   if (!builder) { return std::unexpected(builder.error()); }
 *
 *   builder->gpu_usage_limit_bytes(8ULL << 30);
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
  /// Copy the current settings; subsequent edits affect only the edited builder.
  ///
  /// @code{.cpp}
  /// #include <sirius/context/config_builder.hpp>
  ///
  /// sirius::ContextConfigBuilder original;
  /// sirius::ContextConfigBuilder copy = original;
  /// copy.gpu_usage_limit_fraction(0.5); // The original keeps its defaults.
  /// @endcode
  ContextConfigBuilder(const ContextConfigBuilder&) noexcept;
  /// Replace this builder's settings with an independent copy of another's.
  ///
  /// @code{.cpp}
  /// #include <sirius/context/config_builder.hpp>
  ///
  /// sirius::ContextConfigBuilder defaults;
  /// sirius::ContextConfigBuilder builder;
  /// builder.gpu_usage_limit_fraction(0.5);
  /// builder = defaults; // Replace the edited settings with the defaults.
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
   * Unknown keys and conflicting settings are rejected before overrides can be
   * applied. Editing or deleting the file after loading has no effect on this
   * builder. Loading does not discover hardware or resolve hardware-dependent values.
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
   *   auto config = builder->gpu_usage_limit_fraction(0.5).build();
   * }
   * @endcode
   */
  [[nodiscard]] static std::expected<ContextConfigBuilder, Error> from_yaml(
    const std::filesystem::path& path);

  /**
   * @brief Set the GPU memory usage limit in bytes for each selected GPU.
   *
   * Corresponds to `sirius.memory.gpu.usage_limit_bytes` in YAML. Replaces any
   * earlier byte or fraction choice, including a value loaded from YAML.
   * Reservation limits are configured separately and are unaffected.
   *
   * build() rejects overrides combined with non-empty low-level `sirius.space`
   * lists. Whether each selected GPU has enough physical memory is checked during
   * context creation. A zero limit requests zero GPU capacity; it does not disable
   * GPU use.
   *
   * @param bytes Maximum configured memory capacity per selected GPU, in bytes.
   * @return This builder, for chaining. Validation occurs in build().
   * @throws std::bad_alloc if storing the override fails.
   *
   * @code{.cpp}
   * #include <sirius/context/config_builder.hpp>
   *
   * sirius::ContextConfigBuilder builder;
   * builder.gpu_usage_limit_bytes(8ULL << 30); // 8 GiB per selected GPU.
   * auto config = builder.build();
   * @endcode
   */
  ContextConfigBuilder& gpu_usage_limit_bytes(std::uint64_t bytes);

  /**
   * @brief Set GPU memory capacity as a fraction of each selected GPU's total memory.
   *
   * Corresponds to `sirius.memory.gpu.usage_limit_fraction` in YAML. The built-in
   * default is 0.95. Replaces any earlier byte or fraction choice, including a
   * value loaded from YAML, without changing reservation limits.
   *
   * build() rejects non-finite values, values outside [0, 1], and overrides
   * combined with non-empty low-level `sirius.space` lists. Zero produces zero
   * configured GPU capacity; it does not disable GPU use.
   *
   * @param fraction Fraction of total memory to configure on each selected GPU.
   * @return This builder, for chaining. Validation occurs in build().
   * @throws std::bad_alloc if storing the override fails.
   *
   * @code{.cpp}
   * #include <sirius/context/config_builder.hpp>
   *
   * sirius::ContextConfigBuilder builder;
   * builder.gpu_usage_limit_fraction(0.75); // 75% of each selected GPU's memory.
   * auto config = builder.build();
   * @endcode
   */
  ContextConfigBuilder& gpu_usage_limit_fraction(double fraction);

  /**
   * @brief Validate the settings and produce an immutable configuration snapshot.
   *
   * Explicit overrides win over YAML, then defaults. Fractions remain fractions;
   * GPU selection and capacity-derived operator defaults are resolved during
   * context creation. Explicit YAML operator settings retain precedence over
   * those derived defaults.
   *
   * The source file is never reread, and hardware is never queried. This operation
   * leaves the builder unchanged, including on failure, so invalid settings can
   * be replaced and retried.
   *
   * @return A valid ContextConfig on success, otherwise an Error with code
   *         ErrorCode::invalid_configuration.
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
