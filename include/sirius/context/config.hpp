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
 * @brief Immutable context configuration for Sirius.
 *
 * Include `<sirius/context/config.hpp>` to use this type.
 * Use `<sirius/context/config_builder.hpp>` to construct a configuration.
 */

#pragma once

#include <sirius/export.hpp>

#include <memory>

namespace sirius {

/**
 * @brief An immutable, validated configuration produced by ContextConfigBuilder::build().
 *
 * A snapshot retains its values independently of subsequent builder edits and
 * changes to the source YAML file. Copies share immutable storage and remain
 * valid after the original snapshot or builder is destroyed. Copying from an
 * rvalue also preserves the source snapshot.
 *
 * A configuration describes requested settings without discovering hardware,
 * initializing CUDA or telemetry, or allocating engine resources. Hardware-dependent
 * values are resolved during context creation, after telemetry starts. A valid
 * configuration does not guarantee that the machine can satisfy its requirements.
 */
class SIRIUS_EXPORT ContextConfig {
 public:
  /// Share the immutable snapshot with another configuration.
  ///
  /// @code{.cpp}
  /// #include <sirius/context/config_builder.hpp>
  ///
  /// auto config = sirius::ContextConfigBuilder{}.build();
  /// if (config) {
  ///   sirius::ContextConfig copy = *config;
  /// }
  /// @endcode
  ContextConfig(const ContextConfig&) noexcept;
  /// Replace this snapshot with another configuration's snapshot.
  ///
  /// @code{.cpp}
  /// #include <sirius/context/config_builder.hpp>
  ///
  /// auto config = sirius::ContextConfigBuilder{}.build();
  /// auto replacement = sirius::ContextConfigBuilder{}.gpu_usage_limit_fraction(0.5).build();
  /// if (config && replacement) {
  ///   *config = *replacement;
  /// }
  /// @endcode
  ContextConfig& operator=(const ContextConfig&) noexcept;
  /// Release this configuration's reference to the snapshot.
  ///
  /// @code{.cpp}
  /// #include <sirius/context/config_builder.hpp>
  ///
  /// {
  ///   auto config = sirius::ContextConfigBuilder{}.build();
  /// } // A successfully built configuration is released at the end of the scope.
  /// @endcode
  ~ContextConfig() noexcept;

 private:
  struct Impl;
  explicit ContextConfig(std::shared_ptr<const Impl> impl);
  std::shared_ptr<const Impl> impl_;

  friend class ContextConfigBuilder;
};

}  // namespace sirius
