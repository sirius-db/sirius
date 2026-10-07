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

#pragma once

#include <string>

namespace sirius {

/// @brief Machine-readable categories for failures reported by the public API.
/// Use these categories to handle failures; diagnostic messages may change.
enum class ErrorCode {
  /// The configuration file could not be opened or read.
  configuration_io,
  /// The configuration text could not be parsed as YAML.
  malformed_yaml,
  /// A setting is unknown, invalid, or conflicts with another.
  invalid_configuration,
};

/// @brief A failure returned through std::expected by the public API.
/// Inspect code to choose a recovery action and message to explain the failure.
///
/// @code{.cpp}
/// #include <sirius/context/config_builder.hpp>
/// #include <iostream>
///
/// auto config = sirius::ContextConfigBuilder{}.gpu_usage_limit_fraction(1.5).build();
/// if (!config) {
///   const sirius::Error& error = config.error();
///   if (error.code == sirius::ErrorCode::invalid_configuration) {
///     std::cerr << error.message << '\n';
///   }
/// }
/// @endcode
struct Error {
  /// Category for programmatic error handling.
  ErrorCode code;
  /// Human-readable diagnostic, including file or setting context when available.
  /// Its wording is not a stable interface and should not be parsed.
  std::string message;
};

}  // namespace sirius
