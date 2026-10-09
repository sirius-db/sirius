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

#include <sirius/c/error.h>

#include <memory>
#include <new>
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
  /// An allocation failed. The diagnostic may be empty.
  allocation_failure,
  /// A supplied argument cannot be represented by the C interface.
  invalid_argument,
  /// An unexpected implementation failure or unknown status was reported.
  internal_error,
  /// Hardware resolution or engine initialization failed.
  context_initialization,
};

/// @brief A failure returned through std::expected by the public API.
/// Inspect code to choose a recovery action and message to explain the failure.
///
/// @code{.cpp}
/// #include <sirius/context/config_builder.hpp>
/// #include <iostream>
///
/// auto config = sirius::ContextConfigBuilder::from_yaml("sirius.yaml");
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
  /// May be empty when diagnostic storage is unavailable.
  /// Its wording is not a stable interface and should not be parsed.
  std::string message;

 private:
  static Error from_c(sirius_status status, sirius_error* diagnostic) noexcept
  {
    std::unique_ptr<sirius_error, decltype(&sirius_error_destroy)> owner(diagnostic,
                                                                         sirius_error_destroy);
    auto code = ErrorCode::internal_error;
    switch (status) {
      case SIRIUS_CONFIGURATION_IO: code = ErrorCode::configuration_io; break;
      case SIRIUS_MALFORMED_YAML: code = ErrorCode::malformed_yaml; break;
      case SIRIUS_INVALID_CONFIGURATION: code = ErrorCode::invalid_configuration; break;
      case SIRIUS_ALLOCATION_FAILURE: code = ErrorCode::allocation_failure; break;
      case SIRIUS_INVALID_ARGUMENT: code = ErrorCode::invalid_argument; break;
      case SIRIUS_CONTEXT_INITIALIZATION: code = ErrorCode::context_initialization; break;
      default: break;
    }
    try {
      return {code,
              std::string(sirius_error_message(diagnostic), sirius_error_message_size(diagnostic))};
    } catch (const std::bad_alloc&) {
      return {ErrorCode::allocation_failure, {}};
    } catch (...) {
      return {ErrorCode::internal_error, {}};
    }
  }
  friend class ContextConfigBuilder;
  friend class Context;
};

}  // namespace sirius
