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

#include <log/logging.hpp>

#include <exception>
#include <format>
#include <source_location>
#include <string>
#include <string_view>

inline void log_exception_helper(const std::source_location& loc)
{
  try {
    throw;
  } catch (const std::exception& e) {
    sirius::LogAt(sirius::log_level::error, loc, std::format("Exception caught: {}", e.what()));
  } catch (...) {
    sirius::LogAt(sirius::log_level::error, loc, "UNKNOWN exception caught");
  }
}

template <typename... Args>
void log_exception_helper(const std::source_location& loc, std::string_view fmt_str, Args&&... args)
{
  std::string user_msg = std::vformat(fmt_str, std::make_format_args(args...));

  try {
    throw;
  } catch (const std::exception& e) {
    sirius::LogAt(sirius::log_level::error, loc, std::format("{}: {}", user_msg, e.what()));
  } catch (...) {
    sirius::LogAt(sirius::log_level::error, loc, std::format("{}: UNKNOWN exception", user_msg));
  }
}

#define SIRIUS_TRY_AND_LOG_EXCEPTION(expression, ...)                                 \
  try {                                                                               \
    expression;                                                                       \
  } catch (...) {                                                                     \
    log_exception_helper(std::source_location::current() __VA_OPT__(, ) __VA_ARGS__); \
  }
