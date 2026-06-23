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

#include <spdlog/fmt/bundled/core.h>
#include <spdlog/spdlog.h>

#include <exception>
#include <source_location>
#include <string>
#include <string_view>
#include <utility>

inline void log_exception_helper(const std::source_location& loc)
{
  spdlog::source_loc spd_loc{loc.file_name(), static_cast<int>(loc.line()), loc.function_name()};
  try {
    throw;
  } catch (const std::exception& e) {
    spdlog::log(spd_loc, spdlog::level::err, "Exception caught: {}", e.what());
  } catch (...) {
    spdlog::log(spd_loc, spdlog::level::err, "UNKNOWN exception caught");
  }
}

template <typename... Args>
void log_exception_helper(const std::source_location& loc, std::string_view fmt_str, Args&&... args)
{
  spdlog::source_loc spd_loc{loc.file_name(), static_cast<int>(loc.line()), loc.function_name()};

  // Formats your string cleanly using v1.8.5 syntax
  std::string user_msg = fmt::format(fmt_str, std::forward<Args>(args)...);

  try {
    throw;
  } catch (const std::exception& e) {
    spdlog::log(spd_loc, spdlog::level::err, "{}: {}", user_msg, e.what());
  } catch (...) {
    spdlog::log(spd_loc, spdlog::level::err, "{}: UNKNOWN exception", user_msg);
  }
}

#define SIRIUS_TRY_AND_LOG_EXCEPTION(expression, ...)                                 \
  try {                                                                               \
    expression;                                                                       \
  } catch (...) {                                                                     \
    log_exception_helper(std::source_location::current() __VA_OPT__(, ) __VA_ARGS__); \
  }
