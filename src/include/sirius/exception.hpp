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

#pragma once

#include <cstdarg>
#include <cstdio>
#include <stdexcept>
#include <string>

namespace sirius {

namespace detail {

inline std::string format_message(const char* fmt, ...)
{
  va_list args;
  va_start(args, fmt);
  va_list args_copy;
  va_copy(args_copy, args);
  int size = std::vsnprintf(nullptr, 0, fmt, args);
  va_end(args);
  std::string result(size, '\0');
  std::vsnprintf(result.data(), size + 1, fmt, args_copy);
  va_end(args_copy);
  return result;
}

}  // namespace detail

class internal_exception : public std::runtime_error {
 public:
  explicit internal_exception(const std::string& msg) : std::runtime_error(msg) {}

  template <typename... Args>
  explicit internal_exception(const char* fmt, Args... args)
    : std::runtime_error(detail::format_message(fmt, args...))
  {
  }
};

class not_implemented_exception : public std::runtime_error {
 public:
  explicit not_implemented_exception(const std::string& msg) : std::runtime_error(msg) {}

  template <typename... Args>
  explicit not_implemented_exception(const char* fmt, Args... args)
    : std::runtime_error(detail::format_message(fmt, args...))
  {
  }
};

class invalid_input_exception : public std::runtime_error {
 public:
  explicit invalid_input_exception(const std::string& msg) : std::runtime_error(msg) {}

  template <typename... Args>
  explicit invalid_input_exception(const char* fmt, Args... args)
    : std::runtime_error(detail::format_message(fmt, args...))
  {
  }
};

}  // namespace sirius
