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

#include <format>
#include <stdexcept>
#include <string>

namespace sirius {

class internal_exception : public std::runtime_error {
 public:
  explicit internal_exception(const std::string& msg) : std::runtime_error(msg) {}

  template <typename... Args>
  explicit internal_exception(std::format_string<Args...> fmt, Args&&... args)
    : std::runtime_error(std::format(fmt, std::forward<Args>(args)...))
  {
  }
};

class not_implemented_exception : public std::runtime_error {
 public:
  explicit not_implemented_exception(const std::string& msg) : std::runtime_error(msg) {}

  template <typename... Args>
  explicit not_implemented_exception(std::format_string<Args...> fmt, Args&&... args)
    : std::runtime_error(std::format(fmt, std::forward<Args>(args)...))
  {
  }
};

class invalid_input_exception : public std::runtime_error {
 public:
  explicit invalid_input_exception(const std::string& msg) : std::runtime_error(msg) {}

  template <typename... Args>
  explicit invalid_input_exception(std::format_string<Args...> fmt, Args&&... args)
    : std::runtime_error(std::format(fmt, std::forward<Args>(args)...))
  {
  }
};

}  // namespace sirius
