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

#include "util/env_guard.hpp"

#include <cstdlib>  // for ::setenv / ::unsetenv / ::getenv
#include <utility>

namespace sirius {
namespace util {

env_guard::env_guard(std::string name, const std::string& value) : name_(std::move(name))
{
  if (const char* prev = ::getenv(name_.c_str())) { previous_value_ = std::string(prev); }
  ::setenv(name_.c_str(), value.c_str(), /*overwrite=*/1);
  active_ = true;
}

env_guard::~env_guard() { restore(); }

env_guard::env_guard(env_guard&& other) noexcept
  : name_(std::move(other.name_)),
    previous_value_(std::move(other.previous_value_)),
    active_(other.active_)
{
  other.active_ = false;
}

env_guard& env_guard::operator=(env_guard&& other) noexcept
{
  if (this != &other) {
    restore();
    name_           = std::move(other.name_);
    previous_value_ = std::move(other.previous_value_);
    active_         = other.active_;
    other.active_   = false;
  }
  return *this;
}

void env_guard::restore() noexcept
{
  if (!active_) { return; }
  if (previous_value_) {
    ::setenv(name_.c_str(), previous_value_->c_str(), /*overwrite=*/1);
  } else {
    ::unsetenv(name_.c_str());
  }
  active_ = false;
}

}  // namespace util
}  // namespace sirius
