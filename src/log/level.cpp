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

#include "log/level.hpp"

namespace sirius::log {

bool string_to_enum(std::string_view name, level& lvl)
{
  if (name == "trace") {
    lvl = level::trace;
  } else if (name == "debug") {
    lvl = level::debug;
  } else if (name == "info") {
    lvl = level::info;
  } else if (name == "warn") {
    lvl = level::warn;
  } else if (name == "error") {
    lvl = level::error;
  } else if (name == "critical") {
    lvl = level::critical;
  } else if (name == "off") {
    lvl = level::off;
  } else {
    return false;
  }
  return true;
}

}  // namespace sirius::log
