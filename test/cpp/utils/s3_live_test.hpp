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

#include "catch.hpp"

#include <cstdlib>
#include <exception>
#include <string>

namespace sirius::test::s3 {

inline std::string getenv_or(char const* key, char const* dflt = "")
{
  auto const* value = std::getenv(key);
  return (value && *value) ? value : dflt;
}

inline bool strict_mode_enabled()
{
  auto const value = getenv_or("SIRIUS_TEST_S3_STRICT", "0");
  return value == "1" || value == "true" || value == "TRUE" || value == "yes" ||
         value == "on";
}

inline void handle_live_runtime_failure(std::string const& action,
                                        std::exception const& ex,
                                        std::string const& skip_reason)
{
  auto const detail = action + ": " + ex.what();
  if (strict_mode_enabled()) FAIL(detail);
  WARN(detail);
  SUCCEED(skip_reason);
}

}  // namespace sirius::test::s3
