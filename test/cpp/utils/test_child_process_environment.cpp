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

#include <catch.hpp>
#include <utils/child_process_environment.hpp>

#include <cstdlib>
#include <optional>
#include <string>
#include <string_view>

namespace {

std::optional<std::string> environment_value(char const* name)
{
  auto const* value = std::getenv(name);
  return value == nullptr ? std::nullopt : std::optional<std::string>{value};
}

std::optional<std::string_view> child_environment_value(char* const* environment,
                                                        std::string_view name)
{
  for (auto entry = environment; entry != nullptr && *entry != nullptr; ++entry) {
    std::string_view const value{*entry};
    auto const separator = value.find('=');
    if (separator != std::string_view::npos && value.substr(0, separator) == name) {
      return value.substr(separator + 1);
    }
  }
  return std::nullopt;
}

}  // namespace

TEST_CASE("child process environment applies changes without mutating the parent",
          "[utils][child_process_environment]")
{
  constexpr char const* override_name = "SIRIUS_CHILD_ENV_TEST_OVERRIDE";
  constexpr char const* removal_name  = "SIRIUS_DISABLE";
  auto const parent_override          = environment_value(override_name);
  auto const parent_removal           = environment_value(removal_name);
  auto const parent_path              = environment_value("PATH");

  sirius::test::child_process_environment child_environment{
    {{override_name, "child-only"}, {"PATH", "child-path"}}, {removal_name}};

  CHECK(child_environment_value(child_environment.data(), override_name) == "child-only");
  CHECK(child_environment_value(child_environment.data(), "PATH") == "child-path");
  CHECK_FALSE(child_environment_value(child_environment.data(), removal_name).has_value());

  CHECK(environment_value(override_name) == parent_override);
  CHECK(environment_value(removal_name) == parent_removal);
  CHECK(environment_value("PATH") == parent_path);
}
