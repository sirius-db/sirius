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

#include <catch.hpp>
#include <duckdb.hpp>

#include <cstdint>
#include <string>
#include <utility>

namespace sirius::test {

/** Temporarily overrides a shared Sirius setting; destruction silently attempts restoration. */
class scoped_sirius_setting final {
 public:
  scoped_sirius_setting(duckdb::Connection& connection, std::string name, bool value)
    : scoped_sirius_setting(
        connection, std::move(name), value ? std::string{"true"} : std::string{"false"})
  {
  }

  scoped_sirius_setting(duckdb::Connection& connection, std::string name, std::uint64_t value)
    : scoped_sirius_setting(connection, std::move(name), std::to_string(value))
  {
  }

  ~scoped_sirius_setting() noexcept
  {
    try {
      con_.Query("SET " + name_ + " = " + original_ + ";");
    } catch (...) {
      // Destructors must not mask the failure that caused scope unwinding.
    }
  }

  scoped_sirius_setting(scoped_sirius_setting const&)            = delete;
  scoped_sirius_setting& operator=(scoped_sirius_setting const&) = delete;
  scoped_sirius_setting(scoped_sirius_setting&&)                 = delete;
  scoped_sirius_setting& operator=(scoped_sirius_setting&&)      = delete;

 private:
  scoped_sirius_setting(duckdb::Connection& connection, std::string name, std::string value_literal)
    : con_(connection), name_(std::move(name))
  {
    auto current = con_.Query("SELECT current_setting('" + name_ + "');");
    REQUIRE(current);
    REQUIRE_FALSE(current->HasError());
    original_ = current->GetValue(0, 0).ToString();

    auto applied = con_.Query("SET " + name_ + " = " + value_literal + ";");
    REQUIRE(applied);
    REQUIRE_FALSE(applied->HasError());
  }

  duckdb::Connection& con_;
  std::string name_;
  std::string original_;
};

}  // namespace sirius::test
