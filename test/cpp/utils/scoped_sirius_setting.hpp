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

#include <concepts>
#include <cstdint>
#include <string>
#include <string_view>
#include <utility>

namespace sirius::test {

/** Temporarily overrides a Sirius or DuckDB setting; destruction silently attempts restoration. */
class scoped_sirius_setting final {
 public:
  template <std::same_as<bool> Bool>
  scoped_sirius_setting(duckdb::Connection& connection, std::string name, Bool value)
    : scoped_sirius_setting(
        connection, std::move(name), setting_value{duckdb::Value::BOOLEAN(value)})
  {
  }

  template <std::same_as<std::uint64_t> UInt64>
  scoped_sirius_setting(duckdb::Connection& connection, std::string name, UInt64 value)
    : scoped_sirius_setting(
        connection, std::move(name), setting_value{duckdb::Value::UBIGINT(value)})
  {
  }

  //! @p value is unquoted; the guard quotes it.
  scoped_sirius_setting(duckdb::Connection& connection, std::string name, std::string_view value)
    : scoped_sirius_setting(
        connection, std::move(name), setting_value{duckdb::Value(std::string{value})})
  {
  }

  ~scoped_sirius_setting() noexcept
  {
    try {
      con_.Query("SET " + name_ + " = " + original_.ToSQLString() + ";");
    } catch (...) {
      // Destructors must not mask the failure that caused scope unwinding.
    }
  }

  scoped_sirius_setting(scoped_sirius_setting const&)            = delete;
  scoped_sirius_setting& operator=(scoped_sirius_setting const&) = delete;
  scoped_sirius_setting(scoped_sirius_setting&&)                 = delete;
  scoped_sirius_setting& operator=(scoped_sirius_setting&&)      = delete;

 private:
  //! Wraps a value so the delegating constructor never competes with the public overloads.
  struct setting_value {
    duckdb::Value value;
  };

  scoped_sirius_setting(duckdb::Connection& connection,
                        std::string name,
                        setting_value const& value)
    : con_(connection), name_(std::move(name))
  {
    auto current = con_.Query("SELECT current_setting('" + name_ + "');");
    REQUIRE(current);
    REQUIRE_FALSE(current->HasError());
    original_ = current->GetValue(0, 0);

    auto applied = con_.Query("SET " + name_ + " = " + value.value.ToSQLString() + ";");
    REQUIRE(applied);
    REQUIRE_FALSE(applied->HasError());
  }

  duckdb::Connection& con_;
  std::string name_;
  duckdb::Value original_;
};

}  // namespace sirius::test
