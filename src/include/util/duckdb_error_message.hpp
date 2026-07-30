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

#include <duckdb/common/error_data.hpp>

#include <exception>
#include <string>

namespace sirius {

/// \brief The human-readable message of \p e, without DuckDB's serialized envelope.
///
/// `std::exception::what()` on a DuckDB exception carrying extra info is not
/// prose — it is a JSON object (`{"exception_type":…,"exception_message":…}`).
/// Concatenating it into a user-facing error leaks an internal wire format.
/// `duckdb::ErrorData` parses that envelope and exposes the message alone.
///
/// The parse can fail: a message that merely *starts* with `{` and is not valid
/// JSON makes the parser throw, and the construction can also throw on
/// allocation. Both call sites need this in places where throwing is not an
/// option — inside a DuckDB optimizer extension hook, which has no error path,
/// and inside catch handlers whose exception must not be replaced by a
/// secondary one. So the conversion is treated as fallible by construction:
/// on any failure the original `what()` text is returned unchanged. A worse
/// message is always preferable to losing the original error.
[[nodiscard]] inline std::string sanitized_message(std::exception const& e) noexcept
{
  try {
    return duckdb::ErrorData(e).RawMessage();
  } catch (...) {
    try {
      return e.what();
    } catch (...) {
      return "<unavailable>";
    }
  }
}

}  // namespace sirius
