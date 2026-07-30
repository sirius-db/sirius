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
/// The extraction can fail two ways, and both have to fall back to the raw text.
///
/// It can throw: a message that merely *starts* with `{` without being valid JSON
/// makes the parser throw, and the construction can also throw on allocation.
///
/// And it can succeed while yielding nothing. The parser treats any `{`-leading
/// valid JSON as an envelope but fills the message **only** from an
/// `exception_message` key; every other key goes to extra info. So a legal JSON
/// object that simply is not DuckDB's envelope — `{"error":"missing"}` — extracts
/// to the empty string, and returning that would discard the original error.
///
/// Both call sites need this where throwing is not an option: inside a DuckDB
/// optimizer extension hook, which has no error path, and inside catch handlers
/// whose exception must not be replaced by a secondary one. A worse message is
/// always preferable to losing the original error.
[[nodiscard]] inline std::string sanitized_message(std::exception const& e) noexcept
{
  try {
    auto message = duckdb::ErrorData(e).RawMessage();
    if (!message.empty()) { return message; }
  } catch (...) {  // NOLINT(bugprone-empty-catch) — fall through to the raw text
  }
  try {
    return e.what();
  } catch (...) {
    return "<unavailable>";
  }
}

}  // namespace sirius
