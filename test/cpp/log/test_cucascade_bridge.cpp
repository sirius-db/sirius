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

#include "catch.hpp"
#include "config.hpp"
#include "log/logging.hpp"
#include "sirius_context.hpp"
#include "utils/log_test_utils.hpp"

#include <cucascade/log/logging.hpp>

#include <cstdint>
#include <memory>
#include <optional>
#include <source_location>
#include <string>
#include <string_view>
#include <vector>

using sirius::log::level;
using sirius::log::set_level;
using sirius::log::set_sink;

using recording_sink        = sirius::test::recording_log_sink;
using scoped_recording_sink = sirius::test::scoped_recording_log_sink;

TEST_CASE("cuCascade records reach the Sirius sink with their call site", "[log]")
{
  scoped_recording_sink scoped{"trace"};

  CUCASCADE_LOG_WARN("from cucascade {}", 42);
  const uint32_t expected_line = __LINE__ - 1;

  auto records = scoped.records();
  REQUIRE(records.size() == 1);
  CHECK(records[0].level == level::warn);
  CHECK(records[0].message == "[cucascade] from cucascade 42");
  CHECK(records[0].file.ends_with("test_cucascade_bridge.cpp"));
  CHECK(records[0].line == expected_line);
}

TEST_CASE("cuCascade fatal records arrive as critical", "[log]")
{
  scoped_recording_sink scoped{"trace"};

  CUCASCADE_LOG_FATAL("fatal {}", 1);

  auto records = scoped.records();
  REQUIRE(records.size() == 1);
  CHECK(records[0].level == level::critical);
  CHECK(records[0].message == "[cucascade] fatal 1");
}

TEST_CASE("The Sirius level gates cuCascade records before formatting", "[log]")
{
  scoped_recording_sink scoped{"error"};

  // set_sink(sink, level) mirrored the level into cuCascade's own gate.
  CHECK_FALSE(cucascade::log::is_enabled(cucascade::log::level::warn));
  CHECK(cucascade::log::is_enabled(cucascade::log::level::error));

  CUCASCADE_LOG_WARN("dropped {}", 1);
  CUCASCADE_LOG_ERROR("kept {}", 2);
  auto records = scoped.records();
  REQUIRE(records.size() == 1);
  CHECK(records[0].level == level::error);
  CHECK(records[0].message == "[cucascade] kept 2");

  // set_level re-targets both the Sirius sink and cuCascade's gate.
  set_level(level::trace);
  CHECK(cucascade::log::is_enabled(cucascade::log::level::trace));

  CUCASCADE_LOG_TRACE("traced {}", 3);
  records = scoped.records();
  REQUIRE(records.size() == 2);
  CHECK(records[1].level == level::trace);
  CHECK(records[1].message == "[cucascade] traced 3");
}

TEST_CASE("Swapping the Sirius sink re-routes cuCascade records", "[log]")
{
  scoped_recording_sink scoped{"trace"};

  CUCASCADE_LOG_INFO("to a {}", 1);
  REQUIRE(scoped.sink().count() == 1);

  auto other = std::make_shared<recording_sink>();
  set_sink(other, level::trace);
  CUCASCADE_LOG_INFO("to b {}", 2);

  CHECK(scoped.sink().count() == 1);
  auto records = other->records();
  REQUIRE(records.size() == 1);
  CHECK(records[0].level == level::info);
  CHECK(records[0].message == "[cucascade] to b 2");
}

TEST_CASE("The noop sink mutes cuCascade's gate, and a level change keeps it muted", "[log]")
{
  scoped_recording_sink scoped{"trace"};

  set_sink(nullptr);  // swaps in a noop, detaching the recorder
  CHECK(cucascade::log::get_current_sink() != nullptr);
  // The noop sink accepts nothing, so nothing is even formatted on cuCascade's side.
  CHECK_FALSE(cucascade::log::is_enabled(cucascade::log::level::fatal));
  CHECK_NOTHROW(CUCASCADE_LOG_ERROR("dropped {}", 1));
  CHECK(scoped.sink().count() == 0);

  // SET sirius_log_level under the noop backend must not reopen the gate.
  set_level(level::debug);
  CHECK_FALSE(cucascade::log::is_enabled(cucascade::log::level::debug));
}

TEST_CASE("The one-argument set_sink mirrors the sink's own level onto cuCascade", "[log]")
{
  scoped_recording_sink scoped{"trace"};

  auto other = std::make_shared<recording_sink>();
  other->set_level(level::debug);
  set_sink(other);

  CHECK(cucascade::log::is_enabled(cucascade::log::level::debug));
  CHECK_FALSE(cucascade::log::is_enabled(cucascade::log::level::trace));
}

namespace {

/// Accepts per record, like the DuckDB sink: Sirius cannot know its level in advance.
class dynamic_sink final : public sirius::log::sink {
 public:
  void set_level(level) override {}
  [[nodiscard]] bool should_log(level) const override { return true; }
  void log(level, const std::source_location&, std::string_view message) override
  {
    _messages.emplace_back(message);
  }
  bool flush() override { return true; }
  [[nodiscard]] std::optional<level> min_accepted_level() const override { return std::nullopt; }

  [[nodiscard]] const std::vector<std::string>& messages() const { return _messages; }

 private:
  std::vector<std::string> _messages;
};

}  // namespace

TEST_CASE("A sink that decides per record receives every cuCascade level", "[log]")
{
  scoped_recording_sink scoped{"error"};
  auto dynamic = std::make_shared<dynamic_sink>();
  set_sink(dynamic, level::error);

  CHECK(cucascade::log::is_enabled(cucascade::log::level::trace));
  CUCASCADE_LOG_TRACE("seen {}", 1);
  CHECK(dynamic->messages() == std::vector<std::string>{"[cucascade] seen 1"});
}

TEST_CASE("install_configured_log_sink with the noop backend mutes cuCascade", "[log]")
{
  using duckdb::Config;
  scoped_recording_sink scoped{"trace"};
  const auto previous_backend = Config::LOG_BACKEND;

  Config::LOG_BACKEND = "noop";
  duckdb::install_configured_log_sink(nullptr);
  CHECK_FALSE(cucascade::log::is_enabled(cucascade::log::level::fatal));

  Config::LOG_BACKEND = previous_backend;
  // The scoped sink's destructor reinstalls the configured spdlog sink.
}
