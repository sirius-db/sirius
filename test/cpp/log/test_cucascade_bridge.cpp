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
#include "log/logging.hpp"
#include "utils/log_test_utils.hpp"

#include <cucascade/log/logging.hpp>

#include <cstdint>
#include <memory>

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
  CHECK(records[0].message == "from cucascade 42");
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
  CHECK(records[0].message == "fatal 1");
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
  CHECK(records[0].message == "kept 2");

  // set_level re-targets both the Sirius sink and cuCascade's gate.
  set_level(level::trace);
  CHECK(cucascade::log::is_enabled(cucascade::log::level::trace));

  CUCASCADE_LOG_TRACE("traced {}", 3);
  records = scoped.records();
  REQUIRE(records.size() == 2);
  CHECK(records[1].level == level::trace);
  CHECK(records[1].message == "traced 3");
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
  CHECK(records[0].message == "to b 2");
}

TEST_CASE("cuCascade records vanish under the noop sink", "[log]")
{
  scoped_recording_sink scoped{"trace"};

  set_sink(nullptr);  // swaps in a noop, detaching the recorder
  CHECK(cucascade::log::get_current_sink() != nullptr);
  CHECK_NOTHROW(CUCASCADE_LOG_ERROR("dropped {}", 1));
  CHECK(scoped.sink().count() == 0);
}
