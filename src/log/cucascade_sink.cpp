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

#include "log/cucascade_sink.hpp"

#include "log/sink.hpp"

#include <cucascade/log/logging.hpp>

#include <memory>

namespace sirius::log {

namespace {

level to_sirius_level(cucascade::log::level lvl)
{
  using enum cucascade::log::level;
  switch (lvl) {
    case trace: return level::trace;
    case debug: return level::debug;
    case info: return level::info;
    case warn: return level::warn;
    case error: return level::error;
    case fatal: return level::critical;
    case off: return level::off;
  }
  return level::info;
}

cucascade::log::level to_cucascade_level(level lvl)
{
  using enum level;
  switch (lvl) {
    case trace: return cucascade::log::level::trace;
    case debug: return cucascade::log::level::debug;
    case info: return cucascade::log::level::info;
    case warn: return cucascade::log::level::warn;
    case error: return cucascade::log::level::error;
    case critical: return cucascade::log::level::fatal;
    case off: return cucascade::log::level::off;
  }
  return cucascade::log::level::info;
}

/// Forwards cuCascade records to the Sirius sink active at the time of each
/// record. It holds no sink of its own, so a Sirius set_sink re-routes cuCascade
/// records without touching cuCascade.
class cucascade_forwarding_sink final : public cucascade::log::sink {
 public:
  void write(const cucascade::log::record& rec) override
  {
    // A shared_ptr copy, so a concurrent Sirius set_sink cannot destroy the
    // sink mid-write.
    auto s   = get_sink();
    auto lvl = to_sirius_level(rec.lvl);
    if (s->should_log(lvl)) { s->log(lvl, rec.location, rec.message); }
  }
};

}  // namespace

void attach_cucascade_logging()
{
  // One bridge for the process. Re-installing it would be harmless, but the
  // check keeps the common case (already attached) to one atomic load.
  static const auto bridge = std::make_shared<cucascade_forwarding_sink>();
  if (cucascade::log::get_current_sink() != bridge) { cucascade::log::set_sink(bridge); }
}

void set_cucascade_log_level(level lvl) { cucascade::log::set_min_level(to_cucascade_level(lvl)); }

}  // namespace sirius::log
