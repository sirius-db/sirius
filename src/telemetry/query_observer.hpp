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

#include "telemetry/telemetry_context.hpp"

#include <stdexcept>
#include <utility>

namespace sirius::telemetry {

/// One query observer. The invocation's synchronous owner alone mutates it;
/// consumers receive the UUID, never the mutable Quent FSM. Retains the exporter
/// through completion. Requires a context even when exporting is disabled.
class query_observer {
 public:
  explicit query_observer(std::shared_ptr<const telemetry_context> context,
                          const std::optional<std::string>& query_label   = std::nullopt,
                          const std::optional<std::string>& session_label = std::nullopt)
    : context_(std::move(context))
  {
    if (!context_) { throw std::invalid_argument("query_observer requires a telemetry context"); }
    handle_.emplace(context_->context()
                      .query_observer()
                      ->handle()
                      .init({
                        .instance_name  = query_label.value_or("unnamed_query"),
                        .query_group_id = context_->query_group_id_for(session_label),
                      })
                      .into_dynamic());
  }

  ~query_observer() noexcept { finish(); }
  query_observer(const query_observer&)            = delete;
  query_observer& operator=(const query_observer&) = delete;

  /// Transfer completion responsibility; the source may be destroyed or reassigned.
  query_observer(query_observer&& other) noexcept
    : context_(std::move(other.context_)),
      handle_(std::exchange(other.handle_, std::nullopt)),
      finished_(std::exchange(other.finished_, true))
  {
  }

  query_observer& operator=(query_observer&& other) noexcept
  {
    if (this != &other) {
      finish();
      // Release the old handle while its context is still alive.
      handle_.reset();
      context_  = std::move(other.context_);
      handle_   = std::exchange(other.handle_, std::nullopt);
      finished_ = std::exchange(other.finished_, true);
    }
    return *this;
  }

  [[nodiscard]] const std::shared_ptr<const telemetry_context>& shared_context() const noexcept
  {
    return context_;
  }
  [[nodiscard]] std::optional<quent::Uuid> uuid() const
  {
    if (!handle_) { return std::nullopt; }
    return handle_->id().raw();
  }

  void planning()
  {
    if (handle_ && !finished_) { handle_->planning(); }
  }
  void executing()
  {
    if (handle_ && !finished_) { handle_->executing(); }
  }

  /// At most one terminal emission attempt, including destructor backstops.
  void finish() noexcept
  {
    if (!handle_ || finished_) { return; }
    finished_ = true;
    try {
      handle_->finished();
    } catch (...) {
      sirius::log::log_noexcept([] { SIRIUS_LOG_ERROR("query telemetry finish failed"); });
    }
  }

 private:
  std::shared_ptr<const telemetry_context> context_;
  std::optional<quent::DynamicFsmHandle<quent::Query>> handle_;
  bool finished_ = false;
};

}  // namespace sirius::telemetry
