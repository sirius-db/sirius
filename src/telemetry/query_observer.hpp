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

#include "telemetry/query_activity.hpp"

#include <algorithm>
#include <mutex>
#include <stdexcept>
#include <utility>

namespace sirius::telemetry {

/// One query observer. The invocation owner controls completion;
/// consumers receive the UUID, never the mutable Quent FSM. Retains the exporter
/// through completion. Mutations are serialized for DuckDB execution workers; moves
/// require exclusive access. Requires a context even when exporting is disabled.
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

  ~query_observer() noexcept { complete("abandoned"); }
  query_observer(const query_observer&)            = delete;
  query_observer& operator=(const query_observer&) = delete;

  /// Transfer completion responsibility; the source may be destroyed or reassigned.
  query_observer(query_observer&& other) noexcept
    : metadata_(std::move(other.metadata_)),
      phase_(other.phase_),
      activities_(std::move(other.activities_)),
      context_(std::move(other.context_)),
      handle_(std::exchange(other.handle_, std::nullopt)),
      details_handle_(std::exchange(other.details_handle_, std::nullopt)),
      metadata_revision_(std::exchange(other.metadata_revision_, 0)),
      finished_(std::exchange(other.finished_, true))
  {
  }

  query_observer& operator=(query_observer&& other) noexcept
  {
    if (this != &other) {
      complete("abandoned");
      // Release the old handle while its context is still alive.
      handle_.reset();
      details_handle_.reset();
      context_           = std::move(other.context_);
      handle_            = std::exchange(other.handle_, std::nullopt);
      details_handle_    = std::exchange(other.details_handle_, std::nullopt);
      metadata_revision_ = std::exchange(other.metadata_revision_, 0);
      metadata_          = std::move(other.metadata_);
      phase_             = other.phase_;
      activities_        = std::move(other.activities_);
      finished_          = std::exchange(other.finished_, true);
    }
    return *this;
  }

  [[nodiscard]] const std::shared_ptr<const telemetry_context>& shared_context() const noexcept
  {
    return context_;
  }
  [[nodiscard]] std::optional<quent::Uuid> uuid() const
  {
    std::lock_guard lock(mutex_);
    if (!handle_) { return std::nullopt; }
    return handle_->id().raw();
  }

  /// No implicit current-query or current-parent state. Callers propagate parents.
  [[nodiscard]] query_activity activity(
    const char* phase,
    const char* kind                              = "work",
    std::optional<query_activity::id_type> parent = std::nullopt) noexcept
  {
    std::lock_guard lock(mutex_);
    if (!context_ || !context_->enabled() || !handle_ || finished_) return {};
    try {
      auto handle = context_->context().query_activity_observer()->handle();
      auto track  = context_->context().query_activity_track_observer()->handle();
      track.declaration({.activity_id = handle.id(), .label = phase});
      auto active = std::move(handle)
                      .started({.query_id            = handle_->id(),
                                .parent_activity_id  = parent,
                                .track               = {.target = track.id(), .data = {}},
                                .phase               = phase,
                                .kind                = kind,
                                .planning_generation = metadata_.planning_generation,
                                .execution_query_id  = metadata_.execution_query_id})
                      .into_dynamic();
      auto state = std::make_shared<query_activity::state>(context_, std::move(active));
      std::erase_if(activities_, [](auto const& weak) { return weak.expired(); });
      activities_.push_back(state);
      return query_activity(std::move(state));
    } catch (...) {
      report_failure();
      return {};
    }
  }

  /// Publish a complete snapshot without changing lifecycle state or query identity.
  /// Revisions belong to this owner and survive moves. Metadata may arrive after Finished.
  void update_metadata(quent::records::QueryMetadata metadata)
  {
    std::lock_guard lock(mutex_);
    metadata_ = std::move(metadata);
    publish_metadata();
  }

  void correlate(uint64_t connection, uint64_t ordinal) noexcept
  {
    observe([&] {
      metadata_.connection_id = connection;
      metadata_.query_ordinal = ordinal;
      metadata_.disposition   = "cpu_only";
      publish_metadata();
    });
  }

  void label(const std::optional<std::string>& name) noexcept
  {
    if (!name) return;
    observe([&] {
      metadata_.instance_name = name;
      publish_metadata();
    });
  }

  void planning_generation(uint64_t generation) noexcept
  {
    observe([&] {
      metadata_.planning_generation = generation;
      publish_metadata();
    });
  }

  /// An admitted GPU attempt, including failures before engine construction.
  void execution_window(sirius::query_id_t id) noexcept
  {
    observe([&] {
      metadata_.execution_query_id = sirius::value_of(id);
      metadata_.worker_id          = context_->worker_id();
      if (metadata_.disposition != "gpu_to_cpu") { metadata_.disposition = "gpu"; }
      publish_metadata();
    });
  }

  void cpu_replay() noexcept
  {
    observe([&] {
      metadata_.disposition = "gpu_to_cpu";
      publish_metadata();
    });
  }

  void planning() noexcept
  {
    observe([&] {
      if (phase_ == phase::initial) {
        phase_ = phase::planning;
        handle_->planning();
      }
    });
  }
  void executing() noexcept
  {
    observe([&] {
      if (phase_ != phase::executing) {
        phase_ = phase::executing;
        handle_->executing();
      }
    });
  }

  /// Complete after the outer query's final error (including CPU replay) is known.
  /// A null outcome means the callback cannot distinguish success from abandonment.
  void complete(const char* outcome) noexcept
  {
    std::lock_guard lock(mutex_);
    if (!handle_ || finished_) return;
    try {
      if (outcome) {
        metadata_.outcome = outcome;
      } else {
        metadata_.outcome.reset();
      }
      publish_metadata();
    } catch (...) {
      report_failure();
    }
    finish_activities(outcome);
    finish_locked();
  }

  /// At most one terminal emission attempt, including destructor backstops.
  void finish() noexcept
  {
    std::lock_guard lock(mutex_);
    finish_locked();
  }

 private:
  static void report_failure() noexcept
  {
    sirius::log::log_noexcept([] { SIRIUS_LOG_ERROR("query telemetry emission failed"); });
  }

  template <class F>
  void observe(F&& emit) noexcept
  {
    std::lock_guard lock(mutex_);
    if (!handle_ || finished_) return;
    try {
      emit();
    } catch (...) {
      report_failure();
    }
  }

  void publish_metadata()
  {
    if (!details_handle_) {
      details_handle_.emplace(context_->context().query_details_observer()->handle());
    }
    details_handle_->update(
      {.query_id = handle_->id(), .revision = ++metadata_revision_, .metadata = metadata_});
  }

  void finish_locked() noexcept
  {
    if (!handle_ || finished_) return;
    finish_activities("abandoned");
    finished_ = true;
    try {
      handle_->finished();
    } catch (...) {
      report_failure();
    }
  }

  void finish_activities(const char* outcome) noexcept
  {
    for (auto it = activities_.rbegin(); it != activities_.rend(); ++it) {
      if (auto active = it->lock()) active->finish(outcome);
    }
    activities_.clear();
  }

  enum class phase { initial, planning, executing };
  mutable std::mutex mutex_;
  quent::records::QueryMetadata metadata_;
  phase phase_ = phase::initial;
  std::vector<std::weak_ptr<query_activity::state>> activities_;
  std::shared_ptr<const telemetry_context> context_;
  std::optional<quent::DynamicFsmHandle<quent::Query>> handle_;
  std::optional<quent::Handle<quent::QueryDetails>> details_handle_;
  uint64_t metadata_revision_ = 0;
  bool finished_              = false;
};

}  // namespace sirius::telemetry
