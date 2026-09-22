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

#include "event/query_event_subscriber.hpp"

#include <cstdint>
#include <memory>
#include <stdexcept>

namespace sirius::test {

struct compressed_materialization_stats {
  uint64_t scan_columns_narrowed = 0;
  uint64_t scan_columns_restored = 0;
  uint64_t pin_columns_narrowed  = 0;
  /// Plan-time count of TABLE_SCAN nodes that received a narrow physical
  /// sidecar (post-residency-gate, pre-propagation/pruning — a later pass may
  /// still clear or prune it).
  uint64_t scan_sidecars_installed = 0;
  /// Runtime count of input-batch columns that crossed an engaged hash
  /// PARTITION with a carrier narrower than their native mapping. Derived
  /// from actual batch types, so a regression anywhere in the narrow-carrier
  /// chain drops it to zero.
  uint64_t partition_narrow_columns = 0;
  /// Plan-time count of narrow scan sidecar targets flipped back to native; the keep/retract rule
  /// is `apply_tier_narrowing_policy`'s.
  uint64_t scan_narrow_targets_retracted = 0;
};

/// Only tests aggregate these events. Construct before the operation being measured.
/// snapshot() must run between operations, with all measured reporters quiescent:
/// it drains the old subscription and installs a fresh one for the next operation.
class compressed_materialization_recorder {
 public:
  explicit compressed_materialization_recorder(event::query_event_publisher& publisher)
    : _publisher(publisher.weak_from_this()),
      _subscriber(std::make_unique<subscriber>(publisher, _stats))
  {
  }

  compressed_materialization_recorder(compressed_materialization_recorder const&) = delete;
  compressed_materialization_recorder& operator=(compressed_materialization_recorder const&) =
    delete;

  compressed_materialization_stats snapshot()
  {
    auto publisher = _publisher.lock();
    if (!publisher || !_subscriber->is_subscribed()) {
      throw std::runtime_error("Compressed-materialization observer is stopped");
    }
    _subscriber->stop();
    if (_subscriber->failed()) {
      throw std::runtime_error("Compressed-materialization event delivery incomplete");
    }
    auto const result = _stats;
    _subscriber       = std::make_unique<subscriber>(*publisher, _stats);
    if (!_subscriber->is_subscribed()) {
      throw std::runtime_error("Compressed-materialization observer is stopped");
    }
    return result;
  }

 private:
  class subscriber : public event::query_event_subscriber {
   public:
    subscriber(event::query_event_publisher& publisher, compressed_materialization_stats& stats)
      : query_event_subscriber(publisher,
                               {event::event_type::compressed_materialization},
                               /*drain_on_stop=*/true),
        _stats(stats)
    {
      start();
    }
    ~subscriber() override { stop(); }
    std::string_view name() const noexcept override { return "compressed-materialization-test"; }
    bool failed() const noexcept { return delivery_failed(); }

    void on_compressed_materialization(event::event_id_t,
                                       event::timestamp_t,
                                       event::compressed_materialization_activity activity,
                                       std::uint64_t count) noexcept override
    {
      using enum event::compressed_materialization_activity;
      switch (activity) {
        case scan_columns_narrowed: _stats.scan_columns_narrowed += count; break;
        case scan_columns_restored: _stats.scan_columns_restored += count; break;
        case pin_columns_narrowed: _stats.pin_columns_narrowed += count; break;
        case scan_sidecar_installed: _stats.scan_sidecars_installed += count; break;
        case partition_narrow_columns: _stats.partition_narrow_columns += count; break;
        case scan_narrow_targets_retracted: _stats.scan_narrow_targets_retracted += count; break;
      }
    }

   private:
    compressed_materialization_stats& _stats;
  };

  std::weak_ptr<event::query_event_publisher> _publisher;
  compressed_materialization_stats _stats;
  // Destroyed first, so its worker joins before the counters go away.
  std::unique_ptr<subscriber> _subscriber;
};

}  // namespace sirius::test
