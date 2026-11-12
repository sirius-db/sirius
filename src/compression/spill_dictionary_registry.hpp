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

#include "compression/spill_compression.hpp"

#include <operators/onpair_representation.hpp>

#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <thread>
#include <tuple>

namespace sirius::compression {

/// One column of a query-graph edge.
struct spill_column_key {
  spill_source source;
  std::size_t column{0};

  friend bool operator<(spill_column_key const& a, spill_column_key const& b)
  {
    return std::tuple(value_of(a.source.query), a.source.repository, a.column) <
           std::tuple(value_of(b.source.query), b.source.repository, b.column);
  }
};

/**
 * @brief The OnPair dictionary each spilled string column encodes with, trained once per
 * column in the background.
 *
 * A column's first spilled batch takes its default plan and hands a host sample of its
 * characters to a training thread; later batches encode with the dictionary once it is
 * ready, without probing whether it fits. Whether it still does shows in the ratio:
 * record_ratio() takes the first batch's ratio as the column's baseline and drops the
 * dictionary when a batch falls well below it, so the next batch trains a new one.
 *
 * Entries live until their query ends (erase_query()); the repository pointer in the key
 * is only unique within its query.
 */
class spill_dictionary_registry {
 public:
  static spill_dictionary_registry& global();

  spill_dictionary_registry() = default;
  ~spill_dictionary_registry();
  spill_dictionary_registry(spill_dictionary_registry const&)            = delete;
  spill_dictionary_registry& operator=(spill_dictionary_registry const&) = delete;

  /// The column's dictionary, once trained.
  [[nodiscard]] std::shared_ptr<simpatico::onpair_dictionary const> find(
    spill_column_key const& key) const;

  /// Whether the column has no dictionary and none is training, so a sample is wanted.
  [[nodiscard]] bool needs_training(spill_column_key const& key) const;

  /// Queue training a dictionary for the column on @p sample, unless the column has one
  /// or one is training. Returns whether it was queued.
  bool train_async(spill_column_key const& key,
                   simpatico::onpair_training_sample sample,
                   simpatico::onpair_encoder_config const& config);

  /**
   * @brief Record the ratio a batch of the column reached with @p dict.
   *
   * The first ratio recorded for a dictionary is its baseline. Returns false, and drops
   * the dictionary so the column trains anew, when @p ratio is below @p min_fraction of
   * the baseline: the batch's encoding is then to be discarded. A ratio for a dictionary
   * the column no longer holds is ignored.
   */
  bool record_ratio(spill_column_key const& key,
                    std::shared_ptr<simpatico::onpair_dictionary const> const& dict,
                    double ratio,
                    double min_fraction);

  /// Drop every entry of @p query, including training still queued for it.
  void erase_query(query_id_t query);

  /// Block until no training is queued or running.
  void wait_idle();

  /// Entries, ready or training.
  [[nodiscard]] std::size_t size() const;

 private:
  struct entry {
    std::uint64_t generation{0};
    std::shared_ptr<simpatico::onpair_dictionary const> dict;  ///< Null while training.
    double baseline_ratio{0.0};                                ///< 0 until the first batch.
    bool failed{false};  ///< Training threw; the column keeps its default plan.
  };
  struct job {
    spill_column_key key;
    std::uint64_t generation;
    simpatico::onpair_training_sample sample;
    simpatico::onpair_encoder_config config;
  };

  void run();

  mutable std::mutex _mutex;
  std::condition_variable _wake;
  std::condition_variable _idle;
  std::map<spill_column_key, entry> _entries;
  std::deque<job> _jobs;
  bool _running_job{false};
  bool _stop{false};
  std::uint64_t _next_generation{1};
  std::thread _worker;
};

}  // namespace sirius::compression
