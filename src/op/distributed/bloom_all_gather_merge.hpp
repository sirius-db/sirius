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

#include "exchange/fabric.hpp"
#include "op/dynamic_filter/cross_rank_filter_merge.hpp"
#include "op/dynamic_filter/dynamic_filter_publish_plan.hpp"

#include <chrono>
#include <cstddef>
#include <memory>

namespace cucascade::memory {
class memory_space;
}

namespace sirius::exchange {
class exchange_service;
}

namespace sirius::op {

/**
 * @brief Merges a join's Blooms across the ranks of an exchange, one all-gather per admitted key.
 *
 * After an exchange a rank's build holds only its share of the keys. Each rank builds a Bloom
 * per key, sized from its own rows (see merged_bloom_policy()), all-gathers it with a header (key
 * rep, family, scale, size), folds its own onto the smallest size any rank sent, and ORs the
 * others into it. A key whose filters cannot all be merged publishes nothing: a rank sent none, a
 * header's kind differs, a size does not fold onto the smallest, or a rank missed the timeout.
 */
class bloom_all_gather_merge final : public cross_rank_filter_merge {
 public:
  /// Shorter than all_gather's default: a late rank should cost the filter, not stall the build.
  static constexpr std::chrono::milliseconds default_timeout{std::chrono::minutes{2}};

  /**
   * @param first_stream Admitted key k merges on stream first_stream + k, the same on every rank.
   * @param key_count The join's admitted keys.
   * @param gpu Where this rank's contribution and the gathered filters are allocated.
   * @param timeout A rank that has not sent its filter by then is given up on.
   */
  bloom_all_gather_merge(std::shared_ptr<exchange::exchange_service> service,
                         exchange::stream_id first_stream,
                         std::size_t key_count,
                         cucascade::memory::memory_space& gpu,
                         std::chrono::milliseconds timeout = default_timeout);

  void merge(std::span<std::shared_ptr<sirius_dynamic_filter>> membership,
             ::cuda::stream_ref stream) override;
  void leave() noexcept override;

 private:
  /// Takes this rank's part, with nothing, in the merges of admitted keys [first, count).
  void leave_merges(std::size_t first, std::size_t count) noexcept;

  std::shared_ptr<exchange::exchange_service> _service;
  exchange::stream_id _first_stream;
  std::size_t _key_count;
  cucascade::memory::memory_space* _gpu;
  std::chrono::milliseconds _timeout;
};

/// The bounds of a merged Bloom's size in keys, until the streaming OR (#2375).
inline constexpr std::size_t merged_bloom_min_keys = std::size_t{1} << 16;
inline constexpr std::size_t merged_bloom_max_keys = std::size_t{1} << 30;

/**
 * @brief @p policy, changed so that every rank's filters merge: Blooms only, no zone maps, and no
 * skipping an empty build or drained targets, since the other ranks wait on this rank's part.
 *
 * Each rank sizes its Bloom for its own build rows times @p ranks (a hash-exchanged build holds
 * about one part in @p ranks of the keys), within [@p min_keys, @p max_keys], as a power-of-two
 * number of blocks, so that Blooms of different sizes fold onto the smallest. The planner's
 * estimate does not size it: TPC-H Q21 at SF10000 estimated 120 M keys for a build of about 1 B,
 * and its merged Bloom kept 98% of lineitem.
 *
 * max_bloom_bytes_per_gpu does not cut the size either: the merged Bloom holds every rank's keys,
 * and Q05 at SF3000 on 2 GPUs, cut to 256 MiB, got ~1.5 bits per key, filtered almost nothing and
 * ran 3.5x slower.
 */
[[nodiscard]] dynamic_filter_publication_policy merged_bloom_policy(
  dynamic_filter_publication_policy policy,
  std::size_t ranks,
  std::size_t min_keys = merged_bloom_min_keys,
  std::size_t max_keys = merged_bloom_max_keys) noexcept;

}  // namespace sirius::op
