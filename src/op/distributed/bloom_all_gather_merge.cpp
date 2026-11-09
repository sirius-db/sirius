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

#include "op/distributed/bloom_all_gather_merge.hpp"

#include "cudf/cudf_utils.hpp"
#include "data/data_batch_utils.hpp"
#include "exchange/all_gather.hpp"
#include "exchange/exchange_service.hpp"
#include "log/logging.hpp"
#include "op/dynamic_filter/sirius_dynamic_filter.hpp"

#include <cudf/column/column_factories.hpp>
#include <cudf/table/table.hpp>

#include <rmm/detail/error.hpp>

#include <cucascade/data/data_batch.hpp>
#include <cucascade/memory/memory_space.hpp>

#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <exception>
#include <utility>
#include <vector>

namespace sirius::op {

bloom_all_gather_merge::bloom_all_gather_merge(std::shared_ptr<exchange::exchange_service> service,
                                               exchange::stream_id first_stream,
                                               std::size_t key_count,
                                               cucascade::memory::memory_space& gpu,
                                               std::chrono::milliseconds timeout)
  : _service(std::move(service)),
    _first_stream(first_stream),
    _key_count(key_count),
    _gpu(&gpu),
    _timeout(timeout)
{
}

void bloom_all_gather_merge::merge(std::span<std::shared_ptr<sirius_dynamic_filter>> membership,
                                   ::cuda::stream_ref stream)
{
  // A rank's contribution: rep | family | scale | storage bytes | the Bloom's words, as UINT32.
  constexpr std::size_t header = 4;
  auto const mr                = _gpu->get_default_allocator();
  // The peers wait on this rank in every key's gather, so however this ends, this rank takes part
  // in the gathers it has not entered. Keys from a failed one on publish nothing: this rank's Bloom
  // alone would drop rows another rank's build matches.
  std::size_t entered = 0;
  auto const give_up  = [&](std::size_t from) {
    for (auto j = from; j < membership.size(); ++j) {
      membership[j].reset();
    }
    leave_merges(entered, membership.size());
  };
  try {
    for (std::size_t k = 0; k < membership.size(); ++k) {
      auto* bloom = dynamic_cast<sirius_dynamic_bloom_filter*>(membership[k].get());
      std::array<std::uint32_t, header> head{};
      std::shared_ptr<cucascade::data_batch> mine;
      if (bloom) {
        auto const bytes = bloom->storage_bytes();
        head             = {static_cast<std::uint32_t>(bloom->domain().rep),
                            static_cast<std::uint32_t>(bloom->domain().family),
                            static_cast<std::uint32_t>(bloom->domain().scale),
                            static_cast<std::uint32_t>(bytes)};
        auto column      = cudf::make_numeric_column(
          cudf::data_type{cudf::type_id::UINT32},
          static_cast<cudf::size_type>(header + bytes / sizeof(std::uint32_t)),
          cudf::mask_state::UNALLOCATED,
          stream,
          mr);
        auto* words = column->mutable_view().data<std::uint32_t>();
        RMM_CUDA_TRY(
          cudaMemcpyAsync(words, head.data(), sizeof(head), cudaMemcpyHostToDevice, stream.get()));
        RMM_CUDA_TRY(cudaMemcpyAsync(
          words + header, bloom->storage(), bytes, cudaMemcpyDeviceToDevice, stream.get()));
        std::vector<std::unique_ptr<cudf::column>> columns;
        columns.push_back(std::move(column));
        mine = sirius::make_data_batch(
          std::make_unique<cudf::table>(std::move(columns)), *_gpu, stream, {});
      }
      std::vector<std::shared_ptr<cucascade::data_batch>> all;
      try {
        // all_gather sends and closes to every rank before it waits: once called, this rank has
        // taken its part in key k, even if it throws.
        entered = k + 1;
        all     = exchange::all_gather(
          *_service, _first_stream + k, std::move(mine), *_gpu, stream.get(), _timeout);
      } catch (std::exception const& e) {
        SIRIUS_LOG_WARN(
          "[publish_dynamic_filters] key {}: filters not merged across ranks: {}", k, e.what());
        // A rank missing from this gather is missing from the later ones too.
        give_up(k);
        return;
      }
      if (!bloom) { continue; }
      // Each rank sized its Bloom from its own rows (scaled_bloom_keys), so the sizes may differ
      // by powers of two; every rank folds them all onto the smallest, the same on every rank
      // since every rank reads every header.
      std::vector<std::pair<std::uint32_t const*, std::size_t>> others;
      std::size_t smallest = head[3];
      bool mergeable =
        std::all_of(all.begin(), all.end(), [](auto const& t) { return t != nullptr; });
      if (mergeable) {
        // Every rank's header in one copy-back, so the key costs one host sync, not one per rank.
        std::vector<std::array<std::uint32_t, header>> heads(all.size());
        for (exchange::rank_t r = 0; r < all.size(); ++r) {
          auto const* theirs = sirius::get_cudf_table_view(*all[r]).column(0).data<std::uint32_t>();
          RMM_CUDA_TRY(cudaMemcpyAsync(
            heads[r].data(), theirs, sizeof(heads[r]), cudaMemcpyDeviceToHost, stream.get()));
        }
        stream.sync();
        for (auto const& h : heads) {
          smallest = std::min<std::size_t>(smallest, h[3]);
        }
        auto const foldable = [&](std::size_t bytes) {
          return smallest != 0 && bytes % smallest == 0 && std::has_single_bit(bytes / smallest);
        };
        mergeable = std::all_of(heads.begin(), heads.end(), [&](auto const& h) {
          return h[0] == head[0] && h[1] == head[1] && h[2] == head[2] && foldable(h[3]);
        });
        for (exchange::rank_t r = 0; r < all.size(); ++r) {
          if (r == _service->rank()) { continue; }
          others.emplace_back(
            sirius::get_cudf_table_view(*all[r]).column(0).data<std::uint32_t>() + header,
            heads[r][3]);
        }
      }
      if (!mergeable) {
        SIRIUS_LOG_DEBUG(
          "[publish_dynamic_filters] key {}: a rank sent no filter, another kind, or a size that "
          "does not fold; nothing published for it.",
          k);
        membership[k].reset();
        continue;
      }
      try {
        if (bloom->storage_bytes() != smallest) {
          SIRIUS_LOG_INFO(
            "[publish_dynamic_filters] key {}: folding this rank's {}-byte Bloom onto the "
            "smallest rank's {} bytes.",
            k,
            bloom->storage_bytes(),
            smallest);
          auto folded   = bloom->folded_to(smallest, stream, mr);
          bloom         = folded.get();
          membership[k] = std::move(folded);
        }
        for (auto const& [words, bytes] : others) {
          bloom->merge_from(words, bytes, stream);
        }
      } catch (...) {
        // `all` and this Bloom must outlive every OR already queued on `stream`.
        auto error = std::current_exception();
        stream.sync();
        std::rethrow_exception(error);
      }
      // The gathered filters are freed with `all`, after the merges have read them.
      stream.sync();
    }
  } catch (...) {
    give_up(0);
    throw;
  }
}

void bloom_all_gather_merge::leave() noexcept { leave_merges(0, _key_count); }

void bloom_all_gather_merge::leave_merges(std::size_t first, std::size_t count) noexcept
{
  try {
    for (auto k = first; k < count; ++k) {
      exchange::leave_all_gather(*_service, _first_stream + k);
    }
  } catch (std::exception const& e) {
    SIRIUS_LOG_WARN("[publish_dynamic_filters] taking part in the merge without filters: {}",
                    e.what());
  }
}

dynamic_filter_publication_policy merged_bloom_policy(dynamic_filter_publication_policy policy,
                                                      std::size_t ranks,
                                                      std::size_t min_keys,
                                                      std::size_t max_keys) noexcept
{
  policy.emit_zone_map_filters         = false;
  policy.bloom_only                    = true;
  policy.bloom_rows_scale              = std::max<std::size_t>(ranks, 1);
  policy.scaled_bloom_min_keys         = min_keys;
  policy.scaled_bloom_max_keys         = max_keys;
  policy.publish_when_empty_or_drained = true;
  return policy;
}

}  // namespace sirius::op
