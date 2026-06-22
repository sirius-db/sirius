/*
 * Copyright 2025, Sirius Contributors.
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

#include "data/convertible_data.hpp"
#include "data/sirius_converter_registry.hpp"
#include "log/logging.hpp"

#include <rmm/cuda_stream_view.hpp>

#include <cucascade/data/cpu_data_representation.hpp>
#include <cucascade/data/data_batch.hpp>
#include <cucascade/data/data_repository.hpp>
#include <cucascade/data/disk_data_representation.hpp>
#include <cucascade/data/gpu_data_representation.hpp>
#include <cucascade/memory/common.hpp>
#include <cucascade/memory/memory_reservation.hpp>
#include <cucascade/memory/memory_reservation_manager.hpp>
#include <cucascade/memory/memory_space.hpp>
#include <memory/sirius_memory_reservation_manager.hpp>

#include <cstddef>
#include <memory>
#include <optional>
#include <vector>

namespace sirius {

/**
 * @brief Concrete convertible_data wrapping a cucascade::data_batch.
 *
 * Implements the RAII mutable lock / convert / auto-release pattern for a single
 * data_batch. Acquires mutable_data_batch via get_mutable() (blocking) or
 * try_get_mutable() (non-blocking), calls convert_to on the accessor, and relies on
 * the mutable_data_batch destructor to restore the batch to idle on all exit paths
 * (success, failure, exception). No manual state save/restore is needed.
 */
class convertible_data_batch : public convertible_data {
 public:
  /**
   * @brief Construct from a shared_ptr to a cucascade data_batch.
   * @param batch The data batch to wrap (shared ownership retained).
   */
  explicit convertible_data_batch(std::shared_ptr<cucascade::data_batch> batch)
    : _batch(std::move(batch))
  {
  }

  /**
   * @brief Convert this batch to reside in one of the target memory spaces.
   *
   * Acquires an exclusive (mutable) lock on the batch -- blocking if blocking=true,
   * non-blocking if blocking=false. Iterates target_spaces in order, requesting a
   * reservation in each via the reservation manager. On the first successful reservation
   * the batch is converted via convert_to on the mutable accessor. The mutable_data_batch
   * RAII destructor automatically restores the batch to idle on all exit paths.
   *
   * @param target_spaces  Candidate memory spaces to convert into (tried in order).
   * @param stream         CUDA stream for asynchronous memory operations.
   * @param res_mgr        Reservation manager for acquiring memory in the target space.
   * @param blocking       When true, uses get_mutable() (blocks until exclusive lock acquired).
   *                       When false, uses try_get_mutable() (returns nullopt immediately if
   *                       the lock is unavailable).
   * @return A vector of bytes converted per target space index on success, or nullopt if
   *         no conversion occurred.
   */
  std::optional<std::vector<std::size_t>> convert(
    const std::vector<const cucascade::memory::memory_space*>& target_spaces,
    rmm::cuda_stream_view stream,
    sirius::memory::sirius_memory_reservation_manager& res_mgr,
    bool blocking) override
  {
    std::optional<cucascade::mutable_data_batch> mut_opt;
    if (blocking) {
      mut_opt.emplace(_batch->get_mutable());
    } else {
      mut_opt = _batch->try_get_mutable();
      if (!mut_opt) { return std::nullopt; }
    }
    auto& mut = *mut_opt;

    // Check if the batch is already in the target space
    auto cur_space = mut->get_memory_space();
    for (std::size_t idx = 0; idx < target_spaces.size(); ++idx) {
      const auto* space = target_spaces[idx];
      if (cur_space == space) { return std::nullopt; }
    }

    auto data_size = mut->get_data()->get_size_in_bytes();

    for (std::size_t idx = 0; idx < target_spaces.size(); ++idx) {
      const auto* space = target_spaces[idx];
      auto* mem_space   = res_mgr.get_memory_space(space->get_tier(), space->get_id().device_id);
      if (!mem_space) { continue; }

      // Non-blocking reservation
      auto reservation = mem_space->make_reservation_or_null(data_size);
      if (!reservation) { continue; }

      // When downgrading off the GPU, rebind the source buffers' deallocation stream to this
      // downgrade stream so that when the conversion below frees the GPU representation, the
      // free lands on the active (downgrade) stream rather than the stream the data was
      // originally produced on. We hold the exclusive (mutable) lock here, and convert_to()
      // synchronizes `stream` after the D2H copy and before destroying the source
      // representation, so the free is correctly ordered. No-op for non-GPU-table sources.
      if (cur_space != nullptr && cur_space->get_tier() == cucascade::memory::Tier::GPU) {
        mut->rebind_stream(stream);
      }

      auto& converter_registry = sirius::converter_registry::get();

      switch (space->get_tier()) {
        case cucascade::memory::Tier::GPU:
          mut->convert_to<cucascade::gpu_table_representation>(
            converter_registry, mem_space, stream);
          break;
        case cucascade::memory::Tier::HOST:
          mut->convert_to<cucascade::host_data_representation>(
            converter_registry, mem_space, stream);
          break;
        case cucascade::memory::Tier::DISK:
          mut->convert_to<cucascade::disk_data_representation>(
            converter_registry, mem_space, stream);
          break;
        default: continue;
      }

      // RAII: mutable_data_batch destructor releases the mutable lock and transitions back to idle
      // automatically.
      std::vector<std::size_t> bytes_per_target(target_spaces.size(), 0);
      bytes_per_target[idx] = data_size;
      return bytes_per_target;
    }
    return std::nullopt;
  }

  /**
   * @brief Get the size in bytes of this batch in the specified memory space.
   *
   * Acquires a shared (read-only) lock to access the memory space pointer, then
   * compares with the provided space. Returns the batch size in bytes if the batch
   * resides in the given space, 0 otherwise.
   *
   * @param space The memory space to query.
   * @return The batch size in bytes if the batch resides in the given space, 0 otherwise.
   */
  std::size_t bytes_in_space(cucascade::memory::memory_space* space) const override
  {
    auto ro = _batch->get_read_only();
    if (ro->get_memory_space() == space) { return ro->get_data()->get_size_in_bytes(); }
    return 0;
  }

 private:
  std::shared_ptr<cucascade::data_batch> _batch;
};

/**
 * @brief Concrete convertible_data_provider wrapping a cucascade::shared_data_repository.
 *
 * Iterates partitions and batches within a shared_data_repository, filtering by idle
 * state and matching memory_space. The default iteration order is last-to-first
 * (back-to-front) for both partitions and batches, matching the downgrade eviction
 * pattern of preferring the most recently added data.
 * Only batches in batch_state::idle are considered.
 */
class convertible_data_batch_provider : public convertible_data_provider {
 public:
  /**
   * @brief Construct from a raw pointer to a shared_data_repository.
   * @param repo The repository to iterate (non-owning; caller ensures lifetime).
   */
  explicit convertible_data_batch_provider(cucascade::shared_data_repository* repo) : _repo(repo) {}

  /**
   * @brief Get the next convertible batch matching the given memory space.
   *
   * Iterates partitions and batches. When front_to_back is false (the typical
   * downgrade use case), iterates from last partition to first, and within each
   * partition from last batch to first. Returns the first batch that is idle and
   * resides in the requested memory space.
   *
   * @param space           The memory space to filter by.
   * @param front_to_back   Iteration direction.
   * @return A convertible_data_batch wrapping the matching batch, or nullptr.
   */
  std::unique_ptr<convertible_data> get_next_convertible(cucascade::memory::memory_space* space,
                                                         bool front_to_back) override
  {
    auto num_parts = _repo->num_partitions();
    if (num_parts == 0) { return nullptr; }

    if (front_to_back) {
      for (std::size_t p = 0; p < num_parts; ++p) {
        auto batch_ids = _repo->get_batch_ids(p);
        for (std::size_t i = 0; i < batch_ids.size(); ++i) {
          auto result = try_get_batch(batch_ids[i], p, space);
          if (result) { return result; }
        }
      }
    } else {
      for (std::size_t p = num_parts; p > 0; --p) {
        auto batch_ids = _repo->get_batch_ids(p - 1);
        for (std::size_t i = batch_ids.size(); i > 0; --i) {
          auto result = try_get_batch(batch_ids[i - 1], p - 1, space);
          if (result) { return result; }
        }
      }
    }

    return nullptr;
  }

  /**
   * @brief Get all convertible batches matching the given memory space.
   *
   * Same iteration order as get_next_convertible but collects all matching batches.
   *
   * @param space           The memory space to filter by.
   * @param front_to_back   Iteration direction.
   * @return A vector of convertible_data_batch instances (may be empty).
   */
  std::vector<std::unique_ptr<convertible_data>> get_all_convertible(
    cucascade::memory::memory_space* space, bool front_to_back) override
  {
    std::vector<std::unique_ptr<convertible_data>> results;
    auto num_parts = _repo->num_partitions();
    if (num_parts == 0) { return results; }

    if (front_to_back) {
      for (std::size_t p = 0; p < num_parts; ++p) {
        auto batch_ids = _repo->get_batch_ids(p);
        for (std::size_t i = 0; i < batch_ids.size(); ++i) {
          auto result = try_get_batch(batch_ids[i], p, space);
          if (result) { results.push_back(std::move(result)); }
        }
      }
    } else {
      for (std::size_t p = num_parts; p > 0; --p) {
        auto batch_ids = _repo->get_batch_ids(p - 1);
        for (std::size_t i = batch_ids.size(); i > 0; --i) {
          auto result = try_get_batch(batch_ids[i - 1], p - 1, space);
          if (result) { results.push_back(std::move(result)); }
        }
      }
    }

    return results;
  }

  /**
   * @brief Get the total byte size of all batches in the given memory space.
   *
   * Iterates all partitions front-to-back, summing bytes for batches residing
   * in the specified space. Accesses batch data via get_read_only() as required
   * by the new cucascade API.
   *
   * @param space The memory space to query.
   * @return Total size in bytes.
   */
  std::size_t get_bytes_in_space(cucascade::memory::memory_space* space) const
  {
    std::size_t total = 0;
    auto num_parts    = _repo->num_partitions();

    for (std::size_t p = 0; p < num_parts; ++p) {
      auto batch_ids = _repo->get_batch_ids(p);
      for (auto batch_id : batch_ids) {
        auto batch = _repo->get_data_batch_by_id(batch_id, p);
        if (!batch) { continue; }
        auto ro = batch->get_read_only();
        if (ro->get_memory_space() == space) { total += ro->get_data()->get_size_in_bytes(); }
      }
    }

    return total;
  }

 private:
  /**
   * @brief Try to get a matching batch by id, checking idle state and memory space.
   *
   * Only considers batches in batch_state::idle. Accesses the memory space via
   * get_read_only() as required by the new cucascade API (get_memory_space() is
   * private on idle data_batch).
   *
   * @param batch_id       The batch ID to retrieve.
   * @param partition_idx  The partition containing the batch.
   * @param space          The target memory space to match.
   * @return A convertible_data_batch if the batch matches, nullptr otherwise.
   */
  std::unique_ptr<convertible_data> try_get_batch(uint64_t batch_id,
                                                  std::size_t partition_idx,
                                                  cucascade::memory::memory_space* space) const
  {
    auto batch = _repo->get_data_batch_by_id(batch_id, partition_idx);
    if (!batch) { return nullptr; }

    if (batch->get_state() != cucascade::batch_state::idle) { return nullptr; }

    auto ro = batch->try_get_read_only();
    if (!ro) { return nullptr; }
    if ((*ro)->get_memory_space() == space) {
      return std::make_unique<convertible_data_batch>(std::move(batch));
    }

    return nullptr;
  }

  cucascade::shared_data_repository* _repo;
};

}  // namespace sirius
