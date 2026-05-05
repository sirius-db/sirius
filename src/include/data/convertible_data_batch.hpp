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
 * Generalizes the pattern for converting a data_batch to a target memory space
 * via the post-#117 RAII accessor model. Acquires an exclusive accessor on the
 * batch (which represents the equivalent of the pre-#117 in-transit lock),
 * iterates target memory spaces requesting reservations, and converts via the
 * converter registry. The exclusive lock is released automatically when the
 * accessor goes out of scope (success, failure, or exception).
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
   * Iterates target_spaces in order, requesting a reservation in each via
   * specific_memory_space. On the first successful reservation + conversion the batch
   * is moved to the new tier and a vector of bytes per target space is returned.
   * On failure the batch retains its original representation.
   *
   * Lock semantics (post-#117): a non-blocking exclusive accessor is taken via
   * try_to_mutable(). If the batch is currently locked by another consumer
   * (read_only or mutable), this call returns std::nullopt without converting.
   * On success, the exclusive lock is held for the duration of the conversion
   * and released when the local accessor goes out of scope (RAII).
   *
   * @param target_spaces  Candidate memory spaces to convert into (tried in order).
   * @param stream         CUDA stream for asynchronous memory operations.
   * @param res_mgr        Reservation manager for acquiring memory in the target space.
   * @return A vector of bytes converted per target space index on success, or nullopt if
   *         no conversion occurred.
   */
  std::optional<std::vector<std::size_t>> convert(
    const std::vector<const cucascade::memory::memory_space*>& target_spaces,
    rmm::cuda_stream_view stream,
    sirius::memory::sirius_memory_reservation_manager& res_mgr) override
  {
    if (!_batch) { return std::nullopt; }

    // Non-blocking exclusive lock: equivalent to the pre-#117 transit-lock
    // gate. If another consumer (read_only_data_batch or mutable_data_batch) is
    // currently holding the batch, skip conversion. RAII destruction of `mut`
    // releases the exclusive lock on every exit path (success, failure, exception).
    auto mut_opt = _batch->try_to_mutable();
    if (!mut_opt) { return std::nullopt; }
    auto& mut = *mut_opt;

    auto* data = mut.get_data();
    if (!data) { return std::nullopt; }
    auto data_size = data->get_size_in_bytes();

    for (std::size_t idx = 0; idx < target_spaces.size(); ++idx) {
      const auto* space = target_spaces[idx];
      auto* mem_space   = res_mgr.get_memory_space(space->get_tier(), space->get_id().device_id);
      if (!mem_space) { continue; }

      // Non-blocking reservation
      auto reservation = mem_space->make_reservation_or_null(data_size);

      if (!reservation) { continue; }

      auto& converter_registry = sirius::converter_registry::get();

      switch (space->get_tier()) {
        case cucascade::memory::Tier::GPU:
          mut.convert_to<cucascade::gpu_table_representation>(
            converter_registry, mem_space, stream);
          break;
        case cucascade::memory::Tier::HOST:
          mut.convert_to<cucascade::host_data_representation>(
            converter_registry, mem_space, stream);
          break;
        case cucascade::memory::Tier::DISK:
          mut.convert_to<cucascade::disk_data_representation>(
            converter_registry, mem_space, stream);
          break;
        default: continue;
      }

      std::vector<std::size_t> bytes_per_target(target_spaces.size(), 0);
      bytes_per_target[idx] = data_size;
      return bytes_per_target;
      // mut destroyed here → exclusive lock released
    }

    // No target space succeeded; mut destroyed → exclusive lock released
    return std::nullopt;
  }

  /**
   * @brief Get the size in bytes of this batch in the specified memory space.
   *
   * @param space The memory space to query.
   * @return The batch size in bytes if the batch resides in the given space, 0 otherwise.
   */
  std::size_t bytes_in_space(cucascade::memory::memory_space* space) const override
  {
    if (!_batch) { return 0; }
    // R2 read-only accessor scoped to this single read.
    auto ro = _batch->to_read_only();
    if (ro.get_memory_space() == space && ro.get_data() != nullptr) {
      return ro.get_data()->get_size_in_bytes();
    }
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
 *
 * Note (post-#117): "idle" here means batch_state::idle — i.e. no reader or writer
 * accessor is currently held. A locked batch is not eligible for downgrade until
 * its accessor is released; downstream code path handles polling.
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
   * in the specified space. Each access is mediated through a scoped read-only
   * accessor (R2): the shared lock is held only for the duration of the size
   * read on that single batch.
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
        auto ro = batch->to_read_only();
        if (ro.get_memory_space() == space && ro.get_data() != nullptr) {
          total += ro.get_data()->get_size_in_bytes();
        }
        // ro destroyed → shared lock released
      }
    }

    return total;
  }

 private:
  /**
   * @brief Try to get a matching batch by id, checking idle state and memory space.
   *
   * Lock-free state probe via data_batch::get_state() (lock-free public API);
   * memory-space check via a scoped read-only accessor (R2). The accessor is
   * dropped before constructing the convertible_data_batch wrapper so the
   * downstream convert() call can take its own exclusive accessor without
   * contending against this transient shared lock.
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

    // Lock-free state probe.
    if (batch->get_state() != cucascade::batch_state::idle) { return nullptr; }

    // Scoped read-only accessor for memory_space probe.
    {
      auto ro = batch->to_read_only();
      if (ro.get_memory_space() != space) { return nullptr; }
    }  // ro released here before constructing the wrapper

    return std::make_unique<convertible_data_batch>(std::move(batch));
  }

  cucascade::shared_data_repository* _repo;
};

}  // namespace sirius
