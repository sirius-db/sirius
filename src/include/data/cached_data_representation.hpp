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

// cucascade
#include "host_parquet_representation.hpp"

#include <cucascade/data/common.hpp>
#include <cucascade/data/cpu_data_representation.hpp>
#include <cucascade/memory/memory_space.hpp>

// standard library
#include <concepts>
#include <memory>
#include <type_traits>
#include <vector>

namespace sirius {

template <typename T>
concept data_representation = std::derived_from<T, cucascade::idata_representation>;

template <data_representation T>
class cached_shared_representation : public cucascade::idata_representation {
 public:
  cached_shared_representation(std::unique_ptr<T> representation)
    : idata_representation(representation->get_memory_space()),
      _representation(std::move(representation))
  {
  }

  /**
   * @brief Deep copies the host_parquet_representation.
   *
   * @param[in] stream CUDA stream for memory operations
   * @return A unique pointer to the cloned host_parquet_representation.
   */
  std::unique_ptr<idata_representation> clone(rmm::cuda_stream_view stream) override
  {
    auto cloned                   = _representation->clone(stream);
    std::unique_ptr<T> cloned_ptr = std::unique_ptr<T>(static_cast<T*>(cloned.release()));
    return std::make_unique<cached_shared_representation>(std::move(cloned_ptr));
  }

  /**
   * @brief Shallow copies the host_parquet_representation.
   *
   * @return A unique pointer to the shallow cloned host_parquet_representation.
   */
  std::unique_ptr<cucascade::idata_representation> shallow_clone()
  {
    return std::unique_ptr<cached_shared_representation>(
      new cached_shared_representation(get_memory_space(), _representation));
  }

  /**
   * @brief Gets the size of the representation in bytes (compressed in the multiple blocks
   * allocation).
   *
   * @return The size of the representation in bytes.
   */
  [[nodiscard]] std::size_t get_size_in_bytes() const override
  {
    return _representation->get_size_in_bytes();
  }

  /**
   * @brief Gets the underlying representation.
   *
   * @return A shared_ptr to the underlying representation.
   */
  [[nodiscard]] std::shared_ptr<T> get_representation() { return _representation; }

 private:
  cached_shared_representation(cucascade::memory::memory_space& memory_space,
                               std::shared_ptr<T> representation)
    : idata_representation(memory_space), _representation(std::move(representation))
  {
  }

  std::shared_ptr<T> _representation;
};

using cached_host_parquet_representation =
  cached_shared_representation<host_parquet_representation>;
using cached_host_data_representation =
  cached_shared_representation<cucascade::host_data_representation>;

}  // namespace sirius
