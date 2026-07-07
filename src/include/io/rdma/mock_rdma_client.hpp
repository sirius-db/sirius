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

#include "io/rdma/rdma_client.hpp"

#include <condition_variable>
#include <cstdint>
#include <map>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

namespace sirius::io::rdma {

/**
 * @brief In-memory @c rdma_client test double.
 *
 * Serves seeded objects, delivering into host or device destinations.  Test
 * controls: a gate that blocks @c get while closed (backpressure / drain
 * scenarios), fault injection (thrown transport errors, short reads), and
 * concurrency observability.  Thread-safe.
 */
class mock_rdma_client final : public rdma_client {
 public:
  void put_object(std::string bucket, std::string key, std::vector<std::uint8_t> bytes);

  /// Subsequent get() calls throw std::runtime_error(message) until cleared.
  void fail_gets(std::string message);
  /// The next @p count get() calls throw std::runtime_error(message); later
  /// calls behave normally.  A persistent fail_gets wins if both are set.
  void fail_next_gets(size_t count, std::string message);
  /// Subsequent get() calls deliver at most @p bytes until cleared.
  void short_read(size_t bytes);
  void clear_fault();

  /// While closed, get() blocks (after being counted) until open_gate().
  void close_gate();
  void open_gate();

  [[nodiscard]] size_t gets_issued() const noexcept;
  [[nodiscard]] size_t peak_concurrent_gets() const noexcept;

  size_t head(std::string_view bucket, std::string_view key) override;
  size_t get(
    std::string_view bucket, std::string_view key, size_t offset, size_t size, void* dst) override;

 private:
  mutable std::mutex _mtx;
  std::condition_variable _gate_cv;
  std::map<std::pair<std::string, std::string>, std::vector<std::uint8_t>> _objects;
  std::optional<std::string> _fail_message;
  size_t _fail_next_count{0};
  std::string _fail_next_message;
  std::optional<size_t> _short_read;
  bool _gate_closed{false};
  size_t _gets_issued{0};
  size_t _concurrent{0};
  size_t _peak_concurrent{0};
};

}  // namespace sirius::io::rdma
