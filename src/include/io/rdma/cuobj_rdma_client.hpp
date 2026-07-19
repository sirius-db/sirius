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

#include "io/rdma/rdma_transport_client.hpp"
#include "io/s3/s3_request_authorizer.hpp"

#include <atomic>
#include <memory>
#include <mutex>
#include <string>

namespace sirius::io::rdma {

/**
 * @brief Production control plane: SigV4-signed HTTP over libcurl.
 *
 * One persistent easy handle per client — connection reuse is client state,
 * so back-to-back calls do not re-init or re-connect.  Calls are serialized
 * on an internal mutex (the ioctx makes control calls from one thread at a
 * time today; the mutex keeps the persistent handle safe if that changes).
 * Exactly one HTTP attempt per call; HTTP-level failure is a result, never
 * an exception.
 */
class curl_s3_control_client final : public s3_control_client {
 public:
  curl_s3_control_client(std::shared_ptr<s3::s3_request_authorizer> authorizer,
                         std::string ca_bundle_path = "",
                         bool tls_verify            = true);
  ~curl_s3_control_client() override;

  [[nodiscard]] head_result head(const rx_route& route) override;
  [[nodiscard]] range_get_result range_get(const rx_route& route,
                                           size_t offset,
                                           size_t size,
                                           uint8_t* dst) override;
  [[nodiscard]] uint64_t attempts_total() const noexcept override
  {
    return _attempts_total.load(std::memory_order_relaxed);
  }
  [[nodiscard]] uint64_t connections_total() const noexcept override
  {
    return _connections_total.load(std::memory_order_relaxed);
  }

 private:
  void* ensure_handle();  // CURL*, created once, reset per call

  std::shared_ptr<s3::s3_request_authorizer> _authorizer;
  std::string _ca_bundle_path;
  bool _tls_verify;
  std::mutex _mtx;
  void* _handle{nullptr};  // persistent CURL easy handle
  std::atomic<uint64_t> _attempts_total{0};
  std::atomic<uint64_t> _connections_total{0};
};

/**
 * @brief Production data-plane session factory over the cuObject SDK.
 *
 * Sessions are dormant in this increment: the factory constructs and
 * `acquire` succeeds (so routing and `start()` capability validation are
 * real), but issuing a GET or registering memory fails loudly until the
 * gateway wiring lands.  Building without @c SIRIUS_ENABLE_S3_RDMA behaves
 * identically — the flag gates only the (unreached) SDK calls.
 */
class cuobj_rdma_data_session_factory final : public rdma_data_session_factory {
 public:
  explicit cuobj_rdma_data_session_factory(
    std::shared_ptr<s3::s3_request_authorizer> data_authorizer);

  [[nodiscard]] std::unique_ptr<rdma_data_session> acquire() override;

 private:
  std::shared_ptr<s3::s3_request_authorizer> _data_authorizer;
};

}  // namespace sirius::io::rdma
