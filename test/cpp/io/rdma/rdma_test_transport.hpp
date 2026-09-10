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

#include "io/rdma/mock_rdma_client.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace sirius::test::rdma {

/// Test fixture joining the independently injectable control and data mocks.
/// It owns no transport behavior; calls are forwarded to the production-owned
/// test doubles frozen by the step-5 API appendix.
class mock_transport_fixture {
 public:
  mock_transport_fixture()
    : control(std::make_shared<io::rdma::mock_s3_control_client>()),
      data(std::make_shared<io::rdma::mock_rdma_data_session_factory>())
  {
  }

  void put_object(std::string bucket, std::string key, std::vector<std::uint8_t> bytes)
  {
    control->put_object(bucket, key, bytes);
    data->put_object(std::move(bucket), std::move(key), std::move(bytes));
  }

  [[nodiscard]] io::rdma::rdma_transport_clients clients(
    io::rdma::reply_tag_predicate predicate = &io::rdma::non_empty_reply_tag) const
  {
    return io::rdma::rdma_transport_clients{control, data, predicate};
  }

  void close_get_gate() { data->close_get_gate(); }
  void open_get_gate() { data->open_get_gate(); }
  void fail_gets(std::string message) { data->fail_gets(std::move(message)); }
  void short_write(std::size_t bytes) { data->short_write(bytes); }

  [[nodiscard]] std::size_t gets_issued() const { return data->gets_issued(); }
  [[nodiscard]] std::size_t peak_concurrent_gets() const { return data->peak_concurrent_gets(); }
  [[nodiscard]] std::size_t register_count() const { return data->register_count(); }
  [[nodiscard]] std::size_t deregister_count() const { return data->deregister_count(); }

  std::shared_ptr<io::rdma::mock_s3_control_client> control;
  std::shared_ptr<io::rdma::mock_rdma_data_session_factory> data;
};

inline std::shared_ptr<mock_transport_fixture> seeded_mock_transport(
  std::string bucket, std::string key, std::vector<std::uint8_t> bytes)
{
  auto transport = std::make_shared<mock_transport_fixture>();
  transport->put_object(std::move(bucket), std::move(key), std::move(bytes));
  return transport;
}

}  // namespace sirius::test::rdma
