/* Copyright 2026, Sirius Contributors. SPDX-License-Identifier: Apache-2.0 */
#pragma once

#include <httplib.hpp>

#include <atomic>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace sirius::test {
// A single object that fails reads at a specific byte offset, regardless of request order.
class offset_fault_server {
 public:
  offset_fault_server(std::vector<uint8_t> bytes, std::string etag, uint64_t offset, bool truncate)
    : bytes_(std::move(bytes))
  {
    server_.new_task_queue = [] { return new duckdb_httplib::ThreadPool(2); };
    server_.Get(
      "/.*", [this, etag = std::move(etag), offset, truncate](auto const& request, auto& response) {
        bool fault = false;
        if (request.method == "GET") {
          if (request.get_header_value_count("If-Match") != 1 ||
              request.get_header_value("If-Match") != etag)
            matched_ = false;
          fault =
            request.get_header_value("Range").starts_with("bytes=" + std::to_string(offset) + "-");
          if (fault) ++hits_;
        }
        response.set_header("ETag", etag);
        if (fault && !truncate) {
          response.status = 400;
          return;
        }
        response.set_content_provider(
          bytes_.size(),
          "application/octet-stream",
          [this, fault](size_t start, size_t length, duckdb_httplib::DataSink& sink) {
            auto n = fault ? length / 2 : length;
            if (n && !sink.write(reinterpret_cast<char const*>(bytes_.data() + start), n))
              return false;
            return !fault;
          });
      });
    port_ = server_.bind_to_any_port("127.0.0.1");
    if (port_ < 0) throw std::runtime_error("failed to bind range fault server");
    worker_ = std::thread([this] { server_.listen_after_bind(); });
    server_.wait_until_ready();
  }
  ~offset_fault_server()
  {
    server_.stop();
    worker_.join();
  }
  std::string endpoint() const { return "http://127.0.0.1:" + std::to_string(port_); }
  size_t targeted_reads() const { return hits_.load(); }
  bool all_gets_matched_etag() const { return matched_.load(); }

 private:
  std::vector<uint8_t> const bytes_;
  duckdb_httplib::Server server_;
  std::atomic<size_t> hits_{0};
  std::atomic<bool> matched_{true};
  int port_;
  std::thread worker_;
};
}  // namespace sirius::test
