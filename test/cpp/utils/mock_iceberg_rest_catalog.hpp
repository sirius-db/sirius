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

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <atomic>
#include <map>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>

namespace sirius::test {

/// Read-only Iceberg REST catalog on a loopback port. It answers /v1/config and loadTable, and
/// vends the storage configuration it was given. A hand-written test double: it checks nothing
/// a real catalog would, and no independent client was used to validate it.
class mock_iceberg_rest_catalog {
 public:
  struct table {
    std::string metadata_location;
    std::string metadata_json;
  };

  /// @p vended is returned as the `config` of every loadTable response.
  explicit mock_iceberg_rest_catalog(std::map<std::string, std::string> vended)
    : _vended(std::move(vended))
  {
    _listener = ::socket(AF_INET, SOCK_STREAM, 0);
    sockaddr_in address{};
    address.sin_family      = AF_INET;
    address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    socklen_t length        = sizeof(address);
    if (_listener < 0 || ::bind(_listener, reinterpret_cast<sockaddr*>(&address), length) != 0 ||
        ::listen(_listener, 16) != 0 ||
        ::getsockname(_listener, reinterpret_cast<sockaddr*>(&address), &length) != 0) {
      if (_listener >= 0) ::close(_listener);
      throw std::runtime_error("mock Iceberg catalog could not listen on loopback");
    }
    _port   = ntohs(address.sin_port);
    _server = std::thread([this] { serve(); });
  }

  ~mock_iceberg_rest_catalog()
  {
    _stop = true;
    ::shutdown(_listener, SHUT_RDWR);
    ::close(_listener);
    _server.join();
  }

  mock_iceberg_rest_catalog(mock_iceberg_rest_catalog const&)            = delete;
  mock_iceberg_rest_catalog& operator=(mock_iceberg_rest_catalog const&) = delete;

  void put(std::string const& schema, std::string const& name, table value)
  {
    std::lock_guard lock(_mutex);
    _tables["/v1/namespaces/" + schema + "/tables/" + name] = std::move(value);
  }

  [[nodiscard]] std::string uri() const { return "http://127.0.0.1:" + std::to_string(_port); }
  [[nodiscard]] std::size_t load_table_requests() const { return _load_table_requests; }

 private:
  static std::string json_string(std::string const& value)
  {
    std::string out{"\""};
    for (char c : value) {
      if (c == '"' || c == '\\') out.push_back('\\');
      out.push_back(c);
    }
    return out + "\"";
  }

  std::pair<int, std::string> respond(std::string const& path)
  {
    if (path == "/v1/config") return {200, R"({"defaults":{},"overrides":{}})"};
    std::lock_guard lock(_mutex);
    auto const found = _tables.find(path);
    if (found == _tables.end()) {
      return {404, R"({"error":{"message":"not found","type":"NoSuchTableException","code":404}})"};
    }
    ++_load_table_requests;
    std::string config;
    for (auto const& [key, value] : _vended) {
      config += (config.empty() ? "" : ",") + json_string(key) + ":" + json_string(value);
    }
    return {200,
            "{\"metadata-location\":" + json_string(found->second.metadata_location) +
              ",\"metadata\":" + found->second.metadata_json + ",\"config\":{" + config + "}}"};
  }

  void serve()
  {
    while (!_stop) {
      int const client = ::accept(_listener, nullptr, nullptr);
      if (client < 0) continue;
      std::string request;
      char buffer[4096];
      while (request.find("\r\n\r\n") == std::string::npos) {
        auto const count = ::recv(client, buffer, sizeof(buffer), 0);
        if (count <= 0) break;
        request.append(buffer, static_cast<std::size_t>(count));
      }
      // "GET <path>[?query] HTTP/1.1"
      auto const begin = request.find(' ');
      auto const end   = request.find_first_of(" ?", begin == std::string::npos ? 0 : begin + 1);
      if (begin != std::string::npos && end != std::string::npos) {
        auto const [status, body] = respond(request.substr(begin + 1, end - begin - 1));
        auto const response =
          "HTTP/1.1 " + std::to_string(status) + (status == 200 ? " OK" : " Not Found") +
          "\r\nContent-Type: application/json\r\nConnection: close\r\nContent-Length: " +
          std::to_string(body.size()) + "\r\n\r\n" + body;
        std::size_t sent = 0;
        while (sent < response.size()) {
          auto const count =
            ::send(client, response.data() + sent, response.size() - sent, MSG_NOSIGNAL);
          if (count <= 0) break;
          sent += static_cast<std::size_t>(count);
        }
      }
      ::close(client);
    }
  }

  std::map<std::string, std::string> _vended;
  std::mutex _mutex;
  std::map<std::string, table> _tables;
  std::atomic<std::size_t> _load_table_requests{0};
  std::atomic<bool> _stop{false};
  int _listener{-1};
  int _port{0};
  std::thread _server;
};

}  // namespace sirius::test
