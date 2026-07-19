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

#include "io/rdma/cuobj_rdma_client.hpp"

#include "io/rest/curl_handle.hpp"
#include "io/s3/s3_object_ref.hpp"

#include <cctype>
#include <chrono>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace sirius::io::rdma {

namespace {

constexpr std::chrono::seconds k_presign_ttl{300};
constexpr long k_request_timeout_s = 30;

/// One range response: bounded body delivery + the headers we act on
/// (status line and Content-Range), shared as both the write and header
/// callback userdata so the write callback can see the live status.
struct range_response {
  uint8_t* dst;
  size_t capacity;
  size_t written{0};
  long status{0};             ///< parsed from the status line by @c capture_header
  std::string content_range;  ///< raw Content-Range value; empty if absent
};

size_t write_to_sink(char* ptr, size_t size, size_t nmemb, void* userdata)
{
  auto* sink         = static_cast<range_response*>(userdata);
  const size_t bytes = size * nmemb;
  // An error status carries a diagnostic body (e.g. a 416 / 4xx XML error),
  // not payload: consume and discard it so the transfer completes and the
  // HTTP status becomes the result instead of a fabricated write error.  The
  // over-delivery guard (abort when a success response exceeds the requested
  // range) applies only to a delivering status.
  if (sink->status >= 400) { return bytes; }
  if (sink->written + bytes > sink->capacity) { return 0; }  // abort: server over-delivered
  std::memcpy(sink->dst + sink->written, ptr, bytes);
  sink->written += bytes;
  return bytes;
}

size_t write_discard(char* /*ptr*/, size_t size, size_t nmemb, void* /*userdata*/)
{
  return size * nmemb;
}

std::string_view trim_header_value(std::string_view value)
{
  while (!value.empty() && (value.front() == ' ' || value.front() == '\t')) {
    value.remove_prefix(1);
  }
  while (!value.empty() && (value.back() == '\r' || value.back() == '\n' || value.back() == ' ')) {
    value.remove_suffix(1);
  }
  return value;
}

size_t capture_header(char* buffer, size_t size, size_t nitems, void* userdata)
{
  auto* response     = static_cast<range_response*>(userdata);
  const size_t bytes = size * nitems;
  std::string_view line{buffer, bytes};
  // Status line ("HTTP/1.1 206 Partial Content"): the last one wins (a 100
  // Continue or a redirect precedes the final status).
  if (line.rfind("HTTP/", 0) == 0) {
    if (const auto sp = line.find(' '); sp != std::string_view::npos) {
      long parsed = 0;
      for (size_t i = sp + 1; i < line.size() && line[i] >= '0' && line[i] <= '9'; ++i) {
        parsed = parsed * 10 + (line[i] - '0');
      }
      response->status = parsed;
    }
    return bytes;
  }
  constexpr std::string_view name{"content-range:"};
  if (line.size() > name.size()) {
    bool match = true;
    for (size_t i = 0; i < name.size(); ++i) {
      if (static_cast<char>(std::tolower(static_cast<unsigned char>(line[i]))) != name[i]) {
        match = false;
        break;
      }
    }
    if (match) { response->content_range.assign(trim_header_value(line.substr(name.size()))); }
  }
  return bytes;
}

rest::curl_slist_ptr build_headers(const std::vector<std::pair<std::string, std::string>>& headers)
{
  curl_slist* list = nullptr;
  for (const auto& [name, value] : headers) {
    list = curl_slist_append(list, (name + ": " + value).c_str());
  }
  return rest::curl_slist_ptr{list};
}

std::string range_header_value(size_t offset, size_t size)
{
  return "bytes=" + std::to_string(offset) + "-" + std::to_string(offset + size - 1);
}

}  // namespace

curl_s3_control_client::curl_s3_control_client(
  std::shared_ptr<s3::s3_request_authorizer> authorizer,
  std::string ca_bundle_path,
  bool tls_verify)
  : _authorizer(std::move(authorizer)),
    _ca_bundle_path(std::move(ca_bundle_path)),
    _tls_verify(tls_verify)
{
  if (!_authorizer) {
    throw std::invalid_argument("curl_s3_control_client: null request authorizer");
  }
}

curl_s3_control_client::~curl_s3_control_client()
{
  if (_handle != nullptr) { curl_easy_cleanup(static_cast<CURL*>(_handle)); }
}

void* curl_s3_control_client::ensure_handle()
{
  if (_handle == nullptr) {
    _handle = curl_easy_init();
    if (_handle == nullptr) {
      throw std::runtime_error("curl_s3_control_client: curl_easy_init failed");
    }
    rest::configure_easy_handle(static_cast<CURL*>(_handle),
                                rest::global_curl_context::instance().share_handle());
  }
  // Reset per-call options while KEEPING the handle (and with it the live
  // connection cache — the whole point of the persistent handle); the shared
  // DNS/TLS-session context and defaults are re-applied below per call.
  return _handle;
}

head_result curl_s3_control_client::head(const rx_route& route)
{
  std::lock_guard lk{_mtx};
  _attempts_total.fetch_add(1, std::memory_order_relaxed);
  head_result result;

  try {
    s3::s3_object_ref const obj{route.bucket, route.key};
    auto const authd = _authorizer->authorize(obj, s3::s3_request_method::HEAD, k_presign_ttl);

    auto* h = static_cast<CURL*>(ensure_handle());
    SIRIUS_CURL_CHECK(curl_easy_setopt(h, CURLOPT_TIMEOUT, k_request_timeout_s));
    SIRIUS_CURL_CHECK(curl_easy_setopt(h, CURLOPT_SSL_VERIFYPEER, _tls_verify ? 1L : 0L));
    SIRIUS_CURL_CHECK(curl_easy_setopt(h, CURLOPT_SSL_VERIFYHOST, _tls_verify ? 2L : 0L));
    if (!_ca_bundle_path.empty()) {
      SIRIUS_CURL_CHECK(curl_easy_setopt(h, CURLOPT_CAINFO, _ca_bundle_path.c_str()));
    }
    auto hdrs = build_headers(authd.headers);
    SIRIUS_CURL_CHECK(curl_easy_setopt(h, CURLOPT_URL, authd.url.c_str()));
    SIRIUS_CURL_CHECK(curl_easy_setopt(h, CURLOPT_NOBODY, 1L));
    SIRIUS_CURL_CHECK(curl_easy_setopt(h, CURLOPT_HTTPHEADER, hdrs.get()));
    SIRIUS_CURL_CHECK(curl_easy_setopt(h, CURLOPT_WRITEFUNCTION, &write_discard));
    SIRIUS_CURL_CHECK(curl_easy_setopt(h, CURLOPT_HEADERFUNCTION, nullptr));
    SIRIUS_CURL_CHECK(curl_easy_setopt(h, CURLOPT_HEADERDATA, nullptr));

    CURLcode const rc = curl_easy_perform(h);
    long connects     = 0;
    if (curl_easy_getinfo(h, CURLINFO_NUM_CONNECTS, &connects) == CURLE_OK && connects > 0) {
      _connections_total.fetch_add(static_cast<uint64_t>(connects), std::memory_order_relaxed);
    }
    SIRIUS_CURL_CHECK(curl_easy_setopt(h, CURLOPT_NOBODY, 0L));  // un-stick for later GETs
    // Capture the status even on a CURLcode error: a response may have
    // arrived before the transport failed (operation-specific result).
    long status = 0;
    curl_easy_getinfo(h, CURLINFO_RESPONSE_CODE, &status);
    result.outcome.http_status = status;
    if (status == 200) {
      curl_off_t content_length = -1;
      curl_easy_getinfo(h, CURLINFO_CONTENT_LENGTH_DOWNLOAD_T, &content_length);
      if (content_length >= 0) { result.object_size = static_cast<size_t>(content_length); }
    }
    if (rc != CURLE_OK) { result.outcome.transport_error = curl_easy_strerror(rc); }
    return result;
  } catch (std::exception const& e) {
    // Authorization/setup failure: nothing reached the wire.
    result.outcome.transport_error = e.what();
    return result;
  }
}

range_get_result curl_s3_control_client::range_get(const rx_route& route,
                                                   size_t offset,
                                                   size_t size,
                                                   uint8_t* dst)
{
  std::lock_guard lk{_mtx};
  _attempts_total.fetch_add(1, std::memory_order_relaxed);
  range_get_result result;
  if (size == 0) { return result; }

  try {
    s3::s3_object_ref const obj{route.bucket, route.key};
    auto const authd = _authorizer->authorize(obj, s3::s3_request_method::GET, k_presign_ttl);

    auto* h = static_cast<CURL*>(ensure_handle());
    SIRIUS_CURL_CHECK(curl_easy_setopt(h, CURLOPT_TIMEOUT, k_request_timeout_s));
    SIRIUS_CURL_CHECK(curl_easy_setopt(h, CURLOPT_SSL_VERIFYPEER, _tls_verify ? 1L : 0L));
    SIRIUS_CURL_CHECK(curl_easy_setopt(h, CURLOPT_SSL_VERIFYHOST, _tls_verify ? 2L : 0L));
    if (!_ca_bundle_path.empty()) {
      SIRIUS_CURL_CHECK(curl_easy_setopt(h, CURLOPT_CAINFO, _ca_bundle_path.c_str()));
    }
    auto headers = authd.headers;
    headers.emplace_back("Range", range_header_value(offset, size));
    auto hdrs = build_headers(headers);

    range_response response{dst, size};
    SIRIUS_CURL_CHECK(curl_easy_setopt(h, CURLOPT_URL, authd.url.c_str()));
    SIRIUS_CURL_CHECK(curl_easy_setopt(h, CURLOPT_NOBODY, 0L));
    SIRIUS_CURL_CHECK(curl_easy_setopt(h, CURLOPT_HTTPHEADER, hdrs.get()));
    SIRIUS_CURL_CHECK(curl_easy_setopt(h, CURLOPT_WRITEFUNCTION, &write_to_sink));
    SIRIUS_CURL_CHECK(curl_easy_setopt(h, CURLOPT_WRITEDATA, &response));
    SIRIUS_CURL_CHECK(curl_easy_setopt(h, CURLOPT_HEADERFUNCTION, &capture_header));
    SIRIUS_CURL_CHECK(curl_easy_setopt(h, CURLOPT_HEADERDATA, &response));

    // The persistent handle is reused, so any pointer into this frame's stack
    // objects must be cleared before returning — even on an early/exception
    // exit — so a later perform can never see a dangling WRITEDATA/HEADERDATA.
    struct callback_reset {
      CURL* h;
      ~callback_reset()
      {
        curl_easy_setopt(h, CURLOPT_HEADERFUNCTION, nullptr);
        curl_easy_setopt(h, CURLOPT_HEADERDATA, nullptr);
        curl_easy_setopt(h, CURLOPT_WRITEFUNCTION, &write_discard);
        curl_easy_setopt(h, CURLOPT_WRITEDATA, nullptr);
      }
    } reset{h};

    CURLcode const rc = curl_easy_perform(h);
    long connects     = 0;
    if (curl_easy_getinfo(h, CURLINFO_NUM_CONNECTS, &connects) == CURLE_OK && connects > 0) {
      _connections_total.fetch_add(static_cast<uint64_t>(connects), std::memory_order_relaxed);
    }
    // Capture what actually happened BEFORE interpreting the CURLcode: a
    // response can arrive and then the body transfer fail mid-stream (a
    // partial 206 then a disconnect), and the result must still carry the
    // real status, Content-Range, and bytes written (operation-specific
    // result, contract §4) rather than collapse to status 0.
    long status = 0;
    curl_easy_getinfo(h, CURLINFO_RESPONSE_CODE, &status);
    result.outcome.http_status = status;
    result.delivered_bytes     = (status >= 400) ? 0 : response.written;
    result.content_range       = std::move(response.content_range);
    if (rc != CURLE_OK) { result.outcome.transport_error = curl_easy_strerror(rc); }
    return result;
  } catch (std::exception const& e) {
    result.outcome.transport_error = e.what();
    return result;
  }
}

namespace {

/// Dormant production data session: routing, capability validation, and
/// worker acquisition are real; touching the wire is not wired up yet, so
/// every data operation fails loudly instead of guessing at gateway
/// behavior.
class dormant_cuobj_session final : public rdma_data_session {
 public:
  void register_memory(void* /*base*/, size_t /*bytes*/) override
  {
    throw std::runtime_error(
      "cuobj_rdma_data_session: the RDMA data plane is not wired to a gateway yet; "
      "device registration is unavailable");
  }
  void deregister_memory(void* /*base*/) noexcept override {}
  data_get_result get(const rx_route& /*route*/,
                      size_t /*offset*/,
                      size_t /*size*/,
                      void* /*dst*/) override
  {
    data_get_result result;
    result.commit = data_commit_state::not_sent;
    result.transport_error =
      "cuobj_rdma_data_session: the RDMA data plane is not wired to a gateway yet";
    return result;
  }
};

}  // namespace

cuobj_rdma_data_session_factory::cuobj_rdma_data_session_factory(
  std::shared_ptr<s3::s3_request_authorizer> data_authorizer)
  : _data_authorizer(std::move(data_authorizer))
{
  if (!_data_authorizer) {
    throw std::invalid_argument("cuobj_rdma_data_session_factory: null data-plane authorizer");
  }
}

std::unique_ptr<rdma_data_session> cuobj_rdma_data_session_factory::acquire()
{
  return std::make_unique<dormant_cuobj_session>();
}

}  // namespace sirius::io::rdma
