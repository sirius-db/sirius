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

#include <cuda_runtime.h>

#include <chrono>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#ifdef SIRIUS_ENABLE_S3_RDMA
#include <cuobjclient.h>

#include <mutex>
#endif

namespace sirius::io::rdma {

namespace {

constexpr std::chrono::seconds k_presign_ttl{300};
constexpr long k_request_timeout_s = 30;

struct bounded_sink {
  uint8_t* dst;
  size_t capacity;
  size_t written{0};
};

size_t write_to_sink(char* ptr, size_t size, size_t nmemb, void* userdata)
{
  auto* sink         = static_cast<bounded_sink*>(userdata);
  const size_t bytes = size * nmemb;
  if (sink->written + bytes > sink->capacity) { return 0; }  // abort: server over-delivered
  std::memcpy(sink->dst + sink->written, ptr, bytes);
  sink->written += bytes;
  return bytes;
}

size_t write_discard(char* /*ptr*/, size_t size, size_t nmemb, void* /*userdata*/)
{
  return size * nmemb;
}

bool is_device_pointer(const void* ptr)
{
  cudaPointerAttributes attr{};
  auto err = cudaPointerGetAttributes(&attr, ptr);
  if (err != cudaSuccess) {
    (void)cudaGetLastError();
    return false;
  }
  return attr.type == cudaMemoryTypeDevice;
}

void apply_common_opts(CURL* h, const std::string& ca_bundle_path, bool tls_verify)
{
  SIRIUS_CURL_CHECK(curl_easy_setopt(h, CURLOPT_TIMEOUT, k_request_timeout_s));
  SIRIUS_CURL_CHECK(curl_easy_setopt(h, CURLOPT_SSL_VERIFYPEER, tls_verify ? 1L : 0L));
  SIRIUS_CURL_CHECK(curl_easy_setopt(h, CURLOPT_SSL_VERIFYHOST, tls_verify ? 2L : 0L));
  if (!ca_bundle_path.empty()) {
    SIRIUS_CURL_CHECK(curl_easy_setopt(h, CURLOPT_CAINFO, ca_bundle_path.c_str()));
  }
}

rest::curl_slist_ptr build_headers(const std::vector<std::pair<std::string, std::string>>& headers)
{
  curl_slist* list = nullptr;
  for (const auto& [name, value] : headers) {
    list = curl_slist_append(list, (name + ": " + value).c_str());
  }
  return rest::curl_slist_ptr{list};
}

std::string object_label(std::string_view bucket, std::string_view key)
{
  return "s3://" + std::string(bucket) + "/" + std::string(key);
}

std::string range_header_value(size_t offset, size_t size)
{
  return "bytes=" + std::to_string(offset) + "-" + std::to_string(offset + size - 1);
}

}  // namespace

cuobj_rdma_client::cuobj_rdma_client(std::shared_ptr<s3::s3_request_authorizer> authorizer,
                                     std::string ca_bundle_path,
                                     bool tls_verify)
  : _authorizer(std::move(authorizer)),
    _ca_bundle_path(std::move(ca_bundle_path)),
    _tls_verify(tls_verify)
{
  if (!_authorizer) { throw std::invalid_argument("cuobj_rdma_client: null request authorizer"); }
}

size_t cuobj_rdma_client::head(std::string_view bucket, std::string_view key)
{
  s3::s3_object_ref const obj{std::string(bucket), std::string(key)};
  auto const authd = _authorizer->authorize(obj, s3::s3_request_method::HEAD, k_presign_ttl);

  rest::curl_easy_ptr h{curl_easy_init()};
  if (!h) { throw std::runtime_error("cuobj_rdma_client::head: curl_easy_init failed"); }
  rest::configure_easy_handle(h.get(), rest::global_curl_context::instance().share_handle());
  apply_common_opts(h.get(), _ca_bundle_path, _tls_verify);

  auto hdrs = build_headers(authd.headers);
  SIRIUS_CURL_CHECK(curl_easy_setopt(h.get(), CURLOPT_URL, authd.url.c_str()));
  SIRIUS_CURL_CHECK(curl_easy_setopt(h.get(), CURLOPT_NOBODY, 1L));
  SIRIUS_CURL_CHECK(curl_easy_setopt(h.get(), CURLOPT_HTTPHEADER, hdrs.get()));
  SIRIUS_CURL_CHECK(curl_easy_setopt(h.get(), CURLOPT_WRITEFUNCTION, &write_discard));

  CURLcode const rc = curl_easy_perform(h.get());
  long status       = 0;
  curl_easy_getinfo(h.get(), CURLINFO_RESPONSE_CODE, &status);

  if (rc != CURLE_OK) {
    throw std::runtime_error("cuobj_rdma_client::head: " + object_label(bucket, key) + ": " +
                             curl_easy_strerror(rc));
  }
  if (status != 200) {
    throw std::runtime_error("cuobj_rdma_client::head: " + object_label(bucket, key) + " -> HTTP " +
                             std::to_string(status));
  }
  curl_off_t content_length = -1;
  curl_easy_getinfo(h.get(), CURLINFO_CONTENT_LENGTH_DOWNLOAD_T, &content_length);
  if (content_length < 0) {
    throw std::runtime_error("cuobj_rdma_client::head: " + object_label(bucket, key) +
                             ": no Content-Length");
  }
  return static_cast<size_t>(content_length);
}

size_t cuobj_rdma_client::get(
  std::string_view bucket, std::string_view key, size_t offset, size_t size, void* dst)
{
  if (size == 0) { return 0; }
  if (is_device_pointer(dst)) { return device_get(bucket, key, offset, size, dst); }
  return host_get(bucket, key, offset, size, dst);
}

size_t cuobj_rdma_client::host_get(
  std::string_view bucket, std::string_view key, size_t offset, size_t size, void* dst)
{
  s3::s3_object_ref const obj{std::string(bucket), std::string(key)};
  auto const authd = _authorizer->authorize(obj, s3::s3_request_method::GET, k_presign_ttl);

  rest::curl_easy_ptr h{curl_easy_init()};
  if (!h) { throw std::runtime_error("cuobj_rdma_client::host_get: curl_easy_init failed"); }
  rest::configure_easy_handle(h.get(), rest::global_curl_context::instance().share_handle());
  apply_common_opts(h.get(), _ca_bundle_path, _tls_verify);

  auto headers = authd.headers;
  headers.emplace_back("Range", range_header_value(offset, size));
  auto hdrs = build_headers(headers);

  bounded_sink sink{static_cast<uint8_t*>(dst), size};
  SIRIUS_CURL_CHECK(curl_easy_setopt(h.get(), CURLOPT_URL, authd.url.c_str()));
  SIRIUS_CURL_CHECK(curl_easy_setopt(h.get(), CURLOPT_HTTPHEADER, hdrs.get()));
  SIRIUS_CURL_CHECK(curl_easy_setopt(h.get(), CURLOPT_WRITEFUNCTION, &write_to_sink));
  SIRIUS_CURL_CHECK(curl_easy_setopt(h.get(), CURLOPT_WRITEDATA, &sink));

  CURLcode const rc = curl_easy_perform(h.get());
  long status       = 0;
  curl_easy_getinfo(h.get(), CURLINFO_RESPONSE_CODE, &status);

  // 416: the whole requested range starts at/after EOF — an empty read, not an
  // error (the reactor clips sizes, but callers may probe past the end).
  if (status == 416) { return 0; }
  if (rc != CURLE_OK) {
    throw std::runtime_error("cuobj_rdma_client::host_get: " + object_label(bucket, key) + ": " +
                             curl_easy_strerror(rc));
  }
  if (status != 200 && status != 206) {
    throw std::runtime_error("cuobj_rdma_client::host_get: " + object_label(bucket, key) +
                             " -> HTTP " + std::to_string(status));
  }
  return sink.written;
}

void cuobj_rdma_client::control_get(
  std::string_view bucket,
  std::string_view key,
  const std::vector<std::pair<std::string, std::string>>& extra_headers)
{
  s3::s3_object_ref const obj{std::string(bucket), std::string(key)};
  auto const authd = _authorizer->authorize(obj, s3::s3_request_method::GET, k_presign_ttl);

  rest::curl_easy_ptr h{curl_easy_init()};
  if (!h) { throw std::runtime_error("cuobj_rdma_client::control_get: curl_easy_init failed"); }
  rest::configure_easy_handle(h.get(), rest::global_curl_context::instance().share_handle());
  apply_common_opts(h.get(), _ca_bundle_path, _tls_verify);

  auto headers = authd.headers;
  headers.insert(headers.end(), extra_headers.begin(), extra_headers.end());
  auto hdrs = build_headers(headers);
  SIRIUS_CURL_CHECK(curl_easy_setopt(h.get(), CURLOPT_URL, authd.url.c_str()));
  SIRIUS_CURL_CHECK(curl_easy_setopt(h.get(), CURLOPT_HTTPHEADER, hdrs.get()));
  SIRIUS_CURL_CHECK(curl_easy_setopt(h.get(), CURLOPT_WRITEFUNCTION, &write_discard));

  CURLcode const rc = curl_easy_perform(h.get());
  long status       = 0;
  curl_easy_getinfo(h.get(), CURLINFO_RESPONSE_CODE, &status);
  if (rc != CURLE_OK) {
    throw std::runtime_error("cuobj_rdma_client::control_get: " + object_label(bucket, key) + ": " +
                             curl_easy_strerror(rc));
  }
  if (status != 200 && status != 206) {
    throw std::runtime_error("cuobj_rdma_client::control_get: " + object_label(bucket, key) +
                             " -> HTTP " + std::to_string(status));
  }
}

#ifdef SIRIUS_ENABLE_S3_RDMA

namespace {

/// Per-request context recovered inside the cuObject get callback.
struct cuobj_get_context {
  cuobj_rdma_client* self;
  std::string bucket;
  std::string key;
  std::string range;
};

std::string base64_encode(const char* data, size_t len)
{
  static constexpr char table[] =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  std::string out;
  out.reserve((len + 2) / 3 * 4);
  for (size_t i = 0; i < len; i += 3) {
    uint32_t chunk = static_cast<uint8_t>(data[i]) << 16;
    if (i + 1 < len) { chunk |= static_cast<uint8_t>(data[i + 1]) << 8; }
    if (i + 2 < len) { chunk |= static_cast<uint8_t>(data[i + 2]); }
    out.push_back(table[(chunk >> 18) & 0x3f]);
    out.push_back(table[(chunk >> 12) & 0x3f]);
    out.push_back(i + 1 < len ? table[(chunk >> 6) & 0x3f] : '=');
    out.push_back(i + 2 < len ? table[chunk & 0x3f] : '=');
  }
  return out;
}

/// cuObject control-plane callback: SigV4-signed HTTP GET carrying the RDMA
/// descriptor token; the gateway RDMA_WRITEs into the registered destination
/// and the reply is status-only.  Runs on the cuObjGet caller's thread.
ssize_t cuobj_get_callback(
  const void* handle, char* /*ptr*/, size_t size, loff_t /*offset*/, const cufileRDMAInfo_t* rdma)
{
  auto* ctx = static_cast<cuobj_get_context*>(cuObjClient::getCtx(handle));
  if (ctx == nullptr || rdma == nullptr) { return -1; }
  try {
    std::vector<std::pair<std::string, std::string>> extra;
    extra.emplace_back("x-amz-rdma-token", base64_encode(rdma->desc_str, rdma->desc_len));
    if (!ctx->range.empty()) { extra.emplace_back("Range", ctx->range); }
    ctx->self->control_get(ctx->bucket, ctx->key, extra);
    return static_cast<ssize_t>(size);
  } catch (...) {
    return -1;
  }
}

std::mutex g_cuobj_mtx;

}  // namespace

void* cuobj_rdma_client::ensure_cuobj_client()
{
  std::lock_guard lk{g_cuobj_mtx};
  if (_cuobj == nullptr) {
    static CUObjOps_t ops{};
    ops.get = &cuobj_get_callback;
    ops.put = nullptr;
    _cuobj  = new cuObjClient(ops, CUOBJ_PROTO_RDMA_DC_V1);
  }
  return _cuobj;
}

size_t cuobj_rdma_client::device_get(
  std::string_view bucket, std::string_view key, size_t offset, size_t size, void* dst)
{
  auto* client = static_cast<cuObjClient*>(ensure_cuobj_client());
  cuobj_get_context ctx{
    this, std::string(bucket), std::string(key), range_header_value(offset, size)};
  ssize_t const n = client->cuObjGet(&ctx, dst, size);
  if (n < 0) {
    throw std::runtime_error("cuobj_rdma_client::device_get: " + object_label(bucket, key) +
                             ": cuObjGet failed (rc=" + std::to_string(n) + ")");
  }
  return static_cast<size_t>(n);
}

void cuobj_rdma_client::register_memory(void* base, size_t bytes)
{
  auto* client  = static_cast<cuObjClient*>(ensure_cuobj_client());
  auto const rc = client->cuMemObjGetDescriptor(base, bytes);
  if (rc != CU_OBJ_SUCCESS) {
    throw std::runtime_error("cuobj_rdma_client::register_memory: registration failed (rc=" +
                             std::to_string(static_cast<int>(rc)) + ")");
  }
}

void cuobj_rdma_client::deregister_memory(void* base) noexcept
{
  try {
    if (_cuobj != nullptr) { static_cast<cuObjClient*>(_cuobj)->cuMemObjPutDescriptor(base); }
  } catch (...) {  // NOLINT(bugprone-empty-catch)
  }
}

cuobj_rdma_client::~cuobj_rdma_client() { delete static_cast<cuObjClient*>(_cuobj); }

#else  // !SIRIUS_ENABLE_S3_RDMA

size_t cuobj_rdma_client::device_get(
  std::string_view bucket, std::string_view key, size_t /*offset*/, size_t /*size*/, void* /*dst*/)
{
  throw std::runtime_error(
    "cuobj_rdma_client::device_get: " + object_label(bucket, key) +
    ": device-destination GET requires a build with SIRIUS_ENABLE_S3_RDMA (cuObject SDK)");
}

void cuobj_rdma_client::register_memory(void* /*base*/, size_t /*bytes*/) {}

void cuobj_rdma_client::deregister_memory(void* /*base*/) noexcept {}

cuobj_rdma_client::~cuobj_rdma_client() = default;

#endif  // SIRIUS_ENABLE_S3_RDMA

}  // namespace sirius::io::rdma
