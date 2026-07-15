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

#include "io/rest/rest_reactor.hpp"

#include "cucascade/cuda/event.hpp"
#include "io/details/slot_pool.hpp"
#include "io/rest/curl_handle.hpp"
#include "io/uri_parser.hpp"
#include "log/logging.hpp"

#include <rmm/cuda_device.hpp>

#include <sys/epoll.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <deque>
#include <format>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <system_error>
#include <thread>
#include <unordered_map>
#include <vector>

namespace sirius::io::rest {

namespace {

// Lock-free max: raise @p a to @p v when v is larger.  Backs the *_ns_max perf
// counters (relaxed — perf metrics tolerate reordering).
void atomic_max_relaxed(std::atomic<std::uint64_t>& a, std::uint64_t v) noexcept
{
  std::uint64_t cur = a.load(std::memory_order_relaxed);
  while (v > cur && !a.compare_exchange_weak(cur, v, std::memory_order_relaxed)) {}
}

// ---- libcurl callbacks -----------------------------------------------------

/// Write callback: copy curl's bytes into the sink's destination buffer at the
/// running cursor, and ALWAYS report the full incoming size so curl never
/// aborts the transfer with CURLE_WRITE_ERROR.  Overflow past capacity is
/// counted (total_received) but not stored, so a server that ignored the Range
/// header can be detected after the fact.
size_t write_to_sink(char* ptr, size_t size, size_t nmemb, void* userdata)
{
  auto* sink         = static_cast<buf_sink*>(userdata);
  size_t const bytes = size * nmemb;
  sink->total_received += bytes;
  size_t remaining = bytes;
  auto const* src  = reinterpret_cast<uint8_t const*>(ptr);
  // Scatter across the destination buffers in file order; a fused contiguous
  // GET spills from one buffer into the next as each fills.
  while (remaining > 0 && sink->active < sink->buffers.size()) {
    iovec& b = sink->buffers[sink->active];
    if (b.iov_base == nullptr) { break; }
    if (sink->cursor >= b.iov_len) {
      ++sink->active;
      sink->cursor = 0;
      continue;
    }
    size_t const n = std::min(b.iov_len - sink->cursor, remaining);
    std::memcpy(static_cast<uint8_t*>(b.iov_base) + sink->cursor, src, n);
    sink->cursor += n;
    sink->written += n;
    src += n;
    remaining -= n;
  }
  return bytes;
}

/// Discard callback for HEAD requests (no body expected, but be defensive).
size_t write_discard(char* /*ptr*/, size_t size, size_t nmemb, void* /*userdata*/)
{
  return size * nmemb;
}

/// Lowercase a byte.
char ascii_lower(char c) { return static_cast<char>(std::tolower(static_cast<unsigned char>(c))); }

/// Case-insensitively match @p line against "<name>:" and, on a hit, return the
/// trimmed value; otherwise return empty.
std::string match_header(std::string_view line, std::string_view name)
{
  if (line.size() < name.size() + 1) { return {}; }
  for (size_t i = 0; i < name.size(); ++i) {
    if (ascii_lower(line[i]) != ascii_lower(name[i])) { return {}; }
  }
  if (line[name.size()] != ':') { return {}; }
  std::string_view val = line.substr(name.size() + 1);
  // Trim surrounding whitespace and trailing CRLF.
  while (!val.empty() && (val.front() == ' ' || val.front() == '\t')) {
    val.remove_prefix(1);
  }
  while (!val.empty() &&
         (val.back() == '\r' || val.back() == '\n' || val.back() == ' ' || val.back() == '\t')) {
    val.remove_suffix(1);
  }
  return std::string(val);
}

/// Header callback: capture Content-Range and Retry-After.
size_t capture_header(char* buffer, size_t size, size_t nitems, void* userdata)
{
  auto* hc           = static_cast<header_capture*>(userdata);
  size_t const bytes = size * nitems;
  std::string_view line(buffer, bytes);
  if (auto v = match_header(line, "content-range"); !v.empty()) {
    hc->content_range = std::move(v);
  }
  if (auto v = match_header(line, "retry-after"); !v.empty()) { hc->retry_after = std::move(v); }
  return bytes;
}

/// Shared sink for a suffix-range footer probe: the header callback records the
/// HTTP status (from the status line) plus Content-Range / Retry-After; the body
/// callback consults @c status to abort a non-206 response before it streams a
/// whole object into us.  @c HEADERDATA and @c WRITEDATA point at the same one.
struct suffix_sink {
  std::vector<std::uint8_t> data;
  std::size_t cap{0};
  std::size_t total_received{0};  // wire bytes, incl. those dropped by cap/abort
  long status{0};
  std::string content_range;
  std::string retry_after;
};

/// Header callback for a suffix probe: parse the status code out of the status
/// line so the body callback can abort a non-206 early, and capture the headers
/// the caller needs (Content-Range to verify the 206, Retry-After for backoff).
size_t suffix_header_cb(char* buffer, size_t size, size_t nitems, void* userdata)
{
  auto* s            = static_cast<suffix_sink*>(userdata);
  size_t const bytes = size * nitems;
  std::string_view const line(buffer, bytes);
  if (line.size() >= 5 && ascii_lower(line[0]) == 'h' && ascii_lower(line[1]) == 't' &&
      ascii_lower(line[2]) == 't' && ascii_lower(line[3]) == 'p' && line[4] == '/') {
    if (auto const sp = line.find(' '); sp != std::string_view::npos) {
      long code = 0;
      for (size_t i = sp + 1; i < line.size() && line[i] >= '0' && line[i] <= '9'; ++i) {
        code = code * 10 + (line[i] - '0');
      }
      if (code != 0) { s->status = code; }
    }
  }
  if (auto v = match_header(line, "content-range"); !v.empty()) { s->content_range = std::move(v); }
  if (auto v = match_header(line, "retry-after"); !v.empty()) { s->retry_after = std::move(v); }
  return bytes;
}

/// Body callback for a suffix probe: abort a non-206 response (a deliberate
/// short write, surfacing as CURLE_WRITE_ERROR) so a server that ignores the
/// Range or answers 416/4xx never streams a whole object into us; otherwise
/// append up to @c cap bytes and report the full incoming size to curl.
size_t suffix_write_cb(char* ptr, size_t size, size_t nmemb, void* userdata)
{
  auto* s            = static_cast<suffix_sink*>(userdata);
  size_t const bytes = size * nmemb;
  s->total_received += bytes;
  if (s->status != 206) { return 0; }
  if (s->data.size() < s->cap) {
    size_t const take = std::min(s->cap - s->data.size(), bytes);
    auto const* src   = reinterpret_cast<std::uint8_t const*>(ptr);
    s->data.insert(s->data.end(), src, src + take);
  }
  return bytes;
}

// ---- retry classification --------------------------------------------------

/// HTTP status codes worth retrying (transient server / throttling).  Only the
/// transient 5xx are included: 500 Internal Error, 502 Bad Gateway, 503 Slow
/// Down / Service Unavailable, 504 Gateway Timeout.  Permanent 5xx (501 Not
/// Implemented, 505 HTTP Version Not Supported, ...) are NOT retried — they
/// would only burn the full retry budget on an error that cannot succeed.
bool is_retriable_status(long status) noexcept
{
  return status == 408 || status == 429 || status == 500 || status == 502 || status == 503 ||
         status == 504;
}

/// libcurl error codes worth retrying (transient transport failures).
bool is_retriable_curl(CURLcode rc) noexcept
{
  switch (rc) {
    case CURLE_COULDNT_CONNECT:
    case CURLE_COULDNT_RESOLVE_HOST:
    case CURLE_OPERATION_TIMEDOUT:
    case CURLE_GOT_NOTHING:
    case CURLE_RECV_ERROR:
    case CURLE_SEND_ERROR:
    case CURLE_PARTIAL_FILE:
    case CURLE_SSL_CONNECT_ERROR:
    case CURLE_HTTP2_STREAM: return true;
    default: return false;
  }
}

// ---- per-request helpers ---------------------------------------------------

/// Presigned-URL TTL for a request: the whole-request timeout plus clock-skew
/// headroom, with a sane floor so very short timeouts still leave a usable
/// window.
std::chrono::seconds presign_ttl(const config& cfg) noexcept
{
  long const base = cfg.request_timeout_s > 0 ? cfg.request_timeout_s + 60 : 300;
  return std::chrono::seconds{base};
}

/// Apply per-request TLS + timeout options on top of configure_easy_handle.
void apply_request_opts(CURL* h, const config& cfg)
{
  if (!cfg.ca_bundle_path.empty()) {
    SIRIUS_CURL_CHECK(curl_easy_setopt(h, CURLOPT_CAINFO, cfg.ca_bundle_path.c_str()));
  }
  if (!cfg.tls_verify) {
    SIRIUS_CURL_CHECK(curl_easy_setopt(h, CURLOPT_SSL_VERIFYPEER, 0L));
    SIRIUS_CURL_CHECK(curl_easy_setopt(h, CURLOPT_SSL_VERIFYHOST, 0L));
  }
  if (cfg.request_timeout_s > 0) {
    SIRIUS_CURL_CHECK(curl_easy_setopt(h, CURLOPT_TIMEOUT, cfg.request_timeout_s));
  }
}

/// Build a header list from the authorizer's headers (empty in presigned mode)
/// plus an optional Range header.
curl_slist_ptr build_header_list(std::vector<std::pair<std::string, std::string>> const& headers,
                                 std::string const* range)
{
  curl_slist* list = nullptr;
  for (auto const& [k, v] : headers) {
    std::string const h = k + ": " + v;
    list                = curl_slist_append(list, h.c_str());
  }
  if (range != nullptr) { list = curl_slist_append(list, range->c_str()); }
  return curl_slist_ptr{list};
}

/// "Range: bytes=<lo>-<hi>" (inclusive end) for [offset, offset+size).
std::string range_header(size_t offset, size_t size)
{
  return std::format("Range: bytes={}-{}", offset, offset + size - 1);
}

/// "Range: bytes=-<n>" — the last @p n bytes of an object (a suffix range).
/// Unlike range_header this needs no prior knowledge of the object's size.
std::string suffix_range_header(size_t n) { return std::format("Range: bytes=-{}", n); }

/// Parse the first-byte position out of a Content-Range value of the form
/// "bytes <first>-<last>/<total>" (the trimmed value captured by the header
/// callback).  Returns nullopt for any value that does not start with a
/// well-formed "bytes <first>-" so the caller can reject an unverifiable 206.
std::optional<size_t> content_range_start(std::string const& cr)
{
  constexpr std::string_view kUnit = "bytes";
  std::string_view sv{cr};
  if (sv.size() < kUnit.size()) { return std::nullopt; }
  for (size_t i = 0; i < kUnit.size(); ++i) {
    if (ascii_lower(sv[i]) != kUnit[i]) { return std::nullopt; }
  }
  sv.remove_prefix(kUnit.size());
  while (!sv.empty() && (sv.front() == ' ' || sv.front() == '\t')) {
    sv.remove_prefix(1);
  }
  if (sv.empty() || sv.front() < '0' || sv.front() > '9') { return std::nullopt; }
  size_t value = 0;
  size_t i     = 0;
  for (; i < sv.size() && sv[i] >= '0' && sv[i] <= '9'; ++i) {
    value = value * 10 + static_cast<size_t>(sv[i] - '0');
  }
  // A valid first-byte position is immediately followed by '-' (the range
  // separator); anything else ("*", end of string, ...) is not parseable.
  if (i >= sv.size() || sv[i] != '-') { return std::nullopt; }
  return value;
}

/// Backoff before the next attempt: honor a numeric Retry-After (seconds,
/// capped at 30 s) when present and enabled, else exponential base<<attempt
/// plus uniform jitter.
std::chrono::milliseconds compute_backoff(std::size_t attempt,
                                          std::string const& retry_after,
                                          const config& cfg)
{
  if (cfg.honor_retry_after && !retry_after.empty()) {
    try {
      long const secs = std::stol(retry_after);
      if (secs >= 0) {
        return std::min(std::chrono::milliseconds{secs * 1000}, std::chrono::milliseconds{30'000});
      }
    } catch (...) {
      // Non-numeric (HTTP-date) Retry-After: fall through to exponential.
    }
  }
  std::size_t const shift = std::min<std::size_t>(attempt, 16);
  auto const base         = cfg.retry_backoff_base * (std::size_t{1} << shift);
  std::chrono::milliseconds jitter{0};
  if (cfg.retry_jitter.count() > 0) {
    thread_local std::mt19937 rng{std::random_device{}()};
    std::uniform_int_distribution<long> dist(0, cfg.retry_jitter.count());
    jitter = std::chrono::milliseconds{dist(rng)};
  }
  return base + jitter;
}

/// Group destination segments into ranged-GET chunks.  Fuses runs of
/// file-adjacent segments into one contiguous scatter GET capped at
/// @p chunk_size bytes and @p max_n_chunks buffers.  When @p allow_split is
/// true, a single segment larger than @p chunk_size is also split into
/// chunk_size single-buffer pieces (each a parallel GET) — used by the pure
/// host paths; device staging passes false so each caller buffer stays one
/// buffer (its H2D copy maps 1:1 to that allocation), an oversized one simply
/// becoming a standalone single-buffer GET.  Input segments must be in file
/// order; a null-buffer segment (bounce-staged device read) is kept as a
/// standalone single-buffer output and never fused into a scatter group.  Each
/// output segment's @c size is the contiguous file span its buffers cover.
std::vector<io_object_segment> chunk_host_segments(std::span<const io_object_segment> segs,
                                                   size_t chunk_size,
                                                   size_t max_n_chunks,
                                                   bool allow_split = true)
{
  size_t const cs       = std::max<size_t>(chunk_size, 1);
  size_t const max_bufs = std::max<size_t>(max_n_chunks, 1);
  std::vector<io_object_segment> out;
  out.reserve(segs.size());
  for (size_t i = 0; i < segs.size();) {
    auto const& s = segs[i];
    if (s.size == 0) {
      ++i;
      continue;
    }
    if (allow_split && s.size > cs) {
      // Split an oversized contiguous segment into chunk_size pieces.  A null
      // buffer (bounce-staged) stays null per piece — never `nullptr + pos`
      // (UB): each piece is a standalone single-buffer chunk that submit() backs
      // with its own bounce slot, honoring the standalone-null contract above.
      uint8_t* base = s.data();
      for (size_t pos = 0; pos < s.size; pos += cs) {
        size_t const piece = std::min(cs, s.size - pos);
        out.emplace_back(s.offset + pos, piece, base != nullptr ? base + pos : nullptr);
      }
      ++i;
      continue;
    }
    // Greedily fuse following file-adjacent segments into one scatter GET while
    // the fused span and buffer count stay within their caps.  Never fuse across
    // a null buffer: a null-buffer segment (a reactor-bounce-staged device read,
    // e.g. a prefetch-cache gap) must stay a standalone single-buffer chunk so
    // submit() can back it with one pinned bounce slot and its H2D copy resolves
    // to that slot — fusing it would either break the bounce (one slot per chunk)
    // or leave a stale null-derived copy source.
    io_object_segment group{s.offset, s.size, s.data()};
    size_t j = i + 1;
    while (j < segs.size() && group.n_chunks() < max_bufs && segs[j].size > 0 &&
           group.buffers.back().iov_base != nullptr && segs[j].data() != nullptr &&
           group.offset + group.size == segs[j].offset && group.size + segs[j].size <= cs) {
      group.append(iovec{static_cast<void*>(segs[j].data()), segs[j].size});
      ++j;
    }
    out.push_back(std::move(group));
    i = j;
  }
  return out;
}

}  // namespace

std::optional<size_t> content_range_total(std::string const& cr)
{
  constexpr std::string_view kUnit = "bytes";
  std::string_view sv{cr};
  if (sv.size() < kUnit.size()) { return std::nullopt; }
  for (size_t i = 0; i < kUnit.size(); ++i) {
    if (ascii_lower(sv[i]) != kUnit[i]) { return std::nullopt; }
  }
  sv.remove_prefix(kUnit.size());
  while (!sv.empty() && (sv.front() == ' ' || sv.front() == '\t')) {
    sv.remove_prefix(1);
  }
  // The range part must be a satisfied "<first>-<last>", never "*": a leading
  // digit both rejects "bytes */..." and confirms a total follows the '/'.
  if (sv.empty() || sv.front() < '0' || sv.front() > '9') { return std::nullopt; }
  auto const slash = sv.find('/');
  if (slash == std::string_view::npos) { return std::nullopt; }
  std::string_view const total = sv.substr(slash + 1);
  if (total.empty() || total.front() < '0' || total.front() > '9') { return std::nullopt; }
  size_t value = 0;
  for (char const c : total) {
    if (c < '0' || c > '9') { break; }
    value = value * 10 + static_cast<size_t>(c - '0');
  }
  return value;
}

// ---------------------------------------------------------------------------
// construction / lifecycle
// ---------------------------------------------------------------------------

rest_reactor::rest_reactor(std::shared_ptr<reactor_context> ctx, std::string_view tname)
  : _ctx(std::move(ctx)), _tname(tname)
{
  if (!_ctx) { throw std::invalid_argument("rest_reactor: reactor_context must be non-null"); }
  _config = _ctx->cfg();
  if (!_ctx->authorizer()) {
    throw std::invalid_argument("rest_reactor: context authorizer must be non-null");
  }
  if (_config.max_connections == 0) {
    throw std::invalid_argument("rest_reactor: max_connections must be > 0");
  }
  if (_config.max_retry_attempts == 0) { _config.max_retry_attempts = 1; }
  if (_config.max_read_split == 0) { _config.max_read_split = 1; }

  // Touch the process-wide curl context so global init + the shared cache are
  // ready before any handle is created — including the blocking HEAD that
  // rest_ioctx::create_io_object issues before start() is ever called.
  (void)global_curl_context::instance();

  if (_ctx->host_memory_resource() != nullptr) {
    _bounce_slot_size = _ctx->host_memory_resource()->get_block_size();
  }

  // The wakeup fd is cheap and the worker (started in start()) registers it with
  // epoll, so create it up front.  No pinned bounce allocation and no worker
  // thread until start() — keeps a parked (unused) reactor cheap.
  _wakeup_fd = make_event_fd();
}

void rest_reactor::start()
{
  if (_worker.joinable()) { return; }  // already started

  if (_ctx->host_memory_resource() != nullptr) {
    // One pinned bounce buffer per slot (1:1 with the easy-handle pool), since a
    // slot stages at most one device read at a time.
    _bounce_storage = _ctx->host_memory_resource()->allocate_multiple_blocks(
      _config.max_connections * _bounce_slot_size);
  }

  _worker =
    std::jthread([this](const std::stop_token& st) { worker_loop(st); }, _stop_source.get_token());
  if (!_tname.empty()) {
    std::string const full_name = _tname + "_worker";
    pthread_setname_np(_worker.native_handle(), full_name.c_str());
  }
}

rest_reactor::~rest_reactor() { shutdown(); }

void rest_reactor::interrupt()
{
  // Break the worker out of epoll_wait.
  uint64_t one = 1;
  ssize_t rc   = ::write(_wakeup_fd.get(), &one, sizeof(one));
  (void)rc;  // EAGAIN on a saturated counter is fine — a wakeup is already pending.
}

void rest_reactor::shutdown()
{
  if (_worker.joinable()) {
    _stop_source.request_stop();
    interrupt();
    _worker.join();
  }
}

void rest_reactor::enqueue(request_type_ptr req)
{
  auto chunks = req->get_all_chunks();
  enqueue_chunks(chunks);
}

void rest_reactor::enqueue_chunks(std::span<std::unique_ptr<rest_chunked_rx_request>> batch)
{
  if (batch.empty()) { return; }
  if (_config.perf_instrumentation) {
    auto const now = std::chrono::steady_clock::now();
    for (auto& c : batch) {
      if (c) { c->t_enqueue = now; }
    }
  }
  bool const ok = _requests.enqueue_bulk(std::make_move_iterator(batch.data()), batch.size());
  if (!ok) { throw std::runtime_error("rest_reactor::enqueue_chunks: enqueue_bulk failed"); }
  interrupt();
}

// ---------------------------------------------------------------------------
// request preparation (host paths)
// ---------------------------------------------------------------------------

rest_reactor::request_type_ptr rest_reactor::prep_host_rx_request(const reactor_config_type& cfg,
                                                                  const io_object_type& file,
                                                                  const io_object_segment& segment)
{
  if (segment.size == 0) { return rest_rx_request::create({}); }

  // Break a contiguous host read into N parallel single-buffer ranged GETs so
  // the connection pool fetches them concurrently.  N is the largest count
  // <= max_read_split that keeps every piece at least min_chunk_size; a read
  // below single_request_threshold stays a single GET (the extra round-trips
  // would not pay off).  segment.size is distributed as evenly as possible,
  // spreading the remainder over the leading pieces so every byte is covered
  // exactly once.
  constexpr size_t min_chunk_size           = 1UL << 20;  // 1 MiB
  constexpr size_t single_request_threshold = 2UL << 20;  // 2 MiB

  size_t n_chunks = 1;
  if (segment.size >= single_request_threshold) {
    n_chunks = std::min<size_t>(cfg.max_read_split, segment.size / min_chunk_size);
    n_chunks = std::max<size_t>(n_chunks, 1);
  }
  // Keep every ranged GET piece under 4 GiB, matching the uring backend's
  // 32-bit read-length bound, so a single very large object never produces an
  // oversized read on either backend.
  constexpr size_t max_piece_bytes = size_t{1} << 31;  // 2 GiB
  n_chunks = std::max<size_t>(n_chunks, (segment.size + max_piece_bytes - 1) / max_piece_bytes);

  auto manager       = std::make_shared<request_manager>(segment.size, n_chunks);
  auto const obj     = file.object_ref();
  size_t const fsize = file.size();
  uint8_t* const dst = segment.data();

  size_t const base = segment.size / n_chunks;
  size_t const rem  = segment.size % n_chunks;

  std::vector<std::unique_ptr<rest_chunked_rx_request>> chunks;
  chunks.reserve(n_chunks);
  size_t pos = 0;  // byte offset within the segment
  for (size_t c = 0; c < n_chunks; ++c) {
    size_t const piece = base + (c < rem ? 1 : 0);
    auto req           = std::make_unique<rest_chunked_rx_request>();
    req->object        = obj;
    req->chunk         = io_object_segment{segment.offset + pos, piece, dst + pos};
    req->file_size     = fsize;
    req->manager       = manager;
    chunks.push_back(std::move(req));
    pos += piece;
  }
  return rest_rx_request::create(std::move(chunks));
}

rest_reactor::request_type_ptr rest_reactor::prep_host_rxv_request(
  const reactor_config_type& cfg, const io_object_type& file, std::span<io_object_segment> segments)
{
  if (segments.empty()) { return rest_rx_request::create({}); }

  size_t const fsize = file.size();

  // Clamp each segment to the file end (dropping empties), preserving file
  // order, and total the requested bytes (what the future reports).
  std::vector<io_object_segment> clamped;
  clamped.reserve(segments.size());
  size_t bytes_requested = 0;
  for (auto const& s : segments) {
    size_t const c = s.offset < fsize ? std::min(s.size, fsize - s.offset) : 0;
    if (c == 0) { continue; }
    clamped.emplace_back(s.offset, c, s.data());
    bytes_requested += c;
  }
  if (clamped.empty()) { return rest_rx_request::create({}); }

  // Fuse file-adjacent segments into scatter GETs and split oversized ones;
  // grouping never changes the covered byte total.
  auto groups =
    chunk_host_segments(std::span<const io_object_segment>(clamped.data(), clamped.size()),
                        cfg.chunk_size,
                        cfg.max_n_chunks);

  auto manager   = std::make_shared<request_manager>(bytes_requested, groups.size());
  auto const obj = file.object_ref();

  std::vector<std::unique_ptr<rest_chunked_rx_request>> chunks;
  chunks.reserve(groups.size());
  for (auto& g : groups) {
    auto req       = std::make_unique<rest_chunked_rx_request>();
    req->object    = obj;
    req->chunk     = std::move(g);
    req->file_size = fsize;
    req->manager   = manager;
    chunks.push_back(std::move(req));
  }
  return rest_rx_request::create(std::move(chunks));
}

rest_reactor::request_type_ptr rest_reactor::prep_device_rx_request(const reactor_config_type& cfg,
                                                                    const io_object_type& file,
                                                                    uint8_t* dst,
                                                                    size_t offset,
                                                                    size_t size,
                                                                    rmm::cuda_stream_view stream,
                                                                    int device_id)
{
  if (size == 0) { return rest_rx_request::create({}); }
  if (cfg.bounce_block_size == 0) {
    throw std::runtime_error(
      "rest_reactor::prep_device_rx_request: device reads require a host_memory_resource on the "
      "reactor_context for bounce staging");
  }

  // REST has no GPU-direct path, so the read is staged through reactor-owned
  // pinned bounce slots and H2D-copied to dst.  No block alignment: split the
  // requested range (clamped to the file end) into bounce-sized windows, one
  // staged GET + one H2D copy each.
  size_t const fsize = file.size();
  size_t const end   = std::min(offset + size, fsize);
  if (offset >= end) { return rest_rx_request::create({}); }
  size_t const wanted = end - offset;
  size_t const bounce = cfg.bounce_block_size;
  size_t const n_win  = (wanted + bounce - 1) / bounce;

  auto manager   = std::make_shared<request_manager>(wanted, n_win);
  auto const obj = file.object_ref();

  std::vector<std::unique_ptr<rest_chunked_rx_request>> chunks;
  chunks.reserve(n_win);
  for (size_t w = offset; w < end; w += bounce) {
    size_t const rs = std::min(bounce, end - w);
    auto req        = std::make_unique<rest_chunked_rx_request>();
    req->object     = obj;
    req->chunk      = io_object_segment{w, rs};  // null buffer => reactor stages
    req->file_size  = fsize;
    auto cpy        = std::make_unique<device_cpy_request>();
    cpy->stream     = stream;
    cpy->device_id  = device_id;
    cpy->copies.push_back(device_cpy_request::copy{/*dst=*/dst + (w - offset),
                                                   /*src=*/nullptr,  // resolved to the bounce slot
                                                   /*src_off=*/0,
                                                   /*size=*/rs});
    req->cpy_req = std::move(cpy);
    req->manager = manager;
    chunks.push_back(std::move(req));
  }
  return rest_rx_request::create(std::move(chunks));
}

rest_reactor::request_type_ptr rest_reactor::prep_host_to_device_rx_request(
  const reactor_config_type& cfg,
  const io_object_type& file,
  std::span<io_object_segment> segments,
  uint8_t* dst,
  size_t offset,
  size_t size,
  rmm::cuda_stream_view stream,
  int device_id)
{
  // Device read staged through caller-supplied pinned host buffers.  File-
  // adjacent segments are fused into one contiguous scatter GET (whose response
  // lands across their buffers, like uring's readv); each fused buffer then
  // H2D-copies only the part overlapping the device window [offset, req_end)
  // into dst, as a batch of copies issued on one stream.
  if (size == 0 || segments.empty()) { return rest_rx_request::create({}); }

  size_t const fsize   = file.size();
  size_t const req_end = offset + size;
  auto const obj       = file.object_ref();

  // Validate overlap, total the device-buffer bytes each segment fills (the
  // value reported to the caller — not the host read size, which over-reads to
  // the file end), and clamp each segment's read to the file end (single-buffer,
  // file order preserved for the merge).
  size_t bytes_covered = 0;
  std::vector<io_object_segment> clamped;
  clamped.reserve(segments.size());
  for (auto const& s : segments) {
    size_t const lo = std::max(offset, s.offset);
    size_t const hi = std::min({req_end, s.offset + s.size, fsize});
    if (lo >= hi) {
      throw std::runtime_error(
        "rest_reactor::prep_host_to_device_rx_request: segment does not overlap the requested "
        "device range");
    }
    bytes_covered += hi - lo;
    // hi > lo implies s.offset < fsize, so the clamp is well-defined.
    clamped.push_back(io_object_segment{s.offset, std::min(s.size, fsize - s.offset), s.data()});
  }

  // Fuse contiguous buffers into scatter groups (no sub-splitting: each caller
  // buffer must stay one buffer so its H2D copy maps to that allocation).
  auto groups =
    chunk_host_segments(std::span<const io_object_segment>(clamped.data(), clamped.size()),
                        cfg.chunk_size,
                        cfg.max_n_chunks,
                        /*allow_split=*/false);

  auto manager = std::make_shared<request_manager>(bytes_covered, groups.size());

  std::vector<std::unique_ptr<rest_chunked_rx_request>> chunks;
  chunks.reserve(groups.size());
  for (auto& g : groups) {
    // One copy per buffer in the group, each clipped to the device window and
    // carrying an absolute src (the buffers are separate host allocations).
    auto cpy       = std::make_unique<device_cpy_request>();
    cpy->stream    = stream;
    cpy->device_id = device_id;
    cpy->copies.reserve(g.n_chunks());
    size_t file_lo = g.offset;
    for (auto const& b : g.buffers) {
      size_t const file_hi = file_lo + b.iov_len;
      size_t const data_lo = std::max(offset, file_lo);
      size_t const data_hi = std::min(req_end, file_hi);
      if (data_lo < data_hi) {
        // A null buffer is a bounce-staged sub-range: submit() backs the chunk
        // with a pinned bounce slot (set_data) and the H2D copy must read from
        // that slot, so leave src null and carry the intra-buffer offset in
        // src_off — copy_async then resolves src = bounce_buffer + src_off.
        // Encoding a null buffer as `nullptr + off` (a non-null near-null
        // pointer) instead would bypass that bounce fallback and fault the H2D.
        // chunk_host_segments guarantees a null-buffer segment is standalone and
        // single-buffer, so file_lo == g.offset and the bounce holds the whole
        // segment from offset 0.
        bool const bounce_staged = (b.iov_base == nullptr);
        cpy->copies.push_back(device_cpy_request::copy{
          /*dst=*/dst + (data_lo - offset),
          /*src=*/bounce_staged ? nullptr : static_cast<uint8_t*>(b.iov_base) + (data_lo - file_lo),
          /*src_off=*/bounce_staged ? (data_lo - file_lo) : size_t{0},
          /*size=*/data_hi - data_lo});
      }
      file_lo = file_hi;
    }

    auto req       = std::make_unique<rest_chunked_rx_request>();
    req->object    = obj;
    req->chunk     = std::move(g);
    req->file_size = fsize;
    req->cpy_req   = std::move(cpy);
    req->manager   = manager;
    chunks.push_back(std::move(req));
  }
  return rest_rx_request::create(std::move(chunks));
}

// ---------------------------------------------------------------------------
// synchronous paths
// ---------------------------------------------------------------------------

size_t rest_reactor::host_read(const io_object_type& file, size_t offset, size_t size, uint8_t* dst)
{
  if (size == 0) { return 0; }
  size = std::min(size, file.size() > offset ? file.size() - offset : size_t{0});
  if (size == 0) { return 0; }

  // Serve reads fully inside the suffix-range footer stash locally (the parquet
  // trailer/footer reads after a probe); a straddling read falls through to a GET.
  if (auto const& stash = file.stash(); stash) {
    size_t const lo = file.stash_window_lo();
    size_t const hi = lo + stash->size();
    if (offset >= lo && offset + size <= hi) {
      std::memcpy(dst, stash->data() + (offset - lo), size);
      return size;
    }
  }

  // Drive the blocking read through the worker's async pipeline (pooled
  // connections, parallel ranged GETs, the shared retry/backoff policy) and
  // synchronize on its future — rather than a one-shot easy handle that pays a
  // full TCP+TLS handshake per call and duplicates the retry logic.  Build the
  // request, grab its future BEFORE enqueue (which moves the chunks out), then
  // block: get() rethrows the first reported error or returns the byte count.
  auto req = prep_host_rx_request(_config, file, io_object_segment{offset, size, dst});
  auto fut = req->get_future();
  enqueue(std::move(req));
  return std::move(fut).get();
}

rest_perf_snapshot rest_reactor::perf_snapshot() const noexcept
{
  rest_perf_snapshot s;
  s.chunk_get_ns_total       = _perf.chunk_get_ns_total.load(std::memory_order_relaxed);
  s.chunk_get_count          = _perf.chunk_get_count.load(std::memory_order_relaxed);
  s.chunk_get_ns_max         = _perf.chunk_get_ns_max.load(std::memory_order_relaxed);
  s.queue_wait_ns_total      = _perf.queue_wait_ns_total.load(std::memory_order_relaxed);
  s.queue_wait_count         = _perf.queue_wait_count.load(std::memory_order_relaxed);
  s.ttfb_ns                  = _perf.ttfb_ns.load(std::memory_order_relaxed);
  s.h2d_observed_ns_total    = _perf.h2d_observed_ns_total.load(std::memory_order_relaxed);
  s.h2d_observed_count       = _perf.h2d_observed_count.load(std::memory_order_relaxed);
  s.h2d_observed_ns_max      = _perf.h2d_observed_ns_max.load(std::memory_order_relaxed);
  s.retries_total            = _perf.retries_total.load(std::memory_order_relaxed);
  s.terminal_failures_total  = _perf.terminal_failures_total.load(std::memory_order_relaxed);
  s.device_stream_sync_total = _perf.device_stream_sync_total.load(std::memory_order_relaxed);
  s.payload_bytes_read_total = _perf.payload_bytes_read_total.load(std::memory_order_relaxed);
  return s;
}

size_t rest_reactor::head_object_size(std::string_view bucket, std::string_view key)
{
  s3::s3_object_ref const obj{std::string(bucket), std::string(key)};
  std::string last_error;
  for (std::size_t attempt = 0; attempt < _config.max_retry_attempts; ++attempt) {
    header_capture hc;
    auto const authd =
      _ctx->authorizer()->authorize(obj, s3::s3_request_method::HEAD, presign_ttl(_config));

    curl_easy_ptr h{curl_easy_init()};
    if (!h) { throw std::runtime_error("rest_reactor::head_object_size: curl_easy_init failed"); }
    configure_easy_handle(h.get(), global_curl_context::instance().share_handle());
    apply_request_opts(h.get(), _config);

    curl_slist_ptr hdrs = build_header_list(authd.headers, nullptr);
    SIRIUS_CURL_CHECK(curl_easy_setopt(h.get(), CURLOPT_URL, authd.url.c_str()));
    SIRIUS_CURL_CHECK(curl_easy_setopt(h.get(), CURLOPT_NOBODY, 1L));
    SIRIUS_CURL_CHECK(curl_easy_setopt(h.get(), CURLOPT_HTTPHEADER, hdrs.get()));
    SIRIUS_CURL_CHECK(curl_easy_setopt(h.get(), CURLOPT_WRITEFUNCTION, &write_discard));
    SIRIUS_CURL_CHECK(curl_easy_setopt(h.get(), CURLOPT_HEADERFUNCTION, &capture_header));
    SIRIUS_CURL_CHECK(curl_easy_setopt(h.get(), CURLOPT_HEADERDATA, &hc));

    CURLcode const rc = curl_easy_perform(h.get());
    long status       = 0;
    curl_easy_getinfo(h.get(), CURLINFO_RESPONSE_CODE, &status);

    if (rc == CURLE_OK && status == 200) {
      curl_off_t cl = -1;
      curl_easy_getinfo(h.get(), CURLINFO_CONTENT_LENGTH_DOWNLOAD_T, &cl);
      if (cl < 0) {
        _perf.terminal_failures_total.fetch_add(1, std::memory_order_relaxed);
        throw std::runtime_error("rest_reactor::head_object_size: missing Content-Length for " +
                                 obj.bucket + "/" + obj.key);
      }
      return static_cast<size_t>(cl);
    }

    last_error =
      rc != CURLE_OK ? std::string(curl_easy_strerror(rc)) : ("HTTP " + std::to_string(status));
    bool const retriable =
      (rc != CURLE_OK && is_retriable_curl(rc)) || (rc == CURLE_OK && is_retriable_status(status));
    if (!retriable) {
      _perf.terminal_failures_total.fetch_add(1, std::memory_order_relaxed);
      throw std::runtime_error("rest_reactor::head_object_size: " + last_error + " for " +
                               obj.bucket + "/" + obj.key);
    }
    if (attempt + 1 < _config.max_retry_attempts) {
      _perf.retries_total.fetch_add(1, std::memory_order_relaxed);
      SIRIUS_LOG_WARN("rest_reactor::head_object_size: retrying {}/{} after {} (attempt {}/{})",
                      obj.bucket,
                      obj.key,
                      last_error,
                      attempt + 1,
                      _config.max_retry_attempts);
      std::this_thread::sleep_for(compute_backoff(attempt, hc.retry_after, _config));
    }
  }
  _perf.terminal_failures_total.fetch_add(1, std::memory_order_relaxed);
  throw std::runtime_error("rest_reactor::head_object_size: exhausted retries (" + last_error +
                           ") for " + obj.bucket + "/" + obj.key);
}

footer_probe rest_reactor::fetch_footer_suffix(std::string_view bucket,
                                               std::string_view key,
                                               std::size_t n)
{
  footer_probe probe;
  if (n == 0) { return probe; }

  s3::s3_object_ref const obj{std::string(bucket), std::string(key)};
  std::string last_error;
  for (std::size_t attempt = 0; attempt < _config.max_retry_attempts; ++attempt) {
    suffix_sink sink;
    sink.cap = n;
    auto const authd =
      _ctx->authorizer()->authorize(obj, s3::s3_request_method::GET, presign_ttl(_config));

    curl_easy_ptr h{curl_easy_init()};
    if (!h) {
      throw std::runtime_error("rest_reactor::fetch_footer_suffix: curl_easy_init failed");
    }
    configure_easy_handle(h.get(), global_curl_context::instance().share_handle());
    apply_request_opts(h.get(), _config);

    std::string const range = suffix_range_header(n);
    curl_slist_ptr hdrs     = build_header_list(authd.headers, &range);
    SIRIUS_CURL_CHECK(curl_easy_setopt(h.get(), CURLOPT_URL, authd.url.c_str()));
    SIRIUS_CURL_CHECK(curl_easy_setopt(h.get(), CURLOPT_HTTPHEADER, hdrs.get()));
    SIRIUS_CURL_CHECK(curl_easy_setopt(h.get(), CURLOPT_WRITEFUNCTION, &suffix_write_cb));
    SIRIUS_CURL_CHECK(curl_easy_setopt(h.get(), CURLOPT_WRITEDATA, &sink));
    SIRIUS_CURL_CHECK(curl_easy_setopt(h.get(), CURLOPT_HEADERFUNCTION, &suffix_header_cb));
    SIRIUS_CURL_CHECK(curl_easy_setopt(h.get(), CURLOPT_HEADERDATA, &sink));

    auto const t0     = std::chrono::steady_clock::now();
    CURLcode const rc = curl_easy_perform(h.get());
    long status       = 0;
    curl_easy_getinfo(h.get(), CURLINFO_RESPONSE_CODE, &status);

    // payload_bytes_read_total is always-on and per-attempt (see the async
    // worker's finish()), so credit every attempt's wire bytes outside the
    // perf_instrumentation gate.
    _perf.payload_bytes_read_total.fetch_add(sink.total_received, std::memory_order_relaxed);

    // suffix_write_cb aborts any non-206 body, so a CURLE_WRITE_ERROR here is our
    // own doing and the HTTP status is still valid; only a different curl error
    // (no HTTP status) is a genuine transport failure.
    if (rc != CURLE_OK && rc != CURLE_WRITE_ERROR) {
      last_error = std::string(curl_easy_strerror(rc));
      if (!is_retriable_curl(rc)) {
        _perf.terminal_failures_total.fetch_add(1, std::memory_order_relaxed);
        throw std::runtime_error("rest_reactor::fetch_footer_suffix: " + last_error + " for " +
                                 obj.bucket + "/" + obj.key);
      }
    } else if (status == 206) {
      // Trust the 206 only when the window origin and total both parse and the
      // delivered byte count matches exactly; an unverifiable 206 (missing /
      // "*" Content-Range) reports an empty probe so the caller HEADs instead.
      auto const total = content_range_total(sink.content_range);
      auto const start = content_range_start(sink.content_range);
      if (total && start && *start <= *total && sink.data.size() == *total - *start) {
        // Account the footer suffix GET the same way the async chunk path does
        // (it replaces the tail+body GETs that used to run through the pipeline),
        // so bind-time footer reads stay visible in the perf snapshot.
        if (_config.perf_instrumentation) {
          auto const get_ns =
            static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                         std::chrono::steady_clock::now() - t0)
                                         .count());
          _perf.chunk_get_ns_total.fetch_add(get_ns, std::memory_order_relaxed);
          _perf.chunk_get_count.fetch_add(1, std::memory_order_relaxed);
          atomic_max_relaxed(_perf.chunk_get_ns_max, get_ns);
          std::uint64_t expected = 0;
          _perf.ttfb_ns.compare_exchange_strong(expected, get_ns, std::memory_order_relaxed);
        }
        probe.object_size = *total;
        probe.window_lo   = *start;
        probe.bytes       = std::make_shared<const std::vector<std::uint8_t>>(std::move(sink.data));
      }
      return probe;
    } else if (status == 200 || status == 416) {
      // Range ignored (full body) or unsatisfiable (416): probe unusable but the
      // object exists — report empty so the caller falls back to a HEAD.
      return probe;
    } else if (is_retriable_status(status)) {
      last_error = "HTTP " + std::to_string(status);
    } else {
      // 404 / 403 / 401 / ... — an error a HEAD would not recover from either.
      _perf.terminal_failures_total.fetch_add(1, std::memory_order_relaxed);
      throw std::runtime_error("rest_reactor::fetch_footer_suffix: HTTP " + std::to_string(status) +
                               " for " + obj.bucket + "/" + obj.key);
    }
    if (attempt + 1 < _config.max_retry_attempts) {
      _perf.retries_total.fetch_add(1, std::memory_order_relaxed);
      SIRIUS_LOG_WARN("rest_reactor::fetch_footer_suffix: retrying {}/{} after {} (attempt {}/{})",
                      obj.bucket,
                      obj.key,
                      last_error,
                      attempt + 1,
                      _config.max_retry_attempts);
      std::this_thread::sleep_for(compute_backoff(attempt, sink.retry_after, _config));
    }
  }
  _perf.terminal_failures_total.fetch_add(1, std::memory_order_relaxed);
  throw std::runtime_error("rest_reactor::fetch_footer_suffix: exhausted retries (" + last_error +
                           ") for " + obj.bucket + "/" + obj.key);
}

// ---------------------------------------------------------------------------
// capabilities / factory
// ---------------------------------------------------------------------------

bool rest_reactor::supports(std::string_view path)
{
  try {
    auto const parsed = sirius::io::parse(path);
    return parsed.scheme == "s3";
  } catch (...) {
    return false;
  }
}

std::unique_ptr<rest_reactor::io_object_type> rest_reactor::create_io_object(std::string /*path*/)
{
  // The object size requires a HEAD round-trip and the authorizer, both of
  // which live on the reactor instance — see rest_ioctx::create_io_object.
  throw std::logic_error(
    "rest_reactor::create_io_object: use rest_ioctx::create_io_object (needs HEAD + authorizer)");
}

std::vector<cudf::io::text::byte_range_info> rest_reactor::align_and_coalesce(
  std::span<const cudf::io::text::byte_range_info> ranges, std::optional<size_t> alignment)
{
  // No physical block alignment for REST: honor a caller alignment >= 1 as a
  // lower bound, otherwise treat alignment as 1 (byte) — i.e. pure coalescing.
  size_t const align = std::max<size_t>(alignment.value_or(1), 1);

  std::vector<cudf::io::text::byte_range_info> aligned;
  aligned.reserve(ranges.size());
  for (auto const& r : ranges) {
    if (r.size() <= 0) { continue; }
    auto const offset  = static_cast<size_t>(r.offset());
    auto const end     = offset + static_cast<size_t>(r.size());
    size_t const start = (offset / align) * align;
    size_t const stop  = ((end + align - 1) / align) * align;
    aligned.emplace_back(static_cast<int64_t>(start), static_cast<int64_t>(stop - start));
  }
  if (aligned.empty()) { return aligned; }

  std::sort(aligned.begin(), aligned.end(), [](auto const& a, auto const& b) {
    return a.offset() < b.offset();
  });

  std::vector<cudf::io::text::byte_range_info> coalesced;
  coalesced.reserve(aligned.size());
  coalesced.push_back(aligned.front());
  for (size_t i = 1; i < aligned.size(); ++i) {
    auto& last            = coalesced.back();
    auto const last_start = static_cast<size_t>(last.offset());
    auto const last_end   = last_start + static_cast<size_t>(last.size());
    auto const cur_start  = static_cast<size_t>(aligned[i].offset());
    auto const cur_end    = cur_start + static_cast<size_t>(aligned[i].size());
    if (cur_start <= last_end) {  // overlap or adjacency
      size_t const new_end = std::max(last_end, cur_end);
      last                 = {last.offset(), static_cast<int64_t>(new_end - last_start)};
    } else {
      coalesced.push_back(aligned[i]);
    }
  }
  return coalesced;
}

// ---------------------------------------------------------------------------
// worker loop (epoll + curl_multi_socket_action engine)
// ---------------------------------------------------------------------------

namespace {

/// One reusable I/O slot — the unit of concurrency, mirroring uring_reactor's
/// io_slot.  Owns its persistent easy handle and its dedicated pinned bounce
/// buffer (1:1 with the slot; null when no host memory resource was given), and
/// carries the per-request state whose address curl references for the duration
/// of a transfer (the URL string, header list, write/header targets).  The
/// slot_pool gates which slots are in use; a slot holds its acquisition token
/// while busy, so a slot is also its own bounce-buffer reservation.
struct io_slot {
  // Persistent slot resources (set once at pool creation, never reset):
  curl_easy_ptr easy;
  uint8_t* bounce{nullptr};

  // Held while the slot is in use; releasing it returns the slot to the pool.
  // For a bounce-staged device read it is moved into `copying` so the slot is
  // not reused until the H2D copy off its bounce buffer completes.
  slot_pool::token token;

  // Per-request state (cleared by reset() between requests):
  std::unique_ptr<rest_chunked_rx_request> req;
  std::string url;  // backs CURLOPT_URL
  curl_slist_ptr headers;
  buf_sink sink;
  header_capture hc;

  /// Clear the per-request state and release the slot back to its pool (a
  /// no-op when the token was already moved into `copying`).  The handle and
  /// bounce buffer persist.
  void reset() noexcept
  {
    req.reset();
    url.clear();
    headers.reset();
    sink = buf_sink{};
    hc.reset();
    token = {};
  }
};

/// Worker-loop state reachable from curl's C socket/timer callbacks.
struct worker_state {
  CURLM* multi{nullptr};
  int epoll_fd{-1};
  int curl_timer_fd{-1};
};

/// CURLMOPT_SOCKETFUNCTION: mirror curl's interest in @p s into epoll.
int rest_socket_cb(CURL* /*easy*/, curl_socket_t s, int what, void* userp, void* socketp)
{
  auto* ws = static_cast<worker_state*>(userp);
  if (what == CURL_POLL_REMOVE) {
    // Best-effort delete; the socket may already be gone (ENOENT/EBADF).
    ::epoll_ctl(ws->epoll_fd, EPOLL_CTL_DEL, s, nullptr);
    return 0;
  }
  uint32_t events = 0;
  if (what == CURL_POLL_IN || what == CURL_POLL_INOUT) { events |= EPOLLIN; }
  if (what == CURL_POLL_OUT || what == CURL_POLL_INOUT) { events |= EPOLLOUT; }
  epoll_event ev{};
  ev.events  = events;
  ev.data.fd = s;
  // socketp is curl's per-socket user pointer: null on first sighting (ADD),
  // non-null thereafter (MOD).  We only use it as an "already added" marker.
  int const op = (socketp == nullptr) ? EPOLL_CTL_ADD : EPOLL_CTL_MOD;
  if (socketp == nullptr) { curl_multi_assign(ws->multi, s, ws); }
  ::epoll_ctl(ws->epoll_fd, op, s, &ev);
  return 0;
}

/// CURLMOPT_TIMERFUNCTION: arm/disarm the curl timerfd.
int rest_timer_cb(CURLM* /*multi*/, long timeout_ms, void* userp)
{
  auto* ws = static_cast<worker_state*>(userp);
  itimerspec its{};  // all-zero => disarm
  if (timeout_ms == 0) {
    its.it_value.tv_nsec = 1;  // fire essentially immediately
  } else if (timeout_ms > 0) {
    its.it_value.tv_sec  = timeout_ms / 1000;
    its.it_value.tv_nsec = (timeout_ms % 1000) * 1'000'000L;
  }
  ::timerfd_settime(ws->curl_timer_fd, 0, &its, nullptr);
  return 0;
}

/// Drain (and discard) all pending reads from a non-blocking fd.
void drain_fd(int fd) noexcept
{
  uint64_t v = 0;
  while (::read(fd, &v, sizeof(v)) > 0) {}
}

}  // namespace

void rest_reactor::worker_loop(const std::stop_token& stop_token)
{
  constexpr int MAX_EVENTS = 64;

  // Wake the loop out of epoll_wait when shutdown is requested.
  std::stop_callback const stop_cb(stop_token, [this] { interrupt(); });

  try {
    curl_multi_ptr multi{curl_multi_init()};
    if (!multi) { throw std::runtime_error("rest_reactor: curl_multi_init failed"); }

    file_descriptor epoll_fd        = make_epoll_fd();
    file_descriptor curl_timer_fd   = make_timer_fd();
    file_descriptor retry_timer_fd  = make_timer_fd();
    file_descriptor upkeep_timer_fd = make_timer_fd();
    worker_state ws{multi.get(), epoll_fd.get(), curl_timer_fd.get()};

    SIRIUS_CURLM_CHECK(curl_multi_setopt(multi.get(), CURLMOPT_SOCKETFUNCTION, &rest_socket_cb));
    SIRIUS_CURLM_CHECK(curl_multi_setopt(multi.get(), CURLMOPT_SOCKETDATA, &ws));
    SIRIUS_CURLM_CHECK(curl_multi_setopt(multi.get(), CURLMOPT_TIMERFUNCTION, &rest_timer_cb));
    SIRIUS_CURLM_CHECK(curl_multi_setopt(multi.get(), CURLMOPT_TIMERDATA, &ws));
    // Parallel TCP streams rather than HTTP/2 multiplexing: with multiplexing
    // off, curl opens a separate connection per concurrent transfer (up to
    // MAX_HOST_CONNECTIONS == max_connections) instead of funneling every GET
    // over one h2 connection bounded by the server's stream limit.  Against S3,
    // independent connections give better aggregate throughput on large ranged
    // reads, and the slot pool's "N connections" model then actually holds.
    SIRIUS_CURLM_CHECK(
      curl_multi_setopt(multi.get(), CURLMOPT_PIPELINING, static_cast<long>(CURLPIPE_NOTHING)));
    SIRIUS_CURLM_CHECK(curl_multi_setopt(
      multi.get(), CURLMOPT_MAX_HOST_CONNECTIONS, static_cast<long>(_config.max_connections)));
    SIRIUS_CURLM_CHECK(curl_multi_setopt(
      multi.get(), CURLMOPT_MAXCONNECTS, static_cast<long>(_config.max_connections)));

    auto epoll_add = [&](int fd, uint32_t events) {
      epoll_event ev{};
      ev.events  = events;
      ev.data.fd = fd;
      if (::epoll_ctl(epoll_fd.get(), EPOLL_CTL_ADD, fd, &ev) != 0) {
        throw std::runtime_error(std::string("rest_reactor: epoll_ctl ADD failed: ") +
                                 std::strerror(errno));
      }
    };
    epoll_add(_wakeup_fd.get(), EPOLLIN);
    epoll_add(curl_timer_fd.get(), EPOLLIN);
    epoll_add(retry_timer_fd.get(), EPOLLIN);
    epoll_add(upkeep_timer_fd.get(), EPOLLIN);

    // Idle-connection keepalive: fire the upkeep timer periodically so idle
    // pooled connections get an HTTP/2 PING (gated per-connection by
    // CURLOPT_UPKEEP_INTERVAL_MS).  Disabled when upkeep_interval is zero.
    long const upkeep_ms = static_cast<long>(_config.upkeep_interval.count());
    if (upkeep_ms > 0) {
      itimerspec its{};
      its.it_value.tv_sec = its.it_interval.tv_sec = upkeep_ms / 1000;
      its.it_value.tv_nsec = its.it_interval.tv_nsec = (upkeep_ms % 1000) * 1'000'000L;
      ::timerfd_settime(upkeep_timer_fd.get(), 0, &its, nullptr);
    }

    // Slot pool: max_connections reusable io_slots, each owning its easy handle
    // (configured once with the static performance + TLS/timeout options) and,
    // when device staging is enabled, its dedicated pinned bounce buffer.  The
    // connection cache is shared via a worker-local curl_share so idle
    // connections survive across handles and are reachable by curl_easy_upkeep
    // — safe because only this worker thread touches it.  A slot_pool gates
    // which slots are in use (a free slot also owns a free bounce buffer); each
    // handle carries its slot index in CURLOPT_PRIVATE so a completion maps back
    // to its slot in O(1) with no per-request allocation or hashing.
    curl_share worker_share{/*share_connections=*/true};
    std::vector<io_slot> slots(_config.max_connections);
    slot_pool pool{_config.max_connections};

    std::vector<uint8_t*> bounce_bufs;  // one per slot when device staging is on
    if (_bounce_storage) {
      auto blocks = _bounce_storage->get_blocks();
      bounce_bufs.reserve(blocks.size());
      for (auto* b : blocks) {
        bounce_bufs.push_back(reinterpret_cast<uint8_t*>(b));
      }
    }

    for (std::size_t i = 0; i < _config.max_connections; ++i) {
      curl_easy_ptr h{curl_easy_init()};
      if (!h) { throw std::runtime_error("rest_reactor: curl_easy_init failed"); }
      configure_easy_handle(
        h.get(), worker_share.get(), upkeep_ms, static_cast<long>(_config.conn_max_age.count()));
      apply_request_opts(h.get(), _config);
      // Stable per-handle identity (the slot index); never changes across the
      // requests this handle serves.
      SIRIUS_CURL_CHECK(curl_easy_setopt(
        h.get(), CURLOPT_PRIVATE, reinterpret_cast<void*>(static_cast<intptr_t>(i))));
      slots[i].easy   = std::move(h);
      slots[i].bounce = i < bounce_bufs.size() ? bounce_bufs[i] : nullptr;
    }

    int running  = 0;
    int inflight = 0;  // GETs currently added to the multi (excludes parked H2D)

    // -- retry scheduling --------------------------------------------------
    // A min-heap of chunks keyed by their due time.  When retry_timer_fd fires,
    // every chunk whose due time has passed moves into `ready`, which submit()
    // drains ahead of the inbound queue.  The timer is always armed to the
    // earliest due time in the heap.
    struct retry_entry {
      std::chrono::steady_clock::time_point due;
      std::unique_ptr<rest_chunked_rx_request> req;
    };
    auto const retry_cmp = [](const retry_entry& a, const retry_entry& b) { return a.due > b.due; };
    std::vector<retry_entry> retry_heap;
    std::deque<std::unique_ptr<rest_chunked_rx_request>> ready;

    // -- device staging ----------------------------------------------------
    // A reactor-staged device read (null-buffer chunk) lands in its slot's
    // bounce buffer, then an async H2D copy moves the bytes to the device.  The
    // slot's pool token is held in `copying` (alongside the copy's event) until
    // that copy completes — so the slot, and its bounce buffer, are not reused
    // meanwhile.  Crucially, the chunk is NOT reported complete until the event
    // is observed done: the bytes are not actually on the device until the copy
    // off the bounce buffer finishes, so the manager (and the byte count to
    // credit) ride along here and chunk_complete / report_error is deferred to
    // poll_copy_completions.  Events are pooled per device, indexed by slot.
    struct parked_copy {
      slot_pool::token token;
      cucascade::cuda::cuda_event* event{nullptr};
      std::shared_ptr<request_manager> manager;
      std::size_t bytes{0};
    };
    std::unordered_map<int, std::vector<cucascade::cuda::cuda_event>> copy_events;
    std::vector<parked_copy> copying;
    if (_bounce_storage) {
      int const n_dev = rmm::get_num_cuda_devices();
      for (int d = 0; d < n_dev; ++d) {
        rmm::cuda_set_device_raii const guard{rmm::cuda_device_id{d}};
        auto& evs = copy_events[d];
        evs.reserve(_config.max_connections);
        std::generate_n(std::back_inserter(evs), _config.max_connections, [] {
          return cucascade::cuda::cuda_event{cudaEventDisableTiming};
        });
      }
    }

    auto poll_copy_completions = [&]() {
      using query_status = cucascade::cuda::event::query_result;
      // Credit (or fail) each bounce-staged chunk only now that its H2D copy is
      // actually done — the device bytes were not valid until this point.  On
      // success report the chunk complete; on a copy error fail the request
      // rather than silently dropping it.  Either way the entry is erased, which
      // drops its token and returns the slot (and bounce buffer) to the pool.
      for (auto it = copying.begin(); it != copying.end();) {
        query_status const st = it->event->query();
        if (st == query_status::in_progress) {
          ++it;
          continue;
        }
        if (st == query_status::success) {
          it->manager->chunk_complete(it->bytes);
        } else {
          _perf.terminal_failures_total.fetch_add(1, std::memory_order_relaxed);
          it->manager->report_error(
            std::make_exception_ptr(std::runtime_error("rest_reactor: device H2D copy failed")));
        }
        it = copying.erase(it);
      }
    };

    auto arm_retry_timer = [&]() {
      itimerspec its{};  // all-zero => disarm
      if (!retry_heap.empty()) {
        auto const now = std::chrono::steady_clock::now();
        auto const due = retry_heap.front().due;
        auto ns        = due > now
                           ? std::chrono::duration_cast<std::chrono::nanoseconds>(due - now).count()
                           : std::int64_t{1};
        if (ns <= 0) { ns = 1; }
        its.it_value.tv_sec  = ns / 1'000'000'000;
        its.it_value.tv_nsec = ns % 1'000'000'000;
      }
      ::timerfd_settime(retry_timer_fd.get(), 0, &its, nullptr);
    };

    // Re-enqueue a chunk for a later attempt after a backoff.  @p is_auth picks
    // the bounded HTTP-403 (re-presign) budget instead of the transient-error
    // budget, so a stale presigned URL gets a few fresh-signature retries while
    // a genuine AccessDenied still fails fast.
    auto schedule_retry = [&](std::unique_ptr<rest_chunked_rx_request> req,
                              std::string const& retry_after,
                              bool is_auth,
                              std::string const& reason) {
      std::size_t& counter = is_auth ? req->auth_attempt : req->attempt;
      std::size_t const max_attempts =
        is_auth ? _config.max_auth_retry_attempts : _config.max_retry_attempts;
      if (counter + 1 >= max_attempts) {
        _perf.terminal_failures_total.fetch_add(1, std::memory_order_relaxed);
        req->manager->report_error(std::make_exception_ptr(std::runtime_error(
          "rest_reactor: exhausted retries for " + req->object.bucket + "/" + req->object.key)));
        return;
      }
      // Backoff tracks the transient-attempt count; an auth retry re-presigns
      // and reuses the current step without inflating it.
      auto const delay = compute_backoff(req->attempt, retry_after, _config);
      _perf.retries_total.fetch_add(1, std::memory_order_relaxed);
      SIRIUS_LOG_WARN("rest_reactor: retrying {}/{} after {} (attempt {}/{})",
                      req->object.bucket,
                      req->object.key,
                      reason,
                      counter + 1,
                      max_attempts);
      counter += 1;
      retry_heap.push_back(retry_entry{std::chrono::steady_clock::now() + delay, std::move(req)});
      std::push_heap(retry_heap.begin(), retry_heap.end(), retry_cmp);
      arm_retry_timer();
    };

    auto setup_easy = [&](io_slot& s) {
      CURL* const h = s.easy.get();
      auto authd    = _ctx->authorizer()->authorize(
        s.req->object, s3::s3_request_method::GET, presign_ttl(_config));
      s.url           = std::move(authd.url);
      s.sink.buffers  = std::span<iovec>(s.req->chunk.buffers);
      s.sink.capacity = s.req->chunk.size;
      s.sink.reset();
      s.hc.reset();
      std::string const range = range_header(s.req->chunk.offset, s.req->chunk.size);
      s.headers               = build_header_list(authd.headers, &range);
      SIRIUS_CURL_CHECK(curl_easy_setopt(h, CURLOPT_URL, s.url.c_str()));
      SIRIUS_CURL_CHECK(curl_easy_setopt(h, CURLOPT_HTTPHEADER, s.headers.get()));
      SIRIUS_CURL_CHECK(curl_easy_setopt(h, CURLOPT_WRITEFUNCTION, &write_to_sink));
      SIRIUS_CURL_CHECK(curl_easy_setopt(h, CURLOPT_WRITEDATA, &s.sink));
      SIRIUS_CURL_CHECK(curl_easy_setopt(h, CURLOPT_HEADERFUNCTION, &capture_header));
      SIRIUS_CURL_CHECK(curl_easy_setopt(h, CURLOPT_HEADERDATA, &s.hc));
    };

    auto submit = [&]() {
      // Acquire a slot (and thus a bounce buffer) up front; an invalid token
      // means all slots are busy.  A token taken for a skipped/empty dequeue is
      // released by its RAII destructor at continue/break.
      while (true) {
        slot_pool::token tok = pool.try_acquire_token();
        if (!tok) { break; }

        // Submission priority: due retries (ready) ahead of fresh inbound work
        // so a backed-off request is not starved by new ones.
        std::unique_ptr<rest_chunked_rx_request> dr;
        if (!ready.empty()) {
          dr = std::move(ready.front());
          ready.pop_front();
        } else if (!_requests.try_dequeue(dr)) {
          break;
        }
        if (!dr) { continue; }
        if (dr->manager->has_error()) {
          dr.reset();
          continue;
        }

        int const i = tok.slot_index();
        io_slot& s  = slots[static_cast<size_t>(i)];
        // A bounce-staged device read must re-bind to THIS slot's bounce on
        // every (re)submission.  A retried request still carries the previous
        // attempt's set_data (chunk.data() == that now-freed slot's bounce) and
        // staged_through_bounce, so without the second term needs_bounce would
        // be false on retry: the sink would fill — and the parked H2D would
        // drain from — a foreign slot's bounce while this slot's token is parked.
        bool const needs_bounce =
          dr->is_device() && (!dr->chunk.is_buffer_allocated() || dr->staged_through_bounce);
        if (needs_bounce && s.bounce == nullptr) {
          // Device staging requested but no host memory resource was configured.
          _perf.terminal_failures_total.fetch_add(1, std::memory_order_relaxed);
          dr->manager->report_error(std::make_exception_ptr(std::runtime_error(
            "rest_reactor: device staging unavailable (no host memory resource)")));
          dr.reset();
          continue;  // token released, slot stays free
        }

        s.req = std::move(dr);
        if (_config.perf_instrumentation) {
          auto const now  = std::chrono::steady_clock::now();
          s.req->t_submit = now;
          if (s.req->attempt == 0) {
            auto const wait_ns = static_cast<std::uint64_t>(
              std::chrono::duration_cast<std::chrono::nanoseconds>(now - s.req->t_enqueue).count());
            _perf.queue_wait_ns_total.fetch_add(wait_ns, std::memory_order_relaxed);
            _perf.queue_wait_count.fetch_add(1, std::memory_order_relaxed);
          }
        }
        if (needs_bounce) {
          s.req->chunk.set_data(s.bounce);
          // Record bounce-staging before finish() reads it: set_data has just
          // made is_buffer_allocated() true, so finish() must rely on this flag
          // (not the chunk) to take the event-synchronized recycle path and hold
          // the slot's bounce until the H2D copy off it completes.
          s.req->staged_through_bounce = true;
        }
        s.token = std::move(tok);  // slot holds its token while in use
        setup_easy(s);
        curl_multi_add_handle(multi.get(), s.easy.get());
        ++inflight;
      }
    };

    // Handle one completed transfer.  Returns true iff the slot was parked in
    // `copying` (held until its H2D copy finishes); otherwise the caller
    // recycles the slot immediately.
    auto finish = [&](int i, CURLcode rc, long status) -> bool {
      io_slot& s = slots[static_cast<size_t>(i)];
      // Always-on: credit the HTTP response body bytes this attempt received
      // (write_to_sink already summed them in sink.total_received) BEFORE any
      // success / retry / terminal branching, so a 503 -> retry -> 206 counts
      // both bodies and short / failed reads are reflected in the byte budget.
      _perf.payload_bytes_read_total.fetch_add(s.sink.total_received, std::memory_order_relaxed);
      auto& req           = *s.req;
      bool const ok_range = (status == 206) || (status == 200 && req.chunk.offset == 0);
      // A 206 must report, via Content-Range, that it delivered the exact range
      // we asked for.  The write callback scatters the body in arrival order
      // with no idea what file offset it covers, so a 206 whose body is a
      // *different* range than requested (a misbehaving proxy/CDN/cache, or a
      // multipart/byteranges response curl does not parse) would otherwise be
      // accepted as correct data — silent corruption.  Validate the start
      // offset before trusting any 206 body; a mismatch (or an unparsable /
      // missing Content-Range) is a terminal error, not a transient one.
      if (rc == CURLE_OK && status == 206) {
        auto const start = content_range_start(s.hc.content_range);
        if (!start || *start != req.chunk.offset) {
          _perf.terminal_failures_total.fetch_add(1, std::memory_order_relaxed);
          req.manager->report_error(std::make_exception_ptr(std::runtime_error(
            "rest_reactor: 206 Content-Range mismatch (got '" + s.hc.content_range +
            "', requested offset " + std::to_string(req.chunk.offset) + ") for " +
            req.object.bucket + "/" + req.object.key)));
          return false;
        }
      }
      if (rc == CURLE_OK && ok_range && s.sink.written >= req.chunk.size) {
        if (_config.perf_instrumentation) {
          auto const get_ns =
            static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                         std::chrono::steady_clock::now() - req.t_submit)
                                         .count());
          _perf.chunk_get_ns_total.fetch_add(get_ns, std::memory_order_relaxed);
          _perf.chunk_get_count.fetch_add(1, std::memory_order_relaxed);
          atomic_max_relaxed(_perf.chunk_get_ns_max, get_ns);
          std::uint64_t expected = 0;
          _perf.ttfb_ns.compare_exchange_strong(expected, get_ns, std::memory_order_relaxed);
        }
        if (req.is_device()) {
          // Issue the async H2D copy.  Bounce-staged reads need a CUDA event so
          // the slot (its bounce buffer) is only reused once the copy off it
          // completes; caller-buffer reads detach (the caller's stream orders
          // the copy).
          bool const needs_event          = req.needs_event_for_synchronization();
          cucascade::cuda::cuda_event* ev = nullptr;
          cudaEvent_t cev                 = nullptr;
          if (needs_event) {
            ev  = &copy_events[req.cpy_req->device_id][static_cast<size_t>(i)];
            cev = ev->get();
          }
          std::chrono::steady_clock::time_point h2d_start;
          if (_config.perf_instrumentation) { h2d_start = std::chrono::steady_clock::now(); }
          cudaError_t const err = req.copy_h2d_async(cev);
          if (err != cudaSuccess) {
            _perf.terminal_failures_total.fetch_add(1, std::memory_order_relaxed);
            req.manager->report_error(err);
            return false;
          }
          if (_config.perf_instrumentation) {
            auto const h2d_ns =
              static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                           std::chrono::steady_clock::now() - h2d_start)
                                           .count());
            _perf.h2d_observed_ns_total.fetch_add(h2d_ns, std::memory_order_relaxed);
            _perf.h2d_observed_count.fetch_add(1, std::memory_order_relaxed);
            atomic_max_relaxed(_perf.h2d_observed_ns_max, h2d_ns);
          }
          if (needs_event) {
            // Bounce buffer is still feeding the copy: hand the slot's token to
            // `copying` (with the event, manager and byte count) so the slot —
            // and its bounce buffer — is only reused once the copy completes,
            // and the chunk is credited only then (the device bytes are not yet
            // valid).  Clear the rest of the per-request state now so a shutdown
            // cancel won't touch a done request.
            copying.push_back(parked_copy{std::move(s.token), ev, req.manager, s.sink.written});
            s.reset();
            return true;
          }
          // Caller-buffer device read: the caller's stream orders the copy, so
          // it is safe to credit the chunk now and recycle the slot.
          req.manager->chunk_complete(s.sink.written);
          return false;
        }
        req.manager->chunk_complete(s.sink.written);
        return false;
      }
      if (rc == CURLE_OK && status == 200 && req.chunk.offset != 0) {
        // Server ignored Range and returned the whole object: the bytes start
        // at offset 0, not req.offset — non-retriable, would loop forever.
        _perf.terminal_failures_total.fetch_add(1, std::memory_order_relaxed);
        req.manager->report_error(std::make_exception_ptr(
          std::runtime_error("rest_reactor: server ignored Range (HTTP 200) for " +
                             req.object.bucket + "/" + req.object.key)));
        return false;
      }
      // Error or truncated transfer.  A short read or a transient HTTP / curl
      // failure is retried (re-authorized and re-submitted after a backoff);
      // anything else is terminal.  Retry moves the chunk out of the slot into
      // the heap, so the slot/connection is freed for other work during backoff.
      bool const short_read = rc == CURLE_OK && ok_range && s.sink.written < req.chunk.size;
      bool const retriable  = short_read || (rc != CURLE_OK && is_retriable_curl(rc)) ||
                             (rc == CURLE_OK && is_retriable_status(status));
      // A 403 is most often a presigned URL that expired while queued; re-issue
      // with a fresh signature a bounded number of times before giving up.
      bool const auth_retriable = rc == CURLE_OK && status == 403;
      if (retriable || auth_retriable) {
        std::string const reason =
          rc != CURLE_OK ? std::string(curl_easy_strerror(rc))
                         : (short_read ? "short read" : "HTTP " + std::to_string(status));
        schedule_retry(std::move(s.req),
                       s.hc.retry_after,
                       /*is_auth=*/auth_retriable && !retriable,
                       reason);
        return false;
      }
      std::string const msg = rc != CURLE_OK
                                ? std::string(curl_easy_strerror(rc))
                                : (ok_range ? "short read" : "HTTP " + std::to_string(status));
      _perf.terminal_failures_total.fetch_add(1, std::memory_order_relaxed);
      req.manager->report_error(std::make_exception_ptr(std::runtime_error(
        "rest_reactor: " + msg + " for " + req.object.bucket + "/" + req.object.key)));
      return false;
    };

    auto process_completions = [&]() {
      CURLMsg* msg = nullptr;
      int in_queue = 0;
      while ((msg = curl_multi_info_read(multi.get(), &in_queue)) != nullptr) {
        if (msg->msg != CURLMSG_DONE) { continue; }
        CURL* const h     = msg->easy_handle;
        CURLcode const rc = msg->data.result;
        long status       = 0;
        curl_easy_getinfo(h, CURLINFO_RESPONSE_CODE, &status);
        // Recover the slot index stashed in CURLOPT_PRIVATE at pool creation.
        char* priv = nullptr;
        curl_easy_getinfo(h, CURLINFO_PRIVATE, &priv);
        int const i = static_cast<int>(reinterpret_cast<intptr_t>(priv));
        curl_multi_remove_handle(multi.get(), h);
        --inflight;
        // finish() returns true when it parked the slot in `copying` (device
        // H2D in flight) — poll_copy_completions recycles it once the event
        // clears.  Otherwise reset() releases the slot's token back to the pool.
        if (!finish(i, rc, status)) { slots[static_cast<size_t>(i)].reset(); }
      }
    };

    std::array<epoll_event, MAX_EVENTS> events{};
    submit();  // kickstart anything already queued
    while (!stop_token.stop_requested()) {
      // Block indefinitely when idle; while H2D copies are outstanding, poll on
      // a short timeout so completed copies release their bounce slots promptly.
      int const timeout_ms = copying.empty() ? -1 : 1;
      int const n          = ::epoll_wait(epoll_fd.get(), events.data(), MAX_EVENTS, timeout_ms);
      if (n < 0) {
        if (errno == EINTR) { continue; }
        throw std::runtime_error(std::string("rest_reactor: epoll_wait failed: ") +
                                 std::strerror(errno));
      }
      for (int i = 0; i < n; ++i) {
        int const fd = events[i].data.fd;
        if (fd == _wakeup_fd.get()) {
          drain_fd(_wakeup_fd.get());
        } else if (fd == curl_timer_fd.get()) {
          drain_fd(curl_timer_fd.get());
          curl_multi_socket_action(multi.get(), CURL_SOCKET_TIMEOUT, 0, &running);
        } else if (fd == retry_timer_fd.get()) {
          drain_fd(retry_timer_fd.get());
          auto const now = std::chrono::steady_clock::now();
          while (!retry_heap.empty() && retry_heap.front().due <= now) {
            std::pop_heap(retry_heap.begin(), retry_heap.end(), retry_cmp);
            ready.push_back(std::move(retry_heap.back().req));
            retry_heap.pop_back();
          }
          arm_retry_timer();
        } else if (fd == upkeep_timer_fd.get()) {
          drain_fd(upkeep_timer_fd.get());
          // Keep idle pooled connections warm.  Only when fully idle — an
          // in-flight transfer already keeps its connection active, and this
          // keeps upkeep off the hot path.  One call walks the worker's shared
          // connection cache, so any pooled handle covers all of them.
          if (inflight == 0 && !slots.empty()) { curl_easy_upkeep(slots.front().easy.get()); }
        } else {
          int ev_bitmask = 0;
          if (events[i].events & EPOLLIN) { ev_bitmask |= CURL_CSELECT_IN; }
          if (events[i].events & EPOLLOUT) { ev_bitmask |= CURL_CSELECT_OUT; }
          if (events[i].events & (EPOLLERR | EPOLLHUP)) { ev_bitmask |= CURL_CSELECT_ERR; }
          curl_multi_socket_action(multi.get(), fd, ev_bitmask, &running);
        }
      }
      process_completions();
      poll_copy_completions();
      submit();
    }

    // Drain on shutdown.  First wait for in-flight H2D copies to finish so the
    // bounce storage is not freed (at reactor destruction) while a copy still
    // reads from it.  chunk_complete was deferred for these, so credit each one
    // now that its copy has landed (or fail it if the copy errored) — otherwise
    // the request_manager would never reach total_chunks.
    for (auto& pc : copying) {
      try {
        _perf.device_stream_sync_total.fetch_add(1, std::memory_order_relaxed);
        pc.event->synchronize();
        pc.manager->chunk_complete(pc.bytes);
      } catch (const std::exception& e) {
        SIRIUS_LOG_ERROR("rest_reactor: copy-event synchronize on shutdown failed: {}", e.what());
        pc.manager->report_error(std::make_exception_ptr(std::runtime_error(
          std::string("rest_reactor: device H2D copy failed on shutdown: ") + e.what())));
      }
    }
    copying.clear();

    // Detach in-flight handles and cancel every outstanding request (in-flight,
    // retry-scheduled, ready, queued) so no future is left unfulfilled.
    for (auto& s : slots) {
      if (s.req) {
        curl_multi_remove_handle(multi.get(), s.easy.get());
        s.req->manager->report_error(std::make_error_code(std::errc::operation_canceled));
        s.reset();
      }
    }
    for (auto& e : retry_heap) {
      if (e.req) {
        e.req->manager->report_error(std::make_error_code(std::errc::operation_canceled));
      }
    }
    retry_heap.clear();
    for (auto& r : ready) {
      if (r) { r->manager->report_error(std::make_error_code(std::errc::operation_canceled)); }
    }
    ready.clear();
  } catch (const std::exception& e) {
    SIRIUS_LOG_ERROR("rest_reactor worker_loop: {}", e.what());
  }

  std::unique_ptr<rest_chunked_rx_request> dr;
  while (_requests.try_dequeue(dr)) {
    if (dr) { dr->manager->report_error(std::make_error_code(std::errc::operation_canceled)); }
    dr.reset();
  }
}

}  // namespace sirius::io::rest
