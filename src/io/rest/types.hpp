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

#include "io/io_request.hpp"
#include "io/rest/authorizer.hpp"
#include "io/types.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>

namespace sirius::io::rest {

/// Where a transfer's header callback is within libcurl's delivery: the next
/// line starts a response block (first line = status line), lines are fields
/// of the current response, or everything that follows the final response's
/// headers (trailers) — which never starts a new response.
enum class header_phase : std::uint8_t { response_start, response_fields, trailers };

/// Write-callback target for one in-flight transfer.  libcurl hands the
/// response body to the reactor in arbitrarily-sized pieces; @c buf_sink
/// scatters them across @c buffers (the chunk's destination iovecs, in file
/// order) at a running cursor — so a single contiguous ranged GET that fuses
/// several adjacent segments lands each segment in its own buffer — and ALWAYS
/// reports the full incoming size back to curl (never a short count, which
/// would abort the transfer with CURLE_WRITE_ERROR).  Bytes beyond @c capacity
/// are counted in @c total_received but not stored, so the reactor can detect a
/// server that ignored the Range header (e.g. returned the whole object).  A
/// 3xx body (the XML a redirect carries) is likewise counted but not stored;
/// that decision uses libcurl's own response code for the transfer (@c easy),
/// never a locally parsed status.
struct buf_sink {
  std::span<iovec> buffers;  // destination buffers, in file order
  std::size_t capacity{0};   // Σ buffers[i].iov_len
  std::size_t active{0};     // index of the buffer currently being filled
  std::size_t cursor{0};     // bytes written into buffers[active]
  std::size_t written{0};    // total bytes written across all buffers
  std::size_t total_received{0};
  void* easy{nullptr};  // the transfer's CURL* (CURLINFO_RESPONSE_CODE), or null

  void reset() noexcept
  {
    active         = 0;
    cursor         = 0;
    written        = 0;
    total_received = 0;
  }
};

/// Response metadata for the current HTTP response block.  @c status is the
/// block's own status line (used for framing); completion handling consumes
/// @c location, the validators and the retry metadata.  A new response block
/// clears the preceding block's fields; trailers never do.
struct header_capture {
  header_phase phase{header_phase::response_start};
  long status{0};
  std::string content_range;
  std::string retry_after;
  std::string etag;
  std::string location;

  void reset() noexcept
  {
    phase  = header_phase::response_start;
    status = 0;
    content_range.clear();
    retry_after.clear();
    etag.clear();
    location.clear();
  }
};

/// REST-specific retry envelope around a backend-neutral physical operation.
/// The operation owns any CuCascade staging allocation through staging_owner,
/// so its iovecs remain stable across retries and until a CUDA event drains.
struct rest_io_op_request {
  object_ref object;
  std::string validation_tag;
  std::unique_ptr<io_op_request> op;
  std::size_t attempt{0};
  std::size_t auth_attempt{0};
  bool needs_staging{false};
  std::size_t logical_bytes{0};
  /// Set once this operation's 307 was delegated (config::cluster_redirect):
  /// the Location to request instead of a freshly authorized origin URL.  It
  /// lives on the request, not the connection slot, so a retry after the hop
  /// goes back to the target and a second 307 is refused.
  std::string delegated_url;
  /// Cached-attempt provenance and operation-wide refresh/budget state.
  bool route_cached{false};
  bool route_started_cached{false};
  bool route_refreshed{false};
  std::shared_ptr<const std::string> route_snapshot;

  [[nodiscard]] bool is_device() const noexcept
  {
    return op != nullptr && op->device_copy != nullptr;
  }

  [[nodiscard]] cudaError_t copy_h2d_async(cudaEvent_t event = nullptr) const noexcept
  {
    if (!is_device()) return cudaSuccess;
    return op->device_copy->copy_async(op->io_rng, op->iovecs, event);
  }
};

}  // namespace sirius::io::rest
