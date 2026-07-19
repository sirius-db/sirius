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

#include "io/rdma/rdma_admission_gate.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>

namespace sirius::io::rdma {

/// One control-plane HTTP attempt's transport-level outcome.  The client
/// reports; policy (retry-never, validation, fail-stop) lives above it.
struct control_outcome {
  long http_status{0};          ///< 0 iff the transport failed before any response
  std::string transport_error;  ///< curl error text; empty when an HTTP response arrived
  [[nodiscard]] bool transport_ok() const noexcept { return transport_error.empty(); }
};

struct head_result {
  control_outcome outcome;
  size_t object_size{0};  ///< valid only when outcome.http_status == 200
};

struct range_get_result {
  control_outcome outcome;
  size_t delivered_bytes{0};  ///< bytes actually written into the caller buffer
  std::string content_range;  ///< raw Content-Range response header; empty if absent
};

/// Narrow IOCTX-level control-plane seam.  EXACTLY one HTTP attempt per
/// call; never throws for HTTP-level failure (that is a result), only for
/// programming errors.  No retry, no validation, no timing policy inside.
class s3_control_client {
 public:
  virtual ~s3_control_client()                                   = default;
  [[nodiscard]] virtual head_result head(const rx_route& route)  = 0;
  [[nodiscard]] virtual range_get_result range_get(const rx_route& route,
                                                   size_t offset,
                                                   size_t size,
                                                   uint8_t* dst) = 0;  // HOST memory, by type
  /// One increment per head/range_get call, including failures.
  [[nodiscard]] virtual uint64_t attempts_total() const noexcept = 0;
  /// Sum of per-transfer new-connection counts (CURLINFO_NUM_CONNECTS in the
  /// production client): a reused connection contributes 0.
  [[nodiscard]] virtual uint64_t connections_total() const noexcept = 0;
};

/// Data-plane commit state: an error BEFORE the request left the process is
/// provably-not-issued and must never be conflated with an ambiguous
/// in-flight failure.
enum class data_commit_state : uint8_t { not_sent, sent_unknown, completed };

struct data_get_result {
  data_commit_state commit{data_commit_state::not_sent};
  size_t delivered_bytes{0};
  long http_status{0};
  std::string reply_tag;  ///< x-amz-rdma-reply response header, raw; empty if absent
  std::string transport_error;
};

/// Per-worker data-plane session.  Registration and the token GET are
/// session operations (connection/handle state is session ownership).
class rdma_data_session {
 public:
  virtual ~rdma_data_session()                           = default;
  virtual void register_memory(void* base, size_t bytes) = 0;
  virtual void deregister_memory(void* base) noexcept    = 0;
  /// One token GET into a REGISTERED device destination.  Never throws for
  /// transport/HTTP failure — the result carries the commit state.
  [[nodiscard]] virtual data_get_result get(const rx_route& route,
                                            size_t offset,
                                            size_t size,
                                            void* dst) = 0;
};

/// Non-null capability by construction: the registry builds it or fails;
/// s3_rdma_ioctx::start() validates it and reports an RDMA initialization
/// error when explicit-RDMA config yielded nothing.  Each worker acquires
/// its own session at worker_loop start.
class rdma_data_session_factory {
 public:
  virtual ~rdma_data_session_factory()                               = default;
  [[nodiscard]] virtual std::unique_ptr<rdma_data_session> acquire() = 0;
};

/// Completion-tag acceptance seam: the exact accepted form is frozen from
/// gateway observation; until then production accepts any non-empty tag and
/// tests inject their own predicate.
using reply_tag_predicate = bool (*)(std::string_view tag) noexcept;
bool non_empty_reply_tag(std::string_view tag) noexcept;

/// The bundle the ioctx consumes; the production registry constructs it from
/// config, tests inject mocks.
struct rdma_transport_clients {
  std::shared_ptr<s3_control_client> control;
  std::shared_ptr<rdma_data_session_factory> data_sessions;
  reply_tag_predicate tag_predicate = &non_empty_reply_tag;
};

/// Scrub any `x-amz-rdma-token`-labeled value from diagnostic text.  Applied
/// where a data-plane transport_error enters an exception message or a log
/// line: the token authorizes reads for its TTL, so relayed gateway/verbose
/// diagnostics must not leak it even though well-behaved sessions never emit
/// their own token.
[[nodiscard]] std::string redact_rdma_tokens(std::string_view text);

}  // namespace sirius::io::rdma
