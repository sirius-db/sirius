/*
 * Copyright 2025, Sirius Contributors.
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

#include "duckdb/common/helper.hpp"
#include "duckdb/common/vector.hpp"
#include "exec/exchange_channel.hpp"
#include "op/sirius_physical_streaming_sink.hpp"
#include "op/sirius_physical_streaming_source.hpp"

#include <cucascade/data/data_batch.hpp>
#include <cucascade/data/data_repository.hpp>

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <exception>
#include <future>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <vector>

namespace duckdb {
class SiriusContext;
}  // namespace duckdb

namespace sirius {
namespace pipeline {
class sirius_pipeline;
}  // namespace pipeline

namespace exec {

using stream_id_t = std::uint64_t;
using sender_id_t = std::uint32_t;

/// One external input edge of a streaming plan: an exchange channel + its owner-of-record
/// repository, feeding a STREAMING_SOURCE operator in the plan tree. The producer (a wrapper
/// thread, or a test) pushes batches into this stream by id.
struct stream_input {
  stream_id_t id;
  std::shared_ptr<exchange_channel> channel;
  std::shared_ptr<cucascade::shared_data_repository> repository;
  op::sirius_physical_streaming_source* op;  // non-owning — owned by streaming_plan::root
};

/// One external output edge: a STREAMING_SINK operator in the plan tree pushing to an exchange
/// channel whose batches live in the paired repository. The consumer (wrapper/test) pulls by id.
struct stream_output {
  stream_id_t id;
  std::shared_ptr<exchange_channel> channel;
  std::shared_ptr<cucascade::shared_data_repository> repository;
  op::sirius_physical_streaming_sink* op;  // non-owning — owned by streaming_plan::root
};

/// A lowered physical-operator tree whose leaves are STREAMING_SOURCEs and whose top(s) are
/// STREAMING_SINKs, plus the per-stream channel/repository/operator bindings the session routes
/// push/pull through. v1 tests build this by hand; F2 will emit it from a StarRocks fragment.
struct streaming_plan {
  duckdb::unique_ptr<op::sirius_physical_operator> root;  // owns the whole operator tree
  std::vector<stream_input> inputs;
  std::vector<stream_output> outputs;
};

/// Engine-side session for streaming (multi-shot) execution — issue #839.
///
/// Builds the plan into pipelines, wires the exchange channels + repositories, and runs it on
/// the existing task_scheduler WITHOUT blocking the caller. The producer side pushes batches by
/// stream id and signals per-sender end-of-stream; the consumer side pulls result batches by
/// stream id. Backpressure is task-admission (a full channel simply stops sink-task creation),
/// so engine worker threads never block on a channel — only external producer/consumer threads
/// (in push/wait) do.
///
/// Threading: push/close_input/pull/wait/drained are called from external (wrapper/test)
/// threads and are internally synchronized. Errors surface as thrown exceptions (converted to a
/// Rust Result at the future cxx boundary, #839 decision 3).
///
/// Not copyable and not movable: the streaming source's on_close hook and the built pipelines
/// hold references into this object's graph, so a session must live at a stable address — hand
/// it out via std::unique_ptr.
class stream_session {
 public:
  /// Builds and wires the plan against `ctx` (which owns the scheduler, task_creator,
  /// data_repository_manager, and config). Does NOT start execution — call start(). `ctx` must
  /// outlive the session and stay initialized. Throws invalid_input_exception on a malformed
  /// plan (duplicate/unbound stream ids, an operator that does not match its binding side).
  stream_session(duckdb::SiriusContext& ctx, streaming_plan plan);

  /// Cancels and drains any in-flight work, then releases pipelines, operators, channels, and
  /// repositories. Blocks until the engine has quiesced — safe to destroy mid-stream.
  ~stream_session();

  stream_session(const stream_session&)            = delete;
  stream_session& operator=(const stream_session&) = delete;
  stream_session(stream_session&&)                 = delete;
  stream_session& operator=(stream_session&&)      = delete;

  /// Submits the plan to the scheduler and schedules the initial sources. Returns immediately.
  void start();

  // --- producer side (external threads may block on a full input channel) ---------------

  /// Registers `batch` in the input stream's repository, then pushes its handle to the channel
  /// (blocking if the channel is full), and re-arms source-task creation. Throws if the stream
  /// id is unknown, its input already closed, or the query has errored.
  void push(stream_id_t id, std::shared_ptr<cucascade::data_batch> batch);

  /// Signals end-of-stream from one sender on an input stream. The channel closes (and its
  /// source finishes) once all expected senders have signalled. Idempotent per sender. Default
  /// sender 0 covers the single-sender case.
  void close_input(stream_id_t id, sender_id_t sender = 0);

  // --- consumer side --------------------------------------------------------------------

  /// Non-blocking pull of one result batch from an output stream. Returns nullopt when nothing
  /// is available right now (use drained() to distinguish from end-of-stream). Rethrows a query
  /// error. Also drives the sink's deferred flush-then-close after each pop.
  std::optional<std::shared_ptr<cucascade::data_batch>> pull(stream_id_t id);

  /// Blocks until the output stream has a batch to pull or has reached end-of-stream. Rethrows
  /// a query error on wake.
  void wait(stream_id_t id);

  /// True once the output stream is closed and empty (end-of-stream fully consumed).
  [[nodiscard]] bool drained(stream_id_t id) const;

  // --- control --------------------------------------------------------------------------

  /// Soft-cancel: force-closes all input channels so producers unblock and sources observe EOS,
  /// lets in-flight tasks drain, and closes the output channels. Idempotent.
  void cancel();

  /// True once every output stream has closed (all sink work done) or the query has errored.
  [[nodiscard]] bool finished() const;

 private:
  void build_pipelines();     // construct + wire the pipelines from the plan tree
  void rethrow_if_errored();  // surface a stored query error as an exception
  void close_all_channels();  // force-close every input+output channel (error/cancel path)
  void on_output_closed();    // one output channel reached EOS — completion bookkeeping

  const stream_input& input_for(stream_id_t id) const;
  const stream_output& output_for(stream_id_t id) const;

  duckdb::SiriusContext& _ctx;
  streaming_plan _plan;

  std::map<stream_id_t, const stream_input*> _inputs_by_id;
  std::map<stream_id_t, const stream_output*> _outputs_by_id;

  // Pipelines built from _plan.root, kept alive for the query's lifetime.
  duckdb::vector<duckdb::shared_ptr<pipeline::sirius_pipeline>> _pipelines;
  // Internal-port repositories the session created between pipelines (owned here so they outlive
  // the pipelines that reference them). External edge repos live in the bindings.
  std::vector<std::shared_ptr<cucascade::shared_data_repository>> _internal_repositories;

  // Initial operators the session must schedule to kick execution (streaming sources + scans).
  std::vector<op::sirius_physical_operator*> _initial_ops;

  std::future<void> _completion;  // satisfied (with the exception) only if the query errors
  bool _started{false};
  bool _cancelled{false};
  std::mutex _mutex;  // guards start/cancel bookkeeping

  // Completion detection: streaming has no single completion event, so the session counts its
  // output channels closing (each sink closes its channel on finalize). done when zero remain.
  mutable std::mutex _done_mutex;
  std::condition_variable _done_cv;
  std::size_t _open_outputs{0};

  // Error state, latched from the completion_handler's on_error callback (runs on a worker
  // thread). The exception itself is pulled lazily from _completion on first rethrow.
  std::atomic<bool> _errored{false};
  std::exception_ptr _error;  // guarded by _error_mutex
  std::mutex _error_mutex;
};

}  // namespace exec
}  // namespace sirius
