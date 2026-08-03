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

#include "exec/batch_stream.hpp"
#include "op/sirius_physical_streaming_sink.hpp"
#include "op/sirius_physical_streaming_source.hpp"

#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <vector>

namespace sirius::exec {

/// Session-local address of one stream. Ids are **direction-separated**: an id passed to
/// `push` / `close_input` names an *input* stream, an id passed to `pull` / `wait` / `drained`
/// names an *output* stream, and the two are independent namespaces. Nothing pairs a leaf
/// fragment's output id with a root fragment's input id inside the engine — that routing table
/// is the wrapper's, built from the front end's plan.
using stream_id_t = std::uint64_t;

/// Routes the streaming API by stream id over a plan fragment's already-built streaming
/// operators. One session models one fragment.
///
/// A router and nothing more: each call resolves an id to an operator and forwards. The
/// operators work on their own — a source's re-arm is wired through its pipeline, not here — so
/// the session adds addressing, not behaviour.
///
/// It does not own the operators it routes to. A plan tree owns its operators uniquely
/// (`sirius_physical_operator::children` is a vector of `duckdb::unique_ptr`), so an operator
/// inside a plan cannot be handed out as an owning pointer. Whatever owns the plan must outlive
/// the session.
///
/// Not thread-safe for registration: `add_source` / `add_sink` run at build time, before any
/// producer or consumer thread touches the session. The forwarded calls themselves are as
/// thread-safe as the operators they reach (the stream and the repository both lock).
class stream_session {
 public:
  stream_session()                                     = default;
  stream_session(stream_session&&) noexcept            = default;
  stream_session& operator=(stream_session&&) noexcept = default;
  stream_session(const stream_session&)                = delete;
  stream_session& operator=(const stream_session&)     = delete;

  // -----------------------------------------------------------------------
  // Registration (build time)
  // -----------------------------------------------------------------------

  /// Register a streaming source under the input stream id `id`. The session does not take
  /// ownership; `source` must outlive it.
  /// @throws sirius::invalid_input_exception on a duplicate input id.
  void add_source(stream_id_t id, op::sirius_physical_streaming_source& source);

  /// Register a streaming sink under `ids`, one id per output stream: `ids[i]` addresses the
  /// sink's partition `i`, which is its output repository `i`. A single-destination sink takes
  /// exactly one id. The session does not take ownership; `sink` must outlive it.
  /// @throws sirius::invalid_input_exception on a duplicate output id, or an `ids` size that
  ///         does not match the sink's output stream count.
  void add_sink(std::vector<stream_id_t> ids, op::sirius_physical_streaming_sink& sink);

  /// Registered input stream ids, ascending. Empty for a leaf fragment, which produces but
  /// never receives.
  [[nodiscard]] std::vector<stream_id_t> input_streams() const;

  /// Registered output stream ids, ascending.
  [[nodiscard]] std::vector<stream_id_t> output_streams() const;

  // -----------------------------------------------------------------------
  // Producer side — input streams
  // -----------------------------------------------------------------------

  /// Hand `batch` to the source registered under `id`.
  /// @return false when the stream already reached end-of-stream and the batch was refused.
  /// @throws sirius::invalid_input_exception when `id` is not a registered input stream.
  bool push(stream_id_t id, std::shared_ptr<cucascade::data_batch> batch);

  /// Record that `sender` has finished producing into input stream `id`. Idempotent per
  /// sender; the stream ends only once every expected sender has closed.
  /// @throws sirius::invalid_input_exception when `id` is not a registered input stream, or
  ///         when `sender` is not one of that stream's expected senders.
  void close_input(stream_id_t id, sender_id_t sender);

  // -----------------------------------------------------------------------
  // Consumer side — output streams
  // -----------------------------------------------------------------------

  /// Non-blocking pull from output stream `id`. `nullopt` means "nothing right now", which is
  /// not the same as end-of-stream — `drained(id)` distinguishes them.
  /// @throws sirius::invalid_input_exception when `id` is not a registered output stream.
  std::optional<std::shared_ptr<cucascade::data_batch>> pull(stream_id_t id);

  /// Block until output stream `id` has a batch or has ended. External threads only.
  /// @throws sirius::invalid_input_exception when `id` is not a registered output stream.
  void wait(stream_id_t id);

  /// True when output stream `id` has ended and holds nothing more.
  /// @throws sirius::invalid_input_exception when `id` is not a registered output stream.
  [[nodiscard]] bool drained(stream_id_t id) const;

 private:
  /// An output id resolves to one sink *and* the partition within it. Non-owning.
  struct sink_output {
    op::sirius_physical_streaming_sink* sink;
    std::size_t partition;
  };

  /// @throws sirius::invalid_input_exception when `id` is not a registered input stream.
  [[nodiscard]] op::sirius_physical_streaming_source& resolve_source(stream_id_t id) const;
  /// @throws sirius::invalid_input_exception when `id` is not a registered output stream.
  [[nodiscard]] const sink_output& resolve_sink(stream_id_t id) const;

  std::map<stream_id_t, op::sirius_physical_streaming_source*> _sources;
  std::map<stream_id_t, sink_output> _sinks;
};

}  // namespace sirius::exec
