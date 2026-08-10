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

/// Direction-separated id namespaces: push/close = input; pull/wait/drained = output.
using stream_id_t = std::uint64_t;

/// Id → operator router for one fragment. Non-owning; plan must outlive the session.
/// Registration is build-time only (not thread-safe). Forwarded verbs are as thread-safe as
/// batch_stream + the repository.
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

  /// @throws sirius::invalid_input_exception on a duplicate input id.
  void add_source(stream_id_t id, op::sirius_physical_streaming_source& source);

  /// ids[i] ↔ sink partition i ↔ repository i. Partial insert on duplicate mid-loop (build
  /// aborts the fragment).
  /// @throws sirius::invalid_input_exception on duplicate output id or size mismatch.
  void add_sink(std::vector<stream_id_t> ids, op::sirius_physical_streaming_sink& sink);

  /// Empty for a leaf fragment.
  [[nodiscard]] std::vector<stream_id_t> input_streams() const;

  [[nodiscard]] std::vector<stream_id_t> output_streams() const;

  // -----------------------------------------------------------------------
  // Producer side — input streams
  // -----------------------------------------------------------------------

  /// @return false if terminal. @throws on unknown input id.
  bool push(stream_id_t id, std::shared_ptr<cucascade::data_batch> batch);

  /// Sender-set EOS. @throws on unknown input id or unexpected sender.
  void close_input(stream_id_t id, sender_id_t sender);

  // -----------------------------------------------------------------------
  // Consumer side — output streams
  // -----------------------------------------------------------------------

  /// nullopt = nothing now, not EOS — use drained(id). @throws on unknown output id or poison.
  std::optional<std::shared_ptr<cucascade::data_batch>> pull(stream_id_t id);

  /// External threads only. @throws on unknown output id.
  void wait(stream_id_t id);

  /// @throws on unknown output id.
  [[nodiscard]] bool drained(stream_id_t id) const;

 private:
  struct sink_output {
    op::sirius_physical_streaming_sink* sink;
    std::size_t partition;
  };

  [[nodiscard]] op::sirius_physical_streaming_source& resolve_source(stream_id_t id) const;
  [[nodiscard]] const sink_output& resolve_sink(stream_id_t id) const;

  std::map<stream_id_t, op::sirius_physical_streaming_source*> _sources;
  std::map<stream_id_t, sink_output> _sinks;
};

}  // namespace sirius::exec
