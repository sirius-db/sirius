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

#include "exec/stream_session.hpp"

#include "creator/task_creator.hpp"
#include "op/sirius_physical_operator_type.hpp"
#include "pipeline/pipeline_build_context.hpp"
#include "pipeline/sirius_pipeline.hpp"
#include "pipeline/task_scheduler.hpp"
#include "sirius/exception.hpp"
#include "sirius_context.hpp"
#include "telemetry/telemetry_context.hpp"

#include <cucascade/data/data_repository_manager.hpp>

#include <string>
#include <utility>

namespace sirius {
namespace exec {

stream_session::stream_session(duckdb::SiriusContext& ctx, streaming_plan plan)
  : _ctx(ctx), _plan(std::move(plan))
{
  // Index the bindings by id and validate them. Stream ids must be unique across both sides so a
  // push/pull/close never resolves ambiguously.
  for (const auto& in : _plan.inputs) {
    if (!in.channel || !in.repository || !in.op) {
      throw invalid_input_exception(
        "stream_session: input binding has a null channel, repository, or operator");
    }
    if (!_inputs_by_id.emplace(in.id, &in).second) {
      throw invalid_input_exception("stream_session: duplicate input stream id " +
                                    std::to_string(in.id));
    }
  }
  for (const auto& out : _plan.outputs) {
    if (!out.channel || !out.repository || !out.op) {
      throw invalid_input_exception(
        "stream_session: output binding has a null channel, repository, or operator");
    }
    if (_inputs_by_id.count(out.id) != 0) {
      throw invalid_input_exception("stream_session: stream id " + std::to_string(out.id) +
                                    " is used for both an input and an output");
    }
    if (!_outputs_by_id.emplace(out.id, &out).second) {
      throw invalid_input_exception("stream_session: duplicate output stream id " +
                                    std::to_string(out.id));
    }
  }

  _open_outputs = _plan.outputs.size();
  build_pipelines();
}

stream_session::~stream_session()
{
  try {
    if (_started) {
      if (!finished()) { cancel(); }
      {
        std::unique_lock<std::mutex> lk(_done_mutex);
        _done_cv.wait(lk, [this] { return _open_outputs == 0 || _errored.load(); });
      }
      // Quiesce the scheduler and drain the creation queue so no in-flight task or pending
      // creation request dereferences one of our operators after we destroy them below.
      try {
        _ctx.get_task_scheduler().drain_after_error();
      } catch (...) { /* teardown must proceed */
      }
    }
  } catch (...) {
    // A destructor must not throw.
  }

  // Clear channel callbacks before the operators / channels are destroyed. The source's on_close
  // and our output-completion on_close both capture references that must not fire afterwards.
  for (auto& in : _plan.inputs) {
    if (in.channel) { in.channel->set_on_close(nullptr); }
  }
  for (auto& out : _plan.outputs) {
    if (out.channel) { out.channel->set_on_close(nullptr); }
  }
}

void stream_session::build_pipelines()
{
  auto& repo_mgr = _ctx.get_data_repository_manager();
  auto& creator  = _ctx.get_task_creator();
  pipeline::pipeline_build_context build_ctx{_ctx.get_telemetry_context(), true};

  for (const auto& out : _plan.outputs) {
    op::sirius_physical_operator* sink_op = out.op;

    // v1 supports a single linear source -> streaming-sink chain per output. Intermediate
    // operators (filter / projection / aggregate) and fan-in are deferred (F2 / #838).
    if (sink_op->children.size() != 1) {
      throw invalid_input_exception(
        "stream_session: a streaming sink must have exactly one child (its source) in v1");
    }
    op::sirius_physical_operator* source_leaf = sink_op->children[0].get();
    if (!source_leaf->children.empty()) {
      throw invalid_input_exception(
        "stream_session: v1 supports a source -> streaming-sink chain only "
        "(no intermediate operators yet)");
    }

    // Upstream pipeline: the source leaf is both source and (pipeline) sink — its base-class
    // sink() pushes each produced batch into the streaming sink's input port.
    auto upstream = duckdb::make_shared_ptr<pipeline::sirius_pipeline>(build_ctx);
    pipeline::sirius_pipeline_build_state build_state;
    build_state.set_pipeline_source(*upstream, *source_leaf);
    build_state.set_pipeline_sink(*upstream, source_leaf, 0);
    // Populate operators[] as the normal build does (source/sink live in operators too for a
    // single-op pipeline): the streaming sink's hint recurses into src_pipeline->get_operators()[0]
    // when waiting on upstream, so an empty operators vector would fault.
    build_state.add_pipeline_operator(*upstream, *source_leaf);

    // Sink pipeline: the streaming sink is both source (reads its input port) and sink (pushes to
    // the output channel) — the CONCAT-style boundary shape. Mirrors RESULT_COLLECTOR, which also
    // adds itself to operators[].
    auto sink_pipeline = duckdb::make_shared_ptr<pipeline::sirius_pipeline>(build_ctx);
    build_state.set_pipeline_source(*sink_pipeline, *sink_op);
    build_state.set_pipeline_sink(*sink_pipeline, sink_op, 0);
    build_state.add_pipeline_operator(*sink_pipeline, *sink_op);

    // Internal port repository between the two pipelines, owned by the data_repository_manager so
    // its queued batches are spill-visible. Mirrors materialize_repository_wiring().
    repo_mgr.add_new_repository(sink_op->operator_id,
                                op::sirius_physical_streaming_sink::INPUT_PORT,
                                std::make_unique<cucascade::shared_data_repository>());
    auto* internal_repo =
      repo_mgr.get_repository(sink_op->operator_id, op::sirius_physical_streaming_sink::INPUT_PORT)
        .get();

    sink_op->add_port(op::sirius_physical_streaming_sink::INPUT_PORT,
                      std::make_unique<op::sirius_physical_operator::port>(
                        op::MemoryBarrierType::PIPELINE, internal_repo, upstream, sink_pipeline));
    source_leaf->add_next_port_after_sink(
      {sink_op, op::sirius_physical_streaming_sink::INPUT_PORT});

    // The sink pipeline consumes the upstream pipeline: this records upstream as a dependency and
    // sets the sink pipeline as upstream's parent, so upstream's completion re-schedules the sink
    // (get_output_consumers()) and propagates status.
    sink_pipeline->add_dependency(upstream);

    upstream->set_task_creator(&creator);
    sink_pipeline->set_task_creator(&creator);
    // set_pipeline on the source wires its input channel's on_close -> pipeline finish (handling
    // the zero-task / already-closed case).
    source_leaf->set_pipeline(upstream);
    sink_op->set_pipeline(sink_pipeline);

    // Completion bookkeeping: the sink closes this channel when its pipeline finishes.
    out.channel->set_on_close([this] { on_output_closed(); });

    _pipelines.push_back(upstream);
    _pipelines.push_back(sink_pipeline);
    _initial_ops.push_back(source_leaf);
  }
}

void stream_session::start()
{
  std::lock_guard<std::mutex> lk(_mutex);
  if (_started) { return; }
  _started = true;

  // Register the query (resets a fresh completion handler) and grab the completion future for
  // error propagation. Streaming has no single completion event, so on the success path this
  // future never resolves — the session detects completion via its output channels.
  _ctx.create_query(_pipelines, sirius::telemetry::query_telemetry_info{});

  auto& sched = _ctx.get_task_scheduler();
  _completion = sched.get_query_awaitable();
  sched.set_on_query_error([this] {
    _errored.store(true);
    close_all_channels();
  });

  // Streaming sources are not in the query's scan list, so start_query() would not kick them.
  // Schedule every initial source ourselves.
  auto& creator = _ctx.get_task_creator();
  for (auto* op : _initial_ops) {
    creator.schedule(op);
  }
}

void stream_session::push(stream_id_t id, std::shared_ptr<cucascade::data_batch> batch)
{
  rethrow_if_errored();
  if (!batch) { throw invalid_input_exception("stream_session::push: null batch"); }

  const stream_input& in = input_for(id);
  if (in.channel->closed()) {
    throw invalid_input_exception("stream_session::push: input stream " + std::to_string(id) +
                                  " is already closed");
  }

  // Registration-time size snapshot, read like the sink does (under the batch's read-only view).
  std::size_t size_bytes = 0;
  {
    auto ro = batch->to_read_only();
    if (const auto* d = ro.get_data()) { size_bytes = d->get_size_in_bytes(); }
  }
  const auto batch_id = batch->get_batch_id();

  // Register first (owner of record + spill-visible), then push the handle (blocks while full).
  in.repository->add_data_batch(batch);
  if (!in.channel->push(exchange_batch_handle{batch_id, size_bytes})) {
    // push() only returns false once closed — an error / cancel closed the channel under us.
    in.repository->pop_data_batch_by_id(batch_id);
    rethrow_if_errored();
    throw invalid_input_exception("stream_session::push: input stream " + std::to_string(id) +
                                  " closed during push");
  }
  // Re-arm source-task creation (a push that races the hint's empty-check would otherwise be a
  // lost wake); an already-armed request is dropped for free.
  _ctx.get_task_creator().schedule(in.op);
}

void stream_session::close_input(stream_id_t id, sender_id_t sender)
{
  rethrow_if_errored();
  const stream_input& in = input_for(id);
  in.channel->close_sender(sender);
  // The channel's on_close (wired by the source's set_pipeline) finishes the source pipeline;
  // re-arm so a task is created to observe the drained state if one is needed.
  _ctx.get_task_creator().schedule(in.op);
}

std::optional<std::shared_ptr<cucascade::data_batch>> stream_session::pull(stream_id_t id)
{
  rethrow_if_errored();
  const stream_output& out = output_for(id);

  auto handle = out.channel->try_pop();
  // A pop frees a channel slot: drive the sink's deferred flush-then-close, and re-admit sink
  // tasks (edge-triggered re-arm on the consumer side).
  out.op->try_flush_pending();
  _ctx.get_task_creator().schedule(out.op);

  if (!handle) { return std::nullopt; }

  auto b = out.repository->pop_data_batch_by_id(handle->batch_id);
  if (!b) {
    throw internal_exception("stream_session::pull: batch " + std::to_string(handle->batch_id) +
                             " missing from the output repository");
  }
  return b;
}

void stream_session::wait(stream_id_t id)
{
  const stream_output& out = output_for(id);
  out.channel->wait_readable();
  rethrow_if_errored();
}

bool stream_session::drained(stream_id_t id) const { return output_for(id).channel->drained(); }

void stream_session::cancel()
{
  {
    std::lock_guard<std::mutex> lk(_mutex);
    if (_cancelled) { return; }
    _cancelled = true;
  }
  // Soft-cancel: close every input so producers unblock and sources reach EOS; in-flight work
  // drains to the sinks, which close the output channels on finalize.
  for (const auto& in : _plan.inputs) {
    if (in.channel) { in.channel->close(); }
  }
  if (_started) {
    auto& creator = _ctx.get_task_creator();
    for (auto* op : _initial_ops) {
      creator.schedule(op);
    }
  }
}

bool stream_session::finished() const
{
  if (_errored.load()) { return true; }
  std::lock_guard<std::mutex> lk(_done_mutex);
  return _open_outputs == 0;
}

void stream_session::on_output_closed()
{
  {
    std::lock_guard<std::mutex> lk(_done_mutex);
    if (_open_outputs > 0) { --_open_outputs; }
  }
  _done_cv.notify_all();
}

void stream_session::close_all_channels()
{
  for (const auto& in : _plan.inputs) {
    if (in.channel) { in.channel->close(); }
  }
  for (const auto& out : _plan.outputs) {
    if (out.channel) { out.channel->close(); }
  }
}

void stream_session::rethrow_if_errored()
{
  if (!_errored.load()) { return; }
  std::lock_guard<std::mutex> lk(_error_mutex);
  if (!_error) {
    if (_completion.valid()) {
      try {
        _completion.get();  // ready with the stored exception once report_error ran
      } catch (...) {
        _error = std::current_exception();
      }
    }
    if (!_error) {
      _error = std::make_exception_ptr(internal_exception("stream_session: query errored"));
    }
  }
  std::rethrow_exception(_error);
}

const stream_input& stream_session::input_for(stream_id_t id) const
{
  auto it = _inputs_by_id.find(id);
  if (it == _inputs_by_id.end()) {
    throw invalid_input_exception("stream_session: unknown input stream id " + std::to_string(id));
  }
  return *it->second;
}

const stream_output& stream_session::output_for(stream_id_t id) const
{
  auto it = _outputs_by_id.find(id);
  if (it == _outputs_by_id.end()) {
    throw invalid_input_exception("stream_session: unknown output stream id " + std::to_string(id));
  }
  return *it->second;
}

}  // namespace exec
}  // namespace sirius
