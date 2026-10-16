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

#include "exec/batch_stream.hpp"

#include "sirius/exception.hpp"

#include <cucascade/data/data_batch.hpp>

#include <string>
#include <utility>

namespace sirius::exec {

batch_stream::batch_stream(std::shared_ptr<cucascade::shared_data_repository> repo,
                           std::set<sender_id_t> expected)
  : _repo(std::move(repo)),
    _expected(std::move(expected)),
    // A stream nobody will ever produce into is terminal from the start; without this an
    // empty expected set could never reach EOS, wedging the consumer forever.
    _terminal(_expected.empty())
{
  if (!_repo) {
    throw sirius::invalid_input_exception("batch_stream: repository must not be null");
  }
}

bool batch_stream::push(std::shared_ptr<cucascade::data_batch> batch,
                        std::optional<std::size_t> position)
{
  std::function<void()> on_data;
  {
    std::unique_lock<std::mutex> lock(_mutex);
    if (_terminal) { return false; }
    if (position) {
      _repo->add_data_batch_at_position(std::move(batch), *position);
    } else {
      _repo->add_data_batch(std::move(batch));
    }
    on_data = _on_data;
  }
  _cv.notify_all();
  if (on_data) { on_data(); }  // outside the lock: it calls into the scheduler
  return true;
}

void batch_stream::close(sender_id_t sender)
{
  std::function<void()> eos_hook;
  std::function<void()> data_hook;
  {
    std::unique_lock<std::mutex> lock(_mutex);
    if (_expected.find(sender) == _expected.end()) {
      throw sirius::invalid_input_exception("batch_stream: sender id " + std::to_string(sender) +
                                            " is not in the expected sender set");
    }
    _closed.insert(sender);

    // A repeat close, a close after a failure, or a fan-in still waiting for other senders.
    if (_terminal || _closed.size() < _expected.size()) { return; }

    _terminal = true;
    // Move, don't copy: a throwing copy here would leave the stream terminal with the
    // notify_all() below never reached, stranding every sleeper in wait().
    eos_hook = std::move(_on_end_of_stream);
    if (_repo->get_ordering() == cucascade::data_repository_ordering::ORDERED) {
      data_hook = std::move(_on_data);
    }
  }
  _cv.notify_all();
  if (eos_hook) { eos_hook(); }
  if (data_hook) { data_hook(); }  // EOS makes positioned batches pullable.
}

void batch_stream::fail(std::exception_ptr error)
{
  if (!error) {
    throw sirius::invalid_input_exception("batch_stream: fail requires a non-null error");
  }
  std::function<void()> eos_hook;
  std::function<void()> data_hook;
  {
    std::unique_lock<std::mutex> lock(_mutex);
    if (_error) { return; }  // P2 — the first failure is the cause the consumer sees
    _error = std::move(error);
    // S2 / P4 — announce like a batch (on_data); wait() wakes via notify_all below.
    data_hook = _on_data;
    if (!_terminal) {
      _terminal = true;  // P3 — the stream ends now, waiting for nobody
      eos_hook  = std::move(_on_end_of_stream);
    }
  }
  _cv.notify_all();
  if (data_hook) { data_hook(); }
  if (eos_hook) { eos_hook(); }
}

std::shared_ptr<cucascade::data_batch> batch_stream::try_pull()
{
  return try_pull_with_position().first;
}

cucascade::positioned_data_batch batch_stream::try_pull_with_position()
{
  std::lock_guard<std::mutex> lock(_mutex);
  if (_error) { std::rethrow_exception(_error); }
  if (_repo->get_ordering() != cucascade::data_repository_ordering::ORDERED) {
    return _repo->pop_next_data_batch_with_position();
  }
  if (!_terminal) { return {}; }
  if (!_positions) { _positions = _repo->get_positions(); }
  if (_next_position == _positions->size()) { return {}; }
  auto position = (*_positions)[_next_position++];
  auto batch    = _repo->pop_data_batch_by_position(position);
  if (!batch) { throw sirius::internal_exception("Stream position disappeared after EOS"); }
  return {std::move(batch), position};
}

batch_stream::availability batch_stream::classify() const
{
  std::lock_guard<std::mutex> lock(_mutex);
  if (_error) { return availability::HAS_DATA; }
  if (!_terminal && _repo->get_ordering() == cucascade::data_repository_ordering::ORDERED) {
    return availability::WAITING;
  }
  if (!_repo->all_empty()) { return availability::HAS_DATA; }
  return _terminal ? availability::END_OF_STREAM : availability::WAITING;
}

bool batch_stream::drained() const
{
  std::lock_guard<std::mutex> lock(_mutex);
  return _terminal && !_error && _repo->all_empty();
}

std::exception_ptr batch_stream::pending_error() const
{
  std::lock_guard<std::mutex> lock(_mutex);
  return _error;
}

void batch_stream::wait()
{
  std::unique_lock<std::mutex> lock(_mutex);
  _cv.wait(lock, [&] {
    return _terminal || _error ||
           (_repo->get_ordering() != cucascade::data_repository_ordering::ORDERED &&
            !_repo->all_empty());
  });
}

void batch_stream::set_on_data(std::function<void()> hook)
{
  std::lock_guard<std::mutex> lock(_mutex);
  _on_data = std::move(hook);
}

void batch_stream::set_on_end_of_stream(std::function<void()> hook)
{
  std::function<void()> fire_now;
  {
    std::lock_guard<std::mutex> lock(_mutex);
    _on_end_of_stream = std::move(hook);
    if (_terminal) { fire_now = _on_end_of_stream; }
  }
  if (fire_now) { fire_now(); }
}

bool batch_stream::terminal() const
{
  std::lock_guard<std::mutex> lock(_mutex);
  return _terminal;
}

bool batch_stream::sender_closed(sender_id_t sender) const
{
  std::lock_guard<std::mutex> lock(_mutex);
  return _closed.find(sender) != _closed.end();
}

}  // namespace sirius::exec
