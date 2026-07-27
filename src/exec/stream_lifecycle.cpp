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

#include "exec/stream_lifecycle.hpp"

#include "sirius/exception.hpp"

#include <string>
#include <utility>

namespace sirius::exec {

stream_lifecycle::stream_lifecycle(std::set<sender_id_t> expected)
  : _expected(std::move(expected)),
    // A stream nobody will ever produce into is terminal from the start; without this an
    // empty expected set could never reach EOS, wedging the consumer forever.
    _terminal(_expected.empty())
{
}

bool stream_lifecycle::admit(const std::function<void()>& insert)
{
  std::function<void()> waker;
  {
    std::unique_lock<std::mutex> lock(_mutex);
    if (_terminal) { return false; }
    // Under the lock: close cannot interleave, and the batch is in the repository before any
    // waker can observe it.
    insert();
    waker  = std::move(_waker);
    _waker = nullptr;  // one-shot; the consumer re-arms on its next WAITING classification
  }
  _cv.notify_all();
  // Outside the lock: the waker calls into the scheduler, which must not re-enter here.
  if (waker) { waker(); }
  return true;
}

void stream_lifecycle::mark_sender_done(sender_id_t sender)
{
  std::function<void()> hook;
  {
    std::unique_lock<std::mutex> lock(_mutex);
    if (_expected.find(sender) == _expected.end()) {
      throw sirius::invalid_input_exception("stream_lifecycle: sender id " +
                                            std::to_string(sender) +
                                            " is not in the expected sender set");
    }
    // A set, not a counter: two closes from sender 0 must not stand in for senders {0, 1}.
    _closed.insert(sender);
    if (_terminal || _closed.size() < _expected.size()) { return; }
    _terminal = true;
    hook      = _on_end_of_stream;
  }
  _cv.notify_all();
  if (hook) { hook(); }
}

stream_lifecycle::availability stream_lifecycle::classify(bool repo_empty) const
{
  std::lock_guard<std::mutex> lock(_mutex);
  if (!repo_empty) { return availability::HAS_DATA; }
  return _terminal ? availability::END_OF_STREAM : availability::WAITING;
}

bool stream_lifecycle::drained(bool repo_empty) const
{
  std::lock_guard<std::mutex> lock(_mutex);
  return _terminal && repo_empty;
}

void stream_lifecycle::wait(const std::function<bool()>& repo_empty)
{
  std::unique_lock<std::mutex> lock(_mutex);
  _cv.wait(lock, [&] { return _terminal || !repo_empty(); });
}

bool stream_lifecycle::arm_waker(std::function<void()> waker, const std::function<bool()>& arm_if)
{
  std::lock_guard<std::mutex> lock(_mutex);
  if (!arm_if()) { return false; }
  _waker = std::move(waker);
  return true;
}

void stream_lifecycle::set_on_end_of_stream(std::function<void()> hook)
{
  std::function<void()> fire_now;
  {
    std::lock_guard<std::mutex> lock(_mutex);
    _on_end_of_stream = std::move(hook);
    if (_terminal) { fire_now = _on_end_of_stream; }
  }
  if (fire_now) { fire_now(); }
}

bool stream_lifecycle::terminal() const
{
  std::lock_guard<std::mutex> lock(_mutex);
  return _terminal;
}

bool stream_lifecycle::sender_closed(sender_id_t sender) const
{
  std::lock_guard<std::mutex> lock(_mutex);
  return _closed.find(sender) != _closed.end();
}

}  // namespace sirius::exec
