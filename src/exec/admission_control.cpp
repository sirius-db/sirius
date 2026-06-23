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

#include <exec/admission_control.hpp>

namespace sirius::exec {

admission_control::admission_control(size_t budget) noexcept : _budget(budget) {}

admission_control::slot admission_control::acquire(size_t size, std::stop_token stop)
{
  std::unique_lock lk(_mtx);

  // Wait until either the request fits within the remaining budget, or we
  // are the only one in line (deadlock-avoidance branch — no slots will
  // release to free up space).
  bool fits =
    _cv.wait(lk, stop, [&] { return (_in_use + size <= _budget) || (_active_slots == 0); });
  if (!fits) {
    // Stop requested and predicate still not satisfied.
    return {};
  }

  size_t reserved;
  if (_in_use + size <= _budget) {
    reserved = size;
  } else {
    // _active_slots == 0 here, so _in_use == 0.  Reserve the full budget
    // so the oversized request still makes progress.
    reserved = _budget;
  }
  _in_use += reserved;
  ++_active_slots;
  return slot{this, reserved};
}

void admission_control::release(size_t reserved) noexcept
{
  {
    std::lock_guard lk(_mtx);
    _in_use -= reserved;
    --_active_slots;
  }
  // notify_all (rather than notify_one) is required: both acquire() waiters
  // and wait_for_idle() waiters park on this same _cv, and only the latter
  // care that _active_slots dropped to zero.
  _cv.notify_all();
}

void admission_control::wait_for_idle()
{
  std::unique_lock lk(_mtx);
  _cv.wait(lk, [&] { return _active_slots == 0; });
}

admission_control::slot::~slot()
{
  if (_ctrl) _ctrl->release(_reserved);
}

admission_control::slot::slot(slot&& o) noexcept : _ctrl(o._ctrl), _reserved(o._reserved)
{
  o._ctrl     = nullptr;
  o._reserved = 0;
}

admission_control::slot& admission_control::slot::operator=(slot&& o) noexcept
{
  if (this != &o) {
    if (_ctrl) _ctrl->release(_reserved);
    _ctrl       = o._ctrl;
    _reserved   = o._reserved;
    o._ctrl     = nullptr;
    o._reserved = 0;
  }
  return *this;
}

}  // namespace sirius::exec
