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

#include "helper/helper.hpp"
#include "memory/memory_reservation.hpp"

namespace sirius {
namespace parallel {

/**
 * Interface for concrete task local states.
 */
class itask_local_state {
public:
  virtual ~itask_local_state() = default;

	template <class TargetType>
	TargetType &cast() {
		DynamicCastCheck<TargetType>(this);
		return reinterpret_cast<TargetType &>(*this);
	}

	template <class TargetType>
	const TargetType &cast() const {
		DynamicCastCheck<TargetType>(this);
		return reinterpret_cast<const TargetType &>(*this);
	}
};

/**
 * Interface for concrete task global states.
 */
class itask_global_state {
public:
  virtual ~itask_global_state() = default;

  template <class TargetType>
	TargetType &cast() {
		DynamicCastCheck<TargetType>(this);
		return reinterpret_cast<TargetType &>(*this);
	}

	template <class TargetType>
	const TargetType &cast() const {
		DynamicCastCheck<TargetType>(this);
		return reinterpret_cast<const TargetType &>(*this);
	}
};

/**
 * Interface for concrete executor tasks.
 */
class itask {
public:
  itask(sirius::unique_ptr<itask_local_state> local_state, sirius::shared_ptr<itask_global_state> global_state)
    : _local_state(std::move(local_state)), _global_state(global_state) {}

  virtual ~itask() = default;

  // Non-copyable and movable.
  itask(const itask&) = delete;
  itask& operator=(const itask&) = delete;
  itask(itask&&) = default;
  itask& operator=(itask&&) = default;

  // Execution function.
  virtual void execute() = 0;

protected:
  sirius::unique_ptr<itask_local_state> _local_state;
  sirius::shared_ptr<itask_global_state> _global_state;
};

} // namespace parallel
} // namespace sirius