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

#include "op/sirius_physical_operator.hpp"

#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

namespace sirius::pipeline {

/**
 * @brief Base exception for pipeline task reschedule requests.
 *
 * Carries the operator input data at the point of failure and the operator
 * index to resume from, allowing the executor to requeue the task without
 * falling back to CPU.  Derived types distinguish the failure reason.
 */
class task_reschedule_exception : public std::exception {
 public:
  task_reschedule_exception(std::unique_ptr<op::operator_data> intermediate_data,
                            size_t resume_operator_index,
                            std::string message)
    : _intermediate_data(std::move(intermediate_data)),
      _resume_operator_index(resume_operator_index),
      _message(std::move(message))
  {
  }

  std::unique_ptr<op::operator_data> release_intermediate_data()
  {
    return std::move(_intermediate_data);
  }

  [[nodiscard]] size_t get_resume_operator_index() const noexcept { return _resume_operator_index; }

  [[nodiscard]] const char* what() const noexcept override { return _message.c_str(); }

 private:
  std::unique_ptr<op::operator_data> _intermediate_data;
  size_t _resume_operator_index;
  std::string _message;
};

/**
 * @brief Reschedule request from an out-of-memory failure.
 */
class oom_reschedule_exception : public task_reschedule_exception {
 public:
  using task_reschedule_exception::task_reschedule_exception;
};

/** @brief Reschedule request from a transient CUDA kernel-launch failure. */
class cuda_launch_reschedule_exception : public task_reschedule_exception {
 public:
  cuda_launch_reschedule_exception(std::unique_ptr<op::operator_data> intermediate_data,
                                   size_t resume_operator_index,
                                   int cuda_error_code,
                                   std::string message)
    : task_reschedule_exception(
        std::move(intermediate_data), resume_operator_index, std::move(message)),
      _cuda_error_code(cuda_error_code)
  {
  }

  [[nodiscard]] int get_cuda_error_code() const noexcept { return _cuda_error_code; }

 private:
  int _cuda_error_code;
};

}  // namespace sirius::pipeline
