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
 * @brief Exception thrown when an OOM occurs during pipeline task execution.
 *
 * This exception carries the intermediate operator data and the index of the
 * operator that failed, allowing the executor to reschedule the task to resume
 * from the point of failure.
 */
class oom_reschedule_exception : public std::exception {
 public:
  oom_reschedule_exception(std::unique_ptr<op::operator_data> intermediate_data,
                           size_t resume_operator_index,
                           std::string message)
    : _intermediate_data(std::move(intermediate_data)),
      _resume_operator_index(resume_operator_index),
      _message(std::move(message))
  {
  }

  /**
   * @brief Release ownership of the intermediate data.
   */
  std::unique_ptr<op::operator_data> release_intermediate_data()
  {
    return std::move(_intermediate_data);
  }

  /**
   * @brief Get the operator index to resume from.
   */
  [[nodiscard]] size_t get_resume_operator_index() const noexcept { return _resume_operator_index; }

  [[nodiscard]] const char* what() const noexcept override { return _message.c_str(); }

 private:
  std::unique_ptr<op::operator_data> _intermediate_data;
  size_t _resume_operator_index;
  std::string _message;
};

}  // namespace sirius::pipeline
