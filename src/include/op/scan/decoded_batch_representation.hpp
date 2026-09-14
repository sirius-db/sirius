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

#pragma once

#include <compression/compressed_scan.hpp>
#include <cucascade/cudf/gpu_data_representation.hpp>

#include <cstddef>
#include <memory>
#include <utility>
#include <vector>

namespace sirius {

/**
 * @brief A GPU table representation that reports what its decode did.
 *
 * Constructed by the compression converters (decompress_host_to_gpu /
 * decompress_device_to_gpu) in place of the plain gpu_table_representation
 * whenever the decode has something to report; the plain type is still used
 * when it does not, so nothing changes on the ordinary path and the feature
 * gate being off is byte-identical to before. @c sirius::pushdown_outcome is
 * declared with the decoder that fills it (compression/compressed_scan.hpp).
 *
 * scan_operator_input::prepare_for_processing reads @ref outcome right after
 * convert_to and stamps the split from it.
 *
 * The outcome is a property of THIS decode, so a copy that shares the decoded
 * columns legitimately shares it too — unlike the dynamic-type encoding this
 * replaced, where clone() had to decide what the copy "is" and settled for
 * dropping the information.
 */
class decompression_pushdown_batch_representation final
  : public ::cucascade::gpu_table_representation {
 public:
  decompression_pushdown_batch_representation(std::unique_ptr<cudf::table> table,
                                              ::cucascade::memory::memory_space& memory_space,
                                              rmm::cuda_stream_view writer_stream,
                                              pushdown_outcome outcome)
    : ::cucascade::gpu_table_representation(std::move(table), memory_space, writer_stream),
      _outcome(outcome)
  {
  }

  [[nodiscard]] const pushdown_outcome& outcome() const noexcept { return _outcome; }

 private:
  pushdown_outcome _outcome;
};

}  // namespace sirius
