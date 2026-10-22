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

#include <cstdint>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

/**
 * @file
 * @brief The notifications exchange transports send each other, as binary strings.
 *
 * None depends on arriving in order: a frame may arrive before its batch's first frame, and an
 * end of stream before the frames it counts.
 */
namespace sirius::exchange {

/// Rides with the write of one frame: the receive frame holds bytes [k * frame size, + bytes)
/// of batch @p batch of @p stream. Frame 0 carries the batch's encoded header.
struct frame_message {
  std::uint64_t stream{0};
  std::uint64_t batch{0};
  std::uint32_t k{0};
  std::uint32_t recv_frame{0};
  std::uint64_t bytes{0};
  std::vector<std::uint8_t> header;

  bool operator==(frame_message const&) const = default;
};

/// The receiver copied these frames out: the sender may write them again.
struct credit_message {
  std::vector<std::uint32_t> frames;

  bool operator==(credit_message const&) const = default;
};

/// The sender sent @p batches batches of @p stream in all, and will send no more.
struct end_message {
  std::uint64_t stream{0};
  std::uint64_t batches{0};

  bool operator==(end_message const&) const = default;
};

/// The sender failed; every exchange with it is over.
struct error_message {
  std::string what;

  bool operator==(error_message const&) const = default;
};

using message = std::variant<frame_message, credit_message, end_message, error_message>;

[[nodiscard]] std::string encode(message const& m);

/// @throws sirius::invalid_input_exception on a truncated or malformed message.
[[nodiscard]] message decode(std::string_view bytes);

}  // namespace sirius::exchange
