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

#include "exchange/messages.hpp"

#include "sirius/exception.hpp"

#include <bit>
#include <cstring>

namespace sirius::exchange {
namespace {

static_assert(std::endian::native == std::endian::little, "exchange messages are little-endian");

// u8 kind | body, as native integers:
//   frame:  u64 stream | u64 batch | u32 k | u32 recv frame | u64 bytes | u32 n | n header bytes
//   credit: u32 n | n x u32 frame
//   end:    u64 stream | u64 batches
//   error:  utf-8 text to the end
enum class kind : std::uint8_t { frame = 1, credit = 2, end = 3, error = 4 };

template <typename T>
void put(std::string& out, T value)
{
  out.append(reinterpret_cast<char const*>(&value), sizeof(T));
}

template <typename T>
T take(std::string_view& in)
{
  if (in.size() < sizeof(T)) {
    throw sirius::invalid_input_exception("exchange message: truncated");
  }
  T value;
  std::memcpy(&value, in.data(), sizeof(T));
  in.remove_prefix(sizeof(T));
  return value;
}

void require_end(std::string_view in)
{
  if (!in.empty()) { throw sirius::invalid_input_exception("exchange message: trailing bytes"); }
}

}  // namespace

std::string encode(message const& m)
{
  std::string out;
  if (auto const* f = std::get_if<frame_message>(&m)) {
    put(out, kind::frame);
    put(out, f->stream);
    put(out, f->batch);
    put(out, f->k);
    put(out, f->recv_frame);
    put(out, f->bytes);
    put(out, static_cast<std::uint32_t>(f->header.size()));
    out.append(reinterpret_cast<char const*>(f->header.data()), f->header.size());
  } else if (auto const* c = std::get_if<credit_message>(&m)) {
    put(out, kind::credit);
    put(out, static_cast<std::uint32_t>(c->frames.size()));
    for (auto const frame : c->frames) {
      put(out, frame);
    }
  } else if (auto const* e = std::get_if<end_message>(&m)) {
    put(out, kind::end);
    put(out, e->stream);
    put(out, e->batches);
  } else {
    put(out, kind::error);
    out.append(std::get<error_message>(m).what);
  }
  return out;
}

message decode(std::string_view in)
{
  switch (take<kind>(in)) {
    case kind::frame: {
      frame_message f;
      f.stream     = take<std::uint64_t>(in);
      f.batch      = take<std::uint64_t>(in);
      f.k          = take<std::uint32_t>(in);
      f.recv_frame = take<std::uint32_t>(in);
      f.bytes      = take<std::uint64_t>(in);
      auto const n = take<std::uint32_t>(in);
      if (n != in.size()) {
        throw sirius::invalid_input_exception(
          "exchange message: a header of {} bytes in {} left", n, in.size());
      }
      if ((n == 0) != (f.k != 0)) {
        throw sirius::invalid_input_exception(
          "exchange message: frame {} with a {}-byte header; only frame 0 has one", f.k, n);
      }
      f.header.assign(in.begin(), in.end());
      return f;
    }
    case kind::credit: {
      credit_message c;
      auto const n = take<std::uint32_t>(in);
      if (n != in.size() / sizeof(std::uint32_t)) {
        throw sirius::invalid_input_exception("exchange message: credit count out of range");
      }
      c.frames.reserve(n);
      for (std::uint32_t i = 0; i < n; ++i) {
        c.frames.push_back(take<std::uint32_t>(in));
      }
      require_end(in);
      return c;
    }
    case kind::end: {
      end_message e;
      e.stream  = take<std::uint64_t>(in);
      e.batches = take<std::uint64_t>(in);
      require_end(in);
      return e;
    }
    case kind::error: return error_message{std::string{in}};
  }
  throw sirius::invalid_input_exception("exchange message: unknown kind");
}

}  // namespace sirius::exchange
