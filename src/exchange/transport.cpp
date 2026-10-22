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

#include "exchange/transport.hpp"

#include "exchange/batch_io.hpp"
#include "exchange/messages.hpp"
#include "sirius/exception.hpp"

#include <rmm/cuda_device.hpp>
#include <rmm/detail/error.hpp>

#include <cuda_runtime_api.h>

#include <cucascade/data/data_batch.hpp>

#include <deque>
#include <map>
#include <optional>
#include <tuple>
#include <utility>
#include <vector>

namespace sirius::exchange {
namespace {

using clock = std::chrono::steady_clock;

/// Where sender @p from's frames sit in receiver @p to's ring: one slot per peer, in rank order.
std::uint32_t slot_of(rank_t from, rank_t to) { return from < to ? from : from - 1; }

std::uint32_t frames_for(std::uint64_t stream_bytes, std::size_t frame_bytes)
{
  return static_cast<std::uint32_t>((stream_bytes + frame_bytes - 1) / frame_bytes);
}

/// cudaEvent_t, created once and reused.
class event_pool {
 public:
  ~event_pool()
  {
    for (auto e : _free) {
      static_cast<void>(cudaEventDestroy(e));
    }
  }
  cudaEvent_t record(rmm::cuda_stream_view stream)
  {
    cudaEvent_t e{};
    if (_free.empty()) {
      RMM_CUDA_TRY(cudaEventCreateWithFlags(&e, cudaEventDisableTiming));
    } else {
      e = _free.back();
      _free.pop_back();
    }
    RMM_CUDA_TRY(cudaEventRecord(e, stream.value()));
    return e;
  }
  /// Whether @p e has completed; a completed event goes back to the pool.
  bool done(cudaEvent_t e)
  {
    auto const status = cudaEventQuery(e);
    if (status == cudaErrorNotReady) { return false; }
    RMM_CUDA_TRY(status);
    _free.push_back(e);
    return true;
  }

 private:
  std::vector<cudaEvent_t> _free;
};

}  // namespace

struct transport::engine {
  fabric& fab;
  frame_ring& send_ring;
  frame_ring& recv_ring;
  cucascade::memory::memory_space& gpu;
  cucascade::memory::memory_space* host;
  callbacks cb;
  options opts;
  int device{0};
  std::uint32_t credits_per_peer{0};
  rmm::cuda_stream send_stream;
  rmm::cuda_stream recv_stream;
  event_pool events;

  // ---- shared with senders, under transport::_mutex -----------------------------------------

  /// A batch to send, or (no batch) the end of its stream after `batch` batches.
  struct queued {
    stream_id stream;
    std::uint64_t batch;
    std::shared_ptr<outgoing_batch> out;
  };
  std::vector<std::deque<queued>> queues;
  std::vector<std::map<stream_id, std::uint64_t>> sent;
  std::size_t queued_bytes{0};

  // ---- engine thread only --------------------------------------------------------------------

  struct peer_out {
    std::vector<std::uint32_t> credits;
    std::optional<queued> current;
    std::uint32_t next_k{0};
    std::uint32_t frames{0};
  };
  std::vector<peer_out> out;
  std::vector<std::uint32_t> free_send;

  struct staged_frame {
    cudaEvent_t copied;
    rank_t peer;
    std::uint32_t send_frame;
    frame_message message;
    /// The batch, held until its last frame is staged.
    std::shared_ptr<outgoing_batch> batch;
  };
  std::deque<staged_frame> staged;  // in send-stream order

  struct posted_write {
    std::uint64_t id;
    std::uint32_t send_frame;
  };
  std::vector<posted_write> writes;

  using landing_key = std::tuple<rank_t, stream_id, std::uint64_t>;
  struct landing {
    std::optional<batch_header> header;
    std::unique_ptr<incoming_batch> in;
    std::vector<bool> seen;
    std::uint32_t copied{0};
    std::vector<frame_message> waiting;
  };
  std::map<landing_key, landing> landings;

  struct copy_out {
    cudaEvent_t copied;
    landing_key key;
    std::uint32_t recv_frame;
  };
  std::deque<copy_out> copies;  // in receive-stream order

  struct stream_end {
    std::uint64_t delivered{0};
    std::optional<std::uint64_t> expected;
    bool ended{false};
  };
  std::map<std::pair<rank_t, stream_id>, stream_end> ends;
  std::vector<bool> recv_in_use;
  std::vector<std::vector<std::uint32_t>> credits_back;

  engine(fabric& f,
         frame_ring& s,
         frame_ring& r,
         cucascade::memory::memory_space& g,
         cucascade::memory::memory_space* h,
         callbacks c,
         options o)
    : fab(f), send_ring(s), recv_ring(r), gpu(g), host(h), cb(std::move(c)), opts(o)
  {
    RMM_CUDA_TRY(cudaGetDevice(&device));
    auto const world = fab.world_size();
    auto const peers = world - 1;
    if (peers > 0) {
      credits_per_peer = recv_ring.frames() / peers;
      if (credits_per_peer == 0) {
        throw sirius::invalid_input_exception(
          "exchange: {} receive frames for {} peers", recv_ring.frames(), peers);
      }
    }
    if (send_ring.frame_bytes() != recv_ring.frame_bytes()) {
      throw sirius::invalid_input_exception("exchange: send and receive frames differ in size");
    }
    queues.resize(world);
    sent.resize(world);
    out.resize(world);
    credits_back.resize(world);
    recv_in_use.assign(recv_ring.frames(), false);
    for (rank_t peer = 0; peer < world; ++peer) {
      if (peer == fab.rank()) { continue; }
      auto const first = slot_of(fab.rank(), peer) * credits_per_peer;
      for (std::uint32_t i = 0; i < credits_per_peer; ++i) {
        out[peer].credits.push_back(first + i);
      }
    }
    for (std::uint32_t i = send_ring.frames(); i > 0; --i) {
      free_send.push_back(i - 1);
    }
  }

  ~engine()
  {
    // Staged and landing copies must not outlive the memory they touch.
    send_stream.synchronize_no_throw();
    recv_stream.synchronize_no_throw();
  }

  [[nodiscard]] std::size_t frame_bytes() const noexcept { return send_ring.frame_bytes(); }

  [[nodiscard]] bool busy() const
  {
    if (!landings.empty() || !copies.empty() || !staged.empty() || !writes.empty()) { return true; }
    for (auto const& p : out) {
      if (p.current) { return true; }
    }
    return false;
  }

  // ---- receiving -----------------------------------------------------------------------------

  void on_frame(rank_t from, frame_message m)
  {
    auto const first = slot_of(from, fab.rank()) * credits_per_peer;
    if (m.recv_frame < first || m.recv_frame >= first + credits_per_peer ||
        recv_in_use[m.recv_frame]) {
      throw sirius::invalid_input_exception(
        "exchange: rank {} wrote receive frame {}, which it does not hold", from, m.recv_frame);
    }
    recv_in_use[m.recv_frame] = true;
    landing_key const key{from, m.stream, m.batch};
    auto& l = landings[key];
    if (m.k == 0) {
      if (l.header) {
        throw sirius::invalid_input_exception(
          "exchange: rank {} sent batch {} of stream {} twice", from, m.batch, m.stream);
      }
      l.header = decode_header(m.header);
      l.seen.assign(frames_for(l.header->stream_bytes, frame_bytes()), false);
    }
    l.waiting.push_back(std::move(m));
    try_land(key, l);
  }

  /// Lands @p l if its header has arrived and a tier has room, then copies its waiting frames.
  bool try_land(landing_key const& key, landing& l)
  {
    if (!l.header || l.waiting.empty()) { return false; }
    if (!l.in) {
      l.in = incoming_batch::land(*l.header, gpu, host, recv_stream.view());
      if (!l.in) { return false; }
    }
    for (auto const& m : l.waiting) {
      auto const at = static_cast<std::uint64_t>(m.k) * frame_bytes();
      if (m.k >= l.seen.size() || l.seen[m.k] ||
          m.bytes != std::min<std::uint64_t>(frame_bytes(), l.header->stream_bytes - at)) {
        throw sirius::invalid_input_exception(
          "exchange: frame {} of {} bytes does not fit a {}-byte batch, or came twice",
          m.k,
          m.bytes,
          l.header->stream_bytes);
      }
      l.seen[m.k] = true;
      l.in->copy_in(at, recv_ring.frame(m.recv_frame), m.bytes, recv_stream.view());
      copies.push_back({events.record(recv_stream.view()), key, m.recv_frame});
    }
    l.waiting.clear();
    return true;
  }

  void check_end(rank_t from, stream_id stream)
  {
    auto& e = ends[{from, stream}];
    if (e.ended || !e.expected || e.delivered != *e.expected) { return; }
    e.ended = true;
    cb.on_end(stream, from);
  }

  bool receive()
  {
    bool progress = false;
    for (auto& [key, l] : landings) {
      if (!l.in && try_land(key, l)) { progress = true; }
    }
    while (!copies.empty() && events.done(copies.front().copied)) {
      auto const c = copies.front();
      copies.pop_front();
      progress                         = true;
      recv_in_use[c.recv_frame]        = false;
      auto const [from, stream, batch] = c.key;
      credits_back[from].push_back(c.recv_frame);
      auto it = landings.find(c.key);
      auto& l = it->second;
      if (++l.copied < l.seen.size()) { continue; }
      auto received = l.in->seal(recv_stream.view());
      landings.erase(it);
      cb.on_batch(stream, from, batch, std::move(received));
      ++ends[{from, stream}].delivered;
      check_end(from, stream);
    }
    for (rank_t peer = 0; peer < credits_back.size(); ++peer) {
      if (credits_back[peer].empty()) { continue; }
      fab.notify(peer, encode(credit_message{std::move(credits_back[peer])}));
      credits_back[peer].clear();
    }
    return progress;
  }

  // ---- sending -------------------------------------------------------------------------------

  /// Stages the next frames of each peer's current batch, while send frames and credits last.
  bool stage(std::mutex& mutex)
  {
    bool progress = false;
    for (rank_t peer = 0; peer < out.size(); ++peer) {
      auto& p = out[peer];
      if (!p.current) {
        std::lock_guard lock{mutex};
        if (queues[peer].empty()) { continue; }
        p.current = std::move(queues[peer].front());
        queues[peer].pop_front();
        p.next_k = 0;
        p.frames = p.current->out ? frames_for(p.current->out->stream_bytes(), frame_bytes()) : 0;
      }
      if (!p.current->out) {
        fab.notify(peer, encode(end_message{p.current->stream, p.current->batch}));
        p.current.reset();
        progress = true;
        continue;
      }
      while (p.next_k < p.frames && !free_send.empty() && !p.credits.empty()) {
        auto const send_frame = free_send.back();
        free_send.pop_back();
        auto const recv_frame = p.credits.back();
        p.credits.pop_back();
        auto const& batch = p.current->out;
        auto const at     = static_cast<std::size_t>(p.next_k) * frame_bytes();
        auto const bytes  = std::min<std::size_t>(frame_bytes(), batch->stream_bytes() - at);
        batch->copy_range(at, at + bytes, send_ring.frame(send_frame), send_stream.view());
        frame_message m{p.current->stream, p.current->batch, p.next_k, recv_frame, bytes, {}};
        if (p.next_k == 0) { m.header = encode_header(batch->header()); }
        staged.push_back(
          {events.record(send_stream.view()), peer, send_frame, std::move(m), batch});
        ++p.next_k;
        progress = true;
      }
      if (p.next_k == p.frames) { p.current.reset(); }
    }
    return progress;
  }

  /// Posts the writes of staged frames whose copies are done; frees send frames written.
  /// @return the bytes of batches whose last frame was posted.
  std::size_t post(bool& progress)
  {
    std::size_t released = 0;
    while (!staged.empty() && events.done(staged.front().copied)) {
      auto f = std::move(staged.front());
      staged.pop_front();
      progress         = true;
      auto const bytes = f.message.bytes;
      auto const src   = static_cast<std::size_t>(f.send_frame) * frame_bytes();
      auto const dst   = static_cast<std::size_t>(f.message.recv_frame) * frame_bytes();
      auto const last =
        static_cast<std::uint64_t>(f.message.k) * frame_bytes() + bytes == f.batch->stream_bytes();
      writes.push_back(
        {fab.write(f.peer, src, dst, bytes, encode(std::move(f.message))), f.send_frame});
      if (last) { released += f.batch->stream_bytes(); }
    }
    for (auto it = writes.begin(); it != writes.end();) {
      switch (fab.status(it->id)) {
        case fabric::write_status::pending: ++it; break;
        case fabric::write_status::done:
          free_send.push_back(it->send_frame);
          it       = writes.erase(it);
          progress = true;
          break;
        case fabric::write_status::failed:
          throw sirius::internal_exception("exchange: a write to a peer failed");
      }
    }
    return released;
  }

  // ---- both ----------------------------------------------------------------------------------

  void handle(rank_t from, message m)
  {
    if (auto* f = std::get_if<frame_message>(&m)) {
      on_frame(from, std::move(*f));
    } else if (auto* c = std::get_if<credit_message>(&m)) {
      auto const first = slot_of(fab.rank(), from) * credits_per_peer;
      for (auto const frame : c->frames) {
        if (frame < first || frame >= first + credits_per_peer) {
          throw sirius::invalid_input_exception(
            "exchange: rank {} returned frame {}, not one of ours", from, frame);
        }
        out[from].credits.push_back(frame);
      }
    } else if (auto* e = std::get_if<end_message>(&m)) {
      auto& end = ends[{from, e->stream}];
      if (end.expected) {
        throw sirius::invalid_input_exception(
          "exchange: rank {} ended stream {} twice", from, e->stream);
      }
      end.expected = e->batches;
      check_end(from, e->stream);
    } else if (auto* a = std::get_if<abort_message>(&m)) {
      cb.on_abort(a->query, from, std::move(a->what));
    } else {
      throw sirius::internal_exception(
        "exchange: rank {} failed: {}", from, std::get<error_message>(m).what);
    }
  }
};

transport::transport(fabric& fabric,
                     frame_ring& send,
                     frame_ring& recv,
                     cucascade::memory::memory_space& gpu,
                     cucascade::memory::memory_space* host,
                     callbacks callbacks,
                     options opts)
  : _engine(std::make_unique<engine>(fabric, send, recv, gpu, host, std::move(callbacks), opts))
{
  _thread = std::thread([this] { run(); });
}

transport::~transport()
{
  {
    std::lock_guard lock{_mutex};
    _stop = true;
  }
  _room.notify_all();
  _thread.join();
}

rank_t transport::rank() const noexcept { return _engine->fab.rank(); }
rank_t transport::world_size() const noexcept { return _engine->fab.world_size(); }

void transport::send(stream_id stream, rank_t to, std::shared_ptr<cucascade::data_batch> batch)
{
  auto& e = *_engine;
  if (to >= world_size()) {
    throw sirius::invalid_input_exception("exchange: no rank {} of {}", to, world_size());
  }
  {
    std::lock_guard lock{_mutex};
    if (_error) { std::rethrow_exception(_error); }
  }
  if (to == rank()) {
    std::uint64_t index = 0;
    {
      std::lock_guard lock{_mutex};
      index = e.sent[to][stream]++;
    }
    e.cb.on_batch(stream, to, index, std::move(batch));
    return;
  }
  auto out = outgoing_batch::make(std::move(batch), e.send_stream.view());
  if (!out) { return; }
  auto const bytes = out->stream_bytes();
  std::unique_lock lock{_mutex};
  _room.wait(lock, [&] {
    return _error || _stop || e.queued_bytes == 0 ||
           e.queued_bytes + bytes <= e.opts.max_queued_bytes;
  });
  if (_error) { std::rethrow_exception(_error); }
  e.queued_bytes += bytes;
  e.queues[to].push_back(
    {stream, e.sent[to][stream]++, std::make_shared<outgoing_batch>(std::move(*out))});
}

void transport::close(stream_id stream, rank_t to)
{
  auto& e = *_engine;
  if (to >= world_size()) {
    throw sirius::invalid_input_exception("exchange: no rank {} of {}", to, world_size());
  }
  {
    std::lock_guard lock{_mutex};
    if (_error) { std::rethrow_exception(_error); }
    if (to != rank()) {
      e.queues[to].push_back({stream, e.sent[to][stream], nullptr});
      return;
    }
  }
  e.cb.on_end(stream, to);
}

void transport::abort(std::uint64_t query, std::string what)
{
  std::lock_guard lock{_mutex};
  if (_error) { return; }
  _aborts.push_back({query, std::move(what)});
}

void transport::run()
{
  auto& e = *_engine;
  rmm::cuda_set_device_raii const device{rmm::cuda_device_id{e.device}};
  std::vector<fabric_notification> notes;
  auto last_progress = clock::now();
  try {
    while (true) {
      {
        std::lock_guard lock{_mutex};
        if (_stop) { return; }
      }
      bool progress = false;
      std::vector<abort_message> aborts;
      {
        std::lock_guard lock{_mutex};
        aborts.swap(_aborts);
      }
      for (auto const& a : aborts) {
        for (rank_t peer = 0; peer < e.fab.world_size(); ++peer) {
          if (peer != e.fab.rank()) { e.fab.notify(peer, encode(a)); }
        }
        progress = true;
      }
      notes.clear();
      e.fab.poll(notes);
      for (auto& n : notes) {
        e.handle(n.from, decode(n.body));
        progress = true;
      }
      progress |= e.receive();
      progress |= e.stage(_mutex);
      if (auto const released = e.post(progress); released > 0) {
        {
          std::lock_guard lock{_mutex};
          e.queued_bytes -= released;
        }
        _room.notify_all();
      }
      auto const now = clock::now();
      if (progress) {
        last_progress = now;
        continue;
      }
      if (e.busy() && now - last_progress > e.opts.stall_timeout) {
        throw sirius::internal_exception("exchange: no progress for {} ms",
                                         e.opts.stall_timeout.count());
      }
      std::this_thread::sleep_for(std::chrono::microseconds{e.busy() ? 20 : 200});
    }
  } catch (...) {
    auto const error = std::current_exception();
    std::string what = "unknown error";
    try {
      std::rethrow_exception(error);
    } catch (std::exception const& ex) {
      what = ex.what();
    } catch (...) {
    }
    for (rank_t peer = 0; peer < e.fab.world_size(); ++peer) {
      if (peer == e.fab.rank()) { continue; }
      try {
        e.fab.notify(peer, encode(error_message{what}));
      } catch (...) {
      }
    }
    {
      std::lock_guard lock{_mutex};
      _error = error;
    }
    _room.notify_all();
    e.cb.on_error(error);
  }
}

}  // namespace sirius::exchange
