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
#include "helper/utils.hpp"
#include "log/logging.hpp"
#include "sirius/exception.hpp"

#include <rmm/cuda_device.hpp>
#include <rmm/detail/error.hpp>

#include <cuda_runtime_api.h>

#include <cucascade/data/data_batch.hpp>
#include <cucascade/memory/memory_space.hpp>

#include <algorithm>
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
  return static_cast<std::uint32_t>(utils::ceil_div<std::uint64_t>(stream_bytes, frame_bytes));
}

/// Consecutive frame numbers [first, first + length).
struct frame_run {
  std::uint32_t first{0};
  std::uint32_t length{0};
};

/// The longest run of consecutive frame numbers in @p pool (the first of equals), which it
/// sorts; empty for an empty pool.
frame_run longest_run(std::vector<std::uint32_t>& pool)
{
  std::sort(pool.begin(), pool.end());
  frame_run best{};
  std::size_t start = 0;
  for (std::size_t i = 1; i <= pool.size(); ++i) {
    if (i < pool.size() && pool[i] == pool[i - 1] + 1) { continue; }
    auto const length = static_cast<std::uint32_t>(i - start);
    if (length > best.length) { best = {pool[start], length}; }
    start = i;
  }
  return best;
}

/// Removes the frames of @p run, which a sorted @p pool holds, from it.
void take(std::vector<std::uint32_t>& pool, frame_run run)
{
  auto const at = std::lower_bound(pool.begin(), pool.end(), run.first);
  pool.erase(at, at + run.length);
}

/// cudaEvent_t, created once and reused. Destroys every event it created, checked out or not, so
/// its owner must first synchronize the streams they were recorded on.
class event_pool {
 public:
  ~event_pool()
  {
    for (auto e : _all) {
      static_cast<void>(cudaEventDestroy(e));
    }
  }
  cudaEvent_t record(rmm::cuda_stream_view stream)
  {
    cudaEvent_t e{};
    if (_free.empty()) {
      _all.reserve(_all.size() + 1);  // so the push below cannot throw and orphan the event
      RMM_CUDA_TRY(cudaEventCreateWithFlags(&e, cudaEventDisableTiming));
      _all.push_back(e);
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
  std::vector<cudaEvent_t> _all;
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
  /// The most frames one write covers: half a peer's credits, so that two writes to a peer can
  /// be in flight while the receiver copies the first out.
  std::uint32_t max_write_frames{1};
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
    /// Frames staged for this peer and not yet written: at most its credits.
    std::uint32_t unwritten{0};
  };
  std::vector<peer_out> out;
  /// stage() starts at this peer, one further each pass, so low ranks get no head start on the
  /// shared send frames.
  rank_t first_peer{0};
  std::vector<std::uint32_t> free_send;

  struct staged_frame {
    cudaEvent_t copied;
    rank_t peer;
    /// The first of message.frames send frames not yet written.
    std::uint32_t send_frame;
    /// The frames not yet written; its recv_frame is chosen per write.
    frame_message message;
    /// The batch, held until its last frame is staged.
    std::shared_ptr<outgoing_batch> batch;
    bool copy_done{false};
  };
  std::deque<staged_frame> staged;  // in send-stream order

  struct posted_write {
    std::uint64_t id;
    frame_run send_frames;
  };
  std::vector<posted_write> writes;

  using landing_key = std::tuple<rank_t, stream_id, std::uint64_t>;
  /// A batch being received. Frames that arrive before its header (frame 0) or before a tier has
  /// room wait, holding their receive frames; seen and copied count frames by index.
  struct landing {
    std::optional<batch_header> header;
    std::unique_ptr<incoming_batch> in;
    std::vector<bool> seen;
    std::uint32_t copied{0};
    std::vector<frame_message> waiting;
    bool asked_for_room{false};
  };
  std::map<landing_key, landing> landings;
  /// Landings that found no room are retried no sooner than this.
  clock::time_point next_land_try{};

  struct copy_out {
    cudaEvent_t copied;
    landing_key key;
    frame_run recv_frames;
  };
  std::deque<copy_out> copies;  // in receive-stream order

  struct stream_end {
    std::uint64_t delivered{0};
    std::optional<std::uint64_t> expected;
  };
  std::map<std::pair<rank_t, stream_id>, stream_end> ends;
  /// Ends in `ends` that promised more batches than were delivered so far: work a peer owes.
  std::size_t owed_ends{0};
  std::vector<bool> recv_in_use;
  std::vector<std::vector<std::uint32_t>> credits_back;

  engine(fabric& f,
         frame_ring& s,
         frame_ring& r,
         cucascade::memory::memory_space& g,
         cucascade::memory::memory_space* h,
         callbacks c,
         options o)
    : fab(f), send_ring(s), recv_ring(r), gpu(g), host(h), cb(std::move(c)), opts(std::move(o))
  {
    if (!cb.on_batch || !cb.on_end || !cb.on_error || !cb.on_abort) {
      throw sirius::invalid_input_exception("exchange: every transport callback must be set");
    }
    RMM_CUDA_TRY(cudaGetDevice(&device));
    auto const world = fab.world_size();
    auto const peers = world - 1;
    if (peers > 0) {
      credits_per_peer = recv_ring.frames() / peers;
      if (credits_per_peer == 0) {
        throw sirius::invalid_input_exception(
          "exchange: {} receive frames for {} peers", recv_ring.frames(), peers);
      }
      max_write_frames =
        std::max<std::uint32_t>(1, std::min(credits_per_peer, send_ring.frames()) / 2);
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
    // A peer that ended a stream sends its batches before the end, so they are in flight or lost.
    return owed_ends > 0;
  }

  // ---- receiving -----------------------------------------------------------------------------

  void on_frame(rank_t from, frame_message m)
  {
    auto const first = slot_of(from, fab.rank()) * credits_per_peer;
    if (m.frames == 0 || m.recv_frame < first || m.frames > credits_per_peer ||
        m.recv_frame - first > credits_per_peer - m.frames) {
      throw sirius::invalid_input_exception(
        "exchange: rank {} wrote receive frames [{}, +{}), which it does not hold",
        from,
        m.recv_frame,
        m.frames);
    }
    for (std::uint32_t f = m.recv_frame; f < m.recv_frame + m.frames; ++f) {
      if (recv_in_use[f]) {
        throw sirius::invalid_input_exception(
          "exchange: rank {} wrote receive frame {}, which it does not hold", from, f);
      }
      recv_in_use[f] = true;
    }
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
      if (!l.in) {
        if (!l.asked_for_room && opts.request_room) { opts.request_room(l.header->stream_bytes); }
        l.asked_for_room = true;
        return false;
      }
    }
    for (auto const& m : l.waiting) {
      auto const at = static_cast<std::uint64_t>(m.k) * frame_bytes();
      bool const fits =
        m.k < l.seen.size() && m.frames <= l.seen.size() - m.k &&
        m.bytes == std::min<std::uint64_t>(std::uint64_t{m.frames} * frame_bytes(),
                                           l.header->stream_bytes - at) &&
        frames_for(m.bytes, frame_bytes()) == m.frames &&
        std::none_of(
          l.seen.begin() + m.k, l.seen.begin() + m.k + m.frames, [](bool seen) { return seen; });
      if (!fits) {
        throw sirius::invalid_input_exception(
          "exchange: a write of frames [{}, +{}) and {} bytes does not fit a {}-byte batch, or "
          "came twice",
          m.k,
          m.frames,
          m.bytes,
          l.header->stream_bytes);
      }
      std::fill(l.seen.begin() + m.k, l.seen.begin() + m.k + m.frames, true);
      l.in->copy_in(at, recv_ring.frame(m.recv_frame), m.bytes, recv_stream.view());
      copies.push_back({events.record(recv_stream.view()), key, {m.recv_frame, m.frames}});
    }
    l.waiting.clear();
    return true;
  }

  void check_end(rank_t from, stream_id stream)
  {
    auto const it = ends.find({from, stream});
    if (it == ends.end() || !it->second.expected || it->second.delivered != *it->second.expected) {
      return;
    }
    ends.erase(it);
    try {
      cb.on_end(stream, from);
    } catch (...) {
      fail_query(stream, std::current_exception());
    }
  }

  /// A receiver of @p stream threw: fails its query here and on every peer, not the transport.
  void fail_query(stream_id stream, std::exception_ptr error)
  {
    auto const query = query_of(stream);
    std::string what = "unknown error";
    try {
      std::rethrow_exception(error);
    } catch (std::exception const& ex) {
      what = ex.what();
    } catch (...) {
    }
    cb.on_abort(query, fab.rank(), what);
    for (rank_t peer = 0; peer < fab.world_size(); ++peer) {
      if (peer != fab.rank()) { fab.notify(peer, encode(abort_message{query, what})); }
    }
  }

  bool receive()
  {
    bool progress = false;
    if (auto const now = clock::now(); now >= next_land_try) {
      bool no_room = false;
      for (auto& [key, l] : landings) {
        if (l.in || !l.header || l.waiting.empty()) { continue; }
        if (try_land(key, l)) {
          progress = true;
        } else {
          no_room = true;
        }
      }
      // Room appears as other work frees memory, not within microseconds.
      if (no_room) { next_land_try = now + std::chrono::milliseconds{1}; }
    }
    while (!copies.empty() && events.done(copies.front().copied)) {
      auto const c = copies.front();
      copies.pop_front();
      progress                         = true;
      auto const [from, stream, batch] = c.key;
      for (auto f = c.recv_frames.first; f < c.recv_frames.first + c.recv_frames.length; ++f) {
        recv_in_use[f] = false;
        credits_back[from].push_back(f);
      }
      auto it = landings.find(c.key);
      auto& l = it->second;
      l.copied += c.recv_frames.length;
      if (l.copied < l.seen.size()) { continue; }
      auto received = l.in->seal(recv_stream.view());
      landings.erase(it);
      try {
        cb.on_batch(stream, from, batch, std::move(received));
      } catch (...) {
        fail_query(stream, std::current_exception());
      }
      auto& end = ends[{from, stream}];
      ++end.delivered;
      if (end.expected && end.delivered == *end.expected) { --owed_ends; }
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

  /// Stages the next frames of each peer's current batch into the send ring, while send frames
  /// last and the peer has less than a window (its credits) staged and not yet written. Staging
  /// does not wait for credits, so a returned credit finds the next frames already copied.
  bool stage(std::mutex& mutex)
  {
    bool progress    = false;
    auto const peers = static_cast<rank_t>(out.size());
    first_peer       = (first_peer + 1) % peers;
    for (rank_t i = 0; i < peers; ++i) {
      auto const peer = (first_peer + i) % peers;
      auto& p         = out[peer];
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
      // Each staged run is up to max_write_frames consecutive send frames; post() writes it in
      // as few writes as the peer's consecutive credits allow.
      while (p.next_k < p.frames && !free_send.empty() && p.unwritten < credits_per_peer) {
        auto const sends = longest_run(free_send);
        auto const n     = std::min(
          {p.frames - p.next_k, max_write_frames, sends.length, credits_per_peer - p.unwritten});
        take(free_send, {sends.first, n});
        auto const& batch = p.current->out;
        auto const at     = static_cast<std::size_t>(p.next_k) * frame_bytes();
        auto const bytes =
          std::min<std::size_t>(std::size_t{n} * frame_bytes(), batch->stream_bytes() - at);
        batch->copy_range(at, at + bytes, send_ring.frame(sends.first), send_stream.view());
        // The receive frames are chosen when the run is written.
        frame_message m{p.current->stream, p.current->batch, p.next_k, 0, bytes, {}};
        m.frames = n;
        p.unwritten += n;
        if (p.next_k == 0) { m.header = encode_header(batch->header()); }
        staged.push_back(
          {events.record(send_stream.view()), peer, sends.first, std::move(m), batch});
        p.next_k += n;
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
    // Copies finish in send-stream order: past the first one still running, none is done.
    for (auto it = staged.begin(); it != staged.end();) {
      auto& f = *it;
      if (!f.copy_done) {
        if (!events.done(f.copied)) { break; }
        f.copy_done = true;
      }
      auto& p = out[f.peer];
      // Write the run's frames into the peer's longest runs of consecutive credits.
      while (f.message.frames > 0 && !p.credits.empty()) {
        auto const credits = longest_run(p.credits);
        auto const n       = std::min(credits.length, f.message.frames);
        take(p.credits, {credits.first, n});
        frame_message w = f.message;
        w.recv_frame    = credits.first;
        w.frames        = n;
        w.bytes        = std::min<std::uint64_t>(std::uint64_t{n} * frame_bytes(), f.message.bytes);
        auto const src = static_cast<std::size_t>(f.send_frame) * frame_bytes();
        auto const dst = static_cast<std::size_t>(credits.first) * frame_bytes();
        auto const last =
          static_cast<std::uint64_t>(w.k) * frame_bytes() + w.bytes == f.batch->stream_bytes();
        auto const bytes = w.bytes;
        writes.push_back(
          {fab.write(f.peer, src, dst, bytes, encode(std::move(w))), {f.send_frame, n}});
        f.send_frame += n;
        f.message.k += n;
        f.message.frames -= n;
        f.message.bytes -= bytes;
        f.message.header.clear();  // only the write of frame 0 carries it
        p.unwritten -= n;
        progress = true;
        if (last) { released += f.batch->stream_bytes(); }
      }
      it = f.message.frames == 0 ? staged.erase(it) : std::next(it);
    }
    for (auto it = writes.begin(); it != writes.end();) {
      switch (fab.status(it->id)) {
        case fabric::write_status::pending: ++it; break;
        case fabric::write_status::done:
          for (auto f = it->send_frames.first; f < it->send_frames.first + it->send_frames.length;
               ++f) {
            free_send.push_back(f);
          }
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
      if (end.delivered > e->batches) {
        throw sirius::invalid_input_exception(
          "exchange: rank {} ended stream {} after {} batches, having sent {}",
          from,
          e->stream,
          e->batches,
          end.delivered);
      }
      end.expected = e->batches;
      if (end.delivered < e->batches) { ++owed_ends; }
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
  stop();
  // The engine still holds unsent batches, whose deleters use the members declared after it.
  _engine.reset();
}

void transport::stop()
{
  {
    std::lock_guard lock{_mutex};
    _stop = true;
  }
  _room.notify_all();
  _released.notify_all();
  if (_thread.joinable()) { _thread.join(); }
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
  // Not on send_stream: planning reads string sizes back, which there would wait behind every
  // frame the engine has staged. send_stream waits for the planning instead (the producer's
  // writes and any unslice copy) before the engine stages from the batch.
  rmm::cuda_stream_view const plan_stream{e.gpu.acquire_stream()};
  auto out = outgoing_batch::make(std::move(batch), plan_stream);
  if (!out) { return; }
  cudaEvent_t planned{};
  RMM_CUDA_TRY(cudaEventCreateWithFlags(&planned, cudaEventDisableTiming));
  RMM_CUDA_TRY(cudaEventRecord(planned, plan_stream.value()));
  RMM_CUDA_TRY(cudaStreamWaitEvent(e.send_stream.value(), planned, 0));
  RMM_CUDA_TRY(cudaEventDestroy(planned));
  auto const bytes = out->stream_bytes();
  auto owned       = std::make_unique<outgoing_batch>(std::move(*out));
  // Counted before the deleter can run, on any path that drops the batch.
  auto const query = query_of(stream);
  {
    std::lock_guard lock{_mutex};
    ++_held[query];
  }
  std::shared_ptr<outgoing_batch> held(owned.release(), [this, query](outgoing_batch* b) {
    delete b;
    released(query);
  });
  std::unique_lock lock{_mutex};
  _room.wait(lock, [&] {
    // A batch larger than the cap goes alone, else it would never fit.
    return _error || _stop || _cancelled.contains(query) || e.queued_bytes == 0 ||
           e.queued_bytes + bytes <= e.opts.max_queued_bytes;
  });
  if (_error) { std::rethrow_exception(_error); }
  // Nothing more is queued for a cancelled query, so its senders do not wait for room that only
  // its own cleanup frees; nor once stopped, when nothing drains the queues. `held` is dropped
  // after `lock`, as below.
  if (_stop || _cancelled.contains(query)) { return; }
  // A copy, not a move: if the push throws, `held` stays the last owner and is dropped after `lock`
  // is released, since its deleter takes _mutex.
  auto& next = e.sent[to][stream];
  e.queues[to].push_back({stream, next, held});
  ++next;
  e.queued_bytes += bytes;
}

void transport::released(distributed_query_id query)
{
  {
    std::lock_guard lock{_mutex};
    if (auto it = _held.find(query); it != _held.end() && --it->second == 0) { _held.erase(it); }
  }
  _released.notify_all();
}

bool transport::wait_sent(distributed_query_id query, std::chrono::milliseconds timeout)
{
  std::unique_lock lock{_mutex};
  _released.wait_for(lock, timeout, [&] { return _stop || !_held.contains(query); });
  return !_held.contains(query);
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
    // The count goes with the stream's end: nothing is sent on it after close().
    auto const sent = e.sent[to].extract(stream);
    if (to != rank()) {
      e.queues[to].push_back({stream, sent.empty() ? 0 : sent.mapped(), nullptr});
      return;
    }
  }
  e.cb.on_end(stream, to);
}

void transport::abort(distributed_query_id query, std::string what)
{
  std::lock_guard lock{_mutex};
  if (_error) { return; }
  _aborts.push_back({query, std::move(what)});
}

void transport::cancel(distributed_query_id query)
{
  {
    std::lock_guard lock{_mutex};
    _cancelled.insert(query);
  }
  _room.notify_all();
}

void transport::forget(distributed_query_id first, distributed_query_id last)
{
  std::lock_guard lock{_mutex};
  _cancelled.erase(_cancelled.lower_bound(first), _cancelled.upper_bound(last));
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
        if (_stop) { break; }
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
      // The fabric has no blocking wait. Right after progress, frames move every few
      // microseconds (a 64 MiB write over NVLink takes about 90): spin, as a sleep alone would
      // cost tens of microseconds each time. Then poll often while frames are in flight, less
      // when idle.
      constexpr std::chrono::microseconds spin_after_progress{500};
      constexpr std::chrono::microseconds busy_poll_interval{20};
      constexpr std::chrono::microseconds idle_poll_interval{200};
      if (e.busy() && now - last_progress < spin_after_progress) {
        std::this_thread::yield();
      } else {
        std::this_thread::sleep_for(e.busy() ? busy_poll_interval : idle_poll_interval);
      }
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
      } catch (std::exception const& ex) {
        SIRIUS_LOG_WARN("exchange: could not tell rank {} of the failure: {}", peer, ex.what());
      } catch (...) {
        SIRIUS_LOG_WARN("exchange: could not tell rank {} of the failure", peer);
      }
    }
    // Freed now, not at ~transport: wait_sent() lets their query release its memory once they
    // are gone, and a failed transport is kept until the context ends. send() and close() queue
    // nothing once _error is set.
    std::vector<std::deque<engine::queued>> queued(e.queues.size());
    {
      std::lock_guard lock{_mutex};
      _error = error;
      for (std::size_t peer = 0; peer < queued.size(); ++peer) {
        queued[peer].swap(e.queues[peer]);
      }
    }
    e.send_stream.synchronize_no_throw();
    queued.clear();
    for (auto& p : e.out) {
      p.current.reset();
    }
    e.staged.clear();
    // A landing owns its whole batch from its first frame, and nothing completes one now. The
    // copies into it may still run on the receive stream.
    e.recv_stream.synchronize_no_throw();
    e.copies.clear();
    e.landings.clear();
    _room.notify_all();
    _released.notify_all();
    e.cb.on_error(error);
    return;
  }
  // Stopping. Ends and aborts already asked for need no credits, so they still go: a peer waits
  // for an end, and the rank closing it may tear down at once, as a barrier's last rank does.
  // Queued batches stay in the engine and are dropped with it.
  std::vector<std::pair<rank_t, std::string>> notes_out;
  {
    std::lock_guard lock{_mutex};
    for (auto const& a : _aborts) {
      for (rank_t peer = 0; peer < e.fab.world_size(); ++peer) {
        if (peer != e.fab.rank()) { notes_out.emplace_back(peer, encode(a)); }
      }
    }
    _aborts.clear();
    for (rank_t peer = 0; peer < e.queues.size(); ++peer) {
      for (auto const& q : e.queues[peer]) {
        if (!q.out) { notes_out.emplace_back(peer, encode(end_message{q.stream, q.batch})); }
      }
    }
  }
  for (auto& [peer, note] : notes_out) {
    try {
      e.fab.notify(peer, std::move(note));
    } catch (std::exception const& ex) {
      SIRIUS_LOG_WARN(
        "exchange: could not send rank {} an end or abort at shutdown: {}", peer, ex.what());
    } catch (...) {
      SIRIUS_LOG_WARN("exchange: could not send rank {} an end or abort at shutdown", peer);
    }
  }
}

}  // namespace sirius::exchange
