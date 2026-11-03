#pragma once
// Test-only: included only in generated translation units, never production builds.
#include "io/rest/rest_reactor.hpp"
#include "io/rest/types.hpp"

#include <array>
#include <atomic>
#include <fstream>
#include <map>
#include <stdexcept>
namespace d1 {
using namespace sirius::io::rest;
struct event {
  char const* kind{};
  void const* op{};
  void const* object{};
  void const* context{};
  std::size_t offset{}, size{}, attempt{}, auth{}, bytes{};
  bool enabled{}, reuse{}, strong{}, delegated{}, cached{}, started{}, refreshed{};
  long status{}, rc{}, result{-1};
  rest_reactor::redirect_counters counts{};
  std::shared_ptr<std::string const> selected, current;
};
struct recorder {
  std::array<event, 4096> events{};
  std::size_t capacity{events.size()};
  std::atomic<bool> throw_next{false};
  std::size_t used{};  // single reactor writer; read only after shutdown/join
  std::atomic<bool> failed{false};
  std::atomic<bool> fail_next{false};
};
inline std::atomic<recorder*> active{nullptr};
inline void observe(char const* kind,
                    rest_io_op_request const& r,
                    std::shared_ptr<rest_reactor::reactor_context> const& ctx,
                    config const& cfg,
                    rest_reactor::redirect_counters counts,
                    bool strong,
                    long status       = 0,
                    long rc           = 0,
                    std::size_t bytes = 0,
                    long result       = -1) noexcept
{
  auto* store = active.load(std::memory_order_acquire);
  if (!store) return;
  try {
    if (store->throw_next.exchange(false)) throw std::bad_alloc{};
    if (store->fail_next.exchange(false) || store->used >= store->capacity ||
        store->used >= store->events.size()) {
      store->failed.store(true);
      return;
    }
    event e;
    e.kind      = kind;
    e.op        = r.op.get();
    e.object    = r.op->obj.get();
    e.context   = ctx.get();
    e.offset    = r.op->io_rng.offset;
    e.size      = r.op->io_rng.size;
    e.attempt   = r.attempt;
    e.auth      = r.auth_attempt;
    e.bytes     = bytes;
    e.enabled   = cfg.cluster_redirect.enabled;
    e.reuse     = cfg.cluster_redirect.route_reuse == redirect_route_reuse::same_open;
    e.strong    = strong;
    e.delegated = !r.delegated_url.empty();
    e.cached    = r.route_cached;
    e.started   = r.route_started_cached;
    e.refreshed = r.route_refreshed;
    e.status    = status;
    e.rc        = rc;
    e.result    = result;
    e.counts    = counts;
    e.selected  = r.route_snapshot;
    if (auto const* f = dynamic_cast<rest_io_object const*>(r.op->obj.get()))
      e.current = f->route(ctx);
    store->events[store->used++] = std::move(e);
  } catch (...) {
    store->failed.store(true);
  }
}
struct session {
  recorder data;
  session()
  {
    recorder* expected = nullptr;
    if (!active.compare_exchange_strong(expected, &data))
      throw std::runtime_error("nested D1 observer");
  }
  // Fixture MUST stop/join the reactor before this destructor.
  ~session() { active.store(nullptr); }
};
// All allocation / formatting below happens after the reactor has joined.
inline void dump(recorder const& r,
                 std::ostream& out,
                 void const* expected_object,
                 void const* expected_context)
{
  std::map<void const*, int> pubs;
  std::map<std::size_t, void const*> ops;
  std::map<std::size_t, int> physical_ids;
  auto pub = [&](auto const& p) {
    if (!p) return 0;
    auto [it, added] = pubs.emplace(p.get(), pubs.size() + 1);
    return it->second;
  };
  out << "{\"kind\":\"observer\",\"failed\":" << r.failed.load() << ",\"events\":" << r.used
      << "}\n";
  for (std::size_t i = 0; i < r.used; ++i) {
    auto const& e = r.events[i];
    if (e.object != expected_object || e.context != expected_context || !e.op)
      throw std::runtime_error("D1 identity mismatch");
    auto [it, added] = ops.emplace(e.offset, e.op);
    if (it->second != e.op) throw std::runtime_error("D1 operation changed within range");
    // Pointer identity of retained shared_ptrs, never URL-based identity.
    auto const current = pub(e.current), selected = pub(e.selected);
    auto [physical_it, physical_added] = physical_ids.emplace(e.offset, physical_ids.size() + 1);
    out << "{\"kind\":\"" << e.kind << "\",\"id\":" << e.offset
        << ",\"physical\":" << physical_it->second
        << ",\"open\":1,\"context\":1,\"size\":" << e.size << ",\"current\":" << current
        << ",\"selected\":" << selected << ",\"attempt\":" << e.attempt << ",\"auth\":" << e.auth
        << ",\"enabled\":" << e.enabled << ",\"reuse\":" << e.reuse << ",\"strong\":" << e.strong
        << ",\"delegated\":" << e.delegated << ",\"cached\":" << e.cached
        << ",\"started\":" << e.started << ",\"refreshed\":" << e.refreshed
        << ",\"status\":" << e.status << ",\"rc\":" << e.rc << ",\"bytes\":" << e.bytes
        << ",\"result\":" << e.result << ",\"seeds\":" << e.counts.route_seeds
        << ",\"hits\":" << e.counts.route_hits << ",\"refreshes\":" << e.counts.route_refreshes
        << "}\n";
  }
}
}  // namespace d1
