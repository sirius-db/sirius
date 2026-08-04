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

// Three properties of split_connector's dequeue are load-bearing and are what this file exists to
// pin:
//   - liveness. Preferring a split whose IO has landed is safe; *refusing* one whose IO is still in
//     flight is a permanent hang, because leaving prefetch_progress::loading notifies
//     io::cache::entry_state's atomic and nothing in io/cache holds a reference to this connector's
//     condition variable. select_split_index therefore always returns a valid index.
//   - the arming gate. The selection walk runs only when some pushed split reported
//     can_land_while_queued, i.e. only when a backend on this connector activates its prefetch
//     before the dequeue. No shipped backend does, so the shipped dequeue is the same move-and-
//     pop_front it was before the policy existed: no state reads, no virtual calls, no walks.
//   - boundedness, when the gate IS open. get_next_split runs while the consumer holds
//     sirius_pipeline::_status_mutex, and every task completing on that pipeline blocks behind it.
//     The selection must stay O(1) in the queue length AND in each split's datasource count, which
//     is what kSelectionWindow and kSelectionFoldBudget bound respectively.
//
// The connector-level cases use metadata splits, which the public API cannot produce — the one
// public producer route, drain_cached_provider, only emits resident splits and needs a GPU batch.
// They go in through the split_connector_test_access friend seam instead.
//
// scan_operator_input::prefetch_state and ::can_land_while_queued are virtual precisely so the
// selection policy can be driven end to end from here: arming the gate needs a pre-dequeue
// activation stage and reaching prefetch_progress::cached needs a live armed prefetching cache,
// and no shipped local backend provides either. The synthetic subclass below is the only route to
// both. See the @note on each method before devirtualizing it.

#include "scan_manager/split_connector_test_access.hpp"

#include <catch.hpp>
#include <io/cache/types.hpp>
#include <io/kvikio/kvikio_context.hpp>
#include <io/sirius_datasource.hpp>
#include <op/scan/gpu_ingestible_types.hpp>
#include <op/scan/sirius_gpu_scan_operator_data.hpp>
#include <op/sirius_physical_operator.hpp>

#include <cstddef>
#include <filesystem>
#include <functional>
#include <memory>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

namespace scan = sirius::op::scan;

using sirius::io::cache::prefetch_progress;
using sirius::op::operator_data;
using sirius::scan_manager::prefetch_kind;
using sirius::scan_manager::split_connector;
using sirius::scan_manager::split_connector_test_access;

// scan_operator_input has a user-declared destructor (it reports the split's disposal to the
// per-query prefetch counters), which suppresses the implicit move operations, and its
// unique_ptr<scan_info> variant alternative separately deletes the copy operations. Without the
// explicitly defaulted move CONSTRUCTOR the type would be neither copyable nor movable, and every
// split in this file is moved on its way into the connector.
//
// Move ASSIGNMENT must stay deleted, and that is the create/dispose invariant, not a style rule:
// `a = std::move(b)` would overwrite a's prefetching_state_manager handle without reporting a's
// disposal, so the split a used to be would be counted as created and never disposed and n_live
// would leak for the rest of the query. Move construction has no such hole -- it leaves the
// source's handle null, so the split still accounts for exactly one creation and one disposal.
//
// Asserted at namespace scope so a regression is a compile error rather than a case failure.
static_assert(std::is_move_constructible_v<scan::scan_operator_input>,
              "scan_operator_input must stay move-constructible");
static_assert(!std::is_move_assignable_v<scan::scan_operator_input>,
              "scan_operator_input must NOT be move-assignable: assignment would drop the source "
              "split's disposal report and leak prefetching_state_manager::n_live");

std::size_t select(const std::vector<prefetch_progress>& states)
{
  return split_connector::select_split_index(states);
}

std::filesystem::path project_root()
{
#ifdef SIRIUS_PROJECT_ROOT
  return std::filesystem::path{SIRIUS_PROJECT_ROOT};
#else
  return std::filesystem::current_path();
#endif
}

std::string nation_parquet()
{
  return (project_root() / "test/cpp/integration/data/parquet/nation.parquet").string();
}

/// Metadata-split descriptor carrying a push-order identity, optionally one real datasource, and a
/// count of how many times anything walked this split's datasources.
class probe_scan_info : public scan::scan_info {
 public:
  /// @param id          Push-order identity, readable back through @ref id_of.
  /// @param datasource  The real datasource to visit, or null to visit nothing.
  /// @param declared    What @ref datasource_count reports, which is what the connector records
  ///                    as this split's fold cost. Defaults to the real count. Set it above 1 to
  ///                    stand in for a @c parquet_split_info with many row-group slices without
  ///                    having to open that many files.
  probe_scan_info(std::size_t id,
                  std::shared_ptr<sirius::io::sirius_datasource> datasource,
                  std::size_t declared = 0)
    : declared_datasources(declared != 0 ? declared : (datasource ? 1U : 0U)),
      _id(id),
      _datasource(std::move(datasource))
  {
  }

  void for_each_datasource(
    const std::function<void(sirius::io::sirius_datasource&)>& visit) const override
  {
    ++walks;
    // Kept consistent with datasource_count() so a wide split really does cost that many visits.
    // With a null datasource there is nothing to visit; only staged_split does that, and it
    // overrides both methods that would walk.
    if (!_datasource) { return; }
    for (std::size_t i = 0; i < declared_datasources; ++i) {
      visit(*_datasource);
    }
  }

  [[nodiscard]] std::size_t datasource_count() const noexcept override
  {
    return declared_datasources;
  }

  /// The connector only ever sees a split as an @c operator_data, so the identity rides on the one
  /// @c scan_info value @c scan_operator_input forwards through that interface.
  [[nodiscard]] std::size_t estimated_bytes() const noexcept override { return _id; }

  std::size_t declared_datasources;

  /// How many times this split's datasources were walked, by anything: the connector's one
  /// per-connector arming check, or a consumer-side prefetch_state() fold. Mutable because the
  /// walk is const.
  mutable std::size_t walks{0};

 private:
  std::size_t _id;
  std::shared_ptr<sirius::io::sirius_datasource> _datasource;
};

std::size_t id_of(const operator_data& split)
{
  auto const* input = dynamic_cast<const scan::scan_operator_input*>(&split);
  return input == nullptr ? 0 : input->get_estimated_size_in_bytes();
}

/// A split that reports whichever prefetch state the test asked for, on a connector it arms or
/// leaves unarmed on request.
///
/// The only way to observe @c get_next_split's selection end to end. Arming the gate for real
/// requires a backend that activates its prefetch before the dequeue, and reaching
/// @c prefetch_progress::cached requires a live armed prefetching cache; no shipped local backend
/// provides either, so no *real* split can ever be steered off the FIFO path — which would leave
/// the policy testable only as a pure function. @c split_connector reaches splits through
/// @c dynamic_cast<const scan_operator_input*>, which succeeds for this subclass, so overriding
/// the two virtual accessors is enough.
///
/// The descriptor carries no real datasource but declares a non-zero @c datasource_count, which is
/// what makes the split an io candidate (so the connector runs its arming check on it) and what
/// gives it a fold cost. Nothing walks those declared datasources, because both methods that
/// would are overridden here.
class staged_split : public scan::scan_operator_input {
 public:
  /// @param can_land  What @ref can_land_while_queued reports, i.e. whether this split arms the
  ///                  connector's selection gate.
  /// @param fold_cost What the split declares as its datasource count, which the connector spends
  ///                  against @c kSelectionFoldBudget.
  staged_split(std::size_t id,
               prefetch_progress state,
               bool can_land         = true,
               std::size_t fold_cost = 1)
    : scan_operator_input(std::make_unique<probe_scan_info>(id, nullptr, fold_cost)),
      _state(state),
      _can_land(can_land)
  {
  }

  [[nodiscard]] prefetch_progress prefetch_state() const noexcept override
  {
    ++folds;
    return _state;
  }

  [[nodiscard]] bool can_land_while_queued() const noexcept override { return _can_land; }

  /// How many times the connector read this split's state; mutable because the read is const.
  mutable std::size_t folds{0};

 private:
  prefetch_progress _state;
  bool _can_land;
};

/// Builds @ref staged_split s while keeping a raw handle on each, so a test can read back how
/// often the connector inspected it. The handles are valid only while the splits are alive — i.e.
/// while still queued, or while a pulled split is still held.
struct staged_factory {
  std::vector<staged_split*> probes;

  void push(split_connector& connector,
            std::size_t id,
            prefetch_progress state,
            bool can_land         = true,
            std::size_t fold_cost = 1)
  {
    auto split = std::make_unique<staged_split>(id, state, can_land, fold_cost);
    probes.push_back(split.get());
    split_connector_test_access::push(connector, std::move(split));
  }

  [[nodiscard]] std::size_t total_folds() const
  {
    std::size_t total = 0;
    for (auto const* probe : probes) {
      total += probe->folds;
    }
    return total;
  }
};

/// Builds real metadata splits over one real parquet file, keeping a raw handle on each descriptor
/// so a test can read back how often anything walked its datasources.
///
/// No prefetching cache is wired anywhere in this file — there is no mock ioctx and none is needed.
/// Every datasource here is a real kvikio one, whose activation stage is `none`, so a connector
/// filled from here never arms and every split would report prefetch_progress::empty if anything
/// asked. **That is the shipped local-disk shape**, and these splits are how this file pins it:
/// the dequeue must be a plain pop_front, and the only datasource walk in the connector's whole
/// life is the single arming check on the first push.
///
/// The @ref probes pointers are only valid while the splits are alive — i.e. while they are still
/// queued or while a pulled split is still held.
struct split_factory {
  std::shared_ptr<sirius::io::kvikio_context> ioctx =
    std::make_shared<sirius::io::kvikio_context>();
  std::unique_ptr<sirius::io::sirius_datasource> origin = ioctx->open_datasource(nation_parquet());
  std::vector<probe_scan_info*> probes;

  /// @param id              Push-order identity, readable back through @ref id_of.
  /// @param with_datasource Whether the split has IO to prefetch (@c is_io_prefetchable).
  std::unique_ptr<operator_data> make(std::size_t id, bool with_datasource)
  {
    std::shared_ptr<sirius::io::sirius_datasource> datasource;
    if (with_datasource) {
      datasource = std::shared_ptr<sirius::io::sirius_datasource>{origin->duplicate()};
    }
    auto info = std::make_unique<probe_scan_info>(id, std::move(datasource));
    probes.push_back(info.get());
    return std::make_unique<scan::scan_operator_input>(std::move(info));
  }

  /// Push @p n io-prefetchable splits with ids 1..n, in that order.
  void fill(split_connector& connector, std::size_t n)
  {
    for (std::size_t id = 1; id <= n; ++id) {
      split_connector_test_access::push(connector, make(id, /*with_datasource=*/true));
    }
  }

  [[nodiscard]] std::size_t total_walks() const
  {
    std::size_t total = 0;
    for (auto const* probe : probes) {
      total += probe->walks;
    }
    return total;
  }
};

}  // namespace

TEST_CASE("select_split_index prefers a landed prefetch",
          "[scan_manager][prefetch_api][split_connector]")
{
  SECTION("the front split wins when nothing has landed")
  {
    CHECK(select({prefetch_progress::prepared, prefetch_progress::prepared}) == 0);
  }

  SECTION("a cached split is preferred over an earlier prepared one")
  {
    CHECK(select({prefetch_progress::prepared,
                  prefetch_progress::cached,
                  prefetch_progress::prepared}) == 1);
  }

  SECTION("the first cached split wins")
  {
    CHECK(select({prefetch_progress::cached, prefetch_progress::cached}) == 0);
  }

  SECTION("a loading split is skipped in favour of a later one")
  {
    CHECK(select({prefetch_progress::loading, prefetch_progress::prepared}) == 1);
  }

  SECTION("cached outranks merely not-loading")
  {
    CHECK(select(
            {prefetch_progress::loading, prefetch_progress::prepared, prefetch_progress::cached}) ==
          2);
  }
}

TEST_CASE("select_split_index never starves when every split is loading",
          "[scan_manager][prefetch_api][split_connector]")
{
  // Liveness regression gate, not a tie-break. Refusing a loading split has no wakeup path: the
  // split leaving `loading` notifies io::cache::entry_state's atomic, not this connector's
  // condition variable, and the producer may already have pushed its last split. A future
  // "just wait for a non-loading one" refactor fails here.
  CHECK(select({prefetch_progress::loading}) == 0);
  CHECK(
    select({prefetch_progress::loading, prefetch_progress::loading, prefetch_progress::loading}) ==
    0);
}

TEST_CASE("select_split_index is total for a single split",
          "[scan_manager][prefetch_api][split_connector]")
{
  for (auto const state : {prefetch_progress::empty,
                           prefetch_progress::cancelled,
                           prefetch_progress::prepared,
                           prefetch_progress::loading,
                           prefetch_progress::cached,
                           prefetch_progress::in_use,
                           prefetch_progress::evicting}) {
    INFO("prefetch_progress = " << static_cast<int>(state));
    CHECK(select({state}) == 0);
  }
}

TEST_CASE("n_prefetchable counts io candidates as they are pushed and pulled",
          "[scan_manager][prefetch_api][split_connector]")
{
  split_factory factory;
  split_connector connector;
  split_connector_test_access::push(connector, factory.make(1, /*with_datasource=*/true));
  split_connector_test_access::push(connector, factory.make(2, /*with_datasource=*/true));
  split_connector_test_access::push(connector, factory.make(3, /*with_datasource=*/false));

  CHECK(connector.n_prefetchable(prefetch_kind::io) == 2);
  CHECK(connector.n_prefetchable(prefetch_kind::memory) == 0);

  auto pulled = connector.get_next_split();
  REQUIRE(pulled.has_value());
  CHECK(id_of(**pulled) == 1);

  // The count is maintained incrementally on the consumer path too, not recomputed by an O(n)
  // rescan under the mutex.
  CHECK(connector.n_prefetchable(prefetch_kind::io) == 1);
  CHECK(connector.n_prefetchable(prefetch_kind::memory) == 0);
}

TEST_CASE("prefetch_if honours its look-ahead window",
          "[scan_manager][prefetch_api][split_connector]")
{
  SECTION("a window of zero hints nothing")
  {
    split_factory factory;
    split_connector connector;
    factory.fill(connector, 3);

    std::size_t inspected = 0;
    auto const hinted     = connector.prefetch_if(0, prefetch_kind::io, [&](const operator_data&) {
      ++inspected;
      return true;
    });
    CHECK(hinted == 0);
    CHECK(inspected == 0);
  }

  SECTION("the window bounds how many splits are inspected")
  {
    split_factory factory;
    split_connector connector;
    factory.fill(connector, 5);

    std::vector<std::size_t> inspected;
    auto const hinted =
      connector.prefetch_if(2, prefetch_kind::io, [&](const operator_data& split) {
        inspected.push_back(id_of(split));
        return true;
      });
    CHECK(hinted == 2);
    // Front-to-back and no further: splits 3, 4 and 5 are never looked at.
    CHECK(inspected == std::vector<std::size_t>{1, 2});
  }

  SECTION("the predicate filters within the window")
  {
    split_factory factory;
    split_connector connector;
    factory.fill(connector, 4);

    auto const hinted = connector.prefetch_if(
      4, prefetch_kind::io, [](const operator_data& split) { return id_of(split) % 2 == 0; });
    CHECK(hinted == 2);
  }
}

TEST_CASE("prefetch_if counts a split's ladder rung only the first time it advances",
          "[scan_manager][prefetch_api][split_connector]")
{
  // The ladder must be monotone per split. This walk restarts from the queue front on every
  // invocation and the dequeue fires task_queued again afterwards, so counting inspections rather
  // than advances would grow prefetching_state_manager::n_task_queued without bound on a queue
  // that is not draining -- which is exactly when the depleted hook fires repeatedly.
  split_factory factory;
  split_connector connector;
  factory.fill(connector, 3);

  auto const always = [](const operator_data&) { return true; };
  CHECK(connector.prefetch_if(3, prefetch_kind::io, always) == 3);
  CHECK(connector.prefetch_if(3, prefetch_kind::io, always) == 0);
  CHECK(connector.prefetch_if(3, prefetch_kind::io, always) == 0);
}

TEST_CASE("prefetch_if leaves every split in the queue",
          "[scan_manager][prefetch_api][split_connector]")
{
  split_factory factory;
  split_connector connector;
  factory.fill(connector, 3);

  connector.prefetch_if(3, prefetch_kind::io, [](const operator_data&) { return true; });

  for (std::size_t expected = 1; expected <= 3; ++expected) {
    auto split = connector.get_next_split();
    REQUIRE(split.has_value());
    CHECK(id_of(**split) == expected);
  }
  CHECK_FALSE(connector.has_more_splits());
}

TEST_CASE("an armed connector holding more than the window inspects only the window",
          "[scan_manager][prefetch_api][split_connector]")
{
  constexpr std::size_t kPushed = split_connector::kSelectionWindow * 3;
  staged_factory factory;
  split_connector connector;
  // Nothing landed, so the walk never short-circuits: only kSelectionWindow stops it. One fold
  // each, so the queue stays well inside kSelectionFoldBudget and this case isolates the window.
  for (std::size_t id = 1; id <= kPushed; ++id) {
    factory.push(connector, id, prefetch_progress::prepared);
  }

  // Held for the rest of the case: the fold counters live on the split that was handed out.
  auto pulled = connector.get_next_split();
  REQUIRE(pulled.has_value());
  CHECK(id_of(**pulled) == 1);  // nothing landed, so rule 2/3 still return the queue front

  // The dequeue runs under sirius_pipeline::_status_mutex, where every task completion on the
  // pipeline blocks behind it, so its cost must not scale with the queue length.
  CHECK(factory.total_folds() == split_connector::kSelectionWindow);
  for (std::size_t i = split_connector::kSelectionWindow; i < factory.probes.size(); ++i) {
    INFO("split at index " << i << " is past the selection window");
    CHECK(factory.probes[i]->folds == 0);
  }
}

TEST_CASE("an unarmed connector dequeues strictly fifo and never reads a split's state",
          "[scan_manager][prefetch_api][split_connector]")
{
  // The shipped path, and the whole point of the arming gate. Every datasource here is a real
  // kvikio one, whose activation stage is `none`, so no queued split could ever report `cached`
  // and the selection walk could not beat the queue front even if it ran. It does not run: the
  // dequeue is the same move-and-pop_front it was before the policy existed.
  constexpr std::size_t kPushed = split_connector::kSelectionWindow * 2;
  split_factory factory;
  split_connector connector;
  factory.fill(connector, kPushed);

  // The one datasource walk this connector will ever do: push_split's arming check, on the
  // producer thread, on the first io candidate only.
  CHECK(factory.total_walks() == 1);

  // Pulled splits are kept alive so the counters below are read off live probes.
  std::vector<std::optional<std::unique_ptr<operator_data>>> pulled;
  for (std::size_t expected = 1; expected <= kPushed; ++expected) {
    auto split = connector.get_next_split();
    REQUIRE(split.has_value());
    CHECK(id_of(**split) == expected);
    pulled.push_back(std::move(split));
  }
  CHECK_FALSE(connector.has_more_splits());

  // Still one: kPushed dequeues added no consumer-side folds at all. Any prefetch_state() read on
  // this path would show up here, because scan_operator_input::prefetch_state folds through
  // for_each_datasource.
  CHECK(factory.total_walks() == 1);
}

TEST_CASE("a connector arms only when a split can land while queued",
          "[scan_manager][prefetch_api][split_connector]")
{
  // Identical queue shapes; the only difference is what the splits report for
  // can_land_while_queued. That is the gate, isolated.
  SECTION("an unarmed connector ignores a landed split behind the queue front")
  {
    staged_factory factory;
    split_connector connector;
    factory.push(connector, 1, prefetch_progress::loading, /*can_land=*/false);
    factory.push(connector, 2, prefetch_progress::cached, /*can_land=*/false);

    auto split = connector.get_next_split();
    REQUIRE(split.has_value());
    CHECK(id_of(**split) == 1);         // FIFO: the cached split at index 1 is never looked for
    CHECK(factory.total_folds() == 0);  // and no state was read to find that out
  }

  SECTION("one split that can land arms the connector for all of them")
  {
    // Latched per connector, not per split: the first io candidate decides, and the splits behind
    // it are selected over without being asked again.
    staged_factory factory;
    split_connector connector;
    factory.push(connector, 1, prefetch_progress::loading, /*can_land=*/true);
    factory.push(connector, 2, prefetch_progress::cached, /*can_land=*/false);

    auto split = connector.get_next_split();
    REQUIRE(split.has_value());
    CHECK(id_of(**split) == 2);
    CHECK(factory.total_folds() == 2);
  }
}

TEST_CASE("the fold budget bounds the selection walk",
          "[scan_manager][prefetch_api][split_connector]")
{
  // kSelectionWindow bounds how many splits are inspected; this bounds the work inside them. A
  // parquet_split_info carries one datasource per row-group slice, so a per-split bound alone
  // still leaves the walk unbounded on exactly the split type production queues.
  //
  // The budget governs the OPTIONAL splits -- indices 1..kSelectionWindow-1 -- and nothing else.
  // Split 0's fold is not charged against it, because split 0 is rule 3's mandatory fallback
  // candidate and rules 1 and 2's first candidate: an armed selection cannot answer without
  // classifying it, so its fold is unavoidable and the true per-dequeue worst case is
  // fold_cost(front) + kSelectionFoldBudget.
  SECTION("a wide front split does not spend the budget for the splits behind it")
  {
    // Charging the front split would end the policy on production's split shape: one
    // parquet_split_info of a few hundred datasources exhausts a 32-fold budget before index 1 is
    // considered, the walk stops at one entry, rule 3 fires, and an armed connector degenerates
    // into the FIFO it already is when unarmed -- with kSelectionWindow unreachable.
    staged_factory factory;
    split_connector connector;
    factory.push(connector,
                 1,
                 prefetch_progress::prepared,
                 /*can_land=*/true,
                 /*fold_cost=*/split_connector::kSelectionFoldBudget + 8);
    for (std::size_t id = 2; id <= split_connector::kSelectionWindow; ++id) {
      factory.push(connector, id, prefetch_progress::cached);
    }

    // Index 1 is reached and its landing wins under rule 1 -- which a budget seeded with the front
    // split's cost could never observe.
    auto pulled = connector.get_next_split();
    REQUIRE(pulled.has_value());
    CHECK(id_of(**pulled) == 2);
    // The front split, then the landed one; rule 1 takes the first landed split, so the walk stops
    // there and the six behind it are never read.
    CHECK(factory.total_folds() == 2);
  }

  SECTION("a wide split behind the front is refused before it is inspected")
  {
    // The one thing the budget is now responsible for: index 0's fold is unavoidable, everything
    // after it is optional and has to fit.
    staged_factory factory;
    split_connector connector;
    factory.push(connector, 1, prefetch_progress::prepared, /*can_land=*/true);
    factory.push(connector,
                 2,
                 prefetch_progress::cached,
                 /*can_land=*/true,
                 /*fold_cost=*/split_connector::kSelectionFoldBudget + 1);

    auto pulled = connector.get_next_split();
    REQUIRE(pulled.has_value());
    CHECK(id_of(**pulled) == 1);        // the landed split at index 1 was too wide to look at
    CHECK(factory.total_folds() == 1);  // and its state was never read
  }

  SECTION("the budget stops the walk before the window does")
  {
    // Eight splits at a quarter of the budget each. Split 0 is free, so the four optional splits
    // at indices 1..4 exhaust the budget exactly; index 5 would take the total to 40. The walk
    // therefore stops after five of the eight, short of the window.
    constexpr std::size_t kCost = split_connector::kSelectionFoldBudget / 4;
    staged_factory factory;
    split_connector connector;
    for (std::size_t id = 1; id <= split_connector::kSelectionWindow; ++id) {
      factory.push(connector, id, prefetch_progress::prepared, /*can_land=*/true, kCost);
    }

    auto pulled = connector.get_next_split();
    REQUIRE(pulled.has_value());
    CHECK(id_of(**pulled) == 1);
    CHECK(factory.total_folds() == 5);
  }
}

TEST_CASE("get_next_split hands out the split whose prefetch has landed",
          "[scan_manager][prefetch_api][split_connector]")
{
  // select_split_index's policy driven through the real dequeue rather than as a pure function.
  // The states are synthetic (see staged_split): no shipped backend can produce a landed split.
  SECTION("a landed split jumps ahead of the queue front")
  {
    staged_factory factory;
    split_connector connector;
    factory.push(connector, 1, prefetch_progress::loading);
    factory.push(connector, 2, prefetch_progress::cached);
    factory.push(connector, 3, prefetch_progress::prepared);

    auto landed = connector.get_next_split();
    REQUIRE(landed.has_value());
    CHECK(id_of(**landed) == 2);

    // The policy re-applies on every dequeue, so the remainder is not served FIFO: the queue is
    // now [1 loading, 3 prepared], and rule 2 steps over the loading split again. Skipping 1 is
    // deliberate — materialize_table fires prefetch(disposable), which cancels the handle, so
    // handing out a split whose IO is still in flight would throw that IO away. Split 1 is served
    // last, once nothing better is left (rule 3), which is what keeps the connector live.
    auto next = connector.get_next_split();
    REQUIRE(next.has_value());
    CHECK(id_of(**next) == 3);
    auto last = connector.get_next_split();
    REQUIRE(last.has_value());
    CHECK(id_of(**last) == 1);
  }

  SECTION("the front split is returned when every split is still loading")
  {
    // Liveness, end to end. Refusing a loading split has no wakeup path — leaving `loading`
    // notifies io::cache::entry_state's atomic, never this connector's condition variable — so
    // the dequeue must hand one out rather than wait for a better candidate.
    staged_factory factory;
    split_connector connector;
    factory.push(connector, 1, prefetch_progress::loading);
    factory.push(connector, 2, prefetch_progress::loading);
    factory.push(connector, 3, prefetch_progress::loading);

    auto split = connector.get_next_split();
    REQUIRE(split.has_value());
    CHECK(id_of(**split) == 1);
  }

  SECTION("inspection stops once the front split has landed")
  {
    // The best case must stay as cheap as the plain pop_front it replaced: rule 1 takes the first
    // cached entry, so nothing past it can change the answer and nothing past it is read.
    staged_factory factory;
    split_connector connector;
    factory.push(connector, 1, prefetch_progress::cached);
    for (std::size_t id = 2; id <= split_connector::kSelectionWindow * 2; ++id) {
      factory.push(connector, id, prefetch_progress::prepared);
    }

    // Held for the rest of the section: the fold counter lives on the split that was handed out.
    auto pulled = connector.get_next_split();
    REQUIRE(pulled.has_value());
    CHECK(id_of(**pulled) == 1);
    CHECK(factory.total_folds() == 1);
  }
}
