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

// TODO(phase4): two things change here once the implementations land — drop the [.] tag, and
// restore the broad [scan_manager] tag alongside [prefetch_api].
//
// select_split_index is declared noexcept and its Phase-1 body throws, so an unhidden case here
// would abort the whole test binary instead of failing. A Catch2 test spec *includes* hidden
// cases, so the broad tag stays off until then: with it, running `sirius_unittest "[scan_manager]"`
// would pull these in and abort.
//
// Two properties of split_connector's dequeue are load-bearing and are what this file exists to
// pin:
//   - liveness. Preferring a split whose IO has landed is safe; *refusing* one whose IO is still in
//     flight is a permanent hang, because leaving prefetch_progress::loading notifies
//     io::cache::entry_state's atomic and nothing in io/cache holds a reference to this connector's
//     condition variable. select_split_index therefore always returns a valid index.
//   - boundedness. get_next_split runs while the consumer holds sirius_pipeline::_status_mutex, and
//     every task completing on that pipeline blocks behind it. The selection must stay O(1) in the
//     queue length, which on local disk (where every state reads empty, so no landed split ever
//     short-circuits the walk) is only true because of the kSelectionWindow bound.
//
// The connector-level cases use metadata splits, which the public API cannot produce — the one
// public producer route, drain_cached_provider, only emits resident splits and needs a GPU batch.
// They go in through the split_connector_test_access friend seam instead.
//
// scan_operator_input::prefetch_state is virtual precisely so the selection policy can be driven
// end to end from here: reaching prefetch_progress::cached for real needs a live armed prefetching
// cache, and no shipped local backend arms one, so the synthetic-state subclass below is the only
// route to a non-empty state. See the @note on that method before devirtualizing it.

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
// explicitly defaulted moves on the class the type would be neither copyable nor movable, and
// every split in this file is moved on its way into the connector. Asserted at namespace scope so
// a regression is a compile error rather than a case failure.
static_assert(std::is_move_constructible_v<scan::scan_operator_input>,
              "scan_operator_input must stay move-constructible");
static_assert(std::is_move_assignable_v<scan::scan_operator_input>,
              "scan_operator_input must stay move-assignable");

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
/// count of how many times the connector folded this split's prefetch state.
class probe_scan_info : public scan::scan_info {
 public:
  probe_scan_info(std::size_t id, std::shared_ptr<sirius::io::sirius_datasource> datasource)
    : _id(id), _datasource(std::move(datasource))
  {
  }

  void for_each_datasource(
    const std::function<void(sirius::io::sirius_datasource&)>& visit) const override
  {
    ++folds;
    if (_datasource) { visit(*_datasource); }
  }

  [[nodiscard]] std::size_t datasource_count() const noexcept override
  {
    return _datasource ? 1 : 0;
  }

  /// The connector only ever sees a split as an @c operator_data, so the identity rides on the one
  /// @c scan_info value @c scan_operator_input forwards through that interface.
  [[nodiscard]] std::size_t estimated_bytes() const noexcept override { return _id; }

  /// How many times this split's prefetch state was folded; mutable because the fold is const.
  mutable std::size_t folds{0};

 private:
  std::size_t _id;
  std::shared_ptr<sirius::io::sirius_datasource> _datasource;
};

std::size_t id_of(const operator_data& split)
{
  auto const* input = dynamic_cast<const scan::scan_operator_input*>(&split);
  return input == nullptr ? 0 : input->get_estimated_size_in_bytes();
}

/// A split that reports whichever prefetch state the test asked for.
///
/// The only way to observe @c get_next_split's selection end to end. Reaching
/// @c prefetch_progress::cached for real requires a live armed prefetching cache, and no shipped
/// local backend arms one (both opt out of the ladder), so no *real* split can ever be steered off
/// the FIFO path — which would leave the policy testable only as a pure function.
/// @c split_connector reaches splits through @c dynamic_cast<const scan_operator_input*>, which
/// succeeds for this subclass, so overriding the (virtual) state accessor is enough.
class staged_split : public scan::scan_operator_input {
 public:
  staged_split(std::size_t id, prefetch_progress state)
    : scan_operator_input(std::make_unique<probe_scan_info>(id, nullptr)), _state(state)
  {
  }

  [[nodiscard]] prefetch_progress prefetch_state() const noexcept override
  {
    ++folds;
    return _state;
  }

  /// How many times the connector read this split's state; mutable because the read is const.
  mutable std::size_t folds{0};

 private:
  prefetch_progress _state;
};

/// Builds @ref staged_split s while keeping a raw handle on each, so a test can read back how
/// often the connector inspected it. The handles are valid only while the splits are alive — i.e.
/// while still queued, or while a pulled split is still held.
struct staged_factory {
  std::vector<staged_split*> probes;

  void push(split_connector& connector, std::size_t id, prefetch_progress state)
  {
    auto split = std::make_unique<staged_split>(id, state);
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

/// Builds metadata splits over one real parquet file, keeping a raw handle on each descriptor so a
/// test can read back how often the connector inspected it.
///
/// No prefetching cache is wired anywhere in this file — there is no mock ioctx and none is needed.
/// Every datasource here is a real kvikio one that was never fadvised, so every split reports
/// prefetch_progress::empty. That is the shipped local-disk shape, and it is exactly the case the
/// kSelectionWindow bound exists to keep cheap: with nothing ever landed, the walk never
/// short-circuits.
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

  [[nodiscard]] std::size_t total_folds() const
  {
    std::size_t total = 0;
    for (auto const* probe : probes) {
      total += probe->folds;
    }
    return total;
  }
};

}  // namespace

TEST_CASE("select_split_index prefers a landed prefetch", "[.][prefetch_api][split_connector]")
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
          "[.][prefetch_api][split_connector]")
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

TEST_CASE("select_split_index is total for a single split", "[.][prefetch_api][split_connector]")
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
          "[.][prefetch_api][split_connector]")
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

TEST_CASE("prefetch_if honours its look-ahead window", "[.][prefetch_api][split_connector]")
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

TEST_CASE("prefetch_if leaves every split in the queue", "[.][prefetch_api][split_connector]")
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

TEST_CASE("a connector holding more than the window inspects only the window",
          "[.][prefetch_api][split_connector]")
{
  split_factory factory;
  split_connector connector;
  factory.fill(connector, split_connector::kSelectionWindow * 3);

  // Held for the rest of the case: the fold counters live on the split that was handed out.
  auto pulled = connector.get_next_split();
  REQUIRE(pulled.has_value());

  // The dequeue runs under sirius_pipeline::_status_mutex, where every task completion on the
  // pipeline blocks behind it, so its cost must not scale with the queue length.
  CHECK(factory.total_folds() <= split_connector::kSelectionWindow);
  for (std::size_t i = split_connector::kSelectionWindow; i < factory.probes.size(); ++i) {
    INFO("split at index " << i << " is past the selection window");
    CHECK(factory.probes[i]->folds == 0);
  }
}

TEST_CASE("get_next_split is fifo when nothing has landed", "[.][prefetch_api][split_connector]")
{
  // The local-disk common path: no backend arms the prefetch ladder, so every split reports
  // prefetch_progress::empty and the bounded window must not reorder anything.
  constexpr std::size_t kPushed = split_connector::kSelectionWindow * 2;
  split_factory factory;
  split_connector connector;
  factory.fill(connector, kPushed);

  for (std::size_t expected = 1; expected <= kPushed; ++expected) {
    auto split = connector.get_next_split();
    REQUIRE(split.has_value());
    CHECK(id_of(**split) == expected);
  }
  CHECK_FALSE(connector.has_more_splits());
}

TEST_CASE("get_next_split hands out the split whose prefetch has landed",
          "[.][prefetch_api][split_connector]")
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

    // The two it stepped over keep their relative order and are still there to be served.
    auto next = connector.get_next_split();
    REQUIRE(next.has_value());
    CHECK(id_of(**next) == 1);
    auto last = connector.get_next_split();
    REQUIRE(last.has_value());
    CHECK(id_of(**last) == 3);
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
