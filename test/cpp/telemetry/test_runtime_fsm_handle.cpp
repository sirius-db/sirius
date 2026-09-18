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

#include "catch.hpp"
#include "telemetry/runtime_fsm_handle.hpp"

#include <concepts>
#include <optional>
#include <type_traits>
#include <utility>

namespace {

using query_handle = sirius::telemetry::runtime_fsm_handle<quent::Query,
                                                           quent::query_state::Init,
                                                           quent::query_state::Planning,
                                                           quent::query_state::Executing,
                                                           quent::query_state::Exit>;

using query_init_handle = quent::FsmHandle<quent::Query, quent::query_state::Init>;

using data_batch_handle =
  sirius::telemetry::runtime_fsm_handle<quent::DataBatch,
                                        quent::data_batch_state::Stationary,
                                        quent::data_batch_state::Destructed>;

static_assert(std::move_constructible<query_handle>);
static_assert(std::is_move_assignable_v<query_handle>);
static_assert(!std::copy_constructible<query_handle>);
static_assert(!std::is_copy_assignable_v<query_handle>);
static_assert(!std::default_initializable<query_handle>);

}  // namespace

TEST_CASE("Runtime FSM handle retains typestate across transitions",
          "[telemetry][runtime_fsm_handle]")
{
  auto context    = quent::Context::none();
  auto init       = context.query_observer()->handle().init(quent::query::Init{
          .instance_name  = "runtime-fsm-handle-test",
          .query_group_id = quent::query_group::QueryGroupId(quent::now_v7()),
  });
  const auto uuid = init.id().raw();

  query_handle handle{std::move(init)};
  CHECK(handle.uuid() == uuid);
  CHECK(handle.holds<quent::query_state::Init>());
  CHECK(handle.get_if<quent::query_state::Init>() != nullptr);
  CHECK(handle.get_if<quent::query_state::Planning>() == nullptr);

  REQUIRE(handle.transition<quent::query_state::Init>(
    [](auto&& current) { return std::move(current).planning(); }));
  CHECK(handle.uuid() == uuid);
  CHECK(handle.holds<quent::query_state::Planning>());

  REQUIRE(
    handle.transition<quent::query_state::Init, quent::query_state::Planning>([](auto&& current) {
      using current_handle = std::remove_cvref_t<decltype(current)>;
      if constexpr (std::same_as<current_handle, query_init_handle>) {
        return std::move(current).planning();
      } else {
        return std::move(current).executing();
      }
    }));
  CHECK(handle.holds<quent::query_state::Executing>());

  bool invoked = false;
  CHECK_FALSE(handle.transition<quent::query_state::Init>([&](auto&& current) {
    invoked = true;
    return std::move(current).planning();
  }));
  CHECK_FALSE(invoked);

  const auto& const_handle = handle;
  CHECK(const_handle.get_if<quent::query_state::Executing>() != nullptr);

  REQUIRE(handle.transition<quent::query_state::Executing>(
    [](auto&& current) { return std::move(current).exit(); }));
  CHECK(handle.holds<quent::query_state::Exit>());
}

TEST_CASE("Runtime FSM handle supports self transitions", "[telemetry][runtime_fsm_handle]")
{
  auto context    = quent::Context::none();
  auto stationary = context.data_batch_observer()
                      ->handle()
                      .constructed(quent::data_batch::Constructed{
                        .instance_name          = "runtime-fsm-handle-self-transition-test",
                        .data_batch_id          = 1,
                        .producer_pipeline_uuid = quent::operator_::OperatorId(quent::now_v7()),
                      })
                      .stationary(quent::data_batch::Stationary{.memory = std::nullopt});
  const auto uuid = stationary.id().raw();

  data_batch_handle handle{std::move(stationary)};
  REQUIRE(handle.transition<quent::data_batch_state::Stationary>([](auto&& current) {
    return std::move(current).stationary(quent::data_batch::Stationary{.memory = std::nullopt});
  }));
  CHECK(handle.uuid() == uuid);
  CHECK(handle.holds<quent::data_batch_state::Stationary>());

  REQUIRE(handle.transition<quent::data_batch_state::Stationary>(
    [](auto&& current) { return std::move(current).destructed(); }));
  CHECK(handle.holds<quent::data_batch_state::Destructed>());
}
