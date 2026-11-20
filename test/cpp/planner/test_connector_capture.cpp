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

#include "op/scan/table_scan/bound_read_view.hpp"
#include "transparent/read_view_registry.hpp"
#include "utils/connector_admission_test_utils.hpp"

#include <catch.hpp>
#include <duckdb/common/multi_file/multi_file_function.hpp>
#include <duckdb/planner/operator/logical_get.hpp>

namespace {
using namespace sirius::test::connector_admission;
using namespace sirius::op::scan;

constexpr char capture_key[]   = "connector_capture_probe";
constexpr char declined_text[] = "test connector profile is unsupported";

struct counting_provider {
  unsigned entries   = 0;
  bool saw_admission = false;

  void enter(std::shared_ptr<preparation_admission> const& admission)
  {
    ++entries;
    saw_admission = admission && admission->decision.deferred;
  }
};

struct capture_probe : duckdb::ClientContextState {
  std::string binding_key = "test.capture.k1";
  std::shared_ptr<bound_read_identity const> captured;
  counting_provider provider;
  unsigned captures = 0, capture_begin = 0, capture_end = 0, provider_entry = 0;
  std::vector<unsigned> entries_at_capture_begin, entries_at_capture_end;
  bool pre_decline = false, preparation_seen_in_capture = false;
};

auto capture_for(duckdb::ClientContext& context)
{
  auto probe = context.registered_state->Get<capture_probe>(capture_key);
  if (!probe) throw std::logic_error("connector capture probe is missing");
  return probe;
}

bool verify_binding(duckdb::BaseTableFunction const&,
                    duckdb::FunctionData const*,
                    duckdb::ClientContext& context)
{
  auto probe = capture_for(context);
  return !probe->captured ||
         std::get<provider_read_identity>(probe->captured->data_view).key == probe->binding_key;
}

bound_read_view capture_binding(duckdb::BoundTableFunction const& function,
                                duckdb::FunctionData const* data,
                                duckdb::ClientContext& context)
{
  auto probe     = capture_for(context);
  auto admission = probe_for(context);
  ++probe->captures;
  probe->capture_begin = ++admission->clock;
  probe->entries_at_capture_begin.push_back(probe->provider.entries);
  probe->preparation_seen_in_capture |= !admission->requests.empty();

  auto const* bind = dynamic_cast<duckdb::MultiFileBindData const*>(data);
  if (!bind) throw std::logic_error("capture requires the original Parquet binding");
  bound_read_identity identity;
  for (auto const& entry : sirius::planner::registered_connectors())
    if (entry.function_name == function.GetName())
      identity.source = {entry.function_name, entry.kind, entry.registry_profile};
  identity.data_view   = provider_read_identity{probe->binding_key};
  identity.bound_types = bind->types;
  identity.bound_names = duckdb::IdentifiersToStrings(bind->names);
  bound_read_view view;
  view.replay_policy = {function.GetName().GetIdentifierName(),
                        sirius::transparent::byte_source_class::local_file,
                        true,
                        ""};
  view.identity      = make_bound_read_identity(std::move(identity), {});
  probe->captured    = view.identity;
  probe->entries_at_capture_end.push_back(probe->provider.entries);
  probe->preparation_seen_in_capture |= !admission->requests.empty();
  probe->capture_end = ++admission->clock;
  return view;
}

pre_capture_result pre_capture(duckdb::LogicalGet&,
                               duckdb::ClientContext& context,
                               sirius::planner::scan_contract_provenance&)
{
  pre_capture_result result;
  if (capture_for(context)->pre_decline)
    result.decline =
      pre_decline{eligibility_verdict::unsupported, verdict_reason::carrier_missing, declined_text};
  return result;
}

duckdb::unique_ptr<sirius::op::sirius_physical_operator> enter_provider(
  sirius::op::sirius_physical_table_scan& scan,
  sirius::operator_params const& params,
  duckdb::ClientContext& context,
  dynamic_filter_apply_mode mode)
{
  auto probe     = capture_for(context);
  auto admission = probe_for(context);
  auto request   = admission->requests.find(scan.contract_id);
  probe->provider.enter(request == admission->requests.end() ? nullptr : request->second.admission);
  probe->provider_entry = ++admission->clock;
  return sirius::test::connector_admission::lower_parquet(scan, params, context, mode);
}

void configure(connector& entry)
{
  if (entry.function_name != "parquet_scan" && entry.function_name != "read_parquet") return;
  entry.capture        = capture_binding;
  entry.verify_binding = verify_binding;
  entry.decline_reason = pre_capture;
  entry.lower          = enter_provider;
}

struct capture_fixture : admission_fixture {
  duckdb::shared_ptr<capture_probe> capture = duckdb::make_shared_ptr<capture_probe>();

  capture_fixture() : admission_fixture(configure)
  {
    con.context->registered_state->Insert(capture_key, capture);
  }

  std::string query() const { return "SELECT id FROM " + parquet(); }
};

duckdb::LogicalGet* find_get(duckdb::LogicalOperator& op)
{
  if (op.type == duckdb::LogicalOperatorType::LOGICAL_GET) return &op.Cast<duckdb::LogicalGet>();
  for (auto& child : op.children)
    if (auto* result = find_get(*child)) return result;
  return nullptr;
}

auto const& captured_record(capture_fixture const& fixture)
{
  REQUIRE(fixture.read_views);
  REQUIRE(fixture.read_views->entries().size() == 1);
  auto const& record = fixture.read_views->entries().front();
  REQUIRE(record.contract.view);
  REQUIRE(record.contract.view->identity);
  REQUIRE(
    std::holds_alternative<provider_read_identity>(record.contract.view->identity->data_view));
  CHECK(record.contract.view->identity == fixture.capture->captured);
  CHECK(std::get<provider_read_identity>(record.contract.view->identity->data_view).key ==
        "test.capture.k1");
  CHECK(record.contract.view->identity->bound_types ==
        duckdb::vector<duckdb::LogicalType>{duckdb::LogicalType::INTEGER});
  CHECK(record.contract.view->identity->bound_names == duckdb::vector<std::string>{"id"});
  auto candidates = fixture.read_views->candidate_bindings();
  REQUIRE(candidates.size() == 1);
  CHECK(candidates.front().view == record.contract.view);
  return record;
}
}  // namespace

TEST_CASE("Normal GPU planning captures the provider identity once",
          "[planner][scan_preparation][connector_capture][isolated_context]")
{
  capture_fixture fixture;
  fixture.attempt(fixture.query());
  CHECK(fixture.capture->captures == 1);
  auto const& record = captured_record(fixture);
  CHECK(record.eligibility.verdict == eligibility_verdict::supported);
  REQUIRE(fixture.probe->requests.size() == 1);
  CHECK(fixture.probe->requests.count(record.contract.contract_id) == 1);
  CHECK(fixture.probe->requests.begin()->second.lowerings == 1);
}

TEST_CASE("Pre-declined GPU scans retain the connector capture",
          "[planner][scan_preparation][connector_capture][isolated_context]")
{
  capture_fixture fixture;
  fixture.capture->pre_decline = true;
  auto error                   = expect_refusal(fixture, fixture.query());
  CHECK(fixture.capture->captures == 1);
  auto const& record = captured_record(fixture);
  CHECK(record.eligibility.verdict == eligibility_verdict::unsupported);
  CHECK(record.eligibility.reason == verdict_reason::carrier_missing);
  REQUIRE(error.declined.size() == 1);
  CHECK(error.declined.front().first == record.contract.contract_id);
  CHECK(error.declined.front().second == verdict_reason::carrier_missing);
  CHECK(error.message.find(declined_text) != std::string::npos);
  CHECK(fixture.probe->requests.empty());
  CHECK(fixture.probe->observations == 0);
  CHECK(fixture.provider->backing.seen->requests.load() == 0);
  CHECK(fixture.capture->provider.entries == 0);
  CHECK(fixture.stats().scan_lowerings.empty());
}

TEST_CASE("Provider identity changes after capture fail the classified re-lookup",
          "[planner][scan_preparation][connector_capture][isolated_context]")
{
  capture_fixture fixture;
  auto logical = fixture.con.ExtractPlan(fixture.query());
  REQUIRE(logical);
  auto* get = find_get(*logical);
  REQUIRE(get);
  auto table_index   = get->table_index.index;
  auto before_lookup = sirius::planner::lookup_connector_classified(*get, *fixture.con.context);
  REQUIRE(before_lookup.entry);
  REQUIRE(before_lookup.decline == sirius::planner::lookup_decline::none);
  auto original = capture_bound_read_view(*get, *fixture.con.context);
  REQUIRE(original.identity);
  CHECK(fixture.capture->captures == 1);
  fixture.capture->binding_key = "test.capture.k2";
  auto mismatch = sirius::planner::lookup_connector_classified(*get, *fixture.con.context);
  CHECK_FALSE(mismatch.entry);
  CHECK(mismatch.decline == sirius::planner::lookup_decline::bind_data_mismatch);
  auto before = fixture.stats();
  CHECK_THROWS_AS(fixture.attempt(std::move(logical)), duckdb::NotImplementedException);
  REQUIRE(fixture.read_views);
  REQUIRE(fixture.read_views->declined_lookups().count(table_index) == 1);
  CHECK(fixture.read_views->declined_lookups().at(table_index).reason ==
        verdict_reason::bind_data_mismatch);
  auto reason = static_cast<size_t>(verdict_reason::bind_data_mismatch);
  CHECK(fixture.stats().semantic_declines[reason] == before.semantic_declines[reason] + 1);
  CHECK(fixture.capture->captures == 1);
  CHECK(std::get<provider_read_identity>(original.identity->data_view).key == "test.capture.k1");
  CHECK(fixture.probe->requests.empty());
  CHECK(fixture.capture->provider.entries == 0);
  CHECK(fixture.stats().scan_lowerings.empty());
}

TEST_CASE("The provider entry callback requires granted connector admission",
          "[planner][scan_preparation][connector_capture][isolated_context]")
{
  auto granted = GENERATE(true, false);
  capture_fixture fixture;
  fixture.provider->backing.return_null = !granted;
  if (granted) {
    fixture.attempt(fixture.query());
  } else {
    auto error = expect_refusal(fixture, fixture.query());
    check_refusal(fixture, error, "HOST admission refused");
  }
  REQUIRE(fixture.probe->requests.size() == 1);
  auto const& request = fixture.probe->requests.begin()->second;
  CHECK(fixture.capture->captures == 1);
  CHECK(fixture.capture->provider.entries == (granted ? 1 : 0));
  CHECK(fixture.capture->provider.saw_admission == granted);
  CHECK(request.lowerings == (granted ? 1 : 0));
  CHECK(request.rollbacks == (granted ? 0 : 1));
  if (granted) CHECK(request.bind_order < fixture.capture->provider_entry);
}

TEST_CASE("Connector capture completes before admission and provider entry",
          "[planner][scan_preparation][connector_capture][isolated_context]")
{
  capture_fixture fixture;
  fixture.attempt(fixture.query());
  CHECK(fixture.capture->entries_at_capture_begin == std::vector<unsigned>{0});
  CHECK(fixture.capture->entries_at_capture_end == std::vector<unsigned>{0});
  CHECK_FALSE(fixture.capture->preparation_seen_in_capture);
  REQUIRE(fixture.probe->requests.size() == 1);
  auto const& request = fixture.probe->requests.begin()->second;
  CHECK(fixture.capture->capture_begin < fixture.capture->capture_end);
  CHECK(fixture.capture->capture_end < request.bind_order);
  CHECK(request.bind_order < fixture.capture->provider_entry);
  CHECK(fixture.capture->provider.entries == 1);
  CHECK(request.lowerings == 1);
}
