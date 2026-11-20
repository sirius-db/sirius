// Copyright 2026, Sirius Contributors. SPDX-License-Identifier: Apache-2.0
#include "c_api/object_store_registry_internal.hpp"
#include "object_store_scope/validation.hpp"
#include "test_helpers.hpp"

#include <catch2/catch_message.hpp>
#include <catch2/catch_test_macros.hpp>
#include <sirius-c/object_store/config_builder.h>
#include <sirius-c/object_store/registry.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <filesystem>
#include <future>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

namespace core     = sirius::object_store_scope;
using registry     = sirius_object_store_registry;
using builder      = sirius_object_store_config_builder;
using config       = sirius_object_store_config;
using registry_ptr = std::unique_ptr<registry, decltype(&sirius_object_store_registry_release)>;

template <class T>
T* marker()
{
  return reinterpret_cast<T*>(std::uintptr_t{0x52});
}

struct text_field {
  std::string core::settings::* member;
  sirius_status (*set)(
    const builder*, const char*, std::size_t, builder**, sirius_error**) noexcept;
};
constexpr std::array fields{
  text_field{&core::settings::endpoint, sirius_object_store_config_builder_with_endpoint},
  text_field{&core::settings::region, sirius_object_store_config_builder_with_region},
  text_field{&core::settings::access_key, sirius_object_store_config_builder_with_access_key},
  text_field{&core::settings::secret_key, sirius_object_store_config_builder_with_secret_key},
  text_field{&core::settings::session_token, sirius_object_store_config_builder_with_session_token},
  text_field{&core::settings::ca_bundle_path,
             sirius_object_store_config_builder_with_ca_bundle_path}};

core::settings values(std::string const& id)
{
  return {.endpoint       = "https://RGC-ENDPOINT-" + id + ".invalid:443",
          .region         = "RGC-REGION-" + id,
          .access_key     = "RGC-ACCESS-" + id,
          .secret_key     = "RGC-SECRET-" + id,
          .session_token  = "RGC-TOKEN-" + id,
          .transport      = SIRIUS_OBJECT_STORE_TRANSPORT_HTTP,
          .signing_mode   = SIRIUS_OBJECT_STORE_SIGNING_HEADER,
          .ca_bundle_path = "/missing/RGC-CA-" + id + ".pem",
          .uri_prefix     = {}};
}

template <class T, class Call>
auto make_handle(void (*release)(T*) noexcept, Call call)
{
  T* raw            = nullptr;
  auto const status = call(&raw);
  std::unique_ptr<T, decltype(release)> owned(raw, release);
  REQUIRE(status == SIRIUS_SUCCESS);
  REQUIRE(owned);
  return owned;
}

auto config_for(core::settings const& s)
{
  auto const release = sirius_object_store_config_builder_release;
  auto b             = make_handle(
    release, [](auto out) { return sirius_object_store_config_builder_create(out, nullptr); });
  for (auto const& field : fields) {
    auto const& text = s.*field.member;
    b                = make_handle(release, [&](auto out) {
      return field.set(b.get(), text.data(), text.size(), out, nullptr);
    });
  }
  b = make_handle(release, [&](auto out) {
    return sirius_object_store_config_builder_with_transport(b.get(), s.transport, out, nullptr);
  });
  b = make_handle(release, [&](auto out) {
    return sirius_object_store_config_builder_with_signing_mode(
      b.get(), s.signing_mode, out, nullptr);
  });
  return make_handle(sirius_object_store_config_release, [&](auto out) {
    return sirius_object_store_config_builder_build(b.get(), out, nullptr);
  });
}

struct call_result {
  registry_ptr value;
  std::string diagnostic;
};

template <class Call>
call_result registry_call(Call call,
                          sirius_status wanted = SIRIUS_SUCCESS,
                          bool diagnostic      = true,
                          bool output          = true)
{
  auto* raw         = marker<registry>();
  auto* err         = marker<sirius_error>();
  auto const status = call(output ? &raw : nullptr, diagnostic ? &err : nullptr);
  registry_ptr value(raw == marker<registry>() ? nullptr : raw,
                     sirius_object_store_registry_release);
  std::unique_ptr<sirius_error, decltype(&sirius_error_destroy)> error(
    err == marker<sirius_error>() ? nullptr : err, sirius_error_destroy);
  std::string message;
  if (error) {
    message.assign(sirius_error_message(error.get()), sirius_error_message_size(error.get()));
    for (auto token : {"RGC-ENDPOINT",
                       "RGC-REGION",
                       "RGC-ACCESS",
                       "RGC-SECRET",
                       "RGC-TOKEN",
                       "RGC-CA",
                       "RGC-PREFIX"}) {
      CHECK(message.find(token) == std::string::npos);
    }
  }
  REQUIRE(status == wanted);
  if (wanted == SIRIUS_SUCCESS) {
    if (diagnostic) { CHECK(err == nullptr); }
    REQUIRE(value);
    REQUIRE(value->stores);
  } else {
    if (output) { CHECK(raw == nullptr); }
    if (diagnostic) { REQUIRE(error); }
  }
  return {std::move(value), std::move(message)};
}

registry_ptr empty_registry(bool diagnostic = true)
{
  return registry_call(sirius_object_store_registry_create, SIRIUS_SUCCESS, diagnostic).value;
}

registry_ptr add(registry const* input,
                 std::string_view prefix,
                 config const* c,
                 bool diagnostic = true)
{
  return registry_call(
           [&](auto out, auto err) {
             return sirius_object_store_registry_with_store(
               input, prefix.data(), prefix.size(), c, out, err);
           },
           SIRIUS_SUCCESS,
           diagnostic)
    .value;
}

void check_entry(std::shared_ptr<const core::entry> const& hit,
                 core::settings const& expected,
                 std::string_view prefix)
{
  REQUIRE(hit);
  REQUIRE(hit->value);
  CHECK_FALSE(hit->owner);
  for (auto const& field : fields) {
    CHECK((*hit->value).*field.member == expected.*field.member);
  }
  CHECK(hit->value->transport == expected.transport);
  CHECK(hit->value->signing_mode == expected.signing_mode);
  CHECK(hit->value->uri_prefix == prefix);
}

auto check_hit(registry const& source,
               std::string_view uri,
               core::settings const& expected,
               std::string_view prefix)
{
  auto hit = core::resolve(*source.stores, uri);
  CHECK_FALSE(hit.refused);
  check_entry(hit.hit, expected, prefix);
  return hit.hit;
}

}  // namespace

TEST_CASE("C registry starts unconfigured and accepts either zero-length default key",
          "[object_store_c_registry]")
{
  auto const expected = values("default");
  auto cfg            = config_for(expected);
  for (bool const diagnostic : {false, true}) {
    auto empty = empty_registry(diagnostic);
    CHECK(empty->stores->empty());
    std::array<std::string, 1> const items{"s3://bucket/file"};
    auto miss = core::resolve(*empty->stores, items.front());
    CHECK_FALSE(miss.hit);
    CHECK_FALSE(miss.refused);
    auto refused = core::classify_remote_reads(*empty->stores, items);
    REQUIRE(refused);
    CHECK(refused->kind == core::refusal_kind::unconfigured);
    CHECK_FALSE(core::classify_remote_reads(*empty->stores, {}));
    for (auto key : {std::string_view{}, std::string_view("not part of the prefix", 0)}) {
      auto fallback = add(empty.get(), key, cfg.get(), diagnostic);
      check_hit(*fallback, items.front(), expected, "");
      CHECK(empty->stores->empty());
    }
  }
}

TEST_CASE("C registry replaces only the equal key and retains independent versions",
          "[object_store_c_registry]")
{
  auto first = values("one"), other = values("two"), last = values("three");
  last.transport    = SIRIUS_OBJECT_STORE_TRANSPORT_RDMA;
  last.signing_mode = SIRIUS_OBJECT_STORE_SIGNING_PRESIGNED;
  auto a            = add(empty_registry().get(), "s3://bucket/A/", config_for(first).get());
  auto both         = add(a.get(), "s3://bucket/a/", config_for(other).get());
  auto newer        = add(both.get(), "s3://bucket/A/", config_for(last).get());
  CHECK(newer->stores->size() == 2);
  check_hit(*newer, "s3://bucket/A/file", last, "s3://bucket/A/");
  check_hit(*newer, "s3://bucket/a/file", other, "s3://bucket/a/");
  CHECK(both->stores->size() == 2);
  check_hit(*both, "s3://bucket/A/file", first, "s3://bucket/A/");
  check_hit(*both, "s3://bucket/a/file", other, "s3://bucket/a/");
  CHECK(a->stores->size() == 1);
  check_hit(*a, "s3://bucket/A/file", first, "s3://bucket/A/");
  CHECK_FALSE(core::resolve(*a->stores, "s3://bucket/a/file").hit);
}

TEST_CASE("C registry chooses the longest prefix in every insertion order",
          "[object_store_c_registry]")
{
  std::array<std::string_view, 3> const keys{"", "s3://bucket/", "s3://bucket/deep/"};
  std::array<std::string_view, 3> const uris{
    "s3://other/file", "s3://bucket/file", "s3://bucket/deep/file"};
  std::array const expected{values("fallback"), values("bucket"), values("deep")};
  std::array const configs{
    config_for(expected[0]), config_for(expected[1]), config_for(expected[2])};
  std::array<unsigned, 3> order{0, 1, 2};
  do {
    auto result = empty_registry();
    for (auto i : order) {
      result = add(result.get(), keys[i], configs[i].get());
    }
    REQUIRE(result->stores->size() == keys.size());
    for (auto i : order) {
      check_hit(*result, uris[i], expected[i], keys[i]);
    }
  } while (std::next_permutation(order.begin(), order.end()));
}

TEST_CASE("C registry refusals clear outputs and preserve the source", "[object_store_c_registry]")
{
  auto original = values("original"), changed = values("changed");
  changed.transport    = SIRIUS_OBJECT_STORE_TRANSPORT_RDMA;
  changed.signing_mode = SIRIUS_OBJECT_STORE_SIGNING_PRESIGNED;
  std::array const configs{config_for(original), config_for(changed)};
  auto source    = add(empty_registry().get(), "s3://keep/", configs[0].get());
  auto unchanged = [&] {
    REQUIRE(source->stores->size() == 1);
    check_hit(*source, "s3://keep/file", original, "s3://keep/");
  };
  for (bool const diagnostic : {false, true}) {
    registry_call(sirius_object_store_registry_create, SIRIUS_INVALID_ARGUMENT, diagnostic, false);
    auto refused = [&](bool have_registry,
                       bool have_config,
                       bool have_output,
                       char const* bytes,
                       std::size_t length,
                       sirius_status status,
                       std::string_view message = {}) {
      std::string previous;
      for (std::size_t i = 0; i < configs.size(); ++i) {
        auto result = registry_call(
          [&](auto out, auto err) {
            return sirius_object_store_registry_with_store(have_registry ? source.get() : nullptr,
                                                           bytes,
                                                           length,
                                                           have_config ? configs[i].get() : nullptr,
                                                           out,
                                                           err);
          },
          status,
          diagnostic,
          have_output);
        if (diagnostic && !message.empty()) { CHECK(result.diagnostic == message); }
        // Valid selector values have no unique sentinel: vary them as well as all six texts.
        if (i != 0) { CHECK(result.diagnostic == previous); }
        previous = std::move(result.diagnostic);
        unchanged();
      }
    };
    constexpr std::string_view bad = "RGC-PREFIX/";
    refused(false, true, true, bad.data(), bad.size(), SIRIUS_INVALID_ARGUMENT);
    refused(true, false, true, bad.data(), bad.size(), SIRIUS_INVALID_ARGUMENT);
    refused(true, true, false, bad.data(), bad.size(), SIRIUS_INVALID_ARGUMENT);
    refused(true, true, true, nullptr, 1, SIRIUS_INVALID_ARGUMENT);
    std::string const prefix = "s3://bucket/RGC-PREFIX/";
    for (auto pos : {std::size_t{0}, prefix.size() / 2, prefix.size()}) {
      auto bytes = prefix;
      bytes.insert(pos, 1, '\0');
      refused(true, true, true, bytes.data(), bytes.size(), SIRIUS_INVALID_ARGUMENT);
    }
    struct row {
      std::string_view key;
      core::reason reason;
    };
    for (auto const& test :
         {row{bad, core::reason::prefix_shape},
          row{"s3://bucket/RGC-PREFIX", core::reason::prefix_no_trailing_slash},
          row{"s3://bucket/?RGC-PREFIX/", core::reason::prefix_query_or_fragment},
          row{"s3://bucket/#RGC-PREFIX/", core::reason::prefix_query_or_fragment},
          row{"s3://bucket/RGC-PREFIX/../", core::reason::prefix_unmatchable_segment}}) {
      auto message = "uri_prefix: " + core::describe({"uri_prefix", test.reason});
      refused(
        true, true, true, test.key.data(), test.key.size(), SIRIUS_INVALID_CONFIGURATION, message);
    }
  }
}

TEST_CASE("C registry reads a prefix only after clearing its output slots",
          "[object_store_c_registry]")
{
  auto expected = values("alias");
  auto cfg      = config_for(expected);
  auto source   = add(empty_registry().get(), "s3://keep/", cfg.get());
  for (bool const in_error : {false, true}) {
    registry_call(
      [&](auto out, auto err) {
        auto bytes =
          in_error ? reinterpret_cast<char const*>(err) : reinterpret_cast<char const*>(out);
        auto end     = bytes + (in_error ? sizeof(*err) : sizeof(*out));
        auto nonzero = std::find_if(bytes, end, [](char c) { return c != 0; });
        REQUIRE(nonzero != end);
        return sirius_object_store_registry_with_store(
          source.get(), nonzero, 1, cfg.get(), out, err);
      },
      SIRIUS_INVALID_ARGUMENT);
    REQUIRE(source->stores->size() == 1);
    check_hit(*source, "s3://keep/file", expected, "s3://keep/");
  }
}

TEST_CASE("C registry owns its configuration and prefix after all caller buffers go away",
          "[object_store_c_registry]")
{
  auto const expected   = values("buffer-lifetime");
  std::string const key = "s3://bucket/RGC-PREFIX-buffer-lifetime/";
  auto result           = [&] {
    auto buffers = expected;
    auto prefix  = key + "not-in-range";
    auto cfg     = config_for(buffers);
    auto derived =
      add(empty_registry().get(), std::string_view(prefix.data(), key.size()), cfg.get());
    cfg.reset();
    for (auto const& field : fields) {
      auto& bytes = buffers.*field.member;
      std::fill(bytes.begin(), bytes.end(), '!');
    }
    std::fill(prefix.begin(), prefix.end(), '!');
    return derived;
  }();
  check_hit(*result, key + "file", expected, key);
}

TEST_CASE("C registry retained references and selected entries outlive their sources",
          "[object_store_c_registry]")
{
  sirius_object_store_registry_retain(nullptr);
  sirius_object_store_registry_release(nullptr);
  auto const first = values("held"), last = values("later");
  auto parent = add(empty_registry().get(), "", config_for(first).get());
  sirius_object_store_registry_retain(parent.get());
  registry_ptr alias(parent.get(), sirius_object_store_registry_release);
  parent.reset();
  auto held    = check_hit(*alias, "s3://bucket/file", first, "");
  auto derived = add(alias.get(), "", config_for(last).get());
  alias.reset();
  check_hit(*derived, "s3://bucket/file", last, "");
  derived.reset();
  check_entry(held, first, "");
}

TEST_CASE("C registry defers environmental CA and RDMA checks", "[object_store_c_registry]")
{
  sirius::test::object_store_test::temporary_files files;
  REQUIRE(std::filesystem::is_directory(files.directory));
  for (bool const rdma : {false, true}) {
    auto expected           = values("deferred");
    expected.ca_bundle_path = rdma ? "" : files.directory.string();
    expected.transport =
      rdma ? SIRIUS_OBJECT_STORE_TRANSPORT_RDMA : SIRIUS_OBJECT_STORE_TRANSPORT_HTTP;
    auto refused = core::check_settings(expected, {.rdma_transport = false});
    REQUIRE(refused);
    CHECK(refused->why ==
          (rdma ? core::reason::transport_unavailable : core::reason::ca_bundle_not_regular_file));
    auto derived = add(empty_registry().get(), "", config_for(expected).get());
    check_hit(*derived, "s3://bucket/file", expected, "");
  }
}

TEST_CASE("C registry derives from separately owned references on concurrent workers",
          "[object_store_c_registry]")
{
  auto const expected = values("worker");
  auto cfg            = config_for(expected);
  auto source         = empty_registry();
  std::array<std::string, 4> const keys{"s3://a/", "s3://b/", "s3://c/", "s3://d/"};
  std::vector<std::future<std::pair<sirius_status, registry_ptr>>> workers;
  workers.reserve(keys.size());
  // If launching a worker throws, destroying start unblocks all earlier futures before they join.
  std::promise<void> start;
  auto go = start.get_future().share();
  for (auto const& key : keys) {
    sirius_object_store_registry_retain(source.get());
    registry_ptr ref(source.get(), sirius_object_store_registry_release);
    workers.push_back(
      std::async(std::launch::async, [ref = std::move(ref), go, key, c = cfg.get()]() mutable {
        go.get();
        registry* raw = nullptr;
        auto status   = sirius_object_store_registry_with_store(
          ref.get(), key.data(), key.size(), c, &raw, nullptr);
        ref.reset();
        return std::make_pair(status, registry_ptr(raw, sirius_object_store_registry_release));
      }));
  }
  start.set_value();
  for (auto& worker : workers) {
    worker.wait();
  }
  for (std::size_t i = 0; i < workers.size(); ++i) {
    auto [status, result] = workers[i].get();
    REQUIRE(status == SIRIUS_SUCCESS);
    REQUIRE(result);
    REQUIRE(result->stores);
    REQUIRE(result->stores->size() == 1);
    check_hit(*result, keys[i] + "file", expected, keys[i]);
  }
  CHECK(source->stores->empty());
}
