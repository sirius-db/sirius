// Copyright 2026 Sirius Contributors
// SPDX-License-Identifier: Apache-2.0
#include "c_api/object_store_config_internal.hpp"
#include "test_helpers.hpp"

#include <catch2/catch_message.hpp>
#include <catch2/catch_test_macros.hpp>
#include <sirius-c/error.h>
#include <sirius-c/object_store/config.h>
#include <sirius-c/object_store/config_builder.h>
#include <unistd.h>

#include <array>
#include <cstdint>
#include <fstream>
#include <future>
#include <memory>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

namespace {

using sirius::object_store_scope::settings;
using sirius::test::object_store_test::temporary_files;
using builder     = sirius_object_store_config_builder;
using config      = sirius_object_store_config;
using builder_ptr = std::unique_ptr<builder, decltype(&sirius_object_store_config_builder_release)>;
using config_ptr  = std::unique_ptr<config, decltype(&sirius_object_store_config_release)>;
using error_ptr   = std::unique_ptr<sirius_error, decltype(&sirius_error_destroy)>;
constexpr auto release_builder = sirius_object_store_config_builder_release;
constexpr auto release_config  = sirius_object_store_config_release;

static_assert(std::is_same_v<sirius_object_store_transport, std::uint32_t>);
static_assert(std::is_same_v<sirius_object_store_signing_mode, std::uint32_t>);
static_assert(SIRIUS_OBJECT_STORE_TRANSPORT_AUTO == 0);
static_assert(SIRIUS_OBJECT_STORE_TRANSPORT_HTTP == 1);
static_assert(SIRIUS_OBJECT_STORE_TRANSPORT_RDMA == 2);
static_assert(SIRIUS_OBJECT_STORE_SIGNING_PRESIGNED == 0);
static_assert(SIRIUS_OBJECT_STORE_SIGNING_HEADER == 1);

using text_derivation =
  sirius_status (*)(const builder*, const char*, std::size_t, builder**, sirius_error**) noexcept;

struct text_field {
  std::string settings::* member;
  text_derivation derive;
  char const* sentinel;
};

std::array<text_field, 6> const text_fields{{
  {&settings::endpoint, sirius_object_store_config_builder_with_endpoint, "C_ENDPOINT_SENTINEL"},
  {&settings::region, sirius_object_store_config_builder_with_region, "C_REGION_SENTINEL"},
  {&settings::access_key, sirius_object_store_config_builder_with_access_key, "C_ACCESS_SENTINEL"},
  {&settings::secret_key, sirius_object_store_config_builder_with_secret_key, "C_SECRET_SENTINEL"},
  {&settings::session_token,
   sirius_object_store_config_builder_with_session_token,
   "C_TOKEN_SENTINEL"},
  {&settings::ca_bundle_path,
   sirius_object_store_config_builder_with_ca_bundle_path,
   "C_CA_SENTINEL"},
}};

template <typename T>
T* poison()
{
  return reinterpret_cast<T*>(std::uintptr_t{1});
}

settings const original{.endpoint       = "https://C_ENDPOINT_SENTINEL.example.invalid:443",
                        .region         = "C_REGION_SENTINEL-region",
                        .access_key     = "C_ACCESS_SENTINEL-access",
                        .secret_key     = "C_SECRET_SENTINEL-secret",
                        .session_token  = "C_TOKEN_SENTINEL-session",
                        .transport      = 1,
                        .signing_mode   = 1,
                        .ca_bundle_path = "/missing/C_CA_SENTINEL.pem",
                        .uri_prefix     = ""};

settings const replacement{.endpoint       = "http://replacement.example.invalid:9000",
                           .region         = "replacement-region",
                           .access_key     = "replacement-access",
                           .secret_key     = "replacement-secret",
                           .session_token  = "replacement-token",
                           .transport      = 2,
                           .signing_mode   = 0,
                           .ca_bundle_path = "/missing/replacement.pem",
                           .uri_prefix     = ""};

bool equal_fields(settings const& actual, settings const& expected)
{
  for (auto const& field : text_fields) {
    if (actual.*field.member != expected.*field.member) { return false; }
  }
  return actual.transport == expected.transport && actual.signing_mode == expected.signing_mode &&
         actual.uri_prefix == expected.uri_prefix;
}

void replace_field(settings& value, std::size_t field, settings const& replacement)
{
  if (field < text_fields.size()) {
    value.*text_fields[field].member = replacement.*text_fields[field].member;
  } else {
    (field == text_fields.size() ? value.transport : value.signing_mode) =
      field == text_fields.size() ? replacement.transport : replacement.signing_mode;
  }
}

sirius_status derive(const builder* input,
                     std::size_t field,
                     settings const& value,
                     builder** out,
                     sirius_error** error) noexcept
{
  if (field < text_fields.size()) {
    auto const& bytes = value.*text_fields[field].member;
    return text_fields[field].derive(input, bytes.data(), bytes.size(), out, error);
  }
  if (field == text_fields.size()) {
    return sirius_object_store_config_builder_with_transport(input, value.transport, out, error);
  }
  return sirius_object_store_config_builder_with_signing_mode(
    input, value.signing_mode, out, error);
}

template <typename T, typename Call>
auto checked_call(Call&& call,
                  void (*release)(T*) noexcept,
                  sirius_status wanted     = SIRIUS_SUCCESS,
                  bool diagnostic          = true,
                  bool has_output          = true,
                  std::string_view message = {})
{
  T* raw            = poison<T>();
  sirius_error* err = poison<sirius_error>();
  auto const status = call(&raw, diagnostic ? &err : nullptr);
  std::unique_ptr<T, decltype(release)> result(raw == poison<T>() ? nullptr : raw, release);
  error_ptr error(err == poison<sirius_error>() ? nullptr : err, sirius_error_destroy);
  if (wanted == SIRIUS_SUCCESS) {
    REQUIRE((status == wanted && err == nullptr && result));
  } else {
    CHECK(status == wanted);
    if (has_output) { CHECK(raw == nullptr); }
    if (diagnostic) {
      REQUIRE(error);
      std::string_view const text(sirius_error_message(error.get()),
                                  sirius_error_message_size(error.get()));
      for (auto const& field : text_fields) {
        CHECK(text.find(field.sentinel) == std::string_view::npos);
      }
      if (!message.empty()) { CHECK(text == message); }
    }
  }
  return result;
}

builder_ptr derived(const builder* input, std::size_t field, settings const& value)
{
  return checked_call([&](auto out, auto err) { return derive(input, field, value, out, err); },
                      release_builder);
}

builder_ptr make_builder(settings const& value)
{
  auto result = checked_call(sirius_object_store_config_builder_create, release_builder);
  for (std::size_t field = 0; field < text_fields.size() + 2; ++field) {
    result = derived(result.get(), field, value);
  }
  return result;
}

config_ptr build(const builder* input)
{
  return checked_call(
    [&](auto out, auto err) { return sirius_object_store_config_builder_build(input, out, err); },
    release_config);
}

void check_rebuild(const builder* input, settings const& expected)
{
  CHECK((equal_fields(build(input)->value, expected) && equal_fields(input->value, expected)));
}

template <typename Call>
void invalid_derivation(const builder* input,
                        std::size_t field,
                        Call&& call,
                        bool has_output = true)
{
  for (bool const diagnostic : {false, true}) {
    checked_call(call, release_builder, SIRIUS_INVALID_ARGUMENT, diagnostic, has_output);
    check_rebuild(input, original);
    check_rebuild(derived(input, field, original).get(), original);
  }
}

}  // namespace

TEST_CASE("C object-store builders start incomplete and clear outputs", "[object_store_c_config]")
{
  auto input    = checked_call(sirius_object_store_config_builder_create, release_builder);
  auto expected = original;
  for (auto const& field : text_fields) {
    (expected.*field.member).clear();
  }
  expected.transport = expected.signing_mode = 0;
  CHECK(equal_fields(input->value, expected));
  for (bool const diagnostic : {false, true}) {
    checked_call(
      [](auto, auto err) { return sirius_object_store_config_builder_create(nullptr, err); },
      release_builder,
      SIRIUS_INVALID_ARGUMENT,
      diagnostic,
      false);
  }
}

TEST_CASE("C object-store text derivations preserve byte spans and other fields",
          "[object_store_c_config]")
{
  auto input = make_builder(original);
  for (auto const& field : text_fields) {
    INFO(field.sentinel);
    std::string bytes = "  " + std::string(field.sentinel) + std::string("\xff tail  ", 8);
    for (unsigned mode = 0; mode < 3; ++mode) {
      auto expected          = original;
      expected.*field.member = mode == 0 ? bytes : "";
      auto result            = checked_call(
        [&](auto out, auto err) {
          return field.derive(input.get(),
                              mode == 1 ? nullptr : bytes.data(),
                              mode == 0 ? bytes.size() : 0,
                              out,
                              err);
        },
        release_builder);
      CHECK(equal_fields(result->value, expected));
      check_rebuild(input.get(), original);
    }
  }
}

TEST_CASE("C object-store text derivations reject embedded NUL bytes", "[object_store_c_config]")
{
  auto input = make_builder(original);
  for (std::size_t field = 0; field < text_fields.size(); ++field) {
    auto const& item = text_fields[field];
    for (unsigned position = 0; position < 3; ++position) {
      DYNAMIC_SECTION(item.sentinel << " NUL position " << position)
      {
        std::string bytes = std::string(item.sentinel) + "::" + item.sentinel;
        auto const offset = position == 0 ? 0 : position == 1 ? bytes.size() / 2 : bytes.size();
        bytes.insert(offset, 1, '\0');
        invalid_derivation(input.get(), field, [&](auto out, auto err) {
          return item.derive(input.get(), bytes.data(), bytes.size(), out, err);
        });
      }
    }
  }
}

TEST_CASE("C object-store text checks follow output-slot clearing", "[object_store_c_config]")
{
  auto input = make_builder(original);
  for (auto const& field : text_fields) {
    INFO(field.sentinel);
    for (bool const error_slot : {false, true}) {
      INFO("text in " << (error_slot ? "error slot" : "builder slot"));
      builder* raw            = nullptr;
      sirius_error* raw_error = nullptr;
      auto* text = error_slot ? reinterpret_cast<char*>(&raw_error) : reinterpret_cast<char*>(&raw);
      *text      = 'R';
      // Even a forbidden overlap must not store bytes that were never checked.
      auto const status = field.derive(input.get(), text, 1, &raw, &raw_error);
      builder_ptr result(raw, release_builder);
      error_ptr error(raw_error, sirius_error_destroy);
      CHECK(status == SIRIUS_INVALID_ARGUMENT);
      CHECK(raw == nullptr);
      if (result) { CHECK((result->value.*field.member != std::string(1, '\0'))); }
      check_rebuild(input.get(), original);
    }
  }
}

TEST_CASE("C object-store declared selectors keep their full value", "[object_store_c_config]")
{
  auto input = make_builder(original);
  for (std::size_t field : {text_fields.size(), text_fields.size() + 1}) {
    for (std::uint32_t value = 0; value < (field == text_fields.size() ? 3U : 2U); ++value) {
      auto expected                                                              = original;
      (field == text_fields.size() ? expected.transport : expected.signing_mode) = value;
      auto result = derived(input.get(), field, expected);
      check_rebuild(result.get(), expected);
      check_rebuild(input.get(), original);
    }
  }
}

TEST_CASE("C object-store derivation preserves prior builders and built snapshots",
          "[object_store_c_config]")
{
  for (std::size_t field = 0; field < text_fields.size() + 2; ++field) {
    INFO("field " << field);
    auto buffers = original, supplied = replacement, expected = original;
    replace_field(expected, field, replacement);
    auto first = make_builder(buffers);
    auto a     = build(first.get());
    auto next  = derived(first.get(), field, supplied);
    for (auto* value : {&buffers, &supplied}) {
      for (auto const& item : text_fields) {
        auto& bytes = value->*item.member;
        bytes.assign(bytes.size(), '!');
        std::string{}.swap(bytes);
      }
    }
    auto a2 = build(first.get()), b = build(next.get());
    first.reset();
    next.reset();
    CHECK(equal_fields(a->value, original));
    CHECK(equal_fields(a2->value, original));
    CHECK(equal_fields(b->value, expected));
  }
}

TEST_CASE("C object-store undeclared selectors fail without narrowing or mutation",
          "[object_store_c_config]")
{
  auto input = make_builder(original);
  for (std::size_t field : {text_fields.size(), text_fields.size() + 1}) {
    for (std::uint32_t value : {field == text_fields.size() ? 3U : 2U, 256U, UINT32_MAX}) {
      INFO("field " << field << ", value " << value);
      auto invalid                                                             = original;
      (field == text_fields.size() ? invalid.transport : invalid.signing_mode) = value;
      invalid_derivation(input.get(), field, [&](auto out, auto err) {
        return derive(input.get(), field, invalid, out, err);
      });
    }
  }
}

TEST_CASE("C object-store build reports field refusals in order", "[object_store_c_config]")
{
  struct row {
    std::size_t field;
    char const* bytes;
    char const* message;
  };
  std::array const rows{
    row{0, "", "endpoint: required"},
    row{0, "C_ENDPOINT_SENTINEL:9000", "endpoint: missing scheme (expected http:// or https://)"},
    row{1, "", "region: required"},
    row{2, "", "access_key: required"},
    row{3, "", "secret_key: required"},
  };
  for (std::size_t i = 0; i < rows.size(); ++i) {
    for (std::size_t j = i; j < rows.size(); ++j) {
      if (i != j && rows[i].field == rows[j].field) { continue; }
      auto value                               = original;
      value.*text_fields[rows[i].field].member = rows[i].bytes;
      value.*text_fields[rows[j].field].member = rows[j].bytes;
      auto input                               = make_builder(value);
      for (bool const diagnostic : {false, true}) {
        checked_call(
          [&](auto out, auto err) {
            return sirius_object_store_config_builder_build(input.get(), out, err);
          },
          release_config,
          SIRIUS_INVALID_CONFIGURATION,
          diagnostic,
          true,
          rows[i].message);
        CHECK(equal_fields(input->value, value));
      }
    }
  }
  check_rebuild(make_builder(original).get(), original);
}

TEST_CASE("C object-store build defers CA files and backend support", "[object_store_c_config]")
{
  temporary_files files;
  REQUIRE_FALSE(std::filesystem::exists(files.directory / "missing.pem"));
  REQUIRE(std::filesystem::is_directory(files.directory));
  REQUIRE(std::filesystem::is_regular_file(files.unreadable));
  if (::geteuid() != 0) { REQUIRE_FALSE(std::ifstream(files.unreadable).good()); }
  for (auto const& path : {files.directory / "missing.pem", files.directory, files.unreadable}) {
    auto expected           = original;
    expected.ca_bundle_path = path.string();
    expected.transport      = 2;
    expected.signing_mode   = 1;
    check_rebuild(make_builder(expected).get(), expected);
  }
}

TEST_CASE("C object-store required pointers fail atomically", "[object_store_c_config]")
{
  auto input = make_builder(original);
  for (std::size_t field = 0; field < text_fields.size() + 2; ++field) {
    INFO("field " << field);
    for (bool const null_input : {false, true}) {
      invalid_derivation(
        input.get(),
        field,
        [&](auto out, auto err) {
          return derive(
            null_input ? nullptr : input.get(), field, original, null_input ? out : nullptr, err);
        },
        null_input);
    }
    if (field < text_fields.size()) {
      invalid_derivation(input.get(), field, [&](auto out, auto err) {
        return text_fields[field].derive(input.get(), nullptr, 1, out, err);
      });
    }
  }
  for (bool const diagnostic : {false, true}) {
    for (bool const null_input : {false, true}) {
      checked_call(
        [&](auto out, auto err) {
          return sirius_object_store_config_builder_build(
            null_input ? nullptr : input.get(), null_input ? out : nullptr, err);
        },
        release_config,
        SIRIUS_INVALID_ARGUMENT,
        diagnostic,
        null_input);
      check_rebuild(input.get(), original);
    }
  }
}

TEST_CASE("C object-store owning references survive source release", "[object_store_c_config]")
{
  sirius_object_store_config_builder_retain(nullptr);
  sirius_object_store_config_builder_release(nullptr);
  sirius_object_store_config_retain(nullptr);
  sirius_object_store_config_release(nullptr);
  auto input = make_builder(original);
  sirius_object_store_config_builder_retain(input.get());
  builder_ptr alias(input.get(), release_builder);
  input.reset();
  check_rebuild(alias.get(), original);
  auto next = derived(alias.get(), 0, original);
  alias.reset();
  auto result = build(next.get());
  sirius_object_store_config_retain(result.get());
  config_ptr retained(result.get(), release_config);
  result.reset();
  next.reset();
  CHECK(equal_fields(retained->value, original));
}

TEST_CASE("C object-store immutable handles support concurrent derivation and build",
          "[object_store_c_config]")
{
  auto input = make_builder(original);
  std::vector<std::future<bool>> workers;
  workers.reserve(4);
  // On a failed launch, break the promise before future destructors join waiters.
  std::promise<void> start;
  auto ready = start.get_future().share();
  for (std::size_t worker = 0; worker < 4; ++worker) {
    workers.push_back(std::async(std::launch::async, [&, worker, ready] {
      ready.get();
      sirius_object_store_config_builder_retain(input.get());
      builder_ptr source(input.get(), release_builder);
      auto expected = original;
      replace_field(expected, worker, replacement);
      builder* raw  = nullptr;
      auto const ds = derive(source.get(), worker, replacement, &raw, nullptr);
      builder_ptr next(raw, release_builder);
      config *a = nullptr, *b = nullptr;
      auto const as = sirius_object_store_config_builder_build(source.get(), &a, nullptr);
      config_ptr original_config(a, release_config);
      auto const bs = sirius_object_store_config_builder_build(next.get(), &b, nullptr);
      config_ptr derived_config(b, release_config);
      return ds == SIRIUS_SUCCESS && next && as == SIRIUS_SUCCESS && original_config &&
             bs == SIRIUS_SUCCESS && derived_config &&
             equal_fields(original_config->value, original) &&
             equal_fields(derived_config->value, expected) && equal_fields(source->value, original);
    }));
  }
  start.set_value();
  for (auto& worker : workers) {
    CHECK(worker.get());
  }
  check_rebuild(input.get(), original);
}
