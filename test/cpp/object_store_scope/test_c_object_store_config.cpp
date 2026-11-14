// Copyright 2026 Sirius Contributors
// SPDX-License-Identifier: Apache-2.0
#include "allocation_fault.hpp"
#include "c_api/object_store_config_internal.hpp"
#include "object_store_scope/validation.hpp"
#include "test_helpers.hpp"

#include <catch2/catch_message.hpp>
#include <catch2/catch_test_macros.hpp>
#include <sirius-c/error.h>
#include <sirius-c/object_store/config.h>
#include <sirius-c/object_store/config_builder.h>
#include <unistd.h>

#include <array>
#include <cstdint>
#include <exception>
#include <fstream>
#include <latch>
#include <memory>
#include <string>
#include <string_view>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

namespace core  = sirius::object_store_scope;
namespace fault = sirius::test::allocation_fault;
using sirius::test::object_store_test::operation_deadline;
using sirius::test::object_store_test::temporary_files;
using builder     = sirius_object_store_config_builder;
using config      = sirius_object_store_config;
using builder_ptr = std::unique_ptr<builder, decltype(&sirius_object_store_config_builder_release)>;
using config_ptr  = std::unique_ptr<config, decltype(&sirius_object_store_config_release)>;
using error_ptr   = std::unique_ptr<sirius_error, decltype(&sirius_error_destroy)>;

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
  char const* name;
  std::string core::settings::* member;
  text_derivation derive;
  char const* sentinel;
};

std::array const text_fields{
  text_field{"endpoint",
             &core::settings::endpoint,
             sirius_object_store_config_builder_with_endpoint,
             "C_ENDPOINT_SENTINEL"},
  text_field{"region",
             &core::settings::region,
             sirius_object_store_config_builder_with_region,
             "C_REGION_SENTINEL"},
  text_field{"access_key",
             &core::settings::access_key,
             sirius_object_store_config_builder_with_access_key,
             "C_ACCESS_SENTINEL"},
  text_field{"secret_key",
             &core::settings::secret_key,
             sirius_object_store_config_builder_with_secret_key,
             "C_SECRET_SENTINEL"},
  text_field{"session_token",
             &core::settings::session_token,
             sirius_object_store_config_builder_with_session_token,
             "C_TOKEN_SENTINEL"},
  text_field{"ca_bundle_path",
             &core::settings::ca_bundle_path,
             sirius_object_store_config_builder_with_ca_bundle_path,
             "C_CA_SENTINEL"},
  text_field{"uri_prefix",
             &core::settings::uri_prefix,
             sirius_object_store_config_builder_with_uri_prefix,
             "C_PREFIX_SENTINEL"},
};

template <typename T>
T* poison()
{
  return reinterpret_cast<T*>(std::uintptr_t{1});
}

core::settings empty_settings()
{
  return {.endpoint       = {},
          .region         = {},
          .access_key     = {},
          .secret_key     = {},
          .session_token  = {},
          .transport      = 0,
          .signing_mode   = 0,
          .ca_bundle_path = {},
          .uri_prefix     = {}};
}

core::settings expected_settings()
{
  return {.endpoint       = "https://C_ENDPOINT_SENTINEL.example.invalid:443",
          .region         = "C_REGION_SENTINEL-region",
          .access_key     = "C_ACCESS_SENTINEL-access",
          .secret_key     = "C_SECRET_SENTINEL-secret",
          .session_token  = "C_TOKEN_SENTINEL-session",
          .transport      = 1,
          .signing_mode   = 1,
          .ca_bundle_path = "/missing/C_CA_SENTINEL.pem",
          .uri_prefix     = "s3://bucket/C_PREFIX_SENTINEL/"};
}

core::settings replacement_settings()
{
  return {.endpoint       = "http://replacement.example.invalid:9000",
          .region         = "replacement-region",
          .access_key     = "replacement-access",
          .secret_key     = "replacement-secret",
          .session_token  = "replacement-token",
          .transport      = 2,
          .signing_mode   = 0,
          .ca_bundle_path = "/missing/replacement.pem",
          .uri_prefix     = "s3://other-bucket/replacement/"};
}

void check_fields(core::settings const& actual, core::settings const& expected)
{
  for (auto const& field : text_fields) {
    INFO(field.name);
    CHECK(actual.*field.member == expected.*field.member);
  }
  CHECK(actual.transport == expected.transport);
  CHECK(actual.signing_mode == expected.signing_mode);
}

bool equal_fields(core::settings const& actual, core::settings const& expected)
{
  for (auto const& field : text_fields) {
    if (actual.*field.member != expected.*field.member) { return false; }
  }
  return actual.transport == expected.transport && actual.signing_mode == expected.signing_mode;
}

void replace_field(core::settings& value, std::size_t field, core::settings const& replacement)
{
  if (field < text_fields.size()) {
    value.*text_fields[field].member = replacement.*text_fields[field].member;
  } else if (field == 7) {
    value.transport = replacement.transport;
  } else {
    value.signing_mode = replacement.signing_mode;
  }
}

sirius_status derive(const builder* input,
                     std::size_t field,
                     core::settings const& value,
                     builder** out,
                     sirius_error** error) noexcept
{
  if (field < text_fields.size()) {
    auto const& bytes = value.*text_fields[field].member;
    return text_fields[field].derive(input, bytes.data(), bytes.size(), out, error);
  }
  if (field == 7) {
    return sirius_object_store_config_builder_with_transport(input, value.transport, out, error);
  }
  return sirius_object_store_config_builder_with_signing_mode(
    input, value.signing_mode, out, error);
}

void check_diagnostic(sirius_error* raw,
                      bool required,
                      std::string_view field  = {},
                      std::string_view reason = {})
{
  REQUIRE(raw != poison<sirius_error>());
  error_ptr error(raw, sirius_error_destroy);
  if (required) { REQUIRE(error); }
  std::string_view const message(sirius_error_message(error.get()),
                                 sirius_error_message_size(error.get()));
  for (auto const& item : text_fields) {
    CHECK(message.find(item.sentinel) == std::string_view::npos);
  }
  if (!field.empty()) { CHECK(message.starts_with(field)); }
  if (!reason.empty()) { CHECK(message.find(reason) != std::string_view::npos); }
}

builder_ptr create_builder()
{
  builder* raw      = poison<builder>();
  sirius_error* err = poison<sirius_error>();
  auto const status = sirius_object_store_config_builder_create(&raw, &err);
  builder_ptr result(raw == poison<builder>() ? nullptr : raw,
                     sirius_object_store_config_builder_release);
  check_diagnostic(err, false);
  REQUIRE(status == SIRIUS_SUCCESS);
  REQUIRE(err == nullptr);
  REQUIRE(result);
  return result;
}

builder_ptr derived(const builder* input, std::size_t field, core::settings const& value)
{
  builder* raw      = poison<builder>();
  sirius_error* err = poison<sirius_error>();
  auto const status = derive(input, field, value, &raw, &err);
  builder_ptr result(raw == poison<builder>() ? nullptr : raw,
                     sirius_object_store_config_builder_release);
  check_diagnostic(err, false);
  REQUIRE(status == SIRIUS_SUCCESS);
  REQUIRE(err == nullptr);
  REQUIRE(result);
  return result;
}

builder_ptr make_builder(core::settings const& value)
{
  auto result = create_builder();
  for (std::size_t field = 0; field < 9; ++field) {
    result = derived(result.get(), field, value);
  }
  return result;
}

config_ptr build(const builder* input)
{
  config* raw       = poison<config>();
  sirius_error* err = poison<sirius_error>();
  auto const status = sirius_object_store_config_builder_build(input, &raw, &err);
  config_ptr result(raw == poison<config>() ? nullptr : raw, sirius_object_store_config_release);
  check_diagnostic(err, false);
  REQUIRE(status == SIRIUS_SUCCESS);
  REQUIRE(err == nullptr);
  REQUIRE(result);
  return result;
}

void check_rebuild(const builder* input, core::settings const& expected)
{
  auto result = build(input);
  check_fields(result->value, expected);
  check_fields(input->value, expected);
}

template <typename Call>
void invalid_derivation(const builder* input,
                        core::settings const& expected,
                        std::size_t field,
                        Call&& call,
                        bool has_output = true)
{
  for (bool const diagnostic : {false, true}) {
    builder* out      = poison<builder>();
    sirius_error* err = poison<sirius_error>();
    auto const status = call(&out, diagnostic ? &err : nullptr);
    builder_ptr result(out == poison<builder>() ? nullptr : out,
                       sirius_object_store_config_builder_release);
    CHECK(status == SIRIUS_INVALID_ARGUMENT);
    if (has_output) { CHECK(out == nullptr); }
    if (diagnostic) { check_diagnostic(err, true); }
    check_rebuild(input, expected);
    auto retry = derived(input, field, expected);
    check_rebuild(retry.get(), expected);
  }
}

struct refusal_row {
  std::string core::settings::* member;
  std::string bytes;
  char const* field;
  core::reason reason;
  char const* text;
};

std::vector<refusal_row> refusal_rows()
{
  using core::reason;
  return {
    {&core::settings::endpoint, "", "endpoint", reason::required, "required"},
    {&core::settings::region, "", "region", reason::required, "required"},
    {&core::settings::access_key, "", "access_key", reason::required, "required"},
    {&core::settings::secret_key, "", "secret_key", reason::required, "required"},
    {&core::settings::endpoint,
     "C_ENDPOINT_SENTINEL:9000",
     "endpoint",
     reason::endpoint_missing_scheme,
     "missing scheme (expected http:// or https://)"},
    {&core::settings::endpoint,
     "ftp://C_ENDPOINT_SENTINEL",
     "endpoint",
     reason::endpoint_bad_scheme,
     "scheme must be http or https"},
    {&core::settings::endpoint,
     "https://C_ENDPOINT_SENTINEL host",
     "endpoint",
     reason::endpoint_control_bytes,
     "must not contain control bytes or spaces"},
    {&core::settings::endpoint,
     "https://",
     "endpoint",
     reason::endpoint_missing_host,
     "missing host"},
    {&core::settings::endpoint,
     "https://C_ENDPOINT_SENTINEL/path",
     "endpoint",
     reason::endpoint_has_path,
     "must be scheme://host[:port] with no path, query or fragment"},
    {&core::settings::endpoint,
     "https://C_ENDPOINT_SENTINEL@host",
     "endpoint",
     reason::endpoint_has_user_info,
     "must not contain user-info"},
    {&core::settings::endpoint,
     "https://[::1",
     "endpoint",
     reason::endpoint_unterminated_ipv6,
     "unterminated IPv6 literal"},
    {&core::settings::endpoint,
     "https://[::1]tail",
     "endpoint",
     reason::endpoint_malformed,
     "malformed"},
    {&core::settings::endpoint,
     "https://C_ENDPOINT_SENTINEL:65536",
     "endpoint",
     reason::endpoint_bad_port,
     "port must be a decimal number in 1..65535"},
    {&core::settings::uri_prefix,
     "http://bucket/C_PREFIX_SENTINEL/",
     "uri_prefix",
     reason::prefix_shape,
     "must be s3://bucket/ or s3://bucket/prefix/"},
    {&core::settings::uri_prefix,
     "s3://bucket/C_PREFIX_SENTINEL",
     "uri_prefix",
     reason::prefix_no_trailing_slash,
     "must end with '/'"},
    {&core::settings::uri_prefix,
     "s3://bucket/C_PREFIX_SENTINEL?query/",
     "uri_prefix",
     reason::prefix_query_or_fragment,
     "must not contain a query or fragment"},
    {&core::settings::uri_prefix,
     "s3://bucket/C_PREFIX_SENTINEL/../tail/",
     "uri_prefix",
     reason::prefix_unmatchable_segment,
     "contains a segment no key can match byte-for-byte (empty, '.', '..' or percent-encoded)"},
  };
}

void refused_build(core::settings const& value, std::string_view field, std::string_view reason)
{
  auto input = make_builder(value);
  for (bool const diagnostic : {false, true}) {
    config* out       = poison<config>();
    sirius_error* err = poison<sirius_error>();
    auto const status =
      sirius_object_store_config_builder_build(input.get(), &out, diagnostic ? &err : nullptr);
    config_ptr result(out == poison<config>() ? nullptr : out, sirius_object_store_config_release);
    CHECK(status == SIRIUS_INVALID_CONFIGURATION);
    CHECK(out == nullptr);
    if (diagnostic) { check_diagnostic(err, true, field, reason); }
    check_fields(input->value, value);
  }
}

void overwrite_and_free(core::settings& value)
{
  for (auto const& field : text_fields) {
    auto& bytes = value.*field.member;
    bytes.assign(bytes.size(), '!');
    std::string{}.swap(bytes);
  }
}

}  // namespace

TEST_CASE("C object-store builders start incomplete and clear outputs", "[object_store_c_config]")
{
  auto input = create_builder();
  check_fields(input->value, empty_settings());
  for (bool const diagnostic : {false, true}) {
    sirius_error* err = poison<sirius_error>();
    CHECK(sirius_object_store_config_builder_create(nullptr, diagnostic ? &err : nullptr) ==
          SIRIUS_INVALID_ARGUMENT);
    if (diagnostic) { check_diagnostic(err, true); }
  }
}

TEST_CASE("C object-store text derivations preserve byte spans and other fields",
          "[object_store_c_config]")
{
  auto const original = expected_settings();
  auto input          = make_builder(original);
  for (auto const& field : text_fields) {
    INFO(field.name);
    std::string bytes = "  " + std::string(field.sentinel);
    bytes.push_back('\0');
    bytes.push_back(static_cast<char>(0xff));
    bytes += " tail  ";
    for (unsigned mode = 0; mode < 3; ++mode) {
      auto expected          = original;
      expected.*field.member = mode == 0 ? bytes : "";
      builder* out           = poison<builder>();
      sirius_error* err      = poison<sirius_error>();
      auto const status      = field.derive(
        input.get(), mode == 1 ? nullptr : bytes.data(), mode == 0 ? bytes.size() : 0, &out, &err);
      builder_ptr result(out == poison<builder>() ? nullptr : out,
                         sirius_object_store_config_builder_release);
      check_diagnostic(err, false);
      REQUIRE(status == SIRIUS_SUCCESS);
      REQUIRE(err == nullptr);
      REQUIRE(result);
      check_fields(result->value, expected);
      check_rebuild(input.get(), original);
    }
  }
}

TEST_CASE("C object-store declared selectors keep their full value", "[object_store_c_config]")
{
  auto const original = expected_settings();
  auto input          = make_builder(original);
  for (std::uint32_t value = 0; value <= 2; ++value) {
    auto expected      = original;
    expected.transport = value;
    auto result        = derived(input.get(), 7, expected);
    check_rebuild(result.get(), expected);
    check_rebuild(input.get(), original);
  }
  for (std::uint32_t value = 0; value <= 1; ++value) {
    auto expected         = original;
    expected.signing_mode = value;
    auto result           = derived(input.get(), 8, expected);
    check_rebuild(result.get(), expected);
    check_rebuild(input.get(), original);
  }
}

TEST_CASE("C object-store derivation preserves prior builders and built snapshots",
          "[object_store_c_config]")
{
  auto const original    = expected_settings();
  auto const replacement = replacement_settings();
  for (std::size_t field = 0; field < 9; ++field) {
    INFO("field " << field);
    auto buffers  = original;
    auto supplied = replacement;
    auto expected = original;
    replace_field(expected, field, replacement);
    auto first = make_builder(buffers);
    auto a     = build(first.get());
    auto next  = derived(first.get(), field, supplied);
    overwrite_and_free(buffers);
    overwrite_and_free(supplied);
    auto a2 = build(first.get());
    auto b  = build(next.get());
    first.reset();
    next.reset();
    check_fields(a->value, original);
    check_fields(a2->value, original);
    check_fields(b->value, expected);
  }
}

TEST_CASE("C object-store undeclared selectors fail without narrowing or mutation",
          "[object_store_c_config]")
{
  auto const original = expected_settings();
  auto input          = make_builder(original);
  for (std::size_t field : {std::size_t{7}, std::size_t{8}}) {
    for (std::uint32_t value : {field == 7 ? 3U : 2U, 256U, UINT32_MAX}) {
      INFO("field " << field << ", value " << value);
      auto invalid = original;
      if (field == 7) {
        invalid.transport = value;
      } else {
        invalid.signing_mode = value;
      }
      invalid_derivation(input.get(), original, field, [&](builder** out, sirius_error** err) {
        return derive(input.get(), field, invalid, out, err);
      });
    }
  }
}

TEST_CASE("C object-store build reports every intrinsic field reason", "[object_store_c_config]")
{
  for (auto const& row : refusal_rows()) {
    INFO(row.field << ", reason " << static_cast<unsigned>(row.reason));
    auto value        = expected_settings();
    value.*row.member = row.bytes;
    refused_build(value, row.field, row.text);
  }
  auto const expected = expected_settings();
  auto input          = make_builder(expected);
  check_rebuild(input.get(), expected);
}

TEST_CASE("C object-store build preserves intrinsic field precedence", "[object_store_c_config]")
{
  auto const rows = refusal_rows();
  std::array<std::size_t, 5> const ordered{0, 1, 2, 3, 13};
  for (std::size_t first = 0; first < ordered.size(); ++first) {
    for (std::size_t second = first + 1; second < ordered.size(); ++second) {
      auto const& earlier   = rows[ordered[first]];
      auto const& later     = rows[ordered[second]];
      auto value            = expected_settings();
      value.*earlier.member = earlier.bytes;
      value.*later.member   = later.bytes;
      refused_build(value, earlier.field, earlier.text);
    }
  }
}

TEST_CASE("C object-store build defers CA files and backend support", "[object_store_c_config]")
{
  temporary_files files;
  REQUIRE_FALSE(std::filesystem::exists(files.directory / "missing.pem"));
  REQUIRE(std::filesystem::is_directory(files.directory));
  REQUIRE(std::filesystem::is_regular_file(files.unreadable));
  if (::geteuid() != 0) { REQUIRE_FALSE(std::ifstream(files.unreadable).good()); }
  for (auto const& path : {files.directory / "missing.pem", files.directory, files.unreadable}) {
    auto expected           = expected_settings();
    expected.ca_bundle_path = path.string();
    expected.transport      = 2;
    expected.signing_mode   = 1;
    auto input              = make_builder(expected);
    check_rebuild(input.get(), expected);
  }
}

TEST_CASE("C object-store required pointers fail atomically", "[object_store_c_config]")
{
  auto const original = expected_settings();
  auto input          = make_builder(original);
  for (std::size_t field = 0; field < 9; ++field) {
    INFO("field " << field);
    invalid_derivation(input.get(), original, field, [&](builder** out, sirius_error** err) {
      return derive(nullptr, field, original, out, err);
    });
    invalid_derivation(
      input.get(),
      original,
      field,
      [&](builder**, sirius_error** err) {
        return derive(input.get(), field, original, nullptr, err);
      },
      false);
    if (field < text_fields.size()) {
      invalid_derivation(input.get(), original, field, [&](builder** out, sirius_error** err) {
        return text_fields[field].derive(input.get(), nullptr, 1, out, err);
      });
    }
  }
  for (bool const diagnostic : {false, true}) {
    for (bool const null_input : {false, true}) {
      config* out       = poison<config>();
      sirius_error* err = poison<sirius_error>();
      auto const status =
        sirius_object_store_config_builder_build(null_input ? nullptr : input.get(),
                                                 null_input ? &out : nullptr,
                                                 diagnostic ? &err : nullptr);
      config_ptr result(out == poison<config>() ? nullptr : out,
                        sirius_object_store_config_release);
      CHECK(status == SIRIUS_INVALID_ARGUMENT);
      if (null_input) { CHECK(out == nullptr); }
      if (diagnostic) { check_diagnostic(err, true); }
      check_rebuild(input.get(), original);
    }
  }
}

TEST_CASE("C object-store primary allocation failures preserve inputs", "[object_store_c_config]")
{
  if (!fault::supported()) {
    SKIP("ThreadSanitizer owns the allocation functions; allocation injection is disabled");
  }
  operation_deadline deadline("C object-store primary allocation failures");
  auto const original    = expected_settings();
  auto const replacement = replacement_settings();
  auto input             = make_builder(original);
  for (std::size_t operation = 0; operation < 10; ++operation) {
    INFO("operation " << operation);
    auto expected = original;
    if (operation < 9) { replace_field(expected, operation, replacement); }
    auto attempt = [&](std::size_t ordinal, bool diagnostic) {
      builder* out_builder = poison<builder>();
      config* out_config   = poison<config>();
      sirius_error* err    = poison<sirius_error>();
      fault::arm(ordinal);
      auto const status =
        operation < 9
          ? derive(input.get(), operation, replacement, &out_builder, diagnostic ? &err : nullptr)
          : sirius_object_store_config_builder_build(
              input.get(), &out_config, diagnostic ? &err : nullptr);
      auto const observed = fault::disarm();
      builder_ptr b(out_builder == poison<builder>() ? nullptr : out_builder,
                    sirius_object_store_config_builder_release);
      config_ptr c(out_config == poison<config>() ? nullptr : out_config,
                   sirius_object_store_config_release);
      if (diagnostic) { check_diagnostic(err, false); }
      if (observed.hit) {
        CHECK(status == SIRIUS_ALLOCATION_FAILURE);
        if (operation < 9) {
          CHECK(out_builder == nullptr);
        } else {
          CHECK(out_config == nullptr);
        }
      } else {
        REQUIRE(status == SIRIUS_SUCCESS);
        if (diagnostic) { CHECK(err == nullptr); }
        if (operation < 9) {
          REQUIRE(b);
          check_rebuild(b.get(), expected);
        } else {
          REQUIRE(c);
          check_fields(c->value, expected);
        }
      }
      check_rebuild(input.get(), original);
      return observed;
    };
    auto const counted = attempt(0, false);
    REQUIRE_FALSE(counted.hit);
    REQUIRE(counted.requests > 0);
    for (bool const diagnostic : {false, true}) {
      for (std::size_t ordinal = 1; ordinal <= counted.requests; ++ordinal) {
        INFO("allocation " << ordinal << " of " << counted.requests);
        CHECK(attempt(ordinal, diagnostic).hit);
        CHECK_FALSE(attempt(0, diagnostic).hit);
      }
    }
  }
}

TEST_CASE("C object-store diagnostic allocation failures preserve selected statuses",
          "[object_store_c_config]")
{
  if (!fault::supported()) {
    SKIP("ThreadSanitizer owns the allocation functions; allocation injection is disabled");
  }
  operation_deadline deadline("C object-store diagnostic allocation failures");
  auto incomplete     = create_builder();
  auto const original = expected_settings();
  auto input          = make_builder(original);
  for (unsigned operation = 0; operation < 3; ++operation) {
    INFO("operation " << operation);
    auto const expected_status =
      operation == 0 ? SIRIUS_INVALID_CONFIGURATION : SIRIUS_INVALID_ARGUMENT;
    auto attempt = [&](std::size_t ordinal, bool diagnostic) {
      builder* out_builder = poison<builder>();
      config* out_config   = poison<config>();
      sirius_error* err    = poison<sirius_error>();
      fault::arm(ordinal);
      auto const status =
        operation == 0 ? sirius_object_store_config_builder_build(
                           incomplete.get(), &out_config, diagnostic ? &err : nullptr)
        : operation == 1 ? sirius_object_store_config_builder_with_transport(
                             input.get(), UINT32_MAX, &out_builder, diagnostic ? &err : nullptr)
                         : sirius_object_store_config_builder_with_endpoint(
                             input.get(), nullptr, 1, &out_builder, diagnostic ? &err : nullptr);
      auto const observed = fault::disarm();
      builder_ptr b(out_builder == poison<builder>() ? nullptr : out_builder,
                    sirius_object_store_config_builder_release);
      config_ptr c(out_config == poison<config>() ? nullptr : out_config,
                   sirius_object_store_config_release);
      CHECK(status == expected_status);
      if (operation == 0) {
        CHECK(out_config == nullptr);
      } else {
        CHECK(out_builder == nullptr);
      }
      if (diagnostic) {
        if (observed.hit) {
          CHECK(err == nullptr);
        } else {
          REQUIRE(err != nullptr);
        }
        check_diagnostic(err, !observed.hit);
      }
      check_rebuild(input.get(), original);
      check_fields(incomplete->value, empty_settings());
      return observed;
    };
    auto const without_diagnostic = attempt(0, false);
    REQUIRE(without_diagnostic.requests == 0);
    auto const counted = attempt(0, true);
    REQUIRE(counted.requests > 0);
    for (std::size_t ordinal = 1; ordinal <= counted.requests; ++ordinal) {
      CHECK(attempt(ordinal, true).hit);
    }
    CHECK_FALSE(attempt(0, true).hit);
  }
}

TEST_CASE("C object-store owning references survive source release", "[object_store_c_config]")
{
  sirius_object_store_config_builder_retain(nullptr);
  sirius_object_store_config_builder_release(nullptr);
  sirius_object_store_config_retain(nullptr);
  sirius_object_store_config_release(nullptr);
  auto const expected = expected_settings();
  auto input          = make_builder(expected);
  sirius_object_store_config_builder_retain(input.get());
  builder_ptr alias(input.get(), sirius_object_store_config_builder_release);
  input.reset();
  check_rebuild(alias.get(), expected);
  auto next = derived(alias.get(), 0, expected);
  alias.reset();
  auto result = build(next.get());
  sirius_object_store_config_retain(result.get());
  config_ptr retained(result.get(), sirius_object_store_config_release);
  result.reset();
  next.reset();
  check_fields(retained->value, expected);
}

TEST_CASE("C object-store immutable handles support concurrent derivation and build",
          "[object_store_c_config]")
{
  operation_deadline deadline("C object-store concurrent derivation");
  auto const original    = expected_settings();
  auto const replacement = replacement_settings();
  auto input             = make_builder(original);
  struct worker_result {
    unsigned completed{0};
    std::exception_ptr failure;
  };
  std::array<worker_result, 4> results;
  std::latch start(1);
  std::vector<std::jthread> workers;
  {
    struct release_start {
      std::latch& gate;
      ~release_start() { gate.count_down(); }
    } release{start};
    for (std::size_t worker = 0; worker < results.size(); ++worker) {
      workers.emplace_back([&, worker] {
        start.wait();
        try {
          for (std::size_t iteration = 0; iteration < 18; ++iteration) {
            sirius_object_store_config_builder_retain(input.get());
            builder_ptr source(input.get(), sirius_object_store_config_builder_release);
            auto const field = (iteration + worker) % 9;
            auto expected    = original;
            replace_field(expected, field, replacement);
            builder* raw      = nullptr;
            sirius_error* err = nullptr;
            auto const ds     = derive(source.get(), field, replacement, &raw, &err);
            builder_ptr next(raw, sirius_object_store_config_builder_release);
            error_ptr derivation_error(err, sirius_error_destroy);
            if (ds != SIRIUS_SUCCESS || !next || err) { return; }
            config* a     = nullptr;
            err           = nullptr;
            auto const as = sirius_object_store_config_builder_build(source.get(), &a, &err);
            config_ptr original_config(a, sirius_object_store_config_release);
            error_ptr original_error(err, sirius_error_destroy);
            if (as != SIRIUS_SUCCESS || !original_config || err) { return; }
            config* b     = nullptr;
            err           = nullptr;
            auto const bs = sirius_object_store_config_builder_build(next.get(), &b, &err);
            config_ptr derived_config(b, sirius_object_store_config_release);
            error_ptr derived_error(err, sirius_error_destroy);
            if (bs != SIRIUS_SUCCESS || !derived_config || err) { return; }
            sirius_object_store_config_retain(derived_config.get());
            config_ptr retained(derived_config.get(), sirius_object_store_config_release);
            next.reset();
            derived_config.reset();
            if (!equal_fields(original_config->value, original) ||
                !equal_fields(retained->value, expected) ||
                !equal_fields(source->value, original)) {
              return;
            }
            ++results[worker].completed;
          }
        } catch (...) {
          results[worker].failure = std::current_exception();
        }
      });
    }
  }
  for (auto& worker : workers) {
    worker.join();
  }
  for (auto const& result : results) {
    CHECK_FALSE(result.failure);
    CHECK(result.completed == 18);
  }
  check_rebuild(input.get(), original);
}

TEST_CASE("Object-store intrinsic membership accepts exactly the declared values",
          "[object_store_c_config]")
{
  for (std::uint32_t value = 0; value <= 512; ++value) {
    CHECK(core::is_declared_transport(value) == (value <= 2));
    CHECK(core::is_declared_signing_mode(value) == (value <= 1));
  }
  CHECK_FALSE(core::is_declared_transport(UINT32_MAX));
  CHECK_FALSE(core::is_declared_signing_mode(UINT32_MAX));
}

TEST_CASE("Object-store intrinsic validation shares rules but omits environmental checks",
          "[object_store_c_config]")
{
  for (auto const& row : refusal_rows()) {
    auto value = expected_settings();
    value.ca_bundle_path.clear();
    value.*row.member    = row.bytes;
    auto const intrinsic = core::check_settings_intrinsic(value);
    auto const legacy    = core::check_settings(value, {});
    REQUIRE(intrinsic);
    REQUIRE(legacy);
    CHECK(intrinsic->field == row.field);
    CHECK(intrinsic->why == row.reason);
    CHECK(legacy->field == intrinsic->field);
    CHECK(legacy->why == intrinsic->why);
  }
  for (bool const transport : {false, true}) {
    auto value = expected_settings();
    value.ca_bundle_path.clear();
    if (transport) {
      value.transport = 256;
    } else {
      value.signing_mode = 256;
    }
    auto const intrinsic = core::check_settings_intrinsic(value);
    auto const legacy    = core::check_settings(value, {});
    REQUIRE(intrinsic);
    REQUIRE(legacy);
    CHECK(intrinsic->field == (transport ? "transport" : "signing_mode"));
    CHECK(intrinsic->why ==
          (transport ? core::reason::transport_unknown : core::reason::signing_mode_unknown));
    CHECK(legacy->field == intrinsic->field);
    CHECK(legacy->why == intrinsic->why);
  }
  temporary_files files;
  auto value = expected_settings();
  value.ca_bundle_path.clear();
  CHECK_FALSE(core::check_settings_intrinsic(value));
  CHECK_FALSE(core::check_settings(value, {}));
  auto compare_environment = [&](core::capabilities const& backend, core::reason reason) {
    CHECK_FALSE(core::check_settings_intrinsic(value));
    auto const legacy = core::check_settings(value, backend);
    REQUIRE(legacy);
    CHECK(legacy->why == reason);
  };
  for (auto const& path : {files.directory / "missing.pem", files.directory}) {
    value.ca_bundle_path = path.string();
    compare_environment({}, core::reason::ca_bundle_not_regular_file);
  }
  if (::geteuid() != 0 && !std::ifstream(files.unreadable).good()) {
    value.ca_bundle_path = files.unreadable.string();
    compare_environment({}, core::reason::ca_bundle_unreadable);
  }
  value.ca_bundle_path.clear();
  value.transport = 2;
  compare_environment({}, core::reason::transport_unavailable);
  value.transport = 0;
  compare_environment({false, false, true, "synthetic"}, core::reason::signing_mode_unsupported);
  value.signing_mode   = 0;
  value.ca_bundle_path = files.readable.string();
  compare_environment({false, true, false, "synthetic"}, core::reason::ca_bundle_unsupported);
  value.uri_prefix     = "http://invalid/";
  value.ca_bundle_path = (files.directory / "missing.pem").string();
  auto const intrinsic = core::check_settings_intrinsic(value);
  auto const legacy    = core::check_settings(value, {});
  REQUIRE(intrinsic);
  REQUIRE(legacy);
  CHECK(intrinsic->why == core::reason::prefix_shape);
  CHECK(legacy->why == core::reason::ca_bundle_not_regular_file);
}

TEST_CASE("C object-store build treats impossible selector states as internal errors",
          "[object_store_c_config]")
{
  auto const expected = expected_settings();
  for (bool const transport : {false, true}) {
    for (auto const selector : {transport ? 3U : 2U, 256U, UINT32_MAX}) {
      auto invalid = expected;
      if (transport) {
        invalid.transport = selector;
      } else {
        invalid.signing_mode = selector;
      }
      builder_ptr input(new builder{.value = invalid}, sirius_object_store_config_builder_release);
      for (bool const diagnostic : {false, true}) {
        config* out       = poison<config>();
        sirius_error* err = poison<sirius_error>();
        auto const status =
          sirius_object_store_config_builder_build(input.get(), &out, diagnostic ? &err : nullptr);
        config_ptr result(out == poison<config>() ? nullptr : out,
                          sirius_object_store_config_release);
        error_ptr error(err == poison<sirius_error>() ? nullptr : err, sirius_error_destroy);
        CHECK(status == SIRIUS_INTERNAL_ERROR);
        CHECK(out == nullptr);
        if (diagnostic) {
          REQUIRE(err != poison<sirius_error>());
          REQUIRE(error);
          std::string_view const message(sirius_error_message(error.get()),
                                         sirius_error_message_size(error.get()));
          CHECK(message == "Builder holds settings no derivation produces");
        }
        check_fields(input->value, invalid);
        auto repaired = derived(input.get(), transport ? 7 : 8, expected);
        check_rebuild(repaired.get(), expected);
        check_fields(input->value, invalid);
      }
    }
  }
}
