/* Copyright 2026, Sirius Contributors. SPDX-License-Identifier: Apache-2.0 */
#include "op/scan/arrow/arrow_string_offsets.hpp"
#include "op/scan/arrow_materializer.hpp"

#include <catch.hpp>

#include <array>
#include <cstring>
#include <limits>
#include <string>

using namespace sirius::op::scan;

namespace {
struct flat_input {
  ArrowSchema field{}, schema{};
  ArrowArray column{}, array{};
  ArrowSchema* fields[1]{&field};
  ArrowArray* columns[1]{&column};
  void const* buffers[3]{};
  void const* root_buffers[1]{};

  explicit flat_input(char const* format)
  {
    field.format      = format;
    field.flags       = ARROW_FLAG_NULLABLE;
    field.release     = [](ArrowSchema* s) { s->release = nullptr; };
    schema.format     = "+s";
    schema.n_children = 1;
    schema.children   = fields;
    schema.release    = field.release;
    column.n_buffers  = std::string_view(format) == "u" || std::string_view(format) == "U" ? 3 : 2;
    column.buffers    = buffers;
    column.release    = [](ArrowArray* a) { a->release = nullptr; };
    array.n_buffers   = 1;
    array.buffers     = root_buffers;
    array.n_children  = 1;
    array.children    = columns;
    array.release     = column.release;
  }
};
}  // namespace

TEST_CASE("Arrow host profile preserves every flat scalar and timestamp type",
          "[arrow_profile][materialization]")
{
  using T                 = cudf::type_id;
  auto [format, expected] = GENERATE(table<char const*, T>({{"b", T::BOOL8},
                                                            {"c", T::INT8},
                                                            {"s", T::INT16},
                                                            {"i", T::INT32},
                                                            {"l", T::INT64},
                                                            {"C", T::UINT8},
                                                            {"S", T::UINT16},
                                                            {"I", T::UINT32},
                                                            {"L", T::UINT64},
                                                            {"f", T::FLOAT32},
                                                            {"g", T::FLOAT64},
                                                            {"tdD", T::TIMESTAMP_DAYS},
                                                            {"tss:", T::TIMESTAMP_SECONDS},
                                                            {"tsm:", T::TIMESTAMP_MILLISECONDS},
                                                            {"tsu:", T::TIMESTAMP_MICROSECONDS},
                                                            {"tsn:", T::TIMESTAMP_NANOSECONDS},
                                                            {"u", T::STRING},
                                                            {"U", T::STRING}}));
  CAPTURE(format);
  flat_input input(format);
  auto types = validate_arrow_host_flat(input.schema, input.array);
  REQUIRE(types.size() == 1);
  CHECK(types.front() == cudf::data_type{expected});
}

TEST_CASE("Arrow host profile rejects timezone and unsupported storage formats",
          "[arrow_profile][materialization]")
{
  auto format = GENERATE("tss:UTC",
                         "tsm:UTC",
                         "tsu:UTC",
                         "tsn:UTC",
                         "tsu",
                         "tdm",
                         "z",
                         "Z",
                         "vu",
                         "vz",
                         "+l",
                         "+s",
                         "+m",
                         "+us:0",
                         "+r");
  CAPTURE(format);
  flat_input input(format);
  CHECK_THROWS_AS(validate_arrow_host_flat(input.schema, input.array), std::invalid_argument);
}

TEST_CASE("Arrow decimal scale is preserved independently of precision",
          "[arrow_profile][materialization]")
{
  auto [format, scale] = GENERATE(table<char const*, int32_t>(
    {{"d:9,2", -2}, {"d:9,-3", 3}, {"d:9,10", -10}, {"d:38,0,128", 0}}));
  CAPTURE(format);
  flat_input input(format);
  std::array<__int128_t, 3> values{123, -456, 0};
  input.array.length = input.column.length = values.size();
  input.buffers[1]                         = values.data();
  auto types                               = validate_arrow_host_flat(input.schema, input.array);
  REQUIRE(types.size() == 1);
  CHECK(types.front() == cudf::data_type(cudf::type_id::DECIMAL128, scale));
}

TEST_CASE("Arrow decimal rejects unrepresentable scale and malformed type parameters",
          "[arrow_profile][materialization]")
{
  auto format = GENERATE("d:0,0",
                         "d:39,0",
                         "d:9,-2147483648",
                         "d:9,2147483648",
                         "d:9,2,256",
                         "d:9,2,64",
                         "d:9,2x",
                         "d:9,",
                         "d:,2");
  CAPTURE(format);
  flat_input input(format);
  CHECK_THROWS_AS(validate_arrow_host_flat(input.schema, input.array), std::invalid_argument);
}

TEST_CASE("Arrow string windows check rebased size without allocating their character payload",
          "[arrow_profile][materialization]")
{
  constexpr int64_t max = std::numeric_limits<cudf::size_type>::max();
  CHECK(detail::arrow_string_window_size(0, max) == max);
  CHECK(detail::arrow_string_window_size(max + 17, max + 24) == 7);
  CHECK(detail::arrow_string_window_size(INT64_MAX, INT64_MAX) == 0);
  CHECK_THROWS_AS(detail::arrow_string_window_size(0, max + 1), std::invalid_argument);
  CHECK_THROWS_AS(detail::arrow_string_window_size(1, 0), std::invalid_argument);
  CHECK_THROWS_AS(detail::arrow_string_window_size(-1, 0), std::invalid_argument);
  CHECK_THROWS_AS(detail::arrow_string_window_size(0, INT64_MAX), std::invalid_argument);

  // Real buffers, including an unaligned 64-bit offset array: no character read.
  std::array<int64_t, 2> offsets{max + 17, max + 24};
  std::array<std::byte, sizeof(offsets) + 1> unaligned{};
  std::memcpy(unaligned.data() + 1, offsets.data(), sizeof(offsets));
  detail::arrow_string_offsets source("U", unaligned.data() + 1);
  CHECK(source.width() == sizeof(int64_t));
  CHECK(source[0] == offsets[0]);
  CHECK(source[1] == offsets[1]);
  CHECK_THROWS_AS(source[-1], std::invalid_argument);
  CHECK_THROWS_AS(source[INT64_MAX], std::invalid_argument);
}

TEST_CASE("Arrow large UTF8 validates the selected window and estimates destination storage",
          "[arrow_profile][materialization]")
{
  auto large = GENERATE(false, true);
  auto input = std::make_shared<flat_input>(large ? "U" : "u");
  std::array<int32_t, 7> narrow{-1, -1, 0, 1, 3, 6, -1};
  std::array<int64_t, 7> wide{-1, -1, 0, 1, 3, 6, -1};
  std::array<char, 6> chars{'a', 'b', 'c', 'd', 'e', 'f'};
  uint8_t validity    = 0b00010100;
  input->array.offset = input->column.offset = 1;
  input->array.length                        = 3;
  input->column.length                       = 4;
  input->column.null_count                   = -1;
  input->buffers[0]                          = &validity;
  input->buffers[1] = large ? static_cast<void const*>(wide.data()) : narrow.data();
  input->buffers[2] = chars.data();
  REQUIRE_NOTHROW(validate_arrow_host_flat(input->schema, input->array));
  auto owned = std::make_shared<owned_arrow_batch>(
    input->schema, input->array, input, *std::pmr::new_delete_resource());
  // Identical INT32 destination offsets, chars and conservative mask allowance.
  CHECK(owned->estimated_device_bytes() == 4 * sizeof(int32_t) + chars.size() + 64);
}

TEST_CASE("Arrow large UTF8 rejects invalid destination windows before payload access",
          "[arrow_profile][materialization]")
{
  auto last = GENERATE(int64_t{-1}, int64_t{std::numeric_limits<int32_t>::max()} + 1, INT64_MAX);
  flat_input input("U");
  std::array<int64_t, 2> offsets{0, last};
  input.array.length = input.column.length = 1;
  input.buffers[1]                         = offsets.data();
  CHECK_THROWS_AS(validate_arrow_host_flat(input.schema, input.array), std::invalid_argument);
}

namespace {
struct described_input {
  flat_input input{"i"};
  std::array<int32_t, 8> values{};
  uint8_t validity = 0xff;
  arrow_source_profile profile;
  arrow_bound_schema bound{{{"value", "i", cudf::data_type{cudf::type_id::INT32}, true}}};
  described_input()
  {
    input.field.name    = "value";
    input.array.length  = 3;
    input.column.offset = input.array.offset = 1;
    input.column.length                      = 4;
    input.buffers[0]                         = &validity;
    input.buffers[1]                         = values.data();
    profile.completed                        = true;
    profile.leases                           = std::make_shared<int>(0);
    profile.control_bytes                    = sizeof(*this);
    profile.allocations = {{1, values.data(), sizeof(values)}, {2, &validity, 1}};
    profile.buffers     = {{{arrow_buffer_use{2, 0, 1}}, {arrow_buffer_use{1, 0, sizeof(values)}}}};
    profile.strings.resize(1);
  }
  auto validate() { return validate_arrow_source(input.schema, input.array, profile, bound); }
};
}  // namespace

TEST_CASE("Arrow source profiles validate storage names nullability and the first schema",
          "[arrow_profile][materialization]")
{
  auto mode = GENERATE(0, 1, 2, 3, 4);
  described_input fixture;
  if (mode == 1) fixture.input.field.name = "another";
  if (mode == 2) fixture.input.field.flags = 0;
  if (mode == 3) fixture.input.field.format = "f";
  if (mode == 4) fixture.bound.columns[0].format = "I";
  if (mode)
    CHECK_THROWS_AS(fixture.validate(), std::invalid_argument);
  else {
    auto bytes = fixture.validate();
    CHECK(bytes.retained == sizeof(fixture.values) + 1);
    CHECK(bytes.staging > 0);
    CHECK(bytes.destination >= 3 * sizeof(int32_t));
    CHECK(bytes.total() > bytes.retained);
  }
}

TEST_CASE("Arrow source profile refuses missing evidence and impossible buffer extents",
          "[arrow_profile][materialization]")
{
  auto mode = GENERATE(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12);
  described_input fixture;
  if (mode == 0) fixture.profile.leases.reset();
  if (mode == 1) fixture.profile.completed = false;
  if (mode == 2) fixture.profile.buffers[0][1].reset();
  if (mode == 3) fixture.profile.buffers[0][1]->bytes = 4 * sizeof(int32_t);
  if (mode == 4) fixture.profile.buffers[0][1]->offset = UINT64_MAX;
  if (mode == 5) fixture.profile.buffers[0][1]->allocation = 3;
  if (mode == 6) fixture.profile.allocations.push_back(fixture.profile.allocations.front());
  if (mode == 7) fixture.profile.allocations.push_back({3, fixture.values.data() + 1, 4});
  if (mode == 8) fixture.profile.allocations[0].bytes = UINT64_MAX;
  if (mode == 9) fixture.profile.buffers[0][0]->bytes = 0;
  if (mode == 10) fixture.profile.device = 0;
  if (mode == 11) fixture.profile.memory = arrow_source_memory::device;
  if (mode == 12) fixture.profile.control_bytes = 0;
  CHECK_THROWS_AS(fixture.validate(), std::invalid_argument);
}

TEST_CASE("Arrow source profile separates allocation offsets from logical slice offsets",
          "[arrow_profile][materialization]")
{
  described_input fixture;
  fixture.input.buffers[1]      = fixture.values.data() + 2;
  fixture.profile.buffers[0][1] = arrow_buffer_use{1, 2 * sizeof(int32_t), 6 * sizeof(int32_t)};
  REQUIRE_NOTHROW(fixture.validate());
  fixture.profile.buffers[0][1]->bytes = 4 * sizeof(int32_t);
  CHECK_THROWS_AS(fixture.validate(), std::invalid_argument);
}

TEST_CASE("Arrow control envelope counts retained source storage and spare descriptor capacity",
          "[arrow_profile][materialization]")
{
  described_input fixture;
  auto before = fixture.validate().control;
  fixture.profile.control_bytes += 1024;
  CHECK(fixture.validate().control == before + 1024);
  fixture.profile.allocations.reserve(100);
  CHECK(fixture.validate().control >= before + 1024 + 98 * sizeof(arrow_allocation));
}

TEST_CASE("Arrow device profile sizing never reads a device payload on the host",
          "[arrow_profile][materialization]")
{
  described_input fixture;
  // Metadata-only seam: these bytes deliberately contain no valid decimal payload.
  // No CUDA placement/conversion claim is made by this CPU test.
  fixture.profile.memory          = arrow_source_memory::device;
  fixture.profile.device          = 0;
  fixture.validity                = 0;
  fixture.input.column.null_count = 0;
  auto estimate                   = fixture.validate();
  CHECK(estimate.staging == 0);
  CHECK(estimate.retained == sizeof(fixture.values) + 1);
}

TEST_CASE("Arrow metadata keeps source offsets wide and leaves row limits to admission",
          "[arrow_profile][materialization]")
{
  auto offset = GENERATE(int64_t{0}, int64_t{INT32_MAX} + 1);
  described_input fixture;
  fixture.profile.memory          = arrow_source_memory::device;
  fixture.profile.device          = 0;
  fixture.input.array.offset      = offset;
  fixture.input.array.length      = offset ? 1 : int64_t{INT32_MAX} + 1;
  fixture.input.column.offset     = 0;
  fixture.input.column.length     = fixture.input.array.offset + fixture.input.array.length;
  fixture.input.column.null_count = 0;
  fixture.input.buffers[0]        = nullptr;
  // Allocation metadata seam: intentionally never materialize this virtual address.
  auto base                   = reinterpret_cast<void const*>(uintptr_t{0x100000});
  auto bytes                  = static_cast<uint64_t>(fixture.input.column.length) * 4;
  fixture.input.buffers[1]    = base;
  fixture.profile.allocations = {{1, base, bytes}};
  fixture.profile.buffers     = {{{}, arrow_buffer_use{1, 0, bytes}}};
  auto estimate               = fixture.validate();
  CHECK(estimate.retained == bytes);
  CHECK(estimate.destination >= static_cast<uint64_t>(fixture.input.array.length) * 4);
}
