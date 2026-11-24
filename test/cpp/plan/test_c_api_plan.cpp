// Copyright 2026, Sirius Contributors. SPDX-License-Identifier: Apache-2.0
#include <catch2/catch_test_macros.hpp>
#include <sirius-c/plan/plan.h>

#include <array>
#include <atomic>
#include <cstdint>
#include <latch>
#include <memory>
#include <string>
#include <thread>
#include <vector>

namespace {
template <class T, auto Drop>
using handle            = std::unique_ptr<T, decltype(Drop)>;
using schema_handle     = handle<sirius_schema, sirius_schema_release>;
using node_handle       = handle<sirius_plan_node, sirius_plan_node_release>;
using expression_handle = handle<sirius_expression, sirius_expression_release>;
using plan_handle       = handle<sirius_plan, sirius_plan_release>;

schema_handle schema()
{
  const sirius_field fields[] = {{"id", 2, SIRIUS_TYPE_INT64, 1},
                                 {"active", 6, SIRIUS_TYPE_BOOLEAN, 0}};
  sirius_schema* out          = nullptr;
  REQUIRE(sirius_schema_create(fields, 2, &out, nullptr) == SIRIUS_SUCCESS);
  return {out, sirius_schema_release};
}
node_handle scan(const sirius_schema* schema)
{
  sirius_plan_node* out = nullptr;
  REQUIRE(sirius_plan_scan_parquet("missing.parquet", 15, schema, &out, nullptr) == SIRIUS_SUCCESS);
  return {out, sirius_plan_node_release};
}
expression_handle column(const sirius_plan_node* scan, size_t index)
{
  sirius_expression* out = nullptr;
  REQUIRE(sirius_expression_column(scan, index, &out, nullptr) == SIRIUS_SUCCESS);
  return {out, sirius_expression_release};
}
expression_handle literal(int64_t value)
{
  sirius_expression* out = nullptr;
  REQUIRE(sirius_expression_literal_int64(value, &out, nullptr) == SIRIUS_SUCCESS);
  return {out, sirius_expression_release};
}
expression_handle truth()
{
  sirius_expression* out = nullptr;
  REQUIRE(sirius_expression_literal_boolean(1, &out, nullptr) == SIRIUS_SUCCESS);
  return {out, sirius_expression_release};
}
expression_handle compare(const sirius_expression* left, const sirius_expression* right)
{
  sirius_expression* out = nullptr;
  REQUIRE(sirius_expression_compare(left, SIRIUS_COMPARISON_EQUAL, right, &out, nullptr) ==
          SIRIUS_SUCCESS);
  return {out, sirius_expression_release};
}
node_handle filter(const sirius_plan_node* input, const sirius_expression* predicate)
{
  sirius_plan_node* out = nullptr;
  REQUIRE(sirius_plan_filter(input, predicate, &out, nullptr) == SIRIUS_SUCCESS);
  return {out, sirius_plan_node_release};
}
plan_handle plan(const sirius_plan_node* root)
{
  sirius_plan* out = nullptr;
  REQUIRE(sirius_plan_from_root(root, &out, nullptr) == SIRIUS_SUCCESS);
  return {out, sirius_plan_release};
}
}  // namespace

TEST_CASE("C plan construction owns schemas and paths without opening sources", "[public_plan]")
{
  char name[] = "id";
  sirius_field field{name, 2, SIRIUS_TYPE_INT64, 1};
  sirius_schema* raw_schema = nullptr;
  REQUIRE(sirius_schema_create(&field, 1, &raw_schema, nullptr) == SIRIUS_SUCCESS);
  schema_handle owned_schema(raw_schema, sirius_schema_release);
  name[0]                    = 'x';
  char path[]                = "absent\xff.parquet";
  sirius_plan_node* raw_scan = nullptr;
  REQUIRE(sirius_plan_scan_parquet(
            path, sizeof(path) - 1, owned_schema.get(), &raw_scan, nullptr) == SIRIUS_SUCCESS);
  node_handle owned_scan(raw_scan, sirius_plan_node_release);
  path[0] = 'x';
  owned_schema.reset();
  auto reference = column(owned_scan.get(), 0);
  auto ten       = literal(10);
  auto predicate = compare(reference.get(), ten.get());
  CHECK(sirius_expression_type(predicate.get()) == SIRIUS_TYPE_BOOLEAN);
  CHECK(sirius_expression_nullable(predicate.get()) == 1);
  auto filtered = filter(owned_scan.get(), predicate.get());
  auto result   = plan(filtered.get());
  owned_scan.reset();
  reference.reset();
  predicate.reset();
  ten.reset();
  filtered.reset();
  auto* root = sirius_plan_root(result.get());
  REQUIRE(root != nullptr);
  CHECK(sirius_plan_node_get_kind(root) == SIRIUS_NODE_FILTER);
  CHECK(sirius_plan_node_get_kind(sirius_plan_node_input(root)) == SIRIUS_NODE_PARQUET_SCAN);
  CHECK(sirius_plan_node_input(sirius_plan_node_input(root)) == nullptr);
  auto* output = sirius_plan_node_schema(root);
  CHECK(sirius_schema_field_count(output) == 1);
  sirius_field observed{};
  REQUIRE(sirius_schema_field(output, 0, &observed, nullptr) == SIRIUS_SUCCESS);
  CHECK(std::string(observed.name, observed.name_size) == "id");
  CHECK(observed.type == SIRIUS_TYPE_INT64);
  CHECK(observed.nullable == 1);
  auto description = std::string(sirius_plan_describe(result.get()));
  CHECK(description.find("absent\\xff.parquet") != std::string::npos);
  CHECK(description.find("Filter(") != std::string::npos);
  CHECK(description.find("Int64?") != std::string::npos);
  // Borrowed handles can become independent owning references.
  sirius_plan_node_retain(root);
  node_handle retained(root, sirius_plan_node_release);
  sirius_schema_retain(output);
  schema_handle retained_schema(output, sirius_schema_release);
  result.reset();
  retained.reset();
  REQUIRE(sirius_schema_field(retained_schema.get(), 0, &observed, nullptr) == SIRIUS_SUCCESS);
  CHECK(std::string(observed.name, observed.name_size) == "id");
}

TEST_CASE("Plan descriptions escape literal path bytes", "[public_plan]")
{
  auto fields            = schema();
  const std::string path = "./quote\"\\line\n\t\xff";
  sirius_plan_node* raw  = nullptr;
  REQUIRE(sirius_plan_scan_parquet(path.data(), path.size(), fields.get(), &raw, nullptr) ==
          SIRIUS_SUCCESS);
  node_handle source(raw, sirius_plan_node_release);
  auto result = plan(source.get());
  CHECK(std::string(sirius_plan_describe(result.get()))
          .find(R"(ParquetScan(path="./quote\"\\line\n\t\xff"))") != std::string::npos);
}

TEST_CASE("Plan descriptions escape UTF-8 names and adjacent hexadecimal digits", "[public_plan]")
{
  const sirius_field field{"\xc3\xa9", 2, SIRIUS_TYPE_INT64, 0};
  sirius_schema* raw_schema = nullptr;
  REQUIRE(sirius_schema_create(&field, 1, &raw_schema, nullptr) == SIRIUS_SUCCESS);
  schema_handle fields(raw_schema, sirius_schema_release);
  const std::string path =
    "./\xff"
    "a";
  sirius_plan_node* raw = nullptr;
  REQUIRE(sirius_plan_scan_parquet(path.data(), path.size(), fields.get(), &raw, nullptr) ==
          SIRIUS_SUCCESS);
  node_handle source(raw, sirius_plan_node_release);
  auto result                   = plan(source.get());
  const std::string description = sirius_plan_describe(result.get());
  CHECK(description.find(R"(path="./\xff\x61")") != std::string::npos);
  CHECK(description.find(R"(Output: "\xc3\xa9": Int64)") != std::string::npos);
}

TEST_CASE("C expressions enforce types and exact input scopes", "[public_plan]")
{
  auto fields            = schema();
  auto first             = scan(fields.get());
  auto second            = scan(fields.get());
  auto id                = column(first.get(), 0);
  auto other_id          = column(second.get(), 0);
  auto active            = column(first.get(), 1);
  auto ten               = literal(10);
  sirius_expression* out = nullptr;
  CHECK(
    sirius_expression_compare(id.get(), SIRIUS_COMPARISON_EQUAL, other_id.get(), &out, nullptr) ==
    SIRIUS_INVALID_ARGUMENT);
  CHECK(out == nullptr);
  CHECK(sirius_expression_compare(id.get(), SIRIUS_COMPARISON_EQUAL, active.get(), &out, nullptr) ==
        SIRIUS_INVALID_ARGUMENT);
  CHECK(
    sirius_expression_compare(active.get(), SIRIUS_COMPARISON_LESS, active.get(), &out, nullptr) ==
    SIRIUS_INVALID_ARGUMENT);
  CHECK(sirius_expression_compare(id.get(), UINT32_MAX, ten.get(), &out, nullptr) ==
        SIRIUS_INVALID_ARGUMENT);
  auto reverse = compare(ten.get(), id.get());
  CHECK(filter(first.get(), reverse.get()) != nullptr);
  REQUIRE(sirius_expression_compare(
            active.get(), SIRIUS_COMPARISON_NOT_EQUAL, active.get(), &out, nullptr) ==
          SIRIUS_SUCCESS);
  expression_handle unequal(out, sirius_expression_release);
  out = nullptr;
  CHECK(sirius_expression_type(unequal.get()) == SIRIUS_TYPE_BOOLEAN);
  CHECK(sirius_expression_nullable(unequal.get()) == 0);
  auto predicate            = compare(id.get(), ten.get());
  sirius_plan_node* invalid = nullptr;
  CHECK(sirius_plan_filter(second.get(), predicate.get(), &invalid, nullptr) ==
        SIRIUS_INVALID_ARGUMENT);
  CHECK(sirius_plan_filter(first.get(), id.get(), &invalid, nullptr) == SIRIUS_INVALID_ARGUMENT);
  CHECK(invalid == nullptr);
  auto filtered = filter(first.get(), predicate.get());
  CHECK(sirius_plan_filter(filtered.get(), predicate.get(), &invalid, nullptr) ==
        SIRIUS_INVALID_ARGUMENT);
  auto next_id        = column(filtered.get(), 0);
  auto next_predicate = compare(next_id.get(), ten.get());
  auto another        = filter(filtered.get(), next_predicate.get());
  CHECK(sirius_plan_node_input(another.get()) == filtered.get());
  CHECK(sirius_plan_node_get_kind(first.get()) == SIRIUS_NODE_PARQUET_SCAN);
  auto constant = truth();
  auto unscoped = compare(constant.get(), constant.get());
  CHECK(sirius_expression_nullable(unscoped.get()) == 0);
  auto constant_plan = plan(filter(second.get(), unscoped.get()).get());
  CHECK(std::string(sirius_plan_describe(constant_plan.get())).find("true") != std::string::npos);
  out = nullptr;
  CHECK(sirius_expression_literal_boolean(2, &out, nullptr) == SIRIUS_INVALID_ARGUMENT);
  CHECK(sirius_expression_column(first.get(), 2, &out, nullptr) == SIRIUS_INVALID_ARGUMENT);
}

TEST_CASE("C schemas reject malformed names and field descriptors", "[public_plan]")
{
  const std::array<std::string, 10> invalid_names{"",
                                                  std::string("a\0b", 3),
                                                  "\xff",
                                                  "\xc0\xaf",
                                                  "\xe0\x80\xaf",
                                                  "\xf0\x80\x80\xaf",
                                                  "\xed\xa0\x80",
                                                  "\xf4\x90\x80\x80",
                                                  "\xe2\x82",
                                                  "\x80"};
  for (const auto& name : invalid_names) {
    sirius_field field{name.data(), name.size(), SIRIUS_TYPE_INT64, 0};
    sirius_schema* out = nullptr;
    CHECK(sirius_schema_create(&field, 1, &out, nullptr) == SIRIUS_INVALID_ARGUMENT);
    CHECK(out == nullptr);
  }
  for (const auto& name :
       {std::string("\xc3\xa9"), std::string("\xe2\x82\xac"), std::string("\xf0\x9f\x98\x80")}) {
    sirius_field field{name.data(), name.size(), SIRIUS_TYPE_BOOLEAN, 0};
    sirius_schema* out = nullptr;
    REQUIRE(sirius_schema_create(&field, 1, &out, nullptr) == SIRIUS_SUCCESS);
    sirius_schema_release(out);
  }
  const sirius_field case_distinct[] = {{"id", 2, SIRIUS_TYPE_INT64, 0},
                                        {"ID", 2, SIRIUS_TYPE_INT64, 0}};
  sirius_schema* case_schema         = nullptr;
  REQUIRE(sirius_schema_create(case_distinct, 2, &case_schema, nullptr) == SIRIUS_SUCCESS);
  sirius_schema_release(case_schema);
  sirius_field fields[] = {{"same", 4, SIRIUS_TYPE_INT64, 0}, {"same", 4, SIRIUS_TYPE_BOOLEAN, 1}};
  sirius_schema* out    = nullptr;
  CHECK(sirius_schema_create(fields, 2, &out, nullptr) == SIRIUS_INVALID_ARGUMENT);
  fields[0].type = UINT32_MAX;
  CHECK(sirius_schema_create(fields, 1, &out, nullptr) == SIRIUS_INVALID_ARGUMENT);
  fields[0].type     = SIRIUS_TYPE_INT64;
  fields[0].nullable = 2;
  CHECK(sirius_schema_create(fields, 1, &out, nullptr) == SIRIUS_INVALID_ARGUMENT);
  CHECK(sirius_schema_create(nullptr, 0, &out, nullptr) == SIRIUS_INVALID_ARGUMENT);
}

TEST_CASE("C plan failures clear outputs and provide optional diagnostics", "[public_plan]")
{
  auto fields         = schema();
  auto source         = scan(fields.get());
  auto ten            = literal(10);
  auto* invalid       = reinterpret_cast<sirius_plan_node*>(uintptr_t{1});
  sirius_error* error = nullptr;
  CHECK(sirius_plan_filter(source.get(), ten.get(), &invalid, &error) == SIRIUS_INVALID_ARGUMENT);
  CHECK(invalid == nullptr);
  REQUIRE(error != nullptr);
  CHECK(sirius_error_message_size(error) != 0);
  sirius_error_destroy(error);
  for (const std::string path : {"",
                                 "s3://bucket/file",
                                 "file:/tmp/data.parquet",
                                 "file:data.parquet",
                                 "s3:bucket/file",
                                 "a+b.c-d:x",
                                 "data:1.parquet"}) {
    CHECK(sirius_plan_scan_parquet(path.data(), path.size(), fields.get(), &invalid, nullptr) ==
          SIRIUS_INVALID_ARGUMENT);
  }
  for (const std::string path : {"./data:1.parquet",
                                 "./name:part",
                                 "1a:x.parquet",
                                 "*.parquet",
                                 "a?.parquet",
                                 "a[b].parquet",
                                 "a].parquet"}) {
    sirius_plan_node* local = nullptr;
    REQUIRE(sirius_plan_scan_parquet(path.data(), path.size(), fields.get(), &local, nullptr) ==
            SIRIUS_SUCCESS);
    auto literal_path = plan(local);
    CHECK(std::string(sirius_plan_describe(literal_path.get())).find(path) != std::string::npos);
    sirius_plan_node_release(local);
  }
  CHECK(sirius_plan_scan_parquet("a\0b", 3, fields.get(), &invalid, nullptr) ==
        SIRIUS_INVALID_ARGUMENT);
  CHECK(sirius_plan_scan_parquet(nullptr, 1, fields.get(), &invalid, nullptr) ==
        SIRIUS_INVALID_ARGUMENT);
  CHECK(sirius_plan_scan_parquet("a", 1, nullptr, &invalid, nullptr) == SIRIUS_INVALID_ARGUMENT);
  CHECK(sirius_plan_scan_parquet("a", 1, fields.get(), nullptr, nullptr) ==
        SIRIUS_INVALID_ARGUMENT);
  sirius_expression* expression = nullptr;
  CHECK(sirius_expression_column(nullptr, 0, &expression, nullptr) == SIRIUS_INVALID_ARGUMENT);
  CHECK(sirius_expression_literal_int64(1, nullptr, nullptr) == SIRIUS_INVALID_ARGUMENT);
  CHECK(
    sirius_expression_compare(ten.get(), SIRIUS_COMPARISON_EQUAL, nullptr, &expression, nullptr) ==
    SIRIUS_INVALID_ARGUMENT);
  sirius_plan* result = reinterpret_cast<sirius_plan*>(uintptr_t{1});
  CHECK(sirius_plan_from_root(nullptr, &result, nullptr) == SIRIUS_INVALID_ARGUMENT);
  CHECK(result == nullptr);
  sirius_field field{"stale", 5, SIRIUS_TYPE_INT64, 1};
  CHECK(sirius_schema_field(fields.get(), 99, &field, nullptr) == SIRIUS_INVALID_ARGUMENT);
  CHECK(field.name == nullptr);
  CHECK(field.name_size == 0);
  CHECK(sirius_schema_field(nullptr, 0, &field, nullptr) == SIRIUS_INVALID_ARGUMENT);
  CHECK(sirius_schema_field(fields.get(), 0, nullptr, nullptr) == SIRIUS_INVALID_ARGUMENT);
}

TEST_CASE("C plan null inspection and release are harmless", "[public_plan]")
{
  CHECK(sirius_schema_field_count(nullptr) == 0);
  CHECK(sirius_expression_type(nullptr) == 0);
  CHECK(sirius_expression_nullable(nullptr) == 0);
  CHECK(sirius_plan_node_get_kind(nullptr) == 0);
  CHECK(sirius_plan_node_input(nullptr) == nullptr);
  CHECK(sirius_plan_node_schema(nullptr) == nullptr);
  CHECK(sirius_plan_root(nullptr) == nullptr);
  CHECK(std::string(sirius_plan_describe(nullptr)).empty());
  sirius_schema_retain(nullptr);
  sirius_schema_release(nullptr);
  sirius_expression_retain(nullptr);
  sirius_expression_release(nullptr);
  sirius_plan_node_retain(nullptr);
  sirius_plan_node_release(nullptr);
  sirius_plan_retain(nullptr);
  sirius_plan_release(nullptr);
}

TEST_CASE("C construction bounds depth and describes shared expressions linearly", "[public_plan]")
{
  auto fields    = schema();
  auto source    = scan(fields.get());
  auto predicate = truth();
  // A DAG with only 256 nodes but exponentially many paths if expanded as a tree.
  for (size_t i = 1; i < 256; ++i) {
    predicate = compare(predicate.get(), predicate.get());
  }
  auto result = plan(filter(source.get(), predicate.get()).get());
  CHECK(std::string(sirius_plan_describe(result.get())).size() < 20000);
  sirius_expression* overflow = nullptr;
  CHECK(sirius_expression_compare(
          predicate.get(), SIRIUS_COMPARISON_EQUAL, predicate.get(), &overflow, nullptr) ==
        SIRIUS_INVALID_ARGUMENT);
  CHECK(overflow == nullptr);
  auto yes = truth();
  for (size_t i = 1; i < 256; ++i) {
    source = filter(source.get(), yes.get());
  }
  sirius_plan_node* node = nullptr;
  CHECK(sirius_plan_filter(source.get(), yes.get(), &node, nullptr) == SIRIUS_INVALID_ARGUMENT);
  CHECK(node == nullptr);
}

TEST_CASE("C immutable plans support concurrent ownership and construction", "[public_plan]")
{
  auto fields                   = schema();
  auto source                   = scan(fields.get());
  auto yes                      = truth();
  auto original                 = plan(source.get());
  const std::string description = sirius_plan_describe(original.get());
  std::atomic<unsigned> failures{0};
  {
    std::vector<std::jthread> threads;
    for (int t = 0; t < 4; ++t) {
      threads.emplace_back([&] {
        for (int i = 0; i < 100; ++i) {
          sirius_schema_retain(fields.get());
          sirius_plan_node_retain(source.get());
          sirius_expression_retain(yes.get());
          sirius_plan_retain(original.get());
          sirius_plan_node* next = nullptr;
          if (sirius_plan_filter(source.get(), yes.get(), &next, nullptr) != SIRIUS_SUCCESS)
            ++failures;
          if (std::string(sirius_plan_describe(original.get())) != description) ++failures;
          sirius_plan_node_release(next);
          sirius_plan_release(original.get());
          sirius_expression_release(yes.get());
          sirius_plan_node_release(source.get());
          sirius_schema_release(fields.get());
        }
      });
    }
  }
  CHECK(failures == 0);
}

TEST_CASE("C final expression release tears down combined-depth ownership", "[public_plan]")
{
  auto fields = schema();
  auto source = scan(fields.get());
  for (size_t node_depth = 1; node_depth < 256; ++node_depth) {
    auto predicate = column(source.get(), 1);
    for (size_t expression_depth = 1; expression_depth < 256; ++expression_depth) {
      predicate = compare(predicate.get(), predicate.get());
    }
    source = filter(source.get(), predicate.get());
  }
  // An external expression becomes the sole owner of the entire input chain.
  auto external = column(source.get(), 1);
  for (size_t depth = 1; depth < 256; ++depth) {
    external = compare(external.get(), external.get());
  }
  const auto* borrowed_root = source.get();
  source.reset();
  fields.reset();
  size_t nodes = 0;
  for (auto* node = borrowed_root; node; node = sirius_plan_node_input(node)) {
    ++nodes;
  }
  CHECK(nodes == 256);
  CHECK(sirius_expression_type(external.get()) == SIRIUS_TYPE_BOOLEAN);
  // ASan/LSan check the nested teardown for invalid accesses and leaked references.
  external.reset();
}

TEST_CASE("C shared handles can lose their final references on worker threads", "[public_plan]")
{
  std::atomic<unsigned> failures{0};
  for (int iteration = 0; iteration < 32; ++iteration) {
    auto fields    = schema();
    auto source    = scan(fields.get());
    auto predicate = column(source.get(), 1);
    auto result    = plan(filter(source.get(), predicate.get()).get());
    std::latch start{1};
    std::vector<std::jthread> threads;
    try {
      for (int worker = 0; worker < 4; ++worker) {
        sirius_schema_retain(fields.get());
        schema_handle owned_fields(fields.get(), sirius_schema_release);
        sirius_plan_node_retain(source.get());
        node_handle owned_source(source.get(), sirius_plan_node_release);
        sirius_expression_retain(predicate.get());
        expression_handle owned_predicate(predicate.get(), sirius_expression_release);
        sirius_plan_retain(result.get());
        plan_handle owned_result(result.get(), sirius_plan_release);
        threads.emplace_back([&,
                              fields    = std::move(owned_fields),
                              source    = std::move(owned_source),
                              predicate = std::move(owned_predicate),
                              result    = std::move(owned_result)]() mutable {
          start.wait();
          if (sirius_schema_field_count(fields.get()) != 2 ||
              sirius_plan_node_get_kind(source.get()) != SIRIUS_NODE_PARQUET_SCAN ||
              sirius_expression_type(predicate.get()) != SIRIUS_TYPE_BOOLEAN ||
              sirius_plan_node_input(sirius_plan_root(result.get())) != source.get()) {
            ++failures;
          }
          result.reset();
          predicate.reset();
          source.reset();
          fields.reset();
        });
      }
    } catch (...) {
      // Release waiting workers before their jthread destructors join them.
      start.count_down();
      throw;
    }
    result.reset();
    predicate.reset();
    source.reset();
    fields.reset();
    start.count_down();
  }
  CHECK(failures == 0);
}
