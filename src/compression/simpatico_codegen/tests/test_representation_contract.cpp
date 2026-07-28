// SPDX-License-Identifier: Apache-2.0
//
// Contract tests for the two-tier compressed_representation hierarchy.
//
//  Tier 1 — compressed_representation: storage/metadata only.
//  Tier 2 — standalone_compressed_representation: independently decodable.
//
// Compile-time static_asserts verify that generic representations sit in the
// standalone tier while codegen_fused_representation stays in the base tier.
// Runtime tests verify that decompress_standalone_representation() reports a
// deterministic error when given a storage-only fused representation.

#include "codegen/plan/representation.hpp"

#include <cudf/column/column.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/mr/per_device_resource.hpp>
#include <rmm/resource_ref.hpp>

#include <cstdio>
#include <stdexcept>
#include <string>
#include <type_traits>

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

namespace {

void expect(bool cond, char const* msg)
{
  if (!cond) throw std::runtime_error(msg);
}

// ---------------------------------------------------------------------------
// Compile-time hierarchy contract
// ---------------------------------------------------------------------------

// All generic representations must be standalone-decodable.
static_assert(std::is_base_of_v<simpatico::standalone_compressed_representation,
                                simpatico::identity_compressed_representation>,
              "identity_compressed_representation must derive from standalone");

static_assert(std::is_base_of_v<simpatico::standalone_compressed_representation,
                                simpatico::dictionary_compressed_representation>,
              "dictionary_compressed_representation must derive from standalone");

static_assert(std::is_base_of_v<simpatico::standalone_compressed_representation,
                                simpatico::str_split_compressed_representation>,
              "str_split_compressed_representation must derive from standalone");

static_assert(
  std::is_base_of_v<simpatico::standalone_compressed_representation, simpatico::nvcomp_payload_rep>,
  "nvcomp_payload_rep must derive from standalone");

static_assert(std::is_base_of_v<simpatico::standalone_compressed_representation,
                                simpatico::alp_compressed_representation>,
              "alp_compressed_representation must derive from standalone");

static_assert(std::is_base_of_v<simpatico::standalone_compressed_representation,
                                simpatico::alp_rd_compressed_representation>,
              "alp_rd_compressed_representation must derive from standalone");

static_assert(std::is_base_of_v<simpatico::standalone_compressed_representation,
                                simpatico::bitextract_compressed_representation>,
              "bitextract_compressed_representation must derive from standalone");

// codegen_fused_representation is storage-only: it must NOT derive from the
// standalone tier.
static_assert(!std::is_base_of_v<simpatico::standalone_compressed_representation,
                                 simpatico::codegen_fused_representation>,
              "codegen_fused_representation must NOT derive from standalone");

// Every representation is still reachable from the base storage type.
static_assert(
  std::is_base_of_v<simpatico::compressed_representation, simpatico::codegen_fused_representation>,
  "codegen_fused_representation must derive from compressed_representation");

static_assert(std::is_base_of_v<simpatico::compressed_representation,
                                simpatico::standalone_compressed_representation>,
              "standalone_compressed_representation must derive from compressed_representation");

// ---------------------------------------------------------------------------
// Runtime: decompress_standalone_representation rejects fused reps
// ---------------------------------------------------------------------------

void test_fused_rep_rejected_by_helper()
{
  // Build a minimal fused rep (no buffers needed; helper checks the dynamic
  // type before even calling decompress).
  simpatico::codegen_fused_representation fused{
    simpatico::OpId::Bitpack, cudf::data_type{cudf::type_id::INT32}, 0};

  std::string err;
  auto result = simpatico::decompress_standalone_representation(
    &fused, rmm::cuda_stream_view{}, rmm::mr::get_current_device_resource_ref(), &err);

  expect(result == nullptr, "helper must return nullptr for a fused rep");
  expect(!err.empty(), "helper must set a non-empty error message for a fused rep");
  expect(err.find("storage-only") != std::string::npos || err.find("PlanTree") != std::string::npos,
         "error message must mention 'storage-only' or 'PlanTree'");
}

void test_null_rep_rejected_by_helper()
{
  std::string err;
  auto result = simpatico::decompress_standalone_representation(
    nullptr, rmm::cuda_stream_view{}, rmm::mr::get_current_device_resource_ref(), &err);

  expect(result == nullptr, "helper must return nullptr for null rep");
  expect(!err.empty(), "helper must set a non-empty error message for null rep");
}

void test_fused_rep_rejected_without_error_out()
{
  simpatico::codegen_fused_representation fused{
    simpatico::OpId::Delta, cudf::data_type{cudf::type_id::INT64}, 0};

  // Passing nullptr for error_out must not crash.
  auto result = simpatico::decompress_standalone_representation(
    &fused, rmm::cuda_stream_view{}, rmm::mr::get_current_device_resource_ref(), nullptr);

  expect(result == nullptr, "helper must return nullptr for a fused rep (no error_out)");
}

}  // namespace

int main()
{
  try {
    test_fused_rep_rejected_by_helper();
    test_null_rep_rejected_by_helper();
    test_fused_rep_rejected_without_error_out();
    std::printf("test_representation_contract: PASS\n");
    return 0;
  } catch (std::exception const& e) {
    std::fprintf(stderr, "test_representation_contract: FAIL: %s\n", e.what());
    return 1;
  }
}
