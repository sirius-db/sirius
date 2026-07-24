#include "api/simpatico_codegen.hpp"
#include "codegen/jit/nvrtc_compiler.hpp"
#include "codegen/plan/leaf_desc.hpp"
#include "codegen/plan/representation.hpp"
#include "test_utils.hpp"

#include <cudf/column/column_factories.hpp>
#include <cudf/table/table.hpp>
#include <cudf/types.hpp>

#include <cuda_runtime.h>

#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void expect(bool cond, char const* msg)
{
  if (!cond) throw std::runtime_error(msg);
}

simpatico::compressed_representation const* find_bitpack_leaf(simpatico::PlanTree const& tree)
{
  for (auto const& node : tree.nodes) {
    if (node.rep && node.rep->kind() == simpatico::OpId::Bitpack) { return node.rep.get(); }
    for (auto const& [path, rep] : node.channels) {
      (void)path;
      if (rep && rep->kind() == simpatico::OpId::Bitpack) { return rep.get(); }
    }
  }
  return nullptr;
}

}  // namespace

int main()
{
  if (cudaSetDevice(0) != cudaSuccess) {
    std::fprintf(stderr, "test_leaf_describe: cudaSetDevice failed\n");
    return 1;
  }
  try {
    constexpr int num_rows = 4096;
    // Single INT32 column of small-range values (shared factory); the leaf
    // structure asserted below is independent of the exact payload.
    auto table = make_int32_table(/*num_cols=*/1, num_rows, /*seed=*/0);

    std::string dsl =
      "input -> delta -> differences\n"
      "delta.differences -> rle -> values, runs\n"
      "delta.differences.values -> bitpack\n"
      "delta.differences.runs -> bitpack\n";

    auto ct = simpatico::compress_with_plan(table->view(), dsl);
    expect(ct.columns.size() == 1, "one compressed column");
    expect(ct.columns[0].compound != nullptr, "compound present");

    auto const* rep = find_bitpack_leaf(*ct.columns[0].compound);
    expect(rep != nullptr, "bitpack leaf found");
    expect(rep->kind() == simpatico::OpId::Bitpack, "bitpack kind");
    expect(rep->decoded_type().id() == cudf::type_id::INT32, "decoded type");

    auto meta = rep->describe_meta();
    expect(std::holds_alternative<simpatico::leaf_meta::none>(meta), "no leaf meta");

    std::printf("test_leaf_describe: PASS\n");
    return 0;
  } catch (std::exception const& e) {
    std::fprintf(stderr, "test_leaf_describe: FAIL: %s\n", e.what());
    return 1;
  }
}
