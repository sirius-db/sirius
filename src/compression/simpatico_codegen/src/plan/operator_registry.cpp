// SPDX-License-Identifier: Apache-2.0
#include "codegen/plan/representation.hpp"

#include <string_view>

namespace simpatico {

std::unique_ptr<compressor> make_compressor(std::string const& name)
{
  if (name == "identity") { return std::make_unique<identity_compressor>(); }
  if (name == "dictionary") { return std::make_unique<dictionary_compressor>(); }
  if (name == "for") { return std::make_unique<for_compressor>(); }
  if (name == "alp") { return std::make_unique<alp_compressor>(); }
  if (name == "alp_rd") { return std::make_unique<alp_rd_compressor>(); }
  if (auto suffix = strip_bitextract_prefix(name)) {
    if (!parse_bitextract_spec(*suffix).fields.empty()) {
      return std::make_unique<bitextract_compressor>(*suffix);
    }
  }
  if (name == "ans") { return std::make_unique<ans_compressor>(); }
  if (name == "snappy") { return std::make_unique<snappy_compressor>(); }
  if (name == "lz4") { return std::make_unique<lz4_compressor>(); }
  if (name == "deflate") { return std::make_unique<deflate_compressor>(); }
  if (name == "bitcomp") { return std::make_unique<bitcomp_compressor>(); }
  // `bitcomp_default` / `bitcomp_sparse`. `bitcomp_default` is an
  // explicit alias for bare `bitcomp` (algorithm=0); `bitcomp_sparse`
  // selects nvcomp's sparse algorithm (faster on zero-rich data).
  static constexpr std::string_view kBitcompPrefix = "bitcomp";
  if (name.size() > kBitcompPrefix.size() + 1 &&
      name.compare(0, kBitcompPrefix.size(), kBitcompPrefix) == 0 &&
      name[kBitcompPrefix.size()] == '_') {
    std::string_view suffix(name);
    suffix.remove_prefix(kBitcompPrefix.size() + 1);
    int algorithm = 0;
    if (parse_bitcomp_suffix(suffix, &algorithm)) {
      return std::make_unique<bitcomp_compressor>(algorithm);
    }
  }
  // `nvcomp_cascaded` — nvcomp default opts (num_deltas=1, num_RLEs=2, use_bp=1).
  if (name == "nvcomp_cascaded") { return std::make_unique<cascaded_compressor>(); }
  // `nvcomp_cascaded_<N>D<M>R<K>B` — explicit opts.
  static constexpr std::string_view kCascadedPrefix = "nvcomp_cascaded";
  if (name.size() > kCascadedPrefix.size() + 1 &&
      name.compare(0, kCascadedPrefix.size(), kCascadedPrefix) == 0 &&
      name[kCascadedPrefix.size()] == '_') {
    std::string_view suffix(name);
    suffix.remove_prefix(kCascadedPrefix.size() + 1);
    int deltas = 0, rles = 0, bp = 0;
    if (parse_nvcomp_cascaded_suffix(suffix, &deltas, &rles, &bp)) {
      return std::make_unique<cascaded_compressor>(deltas, rles, bp);
    }
  }
  return nullptr;
}

}  // namespace simpatico
