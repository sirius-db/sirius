// SPDX-License-Identifier: Apache-2.0
//
// The operator registry: one table describing every operator kind, plus the
// name/id/channel lookups and the make_compressor factory that derive from it.
// reconstruct_representation, all_compressor_names, the fusion/explorer
// predicates, and DSL channel canonicalisation all consult this table so the
// operator set has a single source of truth.

#include "codegen/plan/operator_registry.hpp"

#include "codegen/plan/plan_interpreter.hpp"
#include "codegen/plan/representation.hpp"

#include <string_view>

namespace simpatico {
namespace {

// Suffix after `prefix_` (i.e. the parameter part of a `<prefix>_<params>`
// name), or nullopt if `name` is not that parameterised form.
std::optional<std::string_view> after_prefix(std::string const& name, std::string_view prefix)
{
  if (name.size() > prefix.size() + 1 && name.compare(0, prefix.size(), prefix) == 0 &&
      name[prefix.size()] == '_') {
    return std::string_view(name).substr(prefix.size() + 1);
  }
  return std::nullopt;
}

}  // namespace

std::vector<OperatorInfo> const& operator_registry()
{
  // Catalog order: all_compressor_names() flattens catalog_names in this order.
  // channels mirror each rep's named_channels(); {} = variable-arity.
  // clang-format off
  // One row per operator; kept on a single line each (formatting disabled).
  //  id               name              channels                                                                 catalog_names                      term   pre    cg
  static const std::vector<OperatorInfo> kTable = {
    {OpId::Delta,          "delta",           {"differences"},                                                        {"delta"},                         false, true,  true},
    {OpId::Rle,            "rle",             {"runs", "values"},                                                     {"rle"},                           false, true,  true},
    {OpId::Bitpack,        "bitpack",         {"chunk_min", "chunk_count", "chunk_bits", "packed"},                   {"bitpack"},                       false, false, true},
    {OpId::For,            "for",             {"deltas", "references"},                                               {"for"},                           false, true,  true},
    {OpId::Zigzag,         "zigzag",          {"zigzag"},                                                             {"zigzag"},                        false, true,  true},
    {OpId::Dictionary,     "dictionary",      {},                                                                     {"dictionary"},                    false, false, false},
    {OpId::Alp,            "alp",             {"integers", "exceptions", "exception_positions", "metadata"},          {"alp"},                           false, true,  false},
    {OpId::AlpRd,          "alp_rd",          {"right_parts", "dict_indices", "dict", "metadata", "exceptions", "exception_positions"}, {"alp_rd"},      false, true,  false},
    {OpId::Ans,            "ans",             {"output"},                                                             {"ans"},                           true,  false, false},
    {OpId::Bitcomp,        "bitcomp",         {"output"},                                                             {"bitcomp"},                       true,  false, false},
    {OpId::Snappy,         "snappy",          {"output"},                                                             {"snappy"},                        true,  false, false},
    {OpId::Deflate,        "deflate",         {"output"},                                                             {"deflate"},                       true,  false, false},
    {OpId::Lz4,            "lz4",             {"output"},                                                             {"lz4"},                           true,  false, false},
    {OpId::Bitextract,     "bitextract",      {},                                                                     {"bitextract_f32", "bitextract_f64"}, false, true, false},
    {OpId::Identity,       "identity",        {"data"},                                                               {},                                false, false, false},
    {OpId::NvcompCascaded, "nvcomp_cascaded", {"output"},                                                             {},                                false, false, false},
    // 2 or 3 channels: offsets, chars[, null_mask]. null_mask is optional (present only when
    // nullable); the generic named_channels() skips nullptr slots so arity is correct at runtime.
    {OpId::StrSplit,       "str_split",       {"offsets", "chars", "null_mask"},                                     {"str_split"},                     false, true,  false},
  };
  // clang-format on
  return kTable;
}

OperatorInfo const& op_info(OpId id)
{
  for (auto const& e : operator_registry()) {
    if (e.id == id) return e;
  }
  // Unreachable: every OpId has a row.
  return operator_registry().front();
}

std::optional<OpId> op_id_from_name(std::string const& name)
{
  // Parameterised bitextract family: any `bitextract_<spec>` (spec validity is
  // re-checked at construction/reconstruction).
  if (strip_bitextract_prefix(name)) return OpId::Bitextract;

  for (auto const& e : operator_registry()) {
    if (name == e.name) return e.id;
  }

  // "dictionary_fast" is an alias for the dictionary op (same rep/OpId), selecting
  // the 2-buffer keys+indices form via a compressor flag rather than a distinct op.
  if (name == "dictionary_fast") return OpId::Dictionary;

  // Parameterised suffix families sharing a bare-name entry above.
  if (after_prefix(name, "bitcomp")) return OpId::Bitcomp;
  if (after_prefix(name, "nvcomp_cascaded")) return OpId::NvcompCascaded;
  return std::nullopt;
}

std::vector<std::string> const* canonical_channels(OpId id)
{
  auto const& ch = op_info(id).channels;
  return ch.empty() ? nullptr : &ch;
}

std::optional<ChannelId> channel_id(OpId id, std::string_view channel)
{
  auto const& ch = op_info(id).channels;
  for (std::size_t i = 0; i < ch.size(); ++i) {
    if (ch[i] == channel) return static_cast<ChannelId>(i);
  }
  return std::nullopt;
}

// Resolve a DSL compressor name to a compressor instance, or nullptr if the
// name is unknown or its parameters are malformed. The fused ops
// (delta/rle/bitpack/for/zigzag) return nullptr here — they go through the JIT
// codegen encoder, not a compressor factory.
std::unique_ptr<compressor> make_compressor(std::string const& name)
{
  auto id = op_id_from_name(name);
  if (!id) return nullptr;
  switch (*id) {
    case OpId::Identity: return std::make_unique<identity_compressor>();
    case OpId::Dictionary:
      return std::make_unique<dictionary_compressor>(name == "dictionary_fast");
    case OpId::StrSplit: return std::make_unique<str_split_compressor>();
    case OpId::Alp: return std::make_unique<alp_compressor>();
    case OpId::AlpRd: return std::make_unique<alp_rd_compressor>();
    case OpId::Ans: return std::make_unique<ans_compressor>();
    case OpId::Snappy: return std::make_unique<snappy_compressor>();
    case OpId::Lz4: return std::make_unique<lz4_compressor>();
    case OpId::Deflate: return std::make_unique<deflate_compressor>();
    case OpId::Bitextract: {
      if (auto suffix = strip_bitextract_prefix(name)) {
        if (!parse_bitextract_spec(*suffix).fields.empty()) {
          return std::make_unique<bitextract_compressor>(*suffix);
        }
      }
      return nullptr;
    }
    case OpId::Bitcomp: {
      // Bare `bitcomp` (and `bitcomp_default`) select algorithm 0;
      // `bitcomp_sparse` selects nvcomp's sparse algorithm.
      if (name == "bitcomp") return std::make_unique<bitcomp_compressor>();
      int algorithm = 0;
      if (auto suffix = after_prefix(name, "bitcomp");
          suffix && parse_bitcomp_suffix(*suffix, &algorithm)) {
        return std::make_unique<bitcomp_compressor>(algorithm);
      }
      return nullptr;
    }
    case OpId::NvcompCascaded: {
      // `nvcomp_cascaded` — nvcomp default opts; `nvcomp_cascaded_<N>D<M>R<K>B` — explicit opts.
      if (name == "nvcomp_cascaded") return std::make_unique<cascaded_compressor>();
      int deltas = 0, rles = 0, bp = 0;
      if (auto suffix = after_prefix(name, "nvcomp_cascaded");
          suffix && parse_nvcomp_cascaded_suffix(*suffix, &deltas, &rles, &bp)) {
        return std::make_unique<cascaded_compressor>(deltas, rles, bp);
      }
      return nullptr;
    }
    // Codegen-fused ops are built by the JIT encoder, not this factory.
    case OpId::Delta:
    case OpId::Rle:
    case OpId::Bitpack:
    case OpId::For:
    case OpId::Zigzag: return nullptr;
  }
  return nullptr;
}

// A node is codegen-fusable iff its OperatorInfo (registry table above) carries
// the `codegen` flag.
bool is_codegen_compressor(std::string const& op)
{
  auto id = op_id_from_name(op);
  return id && op_info(*id).codegen;
}

}  // namespace simpatico
