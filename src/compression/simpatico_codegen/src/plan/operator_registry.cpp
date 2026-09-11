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

#include <algorithm>
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
  // Catalog order: all_compressor_names() emits explorable ops in this order.
  // channels are canonical DSL output ports used for plan routing; {} =
  // variable-arity. Generic reps usually expose the same names through
  // named_channels(), while fused reps persist separate manifest buffers.
  // Decode-only transients such as Bitpack bp_offsets are never registry ports.
  // clang-format off
  // One row per operator; kept on a single line each (formatting disabled).
  //  id               name              channels                                                                 expl   term   pre    cg
  static const std::vector<OperatorInfo> kTable = {
    {OpId::Delta,          "delta",           {"differences"},                                                        true,  false, true,  true, {{"delta_first", ChannelLayout::per_chunk_metadata}}},
    {OpId::Rle,            "rle",             {"runs", "values"},                                                     true,  false, true,  true, {{"rle_runs_offsets", ChannelLayout::whole_column}}},
    {OpId::Bitpack,        "bitpack",         {"chunk_min", "chunk_count", "chunk_bits", "packed"},                   true,  false, false, true, {{"chunk_min", ChannelLayout::per_chunk_metadata}, {"chunk_count", ChannelLayout::per_chunk_metadata}, {"chunk_bits", ChannelLayout::per_chunk_metadata}, {"packed", ChannelLayout::bulk_variable}}},
    {OpId::For,            "for",             {"deltas", "references"},                                               true,  false, true,  true, {{"references", ChannelLayout::per_chunk_metadata}}},
    {OpId::Zigzag,         "zigzag",          {"zigzag"},                                                             true,  false, true,  true, {{"zigzag", ChannelLayout::bulk_fixed_stride}}},
    // keys_* are the dictionary itself — column state, valid for any subset of the rows that
    // index into it. `indices` is the row-indexed channel. Validity is never a channel here: it
    // is stripped into the tree's sidecar before the walk, so no layout has to describe a
    // one-bit-per-row buffer that byte-per-row arithmetic could not address anyway.
    {OpId::Dictionary,     "dictionary",      {},                                                                     true,  false, false, false, {{"keys_offsets", ChannelLayout::column_state}, {"keys_chars", ChannelLayout::column_state}, {"indices", ChannelLayout::bulk_fixed_stride}}},
    {OpId::Alp,            "alp",             {"integers", "exceptions", "exception_positions", "metadata"},          true,  false, true,  false, {}},
    {OpId::AlpRd,          "alp_rd",          {"right_parts", "dict_indices", "dict", "metadata", "exceptions", "exception_positions"}, true, false, true, false, {}},
    {OpId::Ans,            "ans",             {"output"},                                                             true,  true,  false, false, {{"output", ChannelLayout::whole_column}}},
    {OpId::Bitcomp,        "bitcomp",         {"output"},                                                             true,  true,  false, false, {{"output", ChannelLayout::whole_column}}},
    {OpId::Snappy,         "snappy",          {"output"},                                                             true,  true,  false, false, {{"output", ChannelLayout::whole_column}}},
    {OpId::Deflate,        "deflate",         {"output"},                                                             true,  true,  false, false, {{"output", ChannelLayout::whole_column}}},
    {OpId::Lz4,            "lz4",             {"output"},                                                             true,  true,  false, false, {{"output", ChannelLayout::whole_column}}},
    {OpId::Bitextract,     "bitextract",      {},                                                                     true,  false, true,  false, {}},
    {OpId::Identity,       "identity",        {"data"},                                                               false, false, false, false, {{"data", ChannelLayout::bulk_fixed_stride}}},
    {OpId::NvcompCascaded, "nvcomp_cascaded", {"output"},                                                             false, false, false, false, {{"output", ChannelLayout::whole_column}}},

    {OpId::StrSplit,       "str_split",       {"offsets", "chars"},                                                   true,  false, true,  false, {{"offsets", ChannelLayout::bulk_fixed_stride}, {"chars", ChannelLayout::whole_column}}},

  };
  // clang-format on
  return kTable;
}

std::optional<ChannelLayout> buffer_layout(OpId id, std::string_view buffer)
{
  for (auto const& [name, layout] : op_info(id).persisted_buffers) {
    if (name == buffer) { return layout; }
  }
  return std::nullopt;
}

bool supports_chunk_subset(OpId id)
{
  auto const& bufs = op_info(id).persisted_buffers;
  // Unclassified (an op whose persisted set we have not described, e.g. the variable-arity ones)
  // must fetch whole: an unlisted buffer would otherwise be silently omitted from the subset.
  if (bufs.empty()) { return false; }
  return std::ranges::none_of(
    bufs, [](auto const& b) { return b.second == ChannelLayout::whole_column; });
}

bool channel_is_column_state(OpId id, std::string_view channel)
{
  // The DSL names an edge by its dotted path (`dictionary.keys_offsets`); the port is the last
  // component. Matching on that keeps this working for a nested path without teaching it the
  // path grammar.
  auto const dot  = channel.rfind('.');
  auto const port = dot == std::string_view::npos ? channel : channel.substr(dot + 1);
  for (auto const& [name, layout] : op_info(id).persisted_buffers) {
    if (name == port) { return layout == ChannelLayout::column_state; }
  }
  return false;
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

std::vector<std::string> const& all_compressor_names()
{
  static std::vector<std::string> const names = [] {
    std::vector<std::string> out;
    for (auto const& op : operator_registry()) {
      if (!op.explorable) continue;
      // bitextract's DSL names are its float aliases, not the bare op name.
      if (op.id == OpId::Bitextract) {
        out.emplace_back("bitextract_f32");
        out.emplace_back("bitextract_f64");
      } else {
        out.emplace_back(op.name);
      }
    }
    return out;
  }();
  return names;
}

bool is_terminal_compressor(std::string const& name)
{
  auto id = op_id_from_name(name);
  return id && op_info(*id).terminal;
}

bool is_preprocessing_compressor(std::string const& name)
{
  auto id = op_id_from_name(name);
  return id && op_info(*id).preprocessing;
}

// A node is codegen-fusable iff its OperatorInfo (registry table above) carries
// the `codegen` flag.
bool is_codegen_compressor(std::string const& op)
{
  auto id = op_id_from_name(op);
  return id && op_info(*id).codegen;
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
    case OpId::Dictionary: return std::make_unique<dictionary_compressor>();
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

}  // namespace simpatico
