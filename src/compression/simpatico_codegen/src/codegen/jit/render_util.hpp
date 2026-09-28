// Shared string / entry-symbol helpers for the encode and decode JIT renderers.
// Both renderers walk the same FusedTree IR and emit plain-CUDA source, so these
// small building blocks are identical on both sides.
#pragma once

#include "codegen/jit/fused_tree.hpp"

#include <cctype>
#include <cstddef>
#include <sstream>
#include <string>
#include <string_view>

namespace codegen::jit {

// Same-width unsigned type name, for arithmetic that must wrap modulo 2^N
// instead of relying on signed overflow (UB).
inline const char* unsigned_counterpart(std::size_t elem_size)
{
  return (elem_size == 8) ? "uint64_t" : "uint32_t";
}

inline std::string replace_all(std::string s, std::string_view needle, std::string_view repl)
{
  std::string out;
  out.reserve(s.size());
  std::size_t pos = 0;
  while (true) {
    auto found = s.find(needle, pos);
    if (found == std::string::npos) {
      out.append(s, pos, std::string::npos);
      break;
    }
    out.append(s, pos, found - pos);
    out.append(repl);
    pos = found + needle.size();
  }
  return out;
}

// Walk-order op tag for the entry symbol. Kept short — the kernel-cache key is
// the source hash, not the symbol.
inline void append_op_segment(std::ostringstream& oss, const FusedTree& node)
{
  switch (node.op) {
    case OpKind::Bitpack: oss << "bp"; break;
    case OpKind::Delta: oss << "dl"; break;
    case OpKind::Rle: oss << "rl"; break;
    case OpKind::Raw: oss << "rw"; break;
    case OpKind::For: oss << "fr"; break;
    case OpKind::Zigzag: oss << "zz"; break;
    default: oss << "un"; break;
  }
  for (const auto& [k, child] : node.children) {
    (void)k;
    oss << "_";
    append_op_segment(oss, *child);
  }
}

// Entry-point symbol: <prefix><op-tags>_<sanitized dtype>. `prefix` distinguishes
// the two renderers ("simpatico_encode_" / "simpatico_decode_").
inline std::string make_entry_symbol(const FusedTree& tree,
                                     const std::string& element_dtype,
                                     std::string_view prefix)
{
  std::ostringstream oss;
  oss << prefix;
  append_op_segment(oss, tree);
  oss << "_";
  for (char c : element_dtype)
    oss << (std::isalnum(static_cast<unsigned char>(c)) || c == '_' ? c : '_');
  return oss.str();
}

}  // namespace codegen::jit
