#include "codegen/jit/fused_tree.hpp"

namespace codegen::jit {

std::string child_dtype(OpKind parent_op,
                        const std::string& child_key,
                        const std::string& parent_dtype)
{
  if (parent_op == OpKind::Rle && child_key == "runs") { return "int32_t"; }
  return parent_dtype;
}

std::string op_kind_name(OpKind op)
{
  switch (op) {
    case OpKind::Bitpack: return "BITPACK";
    case OpKind::For: return "FOR";
    case OpKind::Delta: return "DELTA";
    case OpKind::Rle: return "RLE";
    case OpKind::Raw: return "RAW";
    case OpKind::None: return "NONE";
  }
  return "UNKNOWN";
}

}  // namespace codegen::jit
