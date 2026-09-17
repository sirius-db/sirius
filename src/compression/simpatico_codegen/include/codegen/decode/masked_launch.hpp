// Masked JIT decode launchers — the kernels that evaluate a scan's filter
// while decompressing, and decode only the rows that survive it.
//
// Companions to simpatico::launch_decode_fused_tree (codegen_bridge.hpp).
// Every launcher enqueues into the mandatory decode_frame and propagates errors.
// Input, selection, and destination storage must be owned by the frame or remain
// borrowed through its session's completion; the caller establishes semantic
// applicability before launching. No launcher completes the stream.
//
// Mask and chunk_offsets layout: codegen/selection/selection.hpp.  Every
// consuming launcher needs ``mask.chunk_offsets``, i.e. the CNT wave must have
// run, and sizes its output from ``mask.survivor_count``.
//
// One entry point per CONSUMER, mirroring the renderer's axes (DecodeShape in
// decode/jit/renderer.hpp).  The ENUMERATOR is a parameter, not a function:
// pass ``row_indices`` to walk the survivor index list, or nullptr to walk the
// mask bits.  It changes which rows a block visits, never the output or the
// signature — so a consumer that gains the index walk costs no new name here,
// and the two enumerations cannot drift apart in the contract check.
//
#pragma once

#include "codegen/jit/fused_tree.hpp"
#include "codegen/selection/chunk_row_set.hpp"
#include "codegen/selection/selection.hpp"

#include <cstdint>

namespace simpatico {

class decode_frame;

/// How a compacting launcher enumerates survivors. At most one form is set;
/// none means walk the mask bits. It changes which rows a block visits and how
/// many blocks launch — never the output, never the signature.
///
/// The mask and index forms describe a selection the decode itself made, so they
/// are dense over the batch's chunks; a row set is for one that arrives
/// afterwards and launches only the chunks it touches.
struct row_enumeration {
  std::int32_t const* row_indices              = nullptr;
  ::sirius::codegen::chunk_row_set const* rows = nullptr;

  [[nodiscard]] bool by_index() const noexcept { return row_indices != nullptr; }
  [[nodiscard]] bool by_row_set() const noexcept { return rows != nullptr; }
};

/// Range ballot: decode fused with the inclusive range ``pred`` (decoded
/// integer domain), balloting into ``mask.words``; no column output.  Any
/// value_source root.  ``mask.survivor_count`` / ``chunk_offsets`` are left to
/// the CNT wave.  Float columns must NOT be routed here — they decode as
/// bit-reinterpreted integers, so an integer-domain compare is meaningless.
void launch_decode_fused_tree_mask_out(codegen::jit::FusedTree const& tree,
                                       codegen::jit::LabeledBuffers& labeled,
                                       char const* dtype,
                                       std::int64_t num_rows,
                                       ::sirius::codegen::range_predicate pred,
                                       ::sirius::codegen::selection_mask& mask,
                                       decode_frame& frame);

/// Compacting value decode: writes the survivors' values to ``out``, in row
/// order, sized by ``mask.survivor_count``.
///
/// Mask walk (``row_indices == nullptr``): a Bitpack leaf never unpacks a
/// rejected row; a Delta root still reconstructs the chunk (its prefix sum is
/// sequential) and only the STORES are compacted, which saves the full-width
/// write and the downstream gather, not the unpack.
///
/// Index walk (``row_indices`` given): the mask->indices wave's ascending
/// global int32 list, consistent with ``mask.chunk_offsets`` — chunk c's rows
/// occupy row_indices[chunk_offsets[c] .. chunk_offsets[c+1]).  Only listed
/// rows are decoded, by random access, so cost scales with survivors rather
/// than chunk size; the crossover is ~15% selectivity by microbench and the
/// pick is the caller's.  Bitpack leaf roots only — a Delta root is refused
/// (throws a renderer error); choose the mask walk before launch instead.
void launch_decode_fused_tree_compacted(codegen::jit::FusedTree const& tree,
                                        codegen::jit::LabeledBuffers& labeled,
                                        char const* dtype,
                                        std::int64_t num_rows,
                                        ::sirius::codegen::selection_mask const& mask,
                                        row_enumeration rows,
                                        void* out,
                                        decode_frame& frame);

/// Dictionary gather: for dictionary->bitpack string columns with
/// CONSTANT-WIDTH, null-free keys.  ``tree`` is the codes
/// bitpack leaf; for survivor rows only it decodes the code and copies that
/// key's ``key_width`` bytes from ``keys_chars`` (key k at k*key_width) into
/// the compacted ``out_chars``, preserving row order.  Skips both the
/// full-width code column and a separate key gather.  The offsets column is
/// analytic (j*key_width) and the caller assembles it.
void launch_decode_fused_tree_dict_gather(codegen::jit::FusedTree const& tree,
                                          codegen::jit::LabeledBuffers& labeled,
                                          char const* dtype,
                                          std::int64_t num_rows,
                                          ::sirius::codegen::selection_mask const& mask,
                                          row_enumeration rows,
                                          void const* keys_chars,
                                          std::int32_t key_width,
                                          void* out_chars,
                                          decode_frame& frame);

/// str_split gather, phase 1: survivor metadata.  ``tree`` is the string
/// column's OFFSETS subtree (Bitpack- or Delta-rooted, any depth below);
/// ``num_string_rows`` is the STRING count n, and the kernel runs over the
/// n+1 offsets domain internally while the mask stays row-space.  Writes,
/// compacted by rank: ``src_offsets_out`` (char-range starts in the RAW chars
/// buffer) and ``lengths_out`` (byte lengths, for the caller's scan).  Chars
/// are never read here, so entropy-coded chars are out of scope by
/// construction — route those through full decode + gather.
void launch_decode_fused_tree_str_split_meta(codegen::jit::FusedTree const& tree,
                                             codegen::jit::LabeledBuffers& labeled,
                                             char const* dtype,
                                             std::int64_t num_string_rows,
                                             ::sirius::codegen::selection_mask const& mask,
                                             row_enumeration rows,
                                             std::int64_t* src_offsets_out,
                                             std::int32_t* lengths_out,
                                             decode_frame& frame);

/// str_split gather, phase 2: tree-independent byte gather from the RAW chars
/// buffer — survivor j copies out_offsets[j+1]-out_offsets[j] bytes from
/// chars[src_offsets[j]].  ``out_offsets`` is the exclusive scan of phase 1's
/// lengths, in cudf offsets layout, so it doubles as the compacted offsets
/// column.
void launch_masked_char_copy(void const* chars,
                             std::int64_t const* src_offsets,
                             std::int32_t const* out_offsets,
                             std::int64_t n_survivors,
                             void* out_chars,
                             decode_frame& frame);

}  // namespace simpatico
