// SPDX-License-Identifier: Apache-2.0
//
// Every knob that decides whether, and how hard, a decode filters.
//
// One reader per knob, shared by everything that has to agree on it: the scan
// (does it bother analysing its filter?), the memory estimator (will the batch
// come back smaller?) and the decode itself. They used to be copied per
// translation unit, in three different parse styles, and had already drifted —
// one reader accepted only "1" where the decode accepts anything but "0", so a
// value like "true" turned the feature on in one layer and off in another.
//
// All are read once and cached: the gate check sits on the per-batch path.
//
// TODO: these should be Sirius parameters, not environment variables. As env
// vars they are invisible to duckdb_settings(), cannot be set per session, and
// — because the values cache on first read — cannot be varied within one test
// binary, which is why some decode paths can only be reached by a unit test
// rather than end to end. The obstacle is layering: this library has no DuckDB
// dependency, so the values cannot be pulled from the setting registry here.
// They would have to arrive from Sirius, either pushed down once per query or
// passed into the decode call as a policy value. Left as env while the feature
// is experimental; revisit before it ships.

#pragma once

#include <cstddef>

namespace sirius::codegen {

/// Master gate (SIRIUS_EXP_FUSED_SCAN_FILTER). Set and not exactly "0" = on.
bool decompression_pushdown_enabled();

/// Promote the decision trace to INFO / stderr (SIRIUS_EXP_FUSED_SCAN_DIAG),
/// same "set and not exactly 0" contract.
///
/// The trace is permanent tooling, not temporary instrumentation: it records
/// every accept/decline decision, and raising the level is the first move
/// whenever a batch quietly falls back to a plain decode.
bool decompression_pushdown_diag_enabled();

/// Surviving-row fraction above which the decode gives compaction up and
/// produces ordinary full-width columns (SIRIUS_EXP_FUSED_SCAN_MAX_SEL,
/// default 0.35). Every batch that DOES come back compacted is bounded by it,
/// which is what makes sizing a memory reservation off it sound.
///
/// Measured: wins at sel <= .152, losses by .526; the mask walk costs about the
/// plain decode at .5.
double decompression_pushdown_max_selectivity();

/// The same, for a batch with any full-width output
/// (SIRIUS_EXP_FUSED_SCAN_TIERB_MAX_SEL, default 0.10): a full decode plus
/// gather costs about the unfiltered path, so the win is the compacted batch
/// (and, when the decode carries the whole filter, the skipped post-filter),
/// which only pays off at low selectivity.
double decompression_pushdown_full_route_max_selectivity();

/// Surviving-row fraction at or below which walking the survivor index list
/// beats walking the mask bits (SIRIUS_EXP_FUSED_SCAN_K4_MAX_SEL, default
/// 0.15). The microbench crossover sits at 15-50% depending on bit width, so
/// this is the conservative edge. Setting it tiny is the effective kill switch
/// for the index walk (the parse requires > 0).
double decompression_pushdown_index_walk_max_selectivity();

/// How many join-filter probes one decode carries
/// (SIRIUS_EXP_FUSED_SCAN_MAX_MEMBER, default 1).
///
/// Wave-1 probes run at FULL width, so k probes cost k*N rows of probe+adapter
/// work, while the downstream operator's cascade costs ~1*N (later probes see
/// compacted survivors). Measured with 3 probes: the added probe-side cost
/// outweighed the compaction win. One probe is always ~volume-neutral
/// against the cascade's first probe, so the compaction win survives. Sources
/// beyond the cap are DROPPED, which is sound — the mask is conjunctive and
/// such batches are never tagged as fully filtered, so the downstream operator
/// completes the conjunction on the compacted batch. The caller must order the
/// probes by ascending expected keep-rate so the kept prefix is the strongest.
std::size_t decompression_pushdown_max_membership_sources();

}  // namespace sirius::codegen
