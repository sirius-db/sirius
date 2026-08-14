/*
 * Copyright 2026, Sirius Contributors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "planner/late_mat_plan_pass.hpp"

#include "expression/ast/node.hpp"
#include "expression/ast/utils.hpp"
#include "op/sirius_physical_concat.hpp"
#include "op/sirius_physical_filter.hpp"
#include "op/sirius_physical_hash_join.hpp"
#include "op/sirius_physical_operator.hpp"
#include "op/sirius_physical_partition.hpp"
#include "op/sirius_physical_projection.hpp"

#include <algorithm>
#include <optional>
#include <string>
#include <variant>

namespace sirius::planner {

namespace {

using op::sirius_physical_operator;
using op::SiriusPhysicalOperatorType;

/// Whether `expr` reads the column at `pos`.
bool reads_column(ast::node const& expr, std::size_t pos)
{
  bool found = false;
  ast::visit_references(expr, [&](ast::reference const& ref) {
    if (ref.column_index == pos) { found = true; }
  });
  return found;
}

/// What one operator does to a column arriving from `from` at input position
/// `in_pos`.
struct step {
  /// Where the column moves to, or nullopt when its life ends here — because
  /// this operator reads it, or because it is not passed on.
  std::optional<std::size_t> moved_to;
  /// The life ended because a join compares this column. See
  /// column_lifetime::read_as_join_key: such a column leaves candidacy
  /// entirely, which is stronger than stopping here.
  bool as_join_key = false;

  static step reads() { return step{}; }
  static step key() { return step{std::nullopt, true}; }
  static step to(std::optional<std::size_t> position) { return step{position, false}; }
};

step trace_through(sirius_physical_operator const& node,
                   sirius_physical_operator const& from,
                   std::size_t in_pos,
                   bool& nullified)
{
  switch (node.type) {
    // Positionally transparent plumbing. A partition rewrites which rows sit in
    // which batch and a wrap concat glues those batches back together; neither
    // touches the column layout.
    case SiriusPhysicalOperatorType::PARTITION: {
      auto const& partition = static_cast<op::sirius_physical_partition const&>(node);
      // Except that a partition READS the columns it hashes to place a row —
      // and it resolved those positions from its consumer at plan time, so they
      // are known here rather than one hop up at the join. A rowid hashes
      // differently from the value it stands for, so a key riding as one would
      // scatter equal values across partitions and the join would simply miss
      // matches. This is the walk's one silent-wrong-answer shape, and it is
      // why a partition is not transparent by fiat.
      auto const& keys = partition.partition_keys();
      if (std::find(keys.begin(), keys.end(), static_cast<int>(in_pos)) != keys.end()) {
        return step::key();
      }
      return step::to(in_pos);
    }

    case SiriusPhysicalOperatorType::CONCAT: {
      // The generator builds a CONCAT at exactly one site — wrapping one child
      // of one join — so every concat gathers the partitions of ONE flow and a
      // payload crossing it still comes from where the scan said it did.
      //
      // That matters because fan-in at a concat is expressed through PORTS, not
      // tree children: a concat merging two different producers into one
      // repository could not be recognised from here, and if both producers had
      // deferred with type-identical schemas the port's whole-schema match
      // could not tell them apart either. No such concat exists today; the
      // arity check is what would notice if one appeared.
      if (node.children.size() != 1) { return step::reads(); }
      return step::to(in_pos);
    }

    case SiriusPhysicalOperatorType::HASH_JOIN: {
      auto const& join = static_cast<op::sirius_physical_hash_join const&>(node);
      // The types that pass a payload through at all. Semi, anti and mark
      // joins collect a single side's columns, so a payload riding the other
      // is not in the output to be deferred.
      auto const type    = join.join_type;
      bool const carries = type == duckdb::JoinType::INNER || type == duckdb::JoinType::LEFT ||
                           type == duckdb::JoinType::RIGHT || type == duckdb::JoinType::OUTER;
      if (!carries) { return step::reads(); }
      if (node.children.size() != 2) { return step::reads(); }

      bool const from_lhs = node.children[0].get() == &from;
      if (!from_lhs && node.children[1].get() != &from) { return step::reads(); }

      // A key is read: the join compares its values. The conditions' left
      // nodes address the lhs and their right nodes the rhs, so only this
      // side's half is consulted.
      for (auto const& condition : join.conditions) {
        auto const& side = from_lhs ? condition.left : condition.right;
        if (side && reads_column(*side, in_pos)) { return step::key(); }
      }

      std::vector<int> const lhs(join.lhs_output_columns.col_idxs.begin(),
                                 join.lhs_output_columns.col_idxs.end());
      std::vector<int> const rhs(join.rhs_output_columns.col_idxs.begin(),
                                 join.rhs_output_columns.col_idxs.end());
      auto const moved = join_output_position(from_lhs, lhs, rhs, in_pos);

      // An outer join can emit a row this side never matched. The rowid is
      // null there and the column materializes null, which is sound — so the
      // ride continues and the fact is recorded rather than refused.
      //
      // Recorded only for a column that actually RIDES. A column read here as a
      // key, or not projected out, arrives with its own values and stops; the
      // join nullifies its OUTPUT, which that column never reaches. Marking it
      // anyway would cost the key substitution a deferral it is entitled to.
      if (moved.has_value()) {
        nullified = nullified || type == duckdb::JoinType::OUTER ||
                    (from_lhs && type == duckdb::JoinType::RIGHT) ||
                    (!from_lhs && type == duckdb::JoinType::LEFT);
      }
      return step::to(moved);
    }

    // A dynamic filter drops rows a join build has already excluded. It reads
    // only the key columns it probes, and rewrites no column layout, so a
    // payload riding past it is positionally unchanged. Treating it as a reader
    // instead ends the ride at the operator that sits directly above a pinned
    // scan, which is where q10's customer payload was stopping: the bundle
    // materialized one hop from the scan and paid the rowid for nothing.
    //
    // The probed columns are not enumerated here: the filter set is published
    // mid-query and a column this walk decided was payload must not become a
    // probe key afterwards. Deferring a probed key would hand the probe a rowid
    // in place of the value it compares, so it is refused at the source instead
    // -- the scan withholds any column a partition hashes, which is the same
    // column set this operator probes.
    case SiriusPhysicalOperatorType::DYNAMIC_FILTER: return step::to(in_pos);

    case SiriusPhysicalOperatorType::FILTER: {
      auto const& filter = static_cast<op::sirius_physical_filter const&>(node);
      // The predicate is the only thing a filter reads; it decides which ROWS
      // survive, never what is in the columns it passes on.
      if (filter.expression && reads_column(*filter.expression, in_pos)) { return step::reads(); }
      // The output mask is an explicit positional map, so a filter that folds a
      // projection into its gather is still transparent to everything it keeps.
      return step::to(std::visit(
        [&](auto const& mask) -> std::optional<std::size_t> {
          using T = std::decay_t<decltype(mask)>;
          if constexpr (std::is_same_v<T, op::passthrough>) {
            return in_pos;
          } else {
            for (std::size_t out = 0; out < mask.size(); ++out) {
              if (static_cast<std::size_t>(mask[out]) == in_pos) { return out; }
            }
            return std::nullopt;  // dropped here; nothing downstream can want it
          }
        },
        filter.output_columns));
    }

    case SiriusPhysicalOperatorType::PROJECTION: {
      auto const& projection = static_cast<op::sirius_physical_projection const&>(node);
      // A bare column reference MOVES the column. Anything else computes with
      // it, which is a read.
      std::optional<std::size_t> moved_to;
      for (std::size_t out = 0; out < projection.select_list.size(); ++out) {
        auto const& expr = projection.select_list[out];
        if (!expr) { continue; }
        if (auto const* ref = std::get_if<ast::reference>(&expr->v)) {
          if (ref->column_index == in_pos && !moved_to.has_value()) { moved_to = out; }
          continue;
        }
        if (reads_column(*expr, in_pos)) { return step::reads(); }
      }
      return step::to(moved_to);  // nullopt when the projection simply drops it
    }

    default:
      // Fail closed: an unmodelled shape is assumed to read everything. This
      // can only shorten a lifetime, so an operator missing from this switch
      // costs a deferral rather than permitting a wrong one.
      return step::reads();
  }
}

}  // namespace

std::optional<std::size_t> join_output_position(bool from_lhs,
                                                std::vector<int> const& lhs_projection,
                                                std::vector<int> const& rhs_projection,
                                                std::size_t in_position)
{
  auto const& own = from_lhs ? lhs_projection : rhs_projection;
  for (std::size_t i = 0; i < own.size(); ++i) {
    if (static_cast<std::size_t>(own[i]) == in_position) {
      // The output is lhs-then-rhs, so an rhs column carries the lhs's emitted
      // width as an offset — not the lhs's INPUT width, which is larger
      // whenever the join projects only part of its left side.
      return from_lhs ? i : lhs_projection.size() + i;
    }
  }
  return std::nullopt;
}

std::int64_t estimated_value_bytes(sirius::logical_type const& type)
{
  if (type.is_fixed_width()) { return static_cast<std::int64_t>(type.fixed_width_byte_size()); }
  // A variable-width column carries its bytes plus an offset. TPC-H's deferred
  // string columns run 15-72 bytes; 24 sits below most of them, so a bundle
  // that qualifies on this estimate would have qualified on the real widths.
  return 24;
}

std::vector<late_mat::defer_candidate> build_defer_candidates(
  sirius_physical_operator const& scan,
  std::vector<column_lifetime> const& lifetimes,
  std::vector<op::sirius_physical_operator const*>* out_readers)
{
  std::vector<late_mat::defer_candidate> candidates;
  // Slots are labelled by the order their reader is first seen, so a label is
  // stable within one analysis and readable in the census — the identity that
  // matters is "the same operator", not the operator's name.
  std::vector<op::sirius_physical_operator const*> readers;

  for (auto const& life : lifetimes) {
    if (life.first_reader == nullptr) { continue; }
    if (life.scan_output_position >= scan.types.size()) { continue; }
    // A join emits one scan row once per match, so materializing on its OUTPUT
    // gathers a row set larger than the scan produced -- by the join's fan-out,
    // which the per-row value model does not see. q20 rode 2 columns (40 B/row
    // over 6 crossings, comfortably past both floors) into a hash join and cost
    // 0.18 -> 3.33 s. A ride that ends where rows can multiply is refused until
    // the policy can price that fan-out; ports that only ever reduce rows (an
    // aggregate, a top-n) are unaffected, which is where q10's ride lands.
    if (life.first_reader->type == SiriusPhysicalOperatorType::HASH_JOIN) { continue; }

    auto slot = std::find(readers.begin(), readers.end(), life.first_reader);
    if (slot == readers.end()) {
      readers.push_back(life.first_reader);
      slot = std::prev(readers.end());
      late_mat::defer_candidate fresh;
      fresh.slot       = "slot" + std::to_string(readers.size() - 1);
      fresh.boundaries = life.port_crossings;
      candidates.push_back(std::move(fresh));
    }
    auto& candidate = candidates[static_cast<std::size_t>(slot - readers.begin())];
    candidate.columns.push_back(
      late_mat::defer_column{static_cast<std::uint32_t>(life.scan_output_position),
                             estimated_value_bytes(scan.types[life.scan_output_position])});
  }
  if (out_readers != nullptr) { *out_readers = std::move(readers); }
  return candidates;
}

std::vector<column_lifetime> analyze_column_lifetimes(sirius_physical_operator const& scan)
{
  std::vector<column_lifetime> lifetimes(scan.types.size());
  for (std::size_t col = 0; col < lifetimes.size(); ++col) {
    lifetimes[col].scan_output_position = col;
    lifetimes[col].position_at_reader   = col;
  }

  /// A column still travelling: where it sits now, and what the ride has done
  /// to it. Carrying the set through one walk — rather than walking the chain
  /// once per column — is what gives per-column state somewhere to live.
  struct live_column {
    std::size_t index;
    std::size_t position;
    bool nullified = false;
  };
  std::vector<live_column> live;
  live.reserve(lifetimes.size());
  for (std::size_t col = 0; col < lifetimes.size(); ++col) {
    live.push_back(live_column{col, col, false});
  }

  int crossings    = 0;
  auto const* from = static_cast<sirius_physical_operator const*>(&scan);
  for (auto const* node = scan.get_parent_op(); node != nullptr && !live.empty();
       from = node, node = node->get_parent_op()) {
    // Count what carrying actually COSTS. A pipeline sink writes its output to
    // a repository for the next pipeline to read, so leaving one is where the
    // deferred bytes would have been paid for; a filter or projection hands its
    // columns straight on within the pipeline and a wide column rides past it
    // for free. Counting operators instead would let a chain of projections
    // look like a ride worth deferring.
    if (from->is_sink()) { ++crossings; }
    std::vector<live_column> still_travelling;
    still_travelling.reserve(live.size());
    for (auto& col : live) {
      auto const moved = trace_through(*node, *from, col.position, col.nullified);
      if (!moved.moved_to.has_value()) {
        // Read here — or dropped here, which for a deferral is the same
        // answer: this is as far as the column travels.
        auto& life              = lifetimes[col.index];
        life.first_reader       = node;
        life.port_crossings     = crossings;
        life.position_at_reader = col.position;
        life.reader_input       = from;
        life.nullified_on_ride  = col.nullified;
        life.read_as_join_key   = moved.as_join_key;
        continue;
      }
      col.position = *moved.moved_to;
      still_travelling.push_back(col);
    }
    live = std::move(still_travelling);
  }

  // Whatever is still live reached the top of the plan unread.
  for (auto const& col : live) {
    auto& life              = lifetimes[col.index];
    life.port_crossings     = crossings;
    life.position_at_reader = col.position;
    life.reader_input       = from;
    life.nullified_on_ride  = col.nullified;
  }
  return lifetimes;
}

planned_deferral plan_deferral(sirius_physical_operator& scan, late_mat::defer_policy const& policy)
{
  // No env gate here: like the rest of the pass this only reports, and the one
  // caller that acts on it is gated. Keeping the analysis ungated is also what
  // lets it be tested without the gate set in the test binary's environment.
  planned_deferral planned;
  auto const lifetimes = analyze_column_lifetimes(scan);

  // Withdraw the columns an outer join could null BEFORE anything weighs them.
  // Deferring one is sound — a null rowid must materialize a null — but the
  // materializer produces no nulls yet, so the refusal belongs here, where it
  // costs a deferral, rather than at the far end where it would already be an
  // answer.
  std::vector<column_lifetime> weighable;
  weighable.reserve(lifetimes.size());
  for (auto const& life : lifetimes) {
    if (life.nullified_on_ride) {
      ++planned.nullable_columns_skipped;
      continue;
    }
    // A join key leaves candidacy outright — see column_lifetime's note: the
    // partition below the join has already hashed it by the time the port could
    // put its value back.
    if (life.read_as_join_key) {
      ++planned.join_keys_skipped;
      continue;
    }
    weighable.push_back(life);
  }

  std::vector<sirius_physical_operator const*> readers;
  auto const candidates = build_defer_candidates(scan, weighable, &readers);
  planned.census        = late_mat::choose_deferrals(candidates, policy);

  // One rowid rides, so one bundle installs. Widest wins, same rule the policy
  // arbitrates a shared slot by, and the losers are recorded rather than
  // dropped.
  std::optional<std::size_t> best;
  for (std::size_t i = 0; i < planned.census.size(); ++i) {
    if (!planned.census[i].installed()) { continue; }
    if (!best) {
      best = i;
      continue;
    }
    auto const loser = planned.census[i].net_value_bytes > planned.census[*best].net_value_bytes
                         ? std::exchange(*best, i)
                         : i;
    planned.census[loser].refusal = late_mat::defer_refusal::second_bundle;
  }
  if (!best || *best >= readers.size() || readers[*best] == nullptr) { return planned; }

  // The walk reads the plan, which is why it holds const pointers; installing
  // writes to it, and the caller handed us the mutable tree those pointers
  // address.
  planned.port            = const_cast<sirius_physical_operator*>(readers[*best]);
  planned.net_value_bytes = planned.census[*best].net_value_bytes;
  planned.boundaries      = planned.census[*best].boundaries;
  for (auto const& column : candidates[*best].columns) {
    auto const position = static_cast<std::size_t>(column.column_pos);
    planned.positions.push_back(position);
    planned.restored_types.push_back(scan.types[position]);
    planned.port_positions.push_back(lifetimes[position].position_at_reader);
  }
  planned.port_input = lifetimes[planned.positions.front()].reader_input;
  return planned;
}

bool install_deferral(sirius_physical_operator& scan,
                      sirius_physical_operator& port,
                      late_mat::defer_pair pair)
{
  if (!pair.valid()) { return false; }
  // A scan materializing its own deferral would defer nothing across nothing.
  if (&scan == &port) { return false; }
  // Neither half may be overwritten: the second install would substitute
  // against a schema the first one already rewrote.
  if (!scan._deferred_output.empty() || !port._port_directive.empty()) { return false; }
  scan._deferred_output = std::move(pair.scan);
  port._port_directive  = std::move(pair.port);
  late_mat::note_deferral_installed();
  return true;
}

}  // namespace sirius::planner
