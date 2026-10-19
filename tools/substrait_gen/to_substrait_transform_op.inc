// Spliced into DuckDBToSubstrait::TransformOp's switch in the generator's
// patched copy of to_substrait.cpp (see CMakeLists.txt), replacing the
// LOGICAL_COMPARISON_JOIN case. It only emits relations that the unpatched
// from-substrait reader in libsirius already accepts: ReferenceRel to a
// plan-level relation, AggregateRel without measures, and INNER / LEFT / RIGHT
// / SEMI / SINGLE joins with is_not_distinct_from conditions.
//
// ANTI / RIGHT_ANTI become an outer join against the other side plus an
// IS NULL filter on a marker column. That join fans out (every kept row meets
// all its matches before the filter drops them), which is pessimistic for
// anti-join heavy queries such as Q21/Q22 on the GPU. Follow-up: once
// libsirius's reader handles JOIN_TYPE_LEFT_ANTI (upstream duckdb-substrait's
// reader already does), emit LEFT_ANTI directly.
	case LogicalOperatorType::LOGICAL_MATERIALIZED_CTE: {
		// A CTE below the root (DuckDB's common_subplan optimizer creates
		// these): its definition becomes a plan-level relation, read through
		// TransformCTERef's ReferenceRel. Transform before registering, so any
		// relation the definition adds itself keeps the ordinals aligned.
		auto &cte = dop.Cast<LogicalMaterializedCTE>();
		auto definition = TransformOp(*dop.children[0]);
		cte_indices.push_back(cte.table_index);
		plan.add_relations()->set_allocated_rel(definition);
		return TransformOp(*dop.children[1]);
	}
	case LogicalOperatorType::LOGICAL_DELIM_GET: {
		// The duplicate-eliminated input of the innermost DELIM_JOIN: SELECT
		// DISTINCT <duplicate_eliminated_columns> FROM <shared input>.
		if (sirius_substrait_gen_delim_scopes.empty()) {
			throw InternalException("DELIM_GET outside the delim side of a DELIM_JOIN");
		}
		auto &scope = sirius_substrait_gen_delim_scopes.back();
		auto &columns = scope.join->duplicate_eliminated_columns;
		if (columns.size() != dop.types.size()) {
			throw InternalException("DELIM_GET has %llu columns, its DELIM_JOIN eliminates %llu", dop.types.size(),
			                        columns.size());
		}
		auto res = new substrait::Rel();
		auto saggr = res->mutable_aggregate();
		saggr->mutable_input()->mutable_reference()->set_subtree_ordinal(scope.shared_ordinal);
		auto sgrp = saggr->add_groupings();
		for (idx_t i = 0; i < columns.size(); i++) {
			TransformExpr(*columns[i], *saggr->add_grouping_expressions());
			sgrp->add_expression_references(i);
		}
		return res;
	}
	case LogicalOperatorType::LOGICAL_COMPARISON_JOIN:
	case LogicalOperatorType::LOGICAL_DELIM_JOIN: {
		auto &djoin = dop.Cast<LogicalComparisonJoin>();

		// DELIM_JOIN = an ordinary join whose delim side reads DISTINCT
		// duplicate_eliminated_columns of the other side. That other side is
		// emitted once as a plan-level relation and read by reference both
		// here (through a stand-in CTE ref) and by every DELIM_GET.
		const bool is_delim = dop.type == LogicalOperatorType::LOGICAL_DELIM_JOIN;
		const idx_t shared_idx = djoin.delim_flipped ? 1 : 0;
		unique_ptr<LogicalOperator> shared_op;
		if (is_delim) {
			auto shared = TransformOp(*dop.children[shared_idx]);
			const auto ordinal = static_cast<int32_t>(plan.relations_size());
			const auto index = sirius_substrait_gen_next_shared_index++;
			cte_indices.push_back(index);
			plan.add_relations()->set_allocated_rel(shared);
			shared_op = std::move(dop.children[shared_idx]);
			auto &shared_types = shared_op->types;
			auto ref = make_uniq<LogicalCTERef>(0, index, shared_types,
			                                    vector<string>(shared_types.size(), "shared"));
			ref->types = shared_types;
			dop.children[shared_idx] = std::move(ref);
			sirius_substrait_gen_delim_scopes.push_back({ordinal, &djoin});
		}
		auto restore_delim = SiriusSubstraitGenOnExit([&]() {
			if (shared_op) {
				dop.children[shared_idx] = std::move(shared_op);
				sirius_substrait_gen_delim_scopes.pop_back();
			}
		});

		if (djoin.join_type != JoinType::ANTI && djoin.join_type != JoinType::RIGHT_ANTI) {
			return TransformComparisonJoin(dop);
		}

		// Neither direction of duckdb-substrait handles ANTI joins. Emit the
		// equivalent outer join against the other side plus a non-null marker,
		// keep the rows whose marker is NULL (no condition evaluated to true),
		// and project the kept side's columns.
		const idx_t keep = djoin.join_type == JoinType::RIGHT_ANTI ? 1 : 0;
		const idx_t other = 1 - keep;
		auto saved_type = djoin.join_type;
		auto saved_right_map = djoin.right_projection_map;
		if (djoin.left_projection_map.empty()) {
			for (idx_t i = 0; i < dop.children[0]->types.size(); i++) {
				djoin.left_projection_map.push_back(i);
			}
		}
		if (djoin.right_projection_map.empty()) {
			for (idx_t i = 0; i < dop.children[1]->types.size(); i++) {
				djoin.right_projection_map.push_back(i);
			}
		}
		const auto kept = keep == 0 ? djoin.left_projection_map : djoin.right_projection_map;

		auto other_op = std::move(dop.children[other]);
		const idx_t marker_idx = other_op->types.size();
		vector<unique_ptr<Expression>> marker_exprs;
		for (idx_t i = 0; i < marker_idx; i++) {
			marker_exprs.push_back(make_uniq<BoundReferenceExpression>(other_op->types[i], i));
		}
		marker_exprs.push_back(make_uniq<BoundConstantExpression>(Value::BOOLEAN(true)));
		auto marker = make_uniq<LogicalProjection>(0, std::move(marker_exprs));
		marker->types = other_op->types;
		marker->types.push_back(LogicalType::BOOLEAN);
		marker->children.push_back(std::move(other_op));
		dop.children[other] = std::move(marker);
		(keep == 0 ? djoin.right_projection_map : djoin.left_projection_map) = vector<idx_t> {marker_idx};
		djoin.join_type = keep == 0 ? JoinType::LEFT : JoinType::RIGHT;
		// Restore the children and join type, but leave the projection maps
		// describing the rewritten node's real output (the kept columns), as
		// TransformComparisonJoin leaves its maps filled: a parent
		// COMPARISON_JOIN derives this node's column count from them
		// (left_projection_map.size() for ANTI, the sum of both maps for
		// RIGHT_ANTI), so ANTI keeps left = kept, and RIGHT_ANTI keeps
		// right = kept with an empty left map. A DELIM_JOIN child is counted by
		// its types instead, which never change. Either way, a second transform
		// of this node rewrites it identically.
		auto restore_anti = SiriusSubstraitGenOnExit([&]() {
			if (dop.children[other] && dop.children[other]->type == LogicalOperatorType::LOGICAL_PROJECTION &&
			    !dop.children[other]->children.empty()) {
				auto original = std::move(dop.children[other]->children[0]);
				dop.children[other] = std::move(original);
			}
			djoin.join_type = saved_type;
			if (keep == 0) {
				djoin.left_projection_map = kept;
				djoin.right_projection_map = saved_right_map;
			} else {
				djoin.left_projection_map.clear();
				djoin.right_projection_map = kept;
			}
		});

		auto joined = TransformComparisonJoin(dop);
		// The join emits the left side's mapped columns, then the right side's.
		const idx_t joined_count = kept.size() + 1;
		const idx_t marker_pos = keep == 0 ? kept.size() : 0;
		const idx_t first_kept = keep == 0 ? 0 : 1;

		auto filter = new substrait::Rel();
		filter->mutable_filter()->set_allocated_input(joined);
		BoundOperatorExpression is_null(ExpressionType::OPERATOR_IS_NULL, LogicalType::BOOLEAN);
		is_null.children.push_back(make_uniq<BoundReferenceExpression>(LogicalType::BOOLEAN, marker_pos));
		TransformExpr(is_null, *filter->mutable_filter()->mutable_condition());

		auto projection = new substrait::Rel();
		auto sproj = projection->mutable_project();
		sproj->set_allocated_input(filter);
		vector<int32_t> output_mapping;
		for (idx_t i = 0; i < kept.size(); i++) {
			CreateFieldRef(sproj->add_expressions(), first_kept + i);
			// Project expressions are numbered after the joined_count input fields.
			output_mapping.push_back(static_cast<int32_t>(joined_count + i));
		}
		sproj->set_allocated_common(CreateOutputMapping(output_mapping));
		return projection;
	}
