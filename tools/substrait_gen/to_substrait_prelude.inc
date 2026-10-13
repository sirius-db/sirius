// Prepended to the generator's patched copy of duckdb-substrait's
// to_substrait.cpp (see CMakeLists.txt); never compiled into libsirius.
#include "duckdb/function/table_function.hpp"
#include "duckdb/planner/operator/logical_comparison_join.hpp"
#include "duckdb/planner/operator/logical_cteref.hpp"

#include <cstdint>
#include <vector>

namespace duckdb {

// DuckDB 1.5's table_scan only sets statistics_extended; to_substrait.cpp calls
// TableFunction::statistics, which would be a null function pointer.
static unique_ptr<BaseStatistics> SiriusSubstraitGenStatistics(ClientContext &context, const TableFunction &function,
                                                               const FunctionData *bind_data, idx_t column) {
	if (function.statistics_extended) {
		TableFunctionGetStatisticsInput input(bind_data, ColumnIndex(column));
		return function.statistics_extended(context, input);
	}
	if (function.statistics) {
		return function.statistics(context, bind_data, column);
	}
	return nullptr;
}

// A duplicate-eliminated join whose delim side is being transformed: the
// plan-level relation holding its (shared) duplicate-eliminated input and the
// join that owns the duplicate_eliminated_columns. DELIM_GETs resolve to the
// innermost one, as in DuckDB's decorrelation.
struct SiriusSubstraitGenDelimScope {
	int32_t shared_ordinal;
	const LogicalComparisonJoin *join;
};
static std::vector<SiriusSubstraitGenDelimScope> sirius_substrait_gen_delim_scopes;

// cte_indices entries for shared relations that are not DuckDB CTEs; far above
// any real CTE table index, so TransformCTERef never confuses the two.
static uint32_t sirius_substrait_gen_next_shared_index = 0x80000000u;

// Restores a temporarily replaced operator child (and anything else) on scope
// exit, including when the transformation throws.
template <class F>
struct SiriusSubstraitGenRestore {
	F restore;
	~SiriusSubstraitGenRestore() {
		restore();
	}
};
template <class F>
static SiriusSubstraitGenRestore<F> SiriusSubstraitGenOnExit(F f) {
	return SiriusSubstraitGenRestore<F> {f};
}

} // namespace duckdb
