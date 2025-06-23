#pragma once

namespace duckdb {

enum class OrderByType {
	ASCENDING,
	DESCENDING
};

enum class AggregationType {
	SUM,
	MIN,
	MAX,
	COUNT,
    COUNT_STAR,
	AVERAGE,
	FIRST,
	COUNT_DISTINCT
};

}