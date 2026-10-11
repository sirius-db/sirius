target_sources(
  sirius_objects
  PRIVATE aggregate_id.cpp
          ast/constant_range.cpp
          ast/node.cpp
          ast/utils.cpp
          decimal_divide_function.cpp
          from_duckdb.cpp
          function_id.cpp
          join_condition.cpp
          value.cpp)
