target_sources(
  sirius_objects
  PRIVATE aggregate_id.cpp
          ast/constant_range.cpp
          ast/node.cpp
          ast/utils.cpp
          from_duckdb.cpp
          function_id.cpp
          gpu_regexp_replace.cpp
          join_condition.cpp
          value.cpp)
