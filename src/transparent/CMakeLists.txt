target_sources(
  sirius_objects
  PRIVATE plan_source_policy.cpp
          read_view_registry.cpp
          replay_admission.cpp
          connection_provenance.cpp
          join_dimension_after_aggregate.cpp
          narrow_outer_join_counts.cpp
          physical_sirius_execution.cpp
          parquet_footer_statistics.cpp
          push_narrowing_expressions.cpp
          sirius_optimizer_extension.cpp)
