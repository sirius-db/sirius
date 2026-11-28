target_sources(
  sirius_objects
  PRIVATE gds/gds_ioctx.cpp
          gds/gds_stats.cpp
          parquet_helpers.cpp
          path_utils.cpp
          scoped_object_store_configs.cpp
          rest/s3/endpoint.cpp
          s3/duckdb_secret_config.cpp
          s3/scope_policy.cpp
          s3/sirius_httpfs.cpp)
