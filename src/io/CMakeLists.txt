target_sources(
  sirius_objects
  PRIVATE parquet_helpers.cpp path_utils.cpp scoped_object_store_configs.cpp
          s3/duckdb_secret_config.cpp s3/sirius_httpfs.cpp)
