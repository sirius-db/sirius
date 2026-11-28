// Copyright 2026, Sirius Contributors. SPDX-License-Identifier: Apache-2.0
#pragma once

#include "config/block.hpp"
#include "config/option.hpp"
#include "output/parquet_spec.hpp"

#include <string>

namespace sirius::config {

/// `sirius.output.parquet`: files GPU query results are exported to with COPY TO.
struct parquet_output_options : block_base<parquet_output_options> {
  /// Case-insensitive; `none` and `lz4_raw` are accepted as `uncompressed` and `lz4`.
  config_option<output::parquet_compression> compression =
    option<output::parquet_compression>("compression")
      .parser([](const std::string& text, output::parquet_compression& out) -> parse_status {
        out = output::parse_compression(text);
        return {};
      })
      .writer([](output::parquet_compression codec) { return output::compression_name(codec); })
      .fallback(output::parquet_compression::snappy)
      .describe("Parquet codec for COPY TO exports")
      .build();

  auto fields() { return field_list(compression); }
};

/// `sirius.output`: query result export.
struct output_options : block_base<output_options> {
  parquet_output_options parquet;

  auto fields() { return field_list(sub("parquet", parquet)); }
};

}  // namespace sirius::config
