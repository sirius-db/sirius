# I/O (Parquet)

**Status**: USED
**Path**: `cudf/io/`
**Headers we include**: `cudf/io/parquet.hpp`, `cudf/io/datasource.hpp`, `cudf/io/experimental/hybrid_scan.hpp`, `cudf/io/parquet_schema.hpp`, `cudf/io/parquet_io_utils.hpp`, `cudf/io/text/byte_range_info.hpp`

## Summary

Sirius uses cuDF's I/O module exclusively for Parquet reading. The primary integration is through the experimental `hybrid_scan_reader` which enables GPU-accelerated Parquet decoding with CPU fallback. A custom `datasource` implementation provides prefetched data for I/O optimization.

## API Reference

### `cudf::io::datasource`

**Header**: `cudf/io/datasource.hpp`
```cpp
class datasource {
public:
    virtual ~datasource() = default;
    virtual std::unique_ptr<buffer> host_read(size_t offset, size_t size) = 0;
    virtual size_t host_read(size_t offset, size_t size, uint8_t* dst) = 0;
    virtual bool supports_device_read() const;
    virtual std::unique_ptr<device_buffer> device_read(size_t offset, size_t size, ...);
    virtual size_t size() const = 0;

    // Factory methods
    static std::unique_ptr<datasource> create(std::string const& filepath);
    static std::unique_ptr<datasource> create(host_buffer const& buffer);
};
```

**Our usage**:
- `src/include/op/scan/prefetched_data_source.hpp:21` — Custom `prefetched_data_source` subclass for pre-fetched Parquet data
- `src/op/scan/parquet_scan_task.cpp:42` — Creating datasource instances for Parquet files

### `cudf::io::parquet_reader_options`

**Header**: `cudf/io/parquet.hpp`
```cpp
class parquet_reader_options {
    static parquet_reader_options_builder builder(source_info src);
    // Configuration for column selection, row group selection, etc.
};

// Builder pattern
parquet_reader_options_builder& columns(std::vector<std::string> col_names);
parquet_reader_options_builder& row_groups(std::vector<std::vector<size_type>> row_groups);

// Read function
table_with_metadata read_parquet(parquet_reader_options const& options, ...);
```

**Our usage**:
- `src/op/scan/parquet_scan_task.cpp:44` — Configuring Parquet reader for scan tasks

### `cudf::io::experimental::hybrid_scan_reader` (Experimental)

**Header**: `cudf/io/experimental/hybrid_scan.hpp`

**Description**: GPU/CPU hybrid Parquet reader that can decode pages on either device. This is an experimental API that Sirius uses as its primary Parquet ingestion path for better performance.

**Our usage**:
- `src/include/data/host_parquet_representation.hpp:26` — Core data loading infrastructure
- `src/op/scan/parquet_scan_task.cpp:43` — Scan task implementation
- `test/cpp/data/test_host_parquet_representation.cpp:38` — Integration tests

### `cudf::io::parquet_schema`

**Header**: `cudf/io/parquet_schema.hpp`

**Description**: Parquet schema introspection for column type mapping.

**Our usage**:
- `src/op/scan/parquet_scan_task.cpp:45` — Schema inspection during scan setup

### `cudf::io::text::byte_range_info`

**Header**: `cudf/io/text/byte_range_info.hpp`
```cpp
struct byte_range_info {
    size_t offset;
    size_t size;
};
```

**Our usage**:
- `src/include/op/scan/cached_ranges.hpp:19` — Byte range tracking for cached I/O

## APIs Available but Not Used

| API | Header | Brief Description |
|-----|--------|-------------------|
| `cudf::io::write_parquet()` | `parquet.hpp` | Write tables to Parquet files |
| `cudf::io::read_csv()` | `csv.hpp` | CSV reader |
| `cudf::io::read_json()` | `json.hpp` | JSON reader |
| `cudf::io::read_orc()` | `orc.hpp` | ORC reader |
| `cudf::io::chunked_parquet_reader` | `parquet.hpp` | Chunked Parquet reading |
| `cudf::io::data_sink` | `data_sink.hpp` | Output interface for writers |
