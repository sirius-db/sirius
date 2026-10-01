# catalog

**Status**: USED
**Path**: `duckdb/src/include/duckdb/catalog/`
**Headers we include**:
- `duckdb/catalog/catalog.hpp`
- `duckdb/catalog/catalog_entry/table_catalog_entry.hpp`
- `duckdb/catalog/catalog_entry/table_function_catalog_entry.hpp`
- `duckdb/catalog/catalog_transaction.hpp`

## Summary

The `catalog` module manages database metadata — schemas, tables, functions, types. Sirius uses it to register extension functions in the catalog and to look up table/function metadata during query planning.

## API Reference

### Catalog

**Header**: `duckdb/catalog/catalog.hpp`
**Signature**:
```cpp
class Catalog {
public:
    static Catalog &GetSystemCatalog(DatabaseInstance &db);
    static Catalog &GetCatalog(AttachedDatabase &db);

    optional_ptr<CatalogEntry> CreateTableFunction(ClientContext &context, CreateTableFunctionInfo &info);
    optional_ptr<CatalogEntry> GetEntry(ClientContext &context, CatalogType type,
                                         const string &schema, const string &name,
                                         OnEntryNotFound if_not_found = OnEntryNotFound::THROW_EXCEPTION);
};
```

**Description**: The system catalog. Provides access to all database objects (tables, functions, schemas).

**Our usage**:
- `src/sirius_extension.cpp` — `CreateTableFunction()` to register `gpu_execution`, etc.
- `test/cpp/scan/test_parquet_scan_task.cpp` — Look up parquet scan function in catalog
- `test/cpp/integration/test_tpcds_plan_translation.cpp` — Register test functions

### TableCatalogEntry

**Header**: `duckdb/catalog/catalog_entry/table_catalog_entry.hpp`
**Signature**:
```cpp
class TableCatalogEntry : public StandardEntry {
public:
    const ColumnList &GetColumns() const;
    const vector<unique_ptr<Constraint>> &GetConstraints() const;
};
```

**Our usage**:
- `src/sirius_extension.cpp` — Access table metadata
- `test/cpp/scan/test_scan_executor.cpp` — Access table columns for scan tests

### TableFunctionCatalogEntry

**Header**: `duckdb/catalog/catalog_entry/table_function_catalog_entry.hpp`
**Signature**:
```cpp
class TableFunctionCatalogEntry : public FunctionEntry {
public:
    TableFunctionSet functions;
};
```

**Our usage**:
- `test/cpp/scan/test_parquet_scan_task.cpp` — Look up and access parquet scan function

### CatalogTransaction

**Header**: `duckdb/catalog/catalog_transaction.hpp`
**Signature**:
```cpp
class CatalogTransaction {
public:
    static CatalogTransaction GetSystemTransaction(DatabaseInstance &db);
};
```

**Our usage**:
- `test/cpp/scan/test_scan_executor.cpp` — Create system transactions for catalog access

## APIs Available but Not Used

| API | Header | Brief Description |
|-----|--------|-------------------|
| `SchemaCatalogEntry` | `catalog_entry/schema_catalog_entry.hpp` | Schema metadata (indirectly included) |
| `DefaultGenerator` | `default/` | Default catalog entries |
| `ViewCatalogEntry` | `catalog_entry/view_catalog_entry.hpp` | View definitions |
