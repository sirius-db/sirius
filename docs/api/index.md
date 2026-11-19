# Sirius C++ API

Sirius is a GPU-native analytics engine. This reference documents its public C++
API.

## Getting started

New fallible public C++ operations report errors through
`std::expected<T, sirius::Error>`. Factory methods return allocation failures as
`ErrorCode::allocation_failure`; the default builder constructor may throw.
Constructing an argument, such as a filesystem path, can also throw before entering a factory.
The public API therefore requires C++23 and standard-library support for
`std::expected`.

- @ref sirius::Context "Engine context" owns an initialized engine. Only one active context per process is supported.
- @ref sirius::ContextConfigBuilder "Configuration builder" assembles settings from defaults or YAML.
- @ref sirius::ContextConfig "Context configuration" holds an immutable, validated configuration.
- @ref sirius::Plan "Physical plans" describe typed scans and filters without creating a context.
- @ref sirius::Error "Errors" describe failures returned by the API.

The API is under active development. These pages describe the headers on the
repository's default branch; they do not promise ABI stability.

## Construct a physical plan {#cpp_plan_construction}

This example describes a Parquet scan and a filter on its first column. The caller supplies
the file schema. Construction does not open the file or initialize a GPU.

```cpp
#include <sirius/plan/plan.hpp>

std::expected<sirius::Plan, sirius::Error> make_plan()
{
  using namespace sirius;
  Field fields[] = {{"id", DataType::int64, false}};
  auto schema = Schema::create(fields);
  if (!schema) return std::unexpected(schema.error());
  auto scan = PlanNode::scan_parquet("data.parquet", *schema);
  if (!scan) return std::unexpected(scan.error());
  auto id = scan->column(0);
  if (!id) return std::unexpected(id.error());
  auto ten = Expression::literal_int64(10);
  if (!ten) return std::unexpected(ten.error());
  auto predicate = id->compare(Comparison::greater, *ten);
  if (!predicate) return std::unexpected(predicate.error());
  auto filtered = scan->filter(*predicate);
  if (!filtered) return std::unexpected(filtered.error());
  return Plan::from_root(*filtered);
}
```

## More documentation

- [Build and development guide](https://github.com/sirius-db/sirius/blob/main/docs/DEVELOPMENT.md)
- [Engine architecture](https://github.com/sirius-db/sirius/tree/main/docs/super-sirius)
- [Sirius on GitHub](https://github.com/sirius-db/sirius)

## Binary compatibility

The C++ configuration, context, and plan classes are header-only wrappers over the public C ABI.
Their standard-library objects are created and destroyed by the application's
C++ toolchain. Link `sirius::sirius` (or `sirius::sirius_static`) and enable C++23
on your target:

```cmake
target_link_libraries(my_app PRIVATE sirius::sirius)
target_compile_features(my_app PRIVATE cxx_std_23)
```
