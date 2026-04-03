# Fixed Point

**Status**: USED
**Path**: `cudf/fixed_point/`
**Headers we include**: `cudf/fixed_point/fixed_point.hpp`

## Summary

Provides exact decimal arithmetic types (DECIMAL32, DECIMAL64, DECIMAL128) for SQL DECIMAL columns. Sirius uses these for financial/precision-sensitive computations.

## API Reference

### `cudf::numeric::fixed_point<T>`

**Header**: `cudf/fixed_point/fixed_point.hpp`
```cpp
template <typename Rep, typename Radix>
class fixed_point {
    fixed_point(Rep value, scale_type scale);
    Rep value() const;
    scale_type scale() const;
};

using decimal32 = fixed_point<int32_t, numeric::Radix::BASE_10>;
using decimal64 = fixed_point<int64_t, numeric::Radix::BASE_10>;
using decimal128 = fixed_point<__int128_t, numeric::Radix::BASE_10>;

enum class scale_type : int32_t {};
```

**Our usage**:
- `src/op/sirius_physical_ungrouped_aggregate.cpp:29` — DECIMAL result extraction from scalars
- `src/include/expression_executor/gpu_expression_translator.hpp:38` — DECIMAL type handling in expressions
- `test/cpp/utils/data_utils.hpp:22` — Creating test DECIMAL columns
