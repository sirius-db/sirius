# Datetime

**Status**: USED
**Path**: `cudf/datetime.hpp`
**Headers we include**: `cudf/datetime.hpp`

## Summary

Used for SQL date/time functions like EXTRACT(YEAR FROM ...), date arithmetic.

## API Reference

### Date Extraction Functions

**Header**: `cudf/datetime.hpp`
```cpp
std::unique_ptr<column> extract_year(column_view const& timestamps, ...);
std::unique_ptr<column> extract_month(column_view const& timestamps, ...);
std::unique_ptr<column> extract_day(column_view const& timestamps, ...);
```

**Our usage**:
- `src/expression_executor/specializations/gpu_execute_function.cpp:32` — SQL EXTRACT and date_part functions

## APIs Available but Not Used

| API | Header | Brief Description |
|-----|--------|-------------------|
| `cudf::extract_hour/minute/second()` | `datetime.hpp` | Time component extraction |
| `cudf::add_calendrical_months()` | `datetime.hpp` | Date arithmetic |
| `cudf::last_day_of_month()` | `datetime.hpp` | Month-end calculation |
