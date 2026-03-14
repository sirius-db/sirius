# Strings

**Status**: USED
**Path**: `cudf/strings/`
**Headers we include**: `cudf/strings/strings_column_view.hpp`, `cudf/strings/contains.hpp`, `cudf/strings/find.hpp`, `cudf/strings/slice.hpp`, `cudf/strings/replace_re.hpp`, `cudf/strings/attributes.hpp`, `cudf/strings/regex/regex_program.hpp`

## Summary

Sirius uses GPU string operations for SQL string functions (LIKE, SUBSTRING, LENGTH, regex matching). The `strings_column_view` provides the interface to string column data, while individual operation headers provide functions for pattern matching, substring extraction, and string attributes.

## API Reference

### `cudf::strings_column_view`

**Header**: `cudf/strings/strings_column_view.hpp`
```cpp
class strings_column_view : public column_view {
    column_view offsets() const;   // Offset array (int32_t or int64_t)
    column_view chars() const;     // Character data
    size_type chars_size() const;  // Total bytes in character data
};
```

**Our usage**:
- `src/cuda/expression_executor/gpu_dispatch_string.cu:22` — Accessing string data for custom kernels
- `src/op/merge/gpu_merge_impl.cpp:28` — String column introspection for merge optimization

### `cudf::strings::contains_re`

**Header**: `cudf/strings/contains.hpp`
```cpp
std::unique_ptr<column> contains_re(strings_column_view const& input,
                                     regex_program const& prog, ...);
std::unique_ptr<column> like(strings_column_view const& input,
                              string_scalar const& pattern,
                              string_scalar const& escape_char = {}, ...);
std::unique_ptr<column> starts_with(strings_column_view const& input, string_scalar const& target, ...);
std::unique_ptr<column> ends_with(strings_column_view const& input, string_scalar const& target, ...);
```

**Our usage**:
- `src/expression_executor/specializations/gpu_execute_function.cpp` — SQL LIKE, regex matching, prefix/suffix checks

### `cudf::strings::find`

**Header**: `cudf/strings/find.hpp`
```cpp
std::unique_ptr<column> find(strings_column_view const& input, string_scalar const& target, ...);
```

**Our usage**:
- `src/expression_executor/specializations/gpu_execute_function.cpp` — POSITION/INSTR function

### `cudf::strings::slice_strings`

**Header**: `cudf/strings/slice.hpp`
```cpp
std::unique_ptr<column> slice_strings(strings_column_view const& input,
                                       numeric_scalar<size_type> const& start,
                                       numeric_scalar<size_type> const& stop, ...);
std::unique_ptr<column> slice_strings(strings_column_view const& input,
                                       column_view const& starts,
                                       column_view const& stops, ...);
```

**Our usage**:
- `src/expression_executor/specializations/gpu_execute_function.cpp` — SQL SUBSTRING function

### `cudf::strings::count_characters` / `cudf::strings::count_bytes`

**Header**: `cudf/strings/attributes.hpp`
```cpp
std::unique_ptr<column> count_characters(strings_column_view const& input, ...);
std::unique_ptr<column> count_bytes(strings_column_view const& input, ...);
```

**Our usage**:
- `src/expression_executor/specializations/gpu_execute_function.cpp` — SQL LENGTH/CHAR_LENGTH

### `cudf::strings::replace_re`

**Header**: `cudf/strings/replace_re.hpp`
```cpp
std::unique_ptr<column> replace_re(strings_column_view const& input,
                                    regex_program const& prog,
                                    string_scalar const& replacement, ...);
std::unique_ptr<column> replace_with_backrefs(strings_column_view const& input,
                                               regex_program const& prog,
                                               std::string_view replacement, ...);
```

**Our usage**:
- `src/expression_executor/specializations/gpu_execute_function.cpp` — SQL REGEXP_REPLACE

### `cudf::strings::regex_program`

**Header**: `cudf/strings/regex/regex_program.hpp`
```cpp
class regex_program {
    static std::unique_ptr<regex_program> create(std::string_view pattern, regex_flags flags = {}, ...);
};
```

**Our usage**:
- `src/expression_executor/specializations/gpu_execute_function.cpp:37` — Compiling regex patterns for LIKE/REGEXP operations

## APIs Available but Not Used

| API | Header | Brief Description |
|-----|--------|-------------------|
| `cudf::strings::capitalize()` | `capitalize.hpp` | Capitalize strings |
| `cudf::strings::to_lower()` / `to_upper()` | `case.hpp` | Case conversion |
| `cudf::strings::concatenate()` | `combine.hpp` | String concatenation |
| `cudf::strings::strip()` | `strip.hpp` | Trim whitespace (TRIM/LTRIM/RTRIM) |
| `cudf::strings::pad()` | `padding.hpp` | Pad strings (LPAD/RPAD) |
| `cudf::strings::split()` | `split/split.hpp` | Split strings |
| `cudf::strings::findall()` | `findall.hpp` | Find all regex matches |
| `cudf::strings::extract()` | `extract.hpp` | Regex group extraction |
| `cudf::strings::to_integers()` | `convert/convert_integers.hpp` | Parse integers from strings |
