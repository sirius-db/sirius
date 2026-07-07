# Alignment Utilities

**Status**: USED
**Path**: `rmm/aligned.hpp`
**Headers we include**: `<rmm/aligned.hpp>`

## Summary

Memory alignment helper functions and constants. Used for ensuring proper CUDA memory alignment.

## API Reference

### `rmm::CUDA_ALLOCATION_ALIGNMENT`

```cpp
static constexpr std::size_t CUDA_ALLOCATION_ALIGNMENT{256};
```

**Description**: Default alignment for CUDA allocations (256 bytes).

### `rmm::align_up()`

```cpp
std::size_t align_up(std::size_t value, std::size_t alignment) noexcept;
```

**Description**: Rounds `value` up to the nearest multiple of `alignment`.

**Our usage**:
- `src/gpu_buffer_manager.cpp:28` — Aligns buffer sizes for GPU memory allocations

### Other functions

| API | Brief Description |
|-----|-------------------|
| `align_down()` | Round down to alignment |
| `is_aligned()` | Check if value is aligned |
| `is_pow2()` | Check if power of 2 |
| `is_supported_alignment()` | Check if valid alignment |
| `is_pointer_aligned()` | Check pointer alignment |
