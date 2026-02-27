# Stream Check

## 🎯 Primary Goal

Create a preloadable library that intercepts `cudf::get_default_stream()` calls to detect and prevent unintended default stream usage in GPU applications.

## Usage

### In Code

Include the header and use the API:

```cpp
#include <stream_check.hpp>

// Enable logging on default stream access for this thread
enable_log_on_default_stream();

// Your code here - any call to cudf::get_default_stream() will be logged

// Disable logging
disable_log_on_default_stream();
```

### With LD_PRELOAD

Use the helper script:

```bash
./preload_stream_check.sh your_application [args...]
```

Or manually:

```bash
LD_PRELOAD=/path/to/libstream_check.so your_application [args...]
```

### Example Test Program

```cpp
#include <stream_check.hpp>
#include <cudf/types.hpp>
#include <iostream>

int main() {
    // This works fine - returns rmm default stream (no logging)
    auto stream1 = cudf::get_default_stream();
    std::cout << "Got default stream (normal): " << stream1.value() << std::endl;

    // Enable logging
    enable_log_on_default_stream();

    // This will be logged to default_stream_traces.log
    auto stream2 = cudf::get_default_stream();
    std::cout << "Got default stream (logged): " << stream2.value() << std::endl;

    // Disable logging
    disable_log_on_default_stream();

    // This works again without logging
    auto stream3 = cudf::get_default_stream();
    std::cout << "Got default stream (normal): " << stream3.value() << std::endl;

    return 0;
}
```


### Component Diagram

```
┌─────────────────────────────────────────────────┐
│                Application                       │
│  (e.g., sirius_unittest, custom tests)          │
└─────────────────┬───────────────────────────────┘
                  │
                  │ calls cudf::get_default_stream()
                  │
                  ▼
┌─────────────────────────────────────────────────┐
│         libstream_check.so (LD_PRELOAD)         │
│  ┌───────────────────────────────────────────┐  │
│  │ cudf::get_default_stream() {              │  │
│  │   if (g_throw_on_default_stream) {        │  │
│  │     throw runtime_error(...);             │  │
│  │   }                                        │  │
│  │   return rmm::cuda_stream_view{};         │  │
│  │ }                                          │  │
│  └───────────────────────────────────────────┘  │
│                                                  │
│  thread_local atomic<bool>                      │
│  g_throw_on_default_stream{false}               │
└─────────────────────────────────────────────────┘
                  │
                  │ symbol override
                  │
┌─────────────────▼───────────────────────────────┐
│           libcudf.so (original)                 │
│  cudf::get_default_stream() { ... }             │
│  (not called when libstream_check is preloaded) │
└─────────────────────────────────────────────────┘
```

### Data Flow

```
Thread A                         Thread B
   │                               │
   ├─ enable_log_on_default_stream()│
   │  (sets A's flag = true)       │
   │                               │
   ├─ cudf::get_default_stream()   │
   │  └─> logs stack trace         │
   │                               ├─ cudf::get_default_stream()
   │                               │  └─> returns stream (no log)
   │                               │      (B's flag = false)
   ├─ disable_log_on_default_stream()
   │  (sets A's flag = false)      │
   │                               │
   ├─ cudf::get_default_stream()   │
   │  └─> returns stream (no log)  │
```

## 🔧 Technical Details

### Key Technologies

- **C++20**: Thread-local storage, atomic operations
- **CUDA/cuDF**: Stream management API
- **RMM**: RAPIDS Memory Manager for default stream
- **CMake**: Build system integration
- **LD_PRELOAD**: Dynamic symbol interception

### Symbol Interception Mechanism

The library uses the Linux dynamic linker's symbol resolution order:

1. `LD_PRELOAD` libraries are searched first
2. Our `cudf::get_default_stream()` is found
3. Original libcudf symbol is shadowed
4. No source code changes needed in application

### Thread Safety

- ✅ **Thread-safe**: Each thread has independent state
- ✅ **Atomic operations**: Lock-free reads/writes
- ✅ **No synchronization**: No mutex overhead
- ✅ **Concurrent safe**: Multiple threads can call simultaneously
