# Building and Testing

This guide covers how to build Sirius from source, run tests, and verify your installation.

## Prerequisites

### Hardware
- **GPU**: NVIDIA GPU with compute capability 7.0+ (Volta or newer)
  - Recommended: RTX 3090, A100, H100
  - Minimum: GTX 1080, RTX 2060
- **GPU Memory**: 8GB minimum, 16GB+ recommended
- **System RAM**: 32GB+ recommended for building

### Software
- **OS**: Linux (Ubuntu 20.04+, RHEL 8+, or compatible)
- **CUDA**: 11.5 or higher
  - Check: `nvcc --version`
- **GCC**: 9.x or 10.x (for CUDA compatibility)
  - Check: `gcc --version`
- **CMake**: 3.20 or higher
  - Check: `cmake --version`
- **Git**: For cloning repositories
- **Python**: 3.8+ (for Pixi environment)

---

## Build Methods

### Method 1: Using Pixi (Recommended)

Pixi is a package manager that handles all dependencies automatically.

#### Install Pixi

```bash
curl -fsSL https://pixi.sh/install.sh | bash
```

#### Build Sirius

```bash
cd /home/roaramburu/coding/sirius

# Install dependencies and build
pixi run build

# This will:
# - Install CUDA toolkit
# - Install cuDF and RAPIDS dependencies
# - Install DuckDB
# - Build Sirius extension
```

#### Run Tests

```bash
# Run all tests
pixi run test

# Run specific test suites
pixi run test_unit          # C++ unit tests
pixi run test_sql           # SQL integration tests
pixi run test_performance   # Performance tests
```

---

### Method 2: Manual Build

For manual dependency management or custom configurations.

#### Install Dependencies

**CUDA Toolkit**:
```bash
# Ubuntu/Debian
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt-get update
sudo apt-get install cuda-11-8  # or newer
```

**cuDF / RAPIDS** (via conda):
```bash
conda install -c rapidsai -c conda-forge \
    cudf=23.10 \
    rmm=23.10 \
    cuda-version=11.8
```

**DuckDB**:
```bash
# DuckDB is typically built with Sirius
# Or install from package:
pip install duckdb
```

#### Configure Build

```bash
cd /home/roaramburu/coding/sirius
mkdir -p build
cd build

# Configure with CMake
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=80 \
  -DBUILD_TESTS=ON \
  -DBUILD_BENCHMARKS=ON
```

**CUDA Architecture Notes**:
| GPU Family | Compute Capability | CMake Value |
|------------|-------------------|-------------|
| Volta (V100) | 7.0 | 70 |
| Turing (RTX 20xx) | 7.5 | 75 |
| Ampere (A100, RTX 30xx) | 8.0 | 80 |
| Ada Lovelace (RTX 40xx) | 8.9 | 89 |
| Hopper (H100) | 9.0 | 90 |

**Find your GPU's compute capability**:
```bash
nvidia-smi --query-gpu=compute_cap --format=csv
```

#### Build

```bash
# Build with all CPU cores
make -j$(nproc)

# Or specify number of jobs
make -j8
```

#### Install (Optional)

```bash
sudo make install

# This installs to /usr/local/lib/duckdb/ by default
```

---

## Build Configuration Options

### CMake Options

```bash
cmake .. \
  -DCMAKE_BUILD_TYPE=<Debug|Release|RelWithDebInfo> \
  -DCMAKE_CUDA_ARCHITECTURES=<70|75|80|89|90> \
  -DBUILD_TESTS=<ON|OFF> \
  -DBUILD_BENCHMARKS=<ON|OFF> \
  -DBUILD_EXAMPLES=<ON|OFF> \
  -DCMAKE_INSTALL_PREFIX=/custom/path
```

### Build Types

**Release** (default, recommended):
- Full optimizations (-O3)
- No debug symbols
- Best performance

**Debug**:
- No optimizations (-O0)
- Full debug symbols (-g)
- Assertions enabled
- Use for development and debugging

**RelWithDebInfo**:
- Optimizations (-O2)
- Debug symbols (-g)
- Good compromise for profiling

---

## Testing

### Test Structure

```
sirius/
├── test/
│   ├── cpp/                    # C++ unit tests (Catch2)
│   │   ├── operator/          # Operator tests
│   │   ├── pipeline/          # Pipeline tests
│   │   ├── memory/            # Memory management tests
│   │   └── planner/           # Planner tests
│   └── sql/                    # SQL integration tests
│       ├── operators/         # Operator-specific SQL tests
│       ├── tpch/              # TPC-H benchmark queries
│       └── correctness/       # Correctness tests
```

### Running C++ Unit Tests

```bash
cd /home/roaramburu/coding/sirius/build

# Run all unit tests
./test/cpp/sirius_tests

# Run specific test suite
./test/cpp/sirius_tests "operator_*"

# Run specific test case
./test/cpp/sirius_tests "operator_filter"

# Verbose output
./test/cpp/sirius_tests -s
```

**Example Output**:
```
===============================================================================
All tests passed (127 assertions in 23 test cases)
```

### Running SQL Integration Tests

```bash
cd /home/roaramburu/coding/sirius

# Run all SQL tests
pixi run test_sql

# Run specific test file
duckdb < test/sql/operators/test_filter.sql

# Run with error checking
duckdb -no-stdin < test/sql/operators/test_filter.sql
```

**SQL Test Format**:
```sql
-- test/sql/operators/test_filter.sql

-- Load extension
LOAD 'sirius';

-- Test 1: Basic filter
SELECT * FROM gpu_execution('
    SELECT * FROM range(1000) WHERE range > 500
');
-- Expected: 499 rows

-- Test 2: Complex predicate
SELECT * FROM gpu_execution('
    SELECT * FROM generate_series(1, 100) t(x)
    WHERE x % 2 = 0 AND x > 50
');
-- Expected: 25 rows
```

### Running Performance Tests

```bash
cd /home/roaramburu/coding/sirius

# Run TPC-H benchmark
pixi run benchmark_tpch --scale=1

# Run specific query
pixi run benchmark_tpch --scale=1 --query=1

# Run with profiling
pixi run benchmark_tpch --scale=1 --profile
```

---

## Verifying Installation

### Basic Verification

```bash
# Start DuckDB with Sirius
pixi run duckdb

# Or if installed system-wide
duckdb
```

```sql
-- Load extension
LOAD 'sirius';

-- Verify extension loaded
SELECT * FROM duckdb_extensions()
WHERE extension_name = 'sirius';

-- Test basic query
SELECT * FROM gpu_execution('SELECT 42 as answer');
```

**Expected Output**:
```
┌────────┐
│ answer │
│ int32  │
├────────┤
│     42 │
└────────┘
```

### GPU Verification

```sql
-- Test GPU memory allocation
SELECT * FROM gpu_execution('
    SELECT COUNT(*) as count
    FROM range(10000000)
');
```

**Expected**: Should complete in < 1 second on modern GPU.

### Feature Verification

```sql
-- Test aggregation
SELECT * FROM gpu_execution('
    SELECT
        range / 100 as bucket,
        COUNT(*) as count,
        SUM(range) as sum
    FROM range(1000)
    GROUP BY bucket
');

-- Test join
CREATE TABLE t1 AS SELECT range as id, range * 2 as val FROM range(1000);
CREATE TABLE t2 AS SELECT range as id, range * 3 as val FROM range(1000);

SELECT * FROM gpu_execution('
    SELECT t1.id, t1.val, t2.val
    FROM t1 JOIN t2 ON t1.id = t2.id
    LIMIT 10
');
```

---

## Common Build Issues

### Issue 1: CUDA Not Found

**Error**:
```
CMake Error: CUDA not found
```

**Solution**:
```bash
# Set CUDA path
export CUDA_HOME=/usr/local/cuda-11.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Re-run CMake
cmake ..
```

### Issue 2: Incompatible GCC Version

**Error**:
```
nvcc fatal: Unsupported gpu architecture 'compute_89'
```

**Solution**:
```bash
# Use compatible GCC
sudo apt install gcc-10 g++-10

# Set as default
export CC=gcc-10
export CXX=g++-10

# Re-run CMake
cmake ..
```

### Issue 3: cuDF Not Found

**Error**:
```
Could not find cuDF
```

**Solution**:
```bash
# Install via conda
conda install -c rapidsai -c conda-forge cudf=23.10

# Or use Pixi (handles this automatically)
pixi run build
```

### Issue 4: Out of Memory During Build

**Error**:
```
c++: fatal error: Killed signal terminated program cc1plus
```

**Solution**:
```bash
# Reduce parallel jobs
make -j4  # Instead of -j$(nproc)

# Or add swap space
sudo fallocate -l 8G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

### Issue 5: Wrong CUDA Architecture

**Error**:
```
no kernel image is available for execution on the device
```

**Solution**:
```bash
# Check your GPU's compute capability
nvidia-smi --query-gpu=compute_cap --format=csv

# Rebuild with correct architecture
cd build
cmake .. -DCMAKE_CUDA_ARCHITECTURES=80  # Match your GPU
make -j$(nproc)
```

### Issue 6: DuckDB Version Mismatch

**Error**:
```
Incompatible DuckDB version
```

**Solution**:
```bash
# Update DuckDB
git submodule update --init --recursive

# Clean rebuild
rm -rf build
mkdir build && cd build
cmake .. && make -j$(nproc)
```

---

## Incremental Builds

### After Code Changes

```bash
cd /home/roaramburu/coding/sirius/build

# Incremental build (only changed files)
make -j$(nproc)
```

### After CMake Changes

```bash
cd /home/roaramburu/coding/sirius/build

# Re-run CMake
cmake ..

# Rebuild
make -j$(nproc)
```

### Clean Build

```bash
cd /home/roaramburu/coding/sirius

# Remove build directory
rm -rf build

# Full rebuild
mkdir build && cd build
cmake .. && make -j$(nproc)
```

---

## Development Builds

### Debug Build

```bash
mkdir build-debug && cd build-debug

cmake .. \
  -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_CUDA_ARCHITECTURES=80

make -j$(nproc)
```

**Use Debug Build for**:
- Debugging with GDB
- Verbose error messages
- Assertion checking

### Profile Build

```bash
mkdir build-profile && cd build-profile

cmake .. \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DCMAKE_CUDA_ARCHITECTURES=80

make -j$(nproc)
```

**Use Profile Build for**:
- Performance profiling (nvprof, nsys)
- Optimized but debuggable

---

## Continuous Integration

### GitHub Actions (if applicable)

```yaml
# .github/workflows/build-and-test.yml
name: Build and Test

on: [push, pull_request]

jobs:
  build:
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v2

    - name: Install dependencies
      run: |
        curl -fsSL https://pixi.sh/install.sh | bash

    - name: Build
      run: pixi run build

    - name: Test
      run: pixi run test
```

---

## Build Artifacts

After successful build:

```
build/
├── sirius.duckdb_extension         # Main extension library
├── test/
│   └── cpp/
│       └── sirius_tests            # Test executable
└── benchmarks/
    └── benchmark_tpch              # Benchmark executable
```

### Installing Extension

```bash
# Copy to DuckDB extension directory
cp build/sirius.duckdb_extension ~/.duckdb/extensions/v0.9.0/linux_amd64/

# Or use system-wide install
sudo make install
```

---

## Environment Setup

### Setting Up Environment Variables

Create `~/.bashrc` or `~/.zshrc` additions:

```bash
# CUDA
export CUDA_HOME=/usr/local/cuda-11.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Sirius
export SIRIUS_HOME=/home/roaramburu/coding/sirius
export PATH=$SIRIUS_HOME/build:$PATH

# GPU options
export CUDA_VISIBLE_DEVICES=0  # Use first GPU
```

### Pixi Environment

```bash
# Activate Pixi environment
cd /home/roaramburu/coding/sirius
pixi shell

# Now all dependencies available
which nvcc
which cmake
```

---

## Next Steps

- **Debugging**: [Debugging Guide](debugging.md) - Debug Sirius
- **Adding Operators**: [Adding Operators](adding-operators.md) - Extend Sirius
- **Testing Guide**: [Testing Guide](testing-guide.md) - Write tests
- **Code Organization**: [Code Organization](code-organization.md) - Understand codebase structure
