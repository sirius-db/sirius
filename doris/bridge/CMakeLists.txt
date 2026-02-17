# =============================================================================
# Doris Bridge: C++ static library for Rust FFI via cxx
#
# Links against DuckDB (headless instance), the Sirius GPU extension,
# the DuckDB Substrait extension, and Apache Arrow (IPC serialisation).
# =============================================================================

cmake_minimum_required(VERSION 3.24)
project(doris_bridge LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---------------------------------------------------------------------------
# Paths — these can be overridden via -D flags from build.rs / cargo
# ---------------------------------------------------------------------------

# Root of the Sirius repo (parent of doris/)
set(SIRIUS_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../.."
    CACHE PATH "Path to Sirius repo root")

# DuckDB source tree (submodule inside Sirius)
set(DUCKDB_ROOT "${SIRIUS_ROOT}/duckdb"
    CACHE PATH "Path to DuckDB source root")

# Sirius build directory (contains libduckdb_static.a, libsirius_extension.a, etc.)
set(SIRIUS_BUILD_DIR "${SIRIUS_ROOT}/build/release"
    CACHE PATH "Path to Sirius CMake build directory")

# Substrait extension source (submodule inside Sirius)
set(SUBSTRAIT_ROOT "${SIRIUS_ROOT}/substrait"
    CACHE PATH "Path to DuckDB Substrait extension source")

# vcpkg installed packages (Arrow, protobuf, etc.)
set(VCPKG_INSTALLED "${SIRIUS_ROOT}/vcpkg/packages"
    CACHE PATH "Path to vcpkg packages directory")

# ---------------------------------------------------------------------------
# Include directories
# ---------------------------------------------------------------------------

# DuckDB headers
include_directories("${DuckDB_ROOT}/src/include")
include_directories("${SIRIUS_BUILD_DIR}/src/include")  # generated headers

# Sirius headers
include_directories("${SIRIUS_ROOT}/src/include")
include_directories("${SIRIUS_ROOT}/src/include/operator")

# Substrait extension headers (from_substrait.hpp)
include_directories("${SUBSTRAIT_ROOT}/src/include")

# Substrait protobuf headers (substrait/plan.pb.h)
include_directories("${SUBSTRAIT_ROOT}/third_party/substrait/substrait")
include_directories("${SUBSTRAIT_ROOT}/third_party")

# Arrow C++ headers (from vcpkg)
include_directories("${VCPKG_INSTALLED}/arrow_x64-linux/include")

# Protobuf headers (from vcpkg, needed by substrait extension)
include_directories("${VCPKG_INSTALLED}/protobuf_x64-linux/include")

# Bridge headers
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/include")

# ---------------------------------------------------------------------------
# Static library
# ---------------------------------------------------------------------------

add_library(doris_bridge STATIC
    src/doris_bridge.cpp
)

# ---------------------------------------------------------------------------
# Link libraries
# ---------------------------------------------------------------------------

# DuckDB static library
find_library(DUCKDB_STATIC duckdb_static
    PATHS "${SIRIUS_BUILD_DIR}/src"
    NO_DEFAULT_PATH)

# Arrow C++ libraries (from vcpkg)
find_library(ARROW_LIB arrow
    PATHS "${VCPKG_INSTALLED}/arrow_x64-linux/lib"
    NO_DEFAULT_PATH)

if(DUCKDB_STATIC)
    target_link_libraries(doris_bridge PRIVATE ${DUCKDB_STATIC})
endif()

if(ARROW_LIB)
    target_link_libraries(doris_bridge PRIVATE ${ARROW_LIB})
endif()

# CUDA runtime (for cudaMemcpy, cudaMalloc in register_exchange_table)
find_package(CUDAToolkit REQUIRED)
target_link_libraries(doris_bridge PRIVATE CUDA::cudart)

# ---------------------------------------------------------------------------
# Install
# ---------------------------------------------------------------------------

install(TARGETS doris_bridge ARCHIVE DESTINATION lib)
install(DIRECTORY include/ DESTINATION include)
