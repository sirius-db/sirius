set(VCPKG_TARGET_ARCHITECTURE arm64)
set(VCPKG_CRT_LINKAGE dynamic)
set(VCPKG_LIBRARY_LINKAGE static)

set(VCPKG_CMAKE_SYSTEM_NAME Linux)

# Only build release to speed up builds
set(VCPKG_BUILD_TYPE release)

# CUDA version for ports that need version-specific binaries (e.g. nvcomp). Set
# via VCPKG_CUDA_VERSION env var; defaults to 13.
if(DEFINED ENV{VCPKG_CUDA_VERSION})
  set(VCPKG_CUDA_VERSION $ENV{VCPKG_CUDA_VERSION})
else()
  set(VCPKG_CUDA_VERSION 13)
endif()

# Include CUDA version in ABI hash so binary caching distinguishes CUDA 12 vs 13
set(VCPKG_ENV_PASSTHROUGH VCPKG_CUDA_VERSION)
