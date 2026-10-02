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

# Architecture selection and CUDA version both affect the binary cache key.
if(DEFINED ENV{VCPKG_CUDA_ARCHITECTURES}
   AND NOT "$ENV{VCPKG_CUDA_ARCHITECTURES}" STREQUAL "")
  set(VCPKG_CUDA_ARCHITECTURES "$ENV{VCPKG_CUDA_ARCHITECTURES}")
else()
  set(VCPKG_CUDA_ARCHITECTURES RAPIDS)
endif()
set(VCPKG_ENV_PASSTHROUGH VCPKG_CUDA_VERSION VCPKG_CUDA_ARCHITECTURES)
