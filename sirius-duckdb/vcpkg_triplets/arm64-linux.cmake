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
set(VCPKG_CUDA_ARCHITECTURES "$ENV{VCPKG_CUDA_ARCHITECTURES}")
if(VCPKG_CUDA_ARCHITECTURES STREQUAL "")
  message(
    FATAL_ERROR
      "Set VCPKG_CUDA_ARCHITECTURES to an explicit CUDA architecture list")
endif()
set(VCPKG_ENV_PASSTHROUGH VCPKG_CUDA_VERSION VCPKG_CUDA_ARCHITECTURES)

include("${CMAKE_CURRENT_LIST_DIR}/../vcpkg_ports/sirius/source-inputs.cmake")
