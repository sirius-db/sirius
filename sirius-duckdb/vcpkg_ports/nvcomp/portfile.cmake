vcpkg_check_linkage(ONLY_STATIC_LIBRARY)

set(NVCOMP_VERSION "${VERSION}")

# CUDA version from triplet (set via VCPKG_CUDA_VERSION env var)
if(NOT DEFINED VCPKG_CUDA_VERSION)
  message(
    FATAL_ERROR
      "VCPKG_CUDA_VERSION not set. Set the VCPKG_CUDA_VERSION environment variable to 12 or 13."
  )
endif()
set(CUDA_VERSION "${VCPKG_CUDA_VERSION}")

if(VCPKG_TARGET_ARCHITECTURE STREQUAL "x64")
  set(NVCOMP_PLATFORM "linux-x86_64")
  if(CUDA_VERSION STREQUAL "12")
    set(NVCOMP_SHA512
        "f8ca298d09ce77596fc00f4242cb817741539e31162b7305a86839de53570f0e22f79255e7d518a9f5711f642ac2bed8d3d6686db4eb9538f9f6c8435dd3efb5"
    )
  elseif(CUDA_VERSION STREQUAL "13")
    set(NVCOMP_SHA512
        "0894354e7590a90a2c5701a0ede975b0fdb9d8955e0af7028fcc8d98079fb881449f5b58cbfe1ab93d99427b1972f88adca9e31433625ff2546a59641d323532"
    )
  else()
    message(
      FATAL_ERROR "Unsupported CUDA version: ${CUDA_VERSION}. Supported: 12, 13"
    )
  endif()
elseif(VCPKG_TARGET_ARCHITECTURE STREQUAL "arm64")
  set(NVCOMP_PLATFORM "linux-sbsa")
  if(CUDA_VERSION STREQUAL "12")
    set(NVCOMP_SHA512
        "f44a8361935322b2085caa00b5659b47cd02b639a6f1a4287c49f2917a04002da29e74cedc861aca4e33fc62572b0479b7c9a2644f079d3a604b27691613c7f6"
    )
  elseif(CUDA_VERSION STREQUAL "13")
    set(NVCOMP_SHA512
        "13ee964c53f990c966330427f8368ce492014a83a0af589c1b3c6c93c186b0c4fc05d8cbee5f82dbb3da6a412cc40450fb7f2d65f268594ef03d7c49c3c686c7"
    )
  else()
    message(
      FATAL_ERROR "Unsupported CUDA version: ${CUDA_VERSION}. Supported: 12, 13"
    )
  endif()
else()
  message(FATAL_ERROR "Unsupported architecture: ${VCPKG_TARGET_ARCHITECTURE}")
endif()

vcpkg_download_distfile(
  ARCHIVE
  URLS
  "https://developer.download.nvidia.com/compute/nvcomp/redist/nvcomp/${NVCOMP_PLATFORM}/nvcomp-${NVCOMP_PLATFORM}-${NVCOMP_VERSION}_cuda${CUDA_VERSION}-archive.tar.xz"
  FILENAME
  "nvcomp-${NVCOMP_PLATFORM}-${NVCOMP_VERSION}_cuda${CUDA_VERSION}-archive.tar.xz"
  SHA512
  ${NVCOMP_SHA512})

vcpkg_extract_source_archive(SOURCE_PATH ARCHIVE "${ARCHIVE}")

# Install headers
file(GLOB HEADER_FILES "${SOURCE_PATH}/include/*")
file(INSTALL ${HEADER_FILES} DESTINATION "${CURRENT_PACKAGES_DIR}/include")

# Install libraries
file(GLOB LIB_FILES "${SOURCE_PATH}/lib/*.a")
file(INSTALL ${LIB_FILES} DESTINATION "${CURRENT_PACKAGES_DIR}/lib")

# Keep nvCOMP's bundled logger ABI private to its static archives.
vcpkg_cmake_get_vars(NVCOMP_CMAKE_VARS)
include("${NVCOMP_CMAKE_VARS}")
include("${CMAKE_CURRENT_LIST_DIR}/isolate-logging.cmake")
file(GLOB NVCOMP_ARCHIVES "${CURRENT_PACKAGES_DIR}/lib/*.a")
nvcomp_isolate_logging(
  "${VCPKG_DETECTED_CMAKE_NM}" "${VCPKG_DETECTED_CMAKE_OBJCOPY}"
  "${CURRENT_BUILDTREES_DIR}/logging-symbols.txt" ${NVCOMP_ARCHIVES})

# Install CMake config files (targets only, we'll write a custom config.cmake)
file(INSTALL "${SOURCE_PATH}/lib/cmake/nvcomp/nvcomp-config-version.cmake"
     DESTINATION "${CURRENT_PACKAGES_DIR}/share/nvcomp")
file(INSTALL "${SOURCE_PATH}/lib/cmake/nvcomp/nvcomp-targets-static.cmake"
     DESTINATION "${CURRENT_PACKAGES_DIR}/share/nvcomp")
file(INSTALL
     "${SOURCE_PATH}/lib/cmake/nvcomp/nvcomp-targets-static-release.cmake"
     DESTINATION "${CURRENT_PACKAGES_DIR}/share/nvcomp")

# Write a custom config file that works with vcpkg layout
file(
  WRITE "${CURRENT_PACKAGES_DIR}/share/nvcomp/nvcomp-config.cmake"
  "
get_filename_component(PACKAGE_PREFIX_DIR \"\${CMAKE_CURRENT_LIST_DIR}/../../\" ABSOLUTE)

set(nvcomp_VERSION ${VERSION})
set(nvcomp_INCLUDE_DIR \"\${PACKAGE_PREFIX_DIR}/include\")
set(nvcomp_LIBRARY_DIR \"\${PACKAGE_PREFIX_DIR}/lib\")

# Check headers and library directories exist
if(NOT EXISTS \"\${nvcomp_INCLUDE_DIR}/nvcomp.h\")
    message(FATAL_ERROR \"nvcomp headers not found at \${nvcomp_INCLUDE_DIR}\")
endif()

# Load the target definitions
include(\"\${CMAKE_CURRENT_LIST_DIR}/nvcomp-targets-static.cmake\")

# Create alias for compatibility with downstream projects
if(TARGET nvcomp::nvcomp_static AND NOT TARGET nvcomp::nvcomp)
    add_library(nvcomp::nvcomp ALIAS nvcomp::nvcomp_static)
endif()

set(nvcomp_FOUND TRUE)
")

vcpkg_install_copyright(FILE_LIST "${SOURCE_PATH}/LICENSE")
