# NVRTC static redistributable. Mirrors the nvcomp port: downloads NVIDIA's
# cuda_nvrtc redist archive and installs the static libs + header, so Sirius can
# statically link the runtime JIT compiler and the distributed extension no
# longer depends on libnvrtc.so at load time. (The CCCL headers NVRTC compiles
# against are embedded in the Sirius binary separately; see
# src/compression/simpatico_codegen/cmake/embed_cccl_headers.cmake.)
vcpkg_check_linkage(ONLY_STATIC_LIBRARY)

if(NOT DEFINED VCPKG_CUDA_VERSION)
  message(
    FATAL_ERROR
      "VCPKG_CUDA_VERSION not set. Set the VCPKG_CUDA_VERSION environment variable to 12 or 13."
  )
endif()
set(CUDA_VERSION "${VCPKG_CUDA_VERSION}")

# The cuda_nvrtc redist version differs per CUDA major.
if(CUDA_VERSION STREQUAL "13")
  set(NVRTC_VERSION "13.2.51")
elseif(CUDA_VERSION STREQUAL "12")
  set(NVRTC_VERSION "12.9.86")
else()
  message(
    FATAL_ERROR "Unsupported CUDA version: ${CUDA_VERSION}. Supported: 12, 13")
endif()

if(VCPKG_TARGET_ARCHITECTURE STREQUAL "x64")
  set(NVRTC_PLATFORM "linux-x86_64")
  if(CUDA_VERSION STREQUAL "13")
    set(NVRTC_SHA512
        "d3415a8fa3ccc0d64b99da0ed3b3f9d34c9bd1b73a1412d0f0f368ce6d79e9423aed45d672f22c412476ad688d22d272c2797d76bc6e165c84b433ef656b6e81"
    )
  else()
    set(NVRTC_SHA512
        "f40c7ea4da60bacb19d10659e1e74bb165b09f33c7c95693d1ed52d4ae13014979b6061b9080384a23342a28dea717a59357c15f91be1d4b36f93edbf4b7435d"
    )
  endif()
elseif(VCPKG_TARGET_ARCHITECTURE STREQUAL "arm64")
  set(NVRTC_PLATFORM "linux-sbsa")
  if(CUDA_VERSION STREQUAL "13")
    set(NVRTC_SHA512
        "08c363f8f8ebc39921947b739877a50afed16cbdb08e67ed21670cd81640f5ade18da5078b1407072c09d3aa14865eafeaaf1017d832dc1e0eab4a836e7f8e4a"
    )
  else()
    set(NVRTC_SHA512
        "f620a53ff98cea1dc213e5a041c734314332b5bd69b43cf8d90aba526302e2fb9f3c1a60a814d6065439cc5ed0d1ec1eb9ed737fe266f3708c1d883a5a9ade4f"
    )
  endif()
else()
  message(FATAL_ERROR "Unsupported architecture: ${VCPKG_TARGET_ARCHITECTURE}")
endif()

vcpkg_download_distfile(
  ARCHIVE
  URLS
  "https://developer.download.nvidia.com/compute/cuda/redist/cuda_nvrtc/${NVRTC_PLATFORM}/cuda_nvrtc-${NVRTC_PLATFORM}-${NVRTC_VERSION}-archive.tar.xz"
  FILENAME
  "cuda_nvrtc-${NVRTC_PLATFORM}-${NVRTC_VERSION}-archive.tar.xz"
  SHA512
  ${NVRTC_SHA512})

vcpkg_extract_source_archive(SOURCE_PATH ARCHIVE "${ARCHIVE}")

# Header
file(INSTALL "${SOURCE_PATH}/include/nvrtc.h"
     DESTINATION "${CURRENT_PACKAGES_DIR}/include")

# Static libraries (skip the *.alt.a old-ABI variants shipped for CUDA 12).
file(GLOB LIB_FILES "${SOURCE_PATH}/lib/libnvrtc_static.a"
     "${SOURCE_PATH}/lib/libnvrtc-builtins_static.a")
file(INSTALL ${LIB_FILES} DESTINATION "${CURRENT_PACKAGES_DIR}/lib")

# Custom config: expose nvrtc::nvrtc_static. Static NVRTC also requires
# nvptxcompiler_static (from the CUDA toolkit, not this archive) plus system
# libs; declare them as interface link dependencies so consumers link correctly.
file(
  WRITE "${CURRENT_PACKAGES_DIR}/share/nvrtc/nvrtc-config.cmake"
  "
get_filename_component(PACKAGE_PREFIX_DIR \"\${CMAKE_CURRENT_LIST_DIR}/../../\" ABSOLUTE)

set(nvrtc_VERSION ${NVRTC_VERSION})
set(nvrtc_INCLUDE_DIR \"\${PACKAGE_PREFIX_DIR}/include\")
set(nvrtc_LIBRARY_DIR \"\${PACKAGE_PREFIX_DIR}/lib\")

if(NOT EXISTS \"\${nvrtc_INCLUDE_DIR}/nvrtc.h\")
    message(FATAL_ERROR \"nvrtc header not found at \${nvrtc_INCLUDE_DIR}\")
endif()

include(CMakeFindDependencyMacro)
find_dependency(CUDAToolkit)

if(NOT TARGET nvrtc::nvrtc_builtins_static)
    add_library(nvrtc::nvrtc_builtins_static STATIC IMPORTED)
    set_target_properties(nvrtc::nvrtc_builtins_static PROPERTIES
        IMPORTED_LOCATION \"\${nvrtc_LIBRARY_DIR}/libnvrtc-builtins_static.a\")
endif()

if(NOT TARGET nvrtc::nvrtc_static)
    add_library(nvrtc::nvrtc_static STATIC IMPORTED)
    set_target_properties(nvrtc::nvrtc_static PROPERTIES
        IMPORTED_LOCATION \"\${nvrtc_LIBRARY_DIR}/libnvrtc_static.a\"
        INTERFACE_INCLUDE_DIRECTORIES \"\${nvrtc_INCLUDE_DIR}\")
    # nvrtc_static -> nvrtc-builtins_static -> nvptxcompiler_static (toolkit).
    set_property(TARGET nvrtc::nvrtc_static PROPERTY INTERFACE_LINK_LIBRARIES
        nvrtc::nvrtc_builtins_static
        CUDA::nvptxcompiler_static
        \${CMAKE_DL_LIBS}
        Threads::Threads)
endif()

# Alias so downstream can link a plain 'nvrtc' name.
if(TARGET nvrtc::nvrtc_static AND NOT TARGET nvrtc::nvrtc)
    add_library(nvrtc::nvrtc ALIAS nvrtc::nvrtc_static)
endif()

set(nvrtc_FOUND TRUE)
")

vcpkg_install_copyright(FILE_LIST "${SOURCE_PATH}/LICENSE")
