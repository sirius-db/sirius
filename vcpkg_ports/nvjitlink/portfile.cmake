vcpkg_check_linkage(ONLY_STATIC_LIBRARY)

if(NOT DEFINED VCPKG_CUDA_VERSION)
  message(
    FATAL_ERROR
      "VCPKG_CUDA_VERSION not set. Set the VCPKG_CUDA_VERSION environment variable to 12 or 13."
  )
endif()
set(CUDA_VERSION "${VCPKG_CUDA_VERSION}")

# The libnvjitlink redist version differs per CUDA major.
if(CUDA_VERSION STREQUAL "13")
  set(NVJITLINK_VERSION "13.3.33")
elseif(CUDA_VERSION STREQUAL "12")
  set(NVJITLINK_VERSION "12.9.86")
else()
  message(
    FATAL_ERROR "Unsupported CUDA version: ${CUDA_VERSION}. Supported: 12, 13")
endif()

if(VCPKG_TARGET_ARCHITECTURE STREQUAL "x64")
  set(NVJITLINK_PLATFORM "linux-x86_64")
  if(CUDA_VERSION STREQUAL "13")
    set(NVJITLINK_SHA512
        "ed84e6e9e8037143a09ed5663be1147707feb5ba6575a8c411b3fb6338d685565ed7e5d4e0a75f021a2fe988925a5e74358dd922e255fbf22d80042dda8c74bf"
    )
  else()
    set(NVJITLINK_SHA512
        "84ad400c98c52dc8d4370ae4c2617886dc651f5608864b5ca36adb5a5a3d2d05eb9d63871c8da668ab3d3434df906b1d93431a896dc72b5e340aecedf58bc48e"
    )
  endif()
elseif(VCPKG_TARGET_ARCHITECTURE STREQUAL "arm64")
  set(NVJITLINK_PLATFORM "linux-sbsa")
  if(CUDA_VERSION STREQUAL "13")
    set(NVJITLINK_SHA512
        "5c4dbcaec354598dcbdc84d99dafe727e0a8e4c6bcb16f3efa2a68c4b422a9b865a9a6df6c5f51d5044e91359d7ee132aaa48570501623ebf41c8fc407c101f9"
    )
  else()
    set(NVJITLINK_SHA512
        "18fcbbb60edaa77e4ad02b510a70177f506a1775ccb22822c1501c349736fdd12ac5d268671c14429c1925724165a72c4198bf5bacd7231da80400ea1726388a"
    )
  endif()
else()
  message(FATAL_ERROR "Unsupported architecture: ${VCPKG_TARGET_ARCHITECTURE}")
endif()

vcpkg_download_distfile(
  ARCHIVE
  URLS
  "https://developer.download.nvidia.com/compute/cuda/redist/libnvjitlink/${NVJITLINK_PLATFORM}/libnvjitlink-${NVJITLINK_PLATFORM}-${NVJITLINK_VERSION}-archive.tar.xz"
  FILENAME
  "libnvjitlink-${NVJITLINK_PLATFORM}-${NVJITLINK_VERSION}-archive.tar.xz"
  SHA512
  ${NVJITLINK_SHA512})

vcpkg_extract_source_archive(SOURCE_PATH ARCHIVE "${ARCHIVE}")

file(INSTALL "${SOURCE_PATH}/include/nvJitLink.h"
     DESTINATION "${CURRENT_PACKAGES_DIR}/include")
file(INSTALL "${SOURCE_PATH}/lib/libnvJitLink_static.a"
     DESTINATION "${CURRENT_PACKAGES_DIR}/lib")
file(INSTALL "${CMAKE_CURRENT_LIST_DIR}/nvjitlink-config.cmake"
     DESTINATION "${CURRENT_PACKAGES_DIR}/share/nvjitlink")
vcpkg_install_copyright(FILE_LIST "${SOURCE_PATH}/LICENSE")
