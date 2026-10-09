vcpkg_check_linkage(ONLY_STATIC_LIBRARY)

if(NOT DEFINED VCPKG_CUDA_VERSION OR NOT VCPKG_CUDA_VERSION MATCHES "^(12|13)$")
  message(FATAL_ERROR "Set VCPKG_CUDA_VERSION to 12 or 13 in the triplet.")
endif()

if(VCPKG_TARGET_ARCHITECTURE STREQUAL "x64")
  set(_platform "linux-x86_64")
elseif(VCPKG_TARGET_ARCHITECTURE STREQUAL "arm64")
  set(_platform "linux-sbsa")
else()
  message(FATAL_ERROR "Unsupported architecture: ${VCPKG_TARGET_ARCHITECTURE}")
endif()

# Pinned NVIDIA CUDA 12.9.1 / 13.3.0 redistributables.
file(READ "${CMAKE_CURRENT_LIST_DIR}/redistrib.json" _redistrib)
string(JSON _version GET "${_redistrib}" "${VCPKG_CUDA_VERSION}" version)
string(JSON _sha512 GET "${_redistrib}" "${VCPKG_CUDA_VERSION}" "${_platform}"
       sha512)
set(_archive "lib${PORT}-${_platform}-${_version}-archive.tar.xz")

vcpkg_download_distfile(
  ARCHIVE
  URLS
  "https://developer.download.nvidia.com/compute/cuda/redist/lib${PORT}/${_platform}/${_archive}"
  FILENAME
  "${_archive}"
  SHA512
  "${_sha512}")
vcpkg_extract_source_archive(SOURCE_PATH ARCHIVE "${ARCHIVE}")

file(INSTALL "${SOURCE_PATH}/include/"
     DESTINATION "${CURRENT_PACKAGES_DIR}/include")
file(INSTALL "${SOURCE_PATH}/lib/libcublas_static.a"
     "${SOURCE_PATH}/lib/libcublasLt_static.a"
     DESTINATION "${CURRENT_PACKAGES_DIR}/lib")
file(INSTALL "${CMAKE_CURRENT_LIST_DIR}/${PORT}-config.cmake"
     DESTINATION "${CURRENT_PACKAGES_DIR}/share/${PORT}")
vcpkg_install_copyright(FILE_LIST "${SOURCE_PATH}/LICENSE")
