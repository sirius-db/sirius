vcpkg_from_github(
  OUT_SOURCE_PATH
  SOURCE_PATH
  REPO
  NVIDIA/cccl
  REF
  # CCCL 3.4.0 has no release tag yet; pin the commit rapids-cmake 26.06 uses.
  207502e57019cefacf6f21d3bb6045aeebef2a3e
  SHA512
  9dfdb0ba4100f8a37859a1918a7adacea7de06714d7153cbb130342468149ce4b96d306d2262ba346ca326a385a9af053ee57b4c0d41c4736574e5b9825a4686
  HEAD_REF
  main)

# CCCL is header-only, install all headers (including extension-less C++
# standard headers)
file(
  INSTALL "${SOURCE_PATH}/thrust/thrust/"
  DESTINATION "${CURRENT_PACKAGES_DIR}/include/thrust"
  FILES_MATCHING
  PATTERN "*.h"
  PATTERN "*.inl")
file(
  INSTALL "${SOURCE_PATH}/cub/cub/"
  DESTINATION "${CURRENT_PACKAGES_DIR}/include/cub"
  FILES_MATCHING
  PATTERN "*.cuh")
# libcudacxx has both .h/.hpp headers AND extension-less C++ standard headers
# (like climits, cstdint, etc.)
file(COPY "${SOURCE_PATH}/libcudacxx/include/"
     DESTINATION "${CURRENT_PACKAGES_DIR}/include")

# Install CMake config files
file(GLOB CCCL_CMAKE_FILES "${SOURCE_PATH}/lib/cmake/cccl/*")
file(INSTALL ${CCCL_CMAKE_FILES}
     DESTINATION "${CURRENT_PACKAGES_DIR}/share/cccl")
file(GLOB THRUST_CMAKE_FILES "${SOURCE_PATH}/lib/cmake/thrust/*")
file(INSTALL ${THRUST_CMAKE_FILES}
     DESTINATION "${CURRENT_PACKAGES_DIR}/share/thrust")
file(GLOB CUB_CMAKE_FILES "${SOURCE_PATH}/lib/cmake/cub/*")
file(INSTALL ${CUB_CMAKE_FILES} DESTINATION "${CURRENT_PACKAGES_DIR}/share/cub")
file(GLOB LIBCUDACXX_CMAKE_FILES "${SOURCE_PATH}/lib/cmake/libcudacxx/*")
file(INSTALL ${LIBCUDACXX_CMAKE_FILES}
     DESTINATION "${CURRENT_PACKAGES_DIR}/share/libcudacxx")

# Fix header-search.cmake files: upstream configs expect headers relative to the
# CCCL source tree (e.g. ../../../libcudacxx/include). In vcpkg, all headers are
# installed to ${prefix}/include, so fix the paths.
vcpkg_replace_string(
  "${CURRENT_PACKAGES_DIR}/share/libcudacxx/libcudacxx-header-search.cmake"
  [[${CMAKE_CURRENT_LIST_DIR}/../../../libcudacxx/include]]
  [[${CMAKE_CURRENT_LIST_DIR}/../../include]])
vcpkg_replace_string(
  "${CURRENT_PACKAGES_DIR}/share/thrust/thrust-header-search.cmake"
  [[${CMAKE_CURRENT_LIST_DIR}/../../../thrust]]
  [[${CMAKE_CURRENT_LIST_DIR}/../../include]])
vcpkg_replace_string(
  "${CURRENT_PACKAGES_DIR}/share/cub/cub-header-search.cmake"
  [[${CMAKE_CURRENT_LIST_DIR}/../../../cub]]
  [[${CMAKE_CURRENT_LIST_DIR}/../../include]])

vcpkg_install_copyright(FILE_LIST "${SOURCE_PATH}/LICENSE")
