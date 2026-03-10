vcpkg_from_github(
  OUT_SOURCE_PATH
  SOURCE_PATH
  REPO
  NVIDIA/cuCollections
  REF
  d3701ae8e7f2a08f25f9713e182692b4ca544112
  SHA512
  2a35e079a86a62cca7dadf93b45609d86a0bdef1e987c0d5ec59600d1950e6d6a5e26e06cc3d165c389442ce5e25b461caf4cecd67c35369a72d742b1b5851cf
  HEAD_REF
  dev)

vcpkg_from_github(
  OUT_SOURCE_PATH
  RAPIDS_CMAKE_PATH
  REPO
  rapidsai/rapids-cmake
  REF
  v26.02.00
  SHA512
  00d2bb2c005f9e2c4e525af4350c8e8b7d6e67369da9c664c6c3d6080c33d6359b7142a311f3563479f7ac8e1bb0a2a520e8e926719dfeb8e27e49f8cd3e65ca
  HEAD_REF
  main)

# Patch cuco's CMakeLists.txt to use our rapids-cmake directly instead of
# downloading RAPIDS.cmake from GitHub. The download may fail in
# offline/restricted environments.
vcpkg_replace_string(
  "${SOURCE_PATH}/CMakeLists.txt"
  [[if(NOT EXISTS ${CMAKE_CURRENT_BINARY_DIR}/CUCO_RAPIDS.cmake)
    file(DOWNLOAD
      https://raw.githubusercontent.com/rapidsai/rapids-cmake/release/${rapids-cmake-version}/RAPIDS.cmake
         ${CMAKE_CURRENT_BINARY_DIR}/CUCO_RAPIDS.cmake)
endif()
include(${CMAKE_CURRENT_BINARY_DIR}/CUCO_RAPIDS.cmake)]]
  "set(rapids-cmake-dir \"${RAPIDS_CMAKE_PATH}/rapids-cmake\")\nlist(APPEND CMAKE_MODULE_PATH \"\${rapids-cmake-dir}\")"
)

vcpkg_cmake_configure(
  SOURCE_PATH
  "${SOURCE_PATH}"
  OPTIONS
  -DFETCHCONTENT_SOURCE_DIR_RAPIDS-CMAKE=${RAPIDS_CMAKE_PATH}
  -DBUILD_TESTS=OFF
  -DBUILD_BENCHMARKS=OFF
  -DBUILD_EXAMPLES=OFF)

vcpkg_cmake_install()

vcpkg_cmake_config_fixup(CONFIG_PATH lib/cmake/cuco)

file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/debug")

vcpkg_install_copyright(FILE_LIST "${SOURCE_PATH}/LICENSE")
