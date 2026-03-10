vcpkg_check_linkage(ONLY_STATIC_LIBRARY)

vcpkg_from_github(
  OUT_SOURCE_PATH
  SOURCE_PATH
  REPO
  rapidsai/rmm
  REF
  v26.02.00
  SHA512
  bb66832fc1f9e27b09d3d0058923da4f1b3c34d8898f2247c61b6c79c7c0371349cbb50b83c712f3ccd6cdee9a25e601e527e1827291b57ccf887160196467e6
  HEAD_REF
  main)

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

vcpkg_from_github(
  OUT_SOURCE_PATH
  RAPIDS_LOGGER_PATH
  REPO
  rapidsai/rapids-logger
  REF
  4c72b598f99c8aa06af49468b3fc82f3931c6bf6
  SHA512
  5d5997354d3811f6598d1c40296afe88110bdd9e7496ae860e4d3b85528f5d776b70efe3e35cb91d308e4f23e05e04ffa734f05f7b0ed1cdb3278366b0a8309a
  HEAD_REF
  main)

# Patch rapids_logger to use vcpkg's spdlog::spdlog target instead of spdlog
vcpkg_replace_string(
  "${RAPIDS_LOGGER_PATH}/CMakeLists.txt"
  "set_target_properties(spdlog PROPERTIES POSITION_INDEPENDENT_CODE ON)"
  "set_target_properties(spdlog::spdlog PROPERTIES POSITION_INDEPENDENT_CODE ON)"
)

# Ensure vcpkg CCCL headers are found before pixi/conda system CCCL headers. The
# pixi compiler injects -I flags for its bundled CCCL which may be an older
# version.
vcpkg_cmake_configure(
  SOURCE_PATH
  "${SOURCE_PATH}/cpp"
  OPTIONS
  -DFETCHCONTENT_SOURCE_DIR_RAPIDS-CMAKE=${RAPIDS_CMAKE_PATH}
  -DCPM_rapids_logger_SOURCE=${RAPIDS_LOGGER_PATH}
  -DBUILD_TESTS=OFF
  -DBUILD_BENCHMARKS=OFF
  -DCMAKE_CUDA_ARCHITECTURES=RAPIDS
  "-DCMAKE_CXX_FLAGS=-I${CURRENT_INSTALLED_DIR}/include"
  "-DCMAKE_CUDA_FLAGS=-I${CURRENT_INSTALLED_DIR}/include")

vcpkg_cmake_install()

# rapids_logger cmake config is generated but installed to a non-standard
# location. We need to manually install it.
file(
  GLOB
  RAPIDS_LOGGER_CMAKE_FILES
  "${CURRENT_BUILDTREES_DIR}/${TARGET_TRIPLET}-rel/_deps/rapids_logger-build/rapids_logger-*.cmake"
  "${CURRENT_BUILDTREES_DIR}/${TARGET_TRIPLET}-rel/_deps/rapids_logger-build/CMakeFiles/Export/*/rapids_logger-targets*.cmake"
  "${CURRENT_BUILDTREES_DIR}/${TARGET_TRIPLET}-rel/_deps/rapids_logger-build/create_logger_macros.cmake"
)
file(INSTALL ${RAPIDS_LOGGER_CMAKE_FILES}
     DESTINATION "${CURRENT_PACKAGES_DIR}/share/rapids_logger")

# Fix paths in rapids_logger cmake config
file(READ
     "${CURRENT_PACKAGES_DIR}/share/rapids_logger/rapids_logger-config.cmake"
     _config_content)
string(
  REPLACE
    "${CURRENT_BUILDTREES_DIR}/${TARGET_TRIPLET}-rel/_deps/rapids_logger-build"
    "\${CMAKE_CURRENT_LIST_DIR}/../.." _config_content "${_config_content}")
file(WRITE
     "${CURRENT_PACKAGES_DIR}/share/rapids_logger/rapids_logger-config.cmake"
     "${_config_content}")

# Fix rapids_logger-targets.cmake - change from 4 parent dirs to 3
# (share/rapids_logger -> package root). The original file goes up 4 directories
# (assuming lib/cmake/rapids_logger/ layout) but vcpkg uses
# share/rapids_logger/, so we need to go up 3 directories.
execute_process(
  COMMAND
    sed -i "53d"
    "${CURRENT_PACKAGES_DIR}/share/rapids_logger/rapids_logger-targets.cmake")

vcpkg_cmake_config_fixup(PACKAGE_NAME rmm CONFIG_PATH lib/cmake/rmm)

file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/debug/include")
file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/debug/share")

vcpkg_install_copyright(FILE_LIST "${SOURCE_PATH}/LICENSE")
