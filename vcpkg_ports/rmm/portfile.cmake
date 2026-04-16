vcpkg_check_linkage(ONLY_STATIC_LIBRARY)

vcpkg_from_github(
  OUT_SOURCE_PATH
  SOURCE_PATH
  REPO
  rapidsai/rmm
  REF
  v${VERSION}
  SHA512
  9e30d3c1ee5bdcaf545c72934fb8c2dc4c0a65bc51b439d305a8330d1753f496f8ef414a8ffbbbbccda42c63880fd740199bf391217c227bd12bd1125984de47
  HEAD_REF
  main)

vcpkg_from_github(
  OUT_SOURCE_PATH
  RAPIDS_CMAKE_PATH
  REPO
  rapidsai/rapids-cmake
  REF
  v${VERSION}
  SHA512
  de9234549a96b2a73caa4bbbaf10500419c5f16069a430d172432c23e6d404b3ab9e595f6c0958d15f4f190fb50e1cd7dd02945b93cf428180df8feb8822ed94
  HEAD_REF
  main)

vcpkg_from_github(
  OUT_SOURCE_PATH
  RAPIDS_LOGGER_PATH
  REPO
  rapidsai/rapids-logger
  REF
  v0.2.3
  SHA512
  eb7b5ebf6289d10307b8a34d9d1469ffcb63e9371e9dd5ccbda0351923b920ebae8220ceaa8d1d52c9bed57200f35921a6365a5f9a25a209a98314f75195310c
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
  -DCMAKE_CUDA_RUNTIME_LIBRARY=Static
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

# RMM still exports shared CUDA::cudart in its installed target file even when
# built as a static library. That pulls libcudart.so into the final Sirius
# loadable extension alongside libcudart_static.a. Rewrite the exported target
# to prefer the static CUDA runtime for vcpkg consumers.
file(READ "${CURRENT_PACKAGES_DIR}/share/rmm/rmm-targets.cmake" _rmm_targets)
string(REGEX REPLACE "CUDA::cudart([;\">])" "CUDA::cudart_static\\1"
                     _rmm_targets "${_rmm_targets}")
file(WRITE "${CURRENT_PACKAGES_DIR}/share/rmm/rmm-targets.cmake"
     "${_rmm_targets}")

file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/debug/include")
file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/debug/share")

vcpkg_install_copyright(FILE_LIST "${SOURCE_PATH}/LICENSE")
