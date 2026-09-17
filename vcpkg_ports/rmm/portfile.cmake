vcpkg_check_linkage(ONLY_STATIC_LIBRARY)

vcpkg_from_github(
  OUT_SOURCE_PATH
  SOURCE_PATH
  REPO
  rapidsai/rmm
  REF
  v${VERSION}
  SHA512
  0ea7efa882e13431c4faba60bbc805522436735ed91922987448aa5e64137b44937b66fc2d6bcf8658f2bda71f90fcc0f369cc384cdd481848cbefb5b30c0836
  HEAD_REF
  main)

vcpkg_from_github(
  OUT_SOURCE_PATH
  RAPIDS_CMAKE_PATH
  REPO
  rapidsai/rapids-cmake
  REF
  v26.08.00
  SHA512
  472e3bbc0aeedce6632c339f5a383a25524df8462a58891474523cd558b7d8d8bc09b24e8f75da6f2fbfcb83b302caa911fb5a38bc6c04871a570390e7b4a5b8
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
# version. RMM still includes cuda/stream_ref, deprecated by CCCL 3.4.3.
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
  "-DCMAKE_CXX_FLAGS=-I${CURRENT_INSTALLED_DIR}/include -DCCCL_IGNORE_DEPRECATED_STREAM_REF_HEADER"
)

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

# create_logger_macros.cmake calls configure_file() on logger_macros.hpp.in,
# resolved relative to its own install dir. kvikio/cudf 26.06 invoke this macro
# generator (26.04 did not), so the template must sit alongside it.
file(INSTALL "${RAPIDS_LOGGER_PATH}/cmake/logger_macros.hpp.in"
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
