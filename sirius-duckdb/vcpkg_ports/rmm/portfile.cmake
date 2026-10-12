vcpkg_check_linkage(ONLY_STATIC_LIBRARY)

vcpkg_from_github(
  OUT_SOURCE_PATH
  SOURCE_PATH
  REPO
  rapidsai/rmm
  REF
  v${VERSION}
  SHA512
  67b090cc1d3018bcc983f601b7680ea6bb018baf8245b65fe19032b077076bd11e16c22444dfe286e23d87b258f18c0077b1c638dc4eefc696eabdfb9ffbae94
  HEAD_REF
  main)

vcpkg_from_github(
  OUT_SOURCE_PATH
  RAPIDS_CMAKE_PATH
  REPO
  rapidsai/rapids-cmake
  REF
  v26.10.00
  SHA512
  c5c2be8672faae3990713a990b8e0d4e1082307c9dfd5857805bd543243daf0ec2c1fe41bbdb1736d5ea4e4e9fa261de44b2e8b885d812ce822814251d9f65d2
  HEAD_REF
  main)

vcpkg_from_github(
  OUT_SOURCE_PATH
  RAPIDS_LOGGER_PATH
  REPO
  rapidsai/rapids-logger
  REF
  v0.3.0
  SHA512
  0952bcd7c6579b0cc613e848c1d695466942d3a09d3f98af4c8fed97d23f5e7ad86d73603bfc58b55a4f33461b4e567716d0a70957935fefb896907bdf1074d1
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
# version. RMM still includes cuda/stream_ref, deprecated by CCCL 3.5.0.
vcpkg_cmake_configure(
  SOURCE_PATH
  "${SOURCE_PATH}/cpp"
  OPTIONS
  -DFETCHCONTENT_SOURCE_DIR_RAPIDS-CMAKE=${RAPIDS_CMAKE_PATH}
  -DCPM_rapids_logger_SOURCE=${RAPIDS_LOGGER_PATH}
  -DBUILD_TESTS=OFF
  -DBUILD_BENCHMARKS=OFF
  "-DCMAKE_CUDA_ARCHITECTURES=${VCPKG_CUDA_ARCHITECTURES}"
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

# Resolve installed dependencies without build-tree paths or CPM downloads.
configure_file(
  "${CMAKE_CURRENT_LIST_DIR}/rapids_logger-dependencies.cmake"
  "${CURRENT_PACKAGES_DIR}/share/rapids_logger/rapids_logger-dependencies.cmake"
  COPYONLY)

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
