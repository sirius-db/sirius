vcpkg_check_linkage(ONLY_STATIC_LIBRARY)

# raft is exported header-only here (RAFT_COMPILE_LIBRARY OFF), so this port
# produces no compiled library. Its only job is to install raft's headers and
# the raft-config.cmake that exports the raft::raft interface target. cuvs's
# installed config does find_dependency(raft), and Sirius includes raft headers
# directly, so both need raft to be a findable package in the vcpkg tree.
vcpkg_from_github(
  OUT_SOURCE_PATH
  SOURCE_PATH
  REPO
  rapidsai/raft
  REF
  v26.08.00
  SHA512
  cbfe6c618bac35b16f5b9313f1f0315c8c9e331dcb1bdc5038be772951e45798d644f4c602a4370552cd22778676f860b54ed489662d8ac2ac775ba5efe52cf5
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

# Patch rapids_logger to use vcpkg's spdlog::spdlog target instead of spdlog.
# Same fix the rmm and cuvs ports apply.
vcpkg_replace_string(
  "${RAPIDS_LOGGER_PATH}/CMakeLists.txt"
  "set_target_properties(spdlog PROPERTIES POSITION_INDEPENDENT_CODE ON)"
  "set_target_properties(spdlog::spdlog PROPERTIES POSITION_INDEPENDENT_CODE ON)"
)

# Header-only export: RAFT_COMPILE_LIBRARY OFF means no libraft is built, so no
# gtest/benchmark sources are needed either. rmm is already an installed vcpkg
# port, so raft finds it from CMAKE_PREFIX_PATH.
vcpkg_cmake_configure(
  SOURCE_PATH
  "${SOURCE_PATH}/cpp"
  OPTIONS
  -DFETCHCONTENT_SOURCE_DIR_RAPIDS-CMAKE=${RAPIDS_CMAKE_PATH}
  -DCPM_rapids_logger_SOURCE=${RAPIDS_LOGGER_PATH}
  -DRAFT_COMPILE_LIBRARY=OFF
  -DBUILD_TESTS=OFF
  -DRAFT_NVTX=OFF
  -DCMAKE_CUDA_ARCHITECTURES=RAPIDS
  -DCMAKE_CUDA_RUNTIME_LIBRARY=Static
  "-DCMAKE_CXX_FLAGS=-I${CURRENT_INSTALLED_DIR}/include"
  "-DCMAKE_CUDA_FLAGS=-I${CURRENT_INSTALLED_DIR}/include")

vcpkg_cmake_install()

vcpkg_cmake_config_fixup(PACKAGE_NAME raft CONFIG_PATH lib/cmake/raft)

# rmm is the port that owns rapids_logger in this vcpkg layout. raft's build
# produces its own copy of the headers and static lib; drop them so they do not
# conflict with the files rmm already installed. raft-config still resolves
# rapids_logger through find_dependency against rmm's install (same version).
file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/include/rapids_logger"
     "${CURRENT_PACKAGES_DIR}/share/rapids_logger")
# raft is header-only and librapids_logger.a was the only file in lib/, so once
# it is gone lib/ is empty. Remove it (both release and debug) to avoid vcpkg's
# empty-directory post-build warning.
file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/lib"
     "${CURRENT_PACKAGES_DIR}/debug/lib")

file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/debug/include")
file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/debug/share")

vcpkg_install_copyright(FILE_LIST "${SOURCE_PATH}/LICENSE")
