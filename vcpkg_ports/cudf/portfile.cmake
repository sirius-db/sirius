vcpkg_check_linkage(ONLY_STATIC_LIBRARY)

vcpkg_from_github(
    OUT_SOURCE_PATH SOURCE_PATH
    REPO rapidsai/cudf
    REF v26.02.00
    SHA512 3ec3b2184acce64f87d662bb49571d0bee9101893bfb73073ce13b64c8a4050fbccba35504deaef0726ee647e2c13b375c1c68922f7d5caba2aa18ac159eba68
    HEAD_REF main
)

vcpkg_from_github(
    OUT_SOURCE_PATH RAPIDS_CMAKE_PATH
    REPO rapidsai/rapids-cmake
    REF v26.02.00
    SHA512 00d2bb2c005f9e2c4e525af4350c8e8b7d6e67369da9c664c6c3d6080c33d6359b7142a311f3563479f7ac8e1bb0a2a520e8e926719dfeb8e27e49f8cd3e65ca
    HEAD_REF main
)

vcpkg_from_github(
    OUT_SOURCE_PATH RAPIDS_LOGGER_PATH
    REPO rapidsai/rapids-logger
    REF 4c72b598f99c8aa06af49468b3fc82f3931c6bf6
    SHA512 5d5997354d3811f6598d1c40296afe88110bdd9e7496ae860e4d3b85528f5d776b70efe3e35cb91d308e4f23e05e04ffa734f05f7b0ed1cdb3278366b0a8309a
    HEAD_REF main
)

vcpkg_from_github(
    OUT_SOURCE_PATH JITIFY_PATH
    REPO NVIDIA/jitify
    REF 44e978b21fc8bdb6b2d7d8d179523c8350db72e5
    SHA512 c6a175ae6ebae066285f1d662f8a7f73ea595fa17cf1ae7c66261899f5458e0c674eb5d546c404b8840cd1a2e760d72903b7bf6f5a48d32b13ebb5325256a2c4
    HEAD_REF master
)

vcpkg_from_github(
    OUT_SOURCE_PATH BS_THREAD_POOL_PATH
    REPO bshoshany/thread-pool
    REF 097aa718f25d44315cadb80b407144ad455ee4f9
    SHA512 94177c61c5161c3cb5d088058d999239fb8bc446100e948bb9bbae44b73d0a020240c39d5232b13a628b56e233cb55a29e70baa69e511c73a6ba6a2505de1250
    HEAD_REF master
)

# vcpkg sets FETCHCONTENT_FULLY_DISCONNECTED=ON during port builds, which prevents
# CPM from downloading any sources. We must provide all CPM dependencies as local sources.
# zstd: cudf's get_zstd.cmake forces download via CPM_DOWNLOAD_zstd=ON
vcpkg_from_github(
    OUT_SOURCE_PATH ZSTD_PATH
    REPO facebook/zstd
    REF v1.5.7
    SHA512 26e441267305f6e58080460f96ab98645219a90d290a533410b1b0b1d2f870721c95f8384e342ee647c5e968385a5b7e30c2d04340c37f59b3e6d86762c3260c
    HEAD_REF dev
)

# cuco: rapids-cmake always_download=true forces download
vcpkg_from_github(
    OUT_SOURCE_PATH CUCO_PATH
    REPO NVIDIA/cuCollections
    REF d3701ae8e7f2a08f25f9713e182692b4ca544112
    SHA512 2a35e079a86a62cca7dadf93b45609d86a0bdef1e987c0d5ec59600d1950e6d6a5e26e06cc3d165c389442ce5e25b461caf4cecd67c35369a72d742b1b5851cf
    HEAD_REF dev
)

# GTest: cudf builds test utilities even with BUILD_TESTS=OFF. CPM can't download due to
# FETCHCONTENT_FULLY_DISCONNECTED=ON. We provide source here and use -isystem (not -I)
# for vcpkg includes below to avoid header version conflicts with vcpkg's gtest 1.17.0.
vcpkg_from_github(
    OUT_SOURCE_PATH GTEST_PATH
    REPO google/googletest
    REF 6910c9d9165801d8827d628cb72eb7ea9dd538c5
    SHA512 5cb681fb2c1b3283c4c4f3dc31878fda9c3134bb09643c19a38245f41aa2d55ea4c28f802ae5e5862c05452fec004f9aa0a2b4d3e937755949c217330bff9135
    HEAD_REF main
)

# Patch get_zstd.cmake to remove forced download - we provide zstd source via CPM variable
vcpkg_replace_string("${SOURCE_PATH}/cpp/cmake/thirdparty/get_zstd.cmake"
    "set(CPM_DOWNLOAD_zstd ON)"
    "# CPM_DOWNLOAD_zstd removed - using CPM_zstd_SOURCE instead"
)

# Patch dlpack - vcpkg's 0.8 port incorrectly reports version 0.6, so use 0.6
vcpkg_replace_string("${SOURCE_PATH}/cpp/cmake/thirdparty/get_dlpack.cmake"
    "find_and_configure_dlpack(\${CUDF_MIN_VERSION_dlpack})"
    "find_and_configure_dlpack(\"0.6\")"
)

# Patch rapids_logger to use vcpkg's spdlog::spdlog target instead of spdlog
vcpkg_replace_string("${RAPIDS_LOGGER_PATH}/CMakeLists.txt"
    "set_target_properties(spdlog PROPERTIES POSITION_INDEPENDENT_CODE ON)"
    "set_target_properties(spdlog::spdlog PROPERTIES POSITION_INDEPENDENT_CODE ON)"
)

# Patch nanoarrow - vcpkg's nanoarrow_static is an ALIAS target, can't set properties on it
vcpkg_replace_string("${SOURCE_PATH}/cpp/cmake/thirdparty/get_nanoarrow.cmake"
    "set_target_properties(nanoarrow_static PROPERTIES POSITION_INDEPENDENT_CODE ON)"
    "# set_target_properties disabled for vcpkg ALIAS target"
)

# Use -I for vcpkg include dir to ensure vcpkg CCCL 3.2.0 headers beat pixi's older CCCL.
# GTest is excluded from vcpkg deps (see vcpkg.json) to avoid header conflicts with
# the CPM-provided GTest source that cudf needs for test utilities.
vcpkg_cmake_configure(
    SOURCE_PATH "${SOURCE_PATH}/cpp"
    OPTIONS
        -DFETCHCONTENT_SOURCE_DIR_RAPIDS-CMAKE=${RAPIDS_CMAKE_PATH}
        -DCPM_rapids_logger_SOURCE=${RAPIDS_LOGGER_PATH}
        -DCPM_jitify_SOURCE=${JITIFY_PATH}
        -DCPM_bs_thread_pool_SOURCE=${BS_THREAD_POOL_PATH}
        -DCPM_zstd_SOURCE=${ZSTD_PATH}
        -DCPM_cuco_SOURCE=${CUCO_PATH}
        -DCPM_GTest_SOURCE=${GTEST_PATH}
        -DCMAKE_CUDA_ARCHITECTURES=RAPIDS
        -DBUILD_SHARED_LIBS=OFF
        -DCUDA_STATIC_RUNTIME=ON
        -DBUILD_TESTS=OFF
        -DBUILD_BENCHMARKS=OFF
        -DCUDF_KVIKIO_REMOTE_IO=OFF
        "-DCMAKE_CXX_FLAGS=-I${CURRENT_INSTALLED_DIR}/include -Wno-error=sign-compare -Wno-error=parentheses"
        "-DCMAKE_CUDA_FLAGS=-I${CURRENT_INSTALLED_DIR}/include -Xcompiler=-Wno-error=sign-compare -Xcompiler=-Wno-error=parentheses"
)

vcpkg_cmake_install()

# Remove CPM-installed cuco files that conflict with standalone cuco:x64-linux port
file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/include/cuco")
file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/lib/cmake/cuco")

# Remove CPM-installed zstd files that conflict with standalone zstd:x64-linux port
file(REMOVE "${CURRENT_PACKAGES_DIR}/include/zdict.h")
file(REMOVE "${CURRENT_PACKAGES_DIR}/include/zstd.h")
file(REMOVE "${CURRENT_PACKAGES_DIR}/include/zstd_errors.h")
file(REMOVE "${CURRENT_PACKAGES_DIR}/lib/libzstd.a")
file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/lib/pkgconfig")

vcpkg_cmake_config_fixup(
    PACKAGE_NAME cudf
    CONFIG_PATH lib/cmake/cudf
)

file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/debug/include")
file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/debug/share")

# Fix cudf-dependencies.cmake to skip ALIAS targets when setting IMPORTED_GLOBAL.
# ALIAS targets like CCCL::CUB, CCCL::libcudacxx don't support set_target_properties.
vcpkg_replace_string("${CURRENT_PACKAGES_DIR}/share/cudf/cudf-dependencies.cmake"
[[foreach(target IN LISTS rapids_global_targets)
  if(TARGET ${target})
    get_target_property(_is_imported ${target} IMPORTED)
    get_target_property(_already_global ${target} IMPORTED_GLOBAL)
    if(_is_imported AND NOT _already_global)
        set_target_properties(${target} PROPERTIES IMPORTED_GLOBAL TRUE)
    endif()
  endif()
endforeach()]]
[[foreach(target IN LISTS rapids_global_targets)
  if(TARGET ${target})
    get_target_property(_aliased ${target} ALIASED_TARGET)
    if(_aliased)
      # Skip ALIAS targets - can't set properties on them
      continue()
    endif()
    get_target_property(_is_imported ${target} IMPORTED)
    get_target_property(_already_global ${target} IMPORTED_GLOBAL)
    if(_is_imported AND NOT _already_global)
        set_target_properties(${target} PROPERTIES IMPORTED_GLOBAL TRUE)
    endif()
  endif()
endforeach()]]
)

# Remove rapids_logger files that conflict with rmm (rmm already provides them)
file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/include/rapids_logger")
file(REMOVE "${CURRENT_PACKAGES_DIR}/lib/librapids_logger.a")
file(REMOVE "${CURRENT_PACKAGES_DIR}/debug/lib/librapids_logger.a")

# Fix cudf-targets.cmake: remove conda_env from link libraries and its target definition.
# rapids-cmake creates a conda_env target with hardcoded pixi/conda paths that aren't portable.
vcpkg_replace_string("${CURRENT_PACKAGES_DIR}/share/cudf/cudf-targets.cmake"
    ";\$<LINK_ONLY:cudf::conda_env>"
    ""
)

vcpkg_install_copyright(FILE_LIST "${SOURCE_PATH}/LICENSE")
