# Strips device code for GPU architectures outside VCPKG_CUDA_ARCHITECTURES from
# NVIDIA's prebuilt static libraries using nvprune.
#
# Each real target X.y keeps every cubin a CC X.y device can load: sm_Xz for z
# <= y (and their family variants) plus sm_Xya. Family targets (e.g. 100f) also
# cover the higher minors of their major. PTX is kept for virtual entries (e.g.
# a bare 120) and, as the libraries' JIT fallback, for archs up to the lowest
# real target.

# nvprune must match the CUDA major of the libraries it prunes; nvprune 13
# rejects archs it does not know.
set(_cuda_prune_12_version "12.9.82")
set(_cuda_prune_12_archs
    50
    52
    53
    60
    61
    62
    70
    72
    75
    80
    86
    87
    89
    90
    100
    101
    103
    120
    121)
set(_cuda_prune_12_linux-x86_64_sha512
    "698d4f67422d845aa3a8c0a19747517c54c41576ee374fe1aa58cddd0c0891aabc6b2d3ef3f01e2f2512109594d1bb79a23dced74a460c802e19477b37a4de7d"
)
set(_cuda_prune_12_linux-sbsa_sha512
    "1643b7f1c8a0e517064f29ea08ea239001d11543ff040b9056442651941e0de092c5b14ee0b65a17a3f2c77e1097ac5308c5e8c68c007490c3e60d45f69ef86d"
)
set(_cuda_prune_13_version "13.5.37")
set(_cuda_prune_13_archs
    75
    80
    86
    87
    88
    89
    90
    100
    103
    110
    120
    121)
set(_cuda_prune_13_linux-x86_64_sha512
    "ef05089e49d967a2cb8cf00cea7d50f847720f834880cd0062ba69eb89a62c058829d8f1bbafb2e2e65a44eb84e351c33d404f9901194139f7ade2c145a87b17"
)
set(_cuda_prune_13_linux-sbsa_sha512
    "6ae22007d769dc482c3fcb8735a351f63b5afcaf195c64566d5a792ddae8422f9a2212cb6d9ff2a463bf8de79a40cc6d405441b5d717606b880c39f42f9461fe"
)

# Sets OUT_VAR to the nvprune -gencode arguments for VCPKG_CUDA_ARCHITECTURES.
function(cuda_prune_gencode_args OUT_VAR)
  if(NOT DEFINED _cuda_prune_${VCPKG_CUDA_VERSION}_archs)
    message(FATAL_ERROR "Unsupported VCPKG_CUDA_VERSION: ${VCPKG_CUDA_VERSION}")
  endif()
  set(_known "${_cuda_prune_${VCPKG_CUDA_VERSION}_archs}")
  if(VCPKG_CUDA_ARCHITECTURES STREQUAL "")
    message(FATAL_ERROR "VCPKG_CUDA_ARCHITECTURES is empty")
  endif()

  set(_codes "")
  set(_lowest "")
  foreach(_entry IN LISTS VCPKG_CUDA_ARCHITECTURES)
    if(NOT _entry MATCHES "^(([1-9][0-9]*)[0-9])([af]?)(-real|-virtual)?$")
      message(
        FATAL_ERROR
          "Unsupported CUDA architecture '${_entry}' in VCPKG_CUDA_ARCHITECTURES"
      )
    endif()
    set(_arch "${CMAKE_MATCH_1}")
    set(_major "${CMAKE_MATCH_2}")
    set(_variant "${CMAKE_MATCH_3}")
    set(_kind "${CMAKE_MATCH_4}")
    if(NOT _arch IN_LIST _known)
      message(FATAL_ERROR "Unknown CUDA ${VCPKG_CUDA_VERSION} arch: ${_entry}")
    endif()

    if(NOT _kind STREQUAL "-virtual")
      if(_lowest STREQUAL "" OR _arch LESS _lowest)
        set(_lowest "${_arch}")
      endif()
      foreach(_other IN LISTS _known)
        math(EXPR _other_major "${_other} / 10")
        if(NOT _other_major EQUAL _major)
          continue()
        endif()
        if(_other GREATER _arch AND NOT _variant STREQUAL "f")
          continue()
        endif()
        list(APPEND _codes "sm_${_other}")
        if(_other GREATER_EQUAL _arch AND _other GREATER_EQUAL 90)
          list(APPEND _codes "sm_${_other}a")
        endif()
        # nvprune 13 maps family codes to their base arch.
        if(_other GREATER_EQUAL 100 AND VCPKG_CUDA_VERSION STREQUAL "12")
          list(APPEND _codes "sm_${_other}f")
        endif()
      endforeach()
    endif()
    if(NOT _kind STREQUAL "-real")
      list(APPEND _codes "compute_${_arch}${_variant}")
    endif()
  endforeach()
  foreach(_other IN LISTS _known)
    if(NOT _lowest STREQUAL "" AND _other LESS_EQUAL _lowest)
      list(APPEND _codes "compute_${_other}")
    endif()
  endforeach()

  list(REMOVE_DUPLICATES _codes)
  set(_args "")
  foreach(_code IN LISTS _codes)
    string(REGEX REPLACE "^(sm|compute)_" "" _code_arch "${_code}")
    list(APPEND _args "-gencode" "arch=compute_${_code_arch},code=${_code}")
  endforeach()
  set(${OUT_VAR}
      "${_args}"
      PARENT_SCOPE)
endfunction()

# Prunes the given static libraries in place.
function(cuda_prune)
  if(HOST_TRIPLET MATCHES "^x64-")
    set(_platform "linux-x86_64")
  elseif(HOST_TRIPLET MATCHES "^arm64-")
    set(_platform "linux-sbsa")
  else()
    message(FATAL_ERROR "Unsupported host triplet for nvprune: ${HOST_TRIPLET}")
  endif()
  cuda_prune_gencode_args(_gencode)

  set(_version "${_cuda_prune_${VCPKG_CUDA_VERSION}_version}")
  set(_archive "cuda_nvprune-${_platform}-${_version}-archive.tar.xz")
  vcpkg_download_distfile(
    _nvprune_archive
    URLS
    "https://developer.download.nvidia.com/compute/cuda/redist/cuda_nvprune/${_platform}/${_archive}"
    FILENAME
    "${_archive}"
    SHA512
    "${_cuda_prune_${VCPKG_CUDA_VERSION}_${_platform}_sha512}")
  vcpkg_extract_source_archive(_nvprune_dir ARCHIVE "${_nvprune_archive}")

  foreach(_lib IN LISTS ARGN)
    get_filename_component(_name "${_lib}" NAME_WE)
    vcpkg_execute_required_process(
      COMMAND
      "${_nvprune_dir}/bin/nvprune"
      ${_gencode}
      -o
      "${_lib}.pruned"
      "${_lib}"
      WORKING_DIRECTORY
      "${CURRENT_BUILDTREES_DIR}"
      LOGNAME
      "nvprune-${_name}")
    file(RENAME "${_lib}.pruned" "${_lib}")
  endforeach()
endfunction()
