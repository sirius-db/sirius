# TPC-Havoc queries for the C++ tests. TPC-Havoc is BenchBox's set of rewrites
# of the 22 TPC-H queries (https://github.com/BenchBox-dev/BenchBox, MIT
# license). Only its query files are needed, so they are downloaded from a
# pinned commit at configure time instead of vendoring BenchBox. Tests read them
# through test/cpp/utils/tpch_havoc_queries.hpp.
option(SIRIUS_TPCH_HAVOC "Download the TPC-Havoc queries for the C++ tests" ON)
set(SIRIUS_TPCH_HAVOC_DIR
    ""
    CACHE
      PATH
      "Use the TPC-Havoc files in this directory instead of downloading them")

# To move the pin, change the commit and each file's SHA-256 (see
# test/README.md).
set(_tpch_havoc_commit 262eb226675b36d9221f8dc554ffdd03c4913978)
set(_tpch_havoc_url
    "https://raw.githubusercontent.com/BenchBox-dev/BenchBox/${_tpch_havoc_commit}"
)
# <path in BenchBox>=<SHA-256 of the file at the pinned commit>
set(_tpch_havoc_files
    "LICENSE=8a051ce0007bd890d5268188bcfd5eabd5a648b6824ec51dcad54fd0f3c9e4ec"
    "benchbox/core/tpchavoc/variant_sets/q01.yaml=3ff5a678a5fd13fa499bed2fb319807d377763f37da85cf538eb1061fc535dcd"
    "benchbox/core/tpchavoc/variant_sets/q02.yaml=214c53c461a990a1899f56023fab95b0e225ef96be4ee3ccdab4ea955f7e5fcd"
    "benchbox/core/tpchavoc/variant_sets/q03.yaml=eec142ebf1442d68babcfe197bc30bf1b543d5649eac7342100e9d7f33f67c09"
    "benchbox/core/tpchavoc/variant_sets/q04.yaml=d28f6bbd068f380c863c1923d4bb764af3338093e9e59cd9dff78a243683ceda"
    "benchbox/core/tpchavoc/variant_sets/q05.yaml=48d7b1f8d8f193ebf3af415553f494cc1c2b8c1d7bb95e17a6dc0bcf571452c6"
    "benchbox/core/tpchavoc/variant_sets/q06.yaml=0685d2d1b13fc5f269ae0d1d543a02b52037aac1c29608c09df29b1ca33e29c0"
    "benchbox/core/tpchavoc/variant_sets/q07.yaml=54191b0fd951565a8a4037a320db7f6831504294955d52f768b85de877905087"
    "benchbox/core/tpchavoc/variant_sets/q08.yaml=161673e8362dffc15c028a28e9a0883bfe0d55092f689a279cab994939291ca0"
    "benchbox/core/tpchavoc/variant_sets/q09.yaml=5291a64a82508e94ea461c64b8a5217bed54a998ec37c34a9aaa5b0cd953398e"
    "benchbox/core/tpchavoc/variant_sets/q10.yaml=850aa5512090546fa41568d527a3c99871526a143450a36b18315bc6e1daa656"
    "benchbox/core/tpchavoc/variant_sets/q11.yaml=4a8d41d902aee879f80a2c415d4c15d34cf7e2ea79aab856a480ce413841a13b"
    "benchbox/core/tpchavoc/variant_sets/q12.yaml=343743430586f2a47cc9d51b0b8ce4a784df81de63979e187f33dc05848b1edb"
    "benchbox/core/tpchavoc/variant_sets/q13.yaml=8c13bbada59e155427599c3519e2b813eb6ad952aaac933a6046f9a28250fc80"
    "benchbox/core/tpchavoc/variant_sets/q14.yaml=13b8d40576a3ba483be5bf33e7d3339e7ca46023b1ea2b2a50da9e52e9422fcc"
    "benchbox/core/tpchavoc/variant_sets/q15.yaml=eeab1f7b4c335f8686a805a890ae138563f90139148059a35b61609d491a7d36"
    "benchbox/core/tpchavoc/variant_sets/q16.yaml=c1e44f4f53903251f8959ccef02f467a79174a3de36ca08be0a936f371ad6c3d"
    "benchbox/core/tpchavoc/variant_sets/q17.yaml=4519f88681b5e9409f2dee87df7cbb42af67ae37009045c526a300aad3978ede"
    "benchbox/core/tpchavoc/variant_sets/q18.yaml=a1b198e11153d270881cfaad8ff74588e50272f54af3473a55be8a8e72dc9d65"
    "benchbox/core/tpchavoc/variant_sets/q19.yaml=b78821191faa6b9f5371d4c5e153b4666f9e636bd8a75375efb31671d4110e70"
    "benchbox/core/tpchavoc/variant_sets/q20.yaml=f2c311846de928ee84767f0a482a5984ce7bc05281847e83476080357c19fa7e"
    "benchbox/core/tpchavoc/variant_sets/q21.yaml=aec1cb89a48a5f81a92cb25fea2ed8ecd5c43f00bf042c762e473026e2b937da"
    "benchbox/core/tpchavoc/variant_sets/q22.yaml=d03b621ae7d47001693e49ac506bbb9b9cc43b7b40d36a194590d6446a1ec05a"
)

# Puts <path> into <dir> unless a copy with the expected hash is already there.
# With SIRIUS_TPCH_HAVOC_DIR set, the copy must exist; nothing is downloaded.
function(sirius_fetch_tpch_havoc_file dir path sha256)
  cmake_path(GET path FILENAME name)
  set(file "${dir}/${name}")
  if(EXISTS "${file}")
    file(SHA256 "${file}" actual)
    if(actual STREQUAL sha256)
      return()
    endif()
  endif()
  if(SIRIUS_TPCH_HAVOC_DIR)
    message(
      FATAL_ERROR
        "${file} is missing or differs from BenchBox ${_tpch_havoc_commit}")
  endif()
  file(
    DOWNLOAD "${_tpch_havoc_url}/${path}" "${file}.part"
    TLS_VERIFY ON
    STATUS status)
  list(GET status 0 code)
  if(NOT code EQUAL 0)
    list(GET status 1 reason)
    file(REMOVE "${file}.part")
    message(
      FATAL_ERROR
        "Could not download ${_tpch_havoc_url}/${path}: ${reason}. To configure "
        "offline, pass -DSIRIUS_TPCH_HAVOC_DIR=<copy> or -DSIRIUS_TPCH_HAVOC=OFF."
    )
  endif()
  # Checked here rather than with EXPECTED_HASH, whose mismatch doesn't stop
  # configure.
  file(SHA256 "${file}.part" actual)
  if(NOT actual STREQUAL sha256)
    file(REMOVE "${file}.part")
    message(
      FATAL_ERROR
        "${_tpch_havoc_url}/${path} has SHA-256 ${actual}, expected ${sha256}")
  endif()
  file(RENAME "${file}.part" "${file}")
endfunction()

# Configure-time downloads are disabled in vcpkg builds; the tests skip without
# the files.
if(SIRIUS_TPCH_HAVOC AND NOT VCPKG_BUILD)
  if(SIRIUS_TPCH_HAVOC_DIR)
    cmake_path(ABSOLUTE_PATH SIRIUS_TPCH_HAVOC_DIR OUTPUT_VARIABLE
               _tpch_havoc_dir)
  else()
    set(_tpch_havoc_dir "${CMAKE_CURRENT_BINARY_DIR}/test/tpch_havoc")
  endif()
  foreach(_entry IN LISTS _tpch_havoc_files)
    string(REGEX MATCH "^(.+)=([0-9a-f]+)$" _match "${_entry}")
    sirius_fetch_tpch_havoc_file("${_tpch_havoc_dir}" "${CMAKE_MATCH_1}"
                                 "${CMAKE_MATCH_2}")
  endforeach()
  # Only the loader needs the location, so changing it recompiles one file.
  set_source_files_properties(
    test/cpp/utils/tpch_havoc_queries.cpp
    PROPERTIES COMPILE_DEFINITIONS
               SIRIUS_TPCH_HAVOC_DEFAULT_DIR="${_tpch_havoc_dir}")
  message(STATUS "TPC-Havoc queries: ${_tpch_havoc_dir}")
endif()
