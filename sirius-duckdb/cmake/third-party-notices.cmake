# Embeds the third-party notices of an extension binary: one row per component
# for sirius_licenses(), plus a THIRD_PARTY_NOTICES.txt document in the
# .sirius_notices section, which release tooling extracts without loading the
# extension. Components are directories written by sirius_third_party_component.
include_guard(GLOBAL)
include(
  "${CMAKE_CURRENT_LIST_DIR}/../../cmake/sirius-third-party-component.cmake")
if(POLICY CMP0174)
  cmake_policy(SET CMP0174 NEW)
endif()

# sirius_notices_add_duckdb(<dir> <duckdb source dir> <version>)
#
# Writes DuckDB and the libraries vendored in its tree, which every extension
# binary embeds in part. DuckDB's CI configures the loader with this, so an
# unexpected tree only warns: failing would break every extension's build.
function(sirius_notices_add_duckdb dir duckdb_dir version)
  # SPDX expressions of DuckDB's vendored libraries, as licensed in its tree.
  set(licenses
      "brotli=MIT"
      "concurrentqueue=BSD-2-Clause OR BSL-1.0"
      "fast_float=MIT"
      "fastpforlib=Apache-2.0"
      "fmt=MIT"
      "fsst=MIT"
      "httplib=MIT"
      "hyperloglog=MIT"
      "jaro_winkler=MIT"
      "lz4=BSD-2-Clause"
      "mbedtls=Apache-2.0 OR GPL-2.0-or-later"
      "miniz=MIT"
      "parquet=Apache-2.0"
      "pcg=MIT"
      "pdqsort=Zlib"
      "re2=BSD-3-Clause"
      "ska_sort=BSL-1.0"
      "skiplist=MIT"
      "snappy=BSD-3-Clause"
      "tdigest=Apache-2.0"
      "thrift=Apache-2.0"
      "utf8proc=MIT AND Unicode-TOU"
      "vergesort=MIT"
      "yyjson=MIT"
      "zstd=BSD-3-Clause")
  # Test, benchmark and allocator code that extensions do not link.
  set(excluded catch imdb jemalloc)
  file(REMOVE_RECURSE "${dir}")
  sirius_third_party_component(
    "${dir}/duckdb"
    NAME
    "DuckDB"
    VERSION
    "${version}"
    LICENSE
    "MIT"
    HOMEPAGE
    "https://duckdb.org"
    FILES
    "${duckdb_dir}/LICENSE")
  set(alp_license
      "${duckdb_dir}/src/include/duckdb/storage/compression/alp/algorithm/LICENSE"
  )
  if(EXISTS "${alp_license}")
    sirius_third_party_component(
      "${dir}/duckdb-alp"
      NAME
      "ALP (vendored by DuckDB)"
      LICENSE
      "MIT"
      HOMEPAGE
      "https://github.com/cwida/ALP"
      FILES
      "${alp_license}")
  else()
    message(WARNING "DuckDB's ALP license moved from ${alp_license}")
  endif()
  file(
    GLOB vendored CONFIGURE_DEPENDS
    LIST_DIRECTORIES true
    RELATIVE "${duckdb_dir}/third_party"
    "${duckdb_dir}/third_party/*")
  foreach(name IN LISTS vendored)
    if(NOT IS_DIRECTORY "${duckdb_dir}/third_party/${name}" OR name IN_LIST
                                                               excluded)
      continue()
    endif()
    set(license "")
    foreach(entry IN LISTS licenses)
      if(entry MATCHES "^${name}=(.+)$")
        set(license "${CMAKE_MATCH_1}")
      endif()
    endforeach()
    if(license STREQUAL "")
      message(
        WARNING
          "Unknown license for DuckDB third_party/${name}; add it to sirius_notices_add_duckdb"
      )
      set(license "NOASSERTION")
    endif()
    set(homepage
        "https://github.com/duckdb/duckdb/tree/main/third_party/${name}")
    file(
      GLOB_RECURSE candidates CONFIGURE_DEPENDS
      LIST_DIRECTORIES false
      "${duckdb_dir}/third_party/${name}/LICENSE"
      "${duckdb_dir}/third_party/${name}/NOTICES")
    list(SORT candidates)
    # Keep one file per name; nested copies repeat the same terms.
    set(files)
    set(file_names)
    foreach(candidate IN LISTS candidates)
      cmake_path(GET candidate FILENAME file_name)
      if(NOT file_name IN_LIST file_names)
        list(APPEND file_names "${file_name}")
        list(APPEND files "${candidate}")
      endif()
    endforeach()
    set(note "")
    if(NOT files)
      set(note
          "DuckDB vendors this library without a license file; see ${homepage}."
      )
    endif()
    sirius_third_party_component(
      "${dir}/duckdb-${name}"
      NAME
      "${name} (vendored by DuckDB)"
      LICENSE
      "${license}"
      HOMEPAGE
      "${homepage}"
      FILES
      ${files}
      NOTE
      "${note}")
  endforeach()
endfunction()

# sirius_notices_add_vcpkg(<dir> <installed triplet dir>)
#
# Writes one component per installed vcpkg port from its SPDX document and
# copyright file. Components a port installs under share/<port>/third_party are
# collected by sirius_notices_generate.
function(sirius_notices_add_vcpkg dir installed_dir)
  file(REMOVE_RECURSE "${dir}")
  file(
    GLOB ports CONFIGURE_DEPENDS
    LIST_DIRECTORIES true
    RELATIVE "${installed_dir}/share"
    "${installed_dir}/share/*")
  foreach(port IN LISTS ports)
    set(share "${installed_dir}/share/${port}")
    # vcpkg-* are build helpers; sirius is the extension's own row.
    if(port MATCHES "^vcpkg-" OR port STREQUAL "sirius")
      continue()
    endif()
    if(NOT EXISTS "${share}/copyright")
      message(WARNING "vcpkg port ${port} installs no copyright file")
      continue()
    endif()
    set(version "")
    set(license "NOASSERTION")
    set(homepage "")
    if(EXISTS "${share}/vcpkg.spdx.json")
      file(READ "${share}/vcpkg.spdx.json" spdx)
      string(JSON count LENGTH "${spdx}" packages)
      set(indices)
      if(count GREATER 0)
        math(EXPR last "${count} - 1")
        foreach(index RANGE 0 ${last})
          list(APPEND indices ${index})
        endforeach()
      endif()
      foreach(index IN LISTS indices)
        string(
          JSON
          id
          GET
          "${spdx}"
          packages
          ${index}
          SPDXID)
        if(id STREQUAL "SPDXRef-port")
          string(
            JSON
            version
            GET
            "${spdx}"
            packages
            ${index}
            versionInfo)
          string(
            JSON
            license
            GET
            "${spdx}"
            packages
            ${index}
            licenseConcluded)
          string(
            JSON
            homepage
            ERROR_VARIABLE
            no_homepage
            GET
            "${spdx}"
            packages
            ${index}
            homepage)
          if(no_homepage OR homepage STREQUAL "NOASSERTION")
            set(homepage "")
          endif()
        endif()
      endforeach()
      # Drop vcpkg's port revision suffix.
      string(REGEX REPLACE "#[0-9]+$" "" version "${version}")
    endif()
    sirius_third_party_component(
      "${dir}/vcpkg-${port}"
      NAME
      "${port}"
      VERSION
      "${version}"
      LICENSE
      "${license}"
      HOMEPAGE
      "${homepage}"
      FILES
      "${share}/copyright")
  endforeach()
  # The CUDA runtime, device runtime and PTX compiler come from the toolkit and
  # share the CUDA EULA shipped with the culibos redistributable.
  if(EXISTS "${installed_dir}/share/culibos/copyright")
    sirius_third_party_component(
      "${dir}/cuda-toolkit"
      NAME
      "CUDA Runtime, Device Runtime and PTX Compiler static libraries"
      VERSION
      "${CUDAToolkit_VERSION}"
      LICENSE
      "LicenseRef-NVIDIA-CUDA-EULA"
      HOMEPAGE
      "https://docs.nvidia.com/cuda/eula/"
      FILES
      "${installed_dir}/share/culibos/copyright")
  endif()
endfunction()

function(_sirius_notices_cxx_string out_var value)
  string(REPLACE "\\" "\\\\" value "${value}")
  string(REPLACE "\"" "\\\"" value "${value}")
  string(REPLACE "\n" "\\n" value "${value}")
  set(${out_var}
      "\"${value}\""
      PARENT_SCOPE)
endfunction()

# sirius_notices_generate(<out_var> SYMBOL <extension> TITLE <title> VERSION
# <version> SIRIUS_LICENSE <file> DIRECTORIES <dir>...)
#
# Collects every component under DIRECTORIES, deduplicating license texts, and
# sets <out_var> to the generated source holding the table and document. The
# document is written to THIRD_PARTY_NOTICES.txt in the current binary dir.
# Targets compiling the source, sirius_licenses.cpp or code calling it define
# SIRIUS_NOTICES_NAMESPACE as <out_var>_NAMESPACE, so extensions linked into one
# binary keep separate notices.
function(sirius_notices_generate out_var)
  cmake_parse_arguments(PARSE_ARGV 1 arg ""
                        "SYMBOL;TITLE;VERSION;SIRIUS_LICENSE" "DIRECTORIES")
  if(NOT arg_SYMBOL MATCHES "^[A-Za-z_][A-Za-z0-9_]*$")
    message(FATAL_ERROR "sirius_notices_generate needs an identifier SYMBOL")
  endif()
  set(label "sirius_notices_${arg_SYMBOL}")
  set(output_dir "${CMAKE_CURRENT_BINARY_DIR}/third_party_notices")
  sirius_third_party_component(
    "${output_dir}/sirius"
    NAME
    "Sirius"
    VERSION
    "${arg_VERSION}"
    LICENSE
    "Apache-2.0"
    HOMEPAGE
    "https://github.com/sirius-db/sirius"
    FILES
    "${arg_SIRIUS_LICENSE}")

  set(manifests "${output_dir}/sirius/component.cmake")
  foreach(dir IN LISTS arg_DIRECTORIES)
    file(
      GLOB_RECURSE found CONFIGURE_DEPENDS
      LIST_DIRECTORIES false
      "${dir}/component.cmake")
    list(SORT found)
    foreach(manifest IN LISTS found)
      file(REAL_PATH "${manifest}" manifest)
      list(APPEND manifests "${manifest}")
    endforeach()
  endforeach()
  list(REMOVE_DUPLICATES manifests)

  # Sort by name, keeping Sirius first; duplicate name and version pairs (one
  # library vendored by several ports) are listed once.
  set(keys)
  set(seen)
  foreach(manifest IN LISTS manifests)
    unset(component_version)
    include("${manifest}")
    # A tab sorts before any name character, and "0" puts Sirius first.
    string(TOLOWER "1${component_name}" sort_name)
    if(manifest STREQUAL "${output_dir}/sirius/component.cmake")
      set(sort_name "0")
    endif()
    set(identity "${component_name}@${component_version}")
    string(REPLACE ";" "," identity "${identity}")
    string(REPLACE ";" "," sort_name "${sort_name}")
    if(identity IN_LIST seen)
      continue()
    endif()
    list(APPEND seen "${identity}")
    list(APPEND keys "${sort_name}\t${manifest}")
  endforeach()
  list(SORT keys)

  set(text_hashes)
  set(texts)
  set(rows "")
  set(index_lines "")
  set(component_count 0)
  foreach(key IN LISTS keys)
    string(REGEX REPLACE "^[^\t]*\t" "" manifest "${key}")
    foreach(field IN ITEMS name version license homepage note files)
      unset(component_${field})
    endforeach()
    include("${manifest}")
    cmake_path(GET manifest PARENT_PATH component_dir)
    set(text "")
    list(LENGTH component_files file_count)
    foreach(file_name IN LISTS component_files)
      file(READ "${component_dir}/${file_name}" content)
      set_property(
        DIRECTORY
        APPEND
        PROPERTY CMAKE_CONFIGURE_DEPENDS "${component_dir}/${file_name}")
      if(file_count GREATER 1)
        string(APPEND text "---- ${file_name} ----\n\n")
      endif()
      string(STRIP "${content}" content)
      string(APPEND text "${content}\n\n")
    endforeach()
    if(component_note)
      string(APPEND text "${component_note}\n")
    endif()
    string(REPLACE "\r\n" "\n" text "${text}")
    string(REPLACE "\r" "\n" text "${text}")
    string(STRIP "${text}" text)
    string(SHA256 hash "${text}")
    list(FIND text_hashes "${hash}" text_index)
    if(text_index EQUAL -1)
      list(LENGTH text_hashes text_index)
      list(APPEND text_hashes "${hash}")
      set(text_${text_index} "${text}")
      set(text_users_${text_index} "")
    endif()
    list(APPEND text_users_${text_index} "${component_name}")

    set(line "${component_name}")
    if(NOT component_version STREQUAL "")
      string(APPEND line " ${component_version}")
    endif()
    string(APPEND line "\n  License: ${component_license}")
    if(component_homepage)
      string(APPEND line "\n  Homepage: ${component_homepage}")
    endif()
    math(EXPR notice_number "${text_index} + 1")
    string(APPEND line "\n  Notice: ${notice_number}\n")
    string(APPEND index_lines "${line}")

    foreach(field IN ITEMS name version license homepage)
      _sirius_notices_cxx_string(cxx_${field} "${component_${field}}")
    endforeach()
    # Plain strings, not lists: field values may contain semicolons.
    string(
      APPEND
      rows
      "  {${cxx_name}, ${cxx_version}, ${cxx_license}, ${cxx_homepage}, ${text_index}},\n"
    )
    math(EXPR component_count "${component_count} + 1")
  endforeach()

  set(rule
      "================================================================================"
  )
  set(document "${arg_TITLE}\n\n")
  string(
    APPEND
    document
    "This binary includes the following components. Each lists the notice below that contains its license terms.\n\n"
  )
  string(APPEND document "${index_lines}")
  set(spans)
  list(LENGTH text_hashes text_count)
  math(EXPR last_text "${text_count} - 1")
  foreach(index RANGE 0 ${last_text})
    math(EXPR notice_number "${index} + 1")
    list(JOIN text_users_${index} ", " users)
    string(APPEND document
           "\n${rule}\nNotice ${notice_number}: ${users}\n${rule}\n\n")
    string(LENGTH "${document}" offset)
    string(LENGTH "${text_${index}}" length)
    string(APPEND document "${text_${index}}\n")
    list(APPEND spans "{${offset}, ${length}}")
  endforeach()

  # Write through a temporary file so unchanged notices don't rebuild.
  file(WRITE "${output_dir}/THIRD_PARTY_NOTICES.txt.tmp" "${document}")
  set(document_path "${CMAKE_CURRENT_BINARY_DIR}/THIRD_PARTY_NOTICES.txt")
  configure_file("${output_dir}/THIRD_PARTY_NOTICES.txt.tmp" "${document_path}"
                 COPYONLY)
  # .incbin hides the document from compiler caches; its digest changes the
  # source.
  string(SHA256 document_digest "${document}")
  string(REPLACE "\\" "\\\\" asm_path "${document_path}")
  string(REPLACE "\"" "\\\"" asm_path "${asm_path}")
  list(JOIN spans ",\n  " spans)
  file(
    WRITE "${output_dir}/third_party_notices.cpp.tmp"
    "// Generated by sirius-duckdb/cmake/third-party-notices.cmake.
#include \"sirius_licenses.hpp\"

// The document is used by the table below, so the linker keeps the section.
// Global hidden labels resolve across LTO partitions without being exported.
asm(R\"(
  .pushsection .sirius_notices, \"a\"
  .balign 16
  .globl ${label}_begin
  .hidden ${label}_begin
  .globl ${label}_end
  .hidden ${label}_end
  ${label}_begin:
  .incbin \"${asm_path}\"
  ${label}_end:
  .byte 0
  .popsection
)\");

extern \"C\" char const ${label}_begin[];
extern \"C\" char const ${label}_end[];

namespace sirius_duckdb {
namespace {

[[maybe_unused]] constexpr char document_sha256[] = \"${document_digest}\";

constexpr third_party_notices::span text_spans[] = {
  ${spans}};

constexpr third_party_notices::row component_rows[] = {
${rows}};

}  // namespace

std::string_view third_party_notices::document()
{
  return {${label}_begin, static_cast<std::size_t>(${label}_end - ${label}_begin)};
}

third_party_notices::row const* third_party_notices::rows_begin() { return component_rows; }

third_party_notices::row const* third_party_notices::rows_end()
{
  return component_rows + sizeof(component_rows) / sizeof(component_rows[0]);
}

std::string_view third_party_notices::text(std::size_t index)
{
  auto const& span = text_spans[index];
  return document().substr(span.offset, span.length);
}

}  // namespace sirius_duckdb
")
  configure_file("${output_dir}/third_party_notices.cpp.tmp"
                 "${output_dir}/third_party_notices.cpp" COPYONLY)
  set_source_files_properties("${output_dir}/third_party_notices.cpp"
                              PROPERTIES OBJECT_DEPENDS "${document_path}")
  set_property(
    DIRECTORY
    APPEND
    PROPERTY CMAKE_CONFIGURE_DEPENDS ${manifests})
  message(
    STATUS
      "Embedding ${component_count} components with ${text_count} distinct notices"
  )
  set(${out_var}
      "${output_dir}/third_party_notices.cpp"
      PARENT_SCOPE)
  set(${out_var}_NAMESPACE
      "${label}"
      PARENT_SCOPE)
endfunction()
