# Transitively scan the CCCL (<cuda/std/...>, <cub/...>) #include closure the
# runtime NVRTC JIT needs, and emit a .cpp that embeds each header as a
# raw-string literal exposed as an EmbeddedJitHeader table. Passing these to
# nvrtcCreateProgram() as named in-memory headers lets the JIT compile with NO
# -I into a CCCL tree, so a binary distribution needs only the driver + the
# nvrtc runtime it already links. Invoked via `cmake -P`.
#
# Required -D inputs: INCLUDE_DIRS_FILE contains CMake's evaluated CCCL include
# directories, one per line; OUT is the path of the .cpp to generate. DEPFILE
# optionally names a Make/Ninja depfile for the transitively scanned headers.
#
# The scan follows every literal `#include` line (both #ifdef branches) from the
# fixed kernel-prelude roots below. Dependencies reached through macro-expanded
# includes must be seeded explicitly. Recomputed at build time, so it tracks the
# CCCL version in use.

cmake_minimum_required(VERSION 3.24)
include("${CMAKE_CURRENT_LIST_DIR}/jit_header_manifest.cmake")

file(STRINGS "${INCLUDE_DIRS_FILE}" cccl_include_dirs)

# Installed packages share one include root; source packages use separate
# component roots. Search the directories in the order supplied by CMake.
function(find_cccl_header name output)
  set(${output}
      ""
      PARENT_SCOPE)
  cmake_path(SET relative_name NORMALIZE "${name}")
  if(IS_ABSOLUTE "${relative_name}" OR relative_name MATCHES "^\\.\\.(/|$)")
    return()
  endif()
  foreach(include_dir IN LISTS cccl_include_dirs)
    cmake_path(SET candidate NORMALIZE "${include_dir}/${relative_name}")
    # A new header in an earlier search root can shadow a previously selected
    # file. Track the nearest existing parent even for unsuccessful searches.
    get_filename_component(search_parent "${candidate}" DIRECTORY)
    while(NOT IS_DIRECTORY "${search_parent}")
      get_filename_component(next_parent "${search_parent}" DIRECTORY)
      if(next_parent STREQUAL search_parent)
        break()
      endif()
      set(search_parent "${next_parent}")
    endwhile()
    set_property(GLOBAL APPEND PROPERTY simpatico_search_parents
                                        "${search_parent}")
    if(EXISTS "${candidate}" AND NOT IS_DIRECTORY "${candidate}")
      set(${output}
          "${candidate}"
          PARENT_SCOPE)
      return()
    endif()
  endforeach()
endfunction()

foreach(marker cub/version.cuh cuda/std/cstdint thrust/version.h)
  find_cccl_header("${marker}" header)
  if(NOT header)
    message(
      FATAL_ERROR
        "embed_cccl_headers: '${marker}' is missing from CCCL include directories: ${cccl_include_dirs}"
    )
  endif()
endforeach()

# Union of the includes emitted by the encode + decode kernel preludes.
set(roots
    cub/block/block_reduce.cuh
    cub/block/block_scan.cuh
    cub/block/block_exchange.cuh
    cuda/std/cstdint
    cuda/std/cstddef
    cuda/std/climits
    cuda/std/type_traits)
foreach(root IN LISTS roots)
  find_cccl_header("${root}" header)
  if(NOT header)
    message(
      FATAL_ERROR
        "embed_cccl_headers: required prelude header '${root}' is missing from CCCL include directories: ${cccl_include_dirs}"
    )
  endif()
endforeach()

# Some CCCL versions reach these through macro-expanded includes, which the
# literal-include scanner cannot discover. Seed them when present.
foreach(root thrust/system/cpp/detail/execution_policy.h
             thrust/system/cuda/detail/execution_policy.h)
  find_cccl_header("${root}" header)
  if(header)
    list(APPEND roots "${root}")
  else()
    message(STATUS "embed_cccl_headers: skipping optional header '${root}'")
  endif()
endforeach()

set(worklist ${roots})
set(found "")

while(worklist)
  list(POP_FRONT worklist rel)
  if(rel IN_LIST found)
    continue()
  endif()
  find_cccl_header("${rel}" abs)
  if(NOT abs)
    continue()
  endif()
  list(APPEND found "${rel}")

  file(READ "${abs}" content)
  get_filename_component(curdir "${rel}" DIRECTORY)
  # Anchor to line start (a newline followed by only whitespace then '#') so we
  # don't match example `#include` lines inside doc comments — those would drag
  # in large unused subtrees (e.g. all of thrust/). A leading newline is
  # prepended so a directive on the first line still matches.
  string(REGEX MATCHALL "[\r\n][ \t]*#[ \t]*include[ \t]*[<\"][^>\"\r\n]+[>\"]"
               incs "\n${content}")
  foreach(inc IN LISTS incs)
    string(REGEX REPLACE ".*[<\"]([^>\"]+)[>\"].*" "\\1" name "${inc}")
    set(resolved "")
    # 1) resolve through CCCL's include directories (<cuda/...>, <cub/...>)
    find_cccl_header("${name}" header)
    if(header)
      set(resolved "${name}")
    elseif(curdir)
      # 2) resolve relative to the including file's directory (quoted includes)
      cmake_path(SET relative_name NORMALIZE "${curdir}/${name}")
      if(NOT relative_name MATCHES "^\\.\\./")
        find_cccl_header("${relative_name}" header)
        if(header)
          set(resolved "${relative_name}")
        endif()
      endif()
    endif()
    # Names that don't resolve under CCCL are nvrtc built-ins or host headers
    # guarded out under __CUDACC_RTC__ — skip them.
    if(resolved AND NOT resolved IN_LIST found)
      list(APPEND worklist "${resolved}")
    endif()
  endforeach()
endwhile()

list(REMOVE_DUPLICATES found)
list(SORT found)
list(LENGTH found n)
if(n EQUAL 0)
  message(FATAL_ERROR "embed_cccl_headers: the scanned CCCL closure is empty")
endif()

# Raw-string delimiter (<=16 chars, C++ limit) chosen so it cannot appear in a
# CCCL header.
set(D "CCCL_EMB_9F3A")
set(body "// AUTO-GENERATED by embed_cccl_headers.cmake -- DO NOT EDIT.\n")
string(APPEND body
       "// ${n} CCCL headers, the runtime NVRTC JIT include closure.\n")
string(APPEND body "#include \"codegen/jit/cccl_embedded_headers.h\"\n")
string(APPEND body "namespace codegen::jit {\n")

set(idx 0)
set(header_dependencies "")
set(manifest "simpatico-headers-v1\n")
foreach(rel IN LISTS found)
  find_cccl_header("${rel}" header)
  list(APPEND header_dependencies "${header}")
  file(READ "${header}" hsrc)
  jit_normalize_header("${hsrc}" hsrc)
  jit_manifest_record("${rel}" "${hsrc}" record)
  string(APPEND manifest "${record}")
  string(APPEND body "static const char* kCcclName${idx} = \"${rel}\";\n")
  string(APPEND body
         "static const char* kCcclSrc${idx} =\nR\"${D}(${hsrc})${D}\";\n")
  math(EXPR idx "${idx}+1")
endforeach()

string(APPEND body "const EmbeddedJitHeader kCcclEmbeddedHeaders[] = {\n")
set(idx 0)
foreach(rel IN LISTS found)
  string(APPEND body "  {kCcclName${idx}, kCcclSrc${idx}},\n")
  math(EXPR idx "${idx}+1")
endforeach()
string(APPEND body "};\n")
string(APPEND body "const int kCcclEmbeddedHeaderCount = ${n};\n")
string(SHA256 manifest_digest "${manifest}")
string(APPEND body
       "const char kCcclEmbeddedHeadersIdentity[] = \"${manifest_digest}\";\n")
string(APPEND body "}  // namespace codegen::jit\n")

file(WRITE "${OUT}" "${body}")

# Teach the build graph about the dynamically discovered closure. If any scanned
# header changes (including gaining a new literal include), the custom command
# reruns and discovers the updated closure.
if(DEFINED DEPFILE AND NOT DEPFILE STREQUAL "")
  # Escape a path for a Make/Ninja depfile and return it through output.
  function(escape_depfile_path input output)
    set(escaped "${input}")
    string(REPLACE "\\" "/" escaped "${escaped}")
    string(REPLACE "$" "$$" escaped "${escaped}")
    string(REPLACE "#" "\\#" escaped "${escaped}")
    string(REPLACE " " "\\ " escaped "${escaped}")
    string(REPLACE ":" "\\:" escaped "${escaped}")
    set(${output}
        "${escaped}"
        PARENT_SCOPE)
  endfunction()

  escape_depfile_path("${OUT}" depfile_out)
  set(_depfile_body "${depfile_out}:")
  get_property(search_parents GLOBAL PROPERTY simpatico_search_parents)
  list(APPEND header_dependencies ${search_parents})
  list(REMOVE_DUPLICATES header_dependencies)
  foreach(header IN LISTS header_dependencies)
    escape_depfile_path("${header}" depfile_header)
    string(APPEND _depfile_body " \\\n  ${depfile_header}")
  endforeach()
  string(APPEND _depfile_body "\n")
  file(WRITE "${DEPFILE}" "${_depfile_body}")
endif()

message(STATUS "embed_cccl_headers: embedded ${n} CCCL headers into ${OUT}")
