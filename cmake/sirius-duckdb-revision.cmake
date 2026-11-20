# Sets <out> to the commit checked out in <dir>, or to an empty string when
# <dir> is not the root of a git work tree.
function(sirius_duckdb_revision dir out)
  set(${out}
      ""
      PARENT_SCOPE)
  execute_process(
    COMMAND "${CMAKE_COMMAND}" -E env --unset=GIT_DIR --unset=GIT_WORK_TREE git
            -C "${dir}" rev-parse --show-cdup --absolute-git-dir HEAD
    RESULT_VARIABLE result
    OUTPUT_VARIABLE lines
    ERROR_QUIET)
  # A non-empty first line means git answered for an enclosing repository.
  if(NOT result EQUAL 0 OR NOT lines MATCHES "^\n([^\n]+)\n([0-9a-f]+)\n$")
    return()
  endif()
  set(${out}
      "${CMAKE_MATCH_2}"
      PARENT_SCOPE)
  if(CMAKE_SCRIPT_MODE_FILE)
    return()
  endif()
  # Reconfigure when the checkout moves, so a recorded revision cannot go stale.
  foreach(file HEAD logs/HEAD)
    if(EXISTS "${CMAKE_MATCH_1}/${file}")
      set_property(
        DIRECTORY
        APPEND
        PROPERTY CMAKE_CONFIGURE_DEPENDS "${CMAKE_MATCH_1}/${file}")
    endif()
  endforeach()
endfunction()

# Fails unless <library_revision>, the DuckDB commit Sirius was built from, is
# the commit checked out in <source_dir>. Revisions that cannot be determined
# are an error for "static" <linkage> and not compared otherwise.
function(sirius_require_duckdb_revision library_revision source_dir linkage)
  sirius_duckdb_revision("${source_dir}" revision)
  if(NOT "${library_revision}" STREQUAL "" AND NOT "${revision}" STREQUAL "")
    if(NOT "${library_revision}" STREQUAL "${revision}")
      message(
        FATAL_ERROR
          "Sirius was built from DuckDB ${library_revision}, but this extension is built from DuckDB ${revision}"
      )
    endif()
  elseif(linkage STREQUAL "static")
    message(
      FATAL_ERROR
        "Cannot verify that Sirius ('${library_revision}') and this extension ('${revision}') use one DuckDB revision: build both from git checkouts"
    )
  endif()
endfunction()
