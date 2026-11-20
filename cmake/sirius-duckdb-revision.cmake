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
