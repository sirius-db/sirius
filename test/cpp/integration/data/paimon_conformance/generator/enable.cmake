# Used only when regenerating the DELETE fixture; never included by Sirius.
# -DCMAKE_PROJECT_paimon_INCLUDE=/absolute/path/to/enable.cmake
set(PAIMON_FIXTURE_CMAKE_DIR "${CMAKE_CURRENT_LIST_DIR}")
function(add_paimon_delete_fixture)
  include("${PAIMON_FIXTURE_CMAKE_DIR}/target.cmake")
endfunction()
cmake_language(DEFER CALL add_paimon_delete_fixture)
