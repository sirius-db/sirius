# Rename both definitions and references, including template instantiations and
# RTTI and local COMDAT signatures, to avoid a consumer's incompatible ABI.
function(nvcomp_isolate_logging NM OBJCOPY SYMBOL_MAP)
  execute_process(COMMAND "${NM}" --defined-only -j ${ARGN}
                  OUTPUT_VARIABLE SYMBOLS COMMAND_ERROR_IS_FATAL ANY)
  string(REPLACE "\n" ";" SYMBOLS "${SYMBOLS}")
  list(FILTER SYMBOLS INCLUDE REGEX "6spdlog|3fmt|13rapids_logger")
  list(REMOVE_DUPLICATES SYMBOLS)
  list(SORT SYMBOLS)
  if(NOT SYMBOLS)
    message(FATAL_ERROR "No bundled nvCOMP logging symbols found")
  endif()
  file(WRITE "${SYMBOL_MAP}" "")
  foreach(SYMBOL IN LISTS SYMBOLS)
    file(APPEND "${SYMBOL_MAP}" "${SYMBOL} nvcomp_private_${SYMBOL}\n")
  endforeach()
  foreach(ARCHIVE IN LISTS ARGN)
    execute_process(COMMAND "${OBJCOPY}" "--redefine-syms=${SYMBOL_MAP}"
                            "${ARCHIVE}" COMMAND_ERROR_IS_FATAL ANY)
  endforeach()
endfunction()
