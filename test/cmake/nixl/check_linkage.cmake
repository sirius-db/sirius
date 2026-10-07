find_program(READELF readelf REQUIRED)
execute_process(
  COMMAND "${READELF}" -d "${PROBE}"
  RESULT_VARIABLE result
  OUTPUT_VARIABLE dynamic_section
  ERROR_VARIABLE error)
if(NOT result EQUAL 0)
  message(FATAL_ERROR "readelf failed: ${error}")
endif()
if(dynamic_section
   MATCHES
   "NEEDED[^\n]*lib(nixl|plugin_UCX|stream|serdes|absl|ucp|uct|ucs|ucm|ucx|cudart|toml)[^\n]*\\.so"
)
  message(
    FATAL_ERROR
      "Probe depends on a shared NIXL, UCX, Abseil, or CUDA runtime library:\n${dynamic_section}"
  )
endif()
