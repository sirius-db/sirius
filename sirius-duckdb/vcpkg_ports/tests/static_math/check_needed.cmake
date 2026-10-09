execute_process(
  COMMAND "${READELF}" -d "${LIBRARY}"
  RESULT_VARIABLE result
  OUTPUT_VARIABLE dynamic_section
  ERROR_VARIABLE error_output)
if(NOT result EQUAL 0)
  message(FATAL_ERROR "readelf failed: ${error_output}")
endif()

if(dynamic_section
   MATCHES
   "NEEDED[^\n]*lib(cublas[^]]*|cusolver[^]]*|cusparse[^]]*|curand[^]]*|cudart[^]]*|nvJitLink[^]]*|nvrtc[^]]*)\\.so"
)
  message(FATAL_ERROR "Shared CUDA dependency found:\n${dynamic_section}")
endif()
