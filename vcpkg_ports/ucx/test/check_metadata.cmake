file(READ "${PCFILE}" metadata)
if(metadata MATCHES "(^|[ \n])-L([A-Za-z]:)?/")
  message(
    FATAL_ERROR
      "UCX pkg-config metadata contains an absolute library search path:\n${metadata}"
  )
endif()
if(NOT metadata MATCHES "-lcudart_static")
  message(FATAL_ERROR "UCX pkg-config metadata omits the static CUDA runtime")
endif()
