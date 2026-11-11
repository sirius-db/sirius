# The NVRTC device runtime (nvrtc/device_runtime/*.cuh) is embedded into the
# library by cmake/sirius-nvrtc-device-runtime.cmake; device_runtime_check.cu
# compiles the same headers with nvcc so a syntax error fails the build instead
# of the first query.
target_sources(
  sirius_objects
  PRIVATE expression_evaluator_strategy.cpp
          ast_op_counter.cpp
          expression_evaluator.cpp
          gpu_expression_translator.cpp
          nvrtc/device_runtime/device_runtime_check.cu
          nvrtc/filter_sink.cpp
          nvrtc/kernel_abi.cpp
          nvrtc/overflow_site.cpp
          nvrtc/project_sink.cpp
          nvrtc/sink_launch.cpp
          nvrtc/string_constant_cache.cpp
          nvrtc/supportability.cpp
          nvrtc/translation_context.cpp
          regex/regex_playground.cpp
          specializations/between.cpp
          specializations/case.cpp
          specializations/cast.cpp
          specializations/comparison.cpp
          specializations/conjunction.cpp
          specializations/constant.cpp
          specializations/function.cpp
          specializations/narrow_domain.cpp
          specializations/operator.cpp
          specializations/reference.cpp)
