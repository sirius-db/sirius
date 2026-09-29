target_sources(
  sirius_objects
  PRIVATE expression_evaluator_strategy.cpp
          ast_op_counter.cpp
          expression_evaluator.cpp
          gpu_expression_translator.cpp
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
