target_sources(
  sirius_objects
  PRIVATE gpu_pipeline_executor.cpp
          gpu_pipeline_task.cpp
          sirius_pipeline_itask.cpp
          task_scheduler.cpp
          sirius_meta_pipeline.cpp
          sirius_pipeline.cpp
          sirius_pipeline_converter.cpp
          data_size_estimator.cpp
          repository_wiring_materializer.cpp
          sirius_plan_printer.cpp
          task_request.cpp)
