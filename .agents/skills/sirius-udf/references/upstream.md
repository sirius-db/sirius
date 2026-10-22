# Upstream attribution

The workflow draws on NVIDIA's [cuDF Spark UDF agent skills](https://github.com/NVIDIA/cudf-spark/tree/main/skills),
including `udf-gen-test`, `udf-convert-to-sql`, `udf-convert-to-cudf`,
`udf-convert-to-cuda`, and `udf-benchmark`.

Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

The upstream [license](https://github.com/NVIDIA/cudf-spark/blob/main/LICENSE)
licenses documentation and skills under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
and source code under [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0).
This skill's documentation is a Sirius-specific adaptation under CC BY 4.0:
Spark/Maven/Java/JNI guidance has been replaced with DuckDB macros and Sirius's
C++/libcudf/CUDA registration and execution contract. These changes are by Sirius
Contributors. No endorsement by NVIDIA is implied.

The scaffold scripts, templates, examples, and tests are independently authored
Sirius source under Apache 2.0; they do not copy upstream source code. Preserve
this attribution when redistributing the skill documentation, and retain the
applicable notices and licenses if future changes import upstream material.
