// Copyright 2026, Sirius Contributors. SPDX-License-Identifier: Apache-2.0
#include <sirius-c/plan/plan.h>
#include <stdio.h>

// This example is also compiled as C99 by the installed-package link smoke check.
int plan_example(void)
{
  sirius_schema* schema  = NULL;
  sirius_plan_node *scan = NULL, *filtered = NULL;
  sirius_expression *column = NULL, *ten = NULL, *predicate = NULL;
  sirius_plan* plan    = NULL;
  sirius_error* error  = NULL;
  sirius_field field   = {"id", 2, SIRIUS_TYPE_INT64, 1};
  sirius_status status = sirius_schema_create(&field, 1, &schema, &error);
  if (status != SIRIUS_SUCCESS) goto cleanup;
  status = sirius_plan_scan_parquet("data.parquet", 12, schema, &scan, &error);
  if (status != SIRIUS_SUCCESS) goto cleanup;
  status = sirius_expression_column(scan, 0, &column, &error);
  if (status != SIRIUS_SUCCESS) goto cleanup;
  status = sirius_expression_literal_int64(10, &ten, &error);
  if (status != SIRIUS_SUCCESS) goto cleanup;
  status = sirius_expression_compare(column, SIRIUS_COMPARISON_GREATER, ten, &predicate, &error);
  if (status != SIRIUS_SUCCESS) goto cleanup;
  status = sirius_plan_filter(scan, predicate, &filtered, &error);
  if (status != SIRIUS_SUCCESS) goto cleanup;
  status = sirius_plan_from_root(filtered, &plan, &error);
  if (status == SIRIUS_SUCCESS) puts(sirius_plan_describe(plan));
cleanup:
  if (status != SIRIUS_SUCCESS) fprintf(stderr, "%s\n", sirius_error_message(error));
  sirius_error_destroy(error);
  sirius_plan_release(plan);
  sirius_plan_node_release(filtered);
  sirius_expression_release(predicate);
  sirius_expression_release(ten);
  sirius_expression_release(column);
  sirius_plan_node_release(scan);
  sirius_schema_release(schema);
  return status != SIRIUS_SUCCESS;
}
