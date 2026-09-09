# plan-2mig: the top-3 exchange plan adapted to the g7e.4xlarge (1x RTX PRO 6000, MIG 2x 2g.48gb, 124 GiB, SF500)

## Scope and prerequisites

These are experimental runbooks and recorded observations extracted from
`bench/sf500-2-mig-gpus` at `7610840c`. Their dates and measurements describe the
original runs; this documentation PR does not reproduce or certify those results.

The notes distinguish three runtime trees: the integration branch, `perf`, and the
optimized `mcn` / `wp0` work. Features attributed to another tree (including receive
credits, HOST-first ingress, bounded frames, and `CN_SIRIUS_CONFIG_DIR`) are not
introduced by this PR. In particular, `gen-cn-config.sh` in `DISK-TIER.md` belongs to
the separate launcher work; it is not provided here. Reproduction requires the
named runtime revision, benchmark harness, dataset and tool checkout.

For the recorded two-MIG engine, use the documented `GPU_DEVICES=0,1` route.
`MIG_DEVICES=MIG-<uuid>,MIG-<uuid>` is a future compatibility path: the recorded
cuCascade/NVML engine rejects it. See `CHECKLIST-2mig.md` for the failure and the
ordinal workaround. Capacity figures and device identifiers are specific to the
recorded box and should be checked for a new deployment.

## Reading order

Reading order: `BOX-TRANSLATION.md` (numbers), `DISK-TIER.md` (spilling config), `PLAN-top3-2mig.md` (plan), `PROMPT-top3-2mig.md` (box/tools/protocol), `CHECKLIST-2mig.md` (state),
`PROMPT-execute-2mig.md` (paste-ready session prompt). `original/` holds the 2-GPU-box documents verbatim; they remain authoritative for mechanisms and file anchors.
