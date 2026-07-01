# Memory — Push-T Eval Debug Status (2026-05-09)

## Goal
Run a sanity Push-T eval (CPU spot, `num_eval=5`) using checkpoint `s3://leworldduckie/training/pusht/pusht/lewm_object.ckpt`.

## What was fixed
- Added robust failure trap + log uploads to `infra/launch_pusht_eval.sh`.
- Added `--subnet`, `--instance-type`, `--on-demand`, `--keep-on-fail` args.
- Fixed dependency/bootstrap issues:
  - packaging pins
  - swig install for box2d build path
  - moved away from pip boto3 conflict (use `python3-boto3` via apt).
- Uploaded Push-T dataset to S3:
  - `s3://leworldduckie/data/pusht_expert_train.h5` (43.1 GiB)
  - copied to role-readable path: `s3://leworldduckie/training/pusht/pusht_expert_train.h5`
- Fixed eval dataset path mismatch by copying to `/root/.stable-wm/pusht_expert_train.h5` before eval.
- Added CPU fallback patch in cloned `le-wm/eval.py` and runtime `solver.device` override.
- Added diagnostics HDF5 plugin path export.
- Added apt lock wait/retry for `unattended-upgr` dpkg lock.

## Key result observed
- One run reached eval and completed eval phase on CPU:
  - success_rate reported: 20% on `num_eval=5`
  - diagnostics then failed previously (HDF5 plugin path), later patched.

## Current parallel runs
- Run `20260509_091258` (`i-0dd84868a5cbf3d7b`) got stuck waiting on unattended upgrades lock.
- Side run launched: `20260509_102206` (`i-04fbf0406c9ceb15f`) to continue in parallel.

## Next immediate check
- Poll `s3://leworldduckie/evals/pusht/20260509_102206/` for `results.txt`, `diagnostics.txt`, `exit_code.txt`.
