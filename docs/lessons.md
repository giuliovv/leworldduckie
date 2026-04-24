# Lessons Learned & Bugs Fixed

## 1. Policy-Entanglement / FRAMESKIP Bug (critical)

**Symptom**: MPC success rate 0/10 across all runs. Agent spins in circles.

**Root cause**: The dataset is collected at 1-step granularity (every raw env step). With `FRAMESKIP=3` in training, the model learns:

```
predict(z_0, a_0) ≈ z_3
```

But `z_3` was produced by the sequence `a_0 → a_1 → a_2` (three different PD controller actions). The model implicitly learns:

```
E[z_{t+3} | z_t, a_t,  policy applies a_{t+1}, a_{t+2}]
```

At MPC time, the agent **repeats `a_t` three times** (frameskip loop). This is out-of-distribution: the model has never seen a trajectory where the same action is held for three steps. Rollouts are biased toward wherever the PD controller would go.

**Fix**: Set `FRAMESKIP=1` everywhere:
- `lewm_duckie.ipynb` cell 4: `FRAMESKIP = 1`
- `lewm_duckie_run.ipynb` cell 4: `FRAMESKIP = 1`
- `src/mpc_controller.py` default: `FRAMESKIP = 1`
- `infra/launch_mpc_eval.sh`: `--frameskip 1`

No re-collection needed: the dataset is already at 1-step granularity.

**Impact note**: This also means the model trains on all consecutive frame pairs (including high-correlation adjacent frames). Consider collecting with action-repeat in future to allow FRAMESKIP>1 without entanglement.

---

## 2. Stale Generator via Base64 Embedding (fixed 2026-04-23)

**Symptom**: Editing `generate_data.py` had no effect on EC2 runs. Data was being collected with the old `PDController` instead of `LaneFollowController`.

**Root cause**: `launch_datagen.sh` had a hardcoded base64 blob of an old version of `generate_data.py`. EC2 always decoded and ran that blob, ignoring the file on disk.

**Fix**: Removed the base64 blob entirely. `launch_datagen.sh` now:
1. Uploads `generate_data.py` to `s3://leworldduckie/scripts/generate_data.py` before launching
2. EC2 downloads via `boto3` at runtime

This pattern is now used consistently across `launch_datagen.sh`, `launch_eval.sh`, and `launch_mpc_eval.sh`.

---

## 3. boto3 Missing in Colab (silent S3 upload failure)

**Symptom**: Training ran successfully but no checkpoint appeared in S3. Every epoch printed `S3 upload skipped: No module named 'boto3'`.

**Root cause**: `boto3` was not in the pip install list in the notebook setup cell.

**Fix**: Added `'boto3'` to the installs loop in all three notebooks (`lewm_duckie.ipynb`, `lewm_duckie_run.ipynb`, `lewm_duckie_run_pid.ipynb`).

---

## 4. Lag Frames Missing in MPC (fixed 2026-04-24)

**Symptom**: MPC starts in an OOD state — the dynamics model was trained with `skip_initial_steps=4` (gym-duckietown PWM warm-up), so the first 4 env steps were discarded in training data.

**Root cause**: MPC was starting immediately from step 0.

**Fix**: At episode start, burn 4 LaneFollower steps (not added to the context):
```python
--lag-frames 4
```
These match the `LAG_FRAMES=4` config in the notebook and replicate the `skip_initial_steps=4` in `generate_data.py`.

---

## 5. EC2 Spot Capacity Exhausted in us-east-1a and us-east-1b

**Symptom**: `InsufficientInstanceCapacity` error when launching t3.medium spot instances.

**Fix**: Use subnet in us-east-1c: `subnet-01497e4f428a93b98`.

Both `launch_datagen.sh` and `launch_mpc_eval.sh` updated with this subnet.

---

## 6. Wrong Notebook Edited

**Symptom**: Changes made to `lewm_duckie_run_pid.ipynb` (a logging-only variant) instead of the main training notebook `lewm_duckie.ipynb` / `lewm_duckie_run.ipynb`.

**Fix**: Applied all changes (`FRAMESKIP=1`, `boto3`, `BATCH_SIZE=512`) to both `lewm_duckie.ipynb` and `lewm_duckie_run.ipynb`.

**Lesson**: The `_pid` notebook is a stripped-down variant used only for saving training logs. Always edit the main notebooks (`lewm_duckie.ipynb` and `lewm_duckie_run.ipynb`).

---

## 7. gdown `--id` Flag Removed

**Symptom**: `gdown --id FILE_ID` fails — the `--id` flag was removed in newer gdown versions.

**Fix**: Use full URL format:
```bash
gdown "https://drive.google.com/uc?id=FILE_ID"
```

---

## 8. GPU Underutilisation on A100

**Symptom**: Training used only ~13 GB of a 40 GB A100 with `BATCH_SIZE=128`.

**Fix**: Bumped `BATCH_SIZE` from 128 to 512 in all notebooks:
```python
BATCH_SIZE = 32 if not IS_COLAB else 512
```

Expected effect: ~2× speedup per epoch (~70s vs ~140s), GPU usage ~30-35 GB.

---

## Summary Table

| Bug | Impact | Fix |
|-----|--------|-----|
| FRAMESKIP policy-entanglement | MPC 0% success | FRAMESKIP=1 everywhere |
| Base64 stale generator | Wrong data collected | S3 upload+download pattern |
| boto3 missing | No S3 checkpoints saved | Add boto3 to pip installs |
| Lag frames missing | OOD episode starts | --lag-frames 4 |
| us-east-1a/b spot capacity | Launch failures | Switch to us-east-1c |
| Wrong notebook edited | Fixes lost | Edit main notebooks only |
| gdown --id removed | Google Drive download fails | Full URL format |
| BATCH_SIZE=128 on A100 | 2× slower training | BATCH_SIZE=512 |
